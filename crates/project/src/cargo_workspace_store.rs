use feature_flags::{FeatureFlagAppExt as _, RustNativeProfilingExperimentalFeatureFlag};
#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    ffi::{OsStr, OsString},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, ensure};
use async_trait::async_trait;
use collections::HashMap;
use futures::{
    FutureExt as _,
    future::{AbortHandle, Abortable, Either, Shared},
    io::AsyncReadExt as _,
};
use gpui::{
    App, AppContext as _, AsyncApp, BackgroundExecutor, Context, Entity, EventEmitter,
    Subscription, Task, TaskExt as _,
};
use rpc::proto::{ErrorCodeExt as _, ErrorExt as _};
use rpc::{AnyProtoClient, TypedEnvelope, proto};
use sha2::{Digest as _, Sha256};
use util::{paths::PathStyle, rel_path::RelPath};

use crate::{
    ProjectEnvironment, ProjectPath,
    cargo_workspace::{
        CargoCandidateFailure, CargoConfigurationCompleteness,
        CargoConfigurationDiagnosticCategory, CargoHostCompilerModel, CargoHostCompilerStatus,
        CargoHostExecutionConfiguration, CargoSnapshotCompleteness, CargoWorkspaceErrorCategory,
        CargoWorkspaceModel, CargoWorkspaceSnapshot, MAX_CARGO_METADATA_BYTES,
        MAX_CARGO_PASSIVE_CONFIGURATION_BYTES, MAX_RUST_TOOLCHAIN_FILE_BYTES,
        MAX_RUSTC_VERBOSE_VERSION_BYTES, deduplicate_workspaces, enrich_dependency_provenance,
        parse_cargo_host_execution_configuration, parse_cargo_profiles, parse_metadata,
        parse_rust_toolchain, parse_rust_toolchain_configuration, parse_rustc_verbose_version,
        validate_cargo_snapshot_bounds, workspace_from_metadata,
    },
    trusted_worktrees::{TrustedWorktrees, TrustedWorktreesEvent},
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};

const MAX_ERROR_BYTES: usize = 4 * 1024;
const CARGO_METADATA_TIMEOUT: Duration = Duration::from_secs(30);
const CARGO_CONFIGURATION_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const PASSIVE_RUSTUP_POLICY_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_RUSTUP_PROXY_BYTES: u64 = 64 * 1024 * 1024;
#[cfg(any(test, feature = "test-support"))]
static LOCAL_CARGO_PROCESS_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(any(test, feature = "test-support"))]
pub fn local_cargo_process_calls_for_test() -> u64 {
    LOCAL_CARGO_PROCESS_CALLS.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoWorkspaceRemoteErrorKind {
    UnsupportedHost,
    Disconnected,
    Restricted,
}

#[derive(Debug)]
pub struct CargoWorkspaceRemoteError {
    pub kind: CargoWorkspaceRemoteErrorKind,
}

impl std::fmt::Display for CargoWorkspaceRemoteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            CargoWorkspaceRemoteErrorKind::UnsupportedHost => formatter.write_str(
                "This project host does not include Rust tooling. Install or select a rust-tools-capable remote server.",
            ),
            CargoWorkspaceRemoteErrorKind::Disconnected => {
                formatter.write_str("The project host is disconnected. Reconnect and refresh Cargo.")
            }
            CargoWorkspaceRemoteErrorKind::Restricted => formatter.write_str(
                "Cargo refresh was cancelled because worktree trust was revoked on the host. Trust the worktree and refresh Cargo.",
            ),
        }
    }
}

impl std::error::Error for CargoWorkspaceRemoteError {}

#[derive(Clone, Debug)]
pub struct CargoMetadataRequest {
    pub manifest_path: PathBuf,
    pub working_directory: PathBuf,
    pub environment: Option<HashMap<String, String>>,
}

#[async_trait]
pub trait CargoMetadataRunner: Send + Sync {
    async fn run(&self, request: CargoMetadataRequest) -> Result<Vec<u8>>;
}

#[derive(Clone, Debug)]
pub struct CargoConfigurationProbeRequest {
    pub working_directory: PathBuf,
    pub environment: Option<HashMap<String, String>>,
}

#[async_trait]
pub trait CargoConfigurationProbe: Send + Sync {
    async fn run(&self, request: CargoConfigurationProbeRequest) -> Result<Vec<u8>>;
}

struct ProcessCargoMetadataRunner {
    executor: BackgroundExecutor,
}

struct ProcessCargoConfigurationProbe {
    executor: BackgroundExecutor,
}

const PASSIVE_CARGO_PROTECTED_ENVIRONMENT: [&str; 11] = [
    "RUSTC",
    "RUSTDOC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTDOC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
    "PATH",
    "HOME",
    "USERPROFILE",
];

#[derive(Clone)]
struct PassiveRustEnvironment {
    overrides: HashMap<OsString, OsString>,
    inherit_process: bool,
}

impl Default for PassiveRustEnvironment {
    fn default() -> Self {
        Self {
            overrides: HashMap::default(),
            inherit_process: true,
        }
    }
}

impl PassiveRustEnvironment {
    fn from_request(environment: Option<&HashMap<String, String>>) -> Self {
        Self {
            inherit_process: true,
            overrides: environment
                .into_iter()
                .flatten()
                .map(|(name, value)| (name.into(), value.into()))
                .collect(),
        }
    }

    #[cfg(target_os = "macos")]
    fn from_native_recipe(recipe: &crate::terminals::NativeTaskEnvironmentRecipe) -> Self {
        Self {
            overrides: recipe
                .environment()
                .iter()
                .map(|(name, value)| (name.into(), value.into()))
                .collect(),
            inherit_process: false,
        }
    }

    fn value(&self, name: &str) -> Option<OsString> {
        self.overrides
            .iter()
            .find_map(|(key, value)| {
                let matches = key == OsStr::new(name)
                    || (cfg!(windows)
                        && key
                            .to_str()
                            .is_some_and(|key| key.eq_ignore_ascii_case(name)));
                matches.then(|| value.clone())
            })
            .or_else(|| {
                self.inherit_process
                    .then(|| std::env::var_os(name))
                    .flatten()
            })
    }

    fn pin_present_cargo_parent(&mut self) -> Self {
        let mut pinned = Self::default();
        for name in PASSIVE_CARGO_PROTECTED_ENVIRONMENT {
            if let Some(value) = self.value(name) {
                self.insert(name, value.clone());
                pinned.insert(name, value);
            }
        }
        pinned
    }

    fn apply_to_command(&self, command: &mut util::command::Command) {
        if !self.inherit_process {
            command.env_clear();
        }
        command.envs(&self.overrides);
    }

    fn path_for_executable_lookup(&self) -> Option<OsString> {
        self.value("PATH")
            .or_else(|| (!self.inherit_process).then(OsString::new))
    }

    fn insert(&mut self, name: &str, value: OsString) {
        if cfg!(windows) {
            self.overrides.retain(|key, _| {
                !key.to_str()
                    .is_some_and(|key| key.eq_ignore_ascii_case(name))
            });
        }
        self.overrides.insert(name.into(), value);
    }
}

struct PassiveRustExecution {
    child_environment: PassiveRustEnvironment,
    cargo_parent_environment: PassiveRustEnvironment,
    compiler_chain: Vec<OsString>,
    #[cfg(target_os = "macos")]
    delegated_executables: Vec<PathBuf>,
    toolchain: Option<PathBuf>,
    cargo_options: Vec<OsString>,
}

struct NativeCargoConfigurationSeal {
    path: PathBuf,
    requested: Option<fs::Metadata>,
    file: Option<NativeProbeFileSeal>,
    directories: Vec<(PathBuf, u64)>,
}

async fn native_cargo_configuration_directories(
    filesystem: &dyn fs::Fs,
    path: &Path,
) -> Result<Vec<(PathBuf, u64)>> {
    let ancestors = path
        .parent()
        .context("Native Cargo configuration has no parent")?
        .ancestors();
    let mut directories = Vec::new();
    for (depth, parent) in ancestors.enumerate() {
        ensure!(
            depth < 128,
            "Native Cargo configuration route exceeds its bound"
        );
        crate::cargo_workspace::validate_native_cargo_path(parent)?;
        if let Some(metadata) = filesystem.metadata(parent).await? {
            ensure!(
                metadata.is_dir && !metadata.is_symlink && !metadata.is_fifo,
                "Native Cargo configuration parent is not ordinary"
            );
            ensure!(
                filesystem.canonicalize(parent).await? == parent,
                "Native Cargo configuration parent has aliases"
            );
            directories.push((parent.to_path_buf(), metadata.inode));
        }
    }
    ensure!(
        !directories.is_empty(),
        "Native Cargo configuration route is unavailable"
    );
    Ok(directories)
}

async fn read_native_cargo_configuration(
    filesystem: &dyn fs::Fs,
    path: &Path,
) -> Result<(Option<Vec<u8>>, NativeCargoConfigurationSeal)> {
    crate::cargo_workspace::validate_native_cargo_path(path)?;
    let directories = native_cargo_configuration_directories(filesystem, path).await?;
    let requested = filesystem.metadata(path).await?;
    let Some(requested_metadata) = requested else {
        return Ok((
            None,
            NativeCargoConfigurationSeal {
                path: path.to_path_buf(),
                requested,
                file: None,
                directories,
            },
        ));
    };
    ensure!(
        !requested_metadata.is_dir && !requested_metadata.is_fifo,
        "Native Cargo configuration is not a regular file route"
    );
    let canonical = filesystem.canonicalize(path).await?;
    crate::cargo_workspace::validate_native_cargo_path(&canonical)?;
    let root = canonical
        .parent()
        .context("Native Cargo configuration parent disappeared")?
        .to_path_buf();
    let root_metadata = filesystem
        .metadata(&root)
        .await?
        .context("Native Cargo configuration root disappeared")?;
    ensure!(
        root_metadata.is_dir
            && !root_metadata.is_symlink
            && !root_metadata.is_fifo
            && filesystem.canonicalize(&root).await? == root,
        "Native Cargo configuration root is not ordinary"
    );
    let metadata = filesystem
        .metadata(&canonical)
        .await?
        .context("Native Cargo configuration disappeared")?;
    ensure!(
        !metadata.is_dir
            && !metadata.is_symlink
            && !metadata.is_fifo
            && metadata.len <= MAX_CARGO_PASSIVE_CONFIGURATION_BYTES as u64,
        "Native Cargo configuration is not a bounded regular file"
    );
    let relative = canonical.strip_prefix(&root)?.to_path_buf();
    let reader = filesystem
        .open_sync_confined_file(&root, &relative, root_metadata.inode, metadata.inode)
        .await?;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(
        &mut std::io::Read::take(reader, MAX_CARGO_PASSIVE_CONFIGURATION_BYTES as u64 + 1),
        &mut bytes,
    )?;
    ensure!(
        bytes.len() <= MAX_CARGO_PASSIVE_CONFIGURATION_BYTES && bytes.len() as u64 == metadata.len,
        "Native Cargo configuration grew or changed length"
    );
    let after = filesystem
        .metadata(&canonical)
        .await?
        .context("Native Cargo configuration disappeared while reading")?;
    ensure!(
        native_probe_metadata_matches(after, metadata)
            && filesystem
                .metadata(path)
                .await?
                .is_some_and(|current| native_probe_metadata_matches(current, requested_metadata))
            && filesystem.canonicalize(path).await? == canonical
            && native_cargo_configuration_directories(filesystem, path).await? == directories,
        "Native Cargo configuration changed while reading"
    );
    let fingerprint = Sha256::digest(&bytes).into();
    Ok((
        Some(bytes),
        NativeCargoConfigurationSeal {
            path: path.to_path_buf(),
            requested,
            file: Some(NativeProbeFileSeal {
                root,
                root_inode: root_metadata.inode,
                relative,
                metadata,
                fingerprint,
            }),
            directories,
        },
    ))
}

async fn revalidate_native_cargo_configuration(
    filesystem: &dyn fs::Fs,
    seals: &[NativeCargoConfigurationSeal],
) -> Result<()> {
    for seal in seals {
        let (_, current) = read_native_cargo_configuration(filesystem, &seal.path).await?;
        ensure!(
            current.directories == seal.directories
                && match (current.requested, seal.requested) {
                    (None, None) => true,
                    (Some(current), Some(before)) => native_probe_metadata_matches(current, before),
                    _ => false,
                }
                && match (&current.file, &seal.file) {
                    (None, None) => true,
                    (Some(current), Some(before)) =>
                        current.root == before.root
                            && current.root_inode == before.root_inode
                            && current.relative == before.relative
                            && native_probe_metadata_matches(current.metadata, before.metadata)
                            && current.fingerprint == before.fingerprint,
                    _ => false,
                },
            "Native Cargo configuration changed after admission"
        );
    }
    Ok(())
}

struct PassiveCargoConfigurationLoader {
    native_filesystem: Option<Arc<dyn fs::Fs>>,
    native_seals: Vec<NativeCargoConfigurationSeal>,
    active: HashSet<PathBuf>,
    files: usize,
    bytes: usize,
}

impl PassiveCargoConfigurationLoader {
    fn load<'a>(
        &'a mut self,
        path: PathBuf,
        optional: bool,
        depth: usize,
    ) -> futures::future::BoxFuture<'a, Result<CargoHostExecutionConfiguration>> {
        async move {
            if depth >= 32 || self.files >= 128 {
                return Err(anyhow!(
                    "Cargo configuration search exceeds the supported limit"
                ));
            }
            let native_contents = if let Some(filesystem) = &self.native_filesystem {
                ensure!(
                    self.native_seals.len() < 512,
                    "Native Cargo configuration route count exceeds its bound"
                );
                let (contents, seal) =
                    read_native_cargo_configuration(filesystem.as_ref(), &path).await?;
                self.native_seals.push(seal);
                match contents {
                    Some(contents) => Some(contents),
                    None if optional => return Ok(CargoHostExecutionConfiguration::default()),
                    None => anyhow::bail!("Native Cargo configuration disappeared"),
                }
            } else {
                None
            };
            let identity = if let Some(filesystem) = &self.native_filesystem {
                filesystem.canonicalize(&path).await?
            } else {
                match smol::fs::canonicalize(&path).await {
                    Ok(path) => path,
                    Err(error) if optional && error.kind() == std::io::ErrorKind::NotFound => {
                        return Ok(CargoHostExecutionConfiguration::default());
                    }
                    Err(_) => return Err(anyhow!("Cargo configuration cannot be read")),
                }
            };
            if !self.active.insert(identity.clone()) {
                return Err(anyhow!("Cargo configuration includes form a cycle"));
            }
            self.files += 1;
            let contents = if let Some(contents) = native_contents {
                contents
            } else {
                let file = smol::fs::File::open(&path)
                    .await
                    .map_err(|_| anyhow!("Cargo configuration cannot be read"))?;
                let mut contents = Vec::new();
                file.take((MAX_CARGO_PASSIVE_CONFIGURATION_BYTES + 1) as u64)
                    .read_to_end(&mut contents)
                    .await
                    .map_err(|_| anyhow!("Cargo configuration cannot be read"))?;
                contents
            };
            self.bytes += contents.len();
            if self.bytes > 1024 * 1024 {
                return Err(anyhow!(
                    "Cargo configuration search exceeds the supported byte limit"
                ));
            }
            let contents = std::str::from_utf8(&contents)
                .map_err(|_| anyhow!("Cargo configuration is not UTF-8"))?;
            let mut own = parse_cargo_host_execution_configuration(contents, &path)?;
            let mut merged = CargoHostExecutionConfiguration::default();
            for include in std::mem::take(&mut own.includes) {
                merged.merge_values(self.load(include.path, include.optional, depth + 1).await?)?;
            }
            merged.merge_values(own)?;
            self.active.remove(&identity);
            Ok(merged)
        }
        .boxed()
    }
}

async fn passive_cargo_configuration(
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
) -> Result<CargoHostExecutionConfiguration> {
    Ok(
        passive_cargo_configuration_with_native_review(working_directory, environment, None)
            .await?
            .0,
    )
}

async fn passive_cargo_configuration_with_native_review(
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
    native_filesystem: Option<Arc<dyn fs::Fs>>,
) -> Result<(
    CargoHostExecutionConfiguration,
    Vec<NativeCargoConfigurationSeal>,
)> {
    let directory = if let Some(filesystem) = &native_filesystem {
        filesystem.canonicalize(working_directory).await?
    } else {
        smol::fs::canonicalize(working_directory)
            .await
            .map_err(|_| anyhow!("Cargo working directory cannot be resolved"))?
    };
    let home = environment
        .value("CARGO_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            environment
                .value(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
                .map(|home| PathBuf::from(home).join(".cargo"))
        });
    let mut directories = Vec::new();
    if let Some(home) = home {
        directories.push(if home.is_absolute() {
            home
        } else {
            directory.join(home)
        });
    }
    let ancestors = directory.ancestors().collect::<Vec<_>>();
    if ancestors.len() > 128 {
        return Err(anyhow!(
            "Cargo configuration directory search exceeds the supported limit"
        ));
    }
    directories.extend(
        ancestors
            .into_iter()
            .rev()
            .map(|directory| directory.join(".cargo")),
    );
    let mut loader = PassiveCargoConfigurationLoader {
        native_filesystem,
        native_seals: Vec::new(),
        active: HashSet::new(),
        files: 0,
        bytes: 0,
    };
    let mut configuration = CargoHostExecutionConfiguration::default();
    for directory in directories {
        // Cargo prefers its extensionless configuration when both names exist.
        let mut selected = None;
        for name in ["config", "config.toml"] {
            let path = directory.join(name);
            if let Some(filesystem) = &loader.native_filesystem {
                if filesystem.metadata(&path).await?.is_some() {
                    selected = Some(path);
                    break;
                }
                ensure!(
                    loader.native_seals.len() < 512,
                    "Native Cargo configuration route count exceeds its bound"
                );
                let (contents, seal) =
                    read_native_cargo_configuration(filesystem.as_ref(), &path).await?;
                ensure!(
                    contents.is_none(),
                    "Native Cargo configuration appeared during discovery"
                );
                loader.native_seals.push(seal);
                continue;
            }
            match smol::fs::metadata(&path).await {
                Ok(_) => {
                    selected = Some(path);
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(anyhow!("Cargo configuration cannot be inspected")),
            }
        }
        if let Some(path) = selected {
            configuration.merge_values(loader.load(path, false, 0).await?)?;
        }
    }
    if let Some(filesystem) = &loader.native_filesystem {
        revalidate_native_cargo_configuration(filesystem.as_ref(), &loader.native_seals).await?;
    }
    Ok((configuration, loader.native_seals))
}

async fn prepare_passive_rust_execution(
    program: Option<&OsStr>,
    include_rustdoc: bool,
    working_directory: &Path,
    environment: Option<&HashMap<String, String>>,
    executor: &BackgroundExecutor,
) -> Result<PassiveRustExecution> {
    prepare_passive_rust_execution_in_environment(
        program,
        include_rustdoc,
        working_directory,
        PassiveRustEnvironment::from_request(environment),
        executor,
    )
    .await
}

async fn prepare_passive_rust_execution_in_environment(
    program: Option<&OsStr>,
    include_rustdoc: bool,
    working_directory: &Path,
    parent_environment: PassiveRustEnvironment,
    executor: &BackgroundExecutor,
) -> Result<PassiveRustExecution> {
    let preparation = async {
        let mut parent_environment = parent_environment;
        let cargo_parent_environment = if program.is_some_and(|program|
            Path::new(program).file_stem() == Some(OsStr::new("cargo"))) {
            parent_environment.pin_present_cargo_parent()
        } else { PassiveRustEnvironment::default() };
        let installed = if let Some(program) = program {
            check_passive_rustup_executable(program, working_directory, &parent_environment, executor).await?
        } else { None };
        let mut toolchain = installed.as_ref().map(|installed| installed.root.clone());
        let mut delegated_executables = if parent_environment.inherit_process { Vec::new() }
            else { installed.map(|installed| installed.executables).unwrap_or_default() };
        let mut child_environment = parent_environment.clone();
        let mut compiler_chain = Vec::new();
        let mut cargo_options = Vec::new();
        if program.is_none_or(|program| Path::new(program).file_stem() == Some(OsStr::new("cargo"))) {
            let configuration = passive_cargo_configuration(working_directory, &parent_environment).await?;
            for (name, value) in &configuration.environment {
                let is_variable = |expected: &str| name == expected || (cfg!(windows) && name.eq_ignore_ascii_case(expected));
                if ["CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN"].into_iter().any(is_variable) {
                    return Err(anyhow!("Cargo environment configuration cannot select toolchain homes; configure the host environment instead"));
                }
                let (value, force) = value.resolve()?;
                if force || parent_environment.value(name).is_none() {
                    child_environment.insert(name, value);
                }
            }
            let executable = |names: &[&str], configured: Option<&crate::cargo_workspace::CargoConfigurationString>| {
                names.iter().find_map(|name| parent_environment.value(name))
                    .or_else(|| configured.map(|value| value.executable().into_os_string()))
            };
            let mut selected_executables = Vec::new();
            for (field, names, configured, default) in [
                ("rustc-wrapper", ["RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER"], configuration.wrapper.as_ref(), ""),
                ("rustc-workspace-wrapper", ["RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"], configuration.workspace_wrapper.as_ref(), ""),
                ("rustc", ["RUSTC", "CARGO_BUILD_RUSTC"], configuration.compiler.as_ref(), "rustc"),
            ] {
                let selected = executable(&names, configured).unwrap_or_else(|| default.into());
                if field == "rustc" || !selected.is_empty() { compiler_chain.push(selected.clone()); }
                selected_executables.push((field, selected));
            }
            if include_rustdoc {
                selected_executables.push(("rustdoc", executable(&["RUSTDOC", "CARGO_BUILD_RUSTDOC"], configuration.documentation_generator.as_ref())
                    .unwrap_or_else(|| "rustdoc".into())));
            }
            let mut admitted_executables = selected_executables.iter()
                .map(|(_, selected)| selected.clone()).collect::<Vec<_>>();
            // A configured wrapper can delegate through the child environment,
            // whose forced values need not match Cargo's parent-side selection.
            for name in configuration.environment.keys() {
                let is_variable = |expected: &str| name == expected || (cfg!(windows) && name.eq_ignore_ascii_case(expected));
                let compiler = [
                    "RUSTC", "CARGO_BUILD_RUSTC", "RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER",
                    "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
                ].into_iter().any(is_variable);
                let documentation_generator = include_rustdoc && ["RUSTDOC", "CARGO_BUILD_RUSTDOC"].into_iter().any(is_variable);
                if (compiler || documentation_generator) && let Some(selected) = child_environment.value(name)
                    && !admitted_executables.contains(&selected) {
                    admitted_executables.push(selected);
                }
            }
            child_environment.insert("RUSTUP_AUTO_INSTALL", "0".into());
            child_environment.insert("CARGO_NET_OFFLINE", "true".into());
            // Cargo rereads configuration after admission. Preserve its TOML
            // value types so later merges cannot select an unreviewed child value.
            if program.is_some() {
                for (name, original) in &configuration.environment {
                    let is_variable = |expected: &str| name == expected || (cfg!(windows) && name.eq_ignore_ascii_case(expected));
                    let policy_flag = ["RUSTUP_AUTO_INSTALL", "CARGO_NET_OFFLINE"].into_iter().any(is_variable);
                    if !policy_flag && !PASSIVE_CARGO_PROTECTED_ENVIRONMENT.into_iter().any(is_variable) { continue; }
                    let value = child_environment.value(name)
                        .context("Admitted Cargo child environment is unavailable")?;
                    let value = value.to_str()
                        .context("Admitted Cargo child environment is not UTF-8")?;
                    let value = toml::Value::String(value.to_owned()).to_string();
                    match original {
                        crate::cargo_workspace::CargoConfigurationEnvironment::String(_) => {
                            cargo_options.extend([OsString::from("--config"), format!("env.{name}={value}").into()]);
                        }
                        crate::cargo_workspace::CargoConfigurationEnvironment::Table { .. } => {
                            for field in [
                                format!("env.{name}.force={}", !policy_flag),
                                format!("env.{name}.value={value}"),
                                format!("env.{name}.relative=false"),
                            ] { cargo_options.extend([OsString::from("--config"), field.into()]); }
                        }
                    }
                }
                for (name, value) in [("RUSTUP_AUTO_INSTALL", "0"), ("CARGO_NET_OFFLINE", "true")] {
                    if configuration.environment.keys().any(|configured| configured == name
                        || (cfg!(windows) && configured.eq_ignore_ascii_case(name))) { continue; }
                    // These values already exist in the protected parent environment.
                    // Seal absent entries too so a later forced table cannot replace them.
                    for field in [
                        format!("env.{name}.force=false"),
                        format!("env.{name}.value={}", toml::Value::String(value.into())),
                        format!("env.{name}.relative=false"),
                    ] { cargo_options.extend([OsString::from("--config"), field.into()]); }
                }
            }
            if program.is_some() {
                for name in PASSIVE_CARGO_PROTECTED_ENVIRONMENT {
                    if !cargo_parent_environment.overrides.contains_key(OsStr::new(name))
                        || configuration.environment.keys().any(|configured| configured == name
                            || (cfg!(windows) && configured.eq_ignore_ascii_case(name))) { continue; }
                    // Cargo requires a value, but force=false never applies it:
                    // the admitted Some (including empty/non-UTF8) is pinned in the parent.
                    for field in [
                        format!("env.{name}.force=false"),
                        format!("env.{name}.value=\"admitted-parent-present\""),
                        format!("env.{name}.relative=false"),
                    ] { cargo_options.extend([OsString::from("--config"), field.into()]); }
                }
            }
            // CLI build fields outrank reread file configuration and Cargo's
            // build aliases, while dedicated compiler variables keep precedence.
            for (field, selected) in &selected_executables {
                if program.is_some() {
                    let value = selected.to_str().context("Admitted Cargo executable is not UTF-8")?;
                    let value = toml::Value::String(value.to_owned()).to_string();
                    cargo_options.extend([OsString::from("--config"), format!("build.{field}={value}").into()]);
                }
            }
            for selected in &admitted_executables {
                if selected.is_empty() { continue; }
                if let Some(selected) = check_passive_rustup_executable(selected, working_directory, &child_environment, executor).await? {
                    if toolchain.as_ref().is_some_and(|root| root != &selected.root) {
                        return Err(anyhow!("Passive Cargo executables select conflicting installed toolchains"));
                    }
                    if !parent_environment.inherit_process {
                        for executable in selected.executables {
                            if !delegated_executables.contains(&executable) { delegated_executables.push(executable); }
                        }
                    }
                    toolchain = Some(selected.root);
                }
            }
        }
        if let Some(toolchain) = &toolchain {
            child_environment.insert("RUSTUP_TOOLCHAIN", toolchain.as_os_str().into());
        }
        Ok(PassiveRustExecution {
            child_environment,
            cargo_parent_environment,
            compiler_chain,
            #[cfg(target_os = "macos")]
            delegated_executables,
            toolchain,
            cargo_options,
        })
    }.boxed();
    match futures::future::select(
        preparation,
        executor.timer(PASSIVE_RUSTUP_POLICY_TIMEOUT).boxed(),
    )
    .await
    {
        Either::Left((result, _)) => result,
        Either::Right(_) => Err(anyhow!("Rustup passive-download policy check timed out")),
    }
}

pub(crate) async fn prevent_passive_rust_downloads(
    command: &mut util::command::Command,
    include_rustdoc: bool,
    working_directory: &Path,
    environment: Option<&HashMap<String, String>>,
    executor: &BackgroundExecutor,
) -> Result<Vec<OsString>> {
    // Cargo offline does not stop a Rustup proxy, and forced file environment
    // values can override parent flags unless Cargo receives policy options first.
    let execution = prepare_passive_rust_execution(
        Some(command.get_program()),
        include_rustdoc,
        working_directory,
        environment,
        executor,
    )
    .await?;
    execution.cargo_parent_environment.apply_to_command(command);
    command
        .env("RUSTUP_AUTO_INSTALL", "0")
        .env("CARGO_NET_OFFLINE", "true");
    if let Some(toolchain) = execution.toolchain {
        command.env("RUSTUP_TOOLCHAIN", toolchain);
    }
    Ok(execution.cargo_options)
}

struct InstalledRustEnvironmentToolchain {
    root: PathBuf,
    executables: Vec<PathBuf>,
}

async fn check_passive_rustup_executable(
    program: &OsStr,
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
    executor: &BackgroundExecutor,
) -> Result<Option<InstalledRustEnvironmentToolchain>> {
    let path = environment.path_for_executable_lookup();
    let executable = match which::which_in(program, path.as_ref(), working_directory) {
        Ok(executable) => executable,
        // Preserve the collector's normal executable-start error when PATH has no match.
        Err(which::Error::CannotFindBinaryPath) => return Ok(None),
        Err(_) => {
            return Err(anyhow!(
                "Cannot resolve the passive Rust collector executable"
            ));
        }
    };
    let manager_name = if cfg!(windows) {
        "rustup.exe"
    } else {
        "rustup"
    };
    let mut candidates = Vec::new();
    if let Some(parent) = executable.parent() {
        candidates.push(parent.join(manager_name));
    }
    match which::which_in(manager_name, path.as_ref(), working_directory) {
        Ok(manager) if !candidates.contains(&manager) => candidates.push(manager),
        Ok(_) | Err(which::Error::CannotFindBinaryPath) => {}
        Err(_) => return Err(anyhow!("Cannot resolve the Rustup policy executable")),
    }
    for manager in candidates {
        if !same_rustup_executable(&executable, &manager).await? {
            continue;
        }
        let mut probe = util::command::new_command(&manager);
        probe.arg("--version").current_dir(working_directory);
        environment.apply_to_command(&mut probe);
        // Older `rustup --version` also resolves rustc and may install a project toolchain.
        // A custom, uninstalled sentinel prevents that secondary selection from installing.
        probe
            .env(
                "RUSTUP_TOOLCHAIN",
                format!(
                    "rust-workspace-passive-version-probe-{:032x}",
                    rand::random::<u128>()
                ),
            )
            .env("RUSTUP_AUTO_INSTALL", "0")
            .env("CARGO_NET_OFFLINE", "true");
        let output = bounded_cargo_command_output(
            probe,
            "Rustup policy probe",
            MAX_ERROR_BYTES,
            PASSIVE_RUSTUP_POLICY_TIMEOUT,
            executor,
        )
        .await?;
        let version = std::str::from_utf8(&output.stdout)
            .ok()
            .and_then(|output| output.strip_prefix("rustup "))
            .and_then(|output| output.split_whitespace().next());
        let version = version.and_then(|version| semver::Version::parse(version).ok());
        let supported = version.as_ref().is_some_and(|version| {
            version.major == 1 && version.minor >= 28 && version.pre.is_empty()
        });
        if output.status.success()
            && version.as_ref().is_some_and(|version| {
                version.major == 1
                    && version.minor == 27
                    && version.patch == 1
                    && version.pre.is_empty()
            })
        {
            let installed =
                installed_environment_toolchain(&manager, working_directory, environment, executor)
                    .await?;
            if Path::new(program).file_stem() == Some(OsStr::new("rustdoc")) {
                let executable = installed.root.join("bin").join(if cfg!(windows) {
                    "rustdoc.exe"
                } else {
                    "rustdoc"
                });
                let executable = smol::fs::canonicalize(executable).await
                    .map_err(|_| anyhow!("The selected installed Rust toolchain has no verified documentation generator"))?;
                if executable.parent() != Some(installed.root.join("bin").as_path())
                    || !smol::fs::metadata(executable)
                        .await
                        .map_err(|_| {
                            anyhow!("The installed documentation generator cannot be inspected")
                        })?
                        .is_file()
                {
                    return Err(anyhow!(
                        "The installed documentation generator cannot be verified"
                    ));
                }
            }
            return Ok(Some(installed));
        }
        if !output.status.success() || !supported {
            return Err(anyhow!(
                "The configured Rustup proxy cannot guarantee passive discovery without toolchain installation. Update Rustup manually or configure installed compiler binaries; Cargo/Tests will not install a toolchain."
            ));
        }
        if !environment.inherit_process {
            return installed_environment_toolchain(
                &manager,
                working_directory,
                environment,
                executor,
            )
            .await
            .map(Some);
        }
    }
    Ok(None)
}

async fn installed_environment_toolchain(
    manager: &Path,
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
    executor: &BackgroundExecutor,
) -> Result<InstalledRustEnvironmentToolchain> {
    const UNAVAILABLE: &str = "The selected installed Rust toolchain cannot be verified for passive discovery. Configure an installed Rust toolchain or update Rustup manually; Cargo/Tests will not install a toolchain.";
    let selection = environment
        .value("RUSTUP_TOOLCHAIN")
        .filter(|selection| !selection.is_empty());
    let selected = match selection {
        Some(selection) => InstalledRustToolchainSelection {
            selection,
            components: Vec::new(),
            targets: Vec::new(),
        },
        None => {
            legacy_directory_toolchain_selection(manager, working_directory, environment, executor)
                .await
                .map_err(|_| anyhow!(UNAVAILABLE))?
        }
    };
    let selection = &selected.selection;
    let selection_path = Path::new(selection);
    let absolute_root = if selection_path.is_absolute() {
        Some(
            smol::fs::canonicalize(selection_path)
                .await
                .map_err(|_| anyhow!(UNAVAILABLE))?,
        )
    } else {
        None
    };
    let mut installed_root = None;
    let mut executables = Vec::new();
    for binary in ["rustc", "cargo"] {
        let mut probe = util::command::new_command(manager);
        probe.arg("which");
        if absolute_root.is_none() {
            probe.arg("--toolchain").arg(selection);
        }
        probe.arg(binary).current_dir(working_directory);
        environment.apply_to_command(&mut probe);
        if let Some(root) = &absolute_root {
            // Legacy CLI --toolchain rejects absolute paths. Implicit which is safe
            // only when its environment is locked to an existing absolute selection.
            probe.env("RUSTUP_TOOLCHAIN", root);
        }
        probe
            .env("RUSTUP_AUTO_INSTALL", "0")
            .env("CARGO_NET_OFFLINE", "true");
        let output = bounded_cargo_command_output(
            probe,
            "Installed Rustup selection",
            MAX_ERROR_BYTES,
            PASSIVE_RUSTUP_POLICY_TIMEOUT,
            executor,
        )
        .await
        .map_err(|_| anyhow!(UNAVAILABLE))?;
        if !output.status.success() {
            return Err(anyhow!(UNAVAILABLE));
        }
        let path = std::str::from_utf8(&output.stdout)
            .ok()
            .and_then(|path| path.strip_suffix('\n'))
            .map(|path| path.strip_suffix('\r').unwrap_or(path))
            .map(Path::new)
            .filter(|path| path.is_absolute())
            .ok_or_else(|| anyhow!(UNAVAILABLE))?;
        let path = smol::fs::canonicalize(path)
            .await
            .map_err(|_| anyhow!(UNAVAILABLE))?;
        let root = path
            .parent()
            .filter(|parent| parent.file_name() == Some(OsStr::new("bin")))
            .and_then(Path::parent)
            .ok_or_else(|| anyhow!(UNAVAILABLE))?;
        if !smol::fs::metadata(&path)
            .await
            .map_err(|_| anyhow!(UNAVAILABLE))?
            .is_file()
            || absolute_root
                .as_ref()
                .is_some_and(|expected| expected != root)
            || installed_root
                .as_ref()
                .is_some_and(|expected| expected != root)
        {
            return Err(anyhow!(UNAVAILABLE));
        }
        // A custom toolchain's Cargo fallback must not silently select a default.
        installed_root = Some(root.to_path_buf());
        executables.push(path);
    }
    let installed_root = installed_root.ok_or_else(|| anyhow!(UNAVAILABLE))?;
    if !selected.components.is_empty() || !selected.targets.is_empty() {
        verify_installed_rustup_requirements(
            manager,
            &selected,
            &installed_root,
            working_directory,
            environment,
            executor,
        )
        .await
        .map_err(|_| anyhow!(UNAVAILABLE))?;
    }
    Ok(InstalledRustEnvironmentToolchain {
        root: installed_root,
        executables,
    })
}

struct InstalledRustToolchainSelection {
    selection: std::ffi::OsString,
    components: Vec<String>,
    targets: Vec<String>,
}

async fn legacy_directory_toolchain_selection(
    manager: &Path,
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
    executor: &BackgroundExecutor,
) -> Result<InstalledRustToolchainSelection> {
    let overrides = legacy_rustup_read_only_query(
        manager,
        &["override", "list"],
        working_directory,
        environment,
        executor,
    )
    .await?;
    let mut directory_overrides = HashMap::default();
    for line in overrides.lines() {
        if line == "no overrides" {
            continue;
        }
        let (directory, selection) = line
            .split_once('\t')
            .filter(|(directory, selection)| {
                Path::new(directory).is_absolute()
                    && !selection.is_empty()
                    && !selection.contains('\t')
            })
            .ok_or_else(|| anyhow!("invalid installed Rustup directory listing"))?;
        if directory_overrides.len() >= crate::cargo_workspace::MAX_CARGO_CONFIGURATION_ITEMS
            || directory_overrides
                .insert(PathBuf::from(directory), selection.to_string())
                .is_some()
        {
            return Err(anyhow!("unsupported installed Rustup directory listing"));
        }
    }
    let working_directory = smol::fs::canonicalize(working_directory).await?;
    let mut requested_components = Vec::new();
    let mut requested_targets = Vec::new();
    'directories: for (depth, directory) in working_directory.ancestors().enumerate() {
        if depth >= crate::cargo_workspace::MAX_CARGO_CONFIGURATION_ITEMS {
            return Err(anyhow!(
                "Rustup directory search exceeds the supported limit"
            ));
        }
        if let Some(selection) = directory_overrides.get(directory) {
            return Ok(InstalledRustToolchainSelection {
                selection: selection.into(),
                components: requested_components,
                targets: requested_targets,
            });
        }
        for (name, toml_only) in [("rust-toolchain", false), ("rust-toolchain.toml", true)] {
            let file = match smol::fs::File::open(directory.join(name)).await {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            let mut contents = Vec::new();
            file.take((MAX_RUST_TOOLCHAIN_FILE_BYTES + 1) as u64)
                .read_to_end(&mut contents)
                .await?;
            let contents = std::str::from_utf8(&contents)?;
            let configuration = parse_rust_toolchain_configuration(contents, toml_only)?;
            requested_components = configuration.components.unwrap_or_default();
            requested_targets = configuration.targets.unwrap_or_default();
            if let Some(path) = configuration.path {
                return Ok(InstalledRustToolchainSelection {
                    selection: path.into_os_string(),
                    components: requested_components,
                    targets: requested_targets,
                });
            }
            if let Some(channel) = configuration.channel {
                return Ok(InstalledRustToolchainSelection {
                    selection: channel.into(),
                    components: requested_components,
                    targets: requested_targets,
                });
            }
            // A file requesting only components/targets inherits the configured
            // default, not an override in a parent directory.
            break 'directories;
        }
    }
    let default = legacy_rustup_read_only_query(
        manager,
        &["default"],
        &working_directory,
        environment,
        executor,
    )
    .await?;
    let selection = default
        .strip_suffix('\n')
        .map(|value| value.strip_suffix('\r').unwrap_or(value))
        .and_then(|value| value.strip_suffix(" (default)"))
        .filter(|value| !value.is_empty() && !value.contains(['\n', '\r', '\t']))
        .ok_or_else(|| anyhow!("invalid installed Rustup default listing"))?;
    Ok(InstalledRustToolchainSelection {
        selection: selection.into(),
        components: requested_components,
        targets: requested_targets,
    })
}

async fn verify_installed_rustup_requirements(
    manager: &Path,
    selected: &InstalledRustToolchainSelection,
    installed_root: &Path,
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
    executor: &BackgroundExecutor,
) -> Result<()> {
    let selection = selected
        .selection
        .to_str()
        .context("invalid installed Rustup selection")?;
    let targets = legacy_rustup_read_only_query(
        manager,
        &["target", "list", "--installed", "--toolchain", selection],
        working_directory,
        environment,
        executor,
    )
    .await?;
    let installed_targets = targets.lines().collect::<HashSet<_>>();
    if !selected.components.is_empty() {
        let components = legacy_rustup_read_only_query(
            manager,
            &["component", "list", "--installed", "--toolchain", selection],
            working_directory,
            environment,
            executor,
        )
        .await?;
        let mut hosts = components
            .lines()
            .filter_map(|component| component.strip_prefix("rustc-"));
        let host = hosts
            .next()
            .context("installed Rustup compiler component is unavailable")?;
        if hosts.next().is_some() || !installed_targets.contains(host) {
            return Err(anyhow!("installed Rustup compiler component is ambiguous"));
        }
        let host_suffix = format!("-{host}");
        let installed = components
            .lines()
            .filter_map(|component| {
                component.strip_suffix(&host_suffix).or_else(|| {
                    (!installed_targets
                        .iter()
                        .any(|target| component.ends_with(format!("-{target}").as_str())))
                    .then_some(component)
                })
            })
            .collect::<HashSet<_>>();
        let missing = selected
            .components
            .iter()
            .filter(|component| !installed.contains(component.as_str()))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            // Rustup accepts manifest package names as well as renamed display names.
            // Resolve only against this installed toolchain, never a downloaded manifest.
            let file = smol::fs::File::open(
                installed_root.join("lib/rustlib/multirust-channel-manifest.toml"),
            )
            .await?;
            const MAX_INSTALLED_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
            let mut contents = Vec::new();
            file.take(MAX_INSTALLED_MANIFEST_BYTES + 1)
                .read_to_end(&mut contents)
                .await?;
            if contents.len() as u64 > MAX_INSTALLED_MANIFEST_BYTES {
                return Err(anyhow!(
                    "installed Rustup manifest exceeds the supported limit"
                ));
            }
            let aliases = installed_rustup_component_aliases(std::str::from_utf8(&contents)?)?;
            for component in missing {
                if aliases
                    .get(component.as_str())
                    .is_some_and(|canonical| installed.contains(canonical.as_str()))
                {
                    continue;
                }
                return Err(anyhow!(
                    "a requested Rustup component is not verified installed"
                ));
            }
        }
    }
    for target in &selected.targets {
        if !installed_targets.contains(target.as_str()) {
            return Err(anyhow!(
                "a requested Rustup target is not verified installed"
            ));
        }
    }
    Ok(())
}

fn installed_rustup_component_aliases(contents: &str) -> Result<HashMap<String, String>> {
    let manifest = toml::from_str::<toml::Value>(contents)
        .map_err(|_| anyhow!("invalid installed Rustup manifest"))?;
    let renames = manifest
        .get("renames")
        .and_then(toml::Value::as_table)
        .context("installed Rustup component aliases are unavailable")?;
    if renames.len() > crate::cargo_workspace::MAX_CARGO_CONFIGURATION_ITEMS {
        return Err(anyhow!(
            "installed Rustup alias inventory exceeds the supported limit"
        ));
    }
    let mut aliases = HashMap::default();
    for (canonical, rename) in renames {
        let alias = rename
            .get("to")
            .and_then(toml::Value::as_str)
            .context("invalid installed Rustup component alias")?;
        if canonical.is_empty()
            || alias.is_empty()
            || canonical.len() > crate::cargo_workspace::MAX_CARGO_CONFIGURATION_FIELD_BYTES
            || alias.len() > crate::cargo_workspace::MAX_CARGO_CONFIGURATION_FIELD_BYTES
            || aliases
                .insert(alias.to_string(), canonical.clone())
                .is_some()
        {
            return Err(anyhow!("unsupported installed Rustup component alias"));
        }
    }
    Ok(aliases)
}

async fn legacy_rustup_read_only_query(
    manager: &Path,
    arguments: &[&str],
    working_directory: &Path,
    environment: &PassiveRustEnvironment,
    executor: &BackgroundExecutor,
) -> Result<String> {
    let mut command = util::command::new_command(manager);
    command.args(arguments).current_dir(working_directory);
    environment.apply_to_command(&mut command);
    command
        .env(
            "RUSTUP_TOOLCHAIN",
            "rust-workspace-passive-policy-uninstalled",
        )
        .env("RUSTUP_AUTO_INSTALL", "0")
        .env("CARGO_NET_OFFLINE", "true");
    let output = bounded_cargo_command_output(
        command,
        "Installed Rustup configuration",
        MAX_RUST_TOOLCHAIN_FILE_BYTES,
        PASSIVE_RUSTUP_POLICY_TIMEOUT,
        executor,
    )
    .await?;
    if !output.status.success() {
        return Err(anyhow!("installed Rustup configuration is unavailable"));
    }
    Ok(String::from_utf8(output.stdout)?)
}

async fn same_rustup_executable(executable: &Path, manager: &Path) -> Result<bool> {
    let manager_metadata = match smol::fs::metadata(manager).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).context("failed to inspect Rustup policy executable"),
    };
    let executable_metadata = smol::fs::metadata(executable)
        .await
        .context("failed to inspect passive Rust collector executable")?;
    if !manager_metadata.is_file() || executable_metadata.len() != manager_metadata.len() {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if executable_metadata.dev() == manager_metadata.dev()
            && executable_metadata.ino() == manager_metadata.ino()
        {
            return Ok(true);
        }
    }
    if executable_metadata.len() > MAX_RUSTUP_PROXY_BYTES {
        return Err(anyhow!(
            "Rustup proxy identification exceeded the supported file limit"
        ));
    }
    // Windows proxies and manual copies can have different file identities.
    let executable_digest = rustup_executable_digest(executable).await?;
    let manager_digest = rustup_executable_digest(manager).await?;
    Ok(executable_digest == manager_digest)
}

async fn rustup_executable_digest(path: &Path) -> Result<sha2::digest::Output<Sha256>> {
    let mut reader = smol::fs::File::open(path)
        .await
        .context("failed to read Rustup proxy identity")?
        .take(MAX_RUSTUP_PROXY_BYTES + 1);
    let mut digest = Sha256::new();
    let mut buffer = [0; 8192];
    let mut bytes = 0;
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(digest.finalize());
        }
        bytes += read as u64;
        if bytes > MAX_RUSTUP_PROXY_BYTES {
            return Err(anyhow!(
                "Rustup proxy identification exceeded the supported file limit"
            ));
        }
        digest.update(&buffer[..read]);
    }
}

async fn bounded_passive_cargo_command_output(
    mut command: util::command::Command,
    arguments: Vec<OsString>,
    name: &'static str,
    stdout_limit: usize,
    timeout: Duration,
    working_directory: &Path,
    environment: Option<&HashMap<String, String>>,
    executor: &BackgroundExecutor,
) -> Result<std::process::Output> {
    let collection = async {
        if let Some(environment) = environment {
            command.envs(environment);
        }
        let options = prevent_passive_rust_downloads(
            &mut command,
            false,
            working_directory,
            environment,
            executor,
        )
        .await?;
        command.args(options).args(arguments);
        bounded_cargo_command_output(command, name, stdout_limit, timeout, executor).await
    }
    .boxed();
    match futures::future::select(collection, executor.timer(timeout).boxed()).await {
        Either::Left((output, _)) => output,
        Either::Right(_) => Err(anyhow!("{name} timed out")),
    }
}

#[cfg(all(target_os = "macos", feature = "test-support"))]
pub async fn compile_native_profile_test_workload(
    source: &Path,
    executable: &Path,
    executor: &BackgroundExecutor,
) -> Result<()> {
    let root = source
        .parent()
        .context("Rust workload source directory is absent")?;
    ensure!(
        source.is_absolute() && executable.is_absolute() && executable.parent() == Some(root),
        "Rust workload paths must share one owned absolute directory"
    );
    let execution = prepare_passive_rust_execution(None, false, root, None, executor).await?;
    let mut executables = execution.compiler_chain.into_iter();
    let program = executables
        .next()
        .context("Installed workload compiler is unavailable")?;
    let mut command = util::command::new_command(program);
    command
        .args(executables)
        .args(["--crate-name", "native_selected_fixture", "--edition=2024"])
        .arg(source)
        .arg("-o")
        .arg(executable)
        .envs(&execution.child_environment.overrides)
        .current_dir(root);
    let output = bounded_cargo_command_output(
        command,
        "disposable native Rust workload compilation",
        MAX_ERROR_BYTES,
        CARGO_METADATA_TIMEOUT,
        executor,
    )
    .await?;
    ensure!(
        output.status.success(),
        "Rust workload compilation failed: {}",
        bounded_error(&output.stderr)
    );
    Ok(())
}

pub(crate) async fn bounded_cargo_command_output(
    mut command: util::command::Command,
    name: &'static str,
    stdout_limit: usize,
    timeout: Duration,
    executor: &BackgroundExecutor,
) -> Result<std::process::Output> {
    let output = async move {
        let mut child = command
            .stdin(util::command::Stdio::null())
            .stdout(util::command::Stdio::piped())
            .stderr(util::command::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to start {name}"))?;
        let stdout = child
            .stdout
            .take()
            .context("missing Cargo collector stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("missing Cargo collector stderr")?;
        let (stdout, stderr) = futures::try_join!(
            bounded_cargo_command_stream(stdout, stdout_limit, name, "stdout"),
            bounded_cargo_command_stream(stderr, MAX_ERROR_BYTES, name, "stderr"),
        )?;
        let status = child
            .status()
            .await
            .with_context(|| format!("failed to wait for {name}"))?;
        anyhow::Ok(std::process::Output {
            status,
            stdout,
            stderr,
        })
    }
    .boxed();
    match futures::future::select(output, executor.timer(timeout).boxed()).await {
        Either::Left((output, _)) => output,
        Either::Right(_) => Err(anyhow!("{name} timed out")),
    }
}

async fn bounded_cargo_command_stream(
    reader: impl futures::io::AsyncRead + Unpin + Send,
    limit: usize,
    name: &'static str,
    stream: &'static str,
) -> Result<Vec<u8>> {
    match bounded_cargo_metadata_stream(reader, limit).await {
        Ok(output) => Ok(output),
        Err(error) if error.is::<CargoProcessOutputLimitExceeded>() => {
            Err(anyhow!("{name} {stream} exceeded the supported limit"))
        }
        Err(error) => Err(error.context(format!("failed to read {name} {stream}"))),
    }
}

#[cfg(any(test, feature = "test-support"))]
struct UnavailableCargoConfigurationProbe;

#[async_trait]
impl CargoMetadataRunner for ProcessCargoMetadataRunner {
    async fn run(&self, request: CargoMetadataRequest) -> Result<Vec<u8>> {
        #[cfg(any(test, feature = "test-support"))]
        LOCAL_CARGO_PROCESS_CALLS.fetch_add(1, Ordering::Relaxed);
        let mut command = util::command::new_command("cargo");
        command
            .current_dir(&request.working_directory)
            .kill_on_drop(true);
        let mut arguments = [
            "metadata",
            "--offline",
            "--format-version",
            "1",
            "--manifest-path",
        ]
        .into_iter()
        .map(OsString::from)
        .collect::<Vec<_>>();
        arguments.push(request.manifest_path.into_os_string());
        let output = bounded_passive_cargo_command_output(
            command,
            arguments,
            "Cargo metadata",
            MAX_CARGO_METADATA_BYTES,
            CARGO_METADATA_TIMEOUT,
            &request.working_directory,
            request.environment.as_ref(),
            &self.executor,
        )
        .await?;
        if !output.status.success() {
            let message = bounded_error(&output.stderr);
            return Err(anyhow!(
                "Cargo metadata exited with {}: {message}",
                output.status
            ));
        }
        Ok(output.stdout)
    }
}

#[derive(Debug)]
struct CargoProcessOutputLimitExceeded;

impl std::fmt::Display for CargoProcessOutputLimitExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Cargo metadata process output exceeded its byte limit")
    }
}

impl std::error::Error for CargoProcessOutputLimitExceeded {}

async fn bounded_cargo_metadata_stream(
    reader: impl futures::AsyncRead + Unpin,
    maximum: usize,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(u64::try_from(maximum)?.saturating_add(1))
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > maximum {
        return Err(CargoProcessOutputLimitExceeded.into());
    }
    Ok(bytes)
}

#[async_trait]
impl CargoConfigurationProbe for ProcessCargoConfigurationProbe {
    async fn run(&self, request: CargoConfigurationProbeRequest) -> Result<Vec<u8>> {
        #[cfg(any(test, feature = "test-support"))]
        LOCAL_CARGO_PROCESS_CALLS.fetch_add(1, Ordering::Relaxed);
        let collection = async {
            let execution = prepare_passive_rust_execution(
                None,
                false,
                &request.working_directory,
                request.environment.as_ref(),
                &self.executor,
            )
            .await?;
            let mut executables = execution.compiler_chain.into_iter();
            let program = executables
                .next()
                .context("Cargo compiler selection is unavailable")?;
            let mut command = util::command::new_command(program);
            command
                .args(executables)
                .arg("-vV")
                .envs(&execution.child_environment.overrides)
                .current_dir(&request.working_directory)
                .kill_on_drop(true);
            bounded_cargo_command_output(
                command,
                "rustc -vV",
                MAX_RUSTC_VERBOSE_VERSION_BYTES,
                CARGO_CONFIGURATION_PROBE_TIMEOUT,
                &self.executor,
            )
            .await
        }
        .boxed();
        let output = match futures::future::select(
            collection,
            self.executor
                .timer(CARGO_CONFIGURATION_PROBE_TIMEOUT)
                .boxed(),
        )
        .await
        {
            Either::Left((output, _)) => output?,
            Either::Right(_) => return Err(anyhow!("rustc -vV timed out")),
        };
        if !output.status.success() {
            return Err(anyhow!(
                "rustc -vV exited with {}: {}",
                output.status,
                bounded_error(&output.stderr)
            ));
        }
        Ok(output.stdout)
    }
}

#[async_trait]
#[cfg(any(test, feature = "test-support"))]
impl CargoConfigurationProbe for UnavailableCargoConfigurationProbe {
    async fn run(&self, _request: CargoConfigurationProbeRequest) -> Result<Vec<u8>> {
        Err(anyhow!("rustc probe is unavailable in this test store"))
    }
}

#[derive(Clone, Debug)]
pub enum CargoWorkspaceStoreEvent {
    Invalidated,
}

impl EventEmitter<CargoWorkspaceStoreEvent> for CargoWorkspaceStore {}

pub struct CargoWorkspaceStore {
    mode: CargoWorkspaceStoreMode,
    worktree_store: Entity<WorktreeStore>,
    revision: u64,
    cached_snapshot: Option<CargoWorkspaceSnapshot>,
    cached_metadata_facts: Vec<CargoWorkspaceMetadataFacts>,
    cached_metadata_settings_revision: Option<u64>,
    cached_generation: u64,
    cached_scope: Option<HashSet<settings::WorktreeId>>,
    active_remote_request_id: Option<u64>,
    active_remote_request_abort: Option<AbortHandle>,
    disconnected: bool,
    cancelled_remote_requests: HashSet<(proto::PeerId, u64)>,
    active_remote_requests: HashMap<(proto::PeerId, u64), AbortHandle>,
    _subscriptions: Vec<Subscription>,
}

struct CargoMetadataCollection {
    snapshot: CargoWorkspaceSnapshot,
    metadata_facts: Vec<CargoWorkspaceMetadataFacts>,
}

#[derive(Clone)]
struct CargoWorkspaceMetadataFacts {
    runtime_dependencies: CargoNativeDependencyGraph,
    key: crate::cargo_workspace::CargoWorkspaceKey,
    target_directory: PathBuf,
    packages: Vec<CargoPackageMetadataFacts>,
}

#[derive(Clone)]
struct CargoPackageMetadataFacts {
    model_id: String,
    model_name: String,
    model_version: String,
    manifest_path: ProjectPath,
    package_id: String,
    targets: Vec<CargoTargetMetadataFacts>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CargoNativeTargetPolicyFacts {
    Reviewed(crate::cargo_workspace::CargoNativeExecutionPolicy),
    Unavailable(String),
}

#[derive(Clone)]
struct CargoTargetMetadataFacts {
    model: crate::cargo_workspace::CargoTargetModel,
    source_path: PathBuf,
    kinds: Vec<String>,
    execution_policy: CargoNativeTargetPolicyFacts,
}

#[derive(Clone)]
enum CargoNativeDependencyGraph {
    Reviewed(std::collections::BTreeMap<String, std::collections::BTreeSet<String>>),
    Unavailable,
}

impl CargoNativeDependencyGraph {
    fn from_metadata(metadata: &cargo_metadata::Metadata) -> Self {
        let read =
            || -> Result<std::collections::BTreeMap<String, std::collections::BTreeSet<String>>> {
                let resolve = metadata
                    .resolve
                    .as_ref()
                    .context("Cargo runtime dependency graph is unavailable")?;
                ensure!(
                    resolve.nodes.len() <= crate::cargo_workspace::MAX_CARGO_GRAPH_NODES,
                    "Cargo runtime dependency graph node limit reached"
                );
                let mut nodes = HashSet::new();
                let mut edges = 0_usize;
                let mut bytes = 0_usize;
                for node in &resolve.nodes {
                    ensure!(
                        nodes.insert(node.id.repr.as_str()),
                        "Cargo runtime dependency graph node is duplicated"
                    );
                    for id in std::iter::once(&node.id)
                        .chain(node.deps.iter().map(|dependency| &dependency.pkg))
                    {
                        ensure!(
                            !id.repr.is_empty()
                                && id.repr.len()
                                    <= crate::cargo_workspace::MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                                && !id.repr.chars().any(char::is_control),
                            "Cargo runtime dependency identity exceeds its bound"
                        );
                        bytes = bytes
                            .checked_add(id.repr.len() + 64)
                            .context("Cargo runtime graph byte count overflow")?;
                    }
                    let structured_ids = node
                        .deps
                        .iter()
                        .map(|dependency| dependency.pkg.repr.as_str())
                        .collect::<HashSet<_>>();
                    let legacy_ids = node
                        .dependencies
                        .iter()
                        .map(|id| id.repr.as_str())
                        .collect::<HashSet<_>>();
                    ensure!(
                        structured_ids == legacy_ids,
                        "Cargo runtime dependency representations disagree"
                    );
                    edges = edges
                        .checked_add(node.deps.len())
                        .context("Cargo runtime graph edge count overflow")?;
                    ensure!(
                        edges <= crate::cargo_workspace::MAX_CARGO_GRAPH_EDGES
                            && bytes <= crate::cargo_workspace::MAX_CARGO_SNAPSHOT_BYTES,
                        "Cargo runtime dependency graph retained limit reached"
                    );
                }
                ensure!(
                    metadata.packages.len() <= crate::cargo_workspace::MAX_CARGO_GRAPH_NODES,
                    "Cargo runtime metadata package count exceeds its bound"
                );
                let package_ids = metadata
                    .packages
                    .iter()
                    .map(|package| package.id.repr.as_str())
                    .collect::<HashSet<_>>();
                ensure!(
                    package_ids.len() == metadata.packages.len(),
                    "Cargo runtime metadata package identity is duplicated"
                );
                ensure!(
                    nodes.iter().all(|id| package_ids.contains(id))
                        && resolve.nodes.iter().all(|node| node
                            .deps
                            .iter()
                            .all(|dependency| nodes.contains(dependency.pkg.repr.as_str()))),
                    "Cargo runtime dependency graph identity is incomplete"
                );
                Ok(resolve
                    .nodes
                    .iter()
                    .map(|node| {
                        (
                            node.id.repr.clone(),
                            node.deps
                                .iter()
                                .map(|dependency| dependency.pkg.repr.clone())
                                .collect(),
                        )
                    })
                    .collect())
            };
        match read() {
            Ok(graph) => Self::Reviewed(graph),
            Err(_) => Self::Unavailable,
        }
    }

    fn selected_closure(&self, package: &str) -> Option<std::collections::BTreeSet<String>> {
        let Self::Reviewed(graph) = self else {
            return None;
        };
        if !graph.contains_key(package) {
            return None;
        }
        let mut visited = std::collections::BTreeSet::new();
        let mut pending = vec![package.to_owned()];
        while let Some(package) = pending.pop() {
            if !visited.insert(package.clone()) {
                continue;
            }
            let dependencies = graph.get(&package)?;
            pending.extend(
                dependencies
                    .iter()
                    .filter(|id| !visited.contains(*id))
                    .cloned(),
            );
        }
        Some(visited)
    }
}

impl CargoWorkspaceMetadataFacts {
    fn matches_workspace(&self, workspace: &CargoWorkspaceModel) -> bool {
        self.key == workspace.key
            && self.packages.len() == workspace.members.len()
            && self.packages.iter().all(|facts| {
                workspace.members.iter().any(|member| {
                    member.id == facts.model_id
                        && member.name == facts.model_name
                        && member.version == facts.model_version
                        && member.manifest_path == facts.manifest_path
                        && facts
                            .targets
                            .iter()
                            .all(|target| member.targets.contains(&target.model))
                })
            })
    }
}

fn metadata_facts_from_metadata(
    metadata: &cargo_metadata::Metadata,
    workspace: &CargoWorkspaceModel,
    mut resolve_path: impl FnMut(&Path) -> Option<ProjectPath>,
    manifests: &std::collections::BTreeMap<ProjectPath, String>,
) -> Result<CargoWorkspaceMetadataFacts> {
    ensure!(
        metadata.target_directory.is_absolute(),
        "Cargo metadata output root is not absolute"
    );
    let workspace_manifest = resolve_path(metadata.workspace_root.join("Cargo.toml").as_std_path())
        .filter(|path| workspace.root_manifest.as_ref() == Some(path))
        .and_then(|path| manifests.get(&path));
    let workspace_ids = metadata.workspace_members.iter().collect::<HashSet<_>>();
    let mut packages_by_selection = HashMap::default();
    for package in &metadata.packages {
        if !workspace_ids.contains(&package.id) {
            continue;
        }
        let Some(manifest_path) = resolve_path(package.manifest_path.as_std_path()) else {
            continue;
        };
        let key = (
            manifest_path,
            package.name.clone(),
            package.version.to_string(),
        );
        ensure!(
            packages_by_selection.insert(key, package).is_none(),
            "Cargo metadata member identity is ambiguous"
        );
    }
    let mut packages = Vec::new();
    let mut package_ids = HashSet::new();
    for member in &workspace.members {
        let package = packages_by_selection
            .get(&(
                member.manifest_path.clone(),
                member.name.clone(),
                member.version.clone(),
            ))
            .context("Cargo metadata member identity is unavailable")?;
        ensure!(
            !package.id.repr.is_empty() && package_ids.insert(package.id.repr.clone()),
            "Cargo metadata member identity is empty or duplicated"
        );
        let mut targets_by_selection = HashMap::default();
        for target in &package.targets {
            let Some(source) = resolve_path(target.src_path.as_std_path()) else {
                continue;
            };
            let key = (
                crate::cargo_workspace::normalize_target_kind(&target.kind),
                target.name.clone(),
                target.edition.to_string(),
                target.required_features.clone(),
                target
                    .crate_types
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                source,
            );
            ensure!(
                targets_by_selection.insert(key, target).is_none(),
                "Cargo metadata selected target is ambiguous"
            );
        }
        let mut targets = Vec::new();
        for model in &member.targets {
            let Some(source) = &model.source_path else {
                continue;
            };
            let key = (
                model.kind.clone(),
                model.name.clone(),
                model.edition.clone(),
                model.required_features.clone(),
                model.crate_types.clone(),
                source.clone(),
            );
            let target = targets_by_selection
                .get(&key)
                .context("Cargo metadata selected target is unavailable")?;
            ensure!(
                target.src_path.is_absolute(),
                "Cargo metadata target source is not absolute"
            );
            let policy = match model.kind {
                crate::cargo_workspace::CargoTargetKind::Binary
                | crate::cargo_workspace::CargoTargetKind::Example => {
                    Ok(crate::cargo_workspace::CargoNativeExecutionPolicy::Program)
                }
                _ => manifests
                    .get(&member.manifest_path)
                    .context("Native selected target manifest is unavailable")
                    .and_then(|manifest| {
                        crate::cargo_workspace::parse_native_cargo_execution_policy_with_workspace_manifest(
                            manifest,
                            package.manifest_path.as_std_path(),
                            &model.kind,
                            &model.name,
                            target.src_path.as_std_path(),
                            &model.edition,
                            workspace_manifest.map(String::as_str),
                        )
                    }),
            };
            let execution_policy = match policy {
                Ok(policy) => CargoNativeTargetPolicyFacts::Reviewed(policy),
                Err(error) => CargoNativeTargetPolicyFacts::Unavailable(error.to_string()),
            };
            targets.push(CargoTargetMetadataFacts {
                execution_policy,
                model: model.clone(),
                source_path: target.src_path.as_std_path().to_path_buf(),
                kinds: target.kind.iter().map(ToString::to_string).collect(),
            });
        }
        packages.push(CargoPackageMetadataFacts {
            model_id: member.id.clone(),
            model_name: member.name.clone(),
            model_version: member.version.clone(),
            manifest_path: member.manifest_path.clone(),
            package_id: package.id.repr.clone(),
            targets,
        });
    }
    let mut facts = CargoWorkspaceMetadataFacts {
        runtime_dependencies: CargoNativeDependencyGraph::from_metadata(metadata),
        key: workspace.key.clone(),
        target_directory: metadata.target_directory.as_std_path().to_path_buf(),
        packages,
    };
    if let Err(error) = validate_metadata_facts_bounds(std::slice::from_ref(&facts)) {
        if matches!(
            &facts.runtime_dependencies,
            CargoNativeDependencyGraph::Reviewed(_)
        ) {
            facts.runtime_dependencies = CargoNativeDependencyGraph::Unavailable;
            validate_metadata_facts_bounds(std::slice::from_ref(&facts))?;
        } else {
            return Err(error);
        }
    }
    Ok(facts)
}

fn validate_metadata_facts_bounds(workspaces: &[CargoWorkspaceMetadataFacts]) -> Result<()> {
    ensure!(
        workspaces.len() <= 128,
        "Cargo private metadata workspace count exceeded"
    );
    let mut bytes = 0_usize;
    let mut fields = 0_usize;
    let mut field = |value: &str| -> Result<()> {
        ensure!(
            value.len() <= 16 * 1024 && !value.contains('\0'),
            "Cargo private metadata field is invalid"
        );
        fields = fields
            .checked_add(1)
            .context("Cargo private metadata field count overflow")?;
        bytes = bytes
            .checked_add(value.len() + 64)
            .context("Cargo private metadata byte count overflow")?;
        ensure!(
            fields <= 65_536 && bytes <= crate::cargo_workspace::MAX_CARGO_SNAPSHOT_BYTES,
            "Cargo private metadata retained limit exceeded"
        );
        Ok(())
    };
    let mut keys = HashSet::new();
    for workspace in workspaces {
        ensure!(
            keys.insert(&workspace.key),
            "Cargo private metadata workspace is duplicated"
        );
        field(workspace.key.root.path.as_unix_str())?;
        field(
            workspace
                .target_directory
                .to_str()
                .context("Cargo metadata output root is not UTF-8")?,
        )?;
        if let CargoNativeDependencyGraph::Reviewed(graph) = &workspace.runtime_dependencies {
            for (package, dependencies) in graph {
                field(package)?;
                for dependency in dependencies {
                    field(dependency)?;
                }
            }
        }
        for package in &workspace.packages {
            field(&package.model_id)?;
            field(&package.model_name)?;
            field(&package.model_version)?;
            field(package.manifest_path.path.as_unix_str())?;
            field(&package.package_id)?;
            for target in &package.targets {
                if let CargoNativeTargetPolicyFacts::Unavailable(reason) = &target.execution_policy
                {
                    field(reason)?;
                }
                field(&target.model.name)?;
                field(&target.model.edition)?;
                if let crate::cargo_workspace::CargoTargetKind::Other(kind) = &target.model.kind {
                    field(kind)?;
                }
                if let Some(source) = &target.model.source_path {
                    field(source.path.as_unix_str())?;
                }
                if let Some(source) = &target.model.source_display_path {
                    field(source)?;
                }
                field(
                    target
                        .source_path
                        .to_str()
                        .context("Cargo metadata source path is not UTF-8")?,
                )?;
                for value in target
                    .kinds
                    .iter()
                    .chain(&target.model.crate_types)
                    .chain(&target.model.required_features)
                {
                    field(value)?;
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct CargoReviewedMetadataImport {
    revision: u64,
    input_fingerprint: u64,
    settings_revision: u64,
    manifest_path: ProjectPath,
    root: PathBuf,
    root_inode: u64,
    source_snapshot: worktree::Snapshot,
    files: Vec<(ProjectPath, u64, Option<fs::MTime>, u64)>,
}

impl CargoReviewedMetadataImport {
    pub fn manifest_path(&self) -> &ProjectPath {
        &self.manifest_path
    }

    pub async fn preflight_filesystem(&self, filesystem: &dyn fs::Fs) -> Result<()> {
        let root = filesystem
            .metadata(&self.root)
            .await?
            .context("Cargo root is unavailable")?;
        ensure!(
            root.is_dir && !root.is_symlink && root.inode == self.root_inode,
            "Reviewed Cargo root changed"
        );
        for (path, inode, mtime, size) in &self.files {
            let metadata = filesystem
                .metadata(&self.root.join(path.path.as_std_path()))
                .await?
                .context("Reviewed Cargo source disappeared")?;
            #[cfg(any(test, feature = "test-support"))]
            if metadata.inode != *inode
                || Some(metadata.mtime) != *mtime
                || metadata.len != *size
                || metadata.is_symlink
                || metadata.is_dir
                || metadata.is_fifo
            {
                let basename: String = path
                    .path
                    .as_std_path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("<non-UTF-8>")
                    .chars()
                    .filter(|character| !character.is_control())
                    .take(128)
                    .collect();
                eprintln!(
                    "reviewed Cargo input changed: basename={basename} inode_equal={} mtime_equal={} size_equal={} expected_size={} actual_size={} ordinary={} symlink={} directory={} fifo={}",
                    metadata.inode == *inode,
                    Some(metadata.mtime) == *mtime,
                    metadata.len == *size,
                    size,
                    metadata.len,
                    !metadata.is_symlink && !metadata.is_dir && !metadata.is_fifo,
                    metadata.is_symlink,
                    metadata.is_dir,
                    metadata.is_fifo,
                );
            }
            ensure!(
                metadata.inode == *inode
                    && Some(metadata.mtime) == *mtime
                    && metadata.len == *size
                    && !metadata.is_symlink
                    && !metadata.is_dir
                    && !metadata.is_fifo,
                "Reviewed Cargo source changed or is not ordinary"
            );
            let confined = filesystem
                .open_sync_confined_file(
                    &self.root,
                    path.path.as_std_path(),
                    self.root_inode,
                    *inode,
                )
                .await?;
            drop(confined);
        }
        Ok(())
    }
}

pub struct NativeCargoBuildRecipe {
    pub program: PathBuf,
    pub arguments: Vec<String>,
    pub working_directory: PathBuf,
    pub task_environment: HashMap<String, String>,
}

pub struct CargoNativeBuildIntent {
    required_experimental_features: Vec<String>,
    project: gpui::WeakEntity<crate::Project>,
    run_id: uuid::Uuid,
    task_id: task::TaskId,
    invocation_output_root: PathBuf,
    logical_arguments: Vec<String>,
    owner: gpui::EntityId,
    workspace: crate::cargo_workspace::CargoWorkspaceKey,
    member_id: String,
    target: crate::cargo_workspace::CargoTargetModel,
    profile: Option<String>,
    selection: crate::cargo_workspace::CargoExecutableSelection,
    runtime_packages: Option<std::collections::BTreeSet<String>>,
    reviewed: CargoReviewedMetadataImport,
    source_entries: Vec<worktree::Entry>,
    buffer_versions: Vec<(gpui::WeakEntity<language::Buffer>, clock::Global)>,
    sources: Vec<NativeProbeFileSeal>,
    directories: Vec<NativeProbeDirectorySeal>,
    requested_program: PathBuf,
    program: NativeProbeFileSeal,
    arguments: Vec<String>,
    cwd: PathBuf,
    cwd_inode: u64,
    package_source_directory: PathBuf,
    package_source_directory_inode: u64,
    output_root: PathBuf,
    output_root_inode: u64,
    task_environment: HashMap<String, String>,
    #[cfg(target_os = "macos")]
    environment_recipe: Option<crate::terminals::NativeTaskEnvironmentRecipe>,
}

impl CargoNativeBuildIntent {
    pub fn program(&self) -> &Path {
        &self.requested_program
    }
    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }
    pub fn working_directory(&self) -> &Path {
        &self.cwd
    }
    pub fn task_environment(&self) -> &HashMap<String, String> {
        &self.task_environment
    }
    pub fn selection(&self) -> &crate::cargo_workspace::CargoExecutableSelection {
        &self.selection
    }
    pub fn output_root(&self) -> &Path {
        &self.output_root
    }
    pub fn invocation_output_root(&self) -> &Path {
        &self.invocation_output_root
    }
    pub fn run_id(&self) -> uuid::Uuid {
        self.run_id
    }
    pub fn task_id(&self) -> &task::TaskId {
        &self.task_id
    }

    #[cfg(target_os = "macos")]
    pub fn expected_invocation(&self) -> Result<util::macos_task_completion::TaskBuildInvocation> {
        util::macos_task_completion::TaskBuildInvocation::new(
            self.requested_program.clone(),
            self.arguments.clone(),
            self.cwd.clone(),
            self.environment_recipe
                .as_ref()
                .context("Native Build complete host environment is unavailable")?
                .environment()
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        )
        .map_err(|_| anyhow!("Native Build invocation exceeds its supported bounds"))
    }

    #[cfg(target_os = "macos")]
    fn revalidate_environment(&self, cx: &mut gpui::AsyncApp) -> Result<Task<Result<()>>> {
        let recipe = self
            .environment_recipe
            .as_ref()
            .context("Native Build complete host environment is unavailable")?;
        let project = self
            .project
            .upgrade()
            .context("Native Build environment Project disappeared")?;
        project.update(cx, |project, cx| {
            project.revalidate_native_task_environment(recipe, cx)
        })
    }

    pub fn verify_eligible_launch(&self, launch: &task::SpawnInTerminal) -> Result<()> {
        ensure!(
            launch.id == self.task_id
                && launch.command.as_deref() == self.requested_program.to_str()
                && launch.args == self.logical_arguments
                && launch.cwd.as_deref() == Some(self.cwd.as_path())
                && launch.env == self.task_environment
                && launch.literal_prefix_args == launch.args.len(),
            "Native Build eligible Task no longer matches its selected logical recipe"
        );
        Ok(())
    }
}

fn native_build_open_buffer_versions(
    project: &crate::Project,
    worktree_id: worktree::WorktreeId,
    cx: &App,
) -> Result<Vec<(gpui::WeakEntity<language::Buffer>, clock::Global)>> {
    let mut versions = Vec::new();
    for buffer in project.buffer_store().read(cx).buffers() {
        let current = buffer.read(cx);
        if !current
            .file()
            .is_some_and(|file| file.worktree_id(cx) == worktree_id)
        {
            continue;
        }
        ensure!(
            versions.len() < 1024,
            "Native Build open-buffer admission exceeds its limit"
        );
        ensure!(
            !current.is_dirty(),
            "Native Build source has unsaved buffer changes"
        );
        versions.push((buffer.downgrade(), current.version()));
    }
    Ok(versions)
}

fn revalidate_native_build_open_buffers(
    project: &crate::Project,
    intent: &CargoNativeBuildIntent,
    cx: &App,
) -> Result<()> {
    revalidate_native_build_buffer_versions(
        project,
        intent.workspace.root.worktree_id,
        &intent.buffer_versions,
        cx,
    )
}

fn revalidate_native_build_buffer_versions(
    project: &crate::Project,
    worktree_id: worktree::WorktreeId,
    versions: &[(gpui::WeakEntity<language::Buffer>, clock::Global)],
    cx: &App,
) -> Result<()> {
    native_build_open_buffer_versions(project, worktree_id, cx)?;
    for (buffer, version) in versions {
        let buffer = buffer
            .upgrade()
            .context("Native Build admitted buffer disappeared")?;
        let current = buffer.read(cx);
        ensure!(
            !current.is_dirty()
                && &current.version() == version
                && current
                    .file()
                    .is_some_and(|file| file.worktree_id(cx) == worktree_id),
            "Native Build admitted source buffer changed"
        );
    }
    Ok(())
}

fn native_build_file_seals_equal(
    before: &NativeProbeFileSeal,
    after: &NativeProbeFileSeal,
) -> bool {
    before.root == after.root
        && before.root_inode == after.root_inode
        && before.relative == after.relative
        && native_probe_metadata_matches(before.metadata, after.metadata)
        && before.fingerprint == after.fingerprint
}

fn validate_native_created_output_entries(
    before: &[worktree::Entry],
    after: &[worktree::Entry],
    output: &Path,
    root: &Path,
) -> Result<()> {
    ensure!(
        before.len() == after.len(),
        "Native clean-first source entry set changed"
    );
    let parent = output
        .parent()
        .context("Native output parent disappeared")?
        .strip_prefix(root)?;
    let parent = RelPath::new(parent, PathStyle::local())?;
    for (before, after) in before.iter().zip(after) {
        let mut expected = before.clone();
        if expected.path.as_ref() == parent.as_ref() && expected.is_dir() && after.is_dir() {
            expected.mtime = after.mtime;
            expected.size = after.size;
        }
        ensure!(
            &expected == after,
            "Native clean-first changed unrelated source admission"
        );
    }
    Ok(())
}

fn validate_native_created_output_seals(
    before_files: &[NativeProbeFileSeal],
    before_directories: &[NativeProbeDirectorySeal],
    after_files: &[NativeProbeFileSeal],
    after_directories: &[NativeProbeDirectorySeal],
    output: &Path,
) -> Result<()> {
    ensure!(
        before_files.len() == after_files.len()
            && before_directories.len() == after_directories.len(),
        "Native clean-first source closure changed"
    );
    for (before, after) in before_files.iter().zip(after_files) {
        ensure!(
            native_build_file_seals_equal(before, after),
            "Native clean-first source file changed"
        );
    }
    let parent = output
        .parent()
        .context("Native output parent disappeared")?;
    for (before, after) in before_directories.iter().zip(after_directories) {
        ensure!(
            before.path == after.path && before.inode == after.inode,
            "Native clean-first source directory changed"
        );
        let mut expected = before.children.clone();
        if before.path == parent {
            ensure!(
                !expected.iter().any(|path| path == output),
                "Native output appeared before this admission"
            );
            expected.push(output.to_path_buf());
            expected.sort();
        } else {
            ensure!(
                before.mtime == after.mtime,
                "Native clean-first changed unrelated directory metadata"
            );
        }
        ensure!(
            expected == after.children,
            "Native clean-first changed unrelated directory membership"
        );
    }
    Ok(())
}

fn native_build_run_arguments(
    base: &[String],
    output_root: &Path,
    run_id: uuid::Uuid,
) -> Result<(PathBuf, Vec<String>)> {
    ensure!(!run_id.is_nil(), "Native Build run identity is invalid");
    let root = output_root.join(format!("zed-native-profile-build-{}", run_id.simple()));
    crate::cargo_workspace::validate_native_cargo_path(&root)?;
    let mut arguments = base.to_vec();
    ensure!(
        arguments.get(6).is_some_and(|flag| flag == "--target-dir")
            && arguments
                .get(7)
                .is_some_and(|path| Some(path.as_str()) == output_root.to_str()),
        "Native Build output argument contract changed"
    );
    let argument = arguments
        .get_mut(7)
        .context("Native Build output argument disappeared")?;
    *argument = root
        .to_str()
        .context("Native Build run output is not UTF-8")?
        .to_owned();
    Ok((root, arguments))
}

#[cfg(any(target_os = "macos", test))]
fn validate_native_build_run_executable(path: &Path, root: &Path) -> Result<()> {
    crate::cargo_workspace::validate_native_cargo_path(path)?;
    ensure!(
        path.starts_with(root) && path != root,
        "Native Build executable belongs to another output run"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
struct NativeCargoRuntimeAdmission {
    compiler: NativeCargoRuntimeCompilerReceipt,
    loader_scope: NativeCargoRuntimeLoaderScope,
    helper: Arc<crate::profile_results_store::LocalNativeTaskHelperImage>,
}

#[cfg(target_os = "macos")]
enum NativeRuntimeLoaderStage {
    OuterCargo,
    ObservedRuntime,
}

#[cfg(target_os = "macos")]
struct NativeCargoRuntimeLoaderScope {
    run_id: uuid::Uuid,
    compiler_directory: PathBuf,
    compiler_inode: u64,
    compiler_sysroot_directory: PathBuf,
    compiler_sysroot_inode: u64,
    invocation_output_root: PathBuf,
}

#[cfg(target_os = "macos")]
impl NativeCargoRuntimeLoaderScope {
    fn from_compiler(
        intent: &CargoNativeBuildIntent,
        facts: &NativeCargoRuntimeCompilerFacts,
    ) -> Result<Self> {
        let root = intent.output_root.join(format!(
            "zed-native-profile-build-{}",
            intent.run_id.simple()
        ));
        ensure!(
            !intent.run_id.is_nil() && root == intent.invocation_output_root,
            "Native runtime loader scope has no matching producing invocation"
        );
        ensure!(
            facts.target_library_directory.starts_with(&facts.sysroot)
                && facts.target_library_directory != facts.sysroot,
            "Native runtime loader scope has no observed compiler root"
        );
        ensure!(
            facts.sysroot_library_directory.path == facts.sysroot.join("lib")
                && facts.sysroot_library_directory.path != facts.target_library_directory,
            "Native runtime sysroot libraries have no matching observed compiler root"
        );
        Ok(Self {
            run_id: intent.run_id,
            compiler_directory: facts.target_library_directory.clone(),
            compiler_inode: facts.target_library_directory_inode,
            compiler_sysroot_directory: facts.sysroot_library_directory.path.clone(),
            compiler_sysroot_inode: facts.sysroot_library_directory.inode,
            invocation_output_root: root,
        })
    }

    fn matches_compiler(&self, receipt: &NativeCargoRuntimeCompilerReceipt) -> bool {
        self.run_id == receipt.intent.run_id
            && self.invocation_output_root == receipt.intent.invocation_output_root
            && self.compiler_directory == receipt.facts.target_library_directory
            && self.compiler_inode == receipt.facts.target_library_directory_inode
            && self.compiler_sysroot_directory == receipt.facts.sysroot_library_directory.path
            && self.compiler_sysroot_inode == receipt.facts.sysroot_library_directory.inode
            && self.compiler_sysroot_directory == receipt.facts.sysroot.join("lib")
    }

    fn pin_outer_fallback(&self, environment: &mut HashMap<String, String>) -> Result<()> {
        let observed = environment
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        self.paths(&observed, NativeRuntimeLoaderStage::OuterCargo)?;
        if environment
            .get("DYLD_FALLBACK_LIBRARY_PATH")
            .is_none_or(String::is_empty)
        {
            crate::cargo_workspace::validate_native_cargo_path(&self.compiler_directory)?;
            // Cargo appends ambient home/system defaults when this value is empty.
            environment.insert(
                "DYLD_FALLBACK_LIBRARY_PATH".into(),
                self.compiler_directory
                    .to_str()
                    .context("Native runtime sealed compiler library directory is not UTF-8")?
                    .into(),
            );
        }
        Ok(())
    }

    fn paths(
        &self,
        environment: &std::collections::BTreeMap<String, String>,
        stage: NativeRuntimeLoaderStage,
    ) -> Result<std::collections::BTreeSet<PathBuf>> {
        validate_native_runtime_sole_fallback(environment)?;
        let paths = match stage {
            NativeRuntimeLoaderStage::OuterCargo => native_runtime_library_paths(
                environment,
                &self.invocation_output_root,
                &self.compiler_directory,
            )?,
            NativeRuntimeLoaderStage::ObservedRuntime => native_runtime_library_paths_with_sysroot(
                environment,
                &self.invocation_output_root,
                &self.compiler_directory,
                Some(&self.compiler_sysroot_directory),
            )?,
        };
        ensure!(
            matches!(stage, NativeRuntimeLoaderStage::ObservedRuntime)
                || paths.iter().all(|path| path == &self.compiler_directory),
            "Native runtime outer loader environment has no producing child authority"
        );
        ensure!(
            !matches!(stage, NativeRuntimeLoaderStage::ObservedRuntime)
                || paths.contains(&self.compiler_directory),
            "Native runtime observation lost its sealed compiler fallback"
        );
        Ok(paths)
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct NativeCargoRuntimeAdmissionLease {
    owner: gpui::WeakEntity<CargoWorkspaceStore>,
    project: gpui::WeakEntity<crate::Project>,
    run_id: uuid::Uuid,
    helper: Arc<crate::profile_results_store::LocalNativeTaskHelperImage>,
    admission: Arc<parking_lot::Mutex<Option<NativeCargoRuntimeAdmission>>>,
    expected: util::macos_task_completion::TaskBuildInvocation,
}

#[cfg(target_os = "macos")]
impl NativeCargoRuntimeAdmissionLease {
    pub(crate) fn matches_helper_image(
        &self,
        image: &crate::profile_results_store::LocalNativeTaskHelperImage,
    ) -> bool {
        self.helper.same_issuance(image)
    }
    pub(crate) fn same_issuance(&self, other: &Self) -> bool {
        self.owner.entity_id() == other.owner.entity_id()
            && self.run_id == other.run_id
            && Arc::ptr_eq(&self.admission, &other.admission)
    }
    pub(crate) fn run_id(&self) -> uuid::Uuid {
        self.run_id
    }
    pub(crate) fn expected_invocation(&self) -> &util::macos_task_completion::TaskBuildInvocation {
        &self.expected
    }
    pub(crate) fn validate_outer_loader_environment(&self) -> Result<()> {
        let admission = self.admission.lock();
        let admission = admission
            .as_ref()
            .context("Native runtime loader admission is unavailable")?;
        ensure!(
            admission.loader_scope.matches_compiler(&admission.compiler),
            "Native runtime compiler loader authority changed"
        );
        admission.loader_scope.paths(
            self.expected.environment(),
            NativeRuntimeLoaderStage::OuterCargo,
        )?;
        Ok(())
    }
    pub(crate) fn project(&self) -> Result<Entity<crate::Project>> {
        self.project
            .upgrade()
            .context("Native runtime admission Project disappeared")
    }
    pub(crate) fn renew(&self, cx: &mut gpui::AsyncApp) -> Result<Task<Result<()>>> {
        self.owner.update(cx, |store, cx| {
            store.revalidate_native_cargo_runtime_admission(self, cx)
        })?
    }
}

#[cfg(target_os = "macos")]
pub struct CargoNativeRuntimeIntent {
    logical: task::SpawnInTerminal,
    lease: NativeCargoRuntimeAdmissionLease,
    task_environment: HashMap<String, String>,
    workload_arguments: Vec<String>,
}

#[cfg(target_os = "macos")]
impl CargoNativeRuntimeIntent {
    pub fn program(&self) -> &Path {
        self.lease.expected.program()
    }
    pub fn arguments(&self) -> &[String] {
        self.lease.expected.arguments()
    }
    pub fn working_directory(&self) -> &Path {
        self.lease.expected.working_directory()
    }
    pub fn task_environment(&self) -> &HashMap<String, String> {
        &self.task_environment
    }
    pub(crate) fn expected_invocation(&self) -> util::macos_task_completion::TaskBuildInvocation {
        self.lease.expected.clone()
    }
    pub(crate) fn registration_lease(&self) -> NativeCargoRuntimeAdmissionLease {
        self.lease.clone()
    }
    pub fn verify_eligible_launch(&self, launch: &task::SpawnInTerminal) -> Result<()> {
        ensure!(
            launch.id == self.logical.id
                && launch.command == self.logical.command
                && launch.args == self.logical.args
                && launch.cwd == self.logical.cwd
                && launch.env == self.logical.env
                && launch.literal_prefix_args == launch.args.len(),
            "Native runtime eligible Task no longer matches its selected logical recipe"
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn native_runtime_outer_environment(
    complete_terminal_environment: &HashMap<String, String>,
    installed_toolchain: Option<&Path>,
) -> Result<HashMap<String, String>> {
    let mut environment = complete_terminal_environment.clone();
    environment.insert("RUSTUP_AUTO_INSTALL".into(), "0".into());
    environment.insert("CARGO_NET_OFFLINE".into(), "true".into());
    if let Some(toolchain) = installed_toolchain {
        environment.insert(
            "RUSTUP_TOOLCHAIN".into(),
            toolchain
                .to_str()
                .context("Native runtime installed toolchain is not UTF-8")?
                .to_owned(),
        );
    }
    Ok(environment)
}

#[cfg(target_os = "macos")]
fn native_runtime_checked_arguments(
    compiler: &NativeCargoRuntimeCompilerReceipt,
    helper: &crate::profile_results_store::LocalNativeTaskHelperImage,
    workload_arguments: &[String],
) -> Result<Vec<String>> {
    let mut arguments = compiler
        .facts
        .execution
        .cargo_options
        .iter()
        .map(|argument| {
            argument
                .to_str()
                .map(str::to_owned)
                .context("Native runtime compiler policy is not UTF-8")
        })
        .collect::<Result<Vec<_>>>()?;
    let runner = toml::Value::Array(vec![
        toml::Value::String(
            helper
                .helper_path()
                .to_str()
                .context("Native runtime helper path is not UTF-8")?
                .into(),
        ),
        toml::Value::String("--internal-cargo-runtime-observer".into()),
        toml::Value::String("--".into()),
    ]);
    arguments.extend([
        "--config".into(),
        format!(
            "target.{}.runner={runner}",
            toml::Value::String(compiler.host_target().into())
        ),
    ]);
    arguments.extend(
        crate::cargo_workspace::native_cargo_runtime_logical_arguments(
            &compiler.intent.arguments,
            workload_arguments,
        )?,
    );
    Ok(arguments)
}

#[cfg(target_os = "macos")]
struct NativeCargoCompilerExecutableSeal {
    requested: PathBuf,
    file: NativeProbeFileSeal,
}

#[cfg(target_os = "macos")]
pub(crate) struct NativeCargoRuntimeCompilerReceipt {
    intent: CargoNativeBuildIntent,
    facts: NativeCargoRuntimeCompilerFacts,
}

#[cfg(target_os = "macos")]
struct NativeCargoRuntimeCompilerFacts {
    configuration: CargoHostExecutionConfiguration,
    configuration_seals: Vec<NativeCargoConfigurationSeal>,
    execution: PassiveRustExecution,
    executables: Vec<NativeCargoCompilerExecutableSeal>,
    observations: [Vec<u8>; 3],
    host_target: String,
    sysroot: PathBuf,
    sysroot_inode: u64,
    target_library_directory: PathBuf,
    target_library_directory_inode: u64,
    library_members: Vec<PathBuf>,
    libraries: Vec<NativeProbeFileSeal>,
    sysroot_library_directory: NativeCargoRuntimeLibraryDirectory,
}

#[cfg(target_os = "macos")]
impl NativeCargoRuntimeCompilerReceipt {
    pub(crate) fn host_target(&self) -> &str {
        &self.facts.host_target
    }
}

#[cfg(target_os = "macos")]
async fn native_compiler_path_output(
    bytes: &[u8],
    filesystem: &dyn fs::Fs,
) -> Result<(PathBuf, u64)> {
    ensure!(
        bytes.len() <= 4097,
        "Native runtime compiler path output exceeds its bound"
    );
    let text = std::str::from_utf8(bytes).context("Native runtime compiler path is not UTF-8")?;
    let text = text.strip_suffix('\n').unwrap_or(text);
    let text = text.strip_suffix('\r').unwrap_or(text);
    let path = PathBuf::from(text);
    crate::cargo_workspace::validate_native_cargo_path(&path)?;
    let canonical = filesystem.canonicalize(&path).await?;
    ensure!(
        canonical == path,
        "Native runtime compiler directory has aliases"
    );
    let metadata = filesystem
        .metadata(&path)
        .await?
        .context("Native runtime compiler directory disappeared")?;
    ensure!(
        metadata.is_dir && !metadata.is_symlink && !metadata.is_fifo,
        "Native runtime compiler directory is not ordinary"
    );
    Ok((path, metadata.inode))
}

#[cfg(target_os = "macos")]
fn validate_native_precompiler_loader_environment(
    environment: &PassiveRustEnvironment,
    configuration: &CargoHostExecutionConfiguration,
) -> Result<()> {
    ensure!(
        !environment.inherit_process,
        "Native runtime compiler requires its complete admitted environment"
    );
    for (name, value) in &environment.overrides {
        let name = name
            .to_str()
            .context("Native runtime environment name is not UTF-8")?;
        ensure!(
            !(name.starts_with("DYLD_") || name.starts_with("LD_")) || value.is_empty(),
            "Native runtime compiler cannot use loader overrides before compiler provenance"
        );
    }
    for (name, value) in &configuration.environment {
        if name.starts_with("DYLD_") || name.starts_with("LD_") {
            ensure!(
                value.resolve()?.0.is_empty(),
                "Native runtime compiler configuration has no reviewed loader provenance"
            );
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn native_runtime_compiler_observations(
    execution: &PassiveRustExecution,
    cwd: &Path,
    executor: &BackgroundExecutor,
) -> Result<[Vec<u8>; 3]> {
    let mut chain = execution.compiler_chain.iter();
    let first = chain
        .next()
        .context("Native runtime configured compiler is unavailable")?;
    let path = execution.child_environment.value("PATH");
    ensure!(
        Path::new(first).is_absolute() || path.is_some(),
        "Native runtime configured compiler has no admitted PATH"
    );
    let program = which::which_in(first, path.as_ref(), cwd)
        .map_err(|_| anyhow!("Native runtime configured compiler is not installed"))?;
    let tail = chain.cloned().collect::<Vec<_>>();
    let mut observations = Vec::new();
    for arguments in [
        &["-vV"][..],
        &["--print", "sysroot"][..],
        &["--print", "target-libdir"][..],
    ] {
        let mut command = util::command::new_command(&program);
        command
            .args(&tail)
            .args(arguments)
            .env_clear()
            .envs(&execution.child_environment.overrides)
            .current_dir(cwd)
            .stdin(util::command::Stdio::null())
            .kill_on_drop(true);
        let output = bounded_cargo_command_output(
            command,
            "Native runtime compiler probe",
            MAX_RUSTC_VERBOSE_VERSION_BYTES,
            CARGO_CONFIGURATION_PROBE_TIMEOUT,
            executor,
        )
        .await?;
        ensure!(
            output.status.success(),
            "Native runtime configured compiler probe did not succeed"
        );
        observations.push(output.stdout);
    }
    observations
        .try_into()
        .map_err(|_| anyhow!("Native runtime compiler observation count changed"))
}

#[cfg(target_os = "macos")]
async fn seal_native_compiler_executables(
    filesystem: &dyn fs::Fs,
    execution: &PassiveRustExecution,
    cwd: &Path,
) -> Result<Vec<NativeCargoCompilerExecutableSeal>> {
    ensure!(
        !execution.compiler_chain.is_empty() && execution.compiler_chain.len() <= 3,
        "Native runtime configured compiler chain is unsupported"
    );
    let mut programs = execution.compiler_chain.clone();
    programs.extend(
        execution
            .delegated_executables
            .iter()
            .map(|path| path.as_os_str().to_owned()),
    );
    ensure!(
        programs.len() <= 5,
        "Native runtime compiler image count exceeds its bound"
    );
    let mut seals = Vec::new();
    for program in &programs {
        let path = execution.child_environment.value("PATH");
        ensure!(
            Path::new(program).is_absolute() || path.is_some(),
            "Native runtime configured compiler has no admitted PATH"
        );
        let requested = which::which_in(program, path.as_ref(), cwd)
            .map_err(|_| anyhow!("Native runtime configured compiler is not installed"))?;
        crate::cargo_workspace::validate_native_cargo_path(&requested)?;
        let canonical = filesystem.canonicalize(&requested).await?;
        let root = canonical
            .parent()
            .context("Native runtime compiler has no parent")?
            .to_path_buf();
        let root_metadata = filesystem
            .metadata(&root)
            .await?
            .context("Native runtime compiler root disappeared")?;
        ensure!(
            root_metadata.is_dir
                && !root_metadata.is_symlink
                && !root_metadata.is_fifo
                && filesystem.canonicalize(&root).await? == root,
            "Native runtime compiler root is not ordinary"
        );
        let metadata = filesystem
            .metadata(&canonical)
            .await?
            .context("Native runtime compiler disappeared")?;
        ensure!(
            metadata.is_executable && metadata.len > 0,
            "Native runtime compiler is not executable"
        );
        let relative = canonical.strip_prefix(&root)?.to_path_buf();
        let fingerprint = fingerprint_native_probe_file(
            filesystem,
            &root,
            root_metadata.inode,
            &relative,
            metadata,
            MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES,
        )
        .await?;
        seals.push(NativeCargoCompilerExecutableSeal {
            requested,
            file: NativeProbeFileSeal {
                root,
                root_inode: root_metadata.inode,
                relative,
                metadata,
                fingerprint,
            },
        });
    }
    Ok(seals)
}

#[cfg(target_os = "macos")]
struct NativeCompilerLibraryBudget {
    members: usize,
    libraries: usize,
    bytes: u64,
}

#[cfg(target_os = "macos")]
impl Default for NativeCompilerLibraryBudget {
    fn default() -> Self {
        Self {
            members: 1024,
            libraries: 256,
            bytes: 512 * 1024 * 1024,
        }
    }
}

#[cfg(target_os = "macos")]
async fn seal_native_compiler_sysroot_libraries(
    filesystem: &dyn fs::Fs,
    sysroot: &Path,
    budget: &mut NativeCompilerLibraryBudget,
) -> Result<NativeCargoRuntimeLibraryDirectory> {
    let path = sysroot.join("lib");
    ensure!(
        filesystem.canonicalize(&path).await? == path,
        "Native compiler sysroot library directory has aliases"
    );
    let metadata = filesystem
        .metadata(&path)
        .await?
        .context("Native compiler sysroot library directory is unavailable")?;
    ensure!(
        metadata.is_dir && !metadata.is_symlink && !metadata.is_fifo,
        "Native compiler sysroot library directory is not ordinary"
    );
    let (members, libraries) = seal_native_compiler_runtime_libraries_with_budget(
        filesystem,
        &path,
        metadata.inode,
        budget,
    )
    .await?;
    Ok(NativeCargoRuntimeLibraryDirectory {
        path,
        inode: metadata.inode,
        members,
        libraries,
    })
}

#[cfg(target_os = "macos")]
async fn seal_native_compiler_runtime_libraries(
    filesystem: &dyn fs::Fs,
    directory: &Path,
    inode: u64,
) -> Result<(Vec<PathBuf>, Vec<NativeProbeFileSeal>)> {
    let mut budget = NativeCompilerLibraryBudget::default();
    seal_native_compiler_runtime_libraries_with_budget(filesystem, directory, inode, &mut budget)
        .await
}

#[cfg(target_os = "macos")]
async fn seal_native_compiler_runtime_libraries_with_budget(
    filesystem: &dyn fs::Fs,
    directory: &Path,
    inode: u64,
    budget: &mut NativeCompilerLibraryBudget,
) -> Result<(Vec<PathBuf>, Vec<NativeProbeFileSeal>)> {
    let mut remaining = budget.members;
    let members = native_probe_directory_children(filesystem, directory, &mut remaining).await?;
    budget.members = remaining;
    let mut libraries = Vec::new();
    let mut bytes = 0_u64;
    for path in &members {
        ensure!(
            path.parent() == Some(directory),
            "Native compiler library entry escaped its observed directory"
        );
        let metadata = filesystem
            .metadata(path)
            .await?
            .context("Native compiler library entry disappeared")?;
        ensure!(
            !metadata.is_symlink && !metadata.is_fifo,
            "Native compiler library directory contains unsupported entry routes"
        );
        if metadata.is_dir
            || !matches!(
                path.extension().and_then(OsStr::to_str),
                Some("dylib" | "so")
            )
        {
            continue;
        }
        ensure!(
            libraries.len() < 256,
            "Native compiler runtime library count exceeds its bound"
        );
        bytes = bytes
            .checked_add(metadata.len)
            .context("Native compiler runtime library size overflow")?;
        ensure!(
            bytes <= 512 * 1024 * 1024,
            "Native compiler runtime library bytes exceed their bound"
        );
        ensure!(
            filesystem.canonicalize(path).await? == *path,
            "Native compiler runtime library has aliases"
        );
        budget.libraries = budget
            .libraries
            .checked_sub(1)
            .context("Native compiler runtime library count exceeds its aggregate bound")?;
        budget.bytes = budget
            .bytes
            .checked_sub(metadata.len)
            .context("Native compiler runtime libraries exceed their aggregate byte bound")?;
        let relative = path.strip_prefix(directory)?.to_path_buf();
        let fingerprint = fingerprint_native_probe_file(
            filesystem,
            directory,
            inode,
            &relative,
            metadata,
            256 * 1024 * 1024,
        )
        .await?;
        libraries.push(NativeProbeFileSeal {
            root: directory.to_path_buf(),
            root_inode: inode,
            relative,
            metadata,
            fingerprint,
        });
    }
    let mut remaining = 1024;
    ensure!(
        native_probe_directory_children(filesystem, directory, &mut remaining).await? == members,
        "Native compiler library membership changed while sealing"
    );
    Ok((members, libraries))
}

#[cfg(target_os = "macos")]
async fn revalidate_native_compiler_receipt(
    filesystem: Arc<dyn fs::Fs>,
    receipt: &NativeCargoRuntimeCompilerReceipt,
    executor: &BackgroundExecutor,
) -> Result<()> {
    revalidate_native_compiler_facts(filesystem, &receipt.intent, &receipt.facts, executor).await
}

#[cfg(target_os = "macos")]
async fn revalidate_native_compiler_facts(
    filesystem: Arc<dyn fs::Fs>,
    intent: &CargoNativeBuildIntent,
    facts: &NativeCargoRuntimeCompilerFacts,
    executor: &BackgroundExecutor,
) -> Result<()> {
    revalidate_native_build_filesystem(filesystem.as_ref(), intent).await?;
    revalidate_native_cargo_configuration(filesystem.as_ref(), &facts.configuration_seals).await?;
    let recipe = intent
        .environment_recipe
        .as_ref()
        .context("Native runtime complete environment disappeared")?;
    let environment = PassiveRustEnvironment::from_native_recipe(recipe);
    let (configuration, _) = passive_cargo_configuration_with_native_review(
        &intent.cwd,
        &environment,
        Some(filesystem.clone()),
    )
    .await?;
    ensure!(
        configuration == facts.configuration,
        "Native runtime configured compiler selection changed"
    );
    validate_native_precompiler_loader_environment(&environment, &configuration)?;
    let current_execution = prepare_passive_rust_execution_in_environment(
        Some(intent.requested_program.as_os_str()),
        false,
        &intent.cwd,
        environment,
        executor,
    )
    .await?;
    ensure!(
        current_execution.compiler_chain == facts.execution.compiler_chain
            && current_execution.delegated_executables == facts.execution.delegated_executables
            && current_execution.child_environment.overrides
                == facts.execution.child_environment.overrides
            && current_execution.cargo_parent_environment.overrides
                == facts.execution.cargo_parent_environment.overrides
            && current_execution.cargo_options == facts.execution.cargo_options
            && current_execution.toolchain == facts.execution.toolchain,
        "Native runtime compiler execution environment changed"
    );
    for executable in &facts.executables {
        ensure!(
            filesystem.canonicalize(&executable.requested).await?
                == executable.file.root.join(&executable.file.relative),
            "Native runtime compiler route changed"
        );
        ensure!(
            fingerprint_native_probe_file(
                filesystem.as_ref(),
                &executable.file.root,
                executable.file.root_inode,
                &executable.file.relative,
                executable.file.metadata,
                MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES
            )
            .await?
                == executable.file.fingerprint,
            "Native runtime compiler image changed"
        );
    }
    ensure!(
        native_runtime_compiler_observations(&facts.execution, &intent.cwd, executor).await?
            == facts.observations,
        "Native runtime compiler observations changed"
    );
    ensure!(
        facts.sysroot_library_directory.path == facts.sysroot.join("lib"),
        "Native runtime sysroot library route changed"
    );
    for (path, inode) in [
        (&facts.sysroot, facts.sysroot_inode),
        (
            &facts.sysroot_library_directory.path,
            facts.sysroot_library_directory.inode,
        ),
        (
            &facts.target_library_directory,
            facts.target_library_directory_inode,
        ),
    ] {
        ensure!(
            filesystem.canonicalize(path).await? == *path
                && filesystem
                    .metadata(path)
                    .await?
                    .is_some_and(|metadata| metadata.is_dir
                        && !metadata.is_symlink
                        && !metadata.is_fifo
                        && metadata.inode == inode),
            "Native runtime compiler directory identity changed"
        );
    }
    let mut budget = NativeCompilerLibraryBudget::default();
    let (members, libraries) = seal_native_compiler_runtime_libraries_with_budget(
        filesystem.as_ref(),
        &facts.target_library_directory,
        facts.target_library_directory_inode,
        &mut budget,
    )
    .await?;
    ensure!(
        members == facts.library_members
            && libraries.len() == facts.libraries.len()
            && libraries
                .iter()
                .zip(&facts.libraries)
                .all(|(current, before)| current.relative == before.relative
                    && native_probe_metadata_matches(current.metadata, before.metadata)
                    && current.fingerprint == before.fingerprint),
        "Native compiler runtime libraries changed after admission"
    );
    let sysroot_libraries =
        seal_native_compiler_sysroot_libraries(filesystem.as_ref(), &facts.sysroot, &mut budget)
            .await?;
    ensure!(
        sysroot_libraries.path == facts.sysroot_library_directory.path
            && sysroot_libraries.inode == facts.sysroot_library_directory.inode
            && sysroot_libraries.members == facts.sysroot_library_directory.members
            && sysroot_libraries.libraries.len() == facts.sysroot_library_directory.libraries.len()
            && sysroot_libraries
                .libraries
                .iter()
                .zip(&facts.sysroot_library_directory.libraries)
                .all(|(current, before)| current.relative == before.relative
                    && native_probe_metadata_matches(current.metadata, before.metadata)
                    && current.fingerprint == before.fingerprint),
        "Native compiler sysroot runtime libraries changed after admission"
    );
    revalidate_native_cargo_configuration(filesystem.as_ref(), &facts.configuration_seals).await?;
    Ok(())
}

#[cfg(target_os = "macos")]
struct NativeCargoRuntimeLibraryDirectory {
    path: PathBuf,
    inode: u64,
    members: Vec<PathBuf>,
    libraries: Vec<NativeProbeFileSeal>,
}

#[cfg(target_os = "macos")]
fn native_runtime_library_paths(
    environment: &std::collections::BTreeMap<String, String>,
    invocation_output_root: &Path,
    compiler_directory: &Path,
) -> Result<std::collections::BTreeSet<PathBuf>> {
    native_runtime_library_paths_with_sysroot(
        environment,
        invocation_output_root,
        compiler_directory,
        None,
    )
}

#[cfg(target_os = "macos")]
fn native_runtime_library_paths_with_sysroot(
    environment: &std::collections::BTreeMap<String, String>,
    invocation_output_root: &Path,
    compiler_directory: &Path,
    compiler_sysroot_directory: Option<&Path>,
) -> Result<std::collections::BTreeSet<PathBuf>> {
    for (name, value) in environment {
        let descriptor = matches!(
            name.as_str(),
            "CARGO_MAKEFLAGS" | "MAKEFLAGS" | "MFLAGS" | "LISTEN_FDS" | "LISTEN_PID"
        );
        ensure!(
            !descriptor || value.is_empty(),
            "Native runtime environment contains unsupported resource-bearing state"
        );
        let loading = name.starts_with("DYLD_") || name.starts_with("LD_");
        ensure!(
            !loading
                || value.is_empty()
                || matches!(
                    name.as_str(),
                    "DYLD_FALLBACK_LIBRARY_PATH" | "DYLD_LIBRARY_PATH" | "LD_LIBRARY_PATH"
                ),
            "Native runtime contains an unsupported dynamic-loader configuration"
        );
    }
    let mut paths = std::collections::BTreeSet::new();
    for name in [
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ] {
        let Some(value) = environment.get(name).filter(|value| !value.is_empty()) else {
            continue;
        };
        for path in std::env::split_paths(value) {
            ensure!(
                paths.contains(&path) || paths.len() < 128,
                "Native runtime library directory count exceeds its bound"
            );
            crate::cargo_workspace::validate_native_cargo_path(&path)?;
            ensure!(
                path == compiler_directory
                    || compiler_sysroot_directory.is_some_and(|directory| path == directory)
                    || (path.starts_with(&invocation_output_root)
                        && path != invocation_output_root),
                "Native runtime library directory has no producing Task or compiler authority"
            );
            paths.insert(path);
        }
    }
    Ok(paths)
}

#[cfg(target_os = "macos")]
fn validate_native_runtime_sole_fallback(
    environment: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    ensure!(
        environment.iter().all(|(name, value)| value.is_empty()
            || (!name.starts_with("DYLD_") && !name.starts_with("LD_"))
            || name == "DYLD_FALLBACK_LIBRARY_PATH"),
        "Native runtime environment requires unsupported loader configuration"
    );
    if let Some(value) = environment
        .get("DYLD_FALLBACK_LIBRARY_PATH")
        .filter(|value| !value.is_empty())
    {
        ensure!(
            value.len() <= 64 * 1024 && value.split(':').count() <= 128,
            "Native runtime fallback environment exceeds its bound"
        );
        ensure!(
            value
                .split(':')
                .all(|part| !part.is_empty() && !part.chars().any(char::is_control)),
            "Native runtime fallback environment contains an implicit or invalid directory"
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn seal_native_cargo_runtime_library_directories(
    filesystem: &dyn fs::Fs,
    intent: &CargoNativeBuildIntent,
    compiler: &NativeCargoRuntimeCompilerFacts,
    runtime: &util::macos_task_completion::ObservedCargoRuntime,
) -> Result<Vec<NativeCargoRuntimeLibraryDirectory>> {
    let scope = NativeCargoRuntimeLoaderScope::from_compiler(intent, compiler)?;
    let paths = scope.paths(
        runtime.environment(),
        NativeRuntimeLoaderStage::ObservedRuntime,
    )?;
    seal_native_runtime_library_paths(filesystem, paths).await
}

#[cfg(target_os = "macos")]
async fn seal_native_runtime_library_paths(
    filesystem: &dyn fs::Fs,
    paths: std::collections::BTreeSet<PathBuf>,
) -> Result<Vec<NativeCargoRuntimeLibraryDirectory>> {
    let mut directories = Vec::new();
    let mut library_bytes = 0_u64;
    let mut library_count = 0_usize;
    let mut member_count = 0_usize;
    for path in paths {
        ensure!(
            filesystem.canonicalize(&path).await? == path,
            "Native runtime library directory has aliases"
        );
        let metadata = filesystem
            .metadata(&path)
            .await?
            .context("Native runtime library directory disappeared")?;
        ensure!(
            metadata.is_dir && !metadata.is_symlink && !metadata.is_fifo,
            "Native runtime library directory is not ordinary"
        );
        let (members, libraries) =
            seal_native_compiler_runtime_libraries(filesystem, &path, metadata.inode).await?;
        member_count = member_count
            .checked_add(members.len())
            .context("Native runtime library membership count overflow")?;
        ensure!(
            member_count <= MAX_NATIVE_PROBE_ENTRIES,
            "Native runtime library membership exceeds its aggregate bound"
        );
        for library in &libraries {
            library_count = library_count
                .checked_add(1)
                .context("Native runtime library count overflow")?;
            library_bytes = library_bytes
                .checked_add(library.metadata.len)
                .context("Native runtime library byte count overflow")?;
            ensure!(
                library_count <= 1024 && library_bytes <= 512 * 1024 * 1024,
                "Native runtime libraries exceed their aggregate bound"
            );
        }
        directories.push(NativeCargoRuntimeLibraryDirectory {
            path,
            inode: metadata.inode,
            members,
            libraries,
        });
    }
    Ok(directories)
}

#[cfg(target_os = "macos")]
async fn revalidate_native_cargo_runtime_library_directories(
    filesystem: &dyn fs::Fs,
    directories: &[NativeCargoRuntimeLibraryDirectory],
) -> Result<()> {
    for directory in directories {
        let metadata = filesystem
            .metadata(&directory.path)
            .await?
            .context("Native runtime library directory disappeared")?;
        ensure!(
            metadata.is_dir
                && !metadata.is_symlink
                && !metadata.is_fifo
                && metadata.inode == directory.inode
                && filesystem.canonicalize(&directory.path).await? == directory.path,
            "Native runtime library directory identity changed"
        );
        let (members, libraries) =
            seal_native_compiler_runtime_libraries(filesystem, &directory.path, directory.inode)
                .await?;
        ensure!(
            members == directory.members
                && libraries.len() == directory.libraries.len()
                && libraries
                    .iter()
                    .zip(&directory.libraries)
                    .all(|(current, before)| current.relative == before.relative
                        && native_probe_metadata_matches(current.metadata, before.metadata)
                        && current.fingerprint == before.fingerprint),
            "Native runtime library inputs changed after the producing Task"
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
struct NativeCargoRuntimeWorkload {
    compiler: NativeCargoRuntimeCompilerFacts,
    lease: NativeCargoRuntimeAdmissionLease,
    environment: HashMap<String, String>,
    workload_arguments: Vec<String>,
    libraries: Vec<NativeCargoRuntimeLibraryDirectory>,
}

#[cfg(target_os = "macos")]
pub struct NativeCargoWorkload {
    intent: CargoNativeBuildIntent,
    report: crate::profile_results_store::ObservedBuildTaskReport,
    executable: NativeProbeFileSeal,
    runtime_libraries: Option<Vec<crate::cargo_workspace::CargoNativeRuntimeLibraryRecord>>,
    runtime: Option<NativeCargoRuntimeWorkload>,
}

#[cfg(target_os = "macos")]
impl NativeCargoWorkload {
    pub fn runtime_library_records(
        &self,
    ) -> Option<&[crate::cargo_workspace::CargoNativeRuntimeLibraryRecord]> {
        self.runtime_libraries.as_deref()
    }
    pub fn executable_path(&self) -> PathBuf {
        self.executable.root.join(&self.executable.relative)
    }
    pub fn run_id(&self) -> uuid::Uuid {
        self.intent.run_id
    }
    pub fn task_id(&self) -> &task::TaskId {
        &self.intent.task_id
    }
    pub fn selection(&self) -> &crate::cargo_workspace::CargoExecutableSelection {
        &self.intent.selection
    }
    pub fn working_directory(&self) -> &Path {
        &self.intent.cwd
    }

    fn execution_directory(&self) -> Result<&Path> {
        ensure!(
            matches!(
                self.intent.selection.target_kind,
                crate::cargo_workspace::CargoTargetKind::Binary
                    | crate::cargo_workspace::CargoTargetKind::Example
            ) && self.intent.selection.execution_policy
                == crate::cargo_workspace::CargoNativeExecutionPolicy::Program,
            "Native Capture execution-directory policy is unavailable for this Cargo target"
        );
        Ok(&self.intent.cwd)
    }
}

#[cfg(test)]
fn native_build_selection(
    facts: &CargoWorkspaceMetadataFacts,
    workspace: &CargoWorkspaceModel,
    member_id: &str,
    target: &crate::cargo_workspace::CargoTargetModel,
) -> Result<(
    crate::cargo_workspace::CargoExecutableSelection,
    PathBuf,
    Vec<String>,
)> {
    native_build_selection_with_profile(facts, workspace, member_id, target, None)
}

fn native_build_selection_with_profile(
    facts: &CargoWorkspaceMetadataFacts,
    workspace: &CargoWorkspaceModel,
    member_id: &str,
    target: &crate::cargo_workspace::CargoTargetModel,
    profile: Option<&str>,
) -> Result<(
    crate::cargo_workspace::CargoExecutableSelection,
    PathBuf,
    Vec<String>,
)> {
    use crate::cargo_workspace::{
        CargoExecutableSelection, CargoTargetKind, validate_native_cargo_path,
    };
    ensure!(
        facts.matches_workspace(workspace),
        "Native Build metadata no longer matches its visible selection"
    );
    let mut members = workspace
        .members
        .iter()
        .filter(|member| member.id == member_id);
    let member = members
        .next()
        .context("Native Build selected member is unavailable")?;
    ensure!(
        members.next().is_none(),
        "Native Build selected member is ambiguous"
    );
    ensure!(
        workspace
            .members
            .iter()
            .filter(|candidate| candidate.name == member.name)
            .count()
            == 1,
        "Native Build package-name selection is ambiguous"
    );
    ensure!(
        member.targets.contains(target),
        "Native Build target no longer belongs to its selected member"
    );
    let mut packages = facts
        .packages
        .iter()
        .filter(|package| package.model_id == member_id);
    let package = packages
        .next()
        .context("Native Build exact PackageId is unavailable")?;
    ensure!(
        packages.next().is_none(),
        "Native Build exact PackageId is ambiguous"
    );
    let mut targets = package
        .targets
        .iter()
        .filter(|candidate| candidate.model == *target);
    let selected = targets
        .next()
        .context("Native Build exact target is unavailable")?;
    ensure!(
        targets.next().is_none(),
        "Native Build exact target is ambiguous"
    );
    let (flag, kind) = match target.kind {
        CargoTargetKind::Binary => ("--bin", "bin"),
        CargoTargetKind::Example => ("--example", "example"),
        _ => anyhow::bail!("Native Build currently requires a selected binary or example"),
    };
    ensure!(
        selected.kinds.as_slice() == [kind],
        "Native Build target kind is ambiguous"
    );
    ensure!(
        [&package.package_id, &member.name, &target.name]
            .into_iter()
            .all(|value| !value.is_empty()
                && value.len() <= crate::cargo_workspace::MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES
                && !value.chars().any(char::is_control)),
        "Native Build selection fields exceed their bounds"
    );
    ensure!(
        matches!(
            &selected.execution_policy,
            CargoNativeTargetPolicyFacts::Reviewed(
                crate::cargo_workspace::CargoNativeExecutionPolicy::Program
            )
        ),
        "Native Build target execution policy is unavailable or unsupported"
    );
    validate_native_cargo_path(&selected.source_path)?;
    validate_native_cargo_path(&facts.target_directory)?;
    ensure!(
        target.required_features.len() <= 256
            && target
                .required_features
                .iter()
                .all(|value| !value.is_empty()
                    && value.len() <= 1024
                    && !value.contains(',')
                    && !value.chars().any(char::is_control)),
        "Native Build required features are invalid"
    );
    ensure!(
        target
            .required_features
            .iter()
            .try_fold(0_usize, |bytes, value| bytes.checked_add(value.len() + 1))
            .is_some_and(|bytes| bytes <= 4096),
        "Native Build required-features argument exceeds the helper bound"
    );
    let mut arguments = vec![
        "build".into(),
        "--message-format=json".into(),
        "--package".into(),
        member.name.clone(),
        flag.into(),
        target.name.clone(),
        "--target-dir".into(),
        facts
            .target_directory
            .to_str()
            .context("Native Build output root is not UTF-8")?
            .into(),
    ];
    if !target.required_features.is_empty() {
        arguments.extend(["--features".into(), target.required_features.join(",")]);
    }
    if let Some(profile) = profile {
        ensure!(
            workspace
                .configuration
                .profiles
                .iter()
                .filter(|candidate| candidate.name == profile)
                .count()
                == 1,
            "Native Build selected profile is missing or ambiguous"
        );
        arguments.extend(["--profile".into(), profile.into()]);
    }
    crate::cargo_workspace::native_cargo_build_profile(&arguments)?;
    Ok((
        CargoExecutableSelection {
            package_id: package.package_id.clone(),
            target_kind: target.kind.clone(),
            target_name: target.name.clone(),
            source_path: selected.source_path.clone(),
            execution_policy: crate::cargo_workspace::CargoNativeExecutionPolicy::Program,
        },
        facts.target_directory.clone(),
        arguments,
    ))
}

fn native_selected_package_directory(
    facts: &CargoWorkspaceMetadataFacts,
    workspace: &CargoWorkspaceModel,
    selection: &crate::cargo_workspace::CargoExecutableSelection,
    root: &Path,
    entries: &[worktree::Entry],
) -> Result<(PathBuf, u64)> {
    ensure!(
        facts.matches_workspace(workspace),
        "Native workload package no longer matches its current metadata"
    );
    let mut packages = facts
        .packages
        .iter()
        .filter(|package| package.package_id == selection.package_id);
    let package = packages
        .next()
        .context("Native workload exact package is unavailable")?;
    ensure!(
        packages.next().is_none()
            && package.manifest_path.worktree_id == workspace.key.root.worktree_id
            && package.targets.iter().any(|target| {
                target.source_path == selection.source_path
                    && target.model.name == selection.target_name
                    && target.model.kind == selection.target_kind
            }),
        "Native workload package or selected target belongs to another host selection"
    );
    let manifest = entries
        .iter()
        .find(|entry| entry.path == package.manifest_path.path)
        .context("Native workload package manifest is not admitted source")?;
    ensure!(
        manifest.is_file() && !manifest.is_ignored && !manifest.is_private && !manifest.is_external,
        "Native workload package manifest is not ordinary public source"
    );
    let relative = package
        .manifest_path
        .path
        .parent()
        .unwrap_or(RelPath::empty());
    let directory = entries
        .iter()
        .find(|entry| entry.path.as_ref() == relative)
        .context("Native workload package directory is not admitted source")?;
    ensure!(
        directory.is_dir()
            && !directory.is_ignored
            && !directory.is_private
            && !directory.is_external
            && !directory.is_fifo,
        "Native workload package directory is not ordinary public source"
    );
    let path = root.join(relative.as_std_path());
    crate::cargo_workspace::validate_native_cargo_path(&path)?;
    ensure!(
        path.starts_with(root),
        "Native workload package directory is outside its admitted source root"
    );
    Ok((path, directory.inode))
}

const MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_NATIVE_PROBE_SOURCE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_NATIVE_PROBE_ENTRIES: usize = 4096;
const MAX_NATIVE_COLLECTOR_VERSION_BYTES: usize = 4096;

struct NativeProbeFileSeal {
    root: PathBuf,
    root_inode: u64,
    relative: PathBuf,
    metadata: fs::Metadata,
    fingerprint: [u8; 32],
}

struct NativeProbeDirectorySeal {
    path: PathBuf,
    inode: u64,
    mtime: fs::MTime,
    children: Vec<PathBuf>,
}

#[cfg(target_os = "macos")]
pub(crate) struct NativeCargoCapturePlan {
    pub(crate) resolved: task::ResolvedTask,
    workload: NativeCargoWorkload,
    probe: CargoNativeCollectorProbe,
}

#[cfg(target_os = "macos")]
impl NativeCargoCapturePlan {
    pub(crate) fn expected_capture_invocation(
        &self,
        launch: &task::SpawnInTerminal,
    ) -> Result<util::macos_task_completion::TaskBuildInvocation> {
        let expected = &self.resolved.resolved;
        let collector_path = self.probe.collector_path();
        ensure!(
            launch.id == expected.id
                && launch.command == expected.command
                && launch.args == expected.args
                && launch.cwd == expected.cwd
                && launch.env == expected.env
                && launch.literal_command == expected.literal_command
                && launch.literal_prefix_args == expected.literal_prefix_args
                && launch.literal_command
                && launch.literal_prefix_args == launch.args.len()
                && self.resolved.stdout_report().is_none()
                && self.resolved.resolved_artifact().is_none()
                && self.resolved.artifact_directory() == Some(self.workload.intent.cwd.as_path())
                && launch.cwd.as_deref() == Some(self.workload.execution_directory()?),
            "Native capture invocation differs from its bound logical Task"
        );
        let (_, recipe) = self
            .probe
            .environment_source
            .terminal_recipe()
            .context("Native capture requires a complete host environment receipt")?;
        let complete_environment = self
            .probe
            .environment_source
            .complete_environment()
            .context("Native capture complete environment disappeared")?;
        if let Some(proof) = &self.workload.runtime {
            ensure!(
                complete_environment == &proof.environment,
                "Native capture environment belongs to another observed Cargo workload"
            );
        }
        ensure!(
            recipe.working_directory() == self.probe.cwd
                && recipe.logical_environment() == &launch.env
                && launch.cwd.as_deref() == Some(recipe.working_directory())
                && native_capture_environment(complete_environment)? == self.probe.environment
                && launch.command.as_deref().map(Path::new) == Some(collector_path.as_path()),
            "Native capture complete environment or collector differs from its logical Task"
        );
        util::macos_task_completion::TaskBuildInvocation::new(
            collector_path,
            launch.args.clone(),
            self.probe.cwd.clone(),
            self.probe
                .environment
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        )
        .map_err(|_| anyhow!("Native capture invocation exceeds its supported bounds"))
    }

    pub(crate) fn has_observed_runtime(&self) -> bool {
        self.workload.runtime.is_some()
    }

    pub(crate) fn validate_terminal_loader_environment(
        &self,
        expected: &util::macos_task_completion::TaskBuildInvocation,
    ) -> Result<()> {
        ensure!(
            self.expected_capture_invocation(&self.resolved.resolved)? == *expected,
            "Native terminal loader policy belongs to a different capture invocation"
        );
        let Some(proof) = &self.workload.runtime else {
            return crate::profile_results_store::validate_native_runtime_loader_environment(
                expected.environment(),
            );
        };
        let scope =
            NativeCargoRuntimeLoaderScope::from_compiler(&self.workload.intent, &proof.compiler)?;
        let paths = scope.paths(
            expected.environment(),
            NativeRuntimeLoaderStage::ObservedRuntime,
        )?;
        ensure!(
            paths.len() == proof.libraries.len()
                && proof
                    .libraries
                    .iter()
                    .all(|directory| paths.contains(&directory.path)),
            "Native terminal loader directories differ from the sealed producing workload"
        );
        Ok(())
    }

    pub(crate) fn project(&self) -> Result<Entity<crate::Project>> {
        self.workload
            .intent
            .project
            .upgrade()
            .context("Native capture Project disappeared")
    }
}

enum NativeCollectorEnvironment {
    Legacy,
    #[cfg(target_os = "macos")]
    TerminalRecipe {
        project: gpui::WeakEntity<crate::Project>,
        recipe: crate::terminals::NativeTaskEnvironmentRecipe,
    },
    #[cfg(target_os = "macos")]
    CargoRuntime {
        project: gpui::WeakEntity<crate::Project>,
        recipe: crate::terminals::NativeTaskEnvironmentRecipe,
        environment: HashMap<String, String>,
    },
}

impl NativeCollectorEnvironment {
    #[cfg(target_os = "macos")]
    fn terminal_recipe(
        &self,
    ) -> Option<(
        &gpui::WeakEntity<crate::Project>,
        &crate::terminals::NativeTaskEnvironmentRecipe,
    )> {
        match self {
            Self::Legacy => None,
            Self::TerminalRecipe { project, recipe }
            | Self::CargoRuntime {
                project, recipe, ..
            } => Some((project, recipe)),
        }
    }
    #[cfg(target_os = "macos")]
    fn complete_environment(&self) -> Option<&HashMap<String, String>> {
        match self {
            Self::Legacy => None,
            Self::TerminalRecipe { recipe, .. } => Some(recipe.environment()),
            Self::CargoRuntime { environment, .. } => Some(environment),
        }
    }
    fn revalidate(&self, _cx: &mut gpui::AsyncApp) -> Result<Task<Result<()>>> {
        match self {
            Self::Legacy => Ok(Task::ready(Ok(()))),
            #[cfg(target_os = "macos")]
            Self::TerminalRecipe { project, recipe }
            | Self::CargoRuntime {
                project, recipe, ..
            } => {
                let project = project
                    .upgrade()
                    .context("Native collector environment Project disappeared")?;
                project.update(_cx, |project, cx| {
                    project.revalidate_native_task_environment(recipe, cx)
                })
            }
        }
    }
}

pub struct CargoNativeCollectorProbe {
    environment_source: NativeCollectorEnvironment,
    owner: gpui::EntityId,
    reviewed: CargoReviewedMetadataImport,
    source_entries: Vec<worktree::Entry>,
    sources: Vec<NativeProbeFileSeal>,
    directories: Vec<NativeProbeDirectorySeal>,
    requested_collector: PathBuf,
    collector: NativeProbeFileSeal,
    cwd: PathBuf,
    cwd_inode: u64,
    environment: HashMap<String, String>,
    probe_environment_fingerprint: [u8; 32],
    task_environment: HashMap<String, String>,
}

impl CargoNativeCollectorProbe {
    pub fn collector_path(&self) -> PathBuf {
        self.collector.root.join(&self.collector.relative)
    }

    pub fn version(&self) -> &'static str {
        "0.13.1"
    }

    pub fn probe_environment_fingerprint(&self) -> [u8; 32] {
        self.probe_environment_fingerprint
    }
}

fn native_probe_metadata_matches(left: fs::Metadata, right: fs::Metadata) -> bool {
    left.inode == right.inode
        && left.mtime == right.mtime
        && left.len == right.len
        && left.is_dir == right.is_dir
        && left.is_symlink == right.is_symlink
        && left.is_fifo == right.is_fifo
        && left.is_executable == right.is_executable
}

async fn fingerprint_native_probe_file(
    filesystem: &dyn fs::Fs,
    root: &Path,
    root_inode: u64,
    relative: &Path,
    expected: fs::Metadata,
    limit: u64,
) -> Result<[u8; 32]> {
    ensure!(
        !expected.is_dir && !expected.is_symlink && !expected.is_fifo && expected.len <= limit,
        "Native probe input is not a bounded regular file"
    );
    let path = root.join(relative);
    let before = filesystem
        .metadata(&path)
        .await?
        .context("Native probe input disappeared")?;
    ensure!(
        native_probe_metadata_matches(before, expected),
        "Native probe input identity changed"
    );
    let mut reader = filesystem
        .open_sync_confined_file(root, relative, root_inode, expected.inode)
        .await?;
    #[cfg(target_os = "macos")]
    use sha2_helper::Digest as _;
    #[cfg(target_os = "macos")]
    let mut fingerprint = sha2_helper::Sha256::new();
    #[cfg(not(target_os = "macos"))]
    let mut fingerprint = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let remaining = limit
            .saturating_sub(bytes)
            .saturating_add(1)
            .min(buffer.len() as u64) as usize;
        let read = std::io::Read::read(&mut reader, &mut buffer[..remaining])?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .context("Native probe input size overflow")?;
        ensure!(
            bytes <= limit,
            "Native probe input grew beyond its byte limit"
        );
        fingerprint.update(&buffer[..read]);
    }
    ensure!(bytes == expected.len, "Native probe input length changed");
    let after = filesystem
        .metadata(&path)
        .await?
        .context("Native probe input disappeared")?;
    ensure!(
        native_probe_metadata_matches(after, expected),
        "Native probe input changed while reading"
    );
    Ok(fingerprint.finalize().into())
}

async fn revalidate_native_build_filesystem(
    filesystem: &dyn fs::Fs,
    intent: &CargoNativeBuildIntent,
) -> Result<()> {
    intent.reviewed.preflight_filesystem(filesystem).await?;
    revalidate_native_probe_filesystem(filesystem, &intent.sources, &intent.directories).await?;
    for (path, inode) in [
        (&intent.cwd, intent.cwd_inode),
        (
            &intent.package_source_directory,
            intent.package_source_directory_inode,
        ),
        (&intent.output_root, intent.output_root_inode),
    ] {
        ensure!(
            filesystem.canonicalize(path).await? == *path,
            "Native Build directory has aliases"
        );
        let metadata = filesystem
            .metadata(path)
            .await?
            .context("Native Build directory disappeared")?;
        ensure!(
            metadata.is_dir && !metadata.is_symlink && !metadata.is_fifo && metadata.inode == inode,
            "Native Build directory identity changed"
        );
    }
    ensure!(
        filesystem.canonicalize(&intent.requested_program).await?
            == intent.program.root.join(&intent.program.relative),
        "Native Build selected program changed"
    );
    ensure!(
        fingerprint_native_probe_file(
            filesystem,
            &intent.program.root,
            intent.program.root_inode,
            &intent.program.relative,
            intent.program.metadata,
            MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES
        )
        .await?
            == intent.program.fingerprint,
        "Native Build program bytes changed"
    );
    Ok(())
}

async fn native_probe_directory_children(
    filesystem: &dyn fs::Fs,
    path: &Path,
    remaining: &mut usize,
) -> Result<Vec<PathBuf>> {
    use futures::StreamExt as _;
    let mut stream = filesystem.read_dir(path).await?;
    let mut children = Vec::new();
    while let Some(child) = stream.next().await {
        *remaining = remaining
            .checked_sub(1)
            .context("Native source directory-entry budget exceeded")?;
        let child = child?;
        ensure!(
            child.to_str().is_some_and(|path| path.len() <= 4096),
            "Native source entry path is not bounded UTF-8"
        );
        children.push(child);
    }
    children.sort();
    Ok(children)
}

async fn seal_native_probe_sources(
    filesystem: &dyn fs::Fs,
    reviewed: &CargoReviewedMetadataImport,
    entries: &[worktree::Entry],
) -> Result<(Vec<NativeProbeFileSeal>, Vec<NativeProbeDirectorySeal>)> {
    reviewed.preflight_filesystem(filesystem).await?;
    let mut sources = Vec::new();
    let mut directories = Vec::new();
    let mut remaining_entries = MAX_NATIVE_PROBE_ENTRIES;
    for entry in entries {
        let path = reviewed.root.join(entry.path.as_std_path());
        let metadata = filesystem
            .metadata(&path)
            .await?
            .context("Native source input disappeared")?;
        ensure!(
            metadata.inode == entry.inode
                && !metadata.is_symlink
                && !metadata.is_fifo
                && metadata.is_dir == entry.is_dir(),
            "Native source input identity changed"
        );
        if entry.is_dir() {
            directories.push(NativeProbeDirectorySeal {
                path: path.clone(),
                inode: metadata.inode,
                mtime: metadata.mtime,
                children: native_probe_directory_children(
                    filesystem,
                    &path,
                    &mut remaining_entries,
                )
                .await?,
            });
        } else {
            ensure!(
                Some(metadata.mtime) == entry.mtime && metadata.len == entry.size,
                "Native source snapshot is stale"
            );
            let fingerprint = fingerprint_native_probe_file(
                filesystem,
                &reviewed.root,
                reviewed.root_inode,
                entry.path.as_std_path(),
                metadata,
                MAX_NATIVE_PROBE_SOURCE_BYTES,
            )
            .await?;
            sources.push(NativeProbeFileSeal {
                root: reviewed.root.clone(),
                root_inode: reviewed.root_inode,
                relative: entry.path.as_std_path().to_path_buf(),
                metadata,
                fingerprint,
            });
        }
    }
    revalidate_native_probe_filesystem(filesystem, &sources, &directories).await?;
    Ok((sources, directories))
}

async fn revalidate_native_probe_filesystem(
    filesystem: &dyn fs::Fs,
    files: &[NativeProbeFileSeal],
    directories: &[NativeProbeDirectorySeal],
) -> Result<()> {
    let mut remaining_entries = MAX_NATIVE_PROBE_ENTRIES;
    for directory in directories {
        let metadata = filesystem
            .metadata(&directory.path)
            .await?
            .context("Native source directory disappeared")?;
        ensure!(
            metadata.is_dir
                && !metadata.is_symlink
                && metadata.inode == directory.inode
                && metadata.mtime == directory.mtime,
            "Native source directory identity changed"
        );
        ensure!(
            native_probe_directory_children(filesystem, &directory.path, &mut remaining_entries)
                .await?
                == directory.children,
            "Native source directory membership changed"
        );
    }
    for file in files {
        ensure!(
            fingerprint_native_probe_file(
                filesystem,
                &file.root,
                file.root_inode,
                &file.relative,
                file.metadata,
                MAX_NATIVE_PROBE_SOURCE_BYTES
            )
            .await?
                == file.fingerprint,
            "Native source bytes changed"
        );
    }
    Ok(())
}

fn validate_native_build_environment(environment: &HashMap<String, String>) -> Result<()> {
    validate_native_probe_environment(environment)?;
    ensure!(
        !environment.contains_key("ZED_PROFILE_TASK_TOKEN")
            && environment
                .iter()
                .try_fold(0_usize, |bytes, (key, value)| bytes
                    .checked_add(key.len())
                    .and_then(|bytes| bytes.checked_add(value.len())))
                .is_some_and(|bytes| bytes <= 64 * 1024),
        "Native Build environment exceeds its private receipt bound"
    );
    Ok(())
}

fn validate_native_probe_environment(environment: &HashMap<String, String>) -> Result<()> {
    ensure!(
        environment.len() <= 512
            && environment.iter().all(|(key, value)| !key.is_empty()
                && !key.chars().any(|character| matches!(character, '=' | '\0'))
                && key.len() <= 4096
                && value.len() <= 4096
                && !value.contains('\0')),
        "Native collector probe environment exceeds its bounds"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn native_capture_environment(
    environment: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    validate_native_probe_environment(environment)?;
    ensure!(
        environment
            .iter()
            .try_fold(0_usize, |bytes, (key, value)| {
                bytes
                    .checked_add(key.len())
                    .and_then(|bytes| bytes.checked_add(value.len()))
            })
            .is_some_and(|bytes| bytes <= 64 * 1024),
        "Native capture environment exceeds its private receipt bound"
    );
    let mut environment = environment.clone();
    for key in [
        "_NT_SYMBOL_PATH",
        util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
        util::macos_task_completion::TASK_BUILD_INVOCATION_ENV,
        util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV,
    ] {
        environment.remove(key);
    }
    validate_native_build_environment(&environment)?;
    Ok(environment)
}

fn fingerprint_native_probe_environment(environment: &HashMap<String, String>) -> Result<[u8; 32]> {
    validate_native_probe_environment(environment)?;
    let mut entries = environment.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let mut fingerprint = Sha256::new();
    fingerprint.update(b"native-collector-probe-environment-v1");
    for (key, value) in entries {
        fingerprint.update((key.len() as u64).to_le_bytes());
        fingerprint.update(key.as_bytes());
        fingerprint.update((value.len() as u64).to_le_bytes());
        fingerprint.update(value.as_bytes());
    }
    Ok(fingerprint.finalize().into())
}

async fn run_native_collector_version(
    executable: &Path,
    cwd: &Path,
    environment: &HashMap<String, String>,
    executor: &BackgroundExecutor,
) -> Result<()> {
    let mut command = util::command::new_command(executable);
    validate_native_probe_environment(environment)?;
    command
        .env_clear()
        .arg("--version")
        .current_dir(cwd)
        .envs(environment);
    let output = bounded_cargo_command_output(
        command,
        "native collector version probe",
        MAX_NATIVE_COLLECTOR_VERSION_BYTES,
        CARGO_CONFIGURATION_PROBE_TIMEOUT,
        executor,
    )
    .await?;
    ensure!(
        output.status.success(),
        "Native collector version probe was unsuccessful"
    );
    ensure!(
        matches!(
            output.stdout.as_slice(),
            b"samply 0.13.1" | b"samply 0.13.1\n"
        ),
        "Native capture requires the actual installed Samply 0.13.1 collector"
    );
    Ok(())
}

enum CargoWorkspaceStoreMode {
    Local {
        environment: Entity<ProjectEnvironment>,
        runner: Arc<dyn CargoMetadataRunner>,
        configuration_probe: Arc<dyn CargoConfigurationProbe>,
        shared: Option<(u64, AnyProtoClient)>,
    },
    Remote {
        project_id: u64,
        client: AnyProtoClient,
    },
}

struct Candidate {
    project_path: ProjectPath,
    absolute_path: PathBuf,
    worktree_root: PathBuf,
    environment: Shared<Task<Option<HashMap<String, String>>>>,
    manifest_text: Task<Result<String>>,
    lock_text: Option<(ProjectPath, Task<Result<String>>)>,
    toolchain_text: Option<(ProjectPath, Task<Result<String>>)>,
    trusted: bool,
    private_paths: Arc<[Arc<RelPath>]>,
}

impl CargoWorkspaceStore {
    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_get_workspace);
        client.add_entity_request_handler(Self::handle_cancel_workspace);
    }

    pub fn local(
        worktree_store: Entity<WorktreeStore>,
        environment: Entity<ProjectEnvironment>,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut subscriptions = vec![cx.subscribe(&worktree_store, Self::handle_worktree_event)];
        subscriptions.push(cx.observe_global::<settings::SettingsStore>(|store, _| {
            store.clear_metadata_facts();
        }));
        if let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(cx.subscribe(&trusted_worktrees, |store, _, _, cx| {
                store.revision = store.revision.wrapping_add(1);
                store.clear_metadata_facts();
                cx.emit(CargoWorkspaceStoreEvent::Invalidated);
            }));
        }
        Self {
            mode: CargoWorkspaceStoreMode::Local {
                environment,
                runner: Arc::new(ProcessCargoMetadataRunner {
                    executor: cx.background_executor().clone(),
                }),
                configuration_probe: Arc::new(ProcessCargoConfigurationProbe {
                    executor: cx.background_executor().clone(),
                }),
                shared: None,
            },
            worktree_store,
            revision: 0,
            cached_snapshot: None,
            cached_metadata_facts: Vec::new(),
            cached_metadata_settings_revision: None,
            cached_generation: 0,
            cached_scope: None,
            active_remote_request_id: None,
            active_remote_request_abort: None,
            disconnected: false,
            cancelled_remote_requests: HashSet::new(),
            active_remote_requests: HashMap::default(),
            _subscriptions: subscriptions,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_local_runners_for_test(
        &mut self,
        metadata_runner: Arc<dyn CargoMetadataRunner>,
        compiler_probe: Arc<dyn CargoConfigurationProbe>,
    ) -> Result<()> {
        let CargoWorkspaceStoreMode::Local {
            runner,
            configuration_probe,
            ..
        } = &mut self.mode
        else {
            return Err(anyhow!(
                "Cargo collector fixtures require a local host store"
            ));
        };
        *runner = metadata_runner;
        *configuration_probe = compiler_probe;
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn local_with_runner(
        worktree_store: Entity<WorktreeStore>,
        environment: Entity<ProjectEnvironment>,
        runner: Arc<dyn CargoMetadataRunner>,
    ) -> Self {
        Self {
            mode: CargoWorkspaceStoreMode::Local {
                environment,
                runner,
                configuration_probe: Arc::new(UnavailableCargoConfigurationProbe),
                shared: None,
            },
            worktree_store,
            revision: 0,
            cached_snapshot: None,
            cached_metadata_facts: Vec::new(),
            cached_metadata_settings_revision: None,
            cached_generation: 0,
            cached_scope: None,
            active_remote_request_id: None,
            active_remote_request_abort: None,
            disconnected: false,
            cancelled_remote_requests: HashSet::new(),
            active_remote_requests: HashMap::default(),
            _subscriptions: Vec::new(),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn local_with_runners(
        worktree_store: Entity<WorktreeStore>,
        environment: Entity<ProjectEnvironment>,
        runner: Arc<dyn CargoMetadataRunner>,
        configuration_probe: Arc<dyn CargoConfigurationProbe>,
    ) -> Self {
        Self {
            mode: CargoWorkspaceStoreMode::Local {
                environment,
                runner,
                configuration_probe,
                shared: None,
            },
            worktree_store,
            revision: 0,
            cached_snapshot: None,
            cached_metadata_facts: Vec::new(),
            cached_metadata_settings_revision: None,
            cached_generation: 0,
            cached_scope: None,
            active_remote_request_id: None,
            active_remote_request_abort: None,
            disconnected: false,
            cancelled_remote_requests: HashSet::new(),
            active_remote_requests: HashMap::default(),
            _subscriptions: Vec::new(),
        }
    }

    pub fn remote(
        project_id: u64,
        worktree_store: Entity<WorktreeStore>,
        client: AnyProtoClient,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut subscriptions = vec![cx.subscribe(&worktree_store, Self::handle_worktree_event)];
        subscriptions.push(cx.observe_global::<settings::SettingsStore>(|store, _| {
            store.clear_metadata_facts();
        }));
        if let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(cx.subscribe(&trusted_worktrees, |store, _, _, cx| {
                store.revision = store.revision.wrapping_add(1);
                store.clear_metadata_facts();
                cx.emit(CargoWorkspaceStoreEvent::Invalidated);
            }));
        }
        Self {
            mode: CargoWorkspaceStoreMode::Remote { project_id, client },
            worktree_store,
            revision: 0,
            cached_snapshot: None,
            cached_metadata_facts: Vec::new(),
            cached_metadata_settings_revision: None,
            cached_generation: 0,
            cached_scope: None,
            active_remote_request_id: None,
            active_remote_request_abort: None,
            disconnected: false,
            cancelled_remote_requests: HashSet::new(),
            active_remote_requests: HashMap::default(),
            _subscriptions: subscriptions,
        }
    }

    fn handle_worktree_event(
        &mut self,
        _: Entity<WorktreeStore>,
        event: &WorktreeStoreEvent,
        cx: &mut Context<Self>,
    ) {
        let relevant = match event {
            WorktreeStoreEvent::WorktreeUpdatedEntries(_, entries) => entries
                .iter()
                .any(|(path, _, _)| is_cargo_workspace_input(path)),
            WorktreeStoreEvent::WorktreeUpdatedGitRepositories(_, _)
            | WorktreeStoreEvent::WorktreeUpdatedRootRepoCommonDir(_)
            | WorktreeStoreEvent::WorktreeUpdateSent(_) => false,
            WorktreeStoreEvent::WorktreeAdded(_)
            | WorktreeStoreEvent::WorktreeRemoved(_, _)
            | WorktreeStoreEvent::WorktreeReleased(_, _)
            | WorktreeStoreEvent::WorktreeOrderChanged
            | WorktreeStoreEvent::WorktreeDeletedEntry(_, _) => true,
        };
        if relevant {
            self.revision = self.revision.wrapping_add(1);
            self.clear_metadata_facts();
            cx.emit(CargoWorkspaceStoreEvent::Invalidated);
        }
    }

    pub fn shared(&mut self, project_id: u64, client: AnyProtoClient, _cx: &mut Context<Self>) {
        if let CargoWorkspaceStoreMode::Local { shared, .. } = &mut self.mode {
            *shared = Some((project_id, client));
            self.clear_metadata_facts();
        }
    }

    pub fn reconnected_to_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, CargoWorkspaceStoreMode::Remote { .. }) || !self.disconnected {
            return;
        }
        self.disconnected = false;
        cx.emit(CargoWorkspaceStoreEvent::Invalidated);
    }

    pub fn disconnected_from_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, CargoWorkspaceStoreMode::Remote { .. }) {
            return;
        }
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        self.clear_metadata_facts();
        self.revision = self.revision.wrapping_add(1);
        self.active_remote_request_id = None;
        if let Some(abort) = self.active_remote_request_abort.take() {
            abort.abort();
        }
        cx.emit(CargoWorkspaceStoreEvent::Invalidated);
    }

    pub fn refresh(&mut self, cx: &mut Context<Self>) -> Task<Result<CargoWorkspaceSnapshot>> {
        self.refresh_scoped(None, cx)
    }

    fn clear_metadata_facts(&mut self) {
        self.cached_metadata_facts.clear();
        self.cached_metadata_settings_revision = None;
    }

    fn metadata_facts_for_current_snapshot(
        &self,
        cx: &App,
    ) -> Option<&[CargoWorkspaceMetadataFacts]> {
        if !matches!(self.mode, CargoWorkspaceStoreMode::Local { .. })
            || self.cached_metadata_facts.is_empty()
            || self.cached_metadata_settings_revision.is_none()
            || self.cached_metadata_settings_revision
                != cx
                    .try_global::<settings::SettingsStore>()
                    .and_then(|settings| settings.observed_revision())
        {
            return None;
        }
        let (snapshot, current) = self.cached_snapshot(cx)?;
        if !current {
            return None;
        }
        let trust = TrustedWorktrees::try_get_global(cx)?;
        for facts in &self.cached_metadata_facts {
            let tree = self
                .worktree_store
                .read(cx)
                .worktree_for_id(facts.key.root.worktree_id, cx)?;
            if !tree.read(cx).is_visible()
                || tree.read(cx).as_local().is_none()
                || trust.read(cx).observed_worktree_trust(
                    &self.worktree_store,
                    facts.key.root.worktree_id,
                    cx,
                ) != Some(true)
                || !snapshot
                    .workspaces
                    .iter()
                    .any(|workspace| facts.matches_workspace(workspace))
            {
                return None;
            }
        }
        Some(&self.cached_metadata_facts)
    }

    pub fn cached_snapshot(&self, cx: &App) -> Option<(&CargoWorkspaceSnapshot, bool)> {
        let snapshot = self.cached_snapshot.as_ref()?;
        let current = !self.disconnected
            && self.cached_generation == self.revision
            && snapshot.input_fingerprint
                == cargo_input_fingerprint(&self.worktree_store, self.cached_scope.as_ref(), cx);
        Some((snapshot, current))
    }

    pub(crate) fn refresh_scoped(
        &mut self,
        allowed_worktree_ids: Option<HashSet<settings::WorktreeId>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<CargoWorkspaceSnapshot>> {
        let task = self.refresh_scoped_inner(allowed_worktree_ids.clone(), cx);
        let generation = self.revision;
        let settings_revision = cx
            .try_global::<settings::SettingsStore>()
            .and_then(|settings| settings.observed_revision());
        cx.spawn(async move |this, cx| {
            let CargoMetadataCollection {
                snapshot,
                metadata_facts,
            } = task.await?;
            validate_cargo_snapshot_bounds(&snapshot)?;
            this.update(cx, |store, cx| {
                if store.revision == generation
                    && !store.disconnected
                    && (!snapshot.workspaces.is_empty()
                        || snapshot.failures.is_empty()
                        || store.cached_snapshot.is_none())
                {
                    store.cached_metadata_facts = if settings_revision.is_some()
                        && settings_revision
                            == cx
                                .try_global::<settings::SettingsStore>()
                                .and_then(|settings| settings.observed_revision())
                    {
                        metadata_facts
                    } else {
                        Vec::new()
                    };
                    store.cached_metadata_settings_revision =
                        if store.cached_metadata_facts.is_empty() {
                            None
                        } else {
                            settings_revision
                        };
                    store.cached_snapshot = Some(snapshot.clone());
                    store.cached_generation = generation;
                    store.cached_scope = allowed_worktree_ids;
                }
            })?;
            Ok(snapshot)
        })
    }

    fn refresh_scoped_inner(
        &mut self,
        allowed_worktree_ids: Option<HashSet<settings::WorktreeId>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<CargoMetadataCollection>> {
        if self.disconnected {
            return Task::ready(Err(anyhow!(CargoWorkspaceRemoteError {
                kind: CargoWorkspaceRemoteErrorKind::Disconnected,
            })));
        }
        self.revision = self.revision.wrapping_add(1);
        self.clear_metadata_facts();
        let revision = self.revision;
        match &self.mode {
            CargoWorkspaceStoreMode::Remote { project_id, client } => {
                let project_id = *project_id;
                let client = client.clone();
                if let Some(request_id) = self.active_remote_request_id.replace(revision) {
                    cx.background_spawn({
                        let client = client.clone();
                        async move {
                            client
                                .request(proto::CancelCargoWorkspace {
                                    project_id,
                                    request_id,
                                })
                                .await?;
                            Ok::<(), anyhow::Error>(())
                        }
                    })
                    .detach_and_log_err(cx);
                }
                let worktree_ids = self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .map(|worktree| worktree.read(cx).id())
                    .filter(|worktree_id| {
                        allowed_worktree_ids
                            .as_ref()
                            .is_none_or(|allowed| allowed.contains(worktree_id))
                    })
                    .map(|worktree_id| worktree_id.to_proto())
                    .collect();
                let (abort_handle, abort_registration) = AbortHandle::new_pair();
                if let Some(previous) = self.active_remote_request_abort.replace(abort_handle) {
                    previous.abort();
                }
                let request = cx.background_spawn(async move {
                    client
                        .request(proto::GetCargoWorkspace {
                            project_id,
                            request_id: revision,
                            worktree_ids,
                        })
                        .await
                });
                cx.spawn(async move |this, cx| {
                    let response = Abortable::new(request, abort_registration)
                        .await
                        .map_err(|_| {
                            anyhow!(CargoWorkspaceRemoteError {
                                kind: CargoWorkspaceRemoteErrorKind::Disconnected,
                            })
                        })?
                        .map_err(|error| {
                            let unsupported =
                                error
                                    .downcast_ref::<proto::RpcError>()
                                    .is_some_and(|error| {
                                        error.error_tag("reason") == Some("rust_tools_unsupported")
                                            || error.raw_message().contains("was not handled")
                                            || error
                                                .raw_message()
                                                .starts_with("no handler registered for ")
                                    });
                            anyhow!(CargoWorkspaceRemoteError {
                                kind: if error.error_code() == proto::ErrorCode::Forbidden
                                    && error.error_tag("reason") == Some("cargo_trust_revoked")
                                {
                                    CargoWorkspaceRemoteErrorKind::Restricted
                                } else if unsupported {
                                    CargoWorkspaceRemoteErrorKind::UnsupportedHost
                                } else {
                                    CargoWorkspaceRemoteErrorKind::Disconnected
                                },
                            })
                        });
                    this.update(cx, |store, _| {
                        if store.active_remote_request_id == Some(revision) {
                            store.active_remote_request_id = None;
                            store.active_remote_request_abort = None;
                        }
                        if store.revision != revision {
                            return Err(anyhow!(CargoWorkspaceRemoteError {
                                kind: CargoWorkspaceRemoteErrorKind::Disconnected,
                            }));
                        }
                        Ok(())
                    })??;
                    Ok(CargoMetadataCollection {
                        snapshot: snapshot_from_proto(response?)?,
                        metadata_facts: Vec::new(),
                    })
                })
            }
            CargoWorkspaceStoreMode::Local {
                environment,
                runner,
                configuration_probe,
                ..
            } => {
                let wait_for_scan = self.worktree_store.read(cx).wait_for_initial_scan();
                let environment = environment.clone();
                let runner = runner.clone();
                let configuration_probe = configuration_probe.clone();
                let worktree_store = self.worktree_store.clone();
                cx.spawn(async move |_this, cx| {
                    wait_for_scan.await;
                    let (abort_handle, abort_registration) = AbortHandle::new_pair();
                    let (candidates, input_fingerprint, trust_subscription) = cx.update(|cx| {
                        let (candidates, input_fingerprint) = collect_candidates(
                            &worktree_store,
                            &environment,
                            allowed_worktree_ids.as_ref(),
                            cx,
                        );
                        let worktree_store = worktree_store.downgrade();
                        let subscription = TrustedWorktrees::try_get_global(cx).map(|trust| {
                            cx.subscribe(&trust, move |_, event, _| {
                                if matches!(event, TrustedWorktreesEvent::Restricted(store, _) if *store == worktree_store) {
                                    abort_handle.abort();
                                }
                            })
                        });
                        (candidates, input_fingerprint, subscription)
                    });
                    let collection = collect_snapshot(
                        revision,
                        input_fingerprint,
                        candidates,
                        runner,
                        configuration_probe,
                    );
                    let result = Abortable::new(collection, abort_registration)
                        .await
                        .map_err(|_| {
                            proto::ErrorCode::Forbidden
                                .with_tag("reason", "cargo_trust_revoked")
                                .message("Cargo workspace trust was revoked during metadata collection".into())
                        })?;
                    drop(trust_subscription);
                    result
                })
            }
        }
    }

    fn native_probe_source_entries(
        &self,
        reviewed: &CargoReviewedMetadataImport,
        cx: &App,
    ) -> Result<Vec<worktree::Entry>> {
        self.revalidate_reviewed_metadata_import(reviewed, cx)?;
        let tree = self
            .worktree_store
            .read(cx)
            .worktree_for_id(reviewed.manifest_path.worktree_id, cx)
            .context("Native source worktree is unavailable")?;
        let tree = tree.read(cx);
        let snapshot = tree.snapshot();
        let mut entries = Vec::new();
        let mut bytes = 0_u64;
        for entry in snapshot
            .entries(false, 0)
            .filter(|entry| !entry.is_ignored && !entry.is_private)
        {
            ensure!(
                !entry.is_external
                    && !entry.is_fifo
                    && (entry.is_file() || matches!(entry.kind, worktree::EntryKind::Dir)),
                "Native source worktree is not fully indexed and ordinary"
            );
            ensure!(
                entry.path.as_unix_str().len() <= 4096,
                "Native source path exceeds its bound"
            );
            if entry.is_file() {
                bytes = bytes
                    .checked_add(entry.size)
                    .context("Native source byte budget overflow")?;
            }
            ensure!(
                entries.len() < MAX_NATIVE_PROBE_ENTRIES && bytes <= MAX_NATIVE_PROBE_SOURCE_BYTES,
                "Native source admission exceeds its file/count budget"
            );
            entries.push(entry.clone());
        }
        ensure!(
            entries
                .iter()
                .any(|entry| entry.path.is_empty() && entry.is_dir()),
            "Native source root is unavailable"
        );
        Ok(entries)
    }

    fn validate_native_collector_probe(
        &self,
        probe: &CargoNativeCollectorProbe,
        cx: &Context<Self>,
    ) -> Result<()> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            cfg!(target_os = "macos") && cx.entity_id() == probe.owner,
            "Native collector probe belongs to another host owner"
        );
        ensure!(
            self.native_probe_source_entries(&probe.reviewed, cx)? == probe.source_entries,
            "Native collector source admission changed"
        );
        Ok(())
    }

    fn validate_native_build_intent(
        &self,
        intent: &CargoNativeBuildIntent,
        cx: &Context<Self>,
    ) -> Result<()> {
        task::validate_experimental_feature_requirements(&intent.required_experimental_features)?;
        for feature in &intent.required_experimental_features {
            ensure!(
                cx.has_flag_by_name(feature),
                "The original selected Cargo Task experimental opt-in was disabled"
            );
        }
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            cfg!(target_os = "macos") && cx.entity_id() == intent.owner,
            "Native Build intent belongs to another host owner"
        );
        ensure!(
            matches!(
                self.mode,
                CargoWorkspaceStoreMode::Local { shared: None, .. }
            ),
            "Native Build host route changed"
        );
        ensure!(
            self.native_probe_source_entries(&intent.reviewed, cx)? == intent.source_entries,
            "Native Build source admission changed"
        );
        let tree = self
            .worktree_store
            .read(cx)
            .worktree_for_id(intent.workspace.root.worktree_id, cx)
            .context("Native Build output worktree disappeared")?;
        let relative = intent.output_root.strip_prefix(&intent.reviewed.root)?;
        let relative = RelPath::new(relative, PathStyle::local())?;
        let tree = tree.read(cx);
        let snapshot = tree.snapshot();
        let output = snapshot
            .entry_for_path(relative.as_ref())
            .context("Native Build output base is no longer indexed")?;
        ensure!(
            output.is_dir()
                && output.is_ignored
                && !output.is_private
                && !output.is_external
                && output.inode == intent.output_root_inode,
            "Native Build output base policy or identity changed"
        );
        let facts = self
            .metadata_facts_for_current_snapshot(cx)
            .context("Native Build private metadata became stale")?
            .iter()
            .find(|facts| facts.key == intent.workspace)
            .context("Native Build private workspace disappeared")?;
        let workspace = self
            .cached_snapshot(cx)
            .context("Native Build snapshot disappeared")?
            .0
            .workspaces
            .iter()
            .find(|workspace| workspace.key == intent.workspace)
            .context("Native Build selected workspace disappeared")?;
        let (selection, output_root, logical_arguments) = native_build_selection_with_profile(
            facts,
            workspace,
            &intent.member_id,
            &intent.target,
            intent.profile.as_deref(),
        )?;
        let (package_source_directory, package_source_directory_inode) =
            native_selected_package_directory(
                facts,
                workspace,
                &selection,
                &intent.reviewed.root,
                &intent.source_entries,
            )?;
        let (invocation_output_root, arguments) =
            native_build_run_arguments(&logical_arguments, &output_root, intent.run_id)?;
        let project = intent
            .project
            .upgrade()
            .context("Native Build Project disappeared")?;
        ensure!(
            project.read(cx).is_local()
                && !project.read(cx).is_shared()
                && project.read(cx).cargo_workspace_store().entity_id() == cx.entity_id(),
            "Native Build Project owner or route changed"
        );
        revalidate_native_build_open_buffers(project.read(cx), intent, cx)?;
        ensure!(
            selection == intent.selection
                && package_source_directory == intent.package_source_directory
                && package_source_directory_inode == intent.package_source_directory_inode
                && facts
                    .runtime_dependencies
                    .selected_closure(&selection.package_id)
                    == intent.runtime_packages
                && output_root == intent.output_root
                && arguments == intent.arguments
                && logical_arguments == intent.logical_arguments
                && invocation_output_root == intent.invocation_output_root,
            "Native Build selected recipe changed"
        );
        Ok(())
    }

    pub fn prepare_native_cargo_build_recipe(
        &mut self,
        workspace_key: crate::cargo_workspace::CargoWorkspaceKey,
        member_id: String,
        target: crate::cargo_workspace::CargoTargetModel,
        profile: Option<String>,
        task_environment: HashMap<String, String>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoBuildRecipe>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            cfg!(target_os = "macos"),
            "Native Build recipe requires the selected macOS host"
        );
        validate_native_build_environment(&task_environment)?;
        let facts = self
            .metadata_facts_for_current_snapshot(cx)
            .context("Native Build current private metadata is unavailable")?
            .iter()
            .find(|facts| facts.key == workspace_key)
            .context("Native Build private workspace is unavailable")?;
        let workspace = self
            .cached_snapshot(cx)
            .context("Native Build snapshot is unavailable")?
            .0
            .workspaces
            .iter()
            .find(|workspace| workspace.key == workspace_key)
            .context("Native Build selected workspace is unavailable")?;
        let (_, _, arguments) = native_build_selection_with_profile(
            facts,
            workspace,
            &member_id,
            &target,
            profile.as_deref(),
        )?;
        let tree = self
            .worktree_store
            .read(cx)
            .worktree_for_id(workspace_key.root.worktree_id, cx)
            .context("Native Build worktree disappeared")?;
        let working_directory = tree
            .read(cx)
            .abs_path()
            .join(workspace_key.root.path.as_std_path());
        crate::cargo_workspace::validate_native_cargo_path(&working_directory)?;
        let CargoWorkspaceStoreMode::Local {
            environment,
            shared: None,
            ..
        } = &self.mode
        else {
            anyhow::bail!("Native Build recipe requires an unshared local host")
        };
        let environment = environment.update(cx, |environment, cx| {
            environment.directory_environment(Arc::from(working_directory.as_path()), cx)
        });
        let executor = cx.background_executor().clone();
        let work = cx.background_spawn(async move {
            let resolution = async {
                let mut environment = environment
                    .await
                    .context("Native Build host environment is unavailable")?;
                validate_native_build_environment(&environment)?;
                environment.extend(task_environment.clone());
                validate_native_build_environment(&environment)?;
                let program = which::which_in(
                    "cargo",
                    environment.get("PATH").map(OsString::from),
                    &working_directory,
                )
                .map_err(|_| {
                    anyhow!(
                        "Native Build requires an installed Cargo executable on the selected host"
                    )
                })?;
                crate::cargo_workspace::validate_native_cargo_path(&program)?;
                Ok(NativeCargoBuildRecipe {
                    program,
                    arguments,
                    working_directory,
                    task_environment,
                })
            };
            match futures::future::select(
                resolution.boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right(_) => Err(anyhow!("Native Build recipe resolution timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let recipe = work.await?;
            this.read_with(cx, |store, cx| {
                ensure!(
                    matches!(
                        store.mode,
                        CargoWorkspaceStoreMode::Local { shared: None, .. }
                    ),
                    "Native Build host route changed"
                );
                let facts = store
                    .metadata_facts_for_current_snapshot(cx)
                    .context("Native Build private metadata became stale")?
                    .iter()
                    .find(|facts| facts.key == workspace_key)
                    .context("Native Build workspace disappeared")?;
                let workspace = store
                    .cached_snapshot(cx)
                    .context("Native Build snapshot disappeared")?
                    .0
                    .workspaces
                    .iter()
                    .find(|workspace| workspace.key == workspace_key)
                    .context("Native Build workspace disappeared")?;
                let (_, _, arguments) = native_build_selection_with_profile(
                    facts,
                    workspace,
                    &member_id,
                    &target,
                    profile.as_deref(),
                )?;
                ensure!(
                    arguments == recipe.arguments,
                    "Native Build selected recipe changed"
                );
                let tree = store
                    .worktree_store
                    .read(cx)
                    .worktree_for_id(workspace_key.root.worktree_id, cx)
                    .context("Native Build worktree disappeared")?;
                ensure!(
                    tree.read(cx)
                        .abs_path()
                        .join(workspace_key.root.path.as_std_path())
                        == recipe.working_directory,
                    "Native Build cwd changed"
                );
                Ok(())
            })??;
            Ok(recipe)
        }))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn prepare_native_cargo_runtime_compiler_receipt(
        &mut self,
        intent: CargoNativeBuildIntent,
        launch: &task::SpawnInTerminal,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoRuntimeCompilerReceipt>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        intent.verify_eligible_launch(launch)?;
        let checked = self.revalidate_native_cargo_build_intent(intent, cx)?;
        let launch = launch.clone();
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native runtime filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        Ok(cx.spawn(async move |this, cx| {
            let mut intent = checked.await?;
            let project = intent.project.upgrade().context("Native runtime Project disappeared")?;
            let prepared = project.update(cx, |project, cx| project.prepare_native_task_environment(&launch, cx))?;
            let recipe = prepared.await?;
            ensure!(recipe.working_directory() == intent.working_directory()
                && recipe.logical_environment() == intent.task_environment(),
                "Native runtime complete environment no longer matches its logical Task");
            intent.environment_recipe = Some(recipe);
            intent.revalidate_environment(cx)?.await?;
            this.update(cx, |store, cx| store.validate_native_build_intent(&intent, cx))??;
            let work = cx.background_spawn(async move {
                let probe = async {
                    revalidate_native_build_filesystem(filesystem.as_ref(), &intent).await?;
                    let recipe = intent.environment_recipe.as_ref().context("Native runtime complete environment disappeared")?;
                    let environment = PassiveRustEnvironment::from_native_recipe(recipe);
                    let (configuration, configuration_seals) = passive_cargo_configuration_with_native_review(
                        &intent.cwd, &environment, Some(filesystem.clone())).await?;
                    validate_native_precompiler_loader_environment(&environment, &configuration)?;
                    ensure!(!configuration.native_runtime_target_or_runner_configured,
                        "Native runtime preparation does not yet support authored target, runner or unstable host configuration");
                    ensure!(configuration.environment.keys().all(|name| !name.starts_with("ZED_PROFILE_")),
                        "Native runtime configuration conflicts with private Task transport");
                    let execution = prepare_passive_rust_execution_in_environment(
                        Some(intent.requested_program.as_os_str()), false, &intent.cwd, environment, &executor).await?;
                    for (name, value) in &execution.child_environment.overrides {
                        let name = name.to_str().context("Native runtime environment name is not UTF-8")?;
                        let selected_target = name == "CARGO_BUILD_TARGET" || name.starts_with("CARGO_UNSTABLE_")
                            || (name.starts_with("CARGO_TARGET_") && name.ends_with("_RUNNER"));
                        ensure!(!selected_target || value.is_empty(),
                            "Native runtime preparation does not yet support configured target or runner overrides");
                    }
                    let executables = seal_native_compiler_executables(filesystem.as_ref(), &execution, &intent.cwd).await?;
                    revalidate_native_cargo_configuration(filesystem.as_ref(), &configuration_seals).await?;
                    let observations = native_runtime_compiler_observations(&execution, &intent.cwd, &executor).await?;
                    let compiler = crate::cargo_workspace::parse_rustc_verbose_version(&observations[0])?;
                    let host_target = compiler.host_target.context("Native runtime compiler host tuple is unavailable")?;
                    ensure!(host_target.len() <= 128 && host_target.ends_with("-apple-darwin")
                        && host_target.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'),
                        "Native runtime preparation requires an actual macOS compiler host");
                    let (sysroot, sysroot_inode) = native_compiler_path_output(&observations[1], filesystem.as_ref()).await?;
                    let (target_library_directory, target_library_directory_inode) = native_compiler_path_output(&observations[2], filesystem.as_ref()).await?;
                    ensure!(target_library_directory.starts_with(&sysroot) && target_library_directory != sysroot,
                        "Native runtime compiler library directory is outside its observed sysroot");
                    let mut budget = NativeCompilerLibraryBudget::default();
                    let (library_members, libraries) = seal_native_compiler_runtime_libraries_with_budget(filesystem.as_ref(),
                        &target_library_directory, target_library_directory_inode, &mut budget).await?;
                    let sysroot_library_directory = seal_native_compiler_sysroot_libraries(filesystem.as_ref(),
                        &sysroot, &mut budget).await?;
                    let receipt = NativeCargoRuntimeCompilerReceipt { intent, facts: NativeCargoRuntimeCompilerFacts {
                        configuration, configuration_seals, execution, executables, observations,
                        host_target, sysroot, sysroot_inode, target_library_directory, target_library_directory_inode,
                        library_members, libraries, sysroot_library_directory } };
                    revalidate_native_compiler_receipt(filesystem, &receipt, &executor).await?;
                    Ok::<_, anyhow::Error>(receipt)
                };
                match futures::future::select(probe.boxed(), executor.timer(CARGO_METADATA_TIMEOUT).boxed()).await {
                    Either::Left((result, _)) => result,
                    Either::Right(_) => Err(anyhow!("Native runtime compiler receipt preparation timed out")),
                }
            });
            let receipt = work.await?;
            receipt.intent.revalidate_environment(cx)?.await?;
            this.update(cx, |store, cx| store.validate_native_build_intent(&receipt.intent, cx))??;
            Ok(receipt)
        }))
    }

    pub fn prepare_native_cargo_runtime_recipe(
        &mut self,
        workspace_key: crate::cargo_workspace::CargoWorkspaceKey,
        member_id: String,
        target: crate::cargo_workspace::CargoTargetModel,
        profile: Option<String>,
        task_environment: HashMap<String, String>,
        workload_arguments: Vec<String>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoBuildRecipe>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        let selected = self.prepare_native_cargo_build_recipe(
            workspace_key,
            member_id,
            target,
            profile,
            task_environment,
            cx,
        )?;
        Ok(cx.spawn(async move |_, _| {
            let mut recipe = selected.await?;
            recipe.arguments = crate::cargo_workspace::native_cargo_runtime_logical_arguments(
                &recipe.arguments,
                &workload_arguments,
            )?;
            Ok(recipe)
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_native_cargo_runtime_intent_for_launch(
        &mut self,
        project: Entity<crate::Project>,
        run_id: uuid::Uuid,
        launch: &task::SpawnInTerminal,
        continuation: &task::TaskNativeProfileContinuation,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeRuntimeIntent>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            launch.literal_prefix_args == launch.args.len(),
            "Native runtime preparation requires literal selected Task arguments"
        );
        let mut selected_build = launch.clone();
        selected_build.args =
            crate::cargo_workspace::native_cargo_runtime_selected_build_arguments(
                &launch.args,
                &continuation.workload_arguments,
            )?;
        selected_build.literal_prefix_args = selected_build.args.len();
        let prepared = self.prepare_native_cargo_build_intent_for_launch(
            project.clone(),
            run_id,
            &selected_build,
            cx,
        )?;
        let logical = launch.clone();
        let workload_arguments = continuation.workload_arguments.clone();
        Ok(cx.spawn(async move |this, cx| {
            let intent = prepared.await?;
            let prepared = this.update(cx, |store, cx| {
                store.prepare_native_cargo_runtime_compiler_receipt(intent, &selected_build, cx)
            })??;
            let compiler = prepared.await?;
            let prepared = project.update(cx, |project, cx| {
                project.prepare_local_native_cargo_runtime_helper_image(cx)
            })?;
            let helper = Arc::new(prepared.await?);
            let arguments =
                native_runtime_checked_arguments(&compiler, &helper, &workload_arguments)?;
            let loader_scope =
                NativeCargoRuntimeLoaderScope::from_compiler(&compiler.intent, &compiler.facts)?;
            for (name, value) in &compiler.facts.execution.cargo_parent_environment.overrides {
                ensure!(
                    compiler
                        .intent
                        .environment_recipe
                        .as_ref()
                        .context("Native runtime complete environment disappeared")?
                        .environment()
                        .get(
                            name.to_str()
                                .context("Native runtime parent key is not UTF-8")?
                        )
                        .map(|value| OsStr::new(value))
                        == Some(value.as_os_str()),
                    "Native runtime admitted parent environment changed"
                );
            }
            let mut task_environment = native_runtime_outer_environment(
                compiler
                    .intent
                    .environment_recipe
                    .as_ref()
                    .context("Native runtime complete outer environment disappeared")?
                    .environment(),
                compiler.facts.execution.toolchain.as_deref(),
            )?;
            loader_scope.pin_outer_fallback(&mut task_environment)?;
            let expected = util::macos_task_completion::TaskBuildInvocation::new(
                compiler.intent.requested_program.clone(),
                arguments,
                compiler.intent.cwd.clone(),
                task_environment
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
            )
            .map_err(|_| anyhow!("Native runtime invocation exceeds its supported bounds"))?;
            compiler.intent.revalidate_environment(cx)?.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&compiler.intent, cx)
            })??;
            project
                .update(cx, |project, cx| {
                    project.revalidate_local_native_cargo_runtime_helper_image(&helper, cx)
                })?
                .await?;
            loader_scope.paths(expected.environment(), NativeRuntimeLoaderStage::OuterCargo)?;
            let lease = NativeCargoRuntimeAdmissionLease {
                owner: this.clone(),
                project: project.downgrade(),
                run_id,
                helper: helper.clone(),
                expected,
                admission: Arc::new(parking_lot::Mutex::new(Some(NativeCargoRuntimeAdmission {
                    compiler,
                    loader_scope,
                    helper,
                }))),
            };
            Ok(CargoNativeRuntimeIntent {
                logical,
                lease,
                task_environment,
                workload_arguments,
            })
        }))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn revalidate_native_cargo_runtime_admission(
        &mut self,
        lease: &NativeCargoRuntimeAdmissionLease,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            lease.owner.entity_id() == cx.entity_id(),
            "Native runtime admission belongs to another owner"
        );
        let admission = lease
            .admission
            .lock()
            .take()
            .context("Native runtime admission is expired or already in use")?;
        self.validate_native_build_intent(&admission.compiler.intent, cx)?;
        ensure!(
            admission.loader_scope.matches_compiler(&admission.compiler),
            "Native runtime loader scope changed before renewal"
        );
        admission.loader_scope.paths(
            lease.expected.environment(),
            NativeRuntimeLoaderStage::OuterCargo,
        )?;
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native runtime filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let lease = lease.clone();
        Ok(cx.spawn(async move |this, cx| {
            admission
                .compiler
                .intent
                .revalidate_environment(cx)?
                .await?;
            let project = admission
                .compiler
                .intent
                .project
                .upgrade()
                .context("Native runtime Project disappeared")?;
            project
                .update(cx, |project, cx| {
                    project
                        .revalidate_local_native_cargo_runtime_helper_image(&admission.helper, cx)
                })?
                .await?;
            let admission = cx
                .background_spawn(async move {
                    let renewal = revalidate_native_compiler_receipt(
                        filesystem,
                        &admission.compiler,
                        &executor,
                    );
                    match futures::future::select(
                        renewal.boxed(),
                        executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
                    )
                    .await
                    {
                        Either::Left((result, _)) => result?,
                        Either::Right(_) => {
                            anyhow::bail!("Native runtime admission renewal timed out")
                        }
                    }
                    Ok::<_, anyhow::Error>(admission)
                })
                .await?;
            admission
                .compiler
                .intent
                .revalidate_environment(cx)?
                .await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&admission.compiler.intent, cx)
            })??;
            project
                .update(cx, |project, cx| {
                    project
                        .revalidate_local_native_cargo_runtime_helper_image(&admission.helper, cx)
                })?
                .await?;
            ensure!(
                lease.admission.lock().is_none(),
                "Native runtime admission ownership changed"
            );
            *lease.admission.lock() = Some(admission);
            Ok(())
        }))
    }

    pub fn prepare_native_cargo_build_intent_for_launch(
        &mut self,
        project: Entity<crate::Project>,
        run_id: uuid::Uuid,
        launch: &task::SpawnInTerminal,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeBuildIntent>>> {
        crate::task_store::validate_experimental_task_requirements(launch, cx)?;
        let required_experimental_features = launch.required_experimental_features.clone();
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        use crate::cargo_workspace::CargoTargetKind;
        ensure!(
            launch.literal_prefix_args == launch.args.len(),
            "Native Build requires a literal Task recipe"
        );
        let program = launch
            .command
            .as_ref()
            .context("Native Build Task has no executable")?;
        let program = PathBuf::from(program);
        crate::cargo_workspace::validate_native_cargo_path(&program)?;
        let cwd = launch
            .cwd
            .as_ref()
            .context("Native Build Task has no frozen working directory")?;
        crate::cargo_workspace::validate_native_cargo_path(cwd)?;
        validate_native_build_environment(&launch.env)?;
        ensure!(
            matches!(launch.args.len(), 8 | 10 | 12)
                && launch.args.first().is_some_and(|value| value == "build")
                && launch
                    .args
                    .get(1)
                    .is_some_and(|value| value == "--message-format=json")
                && launch.args.get(2).is_some_and(|value| value == "--package"),
            "Native Build Task is outside the supported Cargo recipe"
        );
        let profile =
            crate::cargo_workspace::native_cargo_build_profile(&launch.args)?.map(str::to_owned);
        let kind = match launch.args.get(4).map(String::as_str) {
            Some("--bin") => CargoTargetKind::Binary,
            Some("--example") => CargoTargetKind::Example,
            _ => anyhow::bail!("Native Build requires a selected binary or example"),
        };
        let snapshot = self
            .cached_snapshot(cx)
            .context("Native Build snapshot is unavailable")?
            .0;
        let facts = self
            .metadata_facts_for_current_snapshot(cx)
            .context("Native Build current private metadata is unavailable")?;
        let mut selection = None;
        for workspace in &snapshot.workspaces {
            let Some(private) = facts.iter().find(|facts| facts.key == workspace.key) else {
                continue;
            };
            let tree = self
                .worktree_store
                .read(cx)
                .worktree_for_id(workspace.key.root.worktree_id, cx)
                .context("Native Build worktree disappeared")?;
            let expected_cwd = tree
                .read(cx)
                .abs_path()
                .join(workspace.key.root.path.as_std_path());
            if expected_cwd != *cwd {
                continue;
            }
            for member in workspace
                .members
                .iter()
                .filter(|member| launch.args.get(3) == Some(&member.name))
            {
                for target in member.targets.iter().filter(|target| {
                    target.kind == kind && launch.args.get(5) == Some(&target.name)
                }) {
                    let (_, _, arguments) = native_build_selection_with_profile(
                        private,
                        workspace,
                        &member.id,
                        target,
                        profile.as_deref(),
                    )?;
                    ensure!(
                        arguments == launch.args,
                        "Native Build Task no longer matches its exact selected recipe"
                    );
                    ensure!(
                        selection.is_none(),
                        "Native Build Task selection is ambiguous"
                    );
                    selection = Some((workspace.key.clone(), member.id.clone(), target.clone()));
                }
            }
        }
        let (workspace, member, target) =
            selection.context("Native Build Task has no current exact selection")?;
        let prepared = self.prepare_native_cargo_build_intent(
            project,
            run_id,
            launch.id.clone(),
            workspace,
            member,
            target,
            profile,
            program,
            launch.env.clone(),
            cx,
        )?;
        Ok(cx.spawn(async move |this, cx| {
            let mut intent = prepared.await?;
            intent.required_experimental_features = required_experimental_features;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&intent, cx)
            })??;
            Ok(intent)
        }))
    }

    pub fn prepare_native_cargo_build_intent(
        &mut self,
        project: Entity<crate::Project>,
        run_id: uuid::Uuid,
        task_id: task::TaskId,
        workspace_key: crate::cargo_workspace::CargoWorkspaceKey,
        member_id: String,
        target: crate::cargo_workspace::CargoTargetModel,
        profile: Option<String>,
        requested_program: PathBuf,
        task_environment: HashMap<String, String>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeBuildIntent>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            cfg!(target_os = "macos")
                && matches!(
                    self.mode,
                    CargoWorkspaceStoreMode::Local { shared: None, .. }
                ),
            "Native Build requires an unshared local macOS host"
        );
        ensure!(
            !run_id.is_nil()
                && project.read(cx).cargo_workspace_store().entity_id() == cx.entity_id()
                && project.read(cx).is_local()
                && !project.read(cx).is_shared(),
            "Native Build Project or run identity is invalid"
        );
        crate::cargo_workspace::validate_native_cargo_path(&requested_program)?;
        validate_native_build_environment(&task_environment)?;
        let facts = self
            .metadata_facts_for_current_snapshot(cx)
            .context("Native Build current private metadata is unavailable")?
            .iter()
            .find(|facts| facts.key == workspace_key)
            .context("Native Build private workspace is unavailable")?;
        let workspace = self
            .cached_snapshot(cx)
            .context("Native Build snapshot is unavailable")?
            .0
            .workspaces
            .iter()
            .find(|workspace| workspace.key == workspace_key)
            .context("Native Build selected workspace is unavailable")?;
        let (selection, output_root, logical_arguments) = native_build_selection_with_profile(
            facts,
            workspace,
            &member_id,
            &target,
            profile.as_deref(),
        )?;
        let runtime_packages = facts
            .runtime_dependencies
            .selected_closure(&selection.package_id);
        let (invocation_output_root, arguments) =
            native_build_run_arguments(&logical_arguments, &output_root, run_id)?;
        let manifest = workspace
            .root_manifest
            .clone()
            .context("Native Build workspace manifest is unavailable")?;
        let reviewed = self.prepare_reviewed_metadata_import(manifest, cx)?;
        let source_entries = self.native_probe_source_entries(&reviewed, cx)?;
        let (package_source_directory, package_source_directory_inode) =
            native_selected_package_directory(
                facts,
                workspace,
                &selection,
                &reviewed.root,
                &source_entries,
            )?;
        let buffer_versions = native_build_open_buffer_versions(
            project.read(cx),
            workspace_key.root.worktree_id,
            cx,
        )?;
        let tree = self
            .worktree_store
            .read(cx)
            .worktree_for_id(workspace_key.root.worktree_id, cx)
            .context("Native Build worktree disappeared")?;
        let relative_output = output_root
            .strip_prefix(&reviewed.root)
            .context("Native Build output root is outside its worktree")?;
        let relative_output = RelPath::new(relative_output, PathStyle::local())?;
        let output_entry = tree
            .read(cx)
            .snapshot()
            .entry_for_path(relative_output.as_ref())
            .cloned();
        if output_entry.is_none() {
            let filesystem = self
                .worktree_store
                .read(cx)
                .fs()
                .context("Native Build filesystem is unavailable")?
                .clone();
            let executor = cx.background_executor().clone();
            let preparation = cx.background_spawn({
                let filesystem = filesystem.clone();
                let reviewed = reviewed.clone();
                let entries = source_entries.clone();
                let program = requested_program.clone();
                async move {
                    let work = async {
                        let sources =
                            seal_native_probe_sources(filesystem.as_ref(), &reviewed, &entries)
                                .await?;
                        let canonical = filesystem.canonicalize(&program).await?;
                        let root = canonical
                            .parent()
                            .context("Native Build program root is unavailable")?
                            .to_path_buf();
                        let root_metadata = filesystem
                            .metadata(&root)
                            .await?
                            .context("Native Build program root disappeared")?;
                        let metadata = filesystem
                            .metadata(&canonical)
                            .await?
                            .context("Native Build program disappeared")?;
                        let relative = canonical.strip_prefix(&root)?.to_path_buf();
                        let fingerprint = fingerprint_native_probe_file(
                            filesystem.as_ref(),
                            &root,
                            root_metadata.inode,
                            &relative,
                            metadata,
                            MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES,
                        )
                        .await?;
                        Ok::<_, anyhow::Error>((
                            sources,
                            NativeProbeFileSeal {
                                root,
                                root_inode: root_metadata.inode,
                                relative,
                                metadata,
                                fingerprint,
                            },
                        ))
                    };
                    match futures::future::select(
                        work.boxed(),
                        executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
                    )
                    .await
                    {
                        Either::Left((result, _)) => result,
                        Either::Right(_) => {
                            Err(anyhow!("Native clean-first source admission timed out"))
                        }
                    }
                }
            });
            let relative_output: Arc<RelPath> = Arc::from(relative_output);
            return Ok(cx.spawn(async move |this, cx| {
                let ((before_files, before_directories), before_program) = preparation.await?;
                this.update(cx, |store, cx| {
                    store.revalidate_reviewed_metadata_import(&reviewed, cx)?;
                    ensure!(
                        store.native_probe_source_entries(&reviewed, cx)? == source_entries,
                        "Native clean-first source admission changed before creation"
                    );
                    revalidate_native_build_buffer_versions(
                        project.read(cx),
                        workspace_key.root.worktree_id,
                        &buffer_versions,
                        cx,
                    )
                })??;
                let creation = tree.update(cx, |tree, cx| {
                    tree.as_local()
                        .context("Native clean-first output became remote")
                        .map(|tree| tree.create_ignored_directory_for_task(relative_output, cx))
                })?;
                creation.await?;
                let prepared = this.update(cx, |store, cx| {
                    store.revalidate_reviewed_metadata_import(&reviewed, cx)?;
                    revalidate_native_build_buffer_versions(
                        project.read(cx),
                        workspace_key.root.worktree_id,
                        &buffer_versions,
                        cx,
                    )?;
                    let current = store.native_probe_source_entries(&reviewed, cx)?;
                    validate_native_created_output_entries(
                        &source_entries,
                        &current,
                        &output_root,
                        &reviewed.root,
                    )?;
                    store.prepare_native_cargo_build_intent(
                        project.clone(),
                        run_id,
                        task_id,
                        workspace_key,
                        member_id,
                        target,
                        profile,
                        requested_program,
                        task_environment,
                        cx,
                    )
                })??;
                let intent = prepared.await?;
                validate_native_created_output_seals(
                    &before_files,
                    &before_directories,
                    &intent.sources,
                    &intent.directories,
                    &output_root,
                )?;
                ensure!(
                    native_build_file_seals_equal(&before_program, &intent.program),
                    "Native Build program changed during output creation"
                );
                this.update(cx, |store, cx| {
                    revalidate_native_build_buffer_versions(
                        project.read(cx),
                        intent.workspace.root.worktree_id,
                        &buffer_versions,
                        cx,
                    )?;
                    store.validate_native_build_intent(&intent, cx)
                })??;
                Ok(intent)
            }));
        }
        let output_entry = output_entry.context("Native Build output root is unavailable")?;
        ensure!(
            output_entry.is_dir()
                && output_entry.is_ignored
                && !output_entry.is_private
                && !output_entry.is_external,
            "Native Build output root is not an ordinary permitted ignored directory"
        );
        let source_relative = selection
            .source_path
            .strip_prefix(&reviewed.root)
            .context("Native Build selected source is outside its worktree")?;
        let source_relative = RelPath::new(source_relative, PathStyle::local())?;
        ensure!(
            target
                .source_path
                .as_ref()
                .is_some_and(
                    |source| source.worktree_id == workspace_key.root.worktree_id
                        && source.path.as_ref() == source_relative.as_ref()
                ),
            "Native Build selected source projection changed"
        );
        ensure!(
            source_entries
                .iter()
                .any(|entry| entry.path.as_ref() == source_relative.as_ref() && entry.is_file()),
            "Native Build selected source is private, ignored, unavailable or not ordinary"
        );
        let cwd = reviewed.root.join(workspace_key.root.path.as_std_path());
        crate::cargo_workspace::validate_native_cargo_path(&cwd)?;
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native Build filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let owner = cx.entity_id();
        let work = cx.background_spawn(async move {
            let preparation = async {
                let (sources, directories) =
                    seal_native_probe_sources(filesystem.as_ref(), &reviewed, &source_entries)
                        .await?;
                ensure!(
                    filesystem.canonicalize(&cwd).await? == cwd,
                    "Native Build cwd has aliases"
                );
                let cwd_metadata = filesystem
                    .metadata(&cwd)
                    .await?
                    .context("Native Build cwd disappeared")?;
                ensure!(
                    cwd_metadata.is_dir && !cwd_metadata.is_symlink,
                    "Native Build cwd is not ordinary"
                );
                ensure!(
                    filesystem.canonicalize(&output_root).await? == output_root,
                    "Native Build output root has aliases"
                );
                let output_metadata = filesystem
                    .metadata(&output_root)
                    .await?
                    .context("Native Build output root disappeared")?;
                ensure!(
                    output_metadata.is_dir
                        && !output_metadata.is_symlink
                        && !output_metadata.is_fifo
                        && output_metadata.inode == output_entry.inode,
                    "Native Build output root identity changed"
                );
                let canonical_program = filesystem.canonicalize(&requested_program).await?;
                let root = canonical_program
                    .parent()
                    .context("Native Build program root is unavailable")?
                    .to_path_buf();
                let root_metadata = filesystem
                    .metadata(&root)
                    .await?
                    .context("Native Build program root disappeared")?;
                ensure!(
                    root_metadata.is_dir && !root_metadata.is_symlink,
                    "Native Build program root is not ordinary"
                );
                let relative = canonical_program.strip_prefix(&root)?.to_path_buf();
                let metadata = filesystem
                    .metadata(&canonical_program)
                    .await?
                    .context("Native Build program disappeared")?;
                ensure!(
                    metadata.is_executable && metadata.len > 0,
                    "Native Build program is not executable"
                );
                let fingerprint = fingerprint_native_probe_file(
                    filesystem.as_ref(),
                    &root,
                    root_metadata.inode,
                    &relative,
                    metadata,
                    MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES,
                )
                .await?;
                ensure!(
                    filesystem
                        .metadata(&invocation_output_root)
                        .await?
                        .is_none(),
                    "Native Build run output already exists"
                );
                let intent = CargoNativeBuildIntent {
                    required_experimental_features: Vec::new(),
                    project: project.downgrade(),
                    run_id,
                    task_id,
                    invocation_output_root,
                    logical_arguments,
                    owner,
                    workspace: workspace_key,
                    member_id,
                    target,
                    profile,
                    selection,
                    runtime_packages,
                    reviewed,
                    source_entries,
                    buffer_versions,
                    sources,
                    directories,
                    requested_program,
                    program: NativeProbeFileSeal {
                        root,
                        root_inode: root_metadata.inode,
                        relative,
                        metadata,
                        fingerprint,
                    },
                    arguments,
                    cwd,
                    cwd_inode: cwd_metadata.inode,
                    package_source_directory,
                    package_source_directory_inode,
                    output_root,
                    output_root_inode: output_metadata.inode,
                    task_environment,
                    #[cfg(target_os = "macos")]
                    environment_recipe: None,
                };
                revalidate_native_build_filesystem(filesystem.as_ref(), &intent).await?;
                Ok(intent)
            };
            match futures::future::select(
                preparation.boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right(_) => Err(anyhow!("Native Build preparation timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let intent = work.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&intent, cx)
            })??;
            Ok(intent)
        }))
    }

    pub fn revalidate_native_cargo_build_intent(
        &mut self,
        intent: CargoNativeBuildIntent,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeBuildIntent>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        self.validate_native_build_intent(&intent, cx)?;
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native Build filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let work = cx.background_spawn(async move {
            let revalidation = async {
                revalidate_native_build_filesystem(filesystem.as_ref(), &intent).await?;
                ensure!(
                    filesystem
                        .metadata(&intent.invocation_output_root)
                        .await?
                        .is_none(),
                    "Native Build run output already exists before launch"
                );
                Ok(intent)
            };
            match futures::future::select(
                revalidation.boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right(_) => Err(anyhow!("Native Build revalidation timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let intent = work.await?;
            #[cfg(target_os = "macos")]
            if intent.environment_recipe.is_some() {
                intent.revalidate_environment(cx)?.await?;
            }
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&intent, cx)
            })??;
            Ok(intent)
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_native_cargo_runtime_binding(
        &mut self,
        intent: CargoNativeRuntimeIntent,
        launch: &task::SpawnInTerminal,
        report_limit: usize,
        cx: &mut Context<Self>,
    ) -> Result<
        Task<
            Result<(
                CargoNativeRuntimeIntent,
                crate::profile_results_store::BuildTaskReportBinding,
            )>,
        >,
    > {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            (task::MIN_TASK_STDOUT_REPORT_BYTES..=task::MAX_TASK_STDOUT_REPORT_BYTES)
                .contains(&report_limit),
            "Native runtime declared report limit is invalid"
        );
        intent.verify_eligible_launch(launch)?;
        let checked = self.revalidate_native_cargo_runtime_admission(&intent.lease, cx)?;
        Ok(cx.spawn(async move |this, cx| {
            checked.await?;
            let (run_id, project, helper) = {
                let admission = intent.lease.admission.lock();
                let admission = admission
                    .as_ref()
                    .context("Native runtime admission expired before observation")?;
                (
                    admission.compiler.intent.run_id,
                    admission
                        .compiler
                        .intent
                        .project
                        .upgrade()
                        .context("Native runtime Project disappeared")?,
                    admission.helper.clone(),
                )
            };
            let prepared = project.update(cx, |project, cx| {
                project.prepare_checked_local_cargo_runtime_report(
                    run_id.to_string(),
                    report_limit,
                    intent.expected_invocation(),
                    &helper,
                    intent.registration_lease(),
                    cx,
                )
            })?;
            let binding = prepared.await?;
            ensure!(
                binding.requires_cargo_runtime_observation()
                    && uuid::Uuid::parse_str(binding.run_id()).ok() == Some(run_id),
                "Native runtime observation belongs to another mode or producing Task"
            );
            let checked = this.update(cx, |store, cx| {
                store.revalidate_native_cargo_runtime_admission(&intent.lease, cx)
            })??;
            checked.await?;
            Ok((intent, binding))
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_native_cargo_build_binding(
        &mut self,
        intent: CargoNativeBuildIntent,
        launch: &task::SpawnInTerminal,
        report_limit: usize,
        cx: &mut Context<Self>,
    ) -> Result<
        Task<
            Result<(
                CargoNativeBuildIntent,
                crate::profile_results_store::BuildTaskReportBinding,
            )>,
        >,
    > {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            (task::MIN_TASK_STDOUT_REPORT_BYTES..=task::MAX_TASK_STDOUT_REPORT_BYTES)
                .contains(&report_limit),
            "Native Build declared report limit is invalid"
        );
        intent.verify_eligible_launch(launch)?;
        let checked = self.revalidate_native_cargo_build_intent(intent, cx)?;
        let launch = launch.clone();
        Ok(cx.spawn(async move |this, cx| {
            let mut intent = checked.await?;
            let project = intent
                .project
                .upgrade()
                .context("Native Build Project disappeared")?;
            let prepared = project.update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })?;
            let recipe = prepared.await?;
            ensure!(
                recipe.working_directory() == intent.working_directory()
                    && recipe.logical_environment() == intent.task_environment(),
                "Native Build complete environment no longer matches its logical Task"
            );
            intent.environment_recipe = Some(recipe);
            intent.revalidate_environment(cx)?.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&intent, cx)
            })??;
            let expected = intent.expected_invocation()?;
            let prepared = project.update(cx, |project, cx| {
                project.prepare_checked_local_cargo_build_report(
                    intent.run_id.to_string(),
                    report_limit,
                    expected,
                    cx,
                )
            })?;
            let binding = prepared.await?;
            ensure!(
                binding.requires_checked_invocation()
                    && binding.run_id() == intent.run_id.to_string(),
                "Native Build observer identity changed"
            );
            let checked = this.update(cx, |store, cx| {
                store.revalidate_native_cargo_build_intent(intent, cx)
            })??;
            let intent = checked.await?;
            Ok((intent, binding))
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn finish_native_cargo_runtime(
        &mut self,
        intent: CargoNativeRuntimeIntent,
        report: crate::profile_results_store::ObservedBuildTaskReport,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoWorkload>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            uuid::Uuid::parse_str(report.run_id()).ok() == Some(intent.lease.run_id)
                && report.runtime_admission_matches(&intent.lease)?,
            "Native runtime report has no matching issued producing Task admission"
        );
        ensure!(
            report.observed_cargo_runtime()?.is_some(),
            "Native runtime report has no authenticated Cargo runtime observation"
        );
        let checked = self.revalidate_native_cargo_runtime_admission(&intent.lease, cx)?;
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native runtime filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        Ok(cx.spawn(async move |this, cx| {
            checked.await?;
            let admission = intent
                .lease
                .admission
                .lock()
                .take()
                .context("Native runtime admission expired before completion")?;
            ensure!(
                admission.loader_scope.matches_compiler(&admission.compiler),
                "Native runtime completed loader scope differs from its compiler authority"
            );
            let NativeCargoRuntimeCompilerReceipt {
                intent: selected,
                facts,
            } = admission.compiler;
            let actual = report
                .observed_cargo_runtime()?
                .context("Native runtime observation expired")?;
            ensure!(
                actual.arguments() == intent.workload_arguments.as_slice()
                    && actual.working_directory() == selected.cwd.as_path(),
                "Native runtime observation differs from its selected literal workload or cwd"
            );
            let environment = actual
                .environment()
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
            let proof = NativeCargoRuntimeWorkload {
                compiler: facts,
                lease: intent.registration_lease(),
                environment,
                workload_arguments: intent.workload_arguments,
                libraries: Vec::new(),
            };
            let expected = intent.lease.expected.clone();
            let prepared = this.update(cx, |store, cx| {
                store.finish_native_cargo_build_with_invocation(
                    selected,
                    report,
                    expected,
                    Some(proof),
                    cx,
                )
            })??;
            let workload = prepared.await?;
            let work = cx.background_spawn(async move {
                let sealing = async {
                    let mut workload = workload;
                    let actual = workload
                        .report
                        .observed_cargo_runtime()?
                        .context("Native runtime observation expired")?;
                    ensure!(
                        actual.program() == workload.executable_path(),
                        "Native Cargo runtime executable differs from its exact produced artifact"
                    );
                    let proof = workload
                        .runtime
                        .as_mut()
                        .context("Native runtime authority disappeared")?;
                    proof.libraries = seal_native_cargo_runtime_library_directories(
                        filesystem.as_ref(),
                        &workload.intent,
                        &proof.compiler,
                        actual,
                    )
                    .await?;
                    revalidate_native_compiler_facts(
                        filesystem.clone(),
                        &workload.intent,
                        &proof.compiler,
                        &executor,
                    )
                    .await?;
                    revalidate_native_cargo_runtime_library_directories(
                        filesystem.as_ref(),
                        &proof.libraries,
                    )
                    .await?;
                    Ok::<_, anyhow::Error>(workload)
                };
                match futures::future::select(
                    sealing.boxed(),
                    executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
                )
                .await
                {
                    Either::Left((result, _)) => result,
                    Either::Right(_) => Err(anyhow!("Native runtime library admission timed out")),
                }
            });
            let workload = work.await?;
            this.update(cx, |store, cx| {
                store.revalidate_native_cargo_workload(workload, cx)
            })??
            .await
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn finish_native_cargo_build(
        &mut self,
        intent: CargoNativeBuildIntent,
        report: crate::profile_results_store::ObservedBuildTaskReport,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoWorkload>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        self.validate_native_build_intent(&intent, cx)?;
        let expected = intent.expected_invocation()?;
        self.finish_native_cargo_build_with_invocation(intent, report, expected, None, cx)
    }

    #[cfg(target_os = "macos")]
    fn finish_native_cargo_build_with_invocation(
        &mut self,
        intent: CargoNativeBuildIntent,
        report: crate::profile_results_store::ObservedBuildTaskReport,
        expected: util::macos_task_completion::TaskBuildInvocation,
        runtime: Option<NativeCargoRuntimeWorkload>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoWorkload>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        self.validate_native_build_intent(&intent, cx)?;
        ensure!(
            uuid::Uuid::parse_str(report.run_id()).ok() == Some(intent.run_id),
            "Native Build report belongs to another producing Task"
        );
        ensure!(
            report
                .observed_invocation()?
                .is_some_and(|actual| expected.matches_observed(actual)),
            "Native Build report has no matching actual child invocation"
        );
        let project = intent
            .project
            .upgrade()
            .context("Native Build Project disappeared")?;
        let report_check = project.update(cx, |project, cx| {
            project.revalidate_local_cargo_build_report(&report, cx)
        })?;
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native Build filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let work = cx.background_spawn(async move {
            let sealing = async {
                report_check.await?;
                revalidate_native_build_filesystem(filesystem.as_ref(), &intent).await?;
                let decoded = crate::cargo_workspace::resolve_native_cargo_build_output(
                    report.bytes()?,
                    &intent.selection,
                    true,
                    intent.runtime_packages.as_ref(),
                )?;
                let path = decoded.executable;
                validate_native_build_run_executable(&path, &intent.invocation_output_root)?;
                ensure!(
                    filesystem
                        .canonicalize(&intent.invocation_output_root)
                        .await?
                        == intent.invocation_output_root,
                    "Native Build run output has aliases"
                );
                let root_metadata = filesystem
                    .metadata(&intent.invocation_output_root)
                    .await?
                    .context("Native Build run output disappeared")?;
                ensure!(
                    root_metadata.is_dir && !root_metadata.is_symlink && !root_metadata.is_fifo,
                    "Native Build run output is not ordinary"
                );
                ensure!(
                    filesystem.canonicalize(&path).await? == path,
                    "Native Build executable has aliases"
                );
                let relative = path
                    .strip_prefix(&intent.invocation_output_root)?
                    .to_path_buf();
                let metadata = filesystem
                    .metadata(&path)
                    .await?
                    .context("Native Build executable disappeared")?;
                ensure!(
                    metadata.is_executable && metadata.len > 0,
                    "Native Build selected output is not executable"
                );
                let fingerprint = fingerprint_native_probe_file(
                    filesystem.as_ref(),
                    &intent.invocation_output_root,
                    root_metadata.inode,
                    &relative,
                    metadata,
                    256 * 1024 * 1024,
                )
                .await?;
                let executable = NativeProbeFileSeal {
                    root: intent.invocation_output_root.clone(),
                    root_inode: root_metadata.inode,
                    relative,
                    metadata,
                    fingerprint,
                };
                revalidate_native_build_filesystem(filesystem.as_ref(), &intent).await?;
                Ok(NativeCargoWorkload {
                    intent,
                    report,
                    executable,
                    runtime_libraries: decoded.runtime_libraries,
                    runtime,
                })
            };
            match futures::future::select(
                sealing.boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right(_) => Err(anyhow!("Native Build executable sealing timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let workload = work.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&workload.intent, cx)
            })??;
            let project = workload
                .intent
                .project
                .upgrade()
                .context("Native Build Project disappeared")?;
            let check = project.update(cx, |project, cx| {
                project.revalidate_local_cargo_build_report(&workload.report, cx)
            })?;
            check.await?;
            workload.intent.revalidate_environment(cx)?.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&workload.intent, cx)
            })??;
            Ok(workload)
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn revalidate_native_cargo_workload(
        &mut self,
        workload: NativeCargoWorkload,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoWorkload>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        self.validate_native_build_intent(&workload.intent, cx)?;
        let project = workload
            .intent
            .project
            .upgrade()
            .context("Native Build Project disappeared")?;
        let checked = project.update(cx, |project, cx| {
            project.revalidate_local_cargo_build_report(&workload.report, cx)
        })?;
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native Build filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let work = cx.background_spawn(async move {
            let revalidation = async {
                checked.await?;
                revalidate_native_build_filesystem(filesystem.as_ref(), &workload.intent).await?;
                if let Some(proof) = &workload.runtime {
                    ensure!(
                        workload.report.runtime_admission_matches(&proof.lease)?,
                        "Native runtime producing admission expired"
                    );
                    let actual = workload
                        .report
                        .observed_cargo_runtime()?
                        .context("Native runtime observation expired")?;
                    ensure!(
                        actual.program() == workload.executable_path()
                            && actual.arguments() == proof.workload_arguments.as_slice()
                            && actual.working_directory() == workload.intent.cwd.as_path()
                            && actual
                                .environment()
                                .iter()
                                .all(|(name, value)| proof.environment.get(name) == Some(value))
                            && actual.environment().len() == proof.environment.len(),
                        "Native runtime observation no longer matches its selected workload"
                    );
                    revalidate_native_compiler_facts(
                        filesystem.clone(),
                        &workload.intent,
                        &proof.compiler,
                        &executor,
                    )
                    .await?;
                    revalidate_native_cargo_runtime_library_directories(
                        filesystem.as_ref(),
                        &proof.libraries,
                    )
                    .await?;
                }
                ensure!(
                    fingerprint_native_probe_file(
                        filesystem.as_ref(),
                        &workload.executable.root,
                        workload.executable.root_inode,
                        &workload.executable.relative,
                        workload.executable.metadata,
                        256 * 1024 * 1024
                    )
                    .await?
                        == workload.executable.fingerprint,
                    "Native Build executable bytes changed"
                );
                Ok(workload)
            };
            match futures::future::select(
                revalidation.boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right(_) => Err(anyhow!("Native Build workload revalidation timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let workload = work.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&workload.intent, cx)
            })??;
            let project = workload
                .intent
                .project
                .upgrade()
                .context("Native Build Project disappeared")?;
            let checked = project.update(cx, |project, cx| {
                project.revalidate_local_cargo_build_report(&workload.report, cx)
            })?;
            checked.await?;
            workload.intent.revalidate_environment(cx)?.await?;
            this.update(cx, |store, cx| {
                store.validate_native_build_intent(&workload.intent, cx)
            })??;
            Ok(workload)
        }))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn prepare_native_capture_plan(
        &mut self,
        workload: NativeCargoWorkload,
        declaration: task::TaskNativeProfileContinuation,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoCapturePlan>>> {
        let checked = self.revalidate_native_cargo_workload(workload, cx)?;
        Ok(cx.spawn(async move |this, cx| {
            let workload = checked.await?;
            if let Some(proof) = &workload.runtime {
                ensure!(
                    declaration.workload_arguments == proof.workload_arguments,
                    "Native capture literal arguments differ from the authenticated Cargo runtime"
                );
            }
            let project = workload
                .intent
                .project
                .upgrade()
                .context("Native capture Project disappeared")?;
            let logical_launch = task::SpawnInTerminal {
                cwd: Some(workload.execution_directory()?.to_path_buf()),
                env: workload.intent.task_environment.clone(),
                ..Default::default()
            };
            let prepared_environment = project.update(cx, |project, cx| {
                project.prepare_native_task_environment(&logical_launch, cx)
            })?;
            let recipe = prepared_environment.await?;
            ensure!(
                recipe.working_directory() == workload.execution_directory()?
                    && recipe.logical_environment() == &workload.intent.task_environment,
                "Native capture complete environment differs from the selected workload Task"
            );
            let probe = this
                .update(cx, |store, cx| {
                    store.probe_native_collector_with_environment(
                        workload.intent.reviewed.manifest_path.clone(),
                        PathBuf::from(&declaration.collector_path),
                        workload.execution_directory()?.to_path_buf(),
                        workload.intent.task_environment.clone(),
                        match &workload.runtime {
                            Some(proof) => NativeCollectorEnvironment::CargoRuntime {
                                project: project.downgrade(),
                                recipe,
                                environment: proof.environment.clone(),
                            },
                            None => NativeCollectorEnvironment::TerminalRecipe {
                                project: project.downgrade(),
                                recipe,
                            },
                        },
                        cx,
                    )
                })??
                .await?;
            let checked = this.update(cx, |store, cx| {
                store.revalidate_native_cargo_workload(workload, cx)
            })??;
            let workload = checked.await?;
            let task_template = crate::cargo_workspace::compile_native_profile_task(
                task::TaskTemplate {
                    required_experimental_features: workload
                        .intent
                        .required_experimental_features
                        .clone(),
                    label: "Selected Cargo workload".into(),
                    cwd: Some(
                        workload
                            .execution_directory()?
                            .to_str()
                            .context("Native capture cwd is not UTF-8")?
                            .to_owned(),
                    ),
                    env: workload.intent.task_environment.clone(),
                    save: task::SaveStrategy::None,
                    use_new_terminal: true,
                    ..Default::default()
                },
                "macos",
                &crate::cargo_workspace::NativeProfileCollector {
                    command: probe
                        .collector_path()
                        .to_str()
                        .context("Native collector path is not UTF-8")?
                        .to_owned(),
                    version: probe.version().into(),
                },
                crate::cargo_workspace::NativeProfileCommand {
                    command: workload
                        .executable_path()
                        .to_str()
                        .context("Native workload path is not UTF-8")?
                        .to_owned(),
                    arguments: declaration.workload_arguments,
                    environment: HashMap::default(),
                },
                &declaration.profile_path,
            )?;
            let mut resolved = task_template
                .resolve_task(
                    &crate::TaskSourceKind::UserInput.to_id_base(),
                    &task::TaskContext::default(),
                )
                .context("Native profile Task cannot resolve its literal recipe")?
                .with_artifact_directory(&workload.intent.cwd)?;
            ensure!(
                workload.task_id().0.len() <= 4096,
                "Native producing Task identity exceeds its bound"
            );
            let stable_id = task::TaskId(format!("{}:native-capture", workload.task_id().0));
            resolved.id = stable_id.clone();
            resolved.resolved.id = stable_id;
            Ok(NativeCargoCapturePlan {
                resolved,
                workload,
                probe,
            })
        }))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn revalidate_native_capture_plan(
        &mut self,
        plan: NativeCargoCapturePlan,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoCapturePlan>>> {
        let NativeCargoCapturePlan {
            resolved,
            workload,
            probe,
        } = plan;
        let checked = self.revalidate_native_cargo_workload(workload, cx)?;
        Ok(cx.spawn(async move |this, cx| {
            let workload = checked.await?;
            let checked = this.update(cx, |store, cx| {
                store.revalidate_native_collector_probe(
                    probe,
                    workload.execution_directory()?.to_path_buf(),
                    workload.intent.task_environment.clone(),
                    cx,
                )
            })??;
            let probe = checked.await?;
            let checked = this.update(cx, |store, cx| {
                store.revalidate_native_cargo_workload(workload, cx)
            })??;
            Ok(NativeCargoCapturePlan {
                resolved,
                workload: checked.await?,
                probe,
            })
        }))
    }

    pub fn probe_native_collector(
        &mut self,
        manifest: ProjectPath,
        requested_collector: PathBuf,
        cwd: PathBuf,
        task_environment: HashMap<String, String>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeCollectorProbe>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        self.probe_native_collector_with_environment(
            manifest,
            requested_collector,
            cwd,
            task_environment,
            NativeCollectorEnvironment::Legacy,
            cx,
        )
    }

    fn probe_native_collector_with_environment(
        &mut self,
        manifest: ProjectPath,
        requested_collector: PathBuf,
        cwd: PathBuf,
        task_environment: HashMap<String, String>,
        environment_source: NativeCollectorEnvironment,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeCollectorProbe>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        ensure!(
            cfg!(target_os = "macos"),
            "Native collector probing requires the authoritative macOS host"
        );
        ensure!(
            requested_collector.is_absolute()
                && cwd.is_absolute()
                && cwd.to_str().is_some_and(
                    |value| value.len() <= 4096 && !value.chars().any(char::is_control)
                )
                && requested_collector.to_str().is_some_and(
                    |value| value.len() <= 4096 && !value.chars().any(char::is_control)
                )
                && task_environment.len() <= 256
                && task_environment.iter().all(|(key, value)| !key.is_empty()
                    && !key.contains(['=', '\0'])
                    && key.len() <= 4096
                    && value.len() <= 4096
                    && !value.contains('\0')),
            "Native collector path or Task environment is invalid"
        );
        let reviewed = self.prepare_reviewed_metadata_import(manifest, cx)?;
        let source_entries = self.native_probe_source_entries(&reviewed, cx)?;
        ensure!(
            cwd.starts_with(&reviewed.root),
            "Native collector cwd is outside the admitted worktree"
        );
        let CargoWorkspaceStoreMode::Local {
            environment,
            shared: None,
            ..
        } = &self.mode
        else {
            return Err(anyhow!(
                "Native collector probe requires a local owned host; shared/remote probing remains unsupported"
            ));
        };
        let environment = match &environment_source {
            NativeCollectorEnvironment::Legacy => environment.update(cx, |environment, cx| {
                environment.directory_environment(Arc::from(cwd.as_path()), cx)
            }),
            #[cfg(target_os = "macos")]
            NativeCollectorEnvironment::TerminalRecipe { recipe, .. }
            | NativeCollectorEnvironment::CargoRuntime { recipe, .. } => {
                ensure!(
                    recipe.working_directory() == cwd
                        && recipe.logical_environment() == &task_environment,
                    "Native collector complete recipe differs from its logical Task"
                );
                Task::ready(Some(native_capture_environment(
                    environment_source
                        .complete_environment()
                        .context("Native collector complete environment disappeared")?,
                )?))
                .shared()
            }
        };
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native collector filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let owner = cx.entity_id();
        let (abort, registration) = AbortHandle::new_pair();
        let worktree_subscription = cx.subscribe(&self.worktree_store, {
            let abort = abort.clone();
            move |_, _, _, _| abort.abort()
        });
        let trust_subscription = TrustedWorktrees::try_get_global(cx)
            .map(|trust| cx.subscribe(&trust, move |_, _, _, _| abort.abort()));
        let checked_environment: Task<Result<()>> = match &environment_source {
            NativeCollectorEnvironment::Legacy => Task::ready(Ok(())),
            #[cfg(target_os = "macos")]
            NativeCollectorEnvironment::TerminalRecipe { project, recipe }
            | NativeCollectorEnvironment::CargoRuntime {
                project, recipe, ..
            } => {
                let project = project
                    .upgrade()
                    .context("Native collector environment Project disappeared")?;
                project.update(cx, |project, cx| {
                    project.revalidate_native_task_environment(recipe, cx)
                })?
            }
        };
        let task = cx.background_spawn(async move {
            let preparation = async {
                checked_environment.await?;
                ensure!(
                    filesystem.canonicalize(&cwd).await? == cwd,
                    "Native collector cwd must be a canonical host path"
                );
                let cwd_metadata = filesystem
                    .metadata(&cwd)
                    .await?
                    .context("Native collector cwd disappeared")?;
                ensure!(
                    cwd_metadata.is_dir && !cwd_metadata.is_symlink,
                    "Native collector cwd is not ordinary"
                );
                let mut resolved_environment = environment
                    .await
                    .context("Native collector host environment is unavailable")?;
                if matches!(&environment_source, NativeCollectorEnvironment::Legacy) {
                    resolved_environment.extend(task_environment.clone());
                }
                let probe_environment_fingerprint =
                    fingerprint_native_probe_environment(&resolved_environment)?;
                let (sources, directories) =
                    seal_native_probe_sources(filesystem.as_ref(), &reviewed, &source_entries)
                        .await?;
                let canonical = filesystem.canonicalize(&requested_collector).await?;
                let root = canonical
                    .parent()
                    .context("Native collector has no parent")?
                    .to_path_buf();
                let relative = PathBuf::from(
                    canonical
                        .file_name()
                        .context("Native collector has no filename")?,
                );
                let root_metadata = filesystem
                    .metadata(&root)
                    .await?
                    .context("Native collector directory disappeared")?;
                let metadata = filesystem
                    .metadata(&canonical)
                    .await?
                    .context("Native collector disappeared")?;
                ensure!(
                    root_metadata.is_dir && !root_metadata.is_symlink && metadata.is_executable,
                    "Native collector is not an executable on the admitted host"
                );
                let fingerprint = fingerprint_native_probe_file(
                    filesystem.as_ref(),
                    &root,
                    root_metadata.inode,
                    &relative,
                    metadata,
                    MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES,
                )
                .await?;
                run_native_collector_version(&canonical, &cwd, &resolved_environment, &executor)
                    .await?;
                ensure!(
                    filesystem.canonicalize(&requested_collector).await? == canonical,
                    "Native collector configured path changed during probing"
                );
                ensure!(
                    fingerprint_native_probe_file(
                        filesystem.as_ref(),
                        &root,
                        root_metadata.inode,
                        &relative,
                        metadata,
                        MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES
                    )
                    .await?
                        == fingerprint,
                    "Native collector bytes changed during probing"
                );
                revalidate_native_probe_filesystem(filesystem.as_ref(), &sources, &directories)
                    .await?;
                let final_cwd = filesystem
                    .metadata(&cwd)
                    .await?
                    .context("Native collector cwd disappeared")?;
                ensure!(
                    filesystem.canonicalize(&cwd).await? == cwd
                        && final_cwd.is_dir
                        && !final_cwd.is_symlink
                        && final_cwd.inode == cwd_metadata.inode,
                    "Native collector cwd identity changed"
                );
                Ok(CargoNativeCollectorProbe {
                    environment_source,
                    owner,
                    reviewed,
                    source_entries,
                    sources,
                    directories,
                    requested_collector,
                    collector: NativeProbeFileSeal {
                        root,
                        root_inode: root_metadata.inode,
                        relative,
                        metadata,
                        fingerprint,
                    },
                    cwd,
                    cwd_inode: cwd_metadata.inode,
                    environment: resolved_environment,
                    probe_environment_fingerprint,
                    task_environment,
                })
            };
            match futures::future::select(
                Abortable::new(preparation, registration).boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => {
                    result.map_err(|_| anyhow!("Native collector probe was invalidated"))?
                }
                Either::Right(_) => Err(anyhow!("Native collector admission timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let result = task.await;
            drop((worktree_subscription, trust_subscription));
            let probe = result?;
            probe.environment_source.revalidate(cx)?.await?;
            this.update(cx, |store, cx| {
                store.validate_native_collector_probe(&probe, cx)
            })??;
            Ok(probe)
        }))
    }

    pub fn revalidate_native_collector_probe(
        &mut self,
        probe: CargoNativeCollectorProbe,
        cwd: PathBuf,
        task_environment: HashMap<String, String>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<CargoNativeCollectorProbe>>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        self.validate_native_collector_probe(&probe, cx)?;
        ensure!(
            probe.cwd == cwd && probe.task_environment == task_environment,
            "Native collector Task context changed"
        );
        let CargoWorkspaceStoreMode::Local {
            environment,
            shared: None,
            ..
        } = &self.mode
        else {
            return Err(anyhow!("Native collector host route changed"));
        };
        let checked_environment: Task<Result<()>> = match &probe.environment_source {
            NativeCollectorEnvironment::Legacy => Task::ready(Ok(())),
            #[cfg(target_os = "macos")]
            NativeCollectorEnvironment::TerminalRecipe { project, recipe }
            | NativeCollectorEnvironment::CargoRuntime {
                project, recipe, ..
            } => {
                let project = project
                    .upgrade()
                    .context("Native collector environment Project disappeared")?;
                project.update(cx, |project, cx| {
                    project.revalidate_native_task_environment(recipe, cx)
                })?
            }
        };
        let environment = match &probe.environment_source {
            NativeCollectorEnvironment::Legacy => environment.update(cx, |environment, cx| {
                environment.directory_environment(Arc::from(cwd.as_path()), cx)
            }),
            #[cfg(target_os = "macos")]
            NativeCollectorEnvironment::TerminalRecipe { .. }
            | NativeCollectorEnvironment::CargoRuntime { .. } => {
                Task::ready(Some(native_capture_environment(
                    probe
                        .environment_source
                        .complete_environment()
                        .context("Native collector complete environment disappeared")?,
                )?))
                .shared()
            }
        };
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native collector filesystem is unavailable")?
            .clone();
        let executor = cx.background_executor().clone();
        let task = cx.background_spawn(async move {
            let revalidation = async {
                checked_environment.await?;
                let mut current_environment = environment
                    .await
                    .context("Native collector host environment is unavailable")?;
                if matches!(
                    &probe.environment_source,
                    NativeCollectorEnvironment::Legacy
                ) {
                    current_environment.extend(task_environment);
                }
                ensure!(
                    fingerprint_native_probe_environment(&current_environment)?
                        == probe.probe_environment_fingerprint
                        && current_environment == probe.environment
                        && filesystem.canonicalize(&cwd).await? == cwd,
                    "Native collector probe environment or cwd changed"
                );
                let current_cwd = filesystem
                    .metadata(&cwd)
                    .await?
                    .context("Native collector cwd disappeared")?;
                ensure!(
                    current_cwd.is_dir
                        && !current_cwd.is_symlink
                        && current_cwd.inode == probe.cwd_inode,
                    "Native collector cwd identity changed"
                );
                probe
                    .reviewed
                    .preflight_filesystem(filesystem.as_ref())
                    .await?;
                revalidate_native_probe_filesystem(
                    filesystem.as_ref(),
                    &probe.sources,
                    &probe.directories,
                )
                .await?;
                ensure!(
                    filesystem.canonicalize(&probe.requested_collector).await?
                        == probe.collector_path(),
                    "Native collector configured path changed"
                );
                ensure!(
                    fingerprint_native_probe_file(
                        filesystem.as_ref(),
                        &probe.collector.root,
                        probe.collector.root_inode,
                        &probe.collector.relative,
                        probe.collector.metadata,
                        MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES
                    )
                    .await?
                        == probe.collector.fingerprint,
                    "Native collector identity changed"
                );
                Ok(probe)
            };
            match futures::future::select(
                revalidation.boxed(),
                executor.timer(CARGO_METADATA_TIMEOUT).boxed(),
            )
            .await
            {
                Either::Left((result, _)) => result,
                Either::Right(_) => Err(anyhow!("Native collector revalidation timed out")),
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            let probe = task.await?;
            probe.environment_source.revalidate(cx)?.await?;
            this.update(cx, |store, cx| {
                store.validate_native_collector_probe(&probe, cx)
            })??;
            Ok(probe)
        }))
    }

    pub fn prepare_reviewed_metadata_import(
        &self,
        manifest_path: ProjectPath,
        cx: &App,
    ) -> Result<CargoReviewedMetadataImport> {
        ensure!(
            matches!(
                &self.mode,
                CargoWorkspaceStoreMode::Local { shared: None, .. }
            ) && !self.disconnected,
            "Reviewed Cargo import requires a local owned host"
        );
        let trusted = TrustedWorktrees::try_get_global(cx).and_then(|owner| {
            owner.read(cx).observed_worktree_trust(
                &self.worktree_store,
                manifest_path.worktree_id,
                cx,
            )
        });
        ensure!(
            trusted == Some(true),
            "Reviewed Cargo import requires current project trust"
        );
        let tree = self
            .worktree_store
            .read(cx)
            .worktree_for_id(manifest_path.worktree_id, cx)
            .context("Reviewed Cargo worktree is unavailable")?;
        let tree = tree.read(cx);
        ensure!(
            tree.is_visible() && tree.as_local().is_some(),
            "Cargo import is not a visible local worktree"
        );
        let snapshot = tree.snapshot();
        let manifest = snapshot
            .entry_for_path(&manifest_path.path)
            .context("Selected Cargo manifest is unavailable")?;
        ensure!(
            manifest_path.path.file_name() == Some("Cargo.toml")
                && manifest.is_file()
                && !manifest.is_private
                && !manifest.is_external
                && !manifest.is_ignored,
            "Selected Cargo manifest is not an ordinary visible workspace input"
        );
        let mut files = Vec::new();
        let mut bytes = 0_u64;
        for entry in snapshot.files(false, 0).filter(|entry| {
            is_cargo_workspace_input(&entry.path) && !entry.is_private && !entry.is_ignored
        }) {
            ensure!(
                !entry.is_external && !entry.is_fifo && entry.mtime.is_some(),
                "Cargo input cannot be read through the confined host filesystem"
            );
            bytes = bytes
                .checked_add(entry.size)
                .context("Cargo input budget overflow")?;
            ensure!(
                files.len() < 4096 && bytes <= 2 * 1024 * 1024,
                "Reviewed Cargo source inputs exceed the two-MiB/4096-file limit"
            );
            files.push((
                ProjectPath {
                    worktree_id: manifest_path.worktree_id,
                    path: entry.path.clone(),
                },
                entry.inode,
                entry.mtime,
                entry.size,
            ));
        }
        let root_inode = snapshot
            .root_entry()
            .context("Cargo root disappeared")?
            .inode;
        Ok(CargoReviewedMetadataImport {
            revision: self.revision,
            input_fingerprint: self.current_input_fingerprint(cx),
            settings_revision: cx
                .global::<settings::SettingsStore>()
                .observed_revision()
                .context("Cargo settings authority is unavailable")?,
            manifest_path,
            root: tree.abs_path().to_path_buf(),
            root_inode,
            source_snapshot: snapshot,
            files,
        })
    }

    pub fn revalidate_reviewed_metadata_import(
        &self,
        reviewed: &CargoReviewedMetadataImport,
        cx: &App,
    ) -> Result<()> {
        ensure!(
            reviewed.revision == self.revision,
            "Reviewed Cargo owner generation changed"
        );
        ensure!(
            reviewed.input_fingerprint == self.current_input_fingerprint(cx),
            "Reviewed Cargo input fingerprint changed"
        );
        ensure!(
            Some(reviewed.settings_revision)
                == cx.global::<settings::SettingsStore>().observed_revision(),
            "Reviewed Cargo settings authority changed"
        );
        let current = self.prepare_reviewed_metadata_import(reviewed.manifest_path.clone(), cx)?;
        ensure!(
            current.root == reviewed.root
                && current.root_inode == reviewed.root_inode
                && current.files == reviewed.files,
            "Cargo source identity changed"
        );
        Ok(())
    }

    pub fn admit_reviewed_metadata_import(
        &mut self,
        reviewed: CargoReviewedMetadataImport,
        bytes: Vec<u8>,
        cx: &mut Context<Self>,
    ) -> Task<Result<CargoWorkspaceModel>> {
        let prepared = self
            .revalidate_reviewed_metadata_import(&reviewed, cx)
            .and_then(|()| {
                ensure!(
                    bytes.len() <= crate::cargo_workspace::MAX_CARGO_SNAPSHOT_BYTES,
                    "Reviewed Cargo report exceeds the eight-MiB limit"
                );
                self.worktree_store
                    .read(cx)
                    .fs()
                    .context("Confined Cargo filesystem is unavailable")
            });
        let filesystem = match prepared {
            Ok(filesystem) => filesystem,
            Err(error) => return Task::ready(Err(error)),
        };
        let work = cx.background_spawn({
            let reviewed = reviewed.clone();
            async move { reviewed_cargo_workspace(&reviewed, &bytes, filesystem.as_ref()).await }
        });
        cx.spawn(async move |this, cx| {
            let (workspace, metadata_facts) = work.await?;
            this.update(cx, |store, cx| {
                store.revalidate_reviewed_metadata_import(&reviewed, cx)?;
                let previous_current = store.cached_snapshot(cx).is_none_or(|(_, current)| current);
                let retained_scope_current = previous_current
                    || store.cached_snapshot.as_ref().is_none_or(|previous| {
                        previous
                            .workspaces
                            .iter()
                            .all(|existing| existing.key == workspace.key)
                            && previous
                                .failures
                                .iter()
                                .all(|failure| failure.manifest_path == reviewed.manifest_path)
                    });
                let mut snapshot = CargoWorkspaceSnapshot {
                    revision: store.revision,
                    input_fingerprint: store.current_input_fingerprint(cx),
                    workspaces: Vec::new(),
                    failures: Vec::new(),
                    completeness: CargoSnapshotCompleteness::Partial,
                };
                if let Some(previous) = &store.cached_snapshot {
                    crate::cargo_workspace::validate_cargo_replacement_bounds(
                        previous,
                        &workspace,
                        &reviewed.manifest_path,
                    )?;
                    snapshot.workspaces.extend(
                        previous
                            .workspaces
                            .iter()
                            .filter(|existing| existing.key != workspace.key)
                            .cloned(),
                    );
                    snapshot.failures.extend(
                        previous
                            .failures
                            .iter()
                            .filter(|failure| failure.manifest_path != reviewed.manifest_path)
                            .cloned(),
                    );
                }
                snapshot.workspaces.push(workspace.clone());
                validate_cargo_snapshot_bounds(&snapshot)?;
                let mut retained_facts = store
                    .metadata_facts_for_current_snapshot(cx)
                    .unwrap_or_default()
                    .iter()
                    .filter(|facts| facts.key != workspace.key)
                    .cloned()
                    .collect::<Vec<_>>();
                retained_facts.push(metadata_facts);
                validate_metadata_facts_bounds(&retained_facts)?;
                store.cached_metadata_facts = if retained_scope_current {
                    retained_facts
                } else {
                    Vec::new()
                };
                store.cached_metadata_settings_revision = if store.cached_metadata_facts.is_empty()
                {
                    None
                } else {
                    Some(reviewed.settings_revision)
                };
                store.cached_snapshot = Some(snapshot);
                store.cached_generation = if retained_scope_current {
                    store.revision
                } else {
                    store.revision.wrapping_sub(1)
                };
                store.cached_scope = None;
                cx.notify();
                Ok(workspace)
            })?
        })
    }

    pub fn current_input_fingerprint(&self, cx: &App) -> u64 {
        cargo_input_fingerprint(&self.worktree_store, None, cx)
    }

    async fn handle_get_workspace(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetCargoWorkspace>,
        mut cx: AsyncApp,
    ) -> Result<proto::CargoWorkspaceResponse> {
        let request_id = envelope.payload.request_id;
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let request_key = (peer_id, request_id);
        let allowed_worktree_ids = envelope
            .payload
            .worktree_ids
            .iter()
            .copied()
            .map(settings::WorktreeId::from_proto)
            .collect();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let proceed = this.update(&mut cx, |store, _| {
            if store.cancelled_remote_requests.remove(&request_key) {
                false
            } else {
                store
                    .active_remote_requests
                    .insert(request_key, abort_handle);
                true
            }
        });
        if !proceed {
            return Err(anyhow!("Cargo workspace request was cancelled"));
        }
        let refresh = this.update(&mut cx, |store, cx| {
            store.refresh_scoped(Some(allowed_worktree_ids), cx)
        });
        let snapshot = Abortable::new(refresh, abort_registration)
            .await
            .map_err(|_| anyhow!("Cargo workspace request was cancelled"));
        let was_cancelled = this.update(&mut cx, |store, _| {
            store.active_remote_requests.remove(&request_key);
            store.cancelled_remote_requests.remove(&request_key)
        });
        if was_cancelled {
            return Err(anyhow!("Cargo workspace request was cancelled"));
        }
        let snapshot = snapshot?.map_err(|error| {
            if error
                .downcast_ref::<CargoWorkspaceRemoteError>()
                .is_some_and(|error| error.kind == CargoWorkspaceRemoteErrorKind::UnsupportedHost)
            {
                proto::ErrorCode::Internal
                    .with_tag("reason", "rust_tools_unsupported")
                    .message(error.to_string())
                    .into()
            } else {
                error
            }
        })?;
        Ok(snapshot_to_proto(snapshot, request_id))
    }

    async fn handle_cancel_workspace(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CancelCargoWorkspace>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        this.update(&mut cx, |store, _| {
            let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
            let request_key = (peer_id, envelope.payload.request_id);
            if let Some(handle) = store.active_remote_requests.remove(&request_key) {
                handle.abort();
            } else {
                store.cancelled_remote_requests.insert(request_key);
            }
        });
        Ok(proto::Ack {})
    }
}

fn collect_candidates(
    worktree_store: &Entity<WorktreeStore>,
    environment: &Entity<ProjectEnvironment>,
    allowed_worktree_ids: Option<&HashSet<settings::WorktreeId>>,
    cx: &mut App,
) -> (Vec<Candidate>, u64) {
    let mut candidates = Vec::new();
    let worktrees = worktree_store
        .read(cx)
        .visible_worktrees(cx)
        .collect::<Vec<_>>();
    for worktree in worktrees {
        let (worktree_id, worktree_root, is_single_file, snapshot) =
            worktree.read_with(cx, |tree, _| {
                (
                    tree.id(),
                    tree.abs_path().to_path_buf(),
                    tree.is_single_file(),
                    tree.snapshot(),
                )
            });
        if allowed_worktree_ids.is_some_and(|allowed| !allowed.contains(&worktree_id)) {
            continue;
        }
        if is_single_file {
            continue;
        }
        let trusted = TrustedWorktrees::try_get_global(cx)
            .map(|trusted| {
                trusted.update(cx, |trusted, cx| {
                    trusted.can_trust(worktree_store, worktree_id, cx)
                })
            })
            .unwrap_or(true);
        let worktree_environment = environment.update(cx, |environment, cx| {
            environment.worktree_environment(worktree.clone(), cx)
        });
        let private_paths: Arc<[Arc<RelPath>]> = snapshot
            .entries(false, 0)
            .filter(|entry| entry.is_private)
            .map(|entry| entry.path.clone())
            .collect::<Vec<_>>()
            .into();
        for entry in snapshot.files(false, 0) {
            if entry.path.file_name() != Some("Cargo.toml") || entry.is_private {
                continue;
            }
            let manifest_path = entry.path.clone();
            let manifest_text = load_visible_worktree_text(&worktree, manifest_path.clone(), cx);
            let toolchain_text =
                nearest_toolchain_path(&snapshot, manifest_path.as_ref()).map(|path| {
                    let project_path = ProjectPath {
                        worktree_id,
                        path: path.clone(),
                    };
                    let text = load_visible_worktree_text(&worktree, path, cx);
                    (project_path, text)
                });
            let lock_text =
                nearest_cargo_lock_path(&snapshot, manifest_path.as_ref()).map(|path| {
                    let project_path = ProjectPath {
                        worktree_id,
                        path: path.clone(),
                    };
                    let text = load_visible_worktree_text(&worktree, path, cx);
                    (project_path, text)
                });
            candidates.push(Candidate {
                project_path: ProjectPath {
                    worktree_id,
                    path: manifest_path.clone(),
                },
                absolute_path: worktree_root.join(manifest_path.as_unix_str()),
                worktree_root: worktree_root.clone(),
                environment: worktree_environment.clone(),
                manifest_text,
                lock_text,
                toolchain_text,
                trusted,
                private_paths: private_paths.clone(),
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.project_path
            .worktree_id
            .cmp(&right.project_path.worktree_id)
            .then_with(|| {
                left.project_path
                    .path
                    .components()
                    .count()
                    .cmp(&right.project_path.path.components().count())
            })
            .then_with(|| left.project_path.path.cmp(&right.project_path.path))
    });
    let input_fingerprint = cargo_input_fingerprint(worktree_store, allowed_worktree_ids, cx);
    (candidates, input_fingerprint)
}

fn nearest_cargo_lock_path(
    snapshot: &worktree::Snapshot,
    manifest_path: &RelPath,
) -> Option<Arc<RelPath>> {
    let manifest_directory = manifest_path.parent().unwrap_or(RelPath::empty());
    let lock_name = RelPath::from_unix_str("Cargo.lock").ok()?;
    for directory in manifest_directory.ancestors() {
        let path = directory.join(lock_name);
        if snapshot
            .entry_for_path(path.as_ref())
            .is_some_and(|entry| !entry.is_private && entry.is_file())
        {
            return Some(path.into());
        }
    }
    None
}

fn load_visible_worktree_text(
    worktree: &Entity<worktree::Worktree>,
    path: Arc<RelPath>,
    cx: &mut App,
) -> Task<Result<String>> {
    worktree.update(cx, |worktree, cx| {
        let load = worktree.load_file(path.as_ref(), cx);
        cx.spawn(async move |_worktree, _cx| load.await.map(|loaded| loaded.text))
    })
}

fn nearest_toolchain_path(
    snapshot: &worktree::Snapshot,
    manifest_path: &RelPath,
) -> Option<Arc<RelPath>> {
    let manifest_directory = match manifest_path.parent() {
        Some(parent) => parent,
        None => RelPath::empty(),
    };
    let toml_name = RelPath::from_unix_str("rust-toolchain.toml").ok()?;
    let legacy_name = RelPath::from_unix_str("rust-toolchain").ok()?;
    for directory in manifest_directory.ancestors() {
        for name in [toml_name, legacy_name] {
            let path = directory.join(name);
            if snapshot
                .entry_for_path(path.as_ref())
                .is_some_and(|entry| !entry.is_private && entry.is_file())
            {
                return Some(path.into());
            }
        }
    }
    None
}

fn cargo_input_fingerprint(
    worktree_store: &Entity<WorktreeStore>,
    allowed_worktree_ids: Option<&HashSet<settings::WorktreeId>>,
    cx: &App,
) -> u64 {
    let mut inputs = Vec::new();
    for worktree in worktree_store.read(cx).visible_worktrees(cx) {
        let (worktree_id, snapshot) =
            worktree.read_with(cx, |worktree, _| (worktree.id(), worktree.snapshot()));
        if allowed_worktree_ids.is_some_and(|allowed| !allowed.contains(&worktree_id)) {
            continue;
        }
        for entry in snapshot.files(false, 0) {
            if !entry.is_private && is_cargo_workspace_input(entry.path.as_ref()) {
                inputs.push((
                    worktree_id.to_proto(),
                    entry.path.as_unix_str().to_string(),
                    entry.id.to_proto(),
                    entry.mtime,
                    entry.size,
                ));
            }
        }
    }
    inputs.sort_by(|left, right| (&left.0, &left.1).cmp(&(&right.0, &right.1)));
    let mut hasher = DefaultHasher::new();
    inputs.hash(&mut hasher);
    hasher.finish()
}

fn is_cargo_workspace_input(path: &RelPath) -> bool {
    match path.file_name() {
        Some("Cargo.toml" | "Cargo.lock" | "rust-toolchain.toml" | "rust-toolchain") => true,
        Some("config.toml" | "config") => path
            .parent()
            .is_some_and(|parent| parent.file_name() == Some(".cargo")),
        _ => false,
    }
}

async fn reviewed_cargo_workspace(
    reviewed: &CargoReviewedMetadataImport,
    bytes: &[u8],
    filesystem: &dyn fs::Fs,
) -> Result<(CargoWorkspaceModel, CargoWorkspaceMetadataFacts)> {
    use std::io::Read as _;
    let metadata = parse_metadata(bytes)
        .map_err(|_| anyhow!("Reviewed Cargo report format/schema is invalid or unsupported"))?;
    let selected_root = reviewed
        .manifest_path
        .path
        .parent()
        .unwrap_or(RelPath::empty());
    ensure!(
        metadata.workspace_root.as_std_path() == reviewed.root.join(selected_root.as_std_path()),
        "Cargo report is for another workspace root"
    );
    let mut manifests = std::collections::BTreeMap::new();
    let mut inputs = std::collections::BTreeMap::new();
    for (path, inode, mtime, size) in &reviewed.files {
        let absolute = reviewed.root.join(path.path.as_std_path());
        let before = filesystem
            .metadata(&absolute)
            .await?
            .context("Reviewed Cargo input disappeared")?;
        ensure!(
            before.inode == *inode
                && Some(before.mtime) == *mtime
                && before.len == *size
                && !before.is_symlink
                && !before.is_dir
                && !before.is_fifo,
            "Reviewed Cargo input changed"
        );
        let mut reader = filesystem
            .open_sync_confined_file(
                &reviewed.root,
                path.path.as_std_path(),
                reviewed.root_inode,
                *inode,
            )
            .await?;
        let mut contents = Vec::new();
        reader
            .by_ref()
            .take(size.saturating_add(1))
            .read_to_end(&mut contents)?;
        ensure!(
            contents.len() as u64 == *size,
            "Cargo source input size changed during collection"
        );
        let after = filesystem
            .metadata(&absolute)
            .await?
            .context("Cargo input disappeared during read")?;
        ensure!(
            before.inode == after.inode
                && before.mtime == after.mtime
                && before.len == after.len
                && !after.is_symlink,
            "Cargo input changed during collection"
        );
        let contents = String::from_utf8(contents)?;
        if path.path.file_name() == Some("Cargo.toml") {
            manifests.insert(path.clone(), contents.clone());
        }
        inputs.insert(path.path.clone(), contents);
    }
    let mut workspace = workspace_from_metadata(&metadata, |absolute| {
        let relative = absolute.strip_prefix(&reviewed.root).ok()?;
        let relative = RelPath::new(relative, PathStyle::local()).ok()?;
        let entry = reviewed.source_snapshot.entry_for_path(relative.as_ref())?;
        (!entry.is_private && !entry.is_external && !entry.is_ignored).then(|| ProjectPath {
            worktree_id: reviewed.manifest_path.worktree_id,
            path: relative.into_owned().into(),
        })
    })?;
    ensure!(
        workspace.root_manifest.as_ref() == Some(&reviewed.manifest_path),
        "Selected Cargo root is absent from the report"
    );
    let root_manifest = toml::from_str::<toml::Value>(
        manifests
            .get(&reviewed.manifest_path)
            .context("Reviewed root declaration is unavailable")?,
    )
    .map_err(|_| anyhow!("Reviewed root Cargo declaration is malformed"))?;
    for member in &workspace.members {
        let declared = toml::from_str::<toml::Value>(
            manifests
                .get(&member.manifest_path)
                .context("Reviewed member declaration is unavailable")?,
        )
        .map_err(|_| anyhow!("Reviewed member Cargo declaration is malformed"))?;
        ensure!(
            declared
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                == Some(member.name.as_str()),
            "Cargo report package name differs from current declaration"
        );
        let version = declared
            .get("package")
            .and_then(|package| package.get("version"));
        let inherited = version
            .and_then(|version| version.get("workspace"))
            .and_then(toml::Value::as_bool)
            == Some(true);
        let version = if inherited {
            root_manifest
                .get("workspace")
                .and_then(|workspace| workspace.get("package"))
                .and_then(|package| package.get("version"))
                .and_then(toml::Value::as_str)
                .context("Inherited Cargo package version is unavailable")?
        } else {
            version.and_then(toml::Value::as_str).unwrap_or("0.0.0")
        };
        ensure!(
            version == member.version,
            "Cargo report package version differs from current declaration"
        );
    }
    let lock_path = selected_root.join(RelPath::from_unix_str("Cargo.lock")?);
    enrich_dependency_provenance(
        &mut workspace,
        &manifests,
        inputs
            .get::<RelPath>(lock_path.as_ref())
            .map(String::as_str),
    );
    if let Some(manifest) = manifests.get(&reviewed.manifest_path) {
        match parse_cargo_profiles(manifest) {
            Ok(profiles) => workspace.configuration.profiles = profiles,
            Err(_) => workspace.configuration.add_diagnostic(
                Some(reviewed.manifest_path.clone()),
                CargoConfigurationDiagnosticCategory::Manifest,
                "Reviewed manifest profiles are invalid",
            ),
        }
    }
    if let Some(path) =
        nearest_toolchain_path(&reviewed.source_snapshot, &reviewed.manifest_path.path)
    {
        if let Some(contents) = inputs.get(&path) {
            let project_path = ProjectPath {
                worktree_id: reviewed.manifest_path.worktree_id,
                path,
            };
            match parse_rust_toolchain(project_path.clone(), contents) {
                Ok(toolchain) => workspace.configuration.declared_toolchain = Some(toolchain),
                Err(_) => workspace.configuration.add_diagnostic(
                    Some(project_path),
                    CargoConfigurationDiagnosticCategory::Toolchain,
                    "Reviewed declared toolchain is invalid",
                ),
            }
        }
    }
    workspace.configuration.add_diagnostic(None, CargoConfigurationDiagnosticCategory::CompilerProbe,
        "This reviewed metadata report does not run rustc or establish the compiler, environment, external Cargo configuration or target defaults");
    crate::cargo_workspace::validate_cargo_workspace_bounds(&workspace)?;
    let metadata_facts = metadata_facts_from_metadata(
        &metadata,
        &workspace,
        |absolute| {
            let relative = absolute.strip_prefix(&reviewed.root).ok()?;
            let relative = RelPath::new(relative, PathStyle::local()).ok()?;
            let entry = reviewed.source_snapshot.entry_for_path(relative.as_ref())?;
            (!entry.is_private && !entry.is_external && !entry.is_ignored).then(|| ProjectPath {
                worktree_id: reviewed.manifest_path.worktree_id,
                path: relative.into_owned().into(),
            })
        },
        &manifests,
    )?;
    Ok((workspace, metadata_facts))
}

async fn collect_snapshot(
    revision: u64,
    input_fingerprint: u64,
    candidates: Vec<Candidate>,
    runner: Arc<dyn CargoMetadataRunner>,
    configuration_probe: Arc<dyn CargoConfigurationProbe>,
) -> Result<CargoMetadataCollection> {
    let mut loaded_candidates = Vec::with_capacity(candidates.len());
    let mut manifest_contents = std::collections::BTreeMap::new();
    for candidate in candidates {
        let Candidate {
            project_path,
            absolute_path,
            worktree_root,
            environment,
            manifest_text,
            lock_text,
            toolchain_text,
            trusted,
            private_paths,
        } = candidate;
        let manifest_text = manifest_text.await;
        if let Ok(contents) = &manifest_text {
            manifest_contents.insert(project_path.clone(), contents.clone());
        }
        let lock_text = match lock_text {
            Some((path, text)) => Some((path, text.await)),
            None => None,
        };
        loaded_candidates.push((
            project_path,
            absolute_path,
            worktree_root,
            environment,
            manifest_text,
            lock_text,
            toolchain_text,
            trusted,
            private_paths,
        ));
    }
    let roots = loaded_candidates
        .iter()
        .map(|candidate| {
            (
                candidate.0.worktree_id,
                candidate.2.clone(),
                candidate.8.clone(),
            )
        })
        .collect::<Vec<_>>();
    let mut covered = HashSet::new();
    let mut workspaces = Vec::new();
    let mut metadata_facts = Vec::new();
    let mut failures = Vec::new();
    for candidate in loaded_candidates {
        let (
            project_path,
            absolute_path,
            worktree_root,
            environment,
            manifest_text,
            lock_text,
            toolchain_text,
            trusted,
            _,
        ) = candidate;
        if covered.contains(&absolute_path) {
            continue;
        }
        if !trusted {
            failures.push(failure(
                project_path,
                CargoWorkspaceErrorCategory::Restricted,
                "Cargo metadata is disabled for this restricted worktree. Trust the worktree and refresh.",
            ));
            continue;
        }
        let environment = environment.await;
        let working_directory = absolute_path
            .parent()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| worktree_root.clone());
        let request = CargoMetadataRequest {
            working_directory: working_directory.clone(),
            manifest_path: absolute_path.clone(),
            environment: environment.clone(),
        };
        match runner.run(request).await {
            Ok(output) => match parse_metadata(&output) {
                Ok(metadata) => {
                    for package in &metadata.packages {
                        if metadata.workspace_members.contains(&package.id) {
                            covered.insert(package.manifest_path.as_std_path().to_path_buf());
                        }
                    }
                    covered.insert(
                        metadata
                            .workspace_root
                            .join("Cargo.toml")
                            .into_std_path_buf(),
                    );
                    match workspace_from_metadata(&metadata, |path| {
                        project_path_for_absolute_path(path, &roots)
                    }) {
                        Ok(mut workspace) => {
                            enrich_dependency_provenance(
                                &mut workspace,
                                &manifest_contents,
                                lock_text
                                    .as_ref()
                                    .and_then(|(_, contents)| contents.as_ref().ok())
                                    .map(String::as_str),
                            );
                            enrich_workspace_configuration(
                                &mut workspace,
                                project_path.clone(),
                                manifest_text,
                                toolchain_text,
                                working_directory,
                                environment,
                                &absolute_path,
                                &worktree_root,
                                configuration_probe.as_ref(),
                            )
                            .await;
                            match metadata_facts_from_metadata(
                                &metadata,
                                &workspace,
                                |path| project_path_for_absolute_path(path, &roots),
                                &manifest_contents,
                            ) {
                                Ok(facts) => {
                                    metadata_facts.push(facts);
                                    workspaces.push(workspace);
                                }
                                Err(error) => failures.push(failure(
                                    project_path,
                                    CargoWorkspaceErrorCategory::InvalidMetadata,
                                    &error.to_string(),
                                )),
                            }
                        }
                        Err(error) => failures.push(failure(
                            project_path,
                            CargoWorkspaceErrorCategory::InvalidMetadata,
                            &error.to_string(),
                        )),
                    }
                }
                Err(error) => failures.push(failure(
                    project_path,
                    if error.to_string().contains("unsupported") {
                        CargoWorkspaceErrorCategory::UnsupportedMetadata
                    } else {
                        CargoWorkspaceErrorCategory::InvalidMetadata
                    },
                    &error.to_string(),
                )),
            },
            Err(error) => failures.push(failure(
                project_path.clone(),
                if error.to_string().contains("failed to start Cargo") {
                    CargoWorkspaceErrorCategory::CargoNotFound
                } else {
                    CargoWorkspaceErrorCategory::CargoFailed
                },
                &sanitize_candidate_error(
                    &error.to_string(),
                    &absolute_path,
                    &worktree_root,
                    &project_path,
                ),
            )),
        }
    }
    let completeness = if failures.is_empty()
        && workspaces.iter().all(|workspace| {
            workspace.configuration.completeness == CargoConfigurationCompleteness::Complete
        }) {
        CargoSnapshotCompleteness::Complete
    } else {
        CargoSnapshotCompleteness::Partial
    };
    let snapshot = CargoWorkspaceSnapshot {
        revision,
        input_fingerprint,
        workspaces: deduplicate_workspaces(workspaces),
        failures,
        completeness,
    };
    validate_cargo_snapshot_bounds(&snapshot)?;
    let mut retained_keys = HashSet::new();
    metadata_facts.retain(|facts| {
        retained_keys.insert(facts.key.clone())
            && snapshot
                .workspaces
                .iter()
                .any(|workspace| facts.matches_workspace(workspace))
    });
    validate_metadata_facts_bounds(&metadata_facts)?;
    Ok(CargoMetadataCollection {
        snapshot,
        metadata_facts,
    })
}

#[allow(clippy::too_many_arguments)]
async fn enrich_workspace_configuration(
    workspace: &mut CargoWorkspaceModel,
    manifest_path: ProjectPath,
    manifest_text: Result<String>,
    toolchain_text: Option<(ProjectPath, Task<Result<String>>)>,
    working_directory: PathBuf,
    environment: Option<HashMap<String, String>>,
    absolute_manifest_path: &std::path::Path,
    worktree_root: &std::path::Path,
    configuration_probe: &dyn CargoConfigurationProbe,
) {
    match manifest_text {
        Ok(manifest_text) => match parse_cargo_profiles(&manifest_text) {
            Ok(profiles) => workspace.configuration.profiles = profiles,
            Err(error) => workspace.configuration.add_diagnostic(
                Some(manifest_path.clone()),
                CargoConfigurationDiagnosticCategory::Manifest,
                sanitize_candidate_error(
                    &error.to_string(),
                    absolute_manifest_path,
                    worktree_root,
                    &manifest_path,
                ),
            ),
        },
        Err(error) => workspace.configuration.add_diagnostic(
            Some(manifest_path.clone()),
            CargoConfigurationDiagnosticCategory::Manifest,
            sanitize_candidate_error(
                &error.to_string(),
                absolute_manifest_path,
                worktree_root,
                &manifest_path,
            ),
        ),
    }

    if let Some((toolchain_path, toolchain_text)) = toolchain_text {
        match toolchain_text.await {
            Ok(contents) => match parse_rust_toolchain(toolchain_path.clone(), &contents) {
                Ok(toolchain) => workspace.configuration.declared_toolchain = Some(toolchain),
                Err(error) => workspace.configuration.add_diagnostic(
                    Some(toolchain_path),
                    CargoConfigurationDiagnosticCategory::Toolchain,
                    error.to_string(),
                ),
            },
            Err(error) => workspace.configuration.add_diagnostic(
                Some(toolchain_path),
                CargoConfigurationDiagnosticCategory::Toolchain,
                sanitize_candidate_error(
                    &error.to_string(),
                    absolute_manifest_path,
                    worktree_root,
                    &manifest_path,
                ),
            ),
        }
    }

    match configuration_probe
        .run(CargoConfigurationProbeRequest {
            working_directory,
            environment,
        })
        .await
    {
        Ok(output) => match parse_rustc_verbose_version(&output) {
            Ok(host_compiler) => workspace.configuration.host_compiler = host_compiler,
            Err(error) => {
                workspace.configuration.host_compiler = CargoHostCompilerModel {
                    status: CargoHostCompilerStatus::Failed,
                    release: None,
                    host_target: None,
                    stale: false,
                };
                workspace.configuration.add_diagnostic(
                    None,
                    CargoConfigurationDiagnosticCategory::CompilerProbe,
                    error.to_string(),
                );
            }
        },
        Err(error) => {
            workspace.configuration.host_compiler = CargoHostCompilerModel {
                status: if error.to_string().contains("failed to start rustc") {
                    CargoHostCompilerStatus::Missing
                } else {
                    CargoHostCompilerStatus::Failed
                },
                release: None,
                host_target: None,
                stale: false,
            };
            workspace.configuration.add_diagnostic(
                None,
                CargoConfigurationDiagnosticCategory::CompilerProbe,
                sanitize_candidate_error(
                    &error.to_string(),
                    absolute_manifest_path,
                    worktree_root,
                    &manifest_path,
                ),
            );
        }
    }
}

fn project_path_for_absolute_path(
    path: &std::path::Path,
    roots: &[(settings::WorktreeId, PathBuf, Arc<[Arc<RelPath>]>)],
) -> Option<ProjectPath> {
    roots.iter().find_map(|(worktree_id, root, private_paths)| {
        let relative = path.strip_prefix(root).ok()?;
        let relative = RelPath::new(relative, PathStyle::local()).ok()?;
        if private_paths
            .iter()
            .any(|private_path| relative.starts_with(private_path))
        {
            return None;
        }
        Some(ProjectPath {
            worktree_id: *worktree_id,
            path: relative.into_owned().into(),
        })
    })
}

fn failure(
    manifest_path: ProjectPath,
    category: CargoWorkspaceErrorCategory,
    message: &str,
) -> CargoCandidateFailure {
    CargoCandidateFailure {
        manifest_path,
        category,
        message: bounded_error(message.as_bytes()),
        has_stale_model: false,
    }
}

fn bounded_error(bytes: &[u8]) -> String {
    let start = bytes.len().saturating_sub(MAX_ERROR_BYTES);
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

fn bounded_configuration_value(value: &str) -> String {
    let mut boundary = value
        .len()
        .min(crate::cargo_workspace::MAX_CARGO_CONFIGURATION_FIELD_BYTES);
    while !value.is_char_boundary(boundary) {
        boundary = boundary.saturating_sub(1);
    }
    value[..boundary].to_string()
}

fn sanitize_candidate_error(
    message: &str,
    manifest_path: &std::path::Path,
    worktree_root: &std::path::Path,
    project_path: &ProjectPath,
) -> String {
    let relative = project_path.path.as_unix_str();
    message
        .replace(&manifest_path.to_string_lossy().to_string(), relative)
        .replace(&worktree_root.to_string_lossy().to_string(), "<worktree>")
}

fn snapshot_to_proto(
    snapshot: CargoWorkspaceSnapshot,
    request_id: u64,
) -> proto::CargoWorkspaceResponse {
    proto::CargoWorkspaceResponse {
        request_id,
        revision: snapshot.revision,
        input_fingerprint: snapshot.input_fingerprint,
        workspaces: snapshot
            .workspaces
            .into_iter()
            .map(workspace_to_proto)
            .collect(),
        failures: snapshot
            .failures
            .into_iter()
            .map(|failure| proto::CargoCandidateFailure {
                manifest_path: Some(failure.manifest_path.to_proto()),
                category: error_category_to_proto(failure.category),
                message: bounded_error(failure.message.as_bytes()),
                has_stale_model: failure.has_stale_model,
            })
            .collect(),
        partial: snapshot.completeness == CargoSnapshotCompleteness::Partial,
    }
}

fn workspace_to_proto(workspace: CargoWorkspaceModel) -> proto::CargoWorkspace {
    proto::CargoWorkspace {
        root: Some(workspace.key.root.to_proto()),
        root_manifest: workspace.root_manifest.map(|path| path.to_proto()),
        display_name: workspace.display_name,
        is_virtual: workspace.is_virtual,
        members: workspace
            .members
            .into_iter()
            .map(|package| proto::CargoPackage {
                name: package.name,
                version: package.version,
                manifest_path: Some(package.manifest_path.to_proto()),
                is_default_member: package.is_default_member,
                targets: package.targets.into_iter().map(target_to_proto).collect(),
                features: package
                    .features
                    .into_iter()
                    .map(|feature| proto::CargoFeature {
                        name: feature.name,
                        defined: feature.defined,
                        enabled: match feature.enabled {
                            crate::cargo_workspace::CargoFeatureEnabled::Unknown => 0,
                            crate::cargo_workspace::CargoFeatureEnabled::Enabled => 1,
                            crate::cargo_workspace::CargoFeatureEnabled::Disabled => 2,
                        },
                        expands: feature.expands,
                    })
                    .collect(),
                dependencies: package
                    .dependencies
                    .into_iter()
                    .map(dependency_to_proto)
                    .collect(),
            })
            .collect(),
        configuration: Some(configuration_to_proto(workspace.configuration)),
        dependency_graph: workspace.dependency_graph.map(graph_to_proto),
    }
}

fn source_kind_to_proto(source: crate::cargo_workspace::CargoDependencySourceKind) -> i32 {
    use crate::cargo_workspace::CargoDependencySourceKind;
    match source {
        CargoDependencySourceKind::Other => 0,
        CargoDependencySourceKind::Path => 1,
        CargoDependencySourceKind::Registry => 2,
        CargoDependencySourceKind::Git => 3,
    }
}

fn source_kind_from_proto(source: i32) -> crate::cargo_workspace::CargoDependencySourceKind {
    use crate::cargo_workspace::CargoDependencySourceKind;
    match source {
        1 => CargoDependencySourceKind::Path,
        2 => CargoDependencySourceKind::Registry,
        3 => CargoDependencySourceKind::Git,
        _ => CargoDependencySourceKind::Other,
    }
}

fn graph_to_proto(
    graph: crate::cargo_workspace::CargoDependencyGraph,
) -> proto::CargoDependencyGraph {
    proto::CargoDependencyGraph {
        format_version: graph.format_version,
        metadata_identity: graph.metadata_identity,
        lock_identity: graph.lock_identity,
        roots: graph.roots,
        partial: graph.completeness == CargoSnapshotCompleteness::Partial,
        nodes: graph
            .nodes
            .into_iter()
            .map(|node| proto::CargoDependencyGraphNode {
                identity: node.identity,
                name: node.name,
                version: node.version,
                source_kind: source_kind_to_proto(node.source_kind),
                source_identity: node.source_identity,
                crates_io: node.crates_io,
                workspace_manifest: node.workspace_manifest.map(|path| path.to_proto()),
                enabled_features: node.enabled_features,
                lock_status: match node.lock_status {
                    crate::cargo_workspace::CargoDependencyLockStatus::Unknown => 0,
                    crate::cargo_workspace::CargoDependencyLockStatus::Locked => 1,
                    crate::cargo_workspace::CargoDependencyLockStatus::NotLocked => 2,
                    crate::cargo_workspace::CargoDependencyLockStatus::MissingLockfile => 3,
                },
            })
            .collect(),
        edges: graph
            .edges
            .into_iter()
            .map(|edge| proto::CargoDependencyGraphEdge {
                parent: edge.parent,
                child: edge.child,
                alias: edge.alias,
                kind: match edge.kind {
                    crate::cargo_workspace::CargoDependencyKind::Unknown => 0,
                    crate::cargo_workspace::CargoDependencyKind::Normal => 1,
                    crate::cargo_workspace::CargoDependencyKind::Development => 2,
                    crate::cargo_workspace::CargoDependencyKind::Build => 3,
                },
                target: edge.target,
                version_requirement: edge.version_requirement,
                workspace_inherited: edge.workspace_inherited,
            })
            .collect(),
    }
}

fn graph_from_proto(
    graph: proto::CargoDependencyGraph,
) -> Result<crate::cargo_workspace::CargoDependencyGraph> {
    use crate::cargo_workspace::*;
    ensure!(
        graph.nodes.len() <= MAX_CARGO_GRAPH_NODES
            && graph.edges.len() <= MAX_CARGO_GRAPH_EDGES
            && graph.roots.len() <= MAX_CARGO_GRAPH_NODES,
        "Remote Cargo graph count limit exceeded"
    );
    let graph = CargoDependencyGraph {
        format_version: graph.format_version,
        metadata_identity: graph.metadata_identity,
        lock_identity: graph.lock_identity,
        roots: graph.roots,
        completeness: if graph.partial {
            CargoSnapshotCompleteness::Partial
        } else {
            CargoSnapshotCompleteness::Complete
        },
        nodes: graph
            .nodes
            .into_iter()
            .map(|node| {
                ensure!(
                    node.enabled_features.len() <= MAX_CARGO_DEPENDENCY_FEATURES,
                    "Remote Cargo graph feature limit exceeded"
                );
                Ok(CargoDependencyGraphNode {
                    identity: node.identity,
                    name: node.name,
                    version: node.version,
                    source_kind: source_kind_from_proto(node.source_kind),
                    source_identity: node.source_identity,
                    crates_io: node.crates_io,
                    workspace_manifest: node
                        .workspace_manifest
                        .map(|path| {
                            ProjectPath::from_proto(path)
                                .context("Invalid Cargo graph workspace path")
                        })
                        .transpose()?,
                    enabled_features: node.enabled_features,
                    lock_status: match node.lock_status {
                        1 => CargoDependencyLockStatus::Locked,
                        2 => CargoDependencyLockStatus::NotLocked,
                        3 => CargoDependencyLockStatus::MissingLockfile,
                        _ => CargoDependencyLockStatus::Unknown,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?,
        edges: graph
            .edges
            .into_iter()
            .map(|edge| CargoDependencyGraphEdge {
                parent: edge.parent,
                child: edge.child,
                alias: edge.alias,
                kind: match edge.kind {
                    1 => CargoDependencyKind::Normal,
                    2 => CargoDependencyKind::Development,
                    3 => CargoDependencyKind::Build,
                    _ => CargoDependencyKind::Unknown,
                },
                target: edge.target,
                version_requirement: edge.version_requirement,
                workspace_inherited: edge.workspace_inherited,
            })
            .collect(),
    };
    graph.validate()?;
    Ok(graph)
}

fn configuration_to_proto(
    configuration: crate::cargo_workspace::CargoWorkspaceConfiguration,
) -> proto::CargoWorkspaceConfiguration {
    use crate::cargo_workspace::{
        CargoConfigurationDiagnosticCategory, CargoHostCompilerStatus, CargoProfileOrigin,
        CargoTargetConfiguration, CargoToolchainFormat, MAX_CARGO_CONFIGURATION_DIAGNOSTICS,
        MAX_CARGO_CONFIGURATION_ITEMS,
    };
    proto::CargoWorkspaceConfiguration {
        profiles: configuration
            .profiles
            .into_iter()
            .take(MAX_CARGO_CONFIGURATION_ITEMS)
            .map(|profile| proto::CargoProfile {
                name: bounded_configuration_value(&profile.name),
                origin: match profile.origin {
                    CargoProfileOrigin::Implicit => 0,
                    CargoProfileOrigin::Declared => 1,
                },
            })
            .collect(),
        declared_toolchain: configuration.declared_toolchain.map(|toolchain| {
            proto::CargoToolchain {
                source_path: Some(toolchain.source_path.to_proto()),
                format: match toolchain.format {
                    CargoToolchainFormat::Toml => 0,
                    CargoToolchainFormat::Legacy => 1,
                },
                channel: toolchain
                    .channel
                    .map(|channel| bounded_configuration_value(&channel)),
                components: toolchain
                    .components
                    .into_iter()
                    .take(MAX_CARGO_CONFIGURATION_ITEMS)
                    .map(|component| bounded_configuration_value(&component))
                    .collect(),
                targets: toolchain
                    .targets
                    .into_iter()
                    .take(MAX_CARGO_CONFIGURATION_ITEMS)
                    .map(|target| bounded_configuration_value(&target))
                    .collect(),
            }
        }),
        host_compiler: Some(proto::CargoHostCompiler {
            status: match configuration.host_compiler.status {
                CargoHostCompilerStatus::Unknown => 0,
                CargoHostCompilerStatus::Available => 1,
                CargoHostCompilerStatus::Restricted => 2,
                CargoHostCompilerStatus::Missing => 3,
                CargoHostCompilerStatus::Failed => 4,
            },
            release: configuration
                .host_compiler
                .release
                .map(|release| bounded_configuration_value(&release)),
            host_target: configuration
                .host_compiler
                .host_target
                .map(|host| bounded_configuration_value(&host)),
            stale: configuration.host_compiler.stale,
        }),
        cargo_target: match configuration.cargo_target {
            CargoTargetConfiguration::UnresolvedCargoDefault => 0,
        },
        diagnostics: configuration
            .diagnostics
            .into_iter()
            .take(MAX_CARGO_CONFIGURATION_DIAGNOSTICS)
            .map(|diagnostic| proto::CargoConfigurationDiagnostic {
                source_path: diagnostic.source_path.map(|path| path.to_proto()),
                category: match diagnostic.category {
                    CargoConfigurationDiagnosticCategory::Manifest => 0,
                    CargoConfigurationDiagnosticCategory::Toolchain => 1,
                    CargoConfigurationDiagnosticCategory::CompilerProbe => 2,
                },
                message: bounded_configuration_value(&diagnostic.message),
            })
            .collect(),
        partial: configuration.completeness == CargoConfigurationCompleteness::Partial,
    }
}

fn target_to_proto(target: crate::cargo_workspace::CargoTargetModel) -> proto::CargoTarget {
    use crate::cargo_workspace::CargoTargetKind;
    let (kind, other_kind) = match target.kind {
        CargoTargetKind::Library => (1, String::new()),
        CargoTargetKind::Binary => (2, String::new()),
        CargoTargetKind::Example => (3, String::new()),
        CargoTargetKind::Test => (4, String::new()),
        CargoTargetKind::Bench => (5, String::new()),
        CargoTargetKind::BuildScript => (6, String::new()),
        CargoTargetKind::Other(value) => (0, value),
    };
    proto::CargoTarget {
        name: target.name,
        kind,
        other_kind,
        crate_types: target.crate_types,
        source_path: target.source_path.map(|path| path.to_proto()),
        source_display_path: target.source_display_path,
        required_features: target.required_features,
        edition: target.edition,
    }
}

fn dependency_to_proto(
    dependency: crate::cargo_workspace::CargoDependencyModel,
) -> proto::CargoDependency {
    use crate::cargo_workspace::{
        CargoDependencyDeclarationOrigin, CargoDependencyFeatureCausality, CargoDependencyKind,
        CargoDependencyLockStatus,
    };
    proto::CargoDependency {
        declaration_name: dependency.declaration_name,
        rename: dependency.rename,
        kind: match dependency.kind {
            CargoDependencyKind::Unknown => 0,
            CargoDependencyKind::Normal => 1,
            CargoDependencyKind::Development => 2,
            CargoDependencyKind::Build => 3,
        },
        version_requirement: dependency.version_requirement,
        optional: dependency.optional,
        uses_default_features: dependency.uses_default_features,
        requested_features: dependency.requested_features,
        target: dependency.target,
        source_kind: source_kind_to_proto(dependency.source_kind),
        resolved_name: dependency.resolved_name,
        resolved_version: dependency.resolved_version,
        resolved_workspace_member: dependency
            .resolved_workspace_member
            .map(|path| path.to_proto()),
        local_manifest: dependency.local_manifest.map(|path| path.to_proto()),
        declaration_manifest: dependency.declaration_manifest.map(|path| path.to_proto()),
        declaration_origin: match dependency.declaration_origin {
            CargoDependencyDeclarationOrigin::Unknown => 0,
            CargoDependencyDeclarationOrigin::Direct => 1,
            CargoDependencyDeclarationOrigin::WorkspaceInherited => 2,
        },
        resolved_instances: dependency
            .resolved_instances
            .into_iter()
            .map(|resolved| proto::CargoResolvedDependency {
                name: resolved.name,
                version: resolved.version,
                source_kind: source_kind_to_proto(resolved.source_kind),
                enabled_features: resolved.enabled_features,
                lock_status: match resolved.lock_status {
                    CargoDependencyLockStatus::Unknown => 0,
                    CargoDependencyLockStatus::Locked => 1,
                    CargoDependencyLockStatus::NotLocked => 2,
                    CargoDependencyLockStatus::MissingLockfile => 3,
                },
                workspace_member: resolved.workspace_member.map(|path| path.to_proto()),
                local_manifest: resolved.local_manifest.map(|path| path.to_proto()),
            })
            .collect(),
        resolution_truncated: dependency.resolution_truncated,
        feature_causality: match dependency.feature_causality {
            CargoDependencyFeatureCausality::Unknown => 0,
            CargoDependencyFeatureCausality::Validated => 1,
            CargoDependencyFeatureCausality::Ambiguous => 2,
        },
        cycle_detected: dependency.cycle_detected,
    }
}

fn snapshot_from_proto(response: proto::CargoWorkspaceResponse) -> Result<CargoWorkspaceSnapshot> {
    use proto::Message as _;
    ensure!(
        response.workspaces.len() <= 128
            && response.failures.len() <= 128
            && response.encoded_len() <= crate::cargo_workspace::MAX_CARGO_SNAPSHOT_BYTES,
        "Remote Cargo snapshot payload exceeded its retained limit"
    );
    let workspaces = response
        .workspaces
        .into_iter()
        .map(workspace_from_proto)
        .collect::<Result<Vec<_>>>()?;
    let failures = response
        .failures
        .into_iter()
        .map(|failure| {
            Ok(CargoCandidateFailure {
                manifest_path: ProjectPath::from_proto(
                    failure
                        .manifest_path
                        .context("missing Cargo failure path")?,
                )
                .context("invalid Cargo failure path")?,
                category: error_category_from_proto(failure.category),
                message: bounded_error(failure.message.as_bytes()),
                has_stale_model: failure.has_stale_model,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let snapshot = CargoWorkspaceSnapshot {
        revision: response.revision,
        input_fingerprint: response.input_fingerprint,
        workspaces,
        failures,
        completeness: if response.partial {
            CargoSnapshotCompleteness::Partial
        } else {
            CargoSnapshotCompleteness::Complete
        },
    };
    validate_cargo_snapshot_bounds(&snapshot)?;
    Ok(snapshot)
}

fn workspace_from_proto(workspace: proto::CargoWorkspace) -> Result<CargoWorkspaceModel> {
    let root = ProjectPath::from_proto(workspace.root.context("missing Cargo workspace root")?)
        .context("invalid Cargo workspace root")?;
    let root_manifest = workspace.root_manifest.and_then(ProjectPath::from_proto);
    let members = workspace
        .members
        .into_iter()
        .map(package_from_proto)
        .collect::<Result<Vec<_>>>()?;
    let configuration = workspace
        .configuration
        .map(configuration_from_proto)
        .transpose()?
        .unwrap_or_else(crate::cargo_workspace::CargoWorkspaceConfiguration::unresolved);
    Ok(CargoWorkspaceModel {
        key: crate::cargo_workspace::CargoWorkspaceKey { root },
        root_manifest,
        display_name: workspace.display_name,
        is_virtual: workspace.is_virtual,
        members,
        configuration,
        dependency_graph: workspace
            .dependency_graph
            .map(graph_from_proto)
            .transpose()?,
    })
}

fn configuration_from_proto(
    configuration: proto::CargoWorkspaceConfiguration,
) -> Result<crate::cargo_workspace::CargoWorkspaceConfiguration> {
    use crate::cargo_workspace::*;
    let profiles = configuration
        .profiles
        .into_iter()
        .take(MAX_CARGO_CONFIGURATION_ITEMS)
        .map(|profile| CargoProfileModel {
            name: bounded_configuration_value(&profile.name),
            origin: if profile.origin == 1 {
                CargoProfileOrigin::Declared
            } else {
                CargoProfileOrigin::Implicit
            },
        })
        .collect();
    let declared_toolchain = configuration
        .declared_toolchain
        .map(|toolchain| {
            Ok::<CargoToolchainModel, anyhow::Error>(CargoToolchainModel {
                source_path: ProjectPath::from_proto(
                    toolchain
                        .source_path
                        .context("missing Cargo toolchain source path")?,
                )
                .context("invalid Cargo toolchain source path")?,
                format: if toolchain.format == 1 {
                    CargoToolchainFormat::Legacy
                } else {
                    CargoToolchainFormat::Toml
                },
                channel: toolchain
                    .channel
                    .map(|channel| bounded_configuration_value(&channel)),
                components: toolchain
                    .components
                    .into_iter()
                    .take(MAX_CARGO_CONFIGURATION_ITEMS)
                    .map(|component| bounded_configuration_value(&component))
                    .collect(),
                targets: toolchain
                    .targets
                    .into_iter()
                    .take(MAX_CARGO_CONFIGURATION_ITEMS)
                    .map(|target| bounded_configuration_value(&target))
                    .collect(),
            })
        })
        .transpose()?;
    let host_compiler = configuration
        .host_compiler
        .map(|host| CargoHostCompilerModel {
            status: match host.status {
                1 => CargoHostCompilerStatus::Available,
                2 => CargoHostCompilerStatus::Restricted,
                3 => CargoHostCompilerStatus::Missing,
                4 => CargoHostCompilerStatus::Failed,
                _ => CargoHostCompilerStatus::Unknown,
            },
            release: host
                .release
                .map(|release| bounded_configuration_value(&release)),
            host_target: host
                .host_target
                .map(|target| bounded_configuration_value(&target)),
            stale: host.stale,
        })
        .unwrap_or_else(CargoHostCompilerModel::unknown);
    let diagnostics = configuration
        .diagnostics
        .into_iter()
        .take(MAX_CARGO_CONFIGURATION_DIAGNOSTICS)
        .map(|diagnostic| CargoConfigurationDiagnostic {
            source_path: diagnostic.source_path.and_then(ProjectPath::from_proto),
            category: match diagnostic.category {
                1 => CargoConfigurationDiagnosticCategory::Toolchain,
                2 => CargoConfigurationDiagnosticCategory::CompilerProbe,
                _ => CargoConfigurationDiagnosticCategory::Manifest,
            },
            message: bounded_configuration_value(&diagnostic.message),
        })
        .collect();
    Ok(CargoWorkspaceConfiguration {
        profiles,
        declared_toolchain,
        host_compiler,
        cargo_target: CargoTargetConfiguration::UnresolvedCargoDefault,
        diagnostics,
        completeness: if configuration.partial {
            CargoConfigurationCompleteness::Partial
        } else {
            CargoConfigurationCompleteness::Complete
        },
    })
}

fn package_from_proto(
    package: proto::CargoPackage,
) -> Result<crate::cargo_workspace::CargoPackageModel> {
    use crate::cargo_workspace::*;
    let manifest_path = ProjectPath::from_proto(
        package
            .manifest_path
            .context("missing Cargo package manifest path")?,
    )
    .context("invalid Cargo package manifest path")?;
    let id = stable_package_id(&manifest_path, &package.name, &package.version);
    Ok(CargoPackageModel {
        id,
        name: package.name,
        version: package.version,
        manifest_path,
        is_default_member: package.is_default_member,
        targets: package
            .targets
            .into_iter()
            .map(|target| CargoTargetModel {
                name: target.name,
                kind: match target.kind {
                    1 => CargoTargetKind::Library,
                    2 => CargoTargetKind::Binary,
                    3 => CargoTargetKind::Example,
                    4 => CargoTargetKind::Test,
                    5 => CargoTargetKind::Bench,
                    6 => CargoTargetKind::BuildScript,
                    _ => CargoTargetKind::Other(target.other_kind),
                },
                crate_types: target.crate_types,
                source_path: target.source_path.and_then(ProjectPath::from_proto),
                source_display_path: target.source_display_path,
                required_features: target.required_features,
                edition: target.edition,
            })
            .collect(),
        features: package
            .features
            .into_iter()
            .map(|feature| CargoFeatureModel {
                name: feature.name,
                defined: feature.defined,
                enabled: match feature.enabled {
                    1 => CargoFeatureEnabled::Enabled,
                    2 => CargoFeatureEnabled::Disabled,
                    _ => CargoFeatureEnabled::Unknown,
                },
                expands: feature.expands,
            })
            .collect(),
        dependencies: package
            .dependencies
            .into_iter()
            .map(|dependency| {
                let resolved_instance_count = dependency.resolved_instances.len();
                CargoDependencyModel {
                    declaration_name: dependency.declaration_name,
                    rename: dependency.rename,
                    kind: match dependency.kind {
                        1 => CargoDependencyKind::Normal,
                        2 => CargoDependencyKind::Development,
                        3 => CargoDependencyKind::Build,
                        _ => CargoDependencyKind::Unknown,
                    },
                    version_requirement: dependency.version_requirement,
                    optional: dependency.optional,
                    uses_default_features: dependency.uses_default_features,
                    requested_features: dependency.requested_features,
                    target: dependency.target,
                    source_kind: match dependency.source_kind {
                        1 => CargoDependencySourceKind::Path,
                        2 => CargoDependencySourceKind::Registry,
                        3 => CargoDependencySourceKind::Git,
                        _ => CargoDependencySourceKind::Other,
                    },
                    resolved_name: dependency.resolved_name,
                    resolved_version: dependency.resolved_version,
                    resolved_workspace_member: dependency
                        .resolved_workspace_member
                        .and_then(ProjectPath::from_proto),
                    local_manifest: dependency.local_manifest.and_then(ProjectPath::from_proto),
                    declaration_manifest: dependency
                        .declaration_manifest
                        .and_then(ProjectPath::from_proto),
                    declaration_origin: match dependency.declaration_origin {
                        1 => CargoDependencyDeclarationOrigin::Direct,
                        2 => CargoDependencyDeclarationOrigin::WorkspaceInherited,
                        _ => CargoDependencyDeclarationOrigin::Unknown,
                    },
                    resolved_instances: dependency
                        .resolved_instances
                        .into_iter()
                        .take(MAX_CARGO_DEPENDENCY_INSTANCES)
                        .map(|resolved| CargoResolvedDependencyModel {
                            name: resolved.name,
                            version: resolved.version,
                            source_kind: match resolved.source_kind {
                                1 => CargoDependencySourceKind::Path,
                                2 => CargoDependencySourceKind::Registry,
                                3 => CargoDependencySourceKind::Git,
                                _ => CargoDependencySourceKind::Other,
                            },
                            enabled_features: resolved
                                .enabled_features
                                .into_iter()
                                .take(MAX_CARGO_DEPENDENCY_FEATURES)
                                .collect(),
                            lock_status: match resolved.lock_status {
                                1 => CargoDependencyLockStatus::Locked,
                                2 => CargoDependencyLockStatus::NotLocked,
                                3 => CargoDependencyLockStatus::MissingLockfile,
                                _ => CargoDependencyLockStatus::Unknown,
                            },
                            workspace_member: resolved
                                .workspace_member
                                .and_then(ProjectPath::from_proto),
                            local_manifest: resolved
                                .local_manifest
                                .and_then(ProjectPath::from_proto),
                        })
                        .collect(),
                    resolution_truncated: dependency.resolution_truncated
                        || resolved_instance_count > MAX_CARGO_DEPENDENCY_INSTANCES,
                    feature_causality: match dependency.feature_causality {
                        1 => CargoDependencyFeatureCausality::Validated,
                        2 => CargoDependencyFeatureCausality::Ambiguous,
                        _ => CargoDependencyFeatureCausality::Unknown,
                    },
                    cycle_detected: dependency.cycle_detected,
                }
            })
            .collect(),
    })
}

fn error_category_to_proto(category: CargoWorkspaceErrorCategory) -> i32 {
    match category {
        CargoWorkspaceErrorCategory::Internal => 0,
        CargoWorkspaceErrorCategory::Restricted => 1,
        CargoWorkspaceErrorCategory::CargoNotFound => 2,
        CargoWorkspaceErrorCategory::CargoFailed => 3,
        CargoWorkspaceErrorCategory::InvalidMetadata => 4,
        CargoWorkspaceErrorCategory::UnsupportedMetadata => 5,
        CargoWorkspaceErrorCategory::Disconnected => 6,
        CargoWorkspaceErrorCategory::Cancelled => 7,
    }
}

fn error_category_from_proto(category: i32) -> CargoWorkspaceErrorCategory {
    match category {
        1 => CargoWorkspaceErrorCategory::Restricted,
        2 => CargoWorkspaceErrorCategory::CargoNotFound,
        3 => CargoWorkspaceErrorCategory::CargoFailed,
        4 => CargoWorkspaceErrorCategory::InvalidMetadata,
        5 => CargoWorkspaceErrorCategory::UnsupportedMetadata,
        6 => CargoWorkspaceErrorCategory::Disconnected,
        7 => CargoWorkspaceErrorCategory::Cancelled,
        _ => CargoWorkspaceErrorCategory::Internal,
    }
}

#[cfg(test)]
mod tests {
    use fs::Fs as _;
    use std::sync::Mutex;

    use gpui::{BorrowAppContext as _, TestAppContext};

    use super::*;

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_runtime_installed_proxy_images_and_configured_wrapper_are_sealed(
        cx: &mut TestAppContext,
    ) {
        // Actual child pipe notifications must arrive before virtual time reaches the deadline.
        cx.executor().allow_parking();
        let (directory, root, wrapper, environment) = cx
            .background_spawn(async {
                use std::os::unix::fs::PermissionsExt as _;
                let directory =
                    tempfile::tempdir().expect("private installed native policy fixture");
                let directory_path =
                    std::fs::canonicalize(directory.path()).expect("actual private directory");
                let root = directory_path.join("actual installed selection");
                std::fs::create_dir_all(root.join("bin")).expect("private installed fixture");
                for name in ["rustc", "cargo"] {
                    let path = root.join("bin").join(name);
                    std::fs::write(&path, b"actual selected image bytes")
                        .expect("private selected image");
                    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                        .expect("selected executable mode");
                }
                let wrapper = directory_path.join("configured compiler wrapper");
                std::fs::write(&wrapper, b"distinct configured wrapper image")
                    .expect("private wrapper image");
                std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))
                    .expect("wrapper executable mode");
                let mut environment =
                    passive_rustup_proxy_fixture(&directory_path, "copy", "1.28.0");
                environment.insert(
                    "PASSIVE_TEST_INSTALLED_ROOT".into(),
                    root.to_string_lossy().into_owned(),
                );
                environment.insert("RUSTUP_TOOLCHAIN".into(), "installed-fixture".into());
                environment.insert(
                    "RUSTC_WRAPPER".into(),
                    wrapper.to_string_lossy().into_owned(),
                );
                let mut environment = PassiveRustEnvironment::from_request(Some(&environment));
                environment.inherit_process = false;
                (directory, root, wrapper, environment)
            })
            .await;
        let executor = cx.background_executor.clone();
        let work = cx.background_spawn(async move {
            let cwd = std::fs::canonicalize(directory.path()).expect("same actual private cwd");
            let filesystem = fs::RealFs::new(None, executor.clone());
            let execution = prepare_passive_rust_execution_in_environment(
                Some(OsStr::new("cargo")),
                false,
                &cwd,
                environment.clone(),
                &executor,
            )
            .await
            .expect("genuine supported installed selection");
            assert_eq!(execution.toolchain.as_ref(), Some(&root));
            assert_eq!(
                execution.delegated_executables,
                vec![root.join("bin/rustc"), root.join("bin/cargo")]
            );
            assert_eq!(
                execution.compiler_chain.first(),
                Some(&wrapper.as_os_str().to_owned())
            );
            let seals = seal_native_compiler_executables(&filesystem, &execution, &cwd)
                .await
                .expect("actual proxy wrapper and delegated image seals");
            for path in [&wrapper, &root.join("bin/rustc"), &root.join("bin/cargo")] {
                assert!(
                    seals.iter().any(|seal| &seal.requested == path),
                    "each used actual image must be sealed"
                );
            }
            let compiler = seals
                .iter()
                .find(|seal| seal.requested == root.join("bin/rustc"))
                .expect("observed delegated compiler");
            std::fs::write(root.join("bin/rustc"), b"replaced delegated compiler bytes")
                .expect("owned actual image replacement");
            assert!(
                fingerprint_native_probe_file(
                    &filesystem,
                    &compiler.file.root,
                    compiler.file.root_inode,
                    &compiler.file.relative,
                    compiler.file.metadata,
                    MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES
                )
                .await
                .is_err(),
                "unchanged proxy and paths cannot authorize a changed delegated compiler image"
            );
            let fresh = seal_native_compiler_executables(&filesystem, &execution, &cwd)
                .await
                .expect("fresh actual image seals");
            let fresh_compiler = fresh
                .iter()
                .find(|seal| seal.requested == compiler.requested)
                .expect("same observed selected image");
            assert_ne!(fresh_compiler.file.fingerprint, compiler.file.fingerprint);
            std::fs::remove_file(root.join("bin/rustc"))
                .expect("owned missing installed image control");
            assert!(
                prepare_passive_rust_execution_in_environment(
                    Some(OsStr::new("cargo")),
                    false,
                    &cwd,
                    environment,
                    &executor
                )
                .await
                .is_err(),
                "missing actual installed compiler must fail without install"
            );
            assert!(
                !cwd.join("collector-launched").exists(),
                "read-only policy never executes the workload/compiler"
            );
            drop(directory);
        });
        work.await;
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn native_runtime_policy_commands_use_exact_complete_map_and_preserve_legacy(
        cx: &mut TestAppContext,
    ) {
        // Actual child pipe notifications must arrive before virtual time reaches the deadline.
        cx.executor().allow_parking();
        let marker = OsString::from("NATIVE_RUNTIME_POLICY_CASE");
        let value = OsString::from("owned literal environment");
        let parent_key = std::env::vars_os()
            .find_map(|(key, _)| (key != marker).then_some(key))
            .expect("actual test process has an inherited environment");
        for inherit_process in [false, true] {
            let environment = PassiveRustEnvironment {
                overrides: HashMap::from_iter([(marker.clone(), value.clone())]),
                inherit_process,
            };
            let output = cx
                .background_spawn({
                    let executor = cx.background_executor.clone();
                    async move {
                        let mut command = util::command::new_command("/usr/bin/env");
                        environment.apply_to_command(&mut command);
                        bounded_cargo_command_output(
                            command,
                            "native policy exact environment control",
                            64 * 1024,
                            CARGO_CONFIGURATION_PROBE_TIMEOUT,
                            &executor,
                        )
                        .await
                    }
                })
                .await
                .expect("actual managed environment child");
            assert!(output.status.success());
            let values = std::str::from_utf8(&output.stdout)
                .expect("actual environment is UTF-8")
                .lines()
                .filter_map(|line| line.split_once('='))
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(
                values.get("NATIVE_RUNTIME_POLICY_CASE"),
                Some(&"owned literal environment")
            );
            let parent_key = parent_key.to_str().expect("selected parent name is UTF-8");
            if inherit_process {
                assert!(
                    values.contains_key(parent_key),
                    "legacy policy retains actual ambient parent entry"
                );
            } else {
                assert_eq!(
                    values.len(),
                    1,
                    "native complete-map policy adds no ambient entries"
                );
                assert!(
                    !values.contains_key(parent_key),
                    "native complete-map policy removes actual ambient parent entry"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_outer_map_preserves_parent_selectors_without_promoting_configured_child_values()
     {
        let parent = HashMap::from_iter([
            ("RUSTC".into(), "/owned/parent-rustc".into()),
            ("PATH".into(), "/owned/parent-bin".into()),
            ("RUSTUP_AUTO_INSTALL".into(), "1".into()),
            ("CARGO_NET_OFFLINE".into(), "false".into()),
        ]);
        let mut configured_child = parent.clone();
        configured_child.insert("RUSTC".into(), "/configured/child-rustc".into());
        configured_child.insert("RUSTC_WRAPPER".into(), "/configured/child-wrapper".into());
        configured_child.insert("PATH".into(), "/configured/child-bin".into());
        let observed = native_runtime_outer_environment(
            &parent,
            Some(Path::new("/owned/installed-toolchain")),
        )
        .expect("reviewed outer map");
        assert_eq!(observed.get("RUSTC"), parent.get("RUSTC"));
        assert_ne!(observed.get("RUSTC"), configured_child.get("RUSTC"));
        assert!(!observed.contains_key("RUSTC_WRAPPER"));
        assert_eq!(observed.get("PATH"), parent.get("PATH"));
        assert_eq!(
            observed.get("RUSTUP_AUTO_INSTALL").map(String::as_str),
            Some("0")
        );
        assert_eq!(
            observed.get("CARGO_NET_OFFLINE").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            observed.get("RUSTUP_TOOLCHAIN").map(String::as_str),
            Some("/owned/installed-toolchain")
        );
        assert_eq!(observed.len(), parent.len() + 1);
        let without_parent_selectors = native_runtime_outer_environment(&HashMap::default(), None)
            .expect("empty complete parent map");
        assert!(!without_parent_selectors.contains_key("RUSTC"));
        assert!(!without_parent_selectors.contains_key("RUSTC_WRAPPER"));
        assert!(!without_parent_selectors.contains_key("PATH"));
        assert_eq!(without_parent_selectors.len(), 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_precompiler_loader_paths_cannot_claim_observed_provenance() {
        let mut environment = PassiveRustEnvironment {
            overrides: HashMap::from_iter([(
                OsString::from("PATH"),
                OsString::from("/usr/bin:/bin"),
            )]),
            inherit_process: false,
        };
        let mut configuration = CargoHostExecutionConfiguration::default();
        validate_native_precompiler_loader_environment(&environment, &configuration)
            .expect("ordinary complete environment has no loader injection");
        for key in [
            "DYLD_FALLBACK_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "LD_LIBRARY_PATH",
        ] {
            environment
                .overrides
                .insert(key.into(), "/owned/compiler-looking/path".into());
            assert!(
                validate_native_precompiler_loader_environment(&environment, &configuration)
                    .is_err(),
                "a plausible path cannot construct actual compiler provenance"
            );
            assert!(environment.overrides.remove(OsStr::new(key)).is_some());
        }
        configuration.environment.insert(
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            crate::cargo_workspace::CargoConfigurationEnvironment::String(
                crate::cargo_workspace::CargoConfigurationString {
                    value: "/owned/compiler-looking/path".into(),
                    base_directory: PathBuf::from("/owned"),
                },
            ),
        );
        assert!(
            validate_native_precompiler_loader_environment(&environment, &configuration).is_err(),
            "configured loader overrides must refuse before executing a compiler probe"
        );
        configuration.environment.clear();
        environment.inherit_process = true;
        assert!(
            validate_native_precompiler_loader_environment(&environment, &configuration).is_err()
        );
        environment.inherit_process = false;
        validate_native_precompiler_loader_environment(&environment, &configuration)
            .expect("fresh ordinary recovery");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_sole_fallback_refuses_overrides_implicit_paths_and_excess() {
        for name in [
            "DYLD_LIBRARY_PATH",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_UNKNOWN",
            "LD_UNKNOWN",
        ] {
            let mut environment =
                std::collections::BTreeMap::from([(name.into(), "/private/tmp/foreign".into())]);
            assert!(validate_native_runtime_sole_fallback(&environment).is_err());
            environment.insert(name.into(), String::new());
            validate_native_runtime_sole_fallback(&environment).expect("empty loader setting");
        }
        for value in [
            ":/compiler/lib".into(),
            "/compiler/lib:".into(),
            "/compiler/lib::/compiler/lib".into(),
            "/compiler/lib\n".into(),
            "x".repeat(64 * 1024 + 1),
            vec!["/compiler/lib"; 129].join(":"),
        ] {
            let environment =
                std::collections::BTreeMap::from([("DYLD_FALLBACK_LIBRARY_PATH".into(), value)]);
            assert!(validate_native_runtime_sole_fallback(&environment).is_err());
        }
        validate_native_runtime_sole_fallback(&std::collections::BTreeMap::from([(
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            "/compiler/lib:/owned/run/deps".into(),
        )]))
        .expect("bounded path data alone does not grant loader authority");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_loader_scope_pins_only_sealed_compiler_and_refuses_runtime_erasure() {
        let compiler = PathBuf::from("/native/toolchain/target-lib");
        let root = PathBuf::from("/native/run/one");
        let scope = NativeCargoRuntimeLoaderScope {
            run_id: uuid::Uuid::from_u128(1),
            compiler_directory: compiler.clone(),
            compiler_inode: 7,
            compiler_sysroot_directory: PathBuf::from("/native/toolchain/lib"),
            compiler_sysroot_inode: 8,
            invocation_output_root: root.clone(),
        };
        for empty in [None, Some(String::new())] {
            let mut environment = HashMap::from_iter([("HOME".into(), "/native/home".into())]);
            if let Some(empty) = empty {
                environment.insert("DYLD_FALLBACK_LIBRARY_PATH".into(), empty);
            }
            scope
                .pin_outer_fallback(&mut environment)
                .expect("DATA-only sealed compiler scope pin");
            assert_eq!(
                environment
                    .get("DYLD_FALLBACK_LIBRARY_PATH")
                    .map(String::as_str),
                compiler.to_str()
            );
            assert_eq!(
                environment.get("HOME").map(String::as_str),
                Some("/native/home")
            );
            assert_eq!(environment.len(), 2);
            let observed = environment.into_iter().collect();
            assert_eq!(
                scope
                    .paths(&observed, NativeRuntimeLoaderStage::ObservedRuntime)
                    .expect("DATA-only retained compiler scope"),
                std::collections::BTreeSet::from([compiler.clone()])
            );
        }
        let repeated = format!("{}:{}", compiler.display(), compiler.display());
        let mut environment =
            HashMap::from_iter([("DYLD_FALLBACK_LIBRARY_PATH".into(), repeated.clone())]);
        scope
            .pin_outer_fallback(&mut environment)
            .expect("owned nonempty DATA path");
        assert_eq!(
            environment.get("DYLD_FALLBACK_LIBRARY_PATH"),
            Some(&repeated)
        );
        for unowned in [
            "/native/home/lib",
            "/usr/local/lib",
            "/usr/lib",
            "/native/toolchain/lib",
            "/native/run/two/debug/deps",
            "/native/run/one/debug",
        ] {
            environment.insert("DYLD_FALLBACK_LIBRARY_PATH".into(), unowned.into());
            assert!(scope.pin_outer_fallback(&mut environment).is_err());
            assert_eq!(
                environment
                    .get("DYLD_FALLBACK_LIBRARY_PATH")
                    .map(String::as_str),
                Some(unowned)
            );
        }
        let fallback = format!(
            "{}:{}:{}:{}",
            root.join("debug").display(),
            root.join("debug/deps").display(),
            compiler.display(),
            compiler.display()
        );
        let observed =
            std::collections::BTreeMap::from([("DYLD_FALLBACK_LIBRARY_PATH".into(), fallback)]);
        assert_eq!(
            scope
                .paths(&observed, NativeRuntimeLoaderStage::ObservedRuntime)
                .expect("DATA-only Cargo produced directories")
                .len(),
            3
        );
        for erased in [
            None,
            Some(String::new()),
            Some(root.join("debug/deps").display().to_string()),
        ] {
            let mut observed = std::collections::BTreeMap::new();
            if let Some(erased) = erased {
                observed.insert("DYLD_FALLBACK_LIBRARY_PATH".into(), erased);
            }
            assert!(
                scope
                    .paths(&observed, NativeRuntimeLoaderStage::ObservedRuntime)
                    .is_err()
            );
        }
        for unowned in [
            "/native/home/lib",
            "/usr/local/lib",
            "/usr/lib",
            "/native/run/two/debug",
        ] {
            let observed = std::collections::BTreeMap::from([(
                "DYLD_FALLBACK_LIBRARY_PATH".into(),
                format!("{}:{unowned}", compiler.display()),
            )]);
            assert!(
                scope
                    .paths(&observed, NativeRuntimeLoaderStage::ObservedRuntime)
                    .is_err()
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_loader_scope_allows_only_exact_sealed_sysroot_library_directory() {
        let root = PathBuf::from("/native/run/one");
        let compiler = PathBuf::from("/native/toolchain/lib/rustlib/host/lib");
        let sysroot = PathBuf::from("/native/toolchain/lib");
        let scope = NativeCargoRuntimeLoaderScope {
            run_id: uuid::Uuid::from_u128(1),
            compiler_directory: compiler.clone(),
            compiler_inode: 7,
            compiler_sysroot_directory: sysroot.clone(),
            compiler_sysroot_inode: 8,
            invocation_output_root: root.clone(),
        };
        let observed = std::collections::BTreeMap::from([(
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            format!(
                "{}:{}:{}",
                root.join("debug/deps").display(),
                compiler.display(),
                sysroot.display()
            ),
        )]);
        assert_eq!(
            scope
                .paths(&observed, NativeRuntimeLoaderStage::ObservedRuntime)
                .expect("DATA-only exact two-directory scope")
                .len(),
            3
        );
        assert!(
            scope
                .paths(&observed, NativeRuntimeLoaderStage::OuterCargo)
                .is_err()
        );
        assert!(native_runtime_library_paths(&observed, &root, &compiler).is_err());
        for unowned in [
            "/native/toolchain/lib/subdirectory",
            "/native/toolchain/lib-other",
            "/native/toolchain",
            "/native/other-toolchain/lib",
            "/native/run/two/debug/deps",
            "/native/run/one",
            "/native/home/lib",
            "/usr/lib",
            "/usr/local/lib",
        ] {
            let observed = std::collections::BTreeMap::from([(
                "DYLD_FALLBACK_LIBRARY_PATH".into(),
                format!("{}:{unowned}", compiler.display()),
            )]);
            assert!(
                scope
                    .paths(&observed, NativeRuntimeLoaderStage::ObservedRuntime)
                    .is_err()
            );
        }
        let erased = std::collections::BTreeMap::from([(
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            sysroot.display().to_string(),
        )]);
        assert!(
            scope
                .paths(&erased, NativeRuntimeLoaderStage::ObservedRuntime)
                .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_runtime_probe_fingerprint_matches_sha256_and_refuses_confined_file_drift(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let directory = tempfile::tempdir().expect("owned fingerprint controls");
        let root = filesystem
            .canonicalize(directory.path())
            .await
            .expect("canonical owned root");
        cx.background_spawn(async move {
            let _directory = directory;
            let root_before = filesystem
                .metadata(&root)
                .await
                .expect("root metadata")
                .expect("root");
            let root_inode = root_before.inode;
            let vectors = [
                (
                    Vec::new(),
                    Some([
                        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8,
                        0x99, 0x6f, 0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
                    ]),
                ),
                (
                    b"abc".to_vec(),
                    Some([
                        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde,
                        0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
                    ]),
                ),
                ((0..65536).map(|index| (index % 251) as u8).collect(), None),
                ((0..65537).map(|index| (index % 251) as u8).collect(), None),
            ];
            for (index, (mut bytes, known)) in vectors.into_iter().enumerate() {
                let relative = PathBuf::from(format!("vector-{index}"));
                let path = root.join(&relative);
                filesystem
                    .write(&path, &bytes)
                    .await
                    .expect("controlled vector write");
                let before = filesystem
                    .metadata(&path)
                    .await
                    .expect("file metadata")
                    .expect("file");
                let mut remaining = 16;
                let members =
                    native_probe_directory_children(filesystem.as_ref(), &root, &mut remaining)
                        .await
                        .expect("controlled directory membership");
                let actual = Box::pin(fingerprint_native_probe_file(
                    filesystem.as_ref(),
                    &root,
                    root_inode,
                    &relative,
                    before,
                    bytes.len() as u64,
                ))
                .await
                .expect("actual no-follow bounded native reader");
                let legacy: [u8; 32] = Sha256::digest(&bytes).into();
                assert_eq!(actual, legacy);
                if let Some(known) = known {
                    assert_eq!(actual, known);
                }
                let after = filesystem
                    .metadata(&path)
                    .await
                    .expect("file metadata")
                    .expect("file");
                assert!(native_probe_metadata_matches(before, after));
                let mut remaining = 16;
                assert_eq!(
                    native_probe_directory_children(filesystem.as_ref(), &root, &mut remaining)
                        .await
                        .expect("membership after hash"),
                    members
                );
                let root_after = filesystem
                    .metadata(&root)
                    .await
                    .expect("root metadata")
                    .expect("root");
                assert_eq!(root_after.inode, root_inode);
                assert!(
                    Box::pin(fingerprint_native_probe_file(
                        filesystem.as_ref(),
                        &root,
                        root_inode.checked_add(1).expect("test inode bound"),
                        &relative,
                        before,
                        bytes.len() as u64,
                    ))
                    .await
                    .is_err()
                );
                if !bytes.is_empty() {
                    assert!(
                        Box::pin(fingerprint_native_probe_file(
                            filesystem.as_ref(),
                            &root,
                            root_inode,
                            &relative,
                            before,
                            bytes.len() as u64 - 1,
                        ))
                        .await
                        .is_err()
                    );
                }
                if index == 3 {
                    let first = bytes.first_mut().expect("cross-buffer vector");
                    *first ^= 1;
                    filesystem
                        .write(&path, &bytes)
                        .await
                        .expect("same-inode controlled hash drift");
                    let changed = filesystem
                        .metadata(&path)
                        .await
                        .expect("changed metadata")
                        .expect("file");
                    assert_eq!(changed.inode, before.inode);
                    assert_eq!(changed.len, before.len);
                    let current = Box::pin(fingerprint_native_probe_file(
                        filesystem.as_ref(),
                        &root,
                        root_inode,
                        &relative,
                        changed,
                        bytes.len() as u64,
                    ))
                    .await
                    .expect("fresh drifted no-follow read");
                    assert_ne!(current, actual);
                    assert_eq!(current, <[u8; 32]>::from(Sha256::digest(&bytes)));
                    let alias = PathBuf::from("alias");
                    filesystem
                        .create_symlink(&root.join(&alias), path.clone())
                        .await
                        .expect("owned alias control");
                    assert!(
                        filesystem
                            .open_sync_confined_file(&root, &alias, root_inode, changed.inode)
                            .await
                            .is_err()
                    );
                    assert!(
                        Box::pin(fingerprint_native_probe_file(
                            filesystem.as_ref(),
                            &root,
                            root_inode,
                            &alias,
                            changed,
                            bytes.len() as u64,
                        ))
                        .await
                        .is_err()
                    );
                }
            }
        })
        .await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_runtime_compiler_sysroot_seals_refuse_changes_aliases_and_aggregate_excess(
        cx: &mut TestAppContext,
    ) {
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/native/compiler-seals",
                serde_json::json!({
                    "lib": {"driver.dylib": "original", "target": {"std.dylib": "target"}},
                }),
            )
            .await;
        let sysroot = Path::new("/native/compiler-seals");
        let mut budget = NativeCompilerLibraryBudget::default();
        let reviewed = Box::pin(seal_native_compiler_sysroot_libraries(
            filesystem.as_ref(),
            sysroot,
            &mut budget,
        ))
        .await
        .expect("actual bounded filesystem seal DATA");
        assert_eq!(reviewed.path, sysroot.join("lib"));
        assert_eq!(reviewed.libraries.len(), 1);
        assert_eq!(budget.bytes, 512 * 1024 * 1024 - 8);
        Box::pin(revalidate_native_cargo_runtime_library_directories(
            filesystem.as_ref(),
            std::slice::from_ref(&reviewed),
        ))
        .await
        .expect("unchanged sysroot files");
        let library = reviewed.libraries.first().expect("sealed driver library");
        let path = sysroot.join("lib/driver.dylib");
        filesystem
            .write(&path, b"modified")
            .await
            .expect("same-inode content mutation");
        filesystem.set_next_mtime(library.metadata.mtime.timestamp_for_user());
        filesystem.touch_path(&path).await;
        let changed = filesystem
            .metadata(&path)
            .await
            .expect("metadata")
            .expect("file");
        assert_eq!(changed.inode, library.metadata.inode);
        assert_eq!(changed.len, library.metadata.len);
        assert_eq!(changed.mtime, library.metadata.mtime);
        assert!(
            Box::pin(revalidate_native_cargo_runtime_library_directories(
                filesystem.as_ref(),
                std::slice::from_ref(&reviewed)
            ))
            .await
            .is_err()
        );
        let mut budget = NativeCompilerLibraryBudget::default();
        let current = Box::pin(seal_native_compiler_sysroot_libraries(
            filesystem.as_ref(),
            sysroot,
            &mut budget,
        ))
        .await
        .expect("fresh content seal");
        filesystem
            .insert_file("/native/compiler-seals/lib/new.rmeta", "data".into())
            .await;
        assert!(
            Box::pin(revalidate_native_cargo_runtime_library_directories(
                filesystem.as_ref(),
                std::slice::from_ref(&current)
            ))
            .await
            .is_err()
        );
        let mut budget = NativeCompilerLibraryBudget::default();
        let current = Box::pin(seal_native_compiler_sysroot_libraries(
            filesystem.as_ref(),
            sysroot,
            &mut budget,
        ))
        .await
        .expect("fresh membership recovery");
        Box::pin(revalidate_native_cargo_runtime_library_directories(
            filesystem.as_ref(),
            std::slice::from_ref(&current),
        ))
        .await
        .expect("fresh source seal remains current");
        let mut bytes = NativeCompilerLibraryBudget {
            bytes: 7,
            ..Default::default()
        };
        assert!(
            Box::pin(seal_native_compiler_sysroot_libraries(
                filesystem.as_ref(),
                sysroot,
                &mut bytes
            ))
            .await
            .is_err()
        );
        let mut count = NativeCompilerLibraryBudget {
            libraries: 0,
            ..Default::default()
        };
        assert!(
            Box::pin(seal_native_compiler_sysroot_libraries(
                filesystem.as_ref(),
                sysroot,
                &mut count
            ))
            .await
            .is_err()
        );
        let mut members = NativeCompilerLibraryBudget {
            members: 0,
            ..Default::default()
        };
        assert!(
            Box::pin(seal_native_compiler_sysroot_libraries(
                filesystem.as_ref(),
                sysroot,
                &mut members
            ))
            .await
            .is_err()
        );
        let directory = filesystem
            .metadata(&sysroot.join("lib/target"))
            .await
            .expect("metadata")
            .expect("directory");
        let mut shared = NativeCompilerLibraryBudget {
            bytes: 12,
            ..Default::default()
        };
        Box::pin(seal_native_compiler_runtime_libraries_with_budget(
            filesystem.as_ref(),
            &sysroot.join("lib/target"),
            directory.inode,
            &mut shared,
        ))
        .await
        .expect("first directory consumes shared budget");
        assert!(
            Box::pin(seal_native_compiler_sysroot_libraries(
                filesystem.as_ref(),
                sysroot,
                &mut shared
            ))
            .await
            .is_err()
        );
        filesystem
            .rename(
                &sysroot.join("lib"),
                &sysroot.join("lib-retired"),
                fs::RenameOptions::default(),
            )
            .await
            .expect("move only the owned test directory");
        filesystem
            .insert_tree(
                "/native/compiler-seals/lib",
                serde_json::json!({
                    "driver.dylib": "modified",
                    "new.rmeta": "data",
                    "target": {"std.dylib": "target"},
                }),
            )
            .await;
        let replacement = filesystem
            .metadata(&sysroot.join("lib"))
            .await
            .expect("replacement metadata")
            .expect("replacement directory");
        assert_ne!(replacement.inode, current.inode);
        assert!(
            Box::pin(revalidate_native_cargo_runtime_library_directories(
                filesystem.as_ref(),
                std::slice::from_ref(&current)
            ))
            .await
            .is_err()
        );
        let mut budget = NativeCompilerLibraryBudget::default();
        let replacement = Box::pin(seal_native_compiler_sysroot_libraries(
            filesystem.as_ref(),
            sysroot,
            &mut budget,
        ))
        .await
        .expect("fresh directory identity recovery");
        Box::pin(revalidate_native_cargo_runtime_library_directories(
            filesystem.as_ref(),
            std::slice::from_ref(&replacement),
        ))
        .await
        .expect("fresh replacement seal remains current");
        filesystem
            .insert_symlink("/native/compiler-seals/alias", sysroot.join("lib"))
            .await;
        assert!(
            Box::pin(seal_native_runtime_library_paths(
                filesystem.as_ref(),
                std::collections::BTreeSet::from([sysroot.join("alias")])
            ))
            .await
            .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_library_policy_preserves_only_owned_bounded_paths() {
        let root = Path::new("/native/run/one");
        let compiler = Path::new("/native/toolchain/lib");
        let paths = (0..128)
            .map(|index| root.join(format!("output-{index}")))
            .collect::<Vec<_>>();
        let mut value = std::env::join_paths(&paths)
            .expect("bounded library paths")
            .into_string()
            .expect("UTF-8 paths");
        let mut environment = std::collections::BTreeMap::from([(
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            value.clone(),
        )]);
        assert_eq!(
            native_runtime_library_paths(&environment, root, compiler)
                .expect("inclusive 128 directories")
                .len(),
            128
        );
        value.push(':');
        value.push_str(paths[0].to_str().expect("UTF-8 duplicate"));
        environment.insert("DYLD_FALLBACK_LIBRARY_PATH".into(), value.clone());
        assert_eq!(
            native_runtime_library_paths(&environment, root, compiler)
                .expect("duplicates do not consume new directory capacity")
                .len(),
            128
        );
        value.push_str(":/native/run/one/overflow");
        environment.insert("DYLD_FALLBACK_LIBRARY_PATH".into(), value);
        assert!(native_runtime_library_paths(&environment, root, compiler).is_err());
        for path in [
            "/native/run/two/deps",
            "/native/run/one",
            "/native/toolchain",
            "relative",
            "/native/run/one/../two",
        ] {
            environment.insert("DYLD_FALLBACK_LIBRARY_PATH".into(), path.into());
            assert!(
                native_runtime_library_paths(&environment, root, compiler).is_err(),
                "unowned path {path}"
            );
        }
        environment.insert(
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            "/native/toolchain/lib:/native/run/one/debug/deps".into(),
        );
        assert_eq!(
            native_runtime_library_paths(&environment, root, compiler)
                .expect("actual compiler and producing output scopes")
                .len(),
            2
        );
        for name in [
            "MAKEFLAGS",
            "CARGO_MAKEFLAGS",
            "LISTEN_FDS",
            "DYLD_INSERT_LIBRARIES",
            "LD_PRELOAD",
        ] {
            environment.insert(name.into(), "not-owned-resource".into());
            assert!(
                native_runtime_library_paths(&environment, root, compiler).is_err(),
                "unsupported resource {name}"
            );
            environment.insert(name.into(), String::new());
            assert!(
                native_runtime_library_paths(&environment, root, compiler).is_ok(),
                "explicit empty resource {name}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_runtime_library_seals_reject_content_membership_and_alias_drift(
        cx: &mut TestAppContext,
    ) {
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/native/runtime-libraries",
                serde_json::json!({
                    "output": {"deps": {"owned.dylib": "first", "metadata.rmeta": "data"}},
                    "compiler": {"runtime.dylib": "compiler"}
                }),
            )
            .await;
        let paths = std::collections::BTreeSet::from([
            PathBuf::from("/native/runtime-libraries/output/deps"),
            PathBuf::from("/native/runtime-libraries/compiler"),
        ]);
        let reviewed = seal_native_runtime_library_paths(filesystem.as_ref(), paths.clone())
            .await
            .expect("ordinary confined runtime libraries");
        assert_eq!(
            reviewed
                .iter()
                .map(|entry| entry.libraries.len())
                .sum::<usize>(),
            2
        );
        revalidate_native_cargo_runtime_library_directories(filesystem.as_ref(), &reviewed)
            .await
            .expect("unchanged runtime libraries");
        filesystem
            .insert_file(
                "/native/runtime-libraries/output/deps/owned.dylib",
                "other".into(),
            )
            .await;
        assert!(
            revalidate_native_cargo_runtime_library_directories(filesystem.as_ref(), &reviewed)
                .await
                .is_err()
        );
        let current = seal_native_runtime_library_paths(filesystem.as_ref(), paths.clone())
            .await
            .expect("fresh content review");
        filesystem
            .insert_file(
                "/native/runtime-libraries/output/deps/new.rmeta",
                "new".into(),
            )
            .await;
        assert!(
            revalidate_native_cargo_runtime_library_directories(filesystem.as_ref(), &current)
                .await
                .is_err()
        );
        let current = seal_native_runtime_library_paths(filesystem.as_ref(), paths.clone())
            .await
            .expect("fresh membership review");
        revalidate_native_cargo_runtime_library_directories(filesystem.as_ref(), &current)
            .await
            .expect("fresh library inputs recover");
        filesystem
            .insert_symlink(
                "/native/runtime-libraries/alias",
                PathBuf::from("/native/runtime-libraries/output/deps"),
            )
            .await;
        assert!(
            seal_native_runtime_library_paths(
                filesystem.as_ref(),
                std::collections::BTreeSet::from([PathBuf::from(
                    "/native/runtime-libraries/alias"
                )])
            )
            .await
            .is_err()
        );
    }

    #[test]
    fn native_runtime_complete_environment_does_not_use_process_fallback() {
        let complete = PassiveRustEnvironment {
            overrides: HashMap::default(),
            inherit_process: false,
        };
        assert_eq!(complete.value("PATH"), None);
        assert_eq!(complete.path_for_executable_lookup(), Some(OsString::new()));
        let legacy = PassiveRustEnvironment::from_request(None);
        assert_eq!(legacy.value("PATH"), std::env::var_os("PATH"));
        assert_eq!(legacy.path_for_executable_lookup(), legacy.value("PATH"));
    }

    #[gpui::test]
    async fn native_runtime_configuration_seals_detect_used_and_absent_rule_drift(
        cx: &mut TestAppContext,
    ) {
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/native/runtime", serde_json::json!({
            "home": {".cargo": {}},
            "workspace": {".cargo": {
                "config.toml": "include = [\"base.toml\"]\n[env]\nNATIVE_RUNTIME_CASE = \"leaf\"\n",
                "base.toml": "[env]\nNATIVE_RUNTIME_CASE = \"base\"\n"
            }, "nested": {}}
        })).await;
        let environment = PassiveRustEnvironment {
            overrides: HashMap::from_iter([
                (
                    OsString::from("CARGO_HOME"),
                    OsString::from("/native/runtime/home/.cargo"),
                ),
                (
                    OsString::from("HOME"),
                    OsString::from("/native/runtime/home"),
                ),
            ]),
            inherit_process: false,
        };
        let cwd = Path::new("/native/runtime/workspace/nested");
        let (configuration, seals) = passive_cargo_configuration_with_native_review(
            cwd,
            &environment,
            Some(filesystem.clone()),
        )
        .await
        .expect("actual selected configuration");
        assert_eq!(
            configuration
                .environment
                .get("NATIVE_RUNTIME_CASE")
                .expect("merged leaf")
                .resolve()
                .expect("literal value")
                .0,
            OsString::from("leaf")
        );
        assert!(!configuration.native_runtime_target_or_runner_configured);
        revalidate_native_cargo_configuration(filesystem.as_ref(), &seals)
            .await
            .expect("unchanged owned routes");
        filesystem
            .insert_file(
                "/native/runtime/workspace/.cargo/base.toml",
                b"[env]\nNATIVE_RUNTIME_CASE = \"base\"\n# content drift\n".to_vec(),
            )
            .await;
        assert!(
            revalidate_native_cargo_configuration(filesystem.as_ref(), &seals)
                .await
                .is_err(),
            "used raw include changes must retire the receipt even with equal merged values"
        );
        let (_, fresh) = passive_cargo_configuration_with_native_review(
            cwd,
            &environment,
            Some(filesystem.clone()),
        )
        .await
        .expect("fresh review after include change");
        filesystem
            .insert_file(
                "/native/runtime/home/.cargo/config.toml",
                b"[build]\nrustc = \"different-compiler\"\n".to_vec(),
            )
            .await;
        assert!(
            revalidate_native_cargo_configuration(filesystem.as_ref(), &fresh)
                .await
                .is_err(),
            "new initially absent configuration must retire the receipt"
        );
        let (_, current) = passive_cargo_configuration_with_native_review(
            cwd,
            &environment,
            Some(filesystem.clone()),
        )
        .await
        .expect("fresh review after configuration appeared");
        revalidate_native_cargo_configuration(filesystem.as_ref(), &current)
            .await
            .expect("fresh configuration recovery");
    }

    struct FakeRunner {
        output: Vec<u8>,
        requests: Arc<Mutex<Vec<CargoMetadataRequest>>>,
    }

    struct RoutingRunner {
        requests: Arc<Mutex<Vec<PathBuf>>>,
    }

    #[cfg(unix)]
    fn poll_cargo_process<T>(cx: &mut TestAppContext, mut observe: impl FnMut() -> Option<T>) -> T {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(result) = observe() {
                return result;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "owned Cargo collector did not complete"
            );
            // Real pipe notifications need CPU time outside the deterministic scheduler.
            let pause = if cx.executor().tick() {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(5)
            };
            std::thread::sleep(pause);
        }
    }

    #[cfg(unix)]
    fn require_cargo_process_exit(cx: &mut TestAppContext, process_id: String) {
        let mut retirement = cx.background_executor.spawn(async move {
            loop {
                let status = util::command::new_command("/bin/kill")
                    .args(["-0", &process_id])
                    .stdout(util::command::Stdio::null())
                    .stderr(util::command::Stdio::null())
                    .kill_on_drop(true)
                    .status()
                    .await
                    .context("failed to inspect owned collector")?;
                if !status.success() {
                    return anyhow::Ok(());
                }
            }
        });
        poll_cargo_process(cx, || (&mut retirement).now_or_never())
            .expect("the exact owned process retires");
    }

    #[cfg(unix)]
    fn passive_rustup_proxy_fixture(
        directory: &Path,
        layout: &str,
        version: &str,
    ) -> HashMap<String, String> {
        use std::os::unix::fs::PermissionsExt as _;

        let manager = directory.join("rustup");
        let script = r#"#!/bin/sh
[ "$RUSTUP_AUTO_INSTALL" = 0 ] && [ "$CARGO_NET_OFFLINE" = true ] || exit 73
case "${0##*/}" in
    rustup)
        if [ "$1 $2 $3 $4" = "component list --installed --toolchain" ]; then
            [ "$5" = installed-fixture ] || exit 75
            printf '%s' "$PASSIVE_TEST_INSTALLED_COMPONENTS"
            exit 0
        fi
        if [ "$1 $2 $3 $4" = "target list --installed --toolchain" ]; then
            [ "$5" = installed-fixture ] || exit 75
            printf '%s' "$PASSIVE_TEST_INSTALLED_TARGETS"
            exit 0
        fi
        if [ "$1 $2" = "override list" ]; then
            printf '%s' "$PASSIVE_TEST_DIRECTORY_OVERRIDES"
            exit 0
        fi
        if [ "$1" = default ]; then
            printf '%s' "$PASSIVE_TEST_CONFIGURED_DEFAULT"
            exit 0
        fi
        if [ "$1" = which ]; then
            [ -n "$PASSIVE_TEST_INSTALLED_ROOT" ] || exit 74
            if [ "$2" = --toolchain ]; then
                [ "$3" = installed-fixture ] || exit 75
                binary="$4"
            else
                [ "$RUSTUP_TOOLCHAIN" = "$PASSIVE_TEST_INSTALLED_ROOT" ] || exit 75
                binary="$2"
            fi
            printf '%s\n' "$binary" >> "$PASSIVE_TEST_POLICY_QUERY_MARKER" || exit 79
            root="$PASSIVE_TEST_INSTALLED_ROOT"
            if [ "$binary" = cargo ] && [ -n "$PASSIVE_TEST_CARGO_ROOT" ]; then
                root="$PASSIVE_TEST_CARGO_ROOT"
            fi
            printf '%s/bin/%s\n' "$root" "$binary"
            exit 0
        fi
        [ "$1" = --version ] || exit 74
        case "$RUSTUP_TOOLCHAIN" in rust-workspace-passive-version-probe-*) ;; *) exit 75;; esac
        if [ "__VERSION__" = idle ]; then
            printf '%s' "$$" > "$PASSIVE_TEST_PROBE_PID.pending"
            /bin/mv "$PASSIVE_TEST_PROBE_PID.pending" "$PASSIVE_TEST_PROBE_PID"
            exec /bin/sleep 30
        fi
        printf 'rustup __VERSION__ (controlled fixture)\n'
        ;;
    cargo)
        [ -z "$PASSIVE_TEST_INSTALLED_ROOT" ] || [ "$RUSTUP_TOOLCHAIN" = "$PASSIVE_TEST_INSTALLED_ROOT" ] || exit 78
        case " $* " in *" --offline "*) ;; *) exit 76;; esac
        printf cargo >> "$PASSIVE_TEST_LAUNCH_MARKER"
        printf 'metadata\n'
        ;;
    rustc)
        [ -z "$PASSIVE_TEST_INSTALLED_ROOT" ] || [ "$RUSTUP_TOOLCHAIN" = "$PASSIVE_TEST_INSTALLED_ROOT" ] || exit 78
        printf rustc >> "$PASSIVE_TEST_LAUNCH_MARKER"
        printf 'compiler\n'
        ;;
    *) exit 77;;
esac
"#;
        std::fs::write(&manager, script.replace("__VERSION__", version))
            .expect("write controlled manager");
        std::fs::set_permissions(&manager, std::fs::Permissions::from_mode(0o700))
            .expect("make controlled manager executable");
        for name in ["cargo", "rustc"] {
            let proxy = directory.join(name);
            match layout {
                "hardlink" => std::fs::hard_link(&manager, &proxy).expect("hardlink proxy"),
                "symlink" => std::os::unix::fs::symlink(&manager, &proxy).expect("symlink proxy"),
                "copy" => {
                    std::fs::copy(&manager, &proxy).expect("copied proxy");
                }
                _ => panic!("unknown fixed proxy fixture layout"),
            }
        }
        HashMap::from_iter([
            ("PATH".into(), directory.to_string_lossy().into_owned()),
            ("RUSTUP_AUTO_INSTALL".into(), "1".into()),
            ("CARGO_NET_OFFLINE".into(), "false".into()),
            ("RUSTUP_TOOLCHAIN".into(), "1.0.0".into()),
            (
                "PASSIVE_TEST_LAUNCH_MARKER".into(),
                directory
                    .join("collector-launched")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "PASSIVE_TEST_POLICY_QUERY_MARKER".into(),
                directory
                    .join("policy-queries")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "PASSIVE_TEST_PROBE_PID".into(),
                directory.join("probe.pid").to_string_lossy().into_owned(),
            ),
        ])
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_legacy_file_requirements_need_installed_host_components_and_targets(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for scenario in [
            "installed",
            "installed-alias",
            "alias-not-installed",
            "ambiguous-alias",
            "malformed-manifest",
            "oversized-manifest",
            "missing-component",
            "missing-target",
            "non-host-std",
            "qualified-component",
            "ambiguous-host",
        ] {
            let directory = tempfile::tempdir().expect("installed requirements fixture");
            let root = directory.path().join("installed compiler");
            std::fs::create_dir_all(root.join("bin")).expect("installed root");
            for binary in ["rustc", "cargo"] {
                std::fs::write(root.join("bin").join(binary), b"installed")
                    .expect("installed binary");
            }
            let root = std::fs::canonicalize(root).expect("canonical installed root");
            let mut environment = passive_rustup_proxy_fixture(directory.path(), "copy", "1.27.1");
            environment.insert("RUSTUP_TOOLCHAIN".into(), String::new());
            environment.insert(
                "PASSIVE_TEST_INSTALLED_ROOT".into(),
                root.to_string_lossy().into_owned(),
            );
            environment.insert(
                "PASSIVE_TEST_DIRECTORY_OVERRIDES".into(),
                "no overrides\n".into(),
            );
            environment.insert("PASSIVE_TEST_INSTALLED_COMPONENTS".into(), "rustc-aarch64-apple-darwin\nrustfmt-aarch64-apple-darwin\nrust-src\nrust-std-aarch64-apple-darwin\nrust-std-wasm32-unknown-unknown\n".into());
            environment.insert(
                "PASSIVE_TEST_INSTALLED_TARGETS".into(),
                "aarch64-apple-darwin\nwasm32-unknown-unknown\n".into(),
            );
            let mut component = "rustfmt";
            let mut target = "wasm32-unknown-unknown";
            match scenario {
                "installed" => {}
                "installed-alias"
                | "alias-not-installed"
                | "ambiguous-alias"
                | "malformed-manifest"
                | "oversized-manifest" => {
                    component = "rustfmt-preview";
                    let manifest_directory = root.join("lib/rustlib");
                    std::fs::create_dir_all(&manifest_directory)
                        .expect("installed manifest directory");
                    let contents = match scenario {
                        "alias-not-installed" => "[renames.clippy]\nto = \"rustfmt-preview\"\n".into(),
                        "ambiguous-alias" => "[renames.rustfmt]\nto = \"rustfmt-preview\"\n[renames.clippy]\nto = \"rustfmt-preview\"\n".into(),
                        "malformed-manifest" => "[private invalid manifest".into(),
                        "oversized-manifest" => " ".repeat(2 * 1024 * 1024 + 1),
                        _ => "[renames.rustfmt]\nto = \"rustfmt-preview\"\n".into(),
                    };
                    std::fs::write(
                        manifest_directory.join("multirust-channel-manifest.toml"),
                        contents,
                    )
                    .expect("installed component alias manifest");
                }
                "missing-component" => component = "clippy",
                "missing-target" => target = "wasm32-wasip2",
                "non-host-std" => {
                    component = "rust-std";
                    environment.insert(
                        "PASSIVE_TEST_INSTALLED_COMPONENTS".into(),
                        "rustc-aarch64-apple-darwin\nrust-std-wasm32-unknown-unknown\n".into(),
                    );
                }
                "qualified-component" => component = "rustfmt-aarch64-apple-darwin",
                "ambiguous-host" => {
                    environment.insert("PASSIVE_TEST_INSTALLED_COMPONENTS".into(), "rustc-aarch64-apple-darwin\nrustc-x86_64-apple-darwin\nrustfmt-aarch64-apple-darwin\n".into());
                }
                _ => panic!("fixed installed requirements scenario"),
            }
            std::fs::write(directory.path().join("rust-toolchain.toml"), format!("[toolchain]\nchannel = \"installed-fixture\"\ncomponents = [{component:?}]\ntargets = [{target:?}]\n")).expect("requested installed requirements");
            let runner = ProcessCargoMetadataRunner {
                executor: cx.background_executor.clone(),
            };
            let request = CargoMetadataRequest {
                manifest_path: directory.path().join("Cargo.toml"),
                working_directory: directory.path().to_path_buf(),
                environment: Some(environment.clone()),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            let metadata = poll_cargo_process(cx, || (&mut task).now_or_never());
            let runner = ProcessCargoConfigurationProbe {
                executor: cx.background_executor.clone(),
            };
            let request = CargoConfigurationProbeRequest {
                working_directory: directory.path().to_path_buf(),
                environment: Some(environment),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            let compiler = poll_cargo_process(cx, || (&mut task).now_or_never());
            if matches!(scenario, "installed" | "installed-alias") {
                assert_eq!(
                    metadata.expect("installed metadata requirements"),
                    b"metadata\n"
                );
                assert_eq!(
                    compiler.expect("installed compiler requirements"),
                    b"compiler\n"
                );
            } else {
                metadata.expect_err("unverified component/target must not launch Cargo");
                compiler.expect_err("unverified component/target must not launch rustc");
                assert!(!directory.path().join("collector-launched").exists());
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_legacy_directory_selection_preserves_override_precedence(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for scenario in [
            "default",
            "nearest-file",
            "same-directory-db",
            "file-before-parent-db",
            "legacy-before-toml",
            "absolute-file",
            "empty-component-default",
        ] {
            let directory = tempfile::tempdir().expect("canonical directory fixture");
            let parent = std::fs::canonicalize(directory.path()).expect("canonical fixture");
            let child = parent.join("child");
            let root = parent.join("installed compiler");
            std::fs::create_dir(&child).expect("fixture working directory");
            std::fs::create_dir_all(root.join("bin")).expect("installed compiler directory");
            for binary in ["rustc", "cargo"] {
                std::fs::write(root.join("bin").join(binary), b"installed")
                    .expect("installed binary fixture");
            }
            let mut environment = passive_rustup_proxy_fixture(&parent, "copy", "1.27.1");
            environment.insert("RUSTUP_TOOLCHAIN".into(), String::new());
            environment.insert(
                "PASSIVE_TEST_INSTALLED_ROOT".into(),
                root.to_string_lossy().into_owned(),
            );
            environment.insert(
                "PASSIVE_TEST_DIRECTORY_OVERRIDES".into(),
                "no overrides\n".into(),
            );
            environment.insert(
                "PASSIVE_TEST_CONFIGURED_DEFAULT".into(),
                "installed-fixture (default)\n".into(),
            );
            match scenario {
                "default" => {}
                "nearest-file" | "file-before-parent-db" => {
                    std::fs::write(child.join("rust-toolchain"), "installed-fixture\n")
                        .expect("nearest file");
                    if scenario == "file-before-parent-db" {
                        environment.insert(
                            "PASSIVE_TEST_DIRECTORY_OVERRIDES".into(),
                            format!("{}\tabsent-fixture\n", parent.display()),
                        );
                    }
                    environment.insert(
                        "PASSIVE_TEST_CONFIGURED_DEFAULT".into(),
                        "absent-fixture (default)\n".into(),
                    );
                }
                "same-directory-db" => {
                    std::fs::write(
                        child.join("rust-toolchain.toml"),
                        "invalid private toolchain",
                    )
                    .expect("shadowed invalid file");
                    environment.insert(
                        "PASSIVE_TEST_DIRECTORY_OVERRIDES".into(),
                        format!("{}\tinstalled-fixture\n", child.display()),
                    );
                }
                "legacy-before-toml" => {
                    std::fs::write(child.join("rust-toolchain"), "installed-fixture\n")
                        .expect("legacy file");
                    std::fs::write(
                        child.join("rust-toolchain.toml"),
                        "invalid private toolchain",
                    )
                    .expect("shadowed TOML file");
                }
                "absolute-file" => {
                    std::fs::write(
                        child.join("rust-toolchain.toml"),
                        format!("[toolchain]\npath = {:?}\n", root.to_string_lossy()),
                    )
                    .expect("absolute file");
                }
                "empty-component-default" => {
                    std::fs::write(
                        child.join("rust-toolchain.toml"),
                        "[toolchain]\ncomponents = []\nprofile = \"minimal\"\n",
                    )
                    .expect("default request file");
                    environment.insert(
                        "PASSIVE_TEST_DIRECTORY_OVERRIDES".into(),
                        format!("{}\tabsent-fixture\n", parent.display()),
                    );
                }
                _ => panic!("fixed canonical fixture scenario"),
            }
            let runner = ProcessCargoMetadataRunner {
                executor: cx.background_executor.clone(),
            };
            let request = CargoMetadataRequest {
                manifest_path: child.join("Cargo.toml"),
                working_directory: child.clone(),
                environment: Some(environment.clone()),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            assert_eq!(
                poll_cargo_process(cx, || (&mut task).now_or_never())
                    .expect("canonical metadata selection"),
                b"metadata\n"
            );
            let runner = ProcessCargoConfigurationProbe {
                executor: cx.background_executor.clone(),
            };
            let request = CargoConfigurationProbeRequest {
                working_directory: child,
                environment: Some(environment),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            assert_eq!(
                poll_cargo_process(cx, || (&mut task).now_or_never())
                    .expect("canonical compiler selection"),
                b"compiler\n"
            );
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_legacy_invalid_file_never_substitutes_installed_default(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for contents in [
            "[toolchain]\nchannel = \"absent-fixture\"\n",
            "[toolchain]\nprofile = \"minimal\"\n",
            "[toolchain]\nchannel = \"private missing closing quote",
            "[toolchain]\npath = \"private relative path\"\n",
            "[toolchain]\nchannel = \"installed-fixture\"\ncomponents = [\"unverified\"]\n",
            "[toolchain]\nchannel = \"installed-fixture\"\ntargets = [\"unverified\"]\n",
        ] {
            let directory = tempfile::tempdir().expect("invalid canonical fixture");
            std::fs::write(directory.path().join("rust-toolchain.toml"), contents)
                .expect("invalid selected file");
            let mut environment = passive_rustup_proxy_fixture(directory.path(), "copy", "1.27.1");
            environment.insert("RUSTUP_TOOLCHAIN".into(), String::new());
            environment.insert(
                "PASSIVE_TEST_DIRECTORY_OVERRIDES".into(),
                "no overrides\n".into(),
            );
            environment.insert(
                "PASSIVE_TEST_CONFIGURED_DEFAULT".into(),
                "installed-fixture (default)\n".into(),
            );
            let runner = ProcessCargoMetadataRunner {
                executor: cx.background_executor.clone(),
            };
            let request = CargoMetadataRequest {
                manifest_path: directory.path().join("Cargo.toml"),
                working_directory: directory.path().to_path_buf(),
                environment: Some(environment.clone()),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            let error = poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect_err("invalid selection must be unavailable");
            assert!(!format!("{error:#}").contains("private"));
            let runner = ProcessCargoConfigurationProbe {
                executor: cx.background_executor.clone(),
            };
            let request = CargoConfigurationProbeRequest {
                working_directory: directory.path().to_path_buf(),
                environment: Some(environment),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect_err("invalid compiler must be unavailable");
            assert!(!directory.path().join("collector-launched").exists());
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_direct_cargo_checks_nested_compiler_proxy(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;

        cx.executor().allow_parking();
        for selection in ["installed-fixture", "absent-fixture"] {
            for compiler_configuration in ["path", "rustc", "cargo-build-rustc", "direct"] {
                let directory = tempfile::tempdir().expect("direct Cargo compiler fixture");
                let root = directory.path().join("installed compiler");
                std::fs::create_dir_all(root.join("bin")).expect("installed compiler directory");
                for binary in ["rustc", "cargo"] {
                    std::fs::write(root.join("bin").join(binary), b"installed")
                        .expect("installed binary fixture");
                }
                let root = std::fs::canonicalize(root).expect("canonical installed compiler");
                let mut environment =
                    passive_rustup_proxy_fixture(directory.path(), "copy", "1.27.1");
                environment.insert("RUSTUP_TOOLCHAIN".into(), selection.into());
                environment.insert(
                    "PASSIVE_TEST_INSTALLED_ROOT".into(),
                    root.to_string_lossy().into_owned(),
                );
                let direct = directory.path().join("direct");
                std::fs::create_dir(&direct).expect("direct compiler directory");
                let cargo = direct.join("cargo");
                std::fs::write(
                    &cargo,
                    "#!/bin/sh\ncompiler=\"${RUSTC:-${CARGO_BUILD_RUSTC:-rustc}}\"\nexec \"$compiler\" -vV\n",
                )
                .expect("direct Cargo compiler launch");
                std::fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o700))
                    .expect("executable direct Cargo");
                environment.insert(
                    "PATH".into(),
                    std::env::join_paths([direct.as_path(), directory.path()])
                        .expect("isolated Cargo PATH")
                        .to_string_lossy()
                        .into_owned(),
                );
                match compiler_configuration {
                    "rustc" | "cargo-build-rustc" => {
                        environment.insert(
                            if compiler_configuration == "rustc" {
                                "RUSTC"
                            } else {
                                "CARGO_BUILD_RUSTC"
                            }
                            .into(),
                            directory
                                .path()
                                .join("rustc")
                                .to_string_lossy()
                                .into_owned(),
                        );
                    }
                    "direct" => {
                        let compiler = direct.join("rustc");
                        std::fs::write(&compiler, "#!/bin/sh\nprintf 'configured compiler\\n'\n")
                            .expect("configured direct compiler");
                        std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700))
                            .expect("executable configured compiler");
                        environment.insert("RUSTC".into(), compiler.to_string_lossy().into_owned());
                    }
                    _ => {}
                }
                let runner = ProcessCargoMetadataRunner {
                    executor: cx.background_executor.clone(),
                };
                let request = CargoMetadataRequest {
                    manifest_path: directory.path().join("Cargo.toml"),
                    working_directory: directory.path().to_path_buf(),
                    environment: Some(environment),
                };
                let mut metadata = cx
                    .background_executor
                    .spawn(async move { runner.run(request).await });
                let result = poll_cargo_process(cx, || (&mut metadata).now_or_never());
                if compiler_configuration == "direct" {
                    assert_eq!(
                        result.expect("preserve configured compiler"),
                        b"configured compiler\n"
                    );
                    assert!(!directory.path().join("policy-queries").exists());
                } else if selection == "installed-fixture" {
                    assert_eq!(result.expect("pin nested installed proxy"), b"compiler\n");
                } else {
                    result.expect_err("missing nested selection must not launch");
                    assert!(!directory.path().join("collector-launched").exists());
                }
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_legacy_explicit_installed_selection_pins_nested_proxies(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for layout in ["hardlink", "symlink", "copy"] {
            for absolute_selection in [false, true] {
                let directory = tempfile::tempdir().expect("installed legacy selection fixture");
                let root = directory.path().join("installed compiler");
                std::fs::create_dir_all(root.join("bin")).expect("installed compiler directory");
                for binary in ["rustc", "cargo"] {
                    std::fs::write(root.join("bin").join(binary), b"installed")
                        .expect("installed binary fixture");
                }
                let selection_root = root.clone();
                let root = std::fs::canonicalize(root).expect("canonical installed fixture root");
                let mut environment =
                    passive_rustup_proxy_fixture(directory.path(), layout, "1.27.1");
                environment.insert(
                    "PASSIVE_TEST_INSTALLED_ROOT".into(),
                    root.to_string_lossy().into_owned(),
                );
                environment.insert(
                    "RUSTUP_TOOLCHAIN".into(),
                    if absolute_selection {
                        selection_root.to_string_lossy().into_owned()
                    } else {
                        "installed-fixture".into()
                    },
                );
                let runner = ProcessCargoMetadataRunner {
                    executor: cx.background_executor.clone(),
                };
                let request = CargoMetadataRequest {
                    manifest_path: directory.path().join("Cargo.toml"),
                    working_directory: directory.path().to_path_buf(),
                    environment: Some(environment.clone()),
                };
                let mut metadata = cx
                    .background_executor
                    .spawn(async move { runner.run(request).await });
                assert_eq!(
                    poll_cargo_process(cx, || (&mut metadata).now_or_never())
                        .expect("installed metadata selection"),
                    b"metadata\n"
                );
                let runner = ProcessCargoConfigurationProbe {
                    executor: cx.background_executor.clone(),
                };
                let request = CargoConfigurationProbeRequest {
                    working_directory: directory.path().to_path_buf(),
                    environment: Some(environment),
                };
                let mut compiler = cx
                    .background_executor
                    .spawn(async move { runner.run(request).await });
                assert_eq!(
                    poll_cargo_process(cx, || (&mut compiler).now_or_never())
                        .expect("installed compiler selection"),
                    b"compiler\n"
                );
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_legacy_explicit_selection_rejects_absent_and_fallback_roots(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for missing in [false, true] {
            let directory = tempfile::tempdir().expect("invalid legacy selection fixture");
            let root = directory.path().join("selected private compiler");
            let fallback = directory.path().join("unselected default compiler");
            for root in [&root, &fallback] {
                std::fs::create_dir_all(root.join("bin")).expect("fixture compiler directory");
                for binary in ["rustc", "cargo"] {
                    std::fs::write(root.join("bin").join(binary), b"installed")
                        .expect("fixture binary");
                }
            }
            let root = std::fs::canonicalize(root).expect("canonical selected compiler root");
            let fallback =
                std::fs::canonicalize(fallback).expect("canonical fallback compiler root");
            let mut environment =
                passive_rustup_proxy_fixture(directory.path(), "hardlink", "1.27.1");
            environment.insert(
                "RUSTUP_TOOLCHAIN".into(),
                if missing {
                    directory.path().join("absent compiler")
                } else {
                    root.clone()
                }
                .to_string_lossy()
                .into_owned(),
            );
            environment.insert(
                "PASSIVE_TEST_INSTALLED_ROOT".into(),
                root.to_string_lossy().into_owned(),
            );
            environment.insert(
                "PASSIVE_TEST_CARGO_ROOT".into(),
                fallback.to_string_lossy().into_owned(),
            );
            let runner = ProcessCargoMetadataRunner {
                executor: cx.background_executor.clone(),
            };
            let request = CargoMetadataRequest {
                manifest_path: directory.path().join("Cargo.toml"),
                working_directory: directory.path().to_path_buf(),
                environment: Some(environment),
            };
            let mut metadata = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            let error = poll_cargo_process(cx, || (&mut metadata).now_or_never())
                .expect_err("absent or different root must not launch the collector");
            assert!(error.to_string().contains("will not install a toolchain"));
            assert!(!error.to_string().contains("private compiler"));
            assert!(!error.to_string().contains("default compiler"));
            assert!(!directory.path().join("collector-launched").exists());
            let queries = directory.path().join("policy-queries");
            if missing {
                assert!(!queries.exists());
            } else {
                assert_eq!(
                    std::fs::read(queries).expect("both policy queries observed"),
                    b"rustc\ncargo\n"
                );
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_rustup_guard_rejects_unsafe_proxy_layouts(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        for layout in ["hardlink", "symlink", "copy"] {
            for version in ["1.27.1", "1.28.0", "1.29.0", "1.28.invalid", "1.28.0-alpha"] {
                let directory = tempfile::tempdir().expect("proxy guard fixture");
                let environment = passive_rustup_proxy_fixture(directory.path(), layout, version);
                let runner = ProcessCargoMetadataRunner {
                    executor: cx.background_executor.clone(),
                };
                let request = CargoMetadataRequest {
                    manifest_path: directory.path().join("Cargo.toml"),
                    working_directory: directory.path().to_path_buf(),
                    environment: Some(environment.clone()),
                };
                let mut metadata = cx
                    .background_executor
                    .spawn(async move { runner.run(request).await });
                let metadata = poll_cargo_process(cx, || (&mut metadata).now_or_never());
                let runner = ProcessCargoConfigurationProbe {
                    executor: cx.background_executor.clone(),
                };
                let request = CargoConfigurationProbeRequest {
                    working_directory: directory.path().to_path_buf(),
                    environment: Some(environment),
                };
                let mut compiler = cx
                    .background_executor
                    .spawn(async move { runner.run(request).await });
                let compiler = poll_cargo_process(cx, || (&mut compiler).now_or_never());
                if matches!(version, "1.28.0" | "1.29.0") {
                    assert_eq!(metadata.expect("safe manager metadata"), b"metadata\n");
                    assert_eq!(compiler.expect("safe manager compiler"), b"compiler\n");
                    assert_eq!(
                        std::fs::read(directory.path().join("collector-launched"))
                            .expect("both collectors launch"),
                        b"cargorustc"
                    );
                } else {
                    for result in [metadata, compiler] {
                        let error = result
                            .expect_err("unsafe or unverifiable manager cannot launch a collector");
                        assert!(
                            error.to_string().contains("will not install a toolchain"),
                            "{error:#}"
                        );
                        assert!(
                            !error.to_string().contains("1.0.0"),
                            "project environment values remain private"
                        );
                    }
                    assert!(!directory.path().join("collector-launched").exists());
                }
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_rustup_probe_timeout_and_cancellation_retire_child(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for timeout in [false, true] {
            let directory = tempfile::tempdir().expect("manager lifecycle fixture");
            let environment = passive_rustup_proxy_fixture(directory.path(), "hardlink", "idle");
            let runner = ProcessCargoMetadataRunner {
                executor: cx.background_executor.clone(),
            };
            let request = CargoMetadataRequest {
                manifest_path: directory.path().join("Cargo.toml"),
                working_directory: directory.path().to_path_buf(),
                environment: Some(environment),
            };
            let mut metadata = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            let process_id = poll_cargo_process(cx, || {
                std::fs::read_to_string(directory.path().join("probe.pid"))
                    .ok()
                    .filter(|value| value.parse::<u32>().is_ok())
            });
            if timeout {
                cx.executor().advance_clock(PASSIVE_RUSTUP_POLICY_TIMEOUT);
                let error = poll_cargo_process(cx, || (&mut metadata).now_or_never())
                    .expect_err("manager preflight times out");
                assert!(error.to_string().contains("timed out"), "{error:#}");
            }
            drop(metadata);
            require_cargo_process_exit(cx, process_id);
            assert!(!directory.path().join("collector-launched").exists());
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_process_bounds_and_timeout_retire_owned_child(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        for mode in ["stdout", "stderr", "timeout", "cancel", "exit"] {
            let directory = tempfile::tempdir().expect("Cargo collector bounds fixture");
            let marker = directory.path().join("collector.pid");
            let body = match mode {
                "stdout" => "printf '123456789'",
                "stderr" => "i=0; while [ \"$i\" -le 4096 ]; do printf x >&2; i=$((i + 1)); done",
                "exit" => "printf OK; printf diagnostic >&2; exit 7",
                _ => ":",
            };
            let mut command = util::command::new_command("/bin/sh");
            command
                .arg("-c")
                .arg(format!(
                    "printf '%s' \"$$\" > \"$1.pending\"; /bin/mv \"$1.pending\" \"$1\"; {body}; exec /bin/sleep 30"
                ))
                .arg("cargo-collector-fixture")
                .arg(&marker);
            let executor = cx.background_executor.clone();
            let mut collection = cx.background_executor.spawn(async move {
                bounded_cargo_command_output(
                    command,
                    "fixture collector",
                    8,
                    CARGO_CONFIGURATION_PROBE_TIMEOUT,
                    &executor,
                )
                .await
            });
            let process_id = poll_cargo_process(cx, || {
                std::fs::read_to_string(&marker)
                    .ok()
                    .filter(|value| value.parse::<u32>().is_ok())
            });
            if mode == "timeout" {
                cx.executor()
                    .advance_clock(CARGO_CONFIGURATION_PROBE_TIMEOUT);
            }
            if mode == "exit" {
                let output = poll_cargo_process(cx, || (&mut collection).now_or_never())
                    .expect("collector EOF preserves bounded output and exit status");
                assert_eq!(output.status.code(), Some(7));
                assert_eq!(output.stdout, b"OK");
                assert_eq!(output.stderr, b"diagnostic");
            } else if mode != "cancel" {
                let error = poll_cargo_process(cx, || (&mut collection).now_or_never())
                    .expect_err("overflow or timeout must reject the collector");
                let expected = if mode == "timeout" {
                    "timed out".to_owned()
                } else {
                    format!("{mode} exceeded the supported limit")
                };
                assert!(error.to_string().contains(&expected), "{error:#}");
            }
            drop(collection);
            require_cargo_process_exit(cx, process_id);
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_processes_prevent_downloads(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt;

        cx.executor().allow_parking();
        let directory = tempfile::tempdir().expect("passive command fixture");
        let check_environment = r#"[ "$RUSTUP_AUTO_INSTALL" = 0 ] &&
    [ "$CARGO_NET_OFFLINE" = true ] &&
    [ "$RUST_WORKSPACE_FIXTURE_VALUE" = retained ] || exit 73
"#;
        for (name, body) in [
            (
                "cargo",
                "case \" $* \" in *\" --offline \"*) ;; *) exit 74;; esac\nprintf 'metadata\\n'\n",
            ),
            ("rustc", "printf 'compiler\\n'\n"),
        ] {
            let path = directory.path().join(name);
            std::fs::write(&path, format!("#!/bin/sh\n{check_environment}{body}"))
                .expect("write passive command fixture");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))
                .expect("make passive command fixture executable");
        }
        let environment = HashMap::from_iter([
            (
                "PATH".into(),
                directory.path().to_string_lossy().into_owned(),
            ),
            ("RUSTUP_AUTO_INSTALL".into(), "1".into()),
            ("CARGO_NET_OFFLINE".into(), "false".into()),
            ("RUST_WORKSPACE_FIXTURE_VALUE".into(), "retained".into()),
        ]);
        let metadata_request = CargoMetadataRequest {
            manifest_path: directory.path().join("Cargo.toml"),
            working_directory: directory.path().to_path_buf(),
            environment: Some(environment.clone()),
        };
        let metadata_runner = ProcessCargoMetadataRunner {
            executor: cx.background_executor.clone(),
        };
        let metadata = cx
            .background_executor
            .spawn(async move { metadata_runner.run(metadata_request).await });
        let probe = ProcessCargoConfigurationProbe {
            executor: cx.background_executor.clone(),
        };
        let probe_request = CargoConfigurationProbeRequest {
            working_directory: directory.path().to_path_buf(),
            environment: Some(environment),
        };
        let compiler = cx
            .background_executor
            .spawn(async move { probe.run(probe_request).await });
        for (expected, mut output) in [
            (b"metadata\n".as_slice(), metadata),
            (b"compiler\n".as_slice(), compiler),
        ] {
            let actual = poll_cargo_process(cx, || (&mut output).now_or_never())
                .expect("project environment cannot enable passive downloads");
            assert_eq!(actual, expected);
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn actual_policy_fixture_requests() -> Result<Vec<String>> {
        use std::io::Read as _;
        let path = PathBuf::from(std::env::var("RUST_WORKSPACE_CONTROL_REQUEST_LOG")?);
        ensure!(
            path.is_absolute(),
            "Actual-policy observer journal must be absolute"
        );
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(64 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 64 * 1024,
            "Actual-policy observer journal exceeds its bound"
        );
        let requests: Vec<String> = serde_json::from_slice(&bytes)?;
        ensure!(
            requests.len() <= 256 && requests.iter().all(|request| request.len() <= 4096),
            "Actual-policy observer request inventory exceeds its bound"
        );
        Ok(requests)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    struct ActualPolicyFixture {
        directory: tempfile::TempDir,
        root: PathBuf,
        installed_root: PathBuf,
        environment: HashMap<String, String>,
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    impl ActualPolicyFixture {
        fn new(version: &str, installed: bool, proxy_cargo: bool) -> Result<Self> {
            use std::io::Read as _;
            let origin = std::env::var("RUST_WORKSPACE_CONTROL_ORIGIN")?;
            let port = origin
                .strip_prefix("http://127.0.0.1:")
                .context("Actual-policy observer must be loopback-only")?
                .parse::<u16>()?;
            ensure!(port != 0, "Actual-policy observer needs an assigned port");
            let sdk = PathBuf::from(std::env::var("RUST_WORKSPACE_ACTUAL_MANAGER_ROOT")?);
            let fixture_root = PathBuf::from(std::env::var("RUST_WORKSPACE_ACTUAL_FIXTURE_ROOT")?);
            ensure!(
                sdk.is_absolute() && fixture_root.is_absolute(),
                "Actual-policy fixture roots must be absolute"
            );
            let source = sdk.join(version).join("rustup");
            let (expected_size, expected_digest) = match version {
                "1.27.1" => (
                    7_585_520,
                    "760b18611021deee1a859c345d17200e0087d47f68dfe58278c57abe3a0d3dd0",
                ),
                "1.28.0" => (
                    10_234_464,
                    "545d36042b524291c5b0689ee69e0d40ec825250c28f560d4c406788e417c1ba",
                ),
                _ => anyhow::bail!("Actual-policy fixture manager version is unsupported"),
            };
            ensure!(
                std::fs::metadata(&source)?.len() == expected_size,
                "Actual-policy manager size differs from archived provenance"
            );
            let mut bytes = Vec::new();
            std::fs::File::open(&source)?
                .take(expected_size + 1)
                .read_to_end(&mut bytes)?;
            ensure!(
                bytes.len() as u64 == expected_size
                    && format!("{:x}", Sha256::digest(&bytes)) == expected_digest,
                "Actual-policy manager SHA differs from archived provenance"
            );
            let installed_root = std::fs::canonicalize(
                Path::new(env!("CARGO"))
                    .parent()
                    .and_then(Path::parent)
                    .context("Actual installed Cargo root is unavailable")?,
            )?;
            for binary in ["cargo", "rustc", "rustdoc"] {
                ensure!(
                    installed_root.join("bin").join(binary).is_file(),
                    "Actual installed toolchain fixture is incomplete"
                );
            }
            let directory = tempfile::Builder::new()
                .prefix("actual-passive-policy-")
                .tempdir_in(fixture_root)?;
            let root = std::fs::canonicalize(directory.path())?;
            for path in [
                "src",
                ".cargo",
                "bin",
                "rustup-home/toolchains",
                "cargo-home",
                "target",
                "build",
            ] {
                std::fs::create_dir_all(root.join(path))?;
            }
            std::fs::write(
                root.join("Cargo.toml"),
                "[package]\nname='actual_passive_policy_fixture'\nversion='0.1.0'\nedition='2024'\n",
            )?;
            std::fs::write(root.join("src/lib.rs"), "pub fn retained() -> u32 { 7 }\n")?;
            let build = toml::Value::String(root.join("build").to_string_lossy().into_owned());
            let target = toml::Value::String(root.join("target").to_string_lossy().into_owned());
            std::fs::write(
                root.join(".cargo/config.toml"),
                format!(
                    "[build]\nbuild-dir={build}\ntarget-dir={target}\n[env]\nCARGO_BUILD_BUILD_DIR={{value={build},force=true}}\nCARGO_TARGET_DIR={{value={target},force=true}}\n"
                ),
            )?;
            for proxy in ["rustup", "rustc", "rustdoc"] {
                std::fs::hard_link(&source, root.join("bin").join(proxy))?;
            }
            if proxy_cargo {
                std::fs::hard_link(&source, root.join("bin/cargo"))?;
            } else {
                std::os::unix::fs::symlink(
                    installed_root.join("bin/cargo"),
                    root.join("bin/cargo"),
                )?;
            }
            let installed_name = installed_root
                .file_name()
                .and_then(OsStr::to_str)
                .context("Actual installed toolchain name is not UTF-8")?;
            std::os::unix::fs::symlink(
                &installed_root,
                root.join("rustup-home/toolchains").join(installed_name),
            )?;
            std::fs::write(
                root.join("rustup-home/settings.toml"),
                format!(
                    "version='12'\ndefault_toolchain={}\n",
                    toml::Value::String(installed_name.into())
                ),
            )?;
            let selection = if installed { installed_name } else { "1.0.0" };
            std::fs::write(
                root.join("rust-toolchain.toml"),
                format!(
                    "[toolchain]\nchannel={}\n",
                    toml::Value::String(selection.into())
                ),
            )?;
            let environment = HashMap::from_iter([
                (
                    "PATH".into(),
                    std::env::join_paths([
                        root.join("bin"),
                        PathBuf::from("/usr/bin"),
                        PathBuf::from("/bin"),
                    ])?
                    .into_string()
                    .map_err(|_| anyhow!("Actual fixture PATH is not UTF-8"))?,
                ),
                ("HOME".into(), root.to_string_lossy().into_owned()),
                (
                    "CARGO_HOME".into(),
                    root.join("cargo-home").to_string_lossy().into_owned(),
                ),
                (
                    "RUSTUP_HOME".into(),
                    root.join("rustup-home").to_string_lossy().into_owned(),
                ),
                ("RUSTUP_TOOLCHAIN".into(), selection.into()),
                (
                    "RUSTC".into(),
                    root.join("bin/rustc").to_string_lossy().into_owned(),
                ),
                (
                    "RUSTDOC".into(),
                    root.join("bin/rustdoc").to_string_lossy().into_owned(),
                ),
                (
                    "CARGO_BUILD_RUSTC".into(),
                    root.join("bin/rustc").to_string_lossy().into_owned(),
                ),
                (
                    "CARGO_BUILD_RUSTDOC".into(),
                    root.join("bin/rustdoc").to_string_lossy().into_owned(),
                ),
                ("RUSTC_WRAPPER".into(), String::new()),
                ("RUSTC_WORKSPACE_WRAPPER".into(), String::new()),
                ("CARGO_BUILD_RUSTC_WRAPPER".into(), String::new()),
                ("CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER".into(), String::new()),
                ("RUSTUP_AUTO_INSTALL".into(), "1".into()),
                ("CARGO_NET_OFFLINE".into(), "false".into()),
                ("RUSTUP_DIST_SERVER".into(), origin.clone()),
                ("RUSTUP_UPDATE_ROOT".into(), format!("{origin}/rustup")),
                ("NO_PROXY".into(), "127.0.0.1".into()),
                ("no_proxy".into(), "127.0.0.1".into()),
                ("HTTP_PROXY".into(), String::new()),
                ("HTTPS_PROXY".into(), String::new()),
                ("ALL_PROXY".into(), String::new()),
                ("http_proxy".into(), String::new()),
                ("https_proxy".into(), String::new()),
                ("all_proxy".into(), String::new()),
                (
                    "CARGO_TARGET_DIR".into(),
                    root.join("target").to_string_lossy().into_owned(),
                ),
                (
                    "CARGO_BUILD_BUILD_DIR".into(),
                    root.join("build").to_string_lossy().into_owned(),
                ),
                ("CARGO_INCREMENTAL".into(), "0".into()),
            ]);
            Ok(Self {
                directory,
                root,
                installed_root,
                environment,
            })
        }

        fn build_directory_option(&self) -> String {
            format!(
                "build.build-dir={}",
                toml::Value::String(self.root.join("build").to_string_lossy().into_owned())
            )
        }

        fn unchanged_installation(&self) -> Result<()> {
            let entries = std::fs::read_dir(self.root.join("rustup-home/toolchains"))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(
                entries.len() == 1,
                "Actual-policy control changed the installed toolchain inventory"
            );
            ensure!(
                std::fs::canonicalize(
                    entries
                        .first()
                        .context("Fixture toolchain disappeared")?
                        .path()
                )? == self.installed_root,
                "Actual-policy control replaced its installed toolchain"
            );
            ensure!(
                self.directory.path().exists(),
                "Actual-policy fixture owner expired"
            );
            Ok(())
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[gpui::test]
    #[ignore = "requires checksum-pinned official Rustup fixtures and an explicitly controlled rejecting loopback observer"]
    fn cargo_workspace_passive_actual_manager_and_dependency_controls(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        for version in ["1.27.1", "1.28.0"] {
            let fixture = ActualPolicyFixture::new(version, false, false)
                .expect("Actual missing-toolchain primitive fixture");
            let before =
                actual_policy_fixture_requests().expect("Before missing-toolchain primitive");
            let root = fixture.root.clone();
            let mut environment = fixture.environment.clone();
            environment.insert("RUSTUP_AUTO_INSTALL".into(), "0".into());
            environment.insert("CARGO_NET_OFFLINE".into(), "true".into());
            let executor = cx.background_executor.clone();
            let mut task = cx.background_executor.spawn(async move {
                let mut command = util::command::new_command(root.join("bin/rustc"));
                command
                    .arg("-vV")
                    .envs(&environment)
                    .current_dir(&root)
                    .kill_on_drop(true);
                bounded_cargo_command_output(
                    command,
                    "Actual missing-toolchain primitive",
                    MAX_RUSTC_VERBOSE_VERSION_BYTES,
                    CARGO_CONFIGURATION_PROBE_TIMEOUT,
                    &executor,
                )
                .await
            });
            let output = poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect("Actual primitive result");
            assert!(!output.status.success());
            let observed = actual_policy_fixture_requests().expect("Primitive observer result");
            if version == "1.27.1" {
                assert_eq!(
                    observed
                        .get(before.len()..)
                        .expect("Observer journal retained original prefix"),
                    [
                        "GET /dist/channel-rust-1.0.0.toml.sha256",
                        "GET /dist/rust-1.0.0-aarch64-apple-darwin.tar.gz.sha256"
                    ],
                    "Actual legacy auto-install negative did not reach the observer"
                );
            } else {
                assert_eq!(
                    observed, before,
                    "Rustup1.28.0 missing selection should not install"
                );
            }
            fixture
                .unchanged_installation()
                .expect("Primitive downloaded no toolchain");
            for proxy_cargo in [true, false] {
                for installed in [true, false] {
                    let fixture = ActualPolicyFixture::new(version, installed, proxy_cargo)
                        .expect("Verified actual manager fixture");
                    let before = actual_policy_fixture_requests().expect("Ready observer journal");
                    let root = fixture.root.clone();
                    let environment = fixture.environment.clone();
                    let executor = cx.background_executor.clone();
                    let mut task = cx.background_executor.spawn(async move {
                        ProcessCargoMetadataRunner { executor }
                            .run(CargoMetadataRequest {
                                manifest_path: root.join("Cargo.toml"),
                                working_directory: root,
                                environment: Some(environment),
                            })
                            .await
                    });
                    let metadata = poll_cargo_process(cx, || (&mut task).now_or_never());
                    let request = CargoConfigurationProbeRequest {
                        working_directory: fixture.root.clone(),
                        environment: Some(fixture.environment.clone()),
                    };
                    let runner = ProcessCargoConfigurationProbe {
                        executor: cx.background_executor.clone(),
                    };
                    let mut task = cx
                        .background_executor
                        .spawn(async move { runner.run(request).await });
                    let compiler = poll_cargo_process(cx, || (&mut task).now_or_never());
                    if installed {
                        let metadata =
                            parse_metadata(&metadata.expect("Actual installed Cargo metadata"))
                                .expect("Actual Cargo schema");
                        assert!(!metadata.packages.is_empty());
                        assert_eq!(metadata.workspace_root.as_std_path(), fixture.root);
                        let compiler = compiler.expect("Actual installed compiler probe");
                        assert!(
                            std::str::from_utf8(&compiler)
                                .expect("Compiler identity UTF-8")
                                .starts_with("rustc ")
                        );
                    } else {
                        for result in [metadata, compiler] {
                            let error =
                                result.expect_err("Actual missing selection must be explicit");
                            let diagnostic = format!("{error:#}");
                            assert!(
                                diagnostic.contains("installed Rust toolchain")
                                    || diagnostic.contains("is not installed"),
                                "Unexpected missing-toolchain failure: {diagnostic}"
                            );
                        }
                    }
                    assert_eq!(
                        actual_policy_fixture_requests().expect("Observer result"),
                        before,
                        "Production owners attempted a download"
                    );
                    fixture
                        .unchanged_installation()
                        .expect("No toolchain installed");
                    println!(
                        "actual-manager-owner version={version} proxy_cargo={proxy_cargo} installed={installed} requests=0"
                    );
                }
            }
            let fixture =
                ActualPolicyFixture::new(version, true, false).expect("Actual dependency fixture");
            let origin = std::env::var("RUST_WORKSPACE_CONTROL_ORIGIN").expect("Observer origin");
            std::fs::write(fixture.root.join("Cargo.toml"), "[package]\nname='actual_passive_policy_fixture'\nversion='0.1.0'\nedition='2024'\n[dependencies]\nrust_workspace_never_published_fixture_dependency='=0.0.1'\n").expect("Missing registry dependency");
            let admitted_config = std::fs::read_to_string(fixture.root.join(".cargo/config.toml"))
                .expect("Isolated build directory configuration");
            std::fs::write(fixture.root.join(".cargo/config.toml"), format!("{admitted_config}\n[source.crates-io]\nreplace-with='controlled-loopback'\n[source.controlled-loopback]\nregistry={}\n", toml::Value::String(format!("sparse+{origin}/index/")))).expect("Loopback registry only");
            let before =
                actual_policy_fixture_requests().expect("Before negative registry control");
            let root = fixture.root.clone();
            let environment = fixture.environment.clone();
            let build_directory = fixture.build_directory_option();
            let direct_cargo = fixture.installed_root.join("bin/cargo");
            let executor = cx.background_executor.clone();
            let mut task = cx.background_executor.spawn(async move {
                let mut command = util::command::new_command(direct_cargo);
                command
                    .envs(&environment)
                    .current_dir(&root)
                    .kill_on_drop(true)
                    .args([
                        "--config",
                        &build_directory,
                        "metadata",
                        "--format-version",
                        "1",
                        "--manifest-path",
                    ])
                    .arg(root.join("Cargo.toml"));
                bounded_cargo_command_output(
                    command,
                    "Actual no-offline registry negative",
                    MAX_CARGO_METADATA_BYTES,
                    CARGO_METADATA_TIMEOUT,
                    &executor,
                )
                .await
            });
            let output = poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect("Actual negative Cargo result");
            assert!(!output.status.success(), "Rejecting registry must fail");
            let observed =
                actual_policy_fixture_requests().expect("Registry negative observations");
            assert!(
                observed.len() > before.len()
                    && observed
                        .iter()
                        .skip(before.len())
                        .any(|request| request.contains(" /index/config.json")),
                "Actual Cargo did not reach the rejecting sparse registry: {observed:?}"
            );
            let before = observed;
            let request = CargoMetadataRequest {
                manifest_path: fixture.root.join("Cargo.toml"),
                working_directory: fixture.root.clone(),
                environment: Some(fixture.environment.clone()),
            };
            let runner = ProcessCargoMetadataRunner {
                executor: cx.background_executor.clone(),
            };
            let mut task = cx
                .background_executor
                .spawn(async move { runner.run(request).await });
            let error = poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect_err("Missing offline dependency must fail explicitly");
            assert!(
                format!("{error:#}").contains("no matching package named"),
                "Unexpected missing-dependency failure: {error:#}"
            );
            assert_eq!(
                actual_policy_fixture_requests().expect("Offline registry observations"),
                before,
                "Current metadata owner attempted registry access"
            );
            fixture
                .unchanged_installation()
                .expect("Dependency failure installs no toolchain");
            println!(
                "actual-registry version={version} negative_requests={} guarded_requests=0",
                before.len()
            );
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[gpui::test]
    #[ignore = "requires checksum-pinned official Rustup fixtures, rejecting loopback observer and absent-policy sealing correction"]
    fn cargo_workspace_passive_actual_manager_absent_policy_mutation(cx: &mut TestAppContext) {
        use std::{io::Read as _, os::unix::fs::PermissionsExt as _};
        cx.executor().allow_parking();
        for version in ["1.27.1", "1.28.0"] {
            for (name, value) in [("RUSTUP_AUTO_INSTALL", "1"), ("CARGO_NET_OFFLINE", "false")] {
                for table in [true, false] {
                    let fixture = ActualPolicyFixture::new(version, true, false)
                        .expect("Actual policy mutation fixture");
                    let wrapper = fixture.root.join("reviewed-wrapper");
                    let marker = fixture.root.join("wrapper-policy.log");
                    std::fs::write(&wrapper, "#!/bin/sh\nprintf '%s:%s\\n' \"$RUSTUP_AUTO_INSTALL\" \"$CARGO_NET_OFFLINE\" >> \"$RUST_POLICY_WRAPPER_LOG\" || exit 86\n[ \"$RUSTUP_AUTO_INSTALL\" = 0 ] && [ \"$CARGO_NET_OFFLINE\" = true ] || exit 87\nexec \"$@\"\n").expect("Actual policy observer wrapper");
                    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))
                        .expect("Executable wrapper");
                    let root = fixture.root.clone();
                    let mut environment = fixture.environment.clone();
                    environment.insert(
                        "RUSTC_WRAPPER".into(),
                        wrapper.to_string_lossy().into_owned(),
                    );
                    environment.insert(
                        "RUST_POLICY_WRAPPER_LOG".into(),
                        marker.to_string_lossy().into_owned(),
                    );
                    let build_directory = fixture.build_directory_option();
                    let executor = cx.background_executor.clone();
                    let before =
                        actual_policy_fixture_requests().expect("Before actual policy mutation");
                    let mut task = cx.background_executor.spawn(async move {
                        let mut command = util::command::new_command(root.join("bin/cargo"));
                        command
                            .envs(&environment)
                            .current_dir(&root)
                            .kill_on_drop(true);
                        let options = prevent_passive_rust_downloads(
                            &mut command,
                            false,
                            &root,
                            Some(&environment),
                            &executor,
                        )
                        .await?;
                        let path = root.join(".cargo/config.toml");
                        let mut configuration: toml::Value =
                            toml::from_str(&std::fs::read_to_string(&path)?)?;
                        let variables = configuration
                            .get_mut("env")
                            .and_then(toml::Value::as_table_mut)
                            .context("Fixture environment table missing")?;
                        variables.insert(
                            name.into(),
                            if table {
                                toml::Value::Table(toml::Table::from_iter([
                                    ("value".into(), toml::Value::String(value.into())),
                                    ("force".into(), toml::Value::Boolean(true)),
                                ]))
                            } else {
                                toml::Value::String(value.into())
                            },
                        );
                        std::fs::write(path, toml::to_string(&configuration)?)?;
                        command
                            .args(options)
                            .args([
                                "--config",
                                &build_directory,
                                "check",
                                "--offline",
                                "--target-dir",
                            ])
                            .arg(root.join("target"));
                        bounded_cargo_command_output(
                            command,
                            "Actual manager absent-policy mutation",
                            MAX_CARGO_METADATA_BYTES,
                            CARGO_METADATA_TIMEOUT,
                            &executor,
                        )
                        .await
                    });
                    let output = poll_cargo_process(cx, || (&mut task).now_or_never())
                        .expect("Actual policy mutation result");
                    if table {
                        assert!(
                            output.status.success(),
                            "Actual admitted policy must survive forced {name}: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                        let mut bytes = Vec::new();
                        std::fs::File::open(&marker)
                            .expect("Actual wrapper observations")
                            .take(64 * 1024 + 1)
                            .read_to_end(&mut bytes)
                            .expect("Bounded observations");
                        assert!(!bytes.is_empty() && bytes.len() <= 64 * 1024);
                        let observed =
                            std::str::from_utf8(&bytes).expect("Wrapper observation UTF-8");
                        assert!(observed.lines().all(|line| line == "0:true"), "{observed}");
                    } else {
                        let diagnostic = String::from_utf8_lossy(&output.stderr);
                        assert!(
                            !output.status.success()
                                && diagnostic.contains("failed to merge key `env`")
                                && diagnostic.contains(&format!("failed to merge key `{name}`"))
                                && diagnostic.contains("expected string, but found table"),
                            "Actual Cargo must reject the precise string/table merge: {diagnostic}"
                        );
                        assert!(
                            !marker.exists(),
                            "Type-changing mutation must not launch a compiler wrapper"
                        );
                    }
                    assert_eq!(
                        actual_policy_fixture_requests().expect("After actual policy mutation"),
                        before,
                        "Actual manager mutation attempted a download"
                    );
                    fixture
                        .unchanged_installation()
                        .expect("Mutation installed no toolchain");
                    println!(
                        "actual-policy-mutation version={version} key={name} table={table} requests=0"
                    );
                }
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_checks_forced_child_executables(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;

        cx.executor().allow_parking();
        for name in [
            "RUSTC",
            "CARGO_BUILD_RUSTC",
            "RUSTC_WRAPPER",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            "RUSTDOC",
            "CARGO_BUILD_RUSTDOC",
        ] {
            for (selection, force) in [
                ("installed-fixture", true),
                ("absent-fixture", true),
                ("absent-fixture", false),
            ] {
                let fixture = tempfile::tempdir().expect("Child executable admission fixture");
                let directory = std::fs::canonicalize(fixture.path()).expect("Canonical fixture");
                let root = directory.join("installed compiler");
                std::fs::create_dir_all(root.join("bin")).expect("Installed compiler directory");
                for binary in ["rustc", "cargo", "rustdoc"] {
                    std::fs::write(root.join("bin").join(binary), b"installed")
                        .expect("Installed binary fixture");
                }
                let mut environment = passive_rustup_proxy_fixture(&directory, "copy", "1.27.1");
                std::fs::copy(directory.join("rustup"), directory.join("rustdoc"))
                    .expect("Documentation proxy fixture");
                let compiler = directory.join("direct-compiler");
                std::fs::write(&compiler, "#!/bin/sh\nprintf 'direct compiler\\n'\n")
                    .expect("Direct compiler fixture");
                std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700))
                    .expect("Executable direct compiler");
                for variable in [
                    "RUSTC",
                    "CARGO_BUILD_RUSTC",
                    "RUSTDOC",
                    "CARGO_BUILD_RUSTDOC",
                ] {
                    environment.insert(variable.into(), compiler.to_string_lossy().into_owned());
                }
                for variable in [
                    "RUSTC_WRAPPER",
                    "CARGO_BUILD_RUSTC_WRAPPER",
                    "RUSTC_WORKSPACE_WRAPPER",
                    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
                ] {
                    environment.insert(variable.into(), String::new());
                }
                environment.insert("RUSTUP_TOOLCHAIN".into(), selection.into());
                environment.insert(
                    "PASSIVE_TEST_INSTALLED_ROOT".into(),
                    root.to_string_lossy().into_owned(),
                );
                environment.insert(
                    "CARGO_HOME".into(),
                    directory.join("cargo-home").to_string_lossy().into_owned(),
                );
                let include_rustdoc = name.contains("RUSTDOC");
                let proxy = directory.join(if include_rustdoc { "rustdoc" } else { "rustc" });
                std::fs::create_dir(directory.join(".cargo")).expect("Configuration directory");
                std::fs::write(
                    directory.join(".cargo/config.toml"),
                    format!(
                        "[env]\n{name}={{value={},force={force}}}\n",
                        toml::Value::String(proxy.to_string_lossy().into_owned()),
                    ),
                )
                .expect("Forced child executable configuration");
                let executor = cx.background_executor.clone();
                let selected_root = root.clone();
                let expected_child = if force {
                    proxy.into_os_string()
                } else {
                    environment.get(name).expect("Explicit parent value").into()
                };
                let mut task = cx.background_executor.spawn(async move {
                    let execution = prepare_passive_rust_execution(
                        Some(OsStr::new(env!("CARGO"))),
                        include_rustdoc,
                        &directory,
                        Some(&environment),
                        &executor,
                    )
                    .await?;
                    assert_eq!(execution.compiler_chain, [compiler.into_os_string()]);
                    assert_eq!(
                        execution.child_environment.value(name),
                        Some(expected_child)
                    );
                    assert_eq!(execution.toolchain, force.then(|| selected_root.clone()));
                    if force {
                        assert_eq!(
                            execution.child_environment.value("RUSTUP_TOOLCHAIN"),
                            Some(selected_root.into_os_string())
                        );
                    }
                    anyhow::Ok(())
                });
                let result = poll_cargo_process(cx, || (&mut task).now_or_never());
                if selection == "installed-fixture" || !force {
                    result.expect("Installed configured child proxy is pinned without replacing the parent compiler");
                } else {
                    let error = result
                        .expect_err("Missing configured child proxy must fail before collection");
                    assert!(
                        !format!("{error:#}")
                            .contains(&fixture.path().to_string_lossy().to_string())
                    );
                }
                assert!(
                    !fixture.path().join("collector-launched").exists(),
                    "Admission performs read-only policy queries"
                );
            }
        }
    }

    #[test]
    fn cargo_workspace_passive_parent_snapshot_preserves_present_empty_and_absent_values() {
        let mut environment = PassiveRustEnvironment {
            overrides: HashMap::from_iter([
                (
                    OsString::from("HOME"),
                    OsString::from("fixture parent home"),
                ),
                (OsString::from("RUSTC_WORKSPACE_WRAPPER"), OsString::new()),
                (OsString::from("UNRELATED"), OsString::from("untouched")),
            ]),
            inherit_process: false,
        };
        let pinned = environment.pin_present_cargo_parent();
        assert!(
            pinned.inherit_process,
            "parent pinning must not clear unrelated inherited values"
        );
        assert!(!environment.inherit_process);
        assert_eq!(pinned.overrides.len(), 2);
        assert!(
            pinned.overrides.get(OsStr::new("HOME"))
                == Some(&OsString::from("fixture parent home"))
        );
        assert!(
            pinned.overrides.get(OsStr::new("RUSTC_WORKSPACE_WRAPPER")) == Some(&OsString::new())
        );
        assert!(!pinned.overrides.contains_key(OsStr::new("RUSTC")));
        assert!(!pinned.overrides.contains_key(OsStr::new("UNRELATED")));
        assert!(environment.value("RUSTC").is_none());
        assert!(environment.value("UNRELATED") == Some(OsString::from("untouched")));
        environment.insert("HOME", "later value".into());
        assert!(
            pinned.overrides.get(OsStr::new("HOME"))
                == Some(&OsString::from("fixture parent home"))
        );
        let inherited_path =
            std::env::var_os("PATH").expect("qualification provides inherited PATH");
        let mut inherited = PassiveRustEnvironment::default();
        let pinned = inherited.pin_present_cargo_parent();
        assert!(
            pinned.overrides.get(OsStr::new("PATH")) == Some(&inherited_path),
            "inherited Some must become an explicit parent override"
        );
        assert!(inherited.overrides.get(OsStr::new("PATH")) == Some(&inherited_path));
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt as _;
            let value = OsString::from_vec(vec![b'/', b'h', 0xff]);
            let mut environment = PassiveRustEnvironment {
                overrides: HashMap::from_iter([(OsString::from("HOME"), value.clone())]),
                inherit_process: false,
            };
            let pinned = environment.pin_present_cargo_parent();
            assert!(pinned.overrides.get(OsStr::new("HOME")) == Some(&value));
            assert!(environment.value("HOME") == Some(value));
        }
        #[cfg(windows)]
        {
            let mut environment = PassiveRustEnvironment {
                overrides: HashMap::from_iter([(
                    OsString::from("Home"),
                    OsString::from("fixture mixed-case home"),
                )]),
                inherit_process: false,
            };
            let pinned = environment.pin_present_cargo_parent();
            assert!(
                pinned.overrides.get(OsStr::new("HOME"))
                    == Some(&OsString::from("fixture mixed-case home"))
            );
            assert!(!environment.overrides.contains_key(OsStr::new("Home")));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_seals_unconfigured_present_environment_after_admission(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        let cargo = PathBuf::from(env!("CARGO"));
        let compiler = cargo
            .parent()
            .expect("installed Cargo directory")
            .join("rustc");
        assert!(compiler.is_file(), "installed compiler required");
        for name in ["HOME", "RUSTC"] {
            for shape in ["unchanged", "table", "string"] {
                let fixture = tempfile::tempdir().expect("present parent fixture");
                let directory = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
                std::fs::create_dir_all(directory.join("src")).expect("source directory");
                std::fs::create_dir(directory.join(".cargo")).expect("configuration directory");
                std::fs::write(directory.join("Cargo.toml"), "[package]\nname='present_parent_seal_fixture'\nversion='0.1.0'\nedition='2024'\n").expect("dependency-free manifest");
                std::fs::write(
                    directory.join("src/lib.rs"),
                    "pub fn retained() -> u32 { 7 }\n",
                )
                .expect("fixture source");
                let configuration_path = directory.join(".cargo/config.toml");
                let unrelated = "[env]\nPRESENT_TEST_UNRELATED='from-config'\n";
                std::fs::write(&configuration_path, unrelated)
                    .expect("protected configuration absent");
                let wrapper = directory.join("reviewed-wrapper");
                let observed = directory.join("observed");
                let forbidden = directory.join("forbidden");
                std::fs::write(&wrapper, "#!/bin/sh\nif [ \"$HOME\" != \"$PRESENT_TEST_HOME\" ] || [ \"$RUSTC\" != \"$PRESENT_TEST_RUSTC\" ] || [ \"$PRESENT_TEST_UNRELATED\" != from-config ]; then printf 'child environment changed\\n' > \"$PRESENT_TEST_FORBIDDEN\"; exit 87; fi\nprintf 'retained\\n' >> \"$PRESENT_TEST_OBSERVED\" || exit 86\nshift\nexec \"$RUSTC\" \"$@\"\n").expect("bounded observer wrapper");
                std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))
                    .expect("wrapper executable");
                let home = directory
                    .join("admitted-home")
                    .to_string_lossy()
                    .into_owned();
                let compiler_value = compiler.to_string_lossy().into_owned();
                let environment = HashMap::from_iter([
                    ("HOME".into(), home.clone()),
                    ("RUSTC".into(), compiler_value.clone()),
                    (
                        "RUSTC_WRAPPER".into(),
                        wrapper.to_string_lossy().into_owned(),
                    ),
                    ("RUSTC_WORKSPACE_WRAPPER".into(), String::new()),
                    (
                        "CARGO_HOME".into(),
                        directory.join("cargo-home").to_string_lossy().into_owned(),
                    ),
                    ("PATH".into(), "/usr/bin:/bin".into()),
                    ("PRESENT_TEST_HOME".into(), home),
                    ("PRESENT_TEST_RUSTC".into(), compiler_value),
                    (
                        "PRESENT_TEST_OBSERVED".into(),
                        observed.to_string_lossy().into_owned(),
                    ),
                    (
                        "PRESENT_TEST_FORBIDDEN".into(),
                        forbidden.to_string_lossy().into_owned(),
                    ),
                ]);
                let cargo = cargo.clone();
                let executor = cx.background_executor.clone();
                let mut task = cx.background_executor.spawn(async move {
                    let mut command = util::command::new_command(cargo);
                    command
                        .envs(&environment)
                        .current_dir(&directory)
                        .kill_on_drop(true);
                    let options = prevent_passive_rust_downloads(
                        &mut command,
                        false,
                        &directory,
                        Some(&environment),
                        &executor,
                    )
                    .await?;
                    if shape != "unchanged" {
                        let value = toml::Value::String("unadmitted replacement".into());
                        std::fs::write(
                            configuration_path,
                            if shape == "table" {
                                format!("{unrelated}{name}={{value={value},force=true}}\n")
                            } else {
                                format!("{unrelated}{name}={value}\n")
                            },
                        )?;
                    }
                    command
                        .args(options)
                        .arg("--config")
                        .arg(format!(
                            "build.build-dir={}",
                            toml::Value::String(
                                directory.join("build").to_string_lossy().into_owned()
                            )
                        ))
                        .args(["check", "--offline", "--target-dir"])
                        .arg(directory.join("target"));
                    bounded_cargo_command_output(
                        command,
                        "Actual Cargo present parent mutation",
                        MAX_CARGO_METADATA_BYTES,
                        CARGO_METADATA_TIMEOUT,
                        &executor,
                    )
                    .await
                });
                let output = poll_cargo_process(cx, || (&mut task).now_or_never())
                    .expect("bounded actual Cargo result");
                assert!(
                    !forbidden.exists(),
                    "protected or unrelated child value changed"
                );
                if shape == "string" {
                    assert!(!output.status.success(), "type conflict must refuse");
                    assert!(!observed.exists(), "conflict must precede compiler child");
                    let diagnostic = String::from_utf8_lossy(&output.stderr);
                    assert!(
                        diagnostic.contains("failed to merge key `env`")
                            && diagnostic.contains(&format!("failed to merge key `{name}`")),
                        "expected key-only merge refusal"
                    );
                } else {
                    assert!(
                        output.status.success(),
                        "admitted present parent and unrelated configuration must still compile"
                    );
                    let observed =
                        std::fs::read_to_string(observed).expect("actual wrapper observation");
                    assert!(
                        !observed.is_empty() && observed.lines().all(|line| line == "retained")
                    );
                }
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_seals_absent_policy_flags_after_admission(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        let cargo = PathBuf::from(env!("CARGO"));
        let compiler = cargo
            .parent()
            .expect("Installed Cargo directory")
            .join("rustc");
        assert!(compiler.is_file(), "Actual installed compiler is required");
        for (name, forbidden) in [("RUSTUP_AUTO_INSTALL", "1"), ("CARGO_NET_OFFLINE", "false")] {
            for shape in ["table", "string"] {
                let fixture = tempfile::tempdir().expect("Absent policy mutation fixture");
                let directory = std::fs::canonicalize(fixture.path()).expect("Canonical fixture");
                std::fs::create_dir_all(directory.join("src")).expect("Fixture source");
                std::fs::create_dir(directory.join(".cargo")).expect("Fixture configuration");
                std::fs::write(directory.join("Cargo.toml"), "[package]\nname='absent_policy_seal_fixture'\nversion='0.1.0'\nedition='2024'\n").expect("Dependency-free manifest");
                std::fs::write(
                    directory.join("src/lib.rs"),
                    "pub fn retained() -> u32 { 7 }\n",
                )
                .expect("Fixture implementation");
                let marker = directory.join("wrapper-observed");
                let wrapper = directory.join("reviewed-wrapper");
                std::fs::write(&wrapper, "#!/bin/sh\nprintf '%s:%s\\n' \"$RUSTUP_AUTO_INSTALL\" \"$CARGO_NET_OFFLINE\" >> \"$POLICY_TEST_OBSERVER\" || exit 86\n[ \"$RUSTUP_AUTO_INSTALL\" = 0 ] && [ \"$CARGO_NET_OFFLINE\" = true ] || exit 87\nexec \"$@\"\n").expect("Policy observer wrapper");
                std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))
                    .expect("Executable wrapper");
                let configuration_path = directory.join(".cargo/config.toml");
                std::fs::write(&configuration_path, "").expect("Initially absent policy flags");
                let environment = HashMap::from_iter([
                    ("RUSTC".into(), compiler.to_string_lossy().into_owned()),
                    (
                        "RUSTC_WRAPPER".into(),
                        wrapper.to_string_lossy().into_owned(),
                    ),
                    ("RUSTC_WORKSPACE_WRAPPER".into(), String::new()),
                    (
                        "CARGO_HOME".into(),
                        directory.join("cargo-home").to_string_lossy().into_owned(),
                    ),
                    ("PATH".into(), "/usr/bin:/bin".into()),
                    (
                        "CARGO_BUILD_BUILD_DIR".into(),
                        directory.join("build").to_string_lossy().into_owned(),
                    ),
                    (
                        "POLICY_TEST_OBSERVER".into(),
                        marker.to_string_lossy().into_owned(),
                    ),
                ]);
                let cargo = cargo.clone();
                let executor = cx.background_executor.clone();
                let mut task = cx.background_executor.spawn(async move {
                    let mut command = util::command::new_command(cargo);
                    command
                        .envs(&environment)
                        .current_dir(&directory)
                        .kill_on_drop(true);
                    let options = prevent_passive_rust_downloads(
                        &mut command,
                        false,
                        &directory,
                        Some(&environment),
                        &executor,
                    )
                    .await?;
                    let value = toml::Value::String(forbidden.into()).to_string();
                    std::fs::write(
                        configuration_path,
                        if shape == "table" {
                            format!("[env]\n{name}={{value={value},force=true}}\n")
                        } else {
                            format!("[env]\n{name}={value}\n")
                        },
                    )?;
                    command
                        .args(options)
                        .arg("--config")
                        .arg(format!(
                            "build.build-dir={}",
                            toml::Value::String(
                                directory.join("build").to_string_lossy().into_owned()
                            )
                        ))
                        .args(["check", "--offline", "--target-dir"])
                        .arg(directory.join("target"));
                    bounded_cargo_command_output(
                        command,
                        "Actual Cargo absent policy mutation",
                        MAX_CARGO_METADATA_BYTES,
                        CARGO_METADATA_TIMEOUT,
                        &executor,
                    )
                    .await
                });
                let output = poll_cargo_process(cx, || (&mut task).now_or_never())
                    .expect("Bounded actual Cargo result");
                if shape == "table" {
                    assert!(
                        output.status.success(),
                        "Admitted policy survives new forced {name}: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    let observed = std::fs::read_to_string(&marker)
                        .expect("Actual compiler wrapper observations");
                    assert!(!observed.is_empty());
                    assert!(observed.lines().all(|line| line == "0:true"), "{observed}");
                } else {
                    assert!(
                        !output.status.success(),
                        "Type-changing configuration must reject"
                    );
                    let diagnostic = String::from_utf8_lossy(&output.stderr);
                    assert!(
                        diagnostic.contains("failed to merge key `env`")
                            && diagnostic.contains(&format!("failed to merge key `{name}`"))
                            && diagnostic.contains("expected string, but found table"),
                        "Cargo must reject this env.{name} string/table merge: {diagnostic}"
                    );
                    assert!(
                        !marker.exists(),
                        "Type conflict must fail before any compiler child"
                    );
                }
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_seals_forced_compiler_after_admission(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        let fixture = tempfile::tempdir().expect("Cargo admission mutation fixture");
        let directory = std::fs::canonicalize(fixture.path()).expect("Canonical fixture");
        let cargo = PathBuf::from(env!("CARGO"));
        let compiler = cargo
            .parent()
            .expect("Installed Cargo directory")
            .join("rustc");
        assert!(
            compiler.is_file(),
            "The actual installed compiler is required"
        );
        std::fs::create_dir_all(directory.join("src")).expect("Fixture source");
        std::fs::create_dir_all(directory.join(".cargo")).expect("Fixture configuration");
        std::fs::write(
            directory.join("Cargo.toml"),
            "[package]\nname='rust_passive_configuration_seal_fixture'\nversion='0.1.0'\nedition='2024'\n",
        )
        .expect("Dependency-free manifest");
        std::fs::write(
            directory.join("src/lib.rs"),
            "pub fn retained() -> u32 { 7 }\n",
        )
        .expect("Fixture implementation");
        let wrapper = directory.join("reviewed-wrapper");
        let unreviewed_compiler = directory.join("unreviewed-compiler");
        for (path, contents) in [
            (&wrapper, "#!/bin/sh\nshift\nexec \"$RUSTC\" \"$@\"\n"),
            (
                &unreviewed_compiler,
                "#!/bin/sh\nprintf 'unexpected compiler launch\\n' > \"$RUST_WORKSPACE_FORBIDDEN_MARKER\"\nexit 81\n",
            ),
        ] {
            std::fs::write(path, contents).expect("Controlled compiler/wrapper");
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .expect("Executable fixture");
        }
        let configuration_path = directory.join(".cargo/config.toml");
        let configuration = |compiler: &Path| {
            format!(
                "[env]\nRUSTC={{value={},force=true}}\n",
                serde_json::to_string(compiler.to_str().expect("Fixture compiler path"))
                    .expect("Quoted compiler")
            )
        };
        std::fs::write(&configuration_path, configuration(&compiler))
            .expect("Admitted compiler configuration");
        let marker = directory.join("unreviewed-compiler-launched");
        let environment = HashMap::from_iter([
            ("RUSTC".into(), compiler.to_string_lossy().into_owned()),
            (
                "RUSTC_WRAPPER".into(),
                wrapper.to_string_lossy().into_owned(),
            ),
            ("RUSTC_WORKSPACE_WRAPPER".into(), String::new()),
            (
                "CARGO_HOME".into(),
                directory.join("cargo-home").to_string_lossy().into_owned(),
            ),
            ("PATH".into(), "/usr/bin:/bin".into()),
            (
                "RUST_WORKSPACE_FORBIDDEN_MARKER".into(),
                marker.to_string_lossy().into_owned(),
            ),
        ]);
        let mutated_configuration = configuration(&unreviewed_compiler);
        let executor = cx.background_executor.clone();
        let mut task = cx.background_executor.spawn(async move {
            let mut command = util::command::new_command(cargo);
            command
                .envs(&environment)
                .current_dir(&directory)
                .kill_on_drop(true);
            let options = prevent_passive_rust_downloads(
                &mut command,
                false,
                &directory,
                Some(&environment),
                &executor,
            )
            .await?;
            std::fs::write(configuration_path, mutated_configuration)?;
            command
                .args(options)
                .args(["check", "--offline", "--target-dir"])
                .arg(directory.join("target"));
            bounded_cargo_command_output(
                command,
                "Actual Cargo admission mutation fixture",
                MAX_CARGO_METADATA_BYTES,
                CARGO_METADATA_TIMEOUT,
                &executor,
            )
            .await
        });
        let output = poll_cargo_process(cx, || (&mut task).now_or_never())
            .expect("Bounded actual Cargo result");
        assert!(
            !marker.exists(),
            "Cargo launched an unadmitted compiler after configuration changed"
        );
        assert!(
            output.status.success(),
            "The admitted compiler/wrapper must still run: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_seals_build_executables_after_admission(
        cx: &mut TestAppContext,
    ) {
        use std::{
            io::Read as _,
            os::unix::fs::{MetadataExt as _, PermissionsExt as _},
        };
        cx.executor().allow_parking();
        let cargo = PathBuf::from(env!("CARGO"));
        let compiler_directory = cargo.parent().expect("Installed Cargo directory");
        let compiler = compiler_directory.join("rustc");
        let documentation_generator = compiler_directory.join("rustdoc");
        assert!(compiler.is_file() && documentation_generator.is_file());
        let mut failures = Vec::new();
        for field in [
            "rustc",
            "rustdoc",
            "rustc-wrapper",
            "rustc-workspace-wrapper",
        ] {
            let fixture = tempfile::tempdir().expect("Cargo executable mutation fixture");
            let directory = std::fs::canonicalize(fixture.path()).expect("Canonical fixture");
            std::fs::create_dir_all(directory.join("src")).expect("Fixture source");
            std::fs::create_dir_all(directory.join(".cargo")).expect("Fixture configuration");
            std::fs::write(
                directory.join("Cargo.toml"),
                "[package]\nname='rust_passive_build_seal_fixture'\nversion='0.1.0'\nedition='2024'\n",
            ).expect("Dependency-free manifest");
            std::fs::write(
                directory.join("src/lib.rs"),
                "pub fn retained() -> u32 { 7 }\n",
            )
            .expect("Fixture implementation");
            let wrapper = directory.join("reviewed-wrapper");
            let unreviewed = directory.join("unreviewed-executable");
            let wrapper_observer = directory.join("reviewed-wrapper-invocations");
            for (path, contents) in [
                (
                    &wrapper,
                    "#!/bin/sh\n{ printf 'wrapper=%s\\n' \"$0\"; for argument do printf 'argument=%s\\n' \"$argument\"; done; printf 'end wrapper\\n'; } >> \"$RUST_WORKSPACE_WRAPPER_OBSERVER\" || exit 79\nexec \"$@\"\n",
                ),
                (
                    &unreviewed,
                    "#!/bin/sh\nprintf 'unexpected executable launch\\n' > \"$RUST_WORKSPACE_FORBIDDEN_MARKER\"\nexit 81\n",
                ),
            ] {
                std::fs::write(path, contents).expect("Controlled executable");
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .expect("Executable fixture");
            }
            let mut configuration = toml::Table::new();
            for (name, path) in [
                ("rustc", &compiler),
                ("rustdoc", &documentation_generator),
                ("rustc-wrapper", &wrapper),
                ("rustc-workspace-wrapper", &wrapper),
            ] {
                configuration.insert(
                    name.into(),
                    toml::Value::String(path.to_string_lossy().into_owned()),
                );
            }
            let configuration_path = directory.join(".cargo/config.toml");
            let serialize = |configuration| {
                toml::to_string(&toml::Table::from_iter([(
                    "build".into(),
                    toml::Value::Table(configuration),
                )]))
                .expect("Build configuration")
            };
            std::fs::write(&configuration_path, serialize(configuration.clone()))
                .expect("Admitted executable configuration");
            configuration.insert(
                field.into(),
                toml::Value::String(unreviewed.to_string_lossy().into_owned()),
            );
            let mutated_configuration = serialize(configuration);
            let marker = directory.join("unreviewed-executable-launched");
            let diagnostic_paths = [
                ("cargo", cargo.clone()),
                ("rustc", compiler.clone()),
                ("rustdoc", documentation_generator.clone()),
                ("reviewed-wrapper", wrapper.clone()),
                ("unreviewed-executable", unreviewed.clone()),
            ];
            let environment = HashMap::from_iter([
                (
                    "CARGO_HOME".into(),
                    directory.join("cargo-home").to_string_lossy().into_owned(),
                ),
                ("PATH".into(), "/usr/bin:/bin".into()),
                (
                    "RUST_WORKSPACE_WRAPPER_OBSERVER".into(),
                    wrapper_observer.to_string_lossy().into_owned(),
                ),
                (
                    "RUST_WORKSPACE_FORBIDDEN_MARKER".into(),
                    marker.to_string_lossy().into_owned(),
                ),
            ]);
            let executor = cx.background_executor.clone();
            let cargo = cargo.clone();
            let include_rustdoc = field == "rustdoc";
            let mut task = cx.background_executor.spawn(async move {
                let executable_identities = || {
                    diagnostic_paths.iter().map(|(name, path)| {
                        match std::fs::metadata(path) {
                            Ok(metadata) => serde_json::json!({
                                "name": name, "path": path, "device": metadata.dev(),
                                "inode": metadata.ino(), "bytes": metadata.len(),
                                "mode": metadata.mode(), "modified_seconds": metadata.mtime(),
                                "modified_nanoseconds": metadata.mtime_nsec(),
                            }),
                            Err(error) => serde_json::json!({
                                "name": name, "path": path, "error": error.to_string(),
                            }),
                        }
                    }).collect::<Vec<_>>()
                };
                let configuration_before_admission = std::fs::read_to_string(&configuration_path)?;
                let executables_before_admission = executable_identities();
                let mut command = util::command::new_command(&cargo);
                command
                    .envs(&environment)
                    .current_dir(&directory)
                    .kill_on_drop(true);
                let options = prevent_passive_rust_downloads(
                    &mut command,
                    include_rustdoc,
                    &directory,
                    Some(&environment),
                    &executor,
                )
                .await?;
                let admitted_options = options.iter().map(|option| option.to_string_lossy().into_owned()).collect::<Vec<_>>();
                let configuration_after_admission = std::fs::read_to_string(&configuration_path)?;
                std::fs::write(&configuration_path, mutated_configuration)?;
                let configuration_before_spawn = std::fs::read_to_string(&configuration_path)?;
                let executables_before_spawn = executable_identities();
                command
                    .args(options)
                    .args([
                        if include_rustdoc { "doc" } else { "check" },
                        "--offline",
                        "--target-dir",
                    ])
                    .arg(directory.join("target"));
                let arguments = command.get_args().map(|argument| argument.to_string_lossy().into_owned()).collect::<Vec<_>>();
                let output = bounded_cargo_command_output(
                    command,
                    "Actual Cargo executable mutation fixture",
                    MAX_CARGO_METADATA_BYTES,
                    CARGO_METADATA_TIMEOUT,
                    &executor,
                )
                .await?;
                let wrapper_invocations = match std::fs::File::open(&wrapper_observer) {
                    Ok(file) => {
                        let mut bytes = Vec::new();
                        match file.take(16 * 1024 + 1).read_to_end(&mut bytes) {
                            Ok(_) => {
                                let truncated = bytes.len() > 16 * 1024;
                                bytes.truncate(16 * 1024);
                                serde_json::json!({
                                    "trace": String::from_utf8_lossy(&bytes), "truncated": truncated,
                                })
                            }
                            Err(error) => serde_json::json!({"read_error": error.to_string()}),
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        serde_json::json!({"absent": true})
                    }
                    Err(error) => serde_json::json!({"open_error": error.to_string()}),
                };
                let diagnostic = serde_json::json!({
                    "field": field, "cargo": cargo, "admitted_options": admitted_options,
                    "arguments": arguments, "request_environment": environment,
                    "configuration_before_admission": configuration_before_admission,
                    "configuration_after_admission": configuration_after_admission,
                    "configuration_before_spawn": configuration_before_spawn,
                    "executables_before_admission": executables_before_admission,
                    "executables_before_spawn": executables_before_spawn,
                    "executables_after_completion": executable_identities(),
                    "wrapper_invocations": wrapper_invocations,
                });
                anyhow::Ok((output, diagnostic))
            });
            let (output, diagnostic) = poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect("Bounded actual Cargo result");
            println!(
                "build.{field}: unadmitted_launch={}, status={}",
                marker.exists(),
                output.status
            );
            if marker.exists() || !output.status.success() {
                println!("Cargo mutation failure diagnostic: {diagnostic}");
                failures.push(format!(
                    "build.{field}: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "Cargo executable mutation failures: {failures:#?}"
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_matches_compiler_probe(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        let fixture = tempfile::tempdir().expect("Cargo configuration fixture");
        let directory = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
        let home = directory.join("home");
        let project = directory.join("project");
        std::fs::create_dir_all(home.join(".cargo")).expect("home configuration");
        std::fs::create_dir_all(project.join(".cargo/fragments")).expect("project configuration");
        for (name, contents) in [
            (
                "compiler",
                "#!/bin/sh\nprintf '%s:%s:%s\\n' \"$1\" \"$FIXTURE_VALUE\" \"$RUSTUP_AUTO_INSTALL\"\n",
            ),
            ("wrapper", "#!/bin/sh\nexec \"$@\"\n"),
        ] {
            let path = project.join(name);
            std::fs::write(&path, contents).expect("compiler fixture");
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .expect("executable fixture");
        }
        std::fs::write(
            home.join(".cargo/config.toml"),
            "[env]\nFIXTURE_VALUE={value='home',force=true}\n",
        )
        .expect("home values");
        std::fs::write(project.join(".cargo/fragments/compiler.toml"), "[build]\nrustc='../compiler'\nrustc-wrapper='../wrapper'\n[env]\nFIXTURE_VALUE={value='included',force=true}\nRUSTUP_AUTO_INSTALL={value='1',force=true}\n").expect("included values");
        std::fs::write(project.join(".cargo/config"), "include=['fragments/compiler.toml',{path='absent.toml',optional=true}]\n[env]\nFIXTURE_VALUE={value='project'}\n").expect("preferred configuration");
        std::fs::write(
            project.join(".cargo/config.toml"),
            "invalid TOML ignored by Cargo",
        )
        .expect("lower priority configuration");
        let environment = HashMap::from_iter([
            (
                "CARGO_HOME".into(),
                home.join(".cargo").to_string_lossy().into_owned(),
            ),
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("HOME".into(), home.to_string_lossy().into_owned()),
            ("FIXTURE_VALUE".into(), "parent".into()),
        ]);
        let request = CargoConfigurationProbeRequest {
            working_directory: project.clone(),
            environment: Some(environment.clone()),
        };
        let probe = ProcessCargoConfigurationProbe {
            executor: cx.background_executor.clone(),
        };
        let mut task = cx
            .background_executor
            .spawn(async move { probe.run(request).await });
        assert_eq!(
            poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect("configured compiler and wrapper"),
            b"-vV:project:0\n"
        );
        std::fs::write(project.join("wrapper"), "#!/bin/sh\nif [ \"$1\" = -vV ]; then printf 'unexpected bare wrapper\\n'; exit 0; fi\nexec \"$@\"\n")
            .expect("Wrapper rejects an empty compiler but would accept missing operand");
        let mut empty_environment = environment.clone();
        empty_environment.insert("RUSTC".into(), String::new());
        let request = CargoConfigurationProbeRequest {
            working_directory: project.clone(),
            environment: Some(empty_environment),
        };
        let probe = ProcessCargoConfigurationProbe {
            executor: cx.background_executor.clone(),
        };
        let mut task = cx
            .background_executor
            .spawn(async move { probe.run(request).await });
        assert!(
            poll_cargo_process(cx, || (&mut task).now_or_never()).is_err(),
            "An explicitly empty compiler must not become a bare wrapper probe"
        );
        let mut expected = [
            "--config",
            "env.RUSTUP_AUTO_INSTALL.force=false",
            "--config",
            "env.RUSTUP_AUTO_INSTALL.value=\"0\"",
            "--config",
            "env.RUSTUP_AUTO_INSTALL.relative=false",
            "--config",
            "env.CARGO_NET_OFFLINE.force=false",
            "--config",
            "env.CARGO_NET_OFFLINE.value=\"true\"",
            "--config",
            "env.CARGO_NET_OFFLINE.relative=false",
            "--config",
            "env.PATH.force=false",
            "--config",
            "env.PATH.value=\"admitted-parent-present\"",
            "--config",
            "env.PATH.relative=false",
            "--config",
            "env.HOME.force=false",
            "--config",
            "env.HOME.value=\"admitted-parent-present\"",
            "--config",
            "env.HOME.relative=false",
        ]
        .map(OsString::from)
        .to_vec();
        for (field, value) in [
            (
                "rustc-wrapper",
                project
                    .join(".cargo/../wrapper")
                    .to_string_lossy()
                    .into_owned(),
            ),
            ("rustc-workspace-wrapper", String::new()),
            (
                "rustc",
                project
                    .join(".cargo/../compiler")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ] {
            expected.extend([
                OsString::from("--config"),
                format!("build.{field}={}", toml::Value::String(value)).into(),
            ]);
        }
        expected.extend(["test", "--doc", "--", "--list"].map(OsString::from));
        let executor = cx.background_executor.clone();
        let mut task = cx.background_executor.spawn(async move {
            let mut command = util::command::new_command("cargo");
            command.current_dir(&project).kill_on_drop(true);
            let options = prevent_passive_rust_downloads(
                &mut command,
                false,
                &project,
                Some(&environment),
                &executor,
            )
            .await?;
            command
                .args(options)
                .args(["test", "--doc", "--", "--list"]);
            anyhow::Ok(
                command
                    .get_args()
                    .map(OsStr::to_os_string)
                    .collect::<Vec<_>>(),
            )
        });
        assert_eq!(
            poll_cargo_process(cx, || (&mut task).now_or_never()).expect("policy prefix"),
            expected
        );
    }

    #[cfg(target_os = "linux")]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_probe_preserves_non_utf8_executable_path(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::{ffi::OsStringExt as _, fs::PermissionsExt as _};
        cx.executor().allow_parking();
        let fixture = tempfile::tempdir().expect("Non-UTF-8 compiler fixture");
        let directory = fixture
            .path()
            .join(OsString::from_vec(b"compiler-directory-\xff".to_vec()));
        std::fs::create_dir_all(directory.join(".cargo")).expect("Configuration directory");
        let compiler = directory.join("compiler");
        std::fs::write(
            &compiler,
            "#!/bin/sh\nprintf 'retained compiler path bytes\\n'\nprintf '%s' \"$RUSTC\"\n",
        )
        .expect("Controlled compiler");
        std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o700))
            .expect("Executable compiler");
        std::fs::write(
            directory.join(".cargo/config.toml"),
            "[build]\nrustc='./compiler'\n[env]\nRUSTC={value='./compiler',force=true,relative=true}\n",
        )
        .expect("Source-relative compiler and child environment configuration");
        let mut expected = b"retained compiler path bytes\n".to_vec();
        expected.extend_from_slice(directory.join("./compiler").as_os_str().as_encoded_bytes());
        let request = CargoConfigurationProbeRequest {
            working_directory: directory,
            environment: Some(HashMap::from_iter([
                (
                    "CARGO_HOME".into(),
                    fixture
                        .path()
                        .join("cargo-home")
                        .to_string_lossy()
                        .into_owned(),
                ),
                ("PATH".into(), "/usr/bin:/bin".into()),
            ])),
        };
        let probe = ProcessCargoConfigurationProbe {
            executor: cx.background_executor.clone(),
        };
        let mut task = cx
            .background_executor
            .spawn(async move { probe.run(request).await });
        assert_eq!(
            poll_cargo_process(cx, || (&mut task).now_or_never())
                .expect("Direct probe preserves executable and child environment path bytes"),
            expected
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_rejects_cycles_and_private_errors(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for contents in [
            "include=['config.toml']",
            "[build]\nrustc=['private-source-value']",
            "[env]\nRUSTUP_HOME={value='private-home-value',force=true}",
        ] {
            let fixture = tempfile::tempdir().expect("invalid configuration fixture");
            let directory = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
            std::fs::create_dir(directory.join(".cargo")).expect("configuration directory");
            std::fs::write(directory.join(".cargo/config.toml"), contents)
                .expect("invalid configuration");
            let environment = HashMap::from_iter([
                (
                    "CARGO_HOME".into(),
                    directory.join("home").to_string_lossy().into_owned(),
                ),
                ("PATH".into(), "/usr/bin:/bin".into()),
            ]);
            let executor = cx.background_executor.clone();
            let mut task = cx.background_executor.spawn(async move {
                prepare_passive_rust_execution(
                    None,
                    false,
                    &directory,
                    Some(&environment),
                    &executor,
                )
                .await
            });
            let error = poll_cargo_process(cx, || (&mut task).now_or_never())
                .err()
                .expect("invalid configuration rejected")
                .to_string();
            assert!(
                !error.contains("private-source-value")
                    && !error.contains("private-home-value")
                    && !error.contains(&fixture.path().to_string_lossy().to_string()),
                "{error}"
            );
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn cargo_workspace_passive_configuration_checks_selected_rustdoc(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        for mode in [
            "configuration",
            "rustdoc-env",
            "cargo-build-env",
            "path",
            "direct",
            "missing-binary",
        ] {
            for installed in [true, false] {
                let fixture = tempfile::tempdir().expect("documentation generator fixture");
                let directory = std::fs::canonicalize(fixture.path()).expect("canonical fixture");
                let root = directory.join("installed root");
                std::fs::create_dir_all(root.join("bin")).expect("installed root");
                for name in ["rustc", "cargo", "rustdoc"] {
                    if mode != "missing-binary" || name != "rustdoc" {
                        std::fs::write(root.join("bin").join(name), b"installed")
                            .expect("installed binary");
                    }
                }
                let root = std::fs::canonicalize(root).expect("canonical root");
                let mut environment = passive_rustup_proxy_fixture(&directory, "copy", "1.27.1");
                std::fs::copy(directory.join("rustc"), directory.join("rustdoc"))
                    .expect("Rustup documentation proxy");
                let direct = directory.join("direct");
                std::fs::create_dir(&direct).expect("direct binaries");
                for name in ["cargo", "rustc", "direct-rustdoc"] {
                    let path = direct.join(name);
                    std::fs::write(&path, "#!/bin/sh\nexit 0\n").expect("direct binary fixture");
                    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                        .expect("executable direct fixture");
                }
                environment.insert(
                    "PATH".into(),
                    std::env::join_paths([direct.as_path(), directory.as_path()])
                        .expect("fixture PATH")
                        .to_string_lossy()
                        .into_owned(),
                );
                environment.insert(
                    "RUSTC".into(),
                    direct.join("rustc").to_string_lossy().into_owned(),
                );
                environment.insert("RUSTC_WRAPPER".into(), String::new());
                environment.insert("RUSTC_WORKSPACE_WRAPPER".into(), String::new());
                environment.insert(
                    "CARGO_HOME".into(),
                    directory.join("home").to_string_lossy().into_owned(),
                );
                environment.insert(
                    "RUSTUP_TOOLCHAIN".into(),
                    if installed {
                        "installed-fixture"
                    } else {
                        "absent-fixture"
                    }
                    .into(),
                );
                environment.insert(
                    "PASSIVE_TEST_INSTALLED_ROOT".into(),
                    root.to_string_lossy().into_owned(),
                );
                std::fs::create_dir(directory.join(".cargo")).expect("configuration directory");
                let proxy = directory.join("rustdoc");
                let direct_rustdoc = direct.join("direct-rustdoc");
                let configured = if matches!(mode, "rustdoc-env" | "cargo-build-env") {
                    &direct_rustdoc
                } else {
                    &proxy
                };
                if mode != "path" {
                    std::fs::write(
                        directory.join(".cargo/config.toml"),
                        format!("[build]\nrustdoc={:?}\n", configured.to_string_lossy()),
                    )
                    .expect("documentation configuration");
                }
                match mode {
                    "rustdoc-env" => {
                        environment.insert("RUSTDOC".into(), proxy.to_string_lossy().into_owned());
                    }
                    "cargo-build-env" => {
                        environment.insert(
                            "CARGO_BUILD_RUSTDOC".into(),
                            proxy.to_string_lossy().into_owned(),
                        );
                    }
                    "direct" => {
                        environment.insert(
                            "RUSTDOC".into(),
                            direct_rustdoc.to_string_lossy().into_owned(),
                        );
                    }
                    _ => {}
                }
                let executor = cx.background_executor.clone();
                let working_directory = directory.clone();
                let mut task = cx.background_executor.spawn(async move {
                    let plain = prepare_passive_rust_execution(
                        Some(OsStr::new("cargo")),
                        false,
                        &working_directory,
                        Some(&environment),
                        &executor,
                    )
                    .await?;
                    assert!(
                        plain.toolchain.is_none(),
                        "compiler-only stages do not launch rustdoc"
                    );
                    assert!(!working_directory.join("policy-queries").exists());
                    prepare_passive_rust_execution(
                        Some(OsStr::new("cargo")),
                        true,
                        &working_directory,
                        Some(&environment),
                        &executor,
                    )
                    .await
                });
                let result = poll_cargo_process(cx, || (&mut task).now_or_never());
                if mode == "direct" {
                    assert!(
                        result
                            .expect("preserve direct documentation selection")
                            .toolchain
                            .is_none()
                    );
                    assert!(!directory.join("policy-queries").exists());
                } else if installed && mode != "missing-binary" {
                    assert_eq!(
                        result
                            .expect("verified selected documentation proxy")
                            .toolchain,
                        Some(root)
                    );
                } else {
                    assert!(
                        result.is_err(),
                        "uninstalled documentation selection or component rejected"
                    );
                }
                assert!(
                    !directory.join("collector-launched").exists(),
                    "admission only performs read-only probes"
                );
            }
        }
    }

    struct FakeConfigurationProbe {
        output: Result<Vec<u8>, String>,
        requests: Arc<Mutex<Vec<CargoConfigurationProbeRequest>>>,
    }

    #[async_trait]
    impl CargoConfigurationProbe for FakeConfigurationProbe {
        async fn run(&self, request: CargoConfigurationProbeRequest) -> Result<Vec<u8>> {
            self.requests
                .lock()
                .expect("configuration probe request lock should be available")
                .push(request);
            self.output.clone().map_err(anyhow::Error::msg)
        }
    }

    fn successful_configuration_probe() -> Arc<dyn CargoConfigurationProbe> {
        successful_configuration_probe_with_requests(Arc::new(Mutex::new(Vec::new())))
    }

    fn successful_configuration_probe_with_requests(
        requests: Arc<Mutex<Vec<CargoConfigurationProbeRequest>>>,
    ) -> Arc<dyn CargoConfigurationProbe> {
        Arc::new(FakeConfigurationProbe {
            output: Ok(b"rustc 1.90.0\nbinary: rustc\ncommit-hash: unknown\ncommit-date: unknown\nhost: x86_64-unknown-linux-gnu\nrelease: 1.90.0\nLLVM version: 20.1.0\n".to_vec()),
            requests,
        })
    }

    #[async_trait]
    impl CargoMetadataRunner for RoutingRunner {
        async fn run(&self, request: CargoMetadataRequest) -> Result<Vec<u8>> {
            self.requests
                .lock()
                .expect("routing runner request lock should be available")
                .push(request.manifest_path.clone());
            if request.manifest_path.starts_with("/workspace") {
                Ok(include_bytes!("../test_data/cargo_workspace/workspace-v1.json").to_vec())
            } else if request.manifest_path.starts_with("/standalone") {
                Ok(include_bytes!("../test_data/cargo_workspace/standalone-v1.json").to_vec())
            } else {
                Err(anyhow!(
                    "metadata failed for {}",
                    request.manifest_path.display()
                ))
            }
        }
    }

    fn candidate(
        worktree_id: usize,
        worktree_root: &str,
        manifest_path: &str,
        trusted: bool,
    ) -> Candidate {
        let relative = std::path::Path::new(manifest_path)
            .strip_prefix(worktree_root)
            .expect("fixture manifest should be inside its worktree");
        let relative = relative
            .to_str()
            .expect("fixture manifest should be valid UTF-8");
        let relative = RelPath::new(std::path::Path::new(relative), PathStyle::local())
            .expect("fixture manifest should be project relative");
        let relative: Arc<RelPath> = Arc::from(relative.as_ref());
        Candidate {
            project_path: ProjectPath {
                worktree_id: settings::WorktreeId::from_usize(worktree_id),
                path: relative,
            },
            absolute_path: PathBuf::from(manifest_path),
            worktree_root: PathBuf::from(worktree_root),
            environment: Task::ready(None).shared(),
            manifest_text: Task::ready(Ok(
                "[workspace]\nresolver = \"2\"\n[profile.ship]\ninherits = \"release\"\n"
                    .to_string(),
            )),
            lock_text: None,
            toolchain_text: None,
            trusted,
            private_paths: Arc::from([]),
        }
    }

    #[async_trait]
    impl CargoMetadataRunner for FakeRunner {
        async fn run(&self, request: CargoMetadataRequest) -> Result<Vec<u8>> {
            self.requests
                .lock()
                .expect("fake runner request lock should be available")
                .push(request);
            Ok(self.output.clone())
        }
    }

    #[gpui::test]
    async fn cargo_workspace_fake_runner_uses_candidate_environment(_cx: &mut TestAppContext) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let runner = Arc::new(FakeRunner {
            output: include_bytes!("../test_data/cargo_workspace/workspace-v1.json").to_vec(),
            requests: requests.clone(),
        });
        let environment = collections::HashMap::from_iter([(
            "CARGO_HOME".to_string(),
            "/isolated/cargo".to_string(),
        )]);
        let candidate = Candidate {
            project_path: ProjectPath {
                worktree_id: settings::WorktreeId::from_usize(1),
                path: Arc::from(RelPath::from_unix_str("Cargo.toml").expect("valid fixture path")),
            },
            absolute_path: PathBuf::from("/workspace/Cargo.toml"),
            worktree_root: PathBuf::from("/workspace"),
            environment: Task::ready(Some(environment.clone())).shared(),
            manifest_text: Task::ready(Ok("[workspace]\nresolver = \"2\"\n".to_string())),
            lock_text: None,
            toolchain_text: None,
            trusted: true,
            private_paths: Arc::from([]),
        };
        let snapshot = collect_snapshot(
            5,
            17,
            vec![candidate],
            runner,
            successful_configuration_probe(),
        )
        .await
        .expect("fake metadata should convert")
        .snapshot;
        assert_eq!(snapshot.revision, 5);
        assert_eq!(snapshot.input_fingerprint, 17);
        assert_eq!(snapshot.workspaces.len(), 1);
        assert!(snapshot.failures.is_empty());
        let requests = requests
            .lock()
            .expect("fake runner request lock should be available");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].environment.as_ref(), Some(&environment));
        assert_eq!(
            requests[0].manifest_path,
            PathBuf::from("/workspace/Cargo.toml")
        );
    }

    #[gpui::test]
    async fn cargo_workspace_collects_multiple_roots_skips_covered_members_and_keeps_failures(
        _cx: &mut TestAppContext,
    ) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let snapshot = collect_snapshot(
            9,
            27,
            vec![
                candidate(1, "/workspace", "/workspace/Cargo.toml", true),
                candidate(1, "/workspace", "/workspace/member-one/Cargo.toml", true),
                candidate(2, "/standalone", "/standalone/Cargo.toml", true),
                candidate(3, "/broken", "/broken/Cargo.toml", true),
            ],
            Arc::new(RoutingRunner {
                requests: requests.clone(),
            }),
            successful_configuration_probe(),
        )
        .await
        .expect("partial fixture collection should succeed")
        .snapshot;

        assert_eq!(snapshot.revision, 9);
        assert_eq!(snapshot.input_fingerprint, 27);
        assert_eq!(snapshot.completeness, CargoSnapshotCompleteness::Partial);
        assert_eq!(snapshot.workspaces.len(), 2);
        assert_eq!(snapshot.failures.len(), 1);
        let requests = requests
            .lock()
            .expect("routing runner request lock should be available");
        assert_eq!(requests.len(), 3);
        assert!(!requests.contains(&PathBuf::from("/workspace/member-one/Cargo.toml")));
    }

    #[gpui::test]
    async fn cargo_workspace_restricted_candidate_never_runs_metadata(_cx: &mut TestAppContext) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let probe_requests = Arc::new(Mutex::new(Vec::new()));
        let snapshot = collect_snapshot(
            1,
            2,
            vec![candidate(1, "/workspace", "/workspace/Cargo.toml", false)],
            Arc::new(RoutingRunner {
                requests: requests.clone(),
            }),
            successful_configuration_probe_with_requests(probe_requests.clone()),
        )
        .await
        .expect("restricted collection should return a scoped failure")
        .snapshot;

        assert!(snapshot.workspaces.is_empty());
        assert_eq!(snapshot.failures.len(), 1);
        assert_eq!(
            snapshot.failures[0].category,
            CargoWorkspaceErrorCategory::Restricted
        );
        assert!(
            requests
                .lock()
                .expect("routing runner request lock should be available")
                .is_empty()
        );
        assert!(
            probe_requests
                .lock()
                .expect("configuration probe request lock should be available")
                .is_empty()
        );
    }

    #[gpui::test]
    async fn cargo_workspace_configuration_uses_visible_files_and_project_environment(
        _cx: &mut TestAppContext,
    ) {
        let probe_requests = Arc::new(Mutex::new(Vec::new()));
        let environment = collections::HashMap::from_iter([
            ("RUSTUP_TOOLCHAIN".to_string(), "stable".to_string()),
            ("SECRET_TOKEN".to_string(), "not-for-the-model".to_string()),
        ]);
        let mut candidate = candidate(1, "/workspace", "/workspace/Cargo.toml", true);
        candidate.environment = Task::ready(Some(environment.clone())).shared();
        candidate.manifest_text = Task::ready(Ok(include_str!(
            "../test_data/cargo_workspace/profiles-custom.toml"
        )
        .to_string()));
        candidate.toolchain_text = Some((
            ProjectPath {
                worktree_id: settings::WorktreeId::from_usize(1),
                path: Arc::from(
                    RelPath::from_unix_str("rust-toolchain.toml").expect("valid fixture path"),
                ),
            },
            Task::ready(Ok(include_str!(
                "../test_data/cargo_workspace/rust-toolchain.toml"
            )
            .to_string())),
        ));

        let snapshot = collect_snapshot(
            2,
            3,
            vec![candidate],
            Arc::new(RoutingRunner {
                requests: Arc::new(Mutex::new(Vec::new())),
            }),
            successful_configuration_probe_with_requests(probe_requests.clone()),
        )
        .await
        .expect("configuration fixture should collect")
        .snapshot;
        let configuration = &snapshot.workspaces[0].configuration;
        assert_eq!(
            configuration.completeness,
            CargoConfigurationCompleteness::Complete
        );
        assert_eq!(configuration.profiles.len(), 4);
        assert_eq!(
            configuration
                .declared_toolchain
                .as_ref()
                .and_then(|toolchain| toolchain.channel.as_deref()),
            Some("stable")
        );
        assert_eq!(
            configuration.host_compiler.host_target.as_deref(),
            Some("x86_64-unknown-linux-gnu")
        );
        let requests = probe_requests
            .lock()
            .expect("configuration probe request lock should be available");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].environment.as_ref(), Some(&environment));

        let encoded = format!("{:?}", snapshot_to_proto(snapshot, 4));
        assert!(!encoded.contains("SECRET_TOKEN"));
        assert!(!encoded.contains("not-for-the-model"));
        assert!(!encoded.contains("/workspace"));
    }

    #[gpui::test]
    async fn cargo_workspace_configuration_failures_are_partial_and_isolated(
        _cx: &mut TestAppContext,
    ) {
        let mut candidate = candidate(1, "/workspace", "/workspace/Cargo.toml", true);
        candidate.manifest_text = Task::ready(Ok(include_str!(
            "../test_data/cargo_workspace/profiles-malformed.toml"
        )
        .to_string()));
        let snapshot = collect_snapshot(
            4,
            5,
            vec![candidate],
            Arc::new(RoutingRunner {
                requests: Arc::new(Mutex::new(Vec::new())),
            }),
            Arc::new(FakeConfigurationProbe {
                output: Err("failed to start rustc: not found".to_string()),
                requests: Arc::new(Mutex::new(Vec::new())),
            }),
        )
        .await
        .expect("configuration failures should not discard metadata")
        .snapshot;
        assert_eq!(snapshot.workspaces.len(), 1);
        assert_eq!(snapshot.completeness, CargoSnapshotCompleteness::Partial);
        let configuration = &snapshot.workspaces[0].configuration;
        assert_eq!(
            configuration.profiles,
            crate::cargo_workspace::implicit_cargo_profiles()
        );
        assert_eq!(
            configuration.host_compiler.status,
            CargoHostCompilerStatus::Missing
        );
        assert_eq!(configuration.diagnostics.len(), 2);
    }

    #[test]
    fn cargo_workspace_path_projection_filters_private_and_outside_paths() {
        let roots = vec![(
            settings::WorktreeId::from_usize(1),
            PathBuf::from("/workspace"),
            Arc::from([Arc::from(
                RelPath::from_unix_str("private").expect("valid private fixture path"),
            )]),
        )];
        assert!(
            project_path_for_absolute_path(std::path::Path::new("/outside/Cargo.toml"), &roots)
                .is_none()
        );
        assert!(
            project_path_for_absolute_path(
                std::path::Path::new("/workspace/private/Cargo.toml"),
                &roots,
            )
            .is_none()
        );
        assert_eq!(
            project_path_for_absolute_path(
                std::path::Path::new("/workspace/public/Cargo.toml"),
                &roots,
            )
            .expect("public path should project")
            .path
            .as_unix_str(),
            "public/Cargo.toml"
        );
    }

    #[test]
    fn cargo_workspace_errors_are_bounded_and_lossy_utf8_safe() {
        let mut bytes = vec![b'x'; MAX_ERROR_BYTES + 32];
        bytes.extend_from_slice(&[0xf0, 0x28, 0x8c, 0x28]);
        let message = bounded_error(&bytes);
        assert!(message.len() <= MAX_ERROR_BYTES + 12);
        assert!(message.contains('\u{fffd}'));
    }

    #[test]
    fn cargo_workspace_errors_remove_host_paths() {
        let path = ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(1),
            path: Arc::from(
                RelPath::from_unix_str("member/Cargo.toml").expect("valid fixture path"),
            ),
        };
        let sanitized = sanitize_candidate_error(
            "failed at /host/private/member/Cargo.toml under /host/private",
            std::path::Path::new("/host/private/member/Cargo.toml"),
            std::path::Path::new("/host/private"),
            &path,
        );
        assert!(!sanitized.contains("/host/private"));
        assert!(sanitized.contains("member/Cargo.toml"));
    }

    #[test]
    fn cargo_workspace_input_filter_includes_configuration_files_only() {
        for path in [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "member/rust-toolchain",
            ".cargo/config.toml",
            "member/.cargo/config",
        ] {
            assert!(is_cargo_workspace_input(
                RelPath::from_unix_str(path).expect("fixture path should be valid")
            ));
        }
        for path in ["src/config.toml", ".cargo/notes.txt", "README.md"] {
            assert!(!is_cargo_workspace_input(
                RelPath::from_unix_str(path).expect("fixture path should be valid")
            ));
        }
    }

    #[test]
    fn cargo_workspace_remote_request_keys_are_peer_scoped() {
        let mut requests = HashSet::new();
        requests.insert(((1_u64, 2_u64), 9_u64));
        assert!(!requests.contains(&((1_u64, 3_u64), 9_u64)));
    }

    #[gpui::test]
    async fn cargo_workspace_remote_snapshot_round_trip_preserves_the_host_model(
        _cx: &mut TestAppContext,
    ) {
        let snapshot = collect_snapshot(
            12,
            34,
            vec![candidate(1, "/workspace", "/workspace/Cargo.toml", true)],
            Arc::new(RoutingRunner {
                requests: Arc::new(Mutex::new(Vec::new())),
            }),
            successful_configuration_probe(),
        )
        .await
        .expect("host fixture collection should succeed")
        .snapshot;

        let response = snapshot_to_proto(snapshot.clone(), 56);
        assert_eq!(response.request_id, 56);
        assert_eq!(
            snapshot_from_proto(response).expect("typed remote snapshot should decode"),
            snapshot
        );
    }

    fn cargo_metadata_facts_fixture(
        bytes: &[u8],
        mut resolve_path: impl FnMut(&Path) -> Option<ProjectPath>,
    ) -> (CargoWorkspaceModel, CargoWorkspaceMetadataFacts) {
        let metadata = parse_metadata(bytes).expect("Valid Cargo metadata fixture");
        let workspace =
            workspace_from_metadata(&metadata, &mut resolve_path).expect("Visible Cargo model");
        let facts = metadata_facts_from_metadata(
            &metadata,
            &workspace,
            resolve_path,
            &std::collections::BTreeMap::new(),
        )
        .expect("Private bounded Cargo facts");
        (workspace, facts)
    }

    fn cargo_metadata_facts_fixture_path(path: &Path) -> Option<ProjectPath> {
        let relative = path.strip_prefix("/workspace").ok()?;
        Some(ProjectPath {
            worktree_id: settings::WorktreeId::from_usize(1),
            path: RelPath::new(relative, PathStyle::local())
                .ok()?
                .into_owned()
                .into(),
        })
    }

    #[test]
    fn native_runtime_dependency_graph_closes_exact_raw_ids_and_refuses_incomplete_data() {
        let mut metadata = parse_metadata(include_bytes!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("actual metadata fixture");
        let isolated = parse_metadata(include_bytes!(
            "../test_data/cargo_workspace/standalone-v1.json"
        ))
        .expect("actual independent package fixture");
        metadata.packages.extend(isolated.packages);
        let ids = metadata
            .packages
            .iter()
            .take(3)
            .map(|package| package.id.repr.clone())
            .collect::<Vec<_>>();
        assert_eq!(ids.len(), 3);
        let node = |id: &str, dependencies: &[String]| -> cargo_metadata::Node {
            serde_json::from_value(serde_json::json!({"id":id, "dependencies":dependencies, "features":[],
                "deps":dependencies.iter().enumerate().map(|(index, dependency)| serde_json::json!({"name":format!("renamed-{index}"),"pkg":dependency,"dep_kinds":[{"kind":null,"target":null}]})).collect::<Vec<_>>()
            })).expect("typed raw-ID dependency node")
        };
        metadata.resolve.as_mut().expect("actual graph").nodes = vec![
            node(&ids[0], &[ids[1].clone()]),
            node(&ids[1], &[ids[0].clone()]),
            node(&ids[2], &[]),
        ];
        let graph = CargoNativeDependencyGraph::from_metadata(&metadata);
        assert_eq!(
            graph.selected_closure(&ids[0]),
            Some(std::collections::BTreeSet::from_iter([
                ids[0].clone(),
                ids[1].clone()
            ]))
        );
        assert_eq!(
            graph.selected_closure(&ids[2]),
            Some(std::collections::BTreeSet::from_iter([ids[2].clone()]))
        );
        assert!(graph.selected_closure("unrelated-raw-id").is_none());
        for change in [
            "missing",
            "duplicated-node",
            "dangling-node",
            "wrong-structured-edge",
            "oversized-id",
            "duplicated-package",
        ] {
            let mut changed = metadata.clone();
            match change {
                "missing" => {
                    changed.resolve = None;
                }
                "duplicated-node" => {
                    changed
                        .resolve
                        .as_mut()
                        .expect("graph")
                        .nodes
                        .push(node(&ids[0], &[]));
                }
                "dangling-node" => {
                    changed.resolve.as_mut().expect("graph").nodes[0] =
                        node(&ids[0], &["missing-package".into()]);
                }
                "wrong-structured-edge" => {
                    changed.resolve.as_mut().expect("graph").nodes[0]
                        .deps
                        .clear();
                }
                "oversized-id" => {
                    changed.resolve.as_mut().expect("graph").nodes[0] = node(
                        &"x".repeat(
                            crate::cargo_workspace::MAX_CARGO_BUILD_MESSAGE_FIELD_BYTES + 1,
                        ),
                        &[],
                    );
                }
                "duplicated-package" => {
                    changed.packages.push(changed.packages[0].clone());
                }
                _ => panic!("finite graph scenario"),
            }
            assert!(
                CargoNativeDependencyGraph::from_metadata(&changed)
                    .selected_closure(&ids[0])
                    .is_none(),
                "{change}"
            );
        }
    }

    #[test]
    fn native_target_policy_facts_inherited_edition_joins_exact_root_and_preserves_unavailable_metadata()
     {
        use crate::cargo_workspace::{CargoNativeExecutionPolicy, CargoTargetKind};
        let metadata = parse_metadata(include_bytes!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("typed metadata fixture");
        let workspace = workspace_from_metadata(&metadata, cargo_metadata_facts_fixture_path)
            .expect("visible model");
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("selected member");
        let root_path = workspace
            .root_manifest
            .clone()
            .expect("exact workspace declaration");
        for harness in [true, false] {
            let member_manifest = format!(
                "[package]\nname='member-one'\nversion='0.1.0'\nedition.workspace=true\n[[test]]\nname='integration'\npath='tests/integration.rs'\nharness={harness}\n[[bench]]\nname='throughput'\npath='benches/throughput.rs'\nharness={harness}\n"
            );
            for change in [
                "current",
                "missing-root",
                "wrong-root",
                "wrong-worktree",
                "changed-root-edition",
                "malformed-root",
                "oversized-root",
            ] {
                let mut declared_workspace = workspace.clone();
                let mut manifests = std::collections::BTreeMap::from_iter([
                    (member.manifest_path.clone(), member_manifest.clone()),
                    (root_path.clone(), "[workspace]\nmembers=['member-one','member-two']\n[workspace.package]\nedition='2024'\n".into()),
                ]);
                match change {
                    "missing-root" => {
                        manifests.remove(&root_path);
                    }
                    "wrong-root" => {
                        declared_workspace.root_manifest = Some(member.manifest_path.clone());
                    }
                    "wrong-worktree" => {
                        let mut wrong = root_path.clone();
                        wrong.worktree_id = settings::WorktreeId::from_usize(2);
                        manifests.insert(
                            wrong.clone(),
                            manifests.get(&root_path).expect("root bytes").clone(),
                        );
                        declared_workspace.root_manifest = Some(wrong);
                    }
                    "changed-root-edition" => {
                        manifests.insert(
                            root_path.clone(),
                            "[workspace.package]\nedition='2021'\n".into(),
                        );
                    }
                    "malformed-root" => {
                        manifests.insert(root_path.clone(), "[workspace.package".into());
                    }
                    "oversized-root" => {
                        manifests.insert(
                            root_path.clone(),
                            "#".repeat(
                                crate::cargo_workspace::MAX_CARGO_NATIVE_TARGET_POLICY_BYTES + 1,
                            ),
                        );
                    }
                    "current" => {}
                    _ => panic!("unknown bounded scenario"),
                }
                let facts = metadata_facts_from_metadata(
                    &metadata,
                    &declared_workspace,
                    cargo_metadata_facts_fixture_path,
                    &manifests,
                )
                .expect("ordinary metadata facts retained");
                let package = facts
                    .packages
                    .iter()
                    .find(|package| package.model_id == member.id)
                    .expect("exact private package");
                for kind in [CargoTargetKind::Test, CargoTargetKind::Bench] {
                    let selected = package
                        .targets
                        .iter()
                        .find(|target| target.model.kind == kind)
                        .expect("selected executable target retained");
                    if change == "current" {
                        let expected = match kind {
                            CargoTargetKind::Test => CargoNativeExecutionPolicy::Test { harness },
                            CargoTargetKind::Bench => CargoNativeExecutionPolicy::Bench { harness },
                            _ => panic!("exact finite kind"),
                        };
                        assert_eq!(
                            selected.execution_policy,
                            CargoNativeTargetPolicyFacts::Reviewed(expected)
                        );
                    } else {
                        assert!(
                            matches!(&selected.execution_policy, CargoNativeTargetPolicyFacts::Unavailable(reason) if !reason.is_empty() && reason.len() <= 512),
                            "unsealed or contradictory root bytes cannot mint inherited policy"
                        );
                    }
                    assert!(
                        native_build_selection(
                            &facts,
                            &declared_workspace,
                            &member.id,
                            &selected.model
                        )
                        .is_err(),
                        "typed data alone cannot activate Test/Bench"
                    );
                }
                for kind in [CargoTargetKind::Binary, CargoTargetKind::Example] {
                    let selected = package
                        .targets
                        .iter()
                        .find(|target| target.model.kind == kind)
                        .expect("ordinary executable metadata retained");
                    assert_eq!(
                        selected.execution_policy,
                        CargoNativeTargetPolicyFacts::Reviewed(CargoNativeExecutionPolicy::Program)
                    );
                    assert!(
                        native_build_selection(
                            &facts,
                            &declared_workspace,
                            &member.id,
                            &selected.model
                        )
                        .is_ok(),
                        "existing Program route and metadata remain available"
                    );
                }
            }
        }
    }

    #[test]
    fn native_target_policy_facts_join_exact_manifest_without_activating_capture() {
        use crate::cargo_workspace::{CargoNativeExecutionPolicy, CargoTargetKind};
        let metadata = parse_metadata(include_bytes!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("typed metadata fixture");
        let workspace = workspace_from_metadata(&metadata, cargo_metadata_facts_fixture_path)
            .expect("visible model");
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("selected member");
        for harness in [true, false] {
            let manifest = format!(
                "[package]\nname='member-one'\nversion='0.1.0'\nedition='2024'\n[[test]]\nname='integration'\npath='tests/integration.rs'\nharness={harness}\n[[bench]]\nname='throughput'\npath='benches/throughput.rs'\nharness={harness}\n"
            );
            let manifests =
                std::collections::BTreeMap::from_iter([(member.manifest_path.clone(), manifest)]);
            let facts = metadata_facts_from_metadata(
                &metadata,
                &workspace,
                cargo_metadata_facts_fixture_path,
                &manifests,
            )
            .expect("bounded private policy facts");
            let package = facts
                .packages
                .iter()
                .find(|package| package.model_id == member.id)
                .expect("exact private package");
            for (kind, expected) in [
                (
                    CargoTargetKind::Test,
                    CargoNativeExecutionPolicy::Test { harness },
                ),
                (
                    CargoTargetKind::Bench,
                    CargoNativeExecutionPolicy::Bench { harness },
                ),
            ] {
                let selected = package
                    .targets
                    .iter()
                    .find(|target| target.model.kind == kind)
                    .expect("exact selected target");
                assert_eq!(
                    selected.execution_policy,
                    CargoNativeTargetPolicyFacts::Reviewed(expected)
                );
                assert!(
                    native_build_selection(&facts, &workspace, &member.id, &selected.model)
                        .is_err(),
                    "manifest data cannot activate unsupported Test/Bench execution"
                );
            }
            for kind in [CargoTargetKind::Binary, CargoTargetKind::Example] {
                let selected = package
                    .targets
                    .iter()
                    .find(|target| target.model.kind == kind)
                    .expect("existing selected program");
                assert_eq!(
                    selected.execution_policy,
                    CargoNativeTargetPolicyFacts::Reviewed(CargoNativeExecutionPolicy::Program)
                );
                assert!(
                    native_build_selection(&facts, &workspace, &member.id, &selected.model).is_ok(),
                    "existing exact Program recipe is preserved"
                );
            }
        }
    }

    #[test]
    fn native_target_policy_facts_preserve_unavailable_and_manifest_mutation() {
        use crate::cargo_workspace::{CargoNativeExecutionPolicy, CargoTargetKind};
        let metadata = parse_metadata(include_bytes!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("typed metadata fixture");
        let workspace = workspace_from_metadata(&metadata, cargo_metadata_facts_fixture_path)
            .expect("visible model");
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("selected member");
        let header = "[package]\nname='member-one'\nversion='0.1.0'\nedition='2024'\n";
        let mut reviewed = Vec::new();
        for manifest in [
            None,
            Some("invalid=[".into()),
            Some(" ".repeat(crate::cargo_workspace::MAX_CARGO_NATIVE_TARGET_POLICY_BYTES + 1)),
            Some(format!(
                "{header}[[test]]\nname='integration'\npath='private-policy-sentinel.rs'\n"
            )),
            Some(format!(
                "{header}[[test]]\nname='integration'\nharness=false\n"
            )),
            Some(format!(
                "{header}[[test]]\nname='integration'\nharness=true\n"
            )),
        ] {
            let manifests = manifest
                .map(|contents| {
                    std::collections::BTreeMap::from_iter([(
                        member.manifest_path.clone(),
                        contents,
                    )])
                })
                .unwrap_or_default();
            let facts = metadata_facts_from_metadata(
                &metadata,
                &workspace,
                cargo_metadata_facts_fixture_path,
                &manifests,
            )
            .expect("unavailable native policy does not discard ordinary metadata");
            let selected = facts
                .packages
                .iter()
                .find(|package| package.model_id == member.id)
                .expect("private member")
                .targets
                .iter()
                .find(|target| target.model.kind == CargoTargetKind::Test)
                .expect("test target");
            match &selected.execution_policy {
                CargoNativeTargetPolicyFacts::Reviewed(policy) => reviewed.push(*policy),
                CargoNativeTargetPolicyFacts::Unavailable(reason) => {
                    assert!(!reason.is_empty());
                    assert!(!reason.contains("private-policy-sentinel"));
                }
            }
            assert!(
                native_build_selection(&facts, &workspace, &member.id, &selected.model).is_err()
            );
        }
        assert_eq!(
            reviewed,
            [
                CargoNativeExecutionPolicy::Test { harness: false },
                CargoNativeExecutionPolicy::Test { harness: true },
            ],
            "harness changes are retained as fresh typed data, never executable authority"
        );
    }

    #[test]
    fn native_build_intent_exact_selection_and_pinned_output_recipe() {
        let (workspace, facts) = cargo_metadata_facts_fixture(
            include_bytes!("../test_data/cargo_workspace/workspace-v1.json"),
            cargo_metadata_facts_fixture_path,
        );
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("member");
        let target = member
            .targets
            .iter()
            .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
            .expect("binary");
        let (selection, output, arguments) =
            native_build_selection(&facts, &workspace, &member.id, target)
                .expect("exact selection");
        assert_eq!(
            selection.package_id,
            "path+file:///workspace/member-one#0.1.0"
        );
        assert_eq!(
            selection.source_path,
            PathBuf::from("/workspace/member-one/src/main.rs")
        );
        assert_eq!(output, PathBuf::from("/workspace/target"));
        assert_eq!(
            arguments,
            [
                "build",
                "--message-format=json",
                "--package",
                "member-one",
                "--bin",
                "member-one",
                "--target-dir",
                "/workspace/target"
            ]
        );
        for kind in [
            crate::cargo_workspace::CargoTargetKind::Binary,
            crate::cargo_workspace::CargoTargetKind::Example,
        ] {
            let mut model = workspace.clone();
            let mut private = facts.clone();
            let mut selected = target.clone();
            selected.kind = kind.clone();
            selected.required_features = vec!["first".into(), "second".into()];
            model
                .members
                .iter_mut()
                .find(|candidate| candidate.id == member.id)
                .expect("model member")
                .targets = vec![selected.clone()];
            private
                .packages
                .iter_mut()
                .find(|candidate| candidate.model_id == member.id)
                .expect("private member")
                .targets = vec![CargoTargetMetadataFacts {
                execution_policy: CargoNativeTargetPolicyFacts::Reviewed(
                    crate::cargo_workspace::CargoNativeExecutionPolicy::Program,
                ),
                model: selected.clone(),
                source_path: selection.source_path.clone(),
                kinds: vec![if kind == crate::cargo_workspace::CargoTargetKind::Binary {
                    "bin".into()
                } else {
                    "example".into()
                }],
            }];
            let (exact, _, argv) = native_build_selection(&private, &model, &member.id, &selected)
                .expect("selected exact recipe");
            assert_eq!(exact.target_kind, kind);
            assert_eq!(&argv[8..], &["--features", "first,second"]);
        }
    }

    #[test]
    fn native_build_named_profile_recipe_rejects_removed_ambiguous_and_tampered_selection() {
        let (mut workspace, facts) = cargo_metadata_facts_fixture(
            include_bytes!("../test_data/cargo_workspace/workspace-v1.json"),
            cargo_metadata_facts_fixture_path,
        );
        workspace
            .configuration
            .profiles
            .push(crate::cargo_workspace::CargoProfileModel {
                name: "native-review".into(),
                origin: crate::cargo_workspace::CargoProfileOrigin::Declared,
            });
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("member")
            .clone();
        let target = member
            .targets
            .iter()
            .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
            .expect("binary");
        let (_, root, original) = native_build_selection_with_profile(
            &facts,
            &workspace,
            &member.id,
            target,
            Some("native-review"),
        )
        .expect("declared exact profile");
        assert_eq!(
            &original[original.len() - 2..],
            &["--profile", "native-review"]
        );
        let run = uuid::Uuid::new_v4();
        let (_, bound) = native_build_run_arguments(&original, &root, run)
            .expect("same profile with fresh output");
        assert_eq!(&bound[bound.len() - 2..], &["--profile", "native-review"]);
        let (_, _, changed) = native_build_selection_with_profile(
            &facts,
            &workspace,
            &member.id,
            target,
            Some("release"),
        )
        .expect("different valid profile");
        assert_ne!(
            changed, original,
            "same target cannot rebind a frozen profile recipe"
        );
        let duplicate = workspace
            .configuration
            .profiles
            .last()
            .expect("declared profile")
            .clone();
        workspace.configuration.profiles.push(duplicate);
        assert!(
            native_build_selection_with_profile(
                &facts,
                &workspace,
                &member.id,
                target,
                Some("native-review")
            )
            .is_err()
        );
        workspace
            .configuration
            .profiles
            .retain(|profile| profile.name != "native-review");
        assert!(
            native_build_selection_with_profile(
                &facts,
                &workspace,
                &member.id,
                target,
                Some("native-review")
            )
            .is_err(),
            "a removed profile cannot renew its old intent recipe"
        );
        assert!(
            native_build_selection_with_profile(
                &facts,
                &workspace,
                &member.id,
                target,
                Some("unissued")
            )
            .is_err()
        );
    }

    #[test]
    fn native_build_intent_run_output_binding_rejects_other_run_executable() {
        let base = [
            "build",
            "--message-format=json",
            "--package",
            "member-one",
            "--bin",
            "member-one",
            "--target-dir",
            "/workspace/target",
        ]
        .map(str::to_owned);
        let first = uuid::Uuid::new_v4();
        let second = uuid::Uuid::new_v4();
        let (first_root, first_arguments) =
            native_build_run_arguments(&base, Path::new("/workspace/target"), first)
                .expect("first run");
        let (second_root, second_arguments) =
            native_build_run_arguments(&base, Path::new("/workspace/target"), second)
                .expect("second run");
        assert_ne!(first_root, second_root);
        assert_eq!(
            first_arguments.get(7).map(String::as_str),
            first_root.to_str()
        );
        assert_eq!(
            second_arguments.get(7).map(String::as_str),
            second_root.to_str()
        );
        assert_eq!(&first_arguments[..7], &base[..7]);
        assert_eq!(
            base[7], "/workspace/target",
            "logical persisted recipe remains unbound"
        );
        let first_executable = first_root.join("debug/member-one");
        validate_native_build_run_executable(&first_executable, &first_root).expect("same run");
        assert!(validate_native_build_run_executable(&first_executable, &second_root).is_err());
        assert!(validate_native_build_run_executable(&first_root, &first_root).is_err());
        assert!(
            validate_native_build_run_executable(
                &first_root.join("../other/debug/member-one"),
                &first_root
            )
            .is_err()
        );
        assert!(
            native_build_run_arguments(&base, Path::new("/workspace/target"), uuid::Uuid::nil())
                .is_err()
        );
        let mut changed = base;
        changed[6] = "--manifest-path".into();
        assert!(
            native_build_run_arguments(&changed, Path::new("/workspace/target"), first).is_err()
        );
    }

    #[test]
    fn native_build_intent_rejects_ambiguous_stale_unsupported_and_oversized_selection() {
        let (workspace, facts) = cargo_metadata_facts_fixture(
            include_bytes!("../test_data/cargo_workspace/workspace-v1.json"),
            cargo_metadata_facts_fixture_path,
        );
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("member");
        let target = member
            .targets
            .iter()
            .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
            .expect("binary");
        assert!(
            native_build_selection(
                &facts,
                &workspace,
                &facts.packages.first().expect("private member").package_id,
                target
            )
            .is_err()
        );
        let mut duplicated = facts.clone();
        let package = duplicated
            .packages
            .iter()
            .find(|package| package.model_id == member.id)
            .expect("package")
            .clone();
        duplicated.packages.push(package);
        assert!(native_build_selection(&duplicated, &workspace, &member.id, target).is_err());
        let mut duplicated = facts.clone();
        let package = duplicated
            .packages
            .iter_mut()
            .find(|package| package.model_id == member.id)
            .expect("package");
        let duplicate = package
            .targets
            .iter()
            .find(|candidate| candidate.model == *target)
            .expect("target")
            .clone();
        package.targets.push(duplicate);
        assert!(native_build_selection(&duplicated, &workspace, &member.id, target).is_err());
        let mut stale = target.clone();
        stale.name.push_str("-stale");
        assert!(native_build_selection(&facts, &workspace, &member.id, &stale).is_err());
        for kind in [
            crate::cargo_workspace::CargoTargetKind::Test,
            crate::cargo_workspace::CargoTargetKind::Bench,
        ] {
            let mut unsupported = target.clone();
            unsupported.kind = kind;
            let mut model = workspace.clone();
            model
                .members
                .iter_mut()
                .find(|candidate| candidate.id == member.id)
                .expect("model member")
                .targets = vec![unsupported.clone()];
            let mut private = facts.clone();
            private
                .packages
                .iter_mut()
                .find(|candidate| candidate.model_id == member.id)
                .expect("package")
                .targets = vec![CargoTargetMetadataFacts {
                execution_policy: CargoNativeTargetPolicyFacts::Reviewed(
                    crate::cargo_workspace::CargoNativeExecutionPolicy::Program,
                ),
                model: unsupported.clone(),
                source_path: PathBuf::from("/workspace/member-one/src/main.rs"),
                kinds: vec!["test".into()],
            }];
            assert!(native_build_selection(&private, &model, &member.id, &unsupported).is_err());
        }
        let mut oversized = target.clone();
        oversized.required_features = vec!["x".repeat(1024); 5];
        let mut model = workspace.clone();
        model
            .members
            .iter_mut()
            .find(|candidate| candidate.id == member.id)
            .expect("model member")
            .targets = vec![oversized.clone()];
        let mut private = facts;
        private
            .packages
            .iter_mut()
            .find(|candidate| candidate.model_id == member.id)
            .expect("package")
            .targets = vec![CargoTargetMetadataFacts {
            execution_policy: CargoNativeTargetPolicyFacts::Reviewed(
                crate::cargo_workspace::CargoNativeExecutionPolicy::Program,
            ),
            model: oversized.clone(),
            source_path: PathBuf::from("/workspace/member-one/src/main.rs"),
            kinds: vec!["bin".into()],
        }];
        assert!(native_build_selection(&private, &model, &member.id, &oversized).is_err());
    }

    #[test]
    fn cargo_workspace_private_metadata_preserves_exact_ids_targets_and_custom_output_root() {
        let bytes = include_str!("../test_data/cargo_workspace/workspace-v1.json").replace(
            "\"target_directory\": \"/workspace/target\"",
            "\"target_directory\": \"/private-output/cache/custom-target\"",
        );
        let (workspace, facts) =
            cargo_metadata_facts_fixture(bytes.as_bytes(), cargo_metadata_facts_fixture_path);
        assert!(facts.matches_workspace(&workspace));
        assert_eq!(
            facts.target_directory,
            PathBuf::from("/private-output/cache/custom-target")
        );
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("Current member");
        let package = facts
            .packages
            .iter()
            .find(|package| package.model_id == member.id)
            .expect("Private member identity");
        assert_eq!(
            package.package_id,
            "path+file:///workspace/member-one#0.1.0"
        );
        assert_ne!(
            package.package_id, member.id,
            "Cargo's exact PackageId cannot be replaced by the UI stable ID"
        );
        assert_eq!(member.id, "1:member-one/Cargo.toml:member-one@0.1.0");
        let binary = package
            .targets
            .iter()
            .find(|target| target.model.kind == crate::cargo_workspace::CargoTargetKind::Binary)
            .expect("Exact binary target data");
        assert_eq!(binary.model.name, "member-one");
        assert_eq!(
            binary.source_path,
            PathBuf::from("/workspace/member-one/src/main.rs")
        );
        assert_eq!(binary.kinds, vec!["bin"]);
        let mut changed = workspace.clone();
        changed
            .members
            .iter_mut()
            .find(|member| member.id == package.model_id)
            .expect("Selected member")
            .targets
            .retain(|target| target.kind != crate::cargo_workspace::CargoTargetKind::Binary);
        assert!(
            !facts.matches_workspace(&changed),
            "A removed or changed current target cannot match retained facts"
        );
        let mut changed_identity = workspace.clone();
        changed_identity
            .members
            .iter_mut()
            .find(|member| member.id == package.model_id)
            .expect("Selected stable ID")
            .name = "changed-name-with-retained-ui-id".into();
        assert!(
            !facts.matches_workspace(&changed_identity),
            "Stable UI IDs alone cannot attest exact current package identity"
        );
        let snapshot = CargoWorkspaceSnapshot {
            revision: 1,
            input_fingerprint: 2,
            workspaces: vec![workspace],
            failures: Vec::new(),
            completeness: CargoSnapshotCompleteness::Partial,
        };
        let response = snapshot_to_proto(snapshot.clone(), 3);
        let encoded = format!("{response:?}");
        assert!(!encoded.contains(&package.package_id));
        assert!(!encoded.contains("/private-output"));
        assert!(!encoded.contains("/workspace"));
        assert_eq!(
            snapshot_from_proto(response).expect("Unchanged public wire model"),
            snapshot
        );
    }

    #[test]
    fn cargo_workspace_private_metadata_distinguishes_same_named_binary_and_example() {
        let mut report: serde_json::Value = serde_json::from_str(include_str!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("Fixture JSON");
        let mut example = report["packages"][0]["targets"][1].clone();
        example["kind"] = serde_json::json!(["example"]);
        report["packages"][0]["targets"]
            .as_array_mut()
            .expect("Targets")
            .push(example);
        let bytes = serde_json::to_vec(&report).expect("Metadata bytes");
        let (workspace, facts) =
            cargo_metadata_facts_fixture(&bytes, cargo_metadata_facts_fixture_path);
        let package = facts
            .packages
            .iter()
            .find(|package| package.model_id.ends_with("member-one@0.1.0"))
            .expect("Exact member");
        let selected = package
            .targets
            .iter()
            .filter(|target| target.model.name == "member-one")
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|target| target.model.kind
            == crate::cargo_workspace::CargoTargetKind::Binary
            && target.kinds == ["bin"]));
        assert!(selected.iter().any(|target| target.model.kind
            == crate::cargo_workspace::CargoTargetKind::Example
            && target.kinds == ["example"]));
        assert!(facts.matches_workspace(&workspace));
    }

    #[test]
    fn cargo_workspace_private_metadata_excludes_unresolved_private_target_paths() {
        let sentinel = "/sensitive-source/private-build.rs";
        let bytes = include_str!("../test_data/cargo_workspace/workspace-v1.json")
            .replace("/workspace/member-one/build.rs", sentinel);
        let (workspace, facts) =
            cargo_metadata_facts_fixture(bytes.as_bytes(), cargo_metadata_facts_fixture_path);
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("Member");
        assert!(member.targets.iter().any(|target| target.kind
            == crate::cargo_workspace::CargoTargetKind::BuildScript
            && target.source_path.is_none()));
        assert!(
            facts
                .packages
                .iter()
                .flat_map(|package| &package.targets)
                .all(|target| target.source_path != Path::new(sentinel))
        );
        assert!(facts.matches_workspace(&workspace));
    }

    #[test]
    fn cargo_workspace_private_metadata_joint_retention_budget_is_bounded() {
        let bytes = include_bytes!("../test_data/cargo_workspace/workspace-v1.json");
        let (_, mut facts) = cargo_metadata_facts_fixture(bytes, cargo_metadata_facts_fixture_path);
        facts.packages.truncate(1);
        let package = facts.packages.first_mut().expect("Package");
        package.targets.truncate(1);
        package
            .targets
            .first_mut()
            .expect("Target")
            .model
            .required_features = vec!["x".repeat(4096); 256];
        validate_metadata_facts_bounds(std::slice::from_ref(&facts))
            .expect("Individually bounded private data");
        let retained = (0..9)
            .map(|index| {
                let mut facts = facts.clone();
                facts.key.root.path = Arc::from(
                    RelPath::from_unix_str(&format!("workspace-{index}"))
                        .expect("Distinct workspace root"),
                );
                facts
            })
            .collect::<Vec<_>>();
        let error = validate_metadata_facts_bounds(&retained).expect_err(
            "Joint private cache limit cannot be bypassed by individually valid scopes",
        );
        assert!(error.to_string().contains("retained limit exceeded"));
    }

    #[test]
    fn cargo_workspace_private_metadata_rejects_relative_oversized_and_ambiguous_data() {
        let original: serde_json::Value = serde_json::from_str(include_str!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("Fixture JSON");
        let mut relative = original.clone();
        relative["target_directory"] = serde_json::json!("caller-relative-target");
        let mut oversized = original.clone();
        oversized["target_directory"] =
            serde_json::json!(format!("/private-sentinel-{}", "x".repeat(16 * 1024)));
        let long_id = format!("path+file:///private-id-sentinel/{}", "x".repeat(16 * 1024));
        let old_id = original["packages"][0]["id"].as_str().expect("Package ID");
        let oversized_id: serde_json::Value = serde_json::from_str(
            &serde_json::to_string(&original)
                .expect("Original JSON")
                .replace(old_id, &long_id),
        )
        .expect("Consistent overlong opaque ID");
        let mut duplicate_package = original.clone();
        let package = duplicate_package["packages"][0].clone();
        duplicate_package["packages"]
            .as_array_mut()
            .expect("Packages")
            .push(package);
        let mut duplicate_target = original;
        let target = duplicate_target["packages"][0]["targets"][0].clone();
        duplicate_target["packages"][0]["targets"]
            .as_array_mut()
            .expect("Targets")
            .push(target);
        for invalid in [
            relative,
            oversized,
            oversized_id,
            duplicate_package,
            duplicate_target,
        ] {
            let metadata = parse_metadata(&serde_json::to_vec(&invalid).expect("Case JSON"))
                .expect("Parsable data is not authority");
            let workspace = workspace_from_metadata(&metadata, cargo_metadata_facts_fixture_path)
                .expect("Public model remains distinct from private facts");
            let error = match metadata_facts_from_metadata(
                &metadata,
                &workspace,
                cargo_metadata_facts_fixture_path,
                &std::collections::BTreeMap::new(),
            ) {
                Ok(_) => panic!("Invalid or ambiguous private data must fail explicitly"),
                Err(error) => error,
            };
            assert!(!error.to_string().contains("private-sentinel"));
            assert!(!error.to_string().contains("private-id-sentinel"));
            assert!(!error.to_string().contains("caller-relative-target"));
        }
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_cargo_private_metadata_import_replacement_and_invalidation(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        filesystem
            .atomic_write(
                PathBuf::from("/workspace/member-one/src/main.rs"),
                "fn main() {}\n".into(),
            )
            .await
            .expect("Visible selected target source");
        cx.run_until_parked();
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let original: serde_json::Value = serde_json::from_str(include_str!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("Fixture JSON");
        for target_directory in ["/custom-output/first", "/custom-output/replaced"] {
            let mut report = original.clone();
            report["target_directory"] = serde_json::json!(target_directory);
            let reviewed = store
                .read_with(cx, |store, cx| {
                    store.prepare_reviewed_metadata_import(manifest.clone(), cx)
                })
                .expect("Current metadata review");
            store
                .update(cx, |store, cx| {
                    store.admit_reviewed_metadata_import(
                        reviewed,
                        serde_json::to_vec(&report).expect("Metadata bytes"),
                        cx,
                    )
                })
                .await
                .expect("Reviewed metadata data import");
            store.read_with(cx, |store, cx| {
                let (snapshot, current) = store.cached_snapshot(cx).expect("Current public cache");
                assert!(current);
                assert_eq!(store.cached_metadata_facts.len(), 1);
                assert!(store.metadata_facts_for_current_snapshot(cx).is_some(), "Current private data remains an existing-owner fact, never a producer credential");
                let facts = store
                    .cached_metadata_facts
                    .first()
                    .expect("Exact current facts");
                assert_eq!(facts.target_directory, PathBuf::from(target_directory));
                assert!(
                    facts.matches_workspace(
                        snapshot.workspaces.first().expect("Imported workspace")
                    )
                );
                assert!(
                    facts
                        .packages
                        .iter()
                        .flat_map(|package| &package.targets)
                        .any(|target| target.source_path
                            == Path::new("/workspace/member-one/src/main.rs"))
                );
            });
        }
        let mut invalid = original.clone();
        invalid["target_directory"] = serde_json::json!("relative-output");
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Current review before rejection");
        assert!(
            store
                .update(cx, |store, cx| store.admit_reviewed_metadata_import(
                    reviewed,
                    serde_json::to_vec(&invalid).expect("Invalid report bytes"),
                    cx
                ))
                .await
                .is_err()
        );
        assert_eq!(
            store.read_with(cx, |store, _| store
                .cached_metadata_facts
                .first()
                .expect("Prior valid metadata is preserved on rejected import")
                .target_directory
                .clone()),
            PathBuf::from("/custom-output/replaced")
        );
        filesystem.atomic_write(PathBuf::from("/workspace/Cargo.toml"), "[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n# changed source\n".into()).await.expect("Actual source invalidation");
        cx.run_until_parked();
        store.read_with(cx, |store, cx| {
            assert!(
                store.cached_metadata_facts.is_empty(),
                "Invalidation retires private selection data"
            );
            assert!(
                !store
                    .cached_snapshot(cx)
                    .expect("Old public diagnostics retained")
                    .1
            );
        });
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Fresh review");
        store
            .update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(
                    reviewed,
                    serde_json::to_vec(&original).expect("Original report"),
                    cx,
                )
            })
            .await
            .expect("Fresh data recovery");
        assert!(!store.read_with(cx, |store, _| store.cached_metadata_facts.is_empty()));
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"vim_mode":true}"#, cx)
                    .expect("Settings changed")
            });
        });
        cx.run_until_parked();
        assert!(
            store.read_with(cx, |store, _| store.cached_metadata_facts.is_empty()),
            "Changed settings retire output-root selection data independently of UI diagnostics"
        );
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Current settings review");
        store
            .update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(
                    reviewed,
                    serde_json::to_vec(&original).expect("Report data"),
                    cx,
                )
            })
            .await
            .expect("Fresh data after settings change");
        assert!(!store.read_with(cx, |store, _| store.cached_metadata_facts.is_empty()));
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let trust =
            cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Existing trust owner"));
        trust.update(cx, |trust, cx| {
            trust.restrict(
                worktree_store.downgrade(),
                collections::HashSet::from_iter([crate::trusted_worktrees::PathTrust::Worktree(
                    manifest.worktree_id,
                )]),
                cx,
            )
        });
        cx.run_until_parked();
        store.read_with(cx, |store, cx| {
            assert!(
                store.cached_metadata_facts.is_empty(),
                "Actual restriction retires private facts"
            );
            assert!(
                !store
                    .cached_snapshot(cx)
                    .expect("Prior UI diagnostics retained")
                    .1
            );
        });
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_cargo_private_metadata_refresh_and_failed_refresh_retirement(
        cx: &mut TestAppContext,
    ) {
        let (project, _filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let bytes = include_str!("../test_data/cargo_workspace/workspace-v1.json")
            .replace(
                "\"target_directory\": \"/workspace/target\"",
                "\"target_directory\": \"/configured-cargo-output/target\"",
            )
            .into_bytes();
        for (report, success) in [
            (bytes.clone(), true),
            (b"malformed data".to_vec(), false),
            (bytes, true),
        ] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let snapshot = store
                .update(cx, |store, cx| {
                    store
                        .set_local_runners_for_test(
                            Arc::new(FakeRunner {
                                output: report,
                                requests: requests.clone(),
                            }),
                            successful_configuration_probe(),
                        )
                        .expect("Existing local metadata owner");
                    store.refresh_scoped(Some(HashSet::from([manifest.worktree_id])), cx)
                })
                .await
                .expect("Refresh result");
            let actual_requests = requests.lock().expect("Recorded metadata requests");
            let expected_manifest_paths = if success {
                vec![PathBuf::from("/workspace/Cargo.toml")]
            } else {
                vec![
                    PathBuf::from("/workspace/Cargo.toml"),
                    PathBuf::from("/workspace/member-one/Cargo.toml"),
                    PathBuf::from("/workspace/member-two/Cargo.toml"),
                ]
            };
            assert_eq!(
                actual_requests
                    .iter()
                    .map(|request| request.manifest_path.clone())
                    .collect::<Vec<_>>(),
                expected_manifest_paths
            );
            for request in actual_requests.iter() {
                assert_eq!(
                    Some(request.working_directory.as_path()),
                    request.manifest_path.parent()
                );
            }
            drop(actual_requests);
            store.read_with(cx, |store, cx| {
                if success {
                    assert!(snapshot.failures.is_empty());
                    assert_eq!(snapshot.workspaces.len(), 1);
                    assert_eq!(store.cached_metadata_facts.len(), 1);
                    let facts = store
                        .cached_metadata_facts
                        .first()
                        .expect("Current raw data");
                    assert_eq!(
                        facts.target_directory,
                        PathBuf::from("/configured-cargo-output/target")
                    );
                    assert_eq!(
                        facts.packages.first().expect("Member data").package_id,
                        "path+file:///workspace/member-one#0.1.0"
                    );
                    let workspace = snapshot.workspaces.first().expect("Exact model");
                    assert_eq!(workspace.key.root.worktree_id, manifest.worktree_id);
                    assert_eq!(workspace.key.root.path.as_ref(), RelPath::empty());
                    assert_eq!(workspace.root_manifest.as_ref(), Some(&manifest));
                    assert!(facts.matches_workspace(workspace));
                    let raw_members = facts.packages.iter().map(|package| {
                        assert_eq!(package.manifest_path.worktree_id, manifest.worktree_id);
                        let targets = package.targets.iter().map(|target| (
                            target.model.name.clone(), target.source_path.clone(), target.kinds.clone(),
                        )).collect::<std::collections::BTreeSet<_>>();
                        assert_eq!(targets.len(), package.targets.len(), "Exact distinct target inventory");
                        (package.manifest_path.path.to_string(), (package.package_id.clone(), targets))
                    }).collect::<std::collections::BTreeMap<_, _>>();
                    let expected: serde_json::Value = serde_json::from_str(include_str!(
                        "../test_data/cargo_workspace/workspace-v1.json"
                    )).expect("Exact fixture target inventory");
                    let expected_members = expected.get("packages").and_then(serde_json::Value::as_array)
                        .expect("Fixture members").iter().map(|package| {
                            let manifest = package.get("manifest_path").and_then(serde_json::Value::as_str)
                                .expect("Fixture manifest").strip_prefix("/workspace/").expect("Same scoped worktree");
                            let package_id = package.get("id").and_then(serde_json::Value::as_str).expect("Raw PackageId");
                            let targets = package.get("targets").and_then(serde_json::Value::as_array)
                                .expect("Fixture targets").iter().map(|target| {
                                    let name = target.get("name").and_then(serde_json::Value::as_str).expect("Target name");
                                    let source = target.get("src_path").and_then(serde_json::Value::as_str).expect("Reported source path");
                                    let kinds = target.get("kind").and_then(serde_json::Value::as_array).expect("Raw target kinds")
                                        .iter().map(|kind|kind.as_str().expect("Kind string").to_string()).collect::<Vec<_>>();
                                    (name.to_string(), PathBuf::from(source), kinds)
                                }).collect::<std::collections::BTreeSet<_>>();
                            (manifest.to_string(), (package_id.to_string(), targets))
                        }).collect::<std::collections::BTreeMap<_, _>>();
                    assert_eq!(raw_members, expected_members,
                        "Retained data must match every exact reported target, not a guessed existing-file subset");
                    assert_eq!(raw_members.get("member-one/Cargo.toml").expect("First member").1.len(), 7);
                    assert_eq!(raw_members.get("member-two/Cargo.toml").expect("Second member").1.len(), 1);
                    assert!(store.metadata_facts_for_current_snapshot(cx).is_some());
                    assert!(store.cached_snapshot(cx).expect("Current UI cache").1);
                } else {
                    assert!(snapshot.workspaces.is_empty());
                    assert_eq!(snapshot.failures.len(), 3);
                    let failed_manifests = snapshot
                        .failures
                        .iter()
                        .map(|failure| {
                            assert_eq!(failure.manifest_path.worktree_id, manifest.worktree_id);
                            assert_eq!(
                                failure.category,
                                CargoWorkspaceErrorCategory::InvalidMetadata
                            );
                            failure.manifest_path.path.to_string()
                        })
                        .collect::<std::collections::BTreeSet<_>>();
                    assert_eq!(
                        failed_manifests,
                        std::collections::BTreeSet::from_iter([
                            "Cargo.toml".to_string(),
                            "member-one/Cargo.toml".to_string(),
                            "member-two/Cargo.toml".to_string(),
                        ])
                    );
                    assert!(store.metadata_facts_for_current_snapshot(cx).is_none());
                    assert!(
                        store.cached_metadata_facts.is_empty(),
                        "Failed refresh cannot retain prior raw selection facts"
                    );
                    assert!(
                        !store
                            .cached_snapshot(cx)
                            .expect("Prior diagnostic model preserved")
                            .1
                    );
                }
            });
        }
    }

    #[cfg(feature = "test-support")]
    async fn doctor_cargo_import_fixture(
        cx: &mut TestAppContext,
    ) -> (Entity<crate::Project>, Arc<fs::FakeFs>, ProjectPath) {
        use crate::trusted_worktrees::{DbTrustedPaths, PathTrust};
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::trusted_worktrees::init(DbTrustedPaths::default(), cx);
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{}", cx)
                    .expect("Observed settings");
            });
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        for root in ["/workspace", "/other"] {
            filesystem.insert_tree(root, serde_json::json!({
                "Cargo.toml": "[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n",
                "member-one": {"Cargo.toml": "[package]\nname = \"member-one\"\nversion = \"0.1.0\"\n", "src": {"lib.rs": "preserved source"}},
                "member-two": {"Cargo.toml": "[package]\nname = \"member-two\"\nversion = \"0.2.0\"\n", "src": {"lib.rs": "other source"}},
            })).await;
        }
        let project = crate::Project::test_with_worktree_trust(
            filesystem.clone(),
            [
                std::path::Path::new("/workspace"),
                std::path::Path::new("/other"),
            ],
            cx,
        )
        .await;
        let trees = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .map(|tree| tree.read(cx).id())
                .collect::<Vec<_>>()
        });
        let store = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &store,
                collections::HashSet::from_iter(trees.iter().copied().map(PathTrust::Worktree)),
                cx,
            )
        });
        cx.run_until_parked();
        let manifest = ProjectPath {
            worktree_id: *trees.first().expect("Root"),
            path: Arc::from(RelPath::from_unix_str("Cargo.toml").expect("Manifest")),
        };
        (project, filesystem, manifest)
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_refreshes_sole_scope_without_promoting_stale_neighbors(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let bytes = include_bytes!("../test_data/cargo_workspace/workspace-v1.json").to_vec();
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Initial exact review");
        store
            .update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(reviewed, bytes.clone(), cx)
            })
            .await
            .expect("Initial owned import");
        for retain_neighbor in [false, true] {
            let neighbor = if retain_neighbor {
                let metadata = parse_metadata(
                    String::from_utf8_lossy(&bytes)
                        .replace("/workspace", "/other")
                        .as_bytes(),
                )
                .expect("Other workspace fixture");
                let neighbor = cx
                    .update(|cx| {
                        workspace_from_metadata(&metadata, |path| {
                            project.read(cx).find_project_path(path, cx)
                        })
                    })
                    .expect("Other owner scope");
                store.update(cx, |store, cx| {
                    let fingerprint = store.current_input_fingerprint(cx);
                    let snapshot = store
                        .cached_snapshot
                        .as_mut()
                        .expect("Existing owner snapshot");
                    snapshot.workspaces.push(neighbor.clone());
                    snapshot.input_fingerprint = fingerprint;
                    store.cached_generation = store.revision;
                });
                Some(neighbor)
            } else {
                None
            };
            filesystem.atomic_write(PathBuf::from("/workspace/Cargo.toml"), format!("[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n# reviewed selected input change; retain neighbor={retain_neighbor}\n")).await.expect("Changed selected source input");
            cx.run_until_parked();
            assert!(!store.read_with(cx, |store, cx| {
                store
                    .cached_snapshot(cx)
                    .expect("Retained diagnostic evidence")
                    .1
            }));
            let reviewed = store
                .read_with(cx, |store, cx| {
                    store.prepare_reviewed_metadata_import(manifest.clone(), cx)
                })
                .expect("Current reviewed source metadata");
            reviewed
                .preflight_filesystem(filesystem.as_ref())
                .await
                .expect("Current confined input checks");
            store
                .update(cx, |store, cx| {
                    store.admit_reviewed_metadata_import(reviewed, bytes.clone(), cx)
                })
                .await
                .expect("Actual reviewed replacement");
            store.read_with(cx, |store, cx| {
                let (snapshot, current) = store.cached_snapshot(cx).expect("Fresh selected import retained");
                assert_eq!(current, !retain_neighbor, "Replacing the only stale scope can establish current data; untouched stale scopes must never be promoted");
                if let Some(neighbor) = &neighbor { assert!(snapshot.workspaces.contains(neighbor), "Unrelated snapshot preserved exactly"); }
            });
        }
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/other/member-one/src/lib.rs"))
                .await
                .expect("Unrelated source preserved"),
            "preserved source"
        );
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_admits_exact_inputs_preserving_other_snapshot(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let old_bytes = include_bytes!("../test_data/cargo_workspace/workspace-v1.json");
        let unrelated = parse_metadata(
            String::from_utf8_lossy(old_bytes)
                .replace("/workspace", "/other")
                .as_bytes(),
        )
        .expect("Other metadata");
        let other = cx
            .update(|cx| {
                workspace_from_metadata(&unrelated, |path| {
                    project.read(cx).find_project_path(path, cx)
                })
            })
            .expect("Other workspace");
        store.update(cx, |store, cx| {
            store.cached_snapshot = Some(CargoWorkspaceSnapshot {
                revision: store.revision,
                input_fingerprint: store.current_input_fingerprint(cx),
                workspaces: vec![other.clone()],
                failures: Vec::new(),
                completeness: CargoSnapshotCompleteness::Partial,
            });
            store.cached_generation = store.revision;
        });
        let inputs = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("Passive owner review");
        inputs
            .preflight_filesystem(filesystem.as_ref())
            .await
            .expect("Confined source preflight");
        let process_calls = local_cargo_process_calls_for_test();
        let current = store
            .update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(inputs, old_bytes.to_vec(), cx)
            })
            .await
            .expect("Owner admission");
        assert!(
            current
                .dependency_graph
                .as_ref()
                .is_some_and(|graph| !graph.nodes.is_empty())
        );
        assert_eq!(
            current.configuration.host_compiler.status,
            CargoHostCompilerStatus::Unknown
        );
        assert_eq!(
            current.configuration.completeness,
            CargoConfigurationCompleteness::Partial
        );
        assert_eq!(
            local_cargo_process_calls_for_test(),
            process_calls,
            "Import never invokes Cargo or rustc"
        );
        store.read_with(cx, |store, cx| {
            let (cached, current) = store.cached_snapshot(cx).expect("Existing snapshot owner");
            assert!(current);
            assert_eq!(cached.workspaces.len(), 2);
            assert!(
                cached.workspaces.contains(&other),
                "Unrelated workspace preserved exactly"
            );
            assert_eq!(cached.completeness, CargoSnapshotCompleteness::Partial);
        });
        assert_eq!(
            filesystem
                .load(std::path::Path::new("/workspace/member-one/src/lib.rs"))
                .await
                .expect("Source"),
            "preserved source"
        );
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_replacement_budget_preserves_valid_cache(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let bytes = include_bytes!("../test_data/cargo_workspace/workspace-v1.json");
        let inputs = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Inputs");
        let workspace = reviewed_cargo_workspace(&inputs, bytes, filesystem.as_ref())
            .await
            .expect("Bounded incoming workspace")
            .0;
        let old = store.update(cx, |store, cx| {
            let previous = CargoWorkspaceSnapshot {
                revision: store.revision,
                input_fingerprint: store.current_input_fingerprint(cx),
                workspaces: (0..128)
                    .map(|index| {
                        let mut retained = workspace.clone();
                        retained.key.root.path = Arc::from(
                            RelPath::from_unix_str(&format!("existing-{index}"))
                                .expect("Fixture root"),
                        );
                        retained
                    })
                    .collect(),
                failures: Vec::new(),
                completeness: CargoSnapshotCompleteness::Partial,
            };
            validate_cargo_snapshot_bounds(&previous).expect("Individually valid existing cache");
            store.cached_snapshot = Some(previous.clone());
            store.cached_generation = store.revision;
            previous
        });
        assert!(
            store
                .update(cx, |store, cx| store.admit_reviewed_metadata_import(
                    inputs,
                    bytes.to_vec(),
                    cx
                ))
                .await
                .is_err(),
            "Individually bounded inputs must not exceed the merged retained limit"
        );
        store.read_with(cx, |store, cx| {
            assert_eq!(
                store
                    .cached_snapshot(cx)
                    .expect("Original cache preserved")
                    .0,
                &old
            );
        });
        filesystem
            .atomic_write(
                PathBuf::from("/workspace/Cargo.toml"),
                "PRIVATE_SOURCE_SENTINEL = \"unterminated".into(),
            )
            .await
            .expect("Invalid declaration");
        cx.run_until_parked();
        let invalid = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("Current metadata plan");
        let error = match reviewed_cargo_workspace(&invalid, bytes, filesystem.as_ref()).await {
            Err(error) => error,
            Ok(_) => panic!("Malformed declaration must not admit a graph"),
        };
        assert!(
            !error.to_string().contains("PRIVATE_SOURCE_SENTINEL"),
            "Source content cannot enter an operation failure summary"
        );
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_stale_revoked_malformed_and_cancelled_import(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let bytes = include_bytes!("../test_data/cargo_workspace/workspace-v1.json").to_vec();
        let prepare = |cx: &App| {
            store
                .read(cx)
                .prepare_reviewed_metadata_import(manifest.clone(), cx)
        };
        let inputs = cx.update(|cx| prepare(cx)).expect("Inputs");
        filesystem
            .atomic_write(
                PathBuf::from("/workspace/member-one/Cargo.toml"),
                "[package]\nname = \"member-one\"\nversion = \"0.3.0\"\n".into(),
            )
            .await
            .expect("Changed declaration");
        cx.run_until_parked();
        assert!(
            store
                .update(cx, |store, cx| store.admit_reviewed_metadata_import(
                    inputs,
                    bytes.clone(),
                    cx
                ))
                .await
                .is_err()
        );
        for bad in [
            bytes.clone(),
            b"{}".to_vec(),
            vec![b'x'; crate::cargo_workspace::MAX_CARGO_SNAPSHOT_BYTES + 1],
            String::from_utf8_lossy(&bytes)
                .replace("/workspace", "/unselected")
                .into_bytes(),
        ] {
            let inputs = cx.update(|cx| prepare(cx)).expect("Current inputs");
            assert!(
                store
                    .update(cx, |store, cx| store
                        .admit_reviewed_metadata_import(inputs, bad, cx))
                    .await
                    .is_err()
            );
            assert!(store.read_with(cx, |store, cx| store.cached_snapshot(cx).is_none()));
        }
        filesystem
            .atomic_write(
                PathBuf::from("/workspace/member-one/Cargo.toml"),
                "[package]\nname = \"member-one\"\nversion = \"0.1.0\"\n".into(),
            )
            .await
            .expect("Restore valid fixture for cancellation");
        cx.run_until_parked();
        let inputs = cx.update(|cx| prepare(cx)).expect("Fresh before disposal");
        let pending = store.update(cx, |store, cx| {
            store.admit_reviewed_metadata_import(inputs, bytes.clone(), cx)
        });
        drop(pending);
        cx.run_until_parked();
        assert!(
            store.read_with(cx, |store, cx| store.cached_snapshot(cx).is_none()),
            "Dropped import never mutates cache or replays"
        );
        let inputs = cx
            .update(|cx| prepare(cx))
            .expect("Before settings revocation");
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"vim_mode":true}"#, cx)
                    .expect("Changed review authority");
            })
        });
        assert!(
            store
                .update(cx, |store, cx| store.admit_reviewed_metadata_import(
                    inputs,
                    bytes.clone(),
                    cx
                ))
                .await
                .is_err()
        );
        let inputs = cx
            .update(|cx| prepare(cx))
            .expect("Before trust revocation");
        let trees = project.read_with(cx, |owner, _| owner.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust"));
        trust.update(cx, |trust, cx| {
            trust.restrict(
                trees.downgrade(),
                collections::HashSet::from_iter([crate::trusted_worktrees::PathTrust::Worktree(
                    manifest.worktree_id,
                )]),
                cx,
            )
        });
        assert!(
            store
                .update(cx, |store, cx| store
                    .admit_reviewed_metadata_import(inputs, bytes, cx))
                .await
                .is_err()
        );
        assert!(store.read_with(cx, |store, cx| store.cached_snapshot(cx).is_none()));
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn doctor_dependency_collection_cargo_symlink_and_private_inputs_are_not_public_proof(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let inputs = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Initial input plan");
        filesystem
            .create_symlink(
                std::path::Path::new("/workspace/Cargo.lock"),
                PathBuf::from("/workspace/member-one/Cargo.toml"),
            )
            .await
            .expect("Untrusted input");
        cx.run_until_parked();
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .revalidate_reviewed_metadata_import(&inputs, cx))
                .is_err()
        );
        let linked = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("Indexed plan");
        assert!(
            linked
                .preflight_filesystem(filesystem.as_ref())
                .await
                .is_err(),
            "Internal symlink must block before resolver dispatch"
        );
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"private_files":["Cargo.toml"]}"#, cx)
                    .expect("Private input policy");
            })
        });
        cx.run_until_parked();
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .prepare_reviewed_metadata_import(manifest, cx))
                .is_err()
        );
    }

    #[gpui::test]
    async fn doctor_cargo_graph_wire_roundtrip_old_host_and_malformed_provenance(
        _cx: &mut TestAppContext,
    ) {
        let snapshot = collect_snapshot(
            12,
            34,
            vec![candidate(1, "/workspace", "/workspace/Cargo.toml", true)],
            Arc::new(RoutingRunner {
                requests: Arc::new(Mutex::new(Vec::new())),
            }),
            successful_configuration_probe(),
        )
        .await
        .expect("Existing Cargo owner")
        .snapshot;
        let response = snapshot_to_proto(snapshot.clone(), 56);
        assert_eq!(
            snapshot_from_proto(response.clone()).expect("Versioned graph wire"),
            snapshot
        );
        let mut truncated = response.clone();
        truncated
            .workspaces
            .first_mut()
            .expect("Workspace")
            .dependency_graph
            .as_mut()
            .expect("Graph")
            .partial = true;
        let truncated = snapshot_from_proto(truncated).expect("Partial remote graph");
        let truncated_graph = truncated
            .workspaces
            .first()
            .expect("Workspace")
            .dependency_graph
            .as_ref()
            .expect("Partial graph retained");
        assert_eq!(
            truncated_graph.completeness,
            CargoSnapshotCompleteness::Partial
        );
        assert_eq!(
            truncated_graph.nodes,
            snapshot
                .workspaces
                .first()
                .expect("Workspace")
                .dependency_graph
                .as_ref()
                .expect("Original graph")
                .nodes
        );
        let mut old = response.clone();
        old.workspaces
            .first_mut()
            .expect("Workspace")
            .dependency_graph = None;
        assert!(
            snapshot_from_proto(old)
                .expect("Older host remains compatible")
                .workspaces
                .first()
                .expect("Workspace")
                .dependency_graph
                .is_none()
        );
        let mut malformed = response.clone();
        malformed
            .workspaces
            .first_mut()
            .expect("Workspace")
            .dependency_graph
            .as_mut()
            .expect("Graph")
            .format_version = 99;
        assert!(snapshot_from_proto(malformed).is_err());
        let mut spoofed = response;
        let node = spoofed
            .workspaces
            .first_mut()
            .expect("Workspace")
            .dependency_graph
            .as_mut()
            .expect("Graph")
            .nodes
            .first_mut()
            .expect("Node");
        node.crates_io = true;
        node.source_kind = 2;
        assert!(
            snapshot_from_proto(spoofed).is_err(),
            "Private source fingerprints cannot authorize public coordinates"
        );
    }

    #[gpui::test]
    async fn doctor_cargo_graph_process_stream_bounds_exact_limit_and_overflow(
        _cx: &mut TestAppContext,
    ) {
        let exact = bounded_cargo_metadata_stream(futures::io::Cursor::new(vec![b'x'; 4096]), 4096)
            .await
            .expect("Exact bound");
        assert_eq!(exact.len(), 4096);
        assert!(
            bounded_cargo_metadata_stream(futures::io::Cursor::new(vec![b'x'; 4097]), 4096)
                .await
                .is_err()
        );
    }
    #[cfg(unix)]
    #[gpui::test]
    fn native_collector_probe_actual_version_preserves_cwd_env_and_bounds(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        let directory = tempfile::tempdir().expect("collector version fixture");
        let cwd = std::fs::canonicalize(directory.path()).expect("canonical fixture");
        let executable = cwd.join("collector");
        let ordered = HashMap::from_iter([("AB".into(), "C".into()), ("A".into(), "BC".into())]);
        let reversed = HashMap::from_iter([("A".into(), "BC".into()), ("AB".into(), "C".into())]);
        assert_eq!(
            fingerprint_native_probe_environment(&ordered).expect("bounded environment"),
            fingerprint_native_probe_environment(&reversed).expect("same bounded environment")
        );
        let changed = HashMap::from_iter([("AB".into(), "BC".into()), ("A".into(), "C".into())]);
        assert_ne!(
            fingerprint_native_probe_environment(&ordered).expect("bounded environment"),
            fingerprint_native_probe_environment(&changed).expect("changed bounded environment")
        );
        for (body, expected_error) in [
            (
                "[ \"$#\" = 1 ] && [ \"$1\" = --version ] && [ \"$PROBE_EXPLICIT\" = yes ] && [ \"$(pwd)\" = \"$PROBE_CWD\" ] || exit 31\nprintf 'samply 0.13.1\\n'\n",
                None,
            ),
            (
                "printf 'samply 0.13.2\\n'\n",
                Some("actual installed Samply 0.13.1"),
            ),
            ("printf 'samply 0.13.1\\n'\nexit 19\n", Some("unsuccessful")),
            ("/usr/bin/head -c 4097 /dev/zero\n", Some("stdout exceeded")),
            (
                "/usr/bin/head -c 4097 /dev/zero >&2\nprintf 'samply 0.13.1\\n'\n",
                Some("stderr exceeded"),
            ),
        ] {
            std::fs::write(&executable, format!("#!/bin/sh\n{body}")).expect("collector fixture");
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))
                .expect("executable fixture");
            let executor = cx.background_executor.clone();
            let path = executable.clone();
            let cwd = cwd.clone();
            let environment = HashMap::from_iter([
                ("PROBE_EXPLICIT".into(), "yes".into()),
                ("PROBE_CWD".into(), cwd.to_string_lossy().into_owned()),
            ]);
            let mut task = cx.background_executor.spawn(async move {
                run_native_collector_version(&path, &cwd, &environment, &executor).await
            });
            let result = poll_cargo_process(cx, || (&mut task).now_or_never());
            match expected_error {
                Some(fragment) => assert!(
                    result
                        .expect_err("probe denied")
                        .to_string()
                        .contains(fragment)
                ),
                None => result.expect("actual collector argument/env/cwd receiver"),
            };
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn native_collector_probe_timeout_and_drop_retire_actual_child(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        for timed_out in [false, true] {
            let directory = tempfile::tempdir().expect("held collector fixture");
            let cwd = std::fs::canonicalize(directory.path()).expect("canonical held fixture");
            let executable = cwd.join("collector");
            let pid = cwd.join("child.pid");
            std::fs::write(
                &executable,
                "#!/bin/sh\nprintf '%s' \"$$\" > child.pid\nexec /bin/sleep 30\n",
            )
            .expect("held collector script");
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))
                .expect("held executable");
            let executor = cx.background_executor.clone();
            let mut task = cx.background_executor.spawn(async move {
                run_native_collector_version(&executable, &cwd, &HashMap::default(), &executor)
                    .await
            });
            let process_id = poll_cargo_process(cx, || match std::fs::read_to_string(&pid) {
                Ok(value) if !value.is_empty() => Some(value),
                Ok(_) => None,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => panic!("held PID read failed: {error}"),
            });
            if timed_out {
                cx.executor()
                    .advance_clock(CARGO_CONFIGURATION_PROBE_TIMEOUT);
                let result = poll_cargo_process(cx, || (&mut task).now_or_never());
                assert!(
                    result
                        .expect_err("version timeout")
                        .to_string()
                        .contains("timed out")
                );
            } else {
                drop(task);
            }
            require_cargo_process_exit(cx, process_id);
        }
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn native_collector_probe_source_seals_reject_delayed_watcher_edits(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("reviewed probe config");
        let entries = store
            .read_with(cx, |store, cx| {
                store.native_probe_source_entries(&reviewed, cx)
            })
            .expect("source admission");
        let (sources, directories) =
            seal_native_probe_sources(filesystem.as_ref(), &reviewed, &entries)
                .await
                .expect("confined source seals");
        filesystem.pause_events();
        filesystem
            .write(
                Path::new("/workspace/member-one/src/lib.rs"),
                b"changed source",
            )
            .await
            .expect("unobserved source edit");
        assert_eq!(
            entries,
            store
                .read_with(cx, |store, cx| store
                    .native_probe_source_entries(&reviewed, cx))
                .expect("watcher still paused")
        );
        assert!(
            revalidate_native_probe_filesystem(filesystem.as_ref(), &sources, &directories)
                .await
                .is_err()
        );
        filesystem.unpause_events_and_flush();
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn native_collector_probe_directory_seals_allow_ignored_output_reject_new_source(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        filesystem.insert_tree("/workspace",serde_json::json!({".gitignore":"target/\n","target":{"capture.json":"old output"}})).await;
        cx.run_until_parked();
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest, cx)
            })
            .expect("reviewed config");
        let entries = store
            .read_with(cx, |store, cx| {
                store.native_probe_source_entries(&reviewed, cx)
            })
            .expect("visible source admission");
        assert!(
            !entries
                .iter()
                .any(|entry| entry.path.as_unix_str().starts_with("target/"))
        );
        let (sources, directories) =
            seal_native_probe_sources(filesystem.as_ref(), &reviewed, &entries)
                .await
                .expect("source seals");
        filesystem.pause_events();
        filesystem
            .write(
                Path::new("/workspace/target/capture.json"),
                b"new generated output",
            )
            .await
            .expect("ignored output refresh");
        revalidate_native_probe_filesystem(filesystem.as_ref(), &sources, &directories)
            .await
            .expect("ignored output is outside source seal");
        filesystem
            .write(Path::new("/workspace/member-one/src/new.rs"), b"new source")
            .await
            .expect("unobserved source addition");
        assert!(
            revalidate_native_probe_filesystem(filesystem.as_ref(), &sources, &directories)
                .await
                .is_err()
        );
        filesystem.unpause_events_and_flush();
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn native_collector_probe_admission_rejects_current_trust_and_config_changes(
        cx: &mut TestAppContext,
    ) {
        let (project, filesystem, manifest) = doctor_cargo_import_fixture(cx).await;
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::ffi::OsStringExt as _;
            for cwd in [
                PathBuf::from(format!("/workspace/member-one/{}", "x".repeat(4096))),
                PathBuf::from("/workspace/member-one/new\nline"),
                PathBuf::from(std::ffi::OsString::from_vec(
                    b"/workspace/member-one/invalid\xff".to_vec(),
                )),
            ] {
                let error = store.update(cx, |store, cx| {
                    store
                        .probe_native_collector(
                            manifest.clone(),
                            PathBuf::from("/installed/samply"),
                            cwd,
                            HashMap::default(),
                            cx,
                        )
                        .err()
                        .expect("invalid cwd is denied before an async probe is created")
                });
                assert!(
                    error
                        .to_string()
                        .contains("path or Task environment is invalid")
                );
            }
        }
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("current config");
        filesystem
            .write(
                Path::new("/workspace/member-one/Cargo.toml"),
                b"[package]\nname='replaced'\nversion='0.1.0'\n",
            )
            .await
            .expect("config replacement");
        cx.run_until_parked();
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .native_probe_source_entries(&reviewed, cx))
                .is_err()
        );
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("fresh config");
        let trees = project.read_with(cx, |project, _| project.worktree_store());
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
        trust.update(cx, |trust, cx| {
            trust.restrict(
                trees.downgrade(),
                HashSet::from_iter([crate::trusted_worktrees::PathTrust::Worktree(
                    manifest.worktree_id,
                )]),
                cx,
            )
        });
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .native_probe_source_entries(&reviewed, cx))
                .is_err()
        );
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    struct NativeCollectorPublicFixture {
        _directory: tempfile::TempDir,
        project: Entity<crate::Project>,
        store: Entity<CargoWorkspaceStore>,
        manifest: ProjectPath,
        root: PathBuf,
        collector: PathBuf,
        log: PathBuf,
        environment: HashMap<String, String>,
    }

    #[cfg(all(target_os = "macos", feature = "test-support", feature = "rust-tests"))]
    struct NativeFixtureDiscoveryRunner {
        executor: BackgroundExecutor,
        target_directory: PathBuf,
    }

    #[cfg(all(target_os = "macos", feature = "test-support", feature = "rust-tests"))]
    impl crate::rust_test_provider::RustTestDiscoveryRunner for NativeFixtureDiscoveryRunner {
        fn discover(
            &self,
            mut request: crate::rust_test_provider::RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<
            'static,
            Result<crate::rust_test_provider::RustTestDiscoveryCapture>,
        > {
            // Discovery must not create the output whose absence native clean-first tests review.
            request.environment.get_or_insert_default().insert(
                "CARGO_TARGET_DIR".into(),
                self.target_directory
                    .to_str()
                    .expect("owned discovery target")
                    .into(),
            );
            crate::rust_test_provider::RustTestProviderStore::process_discovery_for_test(
                request,
                self.executor.clone(),
            )
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    async fn refresh_native_fixture_metadata(
        project: &Entity<crate::Project>,
        manifest: &ProjectPath,
        root: &Path,
        cx: &mut TestAppContext,
    ) -> CargoWorkspaceModel {
        let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        #[cfg(feature = "rust-tests")]
        {
            let (discovery, filesystem, worktree) = project.read_with(cx, |project, cx| {
                (
                    project.rust_test_provider_store().clone(),
                    project.fs().clone(),
                    project
                        .worktrees(cx)
                        .find(|tree| tree.read(cx).id() == manifest.worktree_id)
                        .expect("original selected Worktree"),
                )
            });
            {
                let deadline = cx.background_executor.timer(Duration::from_secs(10));
                let settling = async {
                    while !discovery.read_with(cx, |store, _| store.discovery_idle_for_test()) {
                        cx.background_executor
                            .timer(Duration::from_millis(10))
                            .await;
                    }
                    let lock_path: Arc<RelPath> = Arc::from(
                        RelPath::from_unix_str("Cargo.lock").expect("literal Cargo input"),
                    );
                    let metadata = filesystem
                        .metadata(&root.join("Cargo.lock"))
                        .await
                        .expect("actual discovery lock metadata")
                        .expect("actual discovery creates the selected Cargo.lock");
                    let entry = worktree
                        .update(cx, |worktree, cx| {
                            worktree
                                .as_local()
                                .expect("original local Worktree")
                                .refresh_entry(lock_path.clone(), None, cx)
                        })
                        .await
                        .expect("actual discovery lock publication")
                        .expect("actual discovery lock indexed");
                    assert_eq!(
                        (entry.inode, entry.mtime, entry.size),
                        (metadata.inode, Some(metadata.mtime), metadata.len),
                        "The actual compiler lock must be indexed before reviewed metadata admission"
                    );
                    cx.run_until_parked();
                    while !discovery.read_with(cx, |store, _| store.discovery_idle_for_test())
                        || worktree.read_with(cx, |worktree, _| {
                            worktree.scan_id() != worktree.completed_scan_id()
                        })
                    {
                        cx.background_executor
                            .timer(Duration::from_millis(10))
                            .await;
                    }
                    let metadata = filesystem
                        .metadata(&root.join("Cargo.lock"))
                        .await
                        .expect("retired discovery lock metadata")
                        .expect("original discovery lock remains present");
                    worktree.read_with(cx, |worktree, _| {
                        let indexed = worktree.entry_for_path(&lock_path).expect("published Cargo.lock remains indexed");
                        assert_eq!((indexed.inode, indexed.mtime, indexed.size), (metadata.inode, Some(metadata.mtime), metadata.len), "Retired discovery and original Worktree must agree on the final lock identity");
                        assert_eq!(worktree.scan_id(), worktree.completed_scan_id());
                    });
                };
                futures::pin_mut!(settling);
                match futures::future::select(settling, deadline).await {
                    Either::Left(((), _)) => {}
                    Either::Right(_) => panic!(
                        "Original Rust discovery did not retire before native source admission"
                    ),
                }
            }
            assert!(discovery.read_with(cx, |store, _| store.discovery_idle_for_test()));
        }
        let reviewed = store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(manifest.clone(), cx)
            })
            .expect("fresh source admission after original discovery retirement");
        let bytes = include_str!("../test_data/cargo_workspace/workspace-v1.json")
            .replace("/workspace", root.to_str().expect("fixture root"))
            .into_bytes();
        store
            .update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(reviewed, bytes, cx)
            })
            .await
            .expect("fresh metadata through original public admission owner")
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    async fn native_collector_public_fixture(
        cx: &mut TestAppContext,
    ) -> NativeCollectorPublicFixture {
        native_collector_public_fixture_with_output(cx, true).await
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    async fn native_collector_public_fixture_with_output(
        cx: &mut TestAppContext,
        create_output: bool,
    ) -> NativeCollectorPublicFixture {
        use crate::trusted_worktrees::{DbTrustedPaths, PathTrust};
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::trusted_worktrees::init(DbTrustedPaths::default(), cx);
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{}", cx)
                    .expect("observed settings");
            });
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            );
        });
        let directory = tempfile::tempdir().expect("public probe fixture");
        let directory_path = std::fs::canonicalize(directory.path()).expect("canonical fixture");
        let root = directory_path.join("workspace with spaces");
        for path in ["member-one/src", "member-two/src"] {
            std::fs::create_dir_all(root.join(path)).expect("fixture directories");
        }
        if create_output {
            std::fs::create_dir_all(root.join("target")).expect("preexisting fixture output");
        }
        for (path, contents) in [
            (
                "Cargo.toml",
                "[workspace]\nmembers=['member-one','member-two']\nresolver='2'\n",
            ),
            (".gitignore", "target/\n"),
            (
                "member-one/Cargo.toml",
                "[package]\nname='member-one'\nversion='0.1.0'\nedition='2024'\n",
            ),
            ("member-one/src/lib.rs", "pub fn retained() -> u32 { 7 }\n"),
            ("member-one/src/main.rs", "fn main() {}\n"),
            (
                "member-two/Cargo.toml",
                "[package]\nname='member-two'\nversion='0.2.0'\nedition='2021'\n",
            ),
            ("member-two/src/lib.rs", "pub fn other() -> u32 { 11 }\n"),
        ] {
            std::fs::write(root.join(path), contents).expect("fixture source/configuration");
        }
        let installed = directory_path.join("installed");
        std::fs::create_dir(&installed).expect("collector directory");
        let collector = installed.join("collector");
        std::fs::write(&collector,"#!/bin/sh\n[ \"$#\" = 1 ] && [ \"$1\" = --version ] && [ \"$PROBE_EXPLICIT\" = admitted ] && [ \"$(pwd)\" = \"$PROBE_CWD\" ] || exit 31\nprintf '%s\\n' \"$1:$PROBE_EXPLICIT\" >> \"$PROBE_LOG\"\nprintf 'samply 0.13.1\\n'\n").expect("controlled collector");
        std::fs::set_permissions(&collector, std::fs::Permissions::from_mode(0o700))
            .expect("executable collector");
        let log = installed.join("probe.log");
        let environment = HashMap::from_iter([
            ("PROBE_EXPLICIT".into(), "admitted".into()),
            (
                "PROBE_CWD".into(),
                root.to_str().expect("UTF-8 root").to_owned(),
            ),
            (
                "PROBE_LOG".into(),
                log.to_str().expect("UTF-8 log").to_owned(),
            ),
        ]);
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let project =
            crate::Project::test_with_worktree_trust(filesystem, std::iter::empty::<&Path>(), cx)
                .await;
        #[cfg(feature = "rust-tests")]
        {
            let target_directory = directory_path.join("discovery-target");
            std::fs::create_dir(&target_directory).expect("owned discovery output directory");
            assert!(!target_directory.starts_with(&root));
            project.update(cx, |project, cx| {
                project.rust_test_provider_store().update(cx, |store, cx| {
                    store
                        .set_discovery_runner_for_test(Arc::new(NativeFixtureDiscoveryRunner {
                            executor: cx.background_executor().clone(),
                            target_directory,
                        }))
                        .expect("same real discovery owner with isolated output")
                })
            });
        }
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(&root, true, cx)
            })
            .await
            .expect("original selected worktree");
        worktree
            .read_with(cx, |tree, _| {
                tree.as_local()
                    .expect("original local worktree")
                    .scan_complete()
            })
            .await;
        let (trees, store, worktree_store) = project.read_with(cx, |project, cx| {
            (
                project
                    .visible_worktrees(cx)
                    .map(|tree| tree.read(cx).id())
                    .collect::<Vec<_>>(),
                project.cargo_workspace_store().clone(),
                project.worktree_store(),
            )
        });
        let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter(trees.iter().copied().map(PathTrust::Worktree)),
                cx,
            );
        });
        cx.run_until_parked();
        let manifest = ProjectPath {
            worktree_id: *trees.first().expect("fixture worktree"),
            path: Arc::from(RelPath::from_unix_str("Cargo.toml").expect("manifest")),
        };
        let workspace = refresh_native_fixture_metadata(&project, &manifest, &root, cx).await;
        assert_eq!(workspace.members.len(), 2);
        assert_eq!(
            root.join("target").exists(),
            create_output,
            "real test discovery must preserve the native output precondition"
        );
        store.read_with(cx, |store, cx| {
            assert!(
                store
                    .cached_snapshot(cx)
                    .is_some_and(|(_, current)| current)
            );
        });
        NativeCollectorPublicFixture {
            _directory: directory,
            project,
            store,
            manifest,
            root,
            collector,
            log,
            environment,
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_clean_first_worktree_output_admission_is_fallible_and_confined(
        cx: &mut TestAppContext,
    ) {
        for scenario in [
            "fresh",
            "unknown-ignore",
            "private",
            "outside-root",
            "oversized-rule",
            "unknown-repository",
        ] {
            let fixture = native_collector_public_fixture_with_output(cx, false).await;
            assert!(!fixture.root.join("target").exists());
            let facts = fixture.store.read_with(cx, |store, cx| {
                store
                    .metadata_facts_for_current_snapshot(cx)
                    .expect("current private facts")[0]
                    .clone()
            });
            assert_eq!(
                facts.target_directory,
                fixture.root.join("target"),
                "only exact metadata output is considered"
            );
            match scenario {
                "unknown-ignore" => std::fs::write(fixture.root.join(".gitignore"), b"")
                    .expect("remove affirmative ignore"),
                "oversized-rule" => {
                    std::fs::write(fixture.root.join(".gitignore"), vec![b'#'; 64 * 1024 + 1])
                        .expect("oversized actual rule")
                }
                "unknown-repository" => std::fs::create_dir(fixture.root.join(".git"))
                    .expect("unindexed repository authority"),
                "private" => {
                    cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                            settings
                                .set_user_settings(r#"{"private_files":["target"],"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx)
                                .expect("private output policy")
                        })
                    });
                    assert!(
                        cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
                        "Non-flag settings changes must retain native profiling opt-in"
                    );
                    cx.run_until_parked();
                }
                _ => {}
            }
            let trees = fixture
                .project
                .read_with(cx, |project, _| project.worktree_store());
            let tree = trees
                .read_with(cx, |trees, cx| {
                    trees.worktree_for_id(fixture.manifest.worktree_id, cx)
                })
                .expect("actual Worktree owner");
            let relative = if scenario == "outside-root" {
                "outside"
            } else {
                "target"
            };
            let path = Arc::from(RelPath::from_unix_str(relative).expect("relative output"));
            if scenario == "outside-root" {
                let outside = fixture
                    .root
                    .parent()
                    .expect("fixture parent")
                    .join("outside");
                assert!(
                    fixture
                        .project
                        .read_with(cx, |project, cx| project.find_project_path(&outside, cx))
                        .is_none()
                );
                assert!(!outside.exists());
                assert!(!fixture.root.join("target").exists());
                continue;
            }
            let created = tree
                .update(cx, |tree, cx| {
                    tree.as_local()
                        .expect("local Worktree")
                        .create_ignored_directory_for_task(path, cx)
                })
                .await;
            if scenario == "fresh" {
                created.expect("actual clean-first ignored directory creation");
                assert!(fixture.root.join("target").is_dir());
                tree.read_with(cx, |tree, _| {
                    let entry = tree
                        .entry_for_path(RelPath::from_unix_str("target").expect("output"))
                        .expect("indexed created root");
                    assert!(
                        entry.is_dir()
                            && entry.is_ignored
                            && !entry.is_private
                            && !entry.is_external
                    );
                });
            } else {
                assert!(created.is_err(), "rejected {scenario}");
                assert!(!fixture.root.join("target").exists());
            }
            assert!(
                !fixture
                    .root
                    .parent()
                    .expect("fixture parent")
                    .join("outside")
                    .exists()
            );
            assert!(
                !fixture.log.exists() || std::fs::read(&fixture.log).expect("probe log").is_empty(),
                "output preparation cannot launch collector"
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_clean_first_build_preserves_buffer_admission_and_fresh_save_recovery(
        cx: &mut TestAppContext,
    ) {
        use futures::StreamExt as _;
        use std::os::unix::fs::PermissionsExt as _;
        let fixture = native_collector_public_fixture_with_output(cx, false).await;
        let (workspace, facts) = fixture.store.read_with(cx, |store, cx| {
            (
                store.cached_snapshot(cx).expect("snapshot").0.workspaces[0].clone(),
                store
                    .metadata_facts_for_current_snapshot(cx)
                    .expect("private facts")[0]
                    .clone(),
            )
        });
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("member");
        let target = member
            .targets
            .iter()
            .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
            .expect("binary");
        let (_, _, arguments) = native_build_selection(&facts, &workspace, &member.id, target)
            .expect("actual metadata recipe");
        let program = fixture
            .collector
            .parent()
            .expect("installed fixture")
            .join("cargo");
        std::fs::write(&program, "#!/bin/sh\nexit 0\n").expect("declared Build program");
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700))
            .expect("program permissions");
        let source = fixture
            .project
            .update(cx, |project, cx| {
                project.open_buffer(
                    project
                        .find_project_path(fixture.root.join("member-one/src/main.rs"), cx)
                        .expect("source"),
                    cx,
                )
            })
            .await
            .expect("opened source");
        refresh_native_fixture_metadata(&fixture.project, &fixture.manifest, &fixture.root, cx)
            .await;
        let launch = task::SpawnInTerminal {
            id: task::TaskId("clean-first-buffer".into()),
            command: Some(program.to_str().expect("program").into()),
            literal_prefix_args: arguments.len(),
            args: arguments,
            cwd: Some(fixture.root.clone()),
            ..Default::default()
        };
        let pending = fixture
            .store
            .update(cx, |store, cx| {
                store.prepare_native_cargo_build_intent_for_launch(
                    fixture.project.clone(),
                    uuid::Uuid::new_v4(),
                    &launch,
                    cx,
                )
            })
            .expect("actual captured admission");
        source.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "// held dirty source\n")], None, cx)
        });
        assert!(
            pending.await.is_err(),
            "original Buffer version must not be resealed after creation review"
        );
        assert!(!fixture.root.join("target").exists());
        fixture
            .project
            .update(cx, |project, cx| project.save_buffer(source.clone(), cx))
            .await
            .expect("actual fresh save");
        let trees = fixture
            .project
            .read_with(cx, |project, _| project.worktree_store());
        let tree = trees
            .read_with(cx, |trees, cx| {
                trees.worktree_for_id(fixture.manifest.worktree_id, cx)
            })
            .expect("worktree");
        let mut refreshed = tree.read_with(cx, |tree, _| {
            tree.as_local()
                .expect("local")
                .refresh_entries_for_paths(vec![Arc::from(
                    RelPath::from_unix_str("member-one/src/main.rs").expect("saved source"),
                )])
        });
        refreshed.next().await.expect("saved source refresh");
        refresh_native_fixture_metadata(&fixture.project, &fixture.manifest, &fixture.root, cx)
            .await;
        let fresh = fixture
            .store
            .update(cx, |store, cx| {
                store.prepare_native_cargo_build_intent_for_launch(
                    fixture.project.clone(),
                    uuid::Uuid::new_v4(),
                    &launch,
                    cx,
                )
            })
            .expect("fresh declared admission")
            .await
            .expect("clean-first creation after genuine save");
        assert_eq!(fresh.output_root(), fixture.root.join("target"));
        assert!(fixture.root.join("target").is_dir());
        assert!(
            !fixture.log.exists() || std::fs::read(&fixture.log).expect("probe log").is_empty()
        );
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[test]
    fn native_build_intent_helper_process_entry() {
        let Some(socket) = std::env::var_os("ZED_NATIVE_BUILD_TEST_SOCKET") else {
            return;
        };
        let program =
            std::env::var_os("ZED_NATIVE_BUILD_TEST_PROGRAM").expect("controlled program");
        let arguments: Vec<String> = serde_json::from_str(
            &std::env::var("ZED_NATIVE_BUILD_TEST_ARGUMENTS").expect("bounded arguments"),
        )
        .expect("arguments");
        let mut command = std::process::Command::new(program);
        command.args(arguments);
        if std::env::var_os("ZED_NATIVE_BUILD_TEST_CHECKED_NATIVE").is_some() {
            let error =
                util::macos_task_completion::register_and_exec_checked_native_profile_capture(
                    Path::new(&socket),
                    &mut command,
                );
            panic!("checked native controlled helper failed before exec: {error}");
        }
        if std::env::var_os("ZED_NATIVE_BUILD_TEST_NATIVE").is_some() {
            let error = util::macos_task_completion::register_and_exec_native_profile_capture(
                Path::new(&socket),
                &mut command,
            );
            panic!("native controlled helper failed before exec: {error}");
        }
        let status = util::macos_task_completion::register_and_capture_build_stdout(
            Path::new(&socket),
            command,
        )
        .expect("checked producing child");
        util::macos_task_completion::exit_with_observed_task_status(status);
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_selected_package_directory_rejects_wrong_host_and_private_source(
        cx: &mut TestAppContext,
    ) {
        let fixture = native_collector_public_fixture(cx).await;
        let (workspace, facts, entries) = fixture.store.read_with(cx, |store, cx| {
            let reviewed = store
                .prepare_reviewed_metadata_import(fixture.manifest.clone(), cx)
                .expect("genuine current source admission");
            (
                store
                    .cached_snapshot(cx)
                    .expect("current snapshot")
                    .0
                    .workspaces[0]
                    .clone(),
                store
                    .metadata_facts_for_current_snapshot(cx)
                    .expect("current private facts")[0]
                    .clone(),
                store
                    .native_probe_source_entries(&reviewed, cx)
                    .expect("actual indexed sources"),
            )
        });
        let member = workspace
            .members
            .iter()
            .find(|member| member.name == "member-one")
            .expect("nested selected package");
        let target = member
            .targets
            .iter()
            .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
            .expect("selected binary");
        let (selection, _, _) = native_build_selection(&facts, &workspace, &member.id, target)
            .expect("exact selection");
        let reviewed = native_selected_package_directory(
            &facts,
            &workspace,
            &selection,
            &fixture.root,
            &entries,
        )
        .expect("owner-derived nested package directory");
        assert_eq!(reviewed.0, fixture.root.join("member-one"));
        assert_ne!(reviewed.0, fixture.root);
        for flag in [
            "ignored",
            "private",
            "external",
            "missing-directory",
            "missing-manifest",
        ] {
            let mut changed = entries.clone();
            if flag.starts_with("missing") {
                let removed = if flag == "missing-directory" {
                    "member-one"
                } else {
                    "member-one/Cargo.toml"
                };
                changed.retain(|entry| entry.path.as_unix_str() != removed);
            } else {
                let directory = changed
                    .iter_mut()
                    .find(|entry| entry.path.as_unix_str() == "member-one")
                    .expect("actual package directory");
                match flag {
                    "ignored" => directory.is_ignored = true,
                    "private" => directory.is_private = true,
                    "external" => directory.is_external = true,
                    _ => unreachable!("finite test flags"),
                }
            }
            assert!(
                native_selected_package_directory(
                    &facts,
                    &workspace,
                    &selection,
                    &fixture.root,
                    &changed,
                )
                .is_err(),
                "refuse {flag} package source"
            );
        }
        let mut wrong_host = facts.clone();
        let package = wrong_host
            .packages
            .iter_mut()
            .find(|package| package.package_id == selection.package_id)
            .expect("raw selected package");
        package.manifest_path.worktree_id = settings::WorktreeId::from_usize(usize::MAX);
        assert!(
            native_selected_package_directory(
                &wrong_host,
                &workspace,
                &selection,
                &fixture.root,
                &entries,
            )
            .is_err()
        );
        let mut wrong_selection = selection.clone();
        wrong_selection.package_id.push_str("-unissued");
        assert!(
            native_selected_package_directory(
                &facts,
                &workspace,
                &wrong_selection,
                &fixture.root,
                &entries,
            )
            .is_err()
        );
        assert_eq!(
            native_selected_package_directory(
                &facts,
                &workspace,
                &selection,
                &fixture.root,
                &entries,
            )
            .expect("unchanged owner source remains usable after refusal"),
            reviewed
        );
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_capture_service_joins_actual_producer_pair_and_refuses_changed_inputs(
        cx: &mut TestAppContext,
    ) {
        use futures::StreamExt as _;
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        cx.executor().allow_parking();
        let workload_directory = tempfile::tempdir().expect("disposable Rust workload");
        let workload_root = std::fs::canonicalize(workload_directory.path())
            .expect("canonical Rust workload directory");
        let workload_source = workload_root.join("main.rs");
        let workload_executable = workload_root.join("selected-workload");
        std::fs::write(
            &workload_source,
            "fn main() { assert_eq!(std::env::args().skip(1).collect::<Vec<_>>(), [\"selected\"]); assert_eq!(std::env::current_dir().expect(\"cwd\").file_name().and_then(|name| name.to_str()), Some(\"workspace with spaces\")); let manifest = std::fs::read_to_string(\"Cargo.toml\").expect(\"invocation-relative manifest\"); assert!(manifest.contains(\"[workspace]\") && manifest.contains(\"member-one\")); }\n",
        )
        .expect("owned Rust workload source");
        let executor = cx.background_executor.clone();
        let compilation = cx.background_spawn({
            let workload_source = workload_source.clone();
            let workload_executable = workload_executable.clone();
            async move {
                compile_native_profile_test_workload(
                    &workload_source,
                    &workload_executable,
                    &executor,
                )
                .await
            }
        });
        compilation
            .await
            .expect("actual installed compiler workload");
        let workload_metadata = std::fs::symlink_metadata(&workload_executable)
            .expect("actual compiled workload metadata");
        assert!(
            workload_metadata.is_file()
                && !workload_metadata.file_type().is_symlink()
                && (1..=256 * 1024 * 1024).contains(&workload_metadata.len())
                && workload_metadata.permissions().mode() & 0o111 != 0,
            "compiled workload is a bounded ordinary executable",
        );
        let quoted_workload = workload_executable
            .to_str()
            .expect("UTF-8 workload path")
            .replace('\'', "'\\''");
        for scenario in [
            "success",
            "success-twice",
            "capture-environment-review",
            "capture-environment-registration",
            "dirty-review",
            "wrong-pair",
            "poisoned-output",
            "capture-failure",
            "dropped-preparation",
            "dirty-registration",
            "trust-registration",
            "session-registration",
            "fresh-save-recovery",
            "clean-first",
            "clean-first-unknown-ignore",
            "clean-first-private",
            "clean-first-outside-root",
            "package-directory-registration",
            "package-directory-fresh-recovery",
            "wrong-artifact-anchor",
            "wrong-workload-directory",
        ] {
            let fixture = native_collector_public_fixture_with_output(
                cx,
                !scenario.starts_with("clean-first"),
            )
            .await;
            let mut source = fixture
                .project
                .update(cx, |project, cx| {
                    let path = project
                        .find_project_path(fixture.root.join("member-one/src/main.rs"), cx)
                        .expect("selected source");
                    project.open_buffer(path, cx)
                })
                .await
                .expect("actual source buffer");
            refresh_native_fixture_metadata(&fixture.project, &fixture.manifest, &fixture.root, cx)
                .await;
            if scenario == "clean-first-unknown-ignore" {
                std::fs::write(fixture.root.join(".gitignore"), b"")
                    .expect("unknown ignore authority");
            }
            if scenario == "clean-first-outside-root" {
                let mut metadata: serde_json::Value = serde_json::from_slice(include_bytes!(
                    "../test_data/cargo_workspace/workspace-v1.json"
                ))
                .expect("metadata fixture");
                let outside = fixture
                    .root
                    .parent()
                    .expect("fixture parent")
                    .join("outside");
                metadata["target_directory"] =
                    outside.to_str().expect("outside metadata root").into();
                let bytes = serde_json::to_string(&metadata)
                    .expect("metadata JSON")
                    .replace("/workspace", fixture.root.to_str().expect("root"))
                    .into_bytes();
                let admission = fixture
                    .store
                    .read_with(cx, |store, cx| {
                        store.prepare_reviewed_metadata_import(fixture.manifest.clone(), cx)
                    })
                    .expect("public metadata admission");
                fixture
                    .store
                    .update(cx, |store, cx| {
                        store.admit_reviewed_metadata_import(admission, bytes, cx)
                    })
                    .await
                    .expect("actual metadata output root");
            }

            let mut completed_pairs = Vec::new();
            let package_inode = std::fs::symlink_metadata(fixture.root.join("member-one"))
                .expect("original ordinary selected package")
                .ino();
            let mut rejected_package_output = None;
            let mut package_recovered = false;
            for attempt in 0..if matches!(
                scenario,
                "fresh-save-recovery" | "success-twice" | "package-directory-fresh-recovery"
            ) {
                2
            } else {
                1
            } {
                eprintln!("native capture scenario={scenario} attempt={attempt}");
                let (workspace, facts) = fixture.store.read_with(cx, |store, cx| {
                    (
                        store
                            .cached_snapshot(cx)
                            .expect("current metadata")
                            .0
                            .workspaces[0]
                            .clone(),
                        store
                            .metadata_facts_for_current_snapshot(cx)
                            .expect("private facts")[0]
                            .clone(),
                    )
                });
                let member = workspace
                    .members
                    .iter()
                    .find(|member| member.name == "member-one")
                    .expect("selected member");
                let target = member
                    .targets
                    .iter()
                    .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
                    .expect("selected Binary");
                let (selection, _, arguments) =
                    native_build_selection(&facts, &workspace, &member.id, target)
                        .expect("exact metadata selection");
                let mut artifact: serde_json::Value = serde_json::from_str(
                    include_str!("../test_data/rust_test_provider/cargo_messages.jsonl")
                        .lines()
                        .nth(1)
                        .expect("artifact"),
                )
                .expect("typed artifact");
                artifact["package_id"] = selection.package_id.into();
                artifact["target"]["name"] = selection.target_name.into();
                artifact["target"]["src_path"] =
                    selection.source_path.to_str().expect("source").into();
                artifact["target"]["kind"] = serde_json::json!(["bin"]);
                artifact["profile"]["test"] = false.into();
                artifact["executable"] = "%s".into();
                let report = format!(
                    "{}\n{{\"reason\":\"build-finished\",\"success\":true}}\n",
                    serde_json::to_string(&artifact).expect("record")
                )
                .replace('\'', "'\\''");
                let installed = fixture.collector.parent().expect("installed fixture");
                let producer = installed.join("cargo");
                std::fs::write(&producer,format!("#!/bin/sh\n/bin/mkdir -p \"${{8}}/debug\" || exit 31\n/bin/cp '{quoted_workload}' \"${{8}}/debug/member-one\" || exit 32\nprintf '{report}' \"${{8}}/debug/member-one\"\n")).expect("actual producer");
                std::fs::set_permissions(&producer, std::fs::Permissions::from_mode(0o700))
                    .expect("producer permissions");
                if scenario == "fresh-save-recovery" && attempt == 1 {
                    assert!(source.read_with(cx, |buffer, _| buffer.is_dirty()));
                    fixture
                        .project
                        .update(cx, |project, cx| project.save_buffer(source.clone(), cx))
                        .await
                        .expect("actual fresh save after denied registration");
                    assert!(!source.read_with(cx, |buffer, _| buffer.is_dirty()));
                    let trees = fixture
                        .project
                        .read_with(cx, |project, _| project.worktree_store());
                    let tree = trees
                        .read_with(cx, |trees, cx| {
                            trees.worktree_for_id(fixture.manifest.worktree_id, cx)
                        })
                        .expect("saved worktree");
                    let mut refresh = tree.read_with(cx, |tree, _| {
                        tree.as_local()
                            .expect("local save")
                            .refresh_entries_for_paths(vec![Arc::from(
                                RelPath::from_unix_str("member-one/src/main.rs")
                                    .expect("saved source"),
                            )])
                    });
                    refresh.next().await.expect("real saved source refresh");
                    refresh_native_fixture_metadata(
                        &fixture.project,
                        &fixture.manifest,
                        &fixture.root,
                        cx,
                    )
                    .await;
                }
                let launch = task::SpawnInTerminal {
                    id: task::TaskId("selected-actual-capture".into()),
                    command: Some(producer.to_str().expect("program").to_owned()),
                    literal_prefix_args: arguments.len(),
                    args: arguments,
                    cwd: Some(fixture.root.clone()),
                    ..Default::default()
                };
                if scenario == "clean-first-private" {
                    cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                            settings
                                .set_user_settings(r#"{"private_files":["target"],"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx)
                                .expect("actual private output policy")
                        })
                    });
                    assert!(
                        cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
                        "Non-flag settings changes must retain native profiling opt-in"
                    );
                }
                let preparation = fixture.project.update(cx, |project, cx| {
                    project.prepare_selected_native_cargo_build_task(
                        uuid::Uuid::new_v4(),
                        launch,
                        if scenario == "success-twice" {
                            task::MAX_TASK_STDOUT_REPORT_BYTES
                        } else {
                            1024 * 1024
                        },
                        cx,
                    )
                });
                let prepared = match preparation {
                    Ok(task) => task.await,
                    Err(error) => Err(error),
                };
                if matches!(
                    scenario,
                    "clean-first-unknown-ignore"
                        | "clean-first-private"
                        | "clean-first-outside-root"
                ) {
                    assert!(
                        prepared.is_err(),
                        "clean-first refusal must precede any Build helper"
                    );
                    assert!(!fixture.root.join("target").exists());
                    assert!(!installed.join("capture-started").exists());
                    assert!(
                        !fixture
                            .root
                            .parent()
                            .expect("fixture parent")
                            .join("outside")
                            .exists()
                    );
                    continue;
                }
                let prepared = prepared.expect("checked Build preparation");
                if scenario == "clean-first" {
                    assert!(fixture.root.join("target").is_dir());
                }
                let wrapped = prepared.launch().clone();
                assert_eq!(wrapped.cwd.as_deref(), Some(fixture.root.as_path()));
                let mut build =
                    smol::process::Command::new(wrapped.command.as_ref().expect("helper"));
                build
                    .args([
                        "--exact",
                        "cargo_workspace_store::tests::native_build_intent_helper_process_entry",
                        "--nocapture",
                    ])
                    .current_dir(wrapped.cwd.as_ref().expect("cwd"))
                    .envs(&wrapped.env)
                    .env(
                        "ZED_NATIVE_BUILD_TEST_SOCKET",
                        wrapped.args.get(2).expect("socket"),
                    )
                    .env(
                        "ZED_NATIVE_BUILD_TEST_PROGRAM",
                        wrapped.args.get(5).expect("program"),
                    )
                    .env(
                        "ZED_NATIVE_BUILD_TEST_ARGUMENTS",
                        serde_json::to_string(wrapped.args.get(6..).expect("argv")).expect("argv"),
                    )
                    .kill_on_drop(true);
                let mut build = build.spawn().expect("actual Build helper");
                let receipt = fixture
                    .project
                    .update(cx, |project, cx| {
                        project.finish_selected_native_cargo_build_task(prepared, cx)
                    })
                    .expect("real producing Task receipt")
                    .await
                    .expect("genuine selected workload");
                assert!(build.status().await.expect("Build status").success());
                std::fs::write(installed.join("profile-template.json"),include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json")).expect("fixture profile");
                std::fs::write(installed.join("profile-template.syms.json"),include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json")).expect("fixture symbols");
                std::fs::write(
                    installed.join("capture-expected-cwd"),
                    fixture.root.to_str().expect("Cargo invocation cwd"),
                )
                .expect("controlled probe and capture cwd receiver");
                std::fs::write(&fixture.collector,"#!/bin/sh\nbase=$(/usr/bin/dirname \"$0\")\nexpected=$(/bin/cat \"$base/capture-expected-cwd\") || exit 55\n[ \"$(pwd)\" = \"$expected\" ] || exit 56\ncase \"$1\" in\n--version) printf 'samply 0.13.1\\n';;\nrecord) [ -z \"${_NT_SYMBOL_PATH+x}\" ] && [ -z \"${ZED_PROFILE_TASK_TOKEN+x}\" ] || exit 50\nprintf record >> \"$base/capture-started\"\n[ ! -f \"$base/capture-failure\" ] || exit 7\nout=\"${12}\"\nshift 13\n\"$@\" >/dev/null || exit 51\n/bin/cp \"$base/profile-template.json\" \"$out\" || exit 52\n/bin/cp \"$base/profile-template.syms.json\" \"${out%.json}.syms.json\" || exit 53;;\n*) exit 54;;\nesac\n").expect("controlled collector");
                let plan = fixture
                    .project
                    .update(cx, |project, cx| {
                        project.prepare_native_capture_plan(
                            receipt,
                            task::TaskNativeProfileContinuation {
                                collector_path: fixture
                                    .collector
                                    .to_str()
                                    .expect("collector")
                                    .into(),
                                profile_path: "target/capture.json".into(),
                                workload_arguments: vec!["selected".into()],
                            },
                            cx,
                        )
                    })
                    .expect("fresh probe and plan")
                    .await
                    .expect("opaque capture plan");
                assert_eq!(
                    plan.resolved_task().resolved.cwd.as_deref(),
                    Some(fixture.root.as_path())
                );
                assert_eq!(
                    plan.resolved_task().artifact_directory(),
                    Some(fixture.root.as_path())
                );
                assert!(plan.resolved_task().stdout_report().is_none());
                assert!(plan.resolved_task().resolved_artifact().is_none());
                assert_eq!(
                    plan.resolved_task().id.0,
                    "selected-actual-capture:native-capture"
                );
                let run = uuid::Uuid::new_v4();
                let mut bound = plan
                    .resolved_task()
                    .bind_artifact_invocation(&run.to_string())
                    .expect("bound pair");
                let mut group = bound.resolved_artifact_group().expect("pair").clone();
                let output = fixture.root.join(&group.files["profile"].path);
                assert_eq!(
                    bound.resolved.args.get(11),
                    output.to_str().map(String::from).as_ref()
                );
                assert_ne!(output.parent(), bound.resolved.cwd.as_deref());
                let artifact_directory = if scenario == "wrong-artifact-anchor" {
                    Some(fixture.root.join("member-one"))
                } else {
                    bound.artifact_directory().map(Path::to_path_buf)
                };
                if scenario == "wrong-workload-directory" {
                    bound.resolved.cwd = Some(fixture.root.join("member-one"));
                }
                if scenario == "capture-environment-review" {
                    cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                            settings.set_user_settings(
                                r#"{"terminal":{"detect_venv":"off","env":{"NATIVE_CAPTURE_CHANGED_CONTEXT":"after-review"}},"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx,
                            ).expect("changed actual terminal context before admission");
                        });
                    });
                    assert!(
                        cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
                        "Non-flag settings changes must retain native profiling opt-in"
                    );
                }
                if scenario == "dirty-review" {
                    source.update(cx, |buffer, cx| {
                        buffer.edit([(0..0, "// unsaved\n")], None, cx);
                    });
                }
                if scenario == "wrong-pair" {
                    group.files.get_mut("symbols").expect("symbols").path =
                        "target/wrong.syms.json".into();
                }
                if scenario == "poisoned-output" {
                    std::fs::write(&output, b"unrelated output").expect("poisoned absent path");
                }
                let preparation = fixture.project.update(cx, |project, cx| {
                    project.admit_native_capture_task(
                        plan,
                        run,
                        bound.resolved.clone(),
                        group,
                        artifact_directory,
                        cx,
                    )
                });
                let prepared = match preparation {
                    Ok(task) => task.await,
                    Err(error) => Err(error),
                };
                if matches!(
                    scenario,
                    "dirty-review"
                        | "wrong-pair"
                        | "poisoned-output"
                        | "capture-environment-review"
                        | "wrong-artifact-anchor"
                        | "wrong-workload-directory"
                ) {
                    assert!(prepared.is_err());
                    assert!(!installed.join("capture-started").exists());
                    continue;
                }
                let prepared = prepared.expect("approved exact capture");
                if scenario == "dropped-preparation" {
                    drop(prepared);
                    assert!(!installed.join("capture-started").exists());
                    continue;
                }
                if scenario == "capture-failure" {
                    std::fs::write(installed.join("capture-failure"), b"fail")
                        .expect("actual child failure");
                }
                let reject_registration = matches!(
                    scenario,
                    "dirty-registration"
                        | "trust-registration"
                        | "session-registration"
                        | "capture-environment-registration"
                        | "package-directory-registration"
                ) || (matches!(
                    scenario,
                    "fresh-save-recovery" | "package-directory-fresh-recovery"
                ) && attempt == 0);
                if scenario == "dirty-registration"
                    || (scenario == "fresh-save-recovery" && attempt == 0)
                {
                    source.update(cx, |buffer, cx| {
                        buffer.edit([(0..0, "// changed after preparation\n")], None, cx);
                    });
                }
                if scenario == "package-directory-registration"
                    || (scenario == "package-directory-fresh-recovery" && attempt == 0)
                {
                    let member = fixture.root.join("member-one");
                    let replacement = fixture.root.join("moved-member-one");
                    std::fs::rename(&member, &replacement).expect("owned package replacement");
                    std::os::unix::fs::symlink(&replacement, &member)
                        .expect("raced package alias before actual helper registration");
                }
                if scenario == "capture-environment-registration" {
                    cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                            settings.set_user_settings(
                                r#"{"terminal":{"detect_venv":"off","env":{"NATIVE_CAPTURE_CHANGED_CONTEXT":"before-ack"}},"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx,
                            ).expect("changed actual terminal context before registration");
                        });
                    });
                    assert!(
                        cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
                        "Non-flag settings changes must retain native profiling opt-in"
                    );
                }
                if scenario == "trust-registration" {
                    let trees = fixture
                        .project
                        .read_with(cx, |project, _| project.worktree_store());
                    let trust =
                        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
                    trust.update(cx, |trust, cx| {
                        trust.restrict(
                            trees.downgrade(),
                            HashSet::from_iter([crate::trusted_worktrees::PathTrust::Worktree(
                                fixture.manifest.worktree_id,
                            )]),
                            cx,
                        )
                    });
                }
                if scenario == "session-registration" {
                    let client = fixture.project.read_with(cx, |project, _| project.client());
                    fixture
                        .store
                        .update(cx, |store, cx| store.shared(42, client.into(), cx));
                }
                let wrapped = prepared.launch().clone();
                assert_eq!(wrapped.cwd.as_deref(), Some(fixture.root.as_path()));
                assert_eq!(
                    wrapped.args.get(3).map(String::as_str),
                    Some("--checked-native-profile-capture")
                );
                assert!(
                    wrapped
                        .env
                        .contains_key(util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV)
                );
                let mut capture =
                    smol::process::Command::new(wrapped.command.as_ref().expect("helper"));
                capture
                    .args([
                        "--exact",
                        "cargo_workspace_store::tests::native_build_intent_helper_process_entry",
                        "--nocapture",
                    ])
                    .current_dir(wrapped.cwd.as_ref().expect("cwd"))
                    .envs(&wrapped.env)
                    .env(
                        "ZED_NATIVE_BUILD_TEST_SOCKET",
                        wrapped.args.get(2).expect("socket"),
                    )
                    .env(
                        "ZED_NATIVE_BUILD_TEST_PROGRAM",
                        wrapped.args.get(5).expect("program"),
                    )
                    .env(
                        "ZED_NATIVE_BUILD_TEST_ARGUMENTS",
                        serde_json::to_string(wrapped.args.get(6..).expect("argv")).expect("argv"),
                    )
                    .env("ZED_NATIVE_BUILD_TEST_NATIVE", "1")
                    .env("ZED_NATIVE_BUILD_TEST_CHECKED_NATIVE", "1")
                    .env(
                        "_NT_SYMBOL_PATH",
                        "srv*/private/tmp/cache*http://127.0.0.1:1",
                    )
                    .kill_on_drop(true);
                let mut capture = capture.spawn().expect("actual native helper");
                let completion = fixture.project.update(cx, |project, cx| {
                    project.finish_native_capture_task(prepared, cx)
                });
                let finished = match completion {
                    Ok(task) => task.await,
                    Err(error) => Err(error),
                };
                let status = capture.status().await.expect("capture status");
                if reject_registration {
                    assert!(
                        !status.success(),
                        "stale registration must be denied before collector exec"
                    );
                    assert!(finished.is_err());
                    assert!(
                        !installed.join("capture-started").exists(),
                        "collector executed after invalid preparation"
                    );
                    assert!(!output.exists());
                    if scenario == "package-directory-fresh-recovery" {
                        assert_eq!(attempt, 0, "only the first genuine registration is refused");
                        rejected_package_output = Some(output.clone());
                        let member = fixture.root.join("member-one");
                        let replacement = fixture.root.join("moved-member-one");
                        assert!(
                            std::fs::symlink_metadata(&member)
                                .expect("owned rejected alias")
                                .file_type()
                                .is_symlink()
                        );
                        std::fs::remove_file(&member).expect("remove only owned package alias");
                        std::fs::rename(&replacement, &member)
                            .expect("restore the actual original package directory");
                        let restored =
                            std::fs::symlink_metadata(&member).expect("restored package identity");
                        assert!(restored.is_dir() && !restored.file_type().is_symlink());
                        assert_eq!(restored.ino(), package_inode);
                        let trees = fixture
                            .project
                            .read_with(cx, |project, _| project.worktree_store());
                        let tree = trees
                            .read_with(cx, |trees, cx| {
                                trees.worktree_for_id(fixture.manifest.worktree_id, cx)
                            })
                            .expect("actual local source owner");
                        let mut refresh = tree.read_with(cx, |tree, _| {
                            tree.as_local()
                                .expect("local package restoration")
                                .refresh_entries_for_paths(vec![Arc::from(RelPath::empty())])
                        });
                        refresh
                            .next()
                            .await
                            .expect("actual restored source refresh");
                        source = fixture
                            .project
                            .update(cx, |project, cx| {
                                let path = project
                                    .find_project_path(member.join("src/main.rs"), cx)
                                    .expect("refreshed selected source path");
                                project.open_buffer(path, cx)
                            })
                            .await
                            .expect("actual restored source buffer");
                        source.update(cx, |buffer, cx| {
                            buffer.edit(
                                [(0..0, "// freshly saved after package restoration\n")],
                                None,
                                cx,
                            );
                        });
                        assert!(source.read_with(cx, |buffer, _| buffer.is_dirty()));
                        fixture
                            .project
                            .update(cx, |project, cx| project.save_buffer(source.clone(), cx))
                            .await
                            .expect("actual fresh restored-package source save");
                        assert!(!source.read_with(cx, |buffer, _| buffer.is_dirty()));
                        let mut refresh = tree.read_with(cx, |tree, _| {
                            tree.as_local()
                                .expect("local restored-package save")
                                .refresh_entries_for_paths(vec![Arc::from(RelPath::empty())])
                        });
                        refresh
                            .next()
                            .await
                            .expect("actual freshly saved source refresh");
                        refresh_native_fixture_metadata(
                            &fixture.project,
                            &fixture.manifest,
                            &fixture.root,
                            cx,
                        )
                        .await;
                        assert!(!installed.join("capture-started").exists());
                        assert!(!output.exists());
                    }
                    continue;
                }
                if scenario == "capture-failure" {
                    assert!(!status.success());
                    assert!(finished.is_err());
                    assert!(!output.exists());
                } else {
                    assert!(status.success());
                    finished.expect("current kernel success and sealed pair");
                    if scenario == "package-directory-fresh-recovery" {
                        assert_eq!(attempt, 1);
                        let rejected = rejected_package_output
                            .as_ref()
                            .expect("real first registration refusal before recovery");
                        assert_ne!(rejected, &output, "fresh Capture UUID renews pair paths");
                        assert!(!rejected.exists(), "refused pair remains unpublished");
                        assert!(installed.join("capture-started").is_file());
                        assert_eq!(
                            std::fs::symlink_metadata(fixture.root.join("member-one"))
                                .expect("current restored package")
                                .ino(),
                            package_inode
                        );
                        assert!(!source.read_with(cx, |buffer, _| buffer.is_dirty()));
                        package_recovered = true;
                    }
                    assert_eq!(
                        std::fs::read(&output).expect("profile"),
                        include_bytes!(
                            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
                        )
                    );
                    let pair_store = fixture.project.read_with(cx, |project, _| {
                        project
                            .profile_results_store_for_test()
                            .expect("existing pair owner")
                            .clone()
                    });
                    let (reserved, retaining_context) = pair_store
                        .read_with(cx, |store, _| store.checked_native_resources_for_test());
                    assert_eq!(
                        retaining_context, 0,
                        "successful pair retains seals, not checked map/admission"
                    );
                    assert!(
                        reserved <= 8 * 1024 * 1024,
                        "existing aggregate quota remains bounded"
                    );
                    if matches!(
                        scenario,
                        "success-twice" | "package-directory-fresh-recovery"
                    ) {
                        let symbols = output.with_extension("syms.json");
                        assert!(symbols.is_file());
                        let paths = fixture.project.read_with(cx, |project, cx| {
                            [
                                project
                                    .find_project_path(&output, cx)
                                    .expect("current profile path"),
                                project
                                    .find_project_path(&symbols, cx)
                                    .expect("current symbol path"),
                            ]
                        });
                        assert!(
                            !completed_pairs.contains(&paths),
                            "fresh Capture UUID must select a distinct pair"
                        );
                        completed_pairs.push(paths);
                        for paths in &completed_pairs {
                            let import = fixture
                                .project
                                .update(cx, |project, cx| {
                                    project.prepare_diagnostic_profile_import(paths.clone(), cx)
                                })
                                .expect("fresh actual local import admission");
                            let reopened = fixture
                                .project
                                .update(cx, |project, cx| {
                                    project.import_diagnostic_profile(
                                        import,
                                        Arc::new(std::sync::atomic::AtomicBool::new(false)),
                                        cx,
                                    )
                                })
                                .expect("public local loader request")
                                .await
                                .expect("actual pair loader");
                            assert!(reopened.total_samples() > 0);
                            fixture
                                .project
                                .read_with(cx, |project, cx| {
                                    project.revalidate_diagnostic_profile_result(&reopened, cx)
                                })
                                .expect("current independently reopened sealed input result");
                        }
                    }
                }
            }
            if scenario == "success-twice" {
                assert_eq!(completed_pairs.len(), 2);
            }
            if scenario == "package-directory-fresh-recovery" {
                assert_eq!(
                    completed_pairs.len(),
                    1,
                    "only the renewed pair is loadable"
                );
                assert!(rejected_package_output.is_some());
                assert!(
                    package_recovered,
                    "real fresh Build and Capture must complete after refusal"
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_build_intent_actual_child_join_rejects_other_run_failure_and_changed_source(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        for scenario in [
            "valid",
            "other-run",
            "failure",
            "changed-source",
            "dirty-source",
            "changed-buffer-version",
            "report-overflow",
            "named-flag-off",
            "metadata-refresh",
        ] {
            let fixture = native_collector_public_fixture(cx).await;
            if scenario == "named-flag-off" {
                cx.update(|cx| {
                    crate::task_store::set_experimental_feature_for_test(
                        "rust-named-cargo-presets-experimental",
                        true,
                        cx,
                    )
                });
            }
            let source_buffer = fixture
                .project
                .update(cx, |project, cx| {
                    let path = project
                        .find_project_path(fixture.root.join("member-one/src/main.rs"), cx)
                        .expect("selected path");
                    project.open_buffer(path, cx)
                })
                .await
                .expect("actual opened selected source");
            refresh_native_fixture_metadata(&fixture.project, &fixture.manifest, &fixture.root, cx)
                .await;
            let (workspace, facts) = fixture.store.read_with(cx, |store, cx| {
                (
                    store
                        .cached_snapshot(cx)
                        .expect("current workspace")
                        .0
                        .workspaces[0]
                        .clone(),
                    store
                        .metadata_facts_for_current_snapshot(cx)
                        .expect("private facts")[0]
                        .clone(),
                )
            });
            let member = workspace
                .members
                .iter()
                .find(|member| member.name == "member-one")
                .expect("selected member");
            let target = member
                .targets
                .iter()
                .find(|target| target.kind == crate::cargo_workspace::CargoTargetKind::Binary)
                .expect("selected Binary");
            let (selection, _, logical_arguments) =
                native_build_selection(&facts, &workspace, &member.id, target)
                    .expect("current selected recipe");
            let mut artifact: serde_json::Value = serde_json::from_str(
                include_str!("../test_data/rust_test_provider/cargo_messages.jsonl")
                    .lines()
                    .nth(1)
                    .expect("artifact fixture"),
            )
            .expect("typed Cargo artifact");
            artifact["package_id"] = selection.package_id.clone().into();
            artifact["target"]["name"] = selection.target_name.clone().into();
            artifact["target"]["src_path"] =
                selection.source_path.to_str().expect("source path").into();
            artifact["target"]["kind"] = serde_json::json!(["bin"]);
            artifact["profile"]["test"] = false.into();
            artifact["executable"] = "__run_executable__".into();
            let report = format!(
                "{}\n{{\"reason\":\"build-finished\",\"success\":true}}\n",
                serde_json::to_string(&artifact).expect("record")
            );
            let report = report.replace("__run_executable__", "%s");
            let report = report.replace('\'', "'\\''");
            let producer = fixture
                .collector
                .parent()
                .expect("installed fixture")
                .join("cargo");
            let output = if scenario == "other-run" {
                "output=\"$(dirname \"${8}\")/zed-native-profile-build-other/debug\""
            } else {
                "output=\"${8}/debug\""
            };
            let script = format!(
                "#!/bin/sh\n[ \"$1\" = build ] && [ \"$2\" = --message-format=json ] && [ \"$7\" = --target-dir ] || exit 31\n{output}\n/bin/mkdir -p \"$output\" || exit 32\n/bin/cp /bin/echo \"$output/member-one\" || exit 33\nprintf '{report}' \"$output/member-one\"\nexit {}\n",
                if scenario == "failure" { 7 } else { 0 }
            );
            let script = if scenario == "report-overflow" {
                "#!/bin/sh\n/bin/dd if=/dev/zero bs=1048576 count=2 2>/dev/null\nexit 0\n"
                    .to_owned()
            } else {
                script
            };
            std::fs::write(&producer, script).expect("controlled producing command");
            std::fs::set_permissions(&producer, std::fs::Permissions::from_mode(0o700))
                .expect("producer executable");
            let helper = std::env::current_exe().expect("real helper image");
            let report_store = fixture.project.read_with(cx, |project, _| {
                project
                    .profile_results_store_for_test()
                    .expect("existing report owner")
                    .clone()
            });
            report_store.update(cx, |store, _| store.set_local_task_helper(helper));
            let run_id = uuid::Uuid::new_v4();
            let launch = task::SpawnInTerminal {
                required_experimental_features: if scenario == "named-flag-off" {
                    vec!["rust-named-cargo-presets-experimental".into()]
                } else {
                    Vec::new()
                },
                id: task::TaskId("selected-build".into()),
                command: Some(producer.to_str().expect("program").to_owned()),
                args: logical_arguments.clone(),
                literal_prefix_args: logical_arguments.len(),
                cwd: Some(fixture.root.clone()),
                ..Default::default()
            };
            if scenario == "valid" {
                let gateway = fixture
                    .project
                    .update(cx, |project, cx| {
                        project.prepare_selected_native_cargo_build_task(
                            uuid::Uuid::new_v4(),
                            launch.clone(),
                            1024 * 1024,
                            cx,
                        )
                    })
                    .expect("production Project eligibility gateway")
                    .await
                    .expect("opaque ready wrapper");
                assert_eq!(gateway.launch().id, launch.id);
                assert_eq!(
                    gateway.launch().literal_prefix_args,
                    gateway.launch().args.len()
                );
                assert!(
                    gateway
                        .launch()
                        .args
                        .iter()
                        .any(|argument| argument == "--capture-build-stdout")
                );
                assert_eq!(gateway.launch().cwd, launch.cwd);
                assert!(gateway.launch().args.iter().any(|argument| {
                    argument.starts_with(
                        fixture
                            .root
                            .join("target/zed-native-profile-build-")
                            .to_str()
                            .expect("run prefix"),
                    )
                }));
                drop(gateway);
            }
            let intent = fixture
                .store
                .update(cx, |store, cx| {
                    store.prepare_native_cargo_build_intent_for_launch(
                        fixture.project.clone(),
                        run_id,
                        &launch,
                        cx,
                    )
                })
                .expect("public selected admission")
                .await
                .expect("selected source/program seal");
            let prepared = fixture
                .store
                .update(cx, |store, cx| {
                    store.prepare_native_cargo_build_binding(intent, &launch, 1024 * 1024, cx)
                })
                .expect("actual eligible recipe")
                .await
                .expect("checked run binding");
            let (intent, binding) = prepared;
            let token: String = binding
                .token()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let invocation = intent
                .expected_invocation()
                .expect("actual admitted child environment");
            let mut child = smol::process::Command::new(binding.helper_path())
                .args([
                    "--exact",
                    "cargo_workspace_store::tests::native_build_intent_helper_process_entry",
                    "--nocapture",
                ])
                .current_dir(intent.working_directory())
                .envs(invocation.environment())
                .env("ZED_NATIVE_BUILD_TEST_SOCKET", binding.socket_path())
                .env("ZED_NATIVE_BUILD_TEST_PROGRAM", intent.program())
                .env(
                    "ZED_NATIVE_BUILD_TEST_ARGUMENTS",
                    serde_json::to_string(intent.arguments()).expect("exact bound argv"),
                )
                .env(
                    util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                    token,
                )
                .kill_on_drop(true)
                .spawn()
                .expect("actual prepared helper");
            let report = fixture
                .project
                .update(cx, |project, cx| {
                    project.await_local_cargo_build_report(binding, cx)
                })
                .expect("private report observation")
                .await;
            let status = child.status().await.expect("actual child status");
            if scenario == "report-overflow" {
                assert!(
                    report.is_err(),
                    "requested1MiB must reject actual2MiB report"
                );
                continue;
            }
            if scenario == "failure" {
                assert!(!status.success());
                assert!(report.is_err());
                continue;
            }
            assert!(status.success());
            let report = report.expect("kernel-observed successful report");
            if scenario == "changed-source" {
                std::fs::write(
                    fixture.root.join("member-one/src/main.rs"),
                    "fn main() { println!(\"changed\"); }\n",
                )
                .expect("change selected source after actual Build");
            }
            if matches!(scenario, "dirty-source" | "changed-buffer-version") {
                source_buffer.update(cx, |buffer, cx| {
                    buffer.edit([(0..0, "// unsaved\n")], None, cx);
                    if scenario == "changed-buffer-version" {
                        assert!(buffer.undo(cx).is_some());
                        assert!(
                            !buffer.is_dirty(),
                            "undo preserves clean disk bytes but changes version"
                        );
                    } else {
                        assert!(buffer.is_dirty());
                    }
                });
            }
            if scenario == "named-flag-off" {
                cx.update(|cx| {
                    crate::task_store::set_experimental_feature_for_test(
                        "rust-named-cargo-presets-experimental",
                        false,
                        cx,
                    )
                });
            }
            if scenario == "metadata-refresh" {
                let admitted_generation = intent.reviewed.revision;
                let refreshing = fixture.store.update(cx, |store, cx| store.refresh(cx));
                fixture.store.read_with(cx, |store, _| {
                    assert_ne!(store.revision, admitted_generation, "An explicit real metadata refresh must retire the original Cargo generation");
                });
                refreshing
                    .await
                    .expect("original metadata owner refresh completes");
            }
            let finished = fixture.store.update(cx, |store, cx| {
                store.finish_native_cargo_build(intent, report, cx)
            });
            let result = match finished {
                Ok(task) => task.await,
                Err(error) => Err(error),
            };
            if scenario == "named-flag-off" {
                match &result {
                    Err(error) => assert!(
                        error
                            .to_string()
                            .contains("original selected Cargo Task experimental opt-in")
                    ),
                    Ok(_) => panic!(
                        "Named revocation after actual kernel-successful Build must refuse workload"
                    ),
                }
            }
            if scenario == "metadata-refresh" {
                match &result {
                    Err(error) => assert!(
                        error
                            .to_string()
                            .contains("Reviewed Cargo owner generation changed")
                    ),
                    Ok(_) => panic!(
                        "A metadata refresh after actual child success cannot preserve admitted workload authority"
                    ),
                }
            }
            if scenario == "valid" {
                let workload = result.expect("joined observed workload");
                assert!(
                    workload
                        .executable_path()
                        .starts_with(workload.intent.invocation_output_root())
                );
                assert_eq!(workload.selection(), &selection);
                let executable = workload.executable_path();
                std::fs::write(&executable, b"replaced executable")
                    .expect("replace selected executable");
                let revalidation = fixture.store.update(cx, |store, cx| {
                    store.revalidate_native_cargo_workload(workload, cx)
                });
                let result = match revalidation {
                    Ok(task) => task.await,
                    Err(error) => Err(error),
                };
                assert!(
                    result.is_err(),
                    "replacement cannot retain workload authority"
                );
            } else {
                assert!(
                    result.is_err(),
                    "wrong run or changed selected source cannot yield workload authority"
                );
            }
            if scenario == "named-flag-off" {
                cx.update(|cx| {
                    crate::task_store::set_experimental_feature_for_test(
                        "rust-named-cargo-presets-experimental",
                        true,
                        cx,
                    )
                });
                cx.run_until_parked();
                let reviewed = fixture
                    .store
                    .read_with(cx, |store, cx| {
                        store.prepare_reviewed_metadata_import(fixture.manifest.clone(), cx)
                    })
                    .expect("fresh actual metadata source admission");
                let bytes = include_str!("../test_data/cargo_workspace/workspace-v1.json")
                    .replace("/workspace", fixture.root.to_str().expect("fixture root"))
                    .into_bytes();
                fixture
                    .store
                    .update(cx, |store, cx| {
                        store.admit_reviewed_metadata_import(reviewed, bytes, cx)
                    })
                    .await
                    .expect("renew metadata through existing public owner");
                let fresh_run = uuid::Uuid::new_v4();
                let fresh = fixture
                    .store
                    .update(cx, |store, cx| {
                        store.prepare_native_cargo_build_intent_for_launch(
                            fixture.project.clone(),
                            fresh_run,
                            &launch,
                            cx,
                        )
                    })
                    .expect("fresh same-owner selected admission")
                    .await
                    .expect("restored local Named opt-in admits fresh sealed intent");
                assert_eq!(fresh.run_id(), fresh_run);
                assert_ne!(fresh.run_id(), run_id);
                assert_eq!(
                    fresh.required_experimental_features,
                    launch.required_experimental_features
                );
                assert!(
                    fixture
                        .store
                        .update(cx, |store, cx| store
                            .validate_native_build_intent(&fresh, cx))
                        .is_ok()
                );
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    fn native_collector_public_probe(
        fixture: &NativeCollectorPublicFixture,
        cx: &mut TestAppContext,
    ) -> Result<CargoNativeCollectorProbe> {
        let mut task = fixture.store.update(cx, |store, cx| {
            store.probe_native_collector(
                fixture.manifest.clone(),
                fixture.collector.clone(),
                fixture.root.clone(),
                fixture.environment.clone(),
                cx,
            )
        })?;
        poll_cargo_process(cx, || (&mut task).now_or_never())
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    fn native_collector_public_revalidate(
        fixture: &NativeCollectorPublicFixture,
        probe: CargoNativeCollectorProbe,
        cx: &mut TestAppContext,
    ) -> Result<CargoNativeCollectorProbe> {
        let mut task = fixture.store.update(cx, |store, cx| {
            store.revalidate_native_collector_probe(
                probe,
                fixture.root.clone(),
                fixture.environment.clone(),
                cx,
            )
        })?;
        poll_cargo_process(cx, || (&mut task).now_or_never())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_collector_complete_environment_suppression_preserves_data_and_bounds() {
        let original = HashMap::from_iter([
            ("PATH".into(), "/owned/install/bin".into()),
            ("CONTROLLED_SETTING".into(), "retained".into()),
            (
                "_NT_SYMBOL_PATH".into(),
                "srv*http://unadmitted.invalid".into(),
            ),
            (
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV.into(),
                "private-token".into(),
            ),
            (
                util::macos_task_completion::TASK_BUILD_INVOCATION_ENV.into(),
                "private-build-envelope".into(),
            ),
            (
                util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV.into(),
                "private-native-envelope".into(),
            ),
        ]);
        let sanitized = native_capture_environment(&original).expect("bounded complete map");
        assert_eq!(
            sanitized,
            HashMap::from_iter([
                ("PATH".into(), "/owned/install/bin".into()),
                ("CONTROLLED_SETTING".into(), "retained".into()),
            ])
        );
        assert!(
            original.contains_key("_NT_SYMBOL_PATH"),
            "logical/recipe map is not mutated"
        );
        for bad in [
            HashMap::from_iter([("BAD=KEY".into(), "value".into())]),
            HashMap::from_iter([("BAD".into(), "value\0".into())]),
            (0..17)
                .map(|index| (format!("SETTING_{index}"), "x".repeat(4096)))
                .collect(),
        ] {
            assert!(native_capture_environment(&bad).is_err());
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_collector_complete_environment_recipe_drives_actual_probe_and_renewal(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let fixture = native_collector_public_fixture(cx).await;
        cx.update(|cx| cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
            settings.set_user_settings(
                r#"{"terminal":{"detect_venv":"off","env":{"NATIVE_SETTING_ONLY":"settings","NATIVE_PRECEDENCE":"settings","_NT_SYMBOL_PATH":"srv*http://unadmitted.invalid"}},"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx,
            ).expect("actual terminal-only settings");
        }));
        assert!(
            cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
            "Non-flag settings changes must retain native profiling opt-in"
        );
        let reviewed = fixture
            .store
            .read_with(cx, |store, cx| {
                store.prepare_reviewed_metadata_import(fixture.manifest.clone(), cx)
            })
            .expect("fresh metadata review after terminal settings");
        let bytes = std::str::from_utf8(include_bytes!(
            "../test_data/cargo_workspace/workspace-v1.json"
        ))
        .expect("captured metadata UTF-8")
        .replace(
            "/workspace",
            fixture.root.to_str().expect("UTF-8 fixture root"),
        )
        .into_bytes();
        fixture
            .store
            .update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(reviewed, bytes, cx)
            })
            .await
            .expect("fresh genuine metadata after terminal settings");
        std::fs::write(&fixture.collector, "#!/bin/sh\n[ \"$1\" = --version ] || exit 31\n[ \"$NATIVE_SETTING_ONLY\" = settings ] && [ \"$NATIVE_PRECEDENCE\" = logical ] && [ \"$TERM_PROGRAM\" = zed ] || exit 32\n[ -z \"${_NT_SYMBOL_PATH+x}\" ] && [ -z \"${ZED_PROFILE_TASK_TOKEN+x}\" ] && [ -z \"${ZED_PROFILE_BUILD_INVOCATION_ENV+x}\" ] && [ -z \"${ZED_PROFILE_NATIVE_INVOCATION_ENV+x}\" ] || exit 33\nprintf 'samply 0.13.1\\n'\n").expect("owned actual collector probe receiver");
        let launch = task::SpawnInTerminal {
            cwd: Some(fixture.root.clone()),
            env: HashMap::from_iter([("NATIVE_PRECEDENCE".into(), "logical".into())]),
            ..Default::default()
        };
        let recipe = fixture
            .project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("genuine terminal environment request")
            .await
            .expect("genuine terminal environment receipt");
        let probe = fixture
            .store
            .update(cx, |store, cx| {
                store.probe_native_collector_with_environment(
                    fixture.manifest.clone(),
                    fixture.collector.clone(),
                    fixture.root.clone(),
                    launch.env.clone(),
                    NativeCollectorEnvironment::TerminalRecipe {
                        project: fixture.project.downgrade(),
                        recipe,
                    },
                    cx,
                )
            })
            .expect("joined actual probe request")
            .await
            .expect("actual collector received complete sanitized map");
        assert_eq!(probe.task_environment, launch.env);
        assert_eq!(
            probe
                .environment
                .get("NATIVE_SETTING_ONLY")
                .map(String::as_str),
            Some("settings")
        );
        assert_eq!(
            probe
                .environment
                .get("NATIVE_PRECEDENCE")
                .map(String::as_str),
            Some("logical")
        );
        assert!(!probe.task_environment.contains_key("NATIVE_SETTING_ONLY"));
        assert!(!probe.environment.contains_key("_NT_SYMBOL_PATH"));
        let NativeCollectorEnvironment::TerminalRecipe { recipe, .. } = &probe.environment_source
        else {
            panic!("complete probe must retain its genuine terminal recipe");
        };
        fixture
            .project
            .update(cx, |project, cx| {
                project.revalidate_native_task_environment(recipe, cx)
            })
            .expect("actual unchanged terminal receipt renewal request")
            .await
            .expect("actual unchanged terminal receipt renewal");
        let probe = fixture
            .store
            .update(cx, |store, cx| {
                store.revalidate_native_collector_probe(
                    probe,
                    fixture.root.clone(),
                    launch.env.clone(),
                    cx,
                )
            })
            .expect("genuine unchanged renewal request")
            .await
            .expect("genuine unchanged renewed receipt");
        cx.update(|cx| cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
            settings.set_user_settings(
                r#"{"terminal":{"detect_venv":"off","env":{"NATIVE_SETTING_ONLY":"changed","NATIVE_PRECEDENCE":"settings","_NT_SYMBOL_PATH":"srv*http://unadmitted.invalid"}},"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx,
            ).expect("changed terminal-owned complete map");
        }));
        assert!(
            cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
            "Non-flag settings changes must retain native profiling opt-in"
        );
        let NativeCollectorEnvironment::TerminalRecipe { recipe, .. } = &probe.environment_source
        else {
            panic!("complete probe must retain its genuine terminal recipe");
        };
        let changed_recipe = fixture
            .project
            .update(cx, |project, cx| {
                project.revalidate_native_task_environment(recipe, cx)
            })
            .expect("actual changed terminal receipt renewal request")
            .await;
        assert!(
            changed_recipe.is_err(),
            "genuine terminal owner refuses the changed complete map"
        );
        let denied = fixture.store.update(cx, |store, cx| {
            store.revalidate_native_collector_probe(
                probe,
                fixture.root.clone(),
                launch.env.clone(),
                cx,
            )
        });
        let denied = match denied {
            Ok(task) => task.await,
            Err(error) => Err(error),
        };
        assert!(
            denied.is_err(),
            "same logical map cannot hide changed Terminal settings"
        );
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_collector_probe_public_api_admits_exact_collector_and_revalidates(
        cx: &mut TestAppContext,
    ) {
        let fixture = native_collector_public_fixture(cx).await;
        let probe = native_collector_public_probe(&fixture, cx).expect("actual public probe");
        assert_eq!(probe.collector_path(), fixture.collector);
        assert_eq!(probe.version(), "0.13.1");
        let fingerprint = probe.probe_environment_fingerprint();
        assert_eq!(
            std::fs::read_to_string(&fixture.log).expect("actual child receipt"),
            "--version:admitted\n"
        );
        let revalidated = native_collector_public_revalidate(&fixture, probe, cx)
            .expect("actual public revalidation");
        assert_eq!(revalidated.collector_path(), fixture.collector);
        assert_eq!(revalidated.probe_environment_fingerprint(), fingerprint);
        assert_eq!(
            std::fs::read_to_string(&fixture.log).expect("unchanged probe count"),
            "--version:admitted\n"
        );
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_collector_probe_public_api_rejects_source_root_privacy_and_route_changes(
        cx: &mut TestAppContext,
    ) {
        for change in [
            "source-mutation",
            "source-replacement",
            "root-replacement",
            "private-policy",
            "restricted-trust",
            "shared-route",
        ] {
            let fixture = native_collector_public_fixture(cx).await;
            let probe = native_collector_public_probe(&fixture, cx).expect("admitted public probe");
            match change {
                "source-mutation" => {
                    std::fs::write(
                        fixture.root.join("member-one/src/lib.rs"),
                        "pub fn changed() -> u32 { 17 }\n",
                    )
                    .expect("source mutation");
                }
                "source-replacement" => {
                    let replacement = fixture.root.join("member-one/src/replacement.rs");
                    std::fs::write(&replacement, "pub fn retained() -> u32 { 7 }\n")
                        .expect("same-byte new inode");
                    std::fs::rename(replacement, fixture.root.join("member-one/src/lib.rs"))
                        .expect("atomic source replacement");
                }
                "root-replacement" => {
                    let retained = fixture.root.with_file_name("retained-root");
                    std::fs::rename(&fixture.root, retained).expect("old root retained");
                    std::fs::create_dir(&fixture.root).expect("replacement root");
                }
                "private-policy" => {
                    cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                            settings
                                .set_user_settings(r#"{"private_files":["Cargo.toml"],"feature_flags":{"rust-native-profiling-experimental":"on"}}"#, cx)
                                .expect("private input policy");
                        })
                    });
                    assert!(
                        cx.read(|cx| cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()),
                        "Non-flag settings changes must retain native profiling opt-in"
                    );
                }
                "restricted-trust" => {
                    let worktree_store = fixture
                        .project
                        .read_with(cx, |project, _| project.worktree_store());
                    let trust =
                        cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("trust owner"));
                    trust.update(cx, |trust, cx| {
                        trust.restrict(
                            worktree_store.downgrade(),
                            HashSet::from_iter([crate::trusted_worktrees::PathTrust::Worktree(
                                fixture.manifest.worktree_id,
                            )]),
                            cx,
                        );
                    });
                }
                "shared-route" => {
                    let client = fixture.project.read_with(cx, |project, _| project.client());
                    fixture
                        .store
                        .update(cx, |store, cx| store.shared(42, client.into(), cx));
                }
                _ => unreachable!("closed mutation fixture"),
            }
            assert!(
                native_collector_public_revalidate(&fixture, probe, cx).is_err(),
                "public revalidation accepted {change}"
            );
            assert_eq!(
                std::fs::read_to_string(&fixture.log).expect("no launch after invalidation"),
                "--version:admitted\n"
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_collector_probe_public_api_rejects_collector_replacement_and_bounded_failures(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        for change in [
            "collector-replacement",
            "oversized-collector",
            "version-overflow",
        ] {
            let fixture = native_collector_public_fixture(cx).await;
            if change == "collector-replacement" {
                let probe =
                    native_collector_public_probe(&fixture, cx).expect("admitted public probe");
                let replacement = fixture.collector.with_file_name("replacement");
                std::fs::copy(&fixture.collector, &replacement).expect("same-byte replacement");
                std::fs::rename(replacement, &fixture.collector).expect("new collector inode");
                assert!(native_collector_public_revalidate(&fixture, probe, cx).is_err());
                assert_eq!(
                    std::fs::read_to_string(&fixture.log).expect("single original probe"),
                    "--version:admitted\n"
                );
            } else {
                if change == "oversized-collector" {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&fixture.collector)
                        .expect("sparse oversized fixture")
                        .set_len(MAX_NATIVE_COLLECTOR_EXECUTABLE_BYTES + 1)
                        .expect("collector byte-limit overflow");
                } else {
                    std::fs::write(
                        &fixture.collector,
                        "#!/bin/sh\n/usr/bin/head -c 4097 /dev/zero\n",
                    )
                    .expect("bounded output fixture");
                    std::fs::set_permissions(
                        &fixture.collector,
                        std::fs::Permissions::from_mode(0o700),
                    )
                    .expect("executable bounded fixture");
                }
                let error = native_collector_public_probe(&fixture, cx)
                    .err()
                    .expect("public probe denied");
                assert!(
                    error
                        .to_string()
                        .contains(if change == "oversized-collector" {
                            "bounded regular file"
                        } else {
                            "stdout exceeded"
                        })
                );
                assert!(
                    !fixture.log.exists(),
                    "rejected public probe minted no successful child receipt"
                );
            }
        }
    }
}
