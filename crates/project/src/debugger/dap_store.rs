use super::{
    breakpoint_store::BreakpointStore,
    dap_command::EvaluateCommand,
    locators,
    session::{self, Session, SessionStateEvent},
};
use remote::Interactive;

use crate::{
    InlayHint, InlayHintLabel, ProjectEnvironment, ResolveState,
    debugger::session::SessionQuirks,
    project_settings::{DapBinary, ProjectSettings},
    worktree_store::WorktreeStore,
};
use anyhow::{Context as _, Result, anyhow};
use async_trait::async_trait;
use collections::HashMap;
use dap::{
    Capabilities, DapRegistry, DebugRequest, EvaluateArgumentsContext, StackFrameId,
    adapters::{
        DapDelegate, DebugAdapter, DebugAdapterBinary, DebugAdapterName, DebugTaskDefinition,
        TcpArguments,
    },
    client::SessionId,
    inline_value::VariableLookupKind,
    messages::Message,
};
use fs::{Fs, RemoveOptions};
use futures::{
    StreamExt, TryStreamExt as _,
    channel::mpsc::{self, UnboundedSender},
    future::{Shared, join_all},
};
use gpui::{App, AppContext, AsyncApp, Context, Entity, EventEmitter, SharedString, Task, TaskExt};
use http_client::HttpClient;
use language::{Buffer, LanguageToolchainStore};
use node_runtime::NodeRuntime;
use settings::InlayHintKind;
use sha2::{Digest as _, Sha256};

use remote::RemoteClient;
use rpc::{
    AnyProtoClient, TypedEnvelope,
    proto::{self},
};
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsLocation, WorktreeId};
use std::{
    borrow::Borrow,
    collections::BTreeMap,
    ffi::OsStr,
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    sync::{Arc, Once},
};
use task::{DebugScenario, SharedTaskContext, SpawnInTerminal, TaskTemplate};
use util::{ResultExt as _, rel_path::RelPath};
use worktree::Worktree;

pub type DiagnosticDebugAuthority = std::rc::Rc<dyn Fn(&App) -> Result<()>>;

pub struct DiagnosticDebugSession {
    pub session: Entity<Session>,
    pub boot: Task<Result<()>>,
}

pub struct DiagnosticDebugLaunch {
    store_id: gpui::EntityId,
    worktree: Entity<Worktree>,
    program: crate::ProjectPath,
    scan_id: usize,
    adapter: Arc<dyn DebugAdapter>,
    definition: DebugTaskDefinition,
    binary: DebugAdapterBinary,
    settings: Option<DiagnosticDebugSettings>,
    files: Vec<DiagnosticDebugFile>,
    directories: Vec<(PathBuf, u64)>,
}

impl DiagnosticDebugLaunch {
    pub fn binary(&self) -> &DebugAdapterBinary {
        &self.binary
    }

    pub fn definition(&self) -> &DebugTaskDefinition {
        &self.definition
    }

    pub fn program_path(&self) -> &crate::ProjectPath {
        &self.program
    }

    pub fn worktree_id(&self, cx: &App) -> WorktreeId {
        self.worktree.read(cx).id()
    }
}

#[derive(Clone, PartialEq)]
struct DiagnosticDebugSettings {
    custom_binary: Option<String>,
    arguments: Option<Vec<String>>,
    environment: Option<HashMap<String, String>>,
}

#[derive(Clone, PartialEq)]
struct DiagnosticDebugFile {
    path: PathBuf,
    inode: u64,
    modified: fs::MTime,
    size: u64,
    digest: [u8; 32],
    native_arm64: bool,
}

fn diagnostic_debug_configuration_strings(
    value: &serde_json::Value,
    depth: usize,
    remaining: &mut usize,
) -> bool {
    if depth == 0 || *remaining < 8 {
        return false;
    }
    *remaining -= 8;
    match value {
        serde_json::Value::String(value) => {
            if value.len() > 4096
                || value.len() > *remaining
                || value.chars().any(char::is_control)
                || value.contains("${")
            {
                return false;
            }
            *remaining -= value.len();
            true
        }
        serde_json::Value::Array(values) => {
            values.len() <= 256
                && values.iter().all(|value| {
                    diagnostic_debug_configuration_strings(value, depth - 1, remaining)
                })
        }
        serde_json::Value::Object(values) => {
            values.len() <= 64
                && values.iter().all(|(key, value)| {
                    if key.len() > 128
                        || key.len() > *remaining
                        || key.chars().any(char::is_control)
                    {
                        return false;
                    }
                    *remaining -= key.len();
                    diagnostic_debug_configuration_strings(value, depth - 1, remaining)
                })
        }
        _ => true,
    }
}

async fn diagnostic_debug_file(
    filesystem: Arc<dyn Fs>,
    path: PathBuf,
    executor: gpui::BackgroundExecutor,
) -> Result<DiagnosticDebugFile> {
    anyhow::ensure!(
        path.is_absolute() && filesystem.canonicalize(&path).await? == path,
        "Diagnostic executables must be absolute, canonical and free of path aliases"
    );
    let metadata = filesystem
        .metadata(&path)
        .await?
        .context("Diagnostic executable is missing")?;
    anyhow::ensure!(
        !metadata.is_dir
            && !metadata.is_fifo
            && !metadata.is_symlink
            && metadata.is_executable
            && metadata.len <= 256 * 1024 * 1024,
        "Diagnostic executable must be an installed bounded regular native file"
    );
    let parent = path
        .parent()
        .context("Diagnostic executable directory is missing")?;
    let parent_metadata = filesystem
        .metadata(parent)
        .await?
        .context("Diagnostic executable directory is missing")?;
    let name = path
        .file_name()
        .context("Diagnostic executable name is missing")?;
    let mut reader = filesystem
        .open_sync_confined_file(
            parent,
            Path::new(name),
            parent_metadata.inode,
            metadata.inode,
        )
        .await?;
    let size = metadata.len;
    let (digest, native_arm64) = executor
        .spawn(async move {
            use std::io::Read as _;
            let mut header = [0; 8];
            reader.read_exact(&mut header)?;
            let signature = [header[0], header[1], header[2], header[3]];
            anyhow::ensure!(
                matches!(
                    signature,
                    [0x7f, b'E', b'L', b'F']
                        | [0xcf, 0xfa, 0xed, 0xfe]
                        | [0xce, 0xfa, 0xed, 0xfe]
                        | [0xfe, 0xed, 0xfa, 0xcf]
                        | [0xca, 0xfe, 0xba, 0xbe]
                ) || header.starts_with(b"MZ"),
                "Diagnostic execution does not support scripts or shell shims"
            );
            let mut digest = Sha256::new();
            digest.update(header);
            let mut read_bytes = 8u64;
            let mut buffer = [0; 64 * 1024];
            loop {
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                read_bytes = read_bytes
                    .checked_add(count as u64)
                    .context("Executable size overflow")?;
                anyhow::ensure!(
                    read_bytes <= size,
                    "Diagnostic executable changed during inspection"
                );
                digest.update(buffer.get(..count).context("Invalid executable read")?);
            }
            anyhow::ensure!(
                read_bytes == size,
                "Diagnostic executable changed during inspection"
            );
            let native_arm64 = header[..4] == [0xcf, 0xfa, 0xed, 0xfe]
                && u32::from_le_bytes([header[4], header[5], header[6], header[7]]) == 0x0100_000c;
            Ok::<([u8; 32], bool), anyhow::Error>((digest.finalize().into(), native_arm64))
        })
        .await?;
    let current = filesystem
        .metadata(&path)
        .await?
        .context("Diagnostic executable disappeared")?;
    anyhow::ensure!(
        current.inode == metadata.inode
            && current.mtime == metadata.mtime
            && current.len == size
            && current.is_executable
            && !current.is_symlink
            && filesystem.canonicalize(&path).await? == path,
        "Diagnostic executable changed during inspection"
    );
    Ok(DiagnosticDebugFile {
        path,
        inode: metadata.inode,
        modified: metadata.mtime,
        size,
        digest,
        native_arm64,
    })
}

#[derive(Debug)]
pub enum DapStoreEvent {
    DebugClientStarted(SessionId),
    DebugSessionInitialized(SessionId),
    DebugClientShutdown(SessionId),
    DebugClientEvent {
        session_id: SessionId,
        message: Message,
    },
    Notification(String),
    RemoteHasInitialized,
}

enum DapStoreMode {
    Local(LocalDapStore),
    Remote(RemoteDapStore),
    Collab,
}

pub struct LocalDapStore {
    fs: Arc<dyn Fs>,
    node_runtime: NodeRuntime,
    http_client: Arc<dyn HttpClient>,
    environment: Entity<ProjectEnvironment>,
    toolchain_store: Arc<dyn LanguageToolchainStore>,
    is_headless: bool,
}

pub struct RemoteDapStore {
    remote_client: Entity<RemoteClient>,
    upstream_client: AnyProtoClient,
    upstream_project_id: u64,
    node_runtime: NodeRuntime,
    http_client: Arc<dyn HttpClient>,
}

pub struct DapStore {
    mode: DapStoreMode,
    downstream_client: Option<(AnyProtoClient, u64)>,
    breakpoint_store: Entity<BreakpointStore>,
    worktree_store: Entity<WorktreeStore>,
    sessions: BTreeMap<SessionId, Entity<Session>>,
    next_session_id: u32,
    adapter_options: BTreeMap<DebugAdapterName, Arc<PersistedAdapterOptions>>,
}

impl EventEmitter<DapStoreEvent> for DapStore {}

#[derive(Clone, Serialize, Deserialize)]
pub struct PersistedExceptionBreakpoint {
    pub enabled: bool,
}

/// Represents best-effort serialization of adapter state during last session (e.g. watches)
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct PersistedAdapterOptions {
    /// Which exception breakpoints were enabled during the last session with this adapter?
    pub exception_breakpoints: BTreeMap<String, PersistedExceptionBreakpoint>,
}

impl DapStore {
    pub fn init(client: &AnyProtoClient, cx: &mut App) {
        static ADD_LOCATORS: Once = Once::new();
        ADD_LOCATORS.call_once(|| {
            let registry = DapRegistry::global(cx);
            registry.add_locator(Arc::new(locators::cargo::CargoLocator {}));
            registry.add_locator(Arc::new(locators::go::GoLocator {}));
            registry.add_locator(Arc::new(locators::node::NodeLocator));
            registry.add_locator(Arc::new(locators::python::PythonLocator));
        });
        client.add_entity_request_handler(Self::handle_run_debug_locator);
        client.add_entity_request_handler(Self::handle_get_debug_adapter_binary);
        client.add_entity_message_handler(Self::handle_log_to_debug_console);
    }

    #[expect(clippy::too_many_arguments)]
    pub fn new_local(
        http_client: Arc<dyn HttpClient>,
        node_runtime: NodeRuntime,
        fs: Arc<dyn Fs>,
        environment: Entity<ProjectEnvironment>,
        toolchain_store: Arc<dyn LanguageToolchainStore>,
        worktree_store: Entity<WorktreeStore>,
        breakpoint_store: Entity<BreakpointStore>,
        is_headless: bool,
        cx: &mut Context<Self>,
    ) -> Self {
        let mode = DapStoreMode::Local(LocalDapStore {
            fs: fs.clone(),
            environment,
            http_client,
            node_runtime,
            toolchain_store,
            is_headless,
        });

        Self::new(mode, breakpoint_store, worktree_store, fs, cx)
    }

    pub fn new_remote(
        project_id: u64,
        remote_client: Entity<RemoteClient>,
        breakpoint_store: Entity<BreakpointStore>,
        worktree_store: Entity<WorktreeStore>,
        node_runtime: NodeRuntime,
        http_client: Arc<dyn HttpClient>,
        fs: Arc<dyn Fs>,
        cx: &mut Context<Self>,
    ) -> Self {
        let mode = DapStoreMode::Remote(RemoteDapStore {
            upstream_client: remote_client.read(cx).proto_client(),
            remote_client,
            upstream_project_id: project_id,
            node_runtime,
            http_client,
        });

        Self::new(mode, breakpoint_store, worktree_store, fs, cx)
    }

    pub fn new_collab(
        _project_id: u64,
        _upstream_client: AnyProtoClient,
        breakpoint_store: Entity<BreakpointStore>,
        worktree_store: Entity<WorktreeStore>,
        fs: Arc<dyn Fs>,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(
            DapStoreMode::Collab,
            breakpoint_store,
            worktree_store,
            fs,
            cx,
        )
    }

    fn new(
        mode: DapStoreMode,
        breakpoint_store: Entity<BreakpointStore>,
        worktree_store: Entity<WorktreeStore>,
        fs: Arc<dyn Fs>,
        cx: &mut Context<Self>,
    ) -> Self {
        if let Some(trusted_worktrees) =
            crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
        {
            cx.subscribe(&trusted_worktrees, |store, _, event, cx| {
                if matches!(event, crate::trusted_worktrees::TrustedWorktreesEvent::Restricted(worktrees, _) if *worktrees == store.worktree_store.downgrade()) {
                    let sessions = store
                        .sessions
                        .iter()
                        .filter(|(_, session)| {
                            let session = session.read(cx);
                            session.diagnostic_observation().is_some()
                                || (cfg!(feature = "jvm-tools") && session.adapter().0.as_ref() == "Java")
                        })
                        .map(|(session_id, _)| *session_id)
                        .collect::<Vec<_>>();
                    if !sessions.is_empty() {
                        cx.emit(DapStoreEvent::Notification(
                            "Reviewed debugging stopped because project trust was restricted; trust the project and review a new session".to_string(),
                        ));
                    }
                    for session_id in sessions {
                        if store.sessions.contains_key(&session_id) {
                            store.shutdown_session(session_id, cx).detach_and_log_err(cx);
                        }
                    }
                }
            })
            .detach();
        }
        cx.background_spawn(async move {
            let dir = paths::debug_adapters_dir().join("js-debug-companion");

            let mut children = fs.read_dir(&dir).await?.try_collect::<Vec<_>>().await?;
            children.sort_by_key(|child| semver::Version::parse(child.file_name()?.to_str()?).ok());

            if let Some(child) = children.last()
                && let Some(name) = child.file_name()
                && let Some(name) = name.to_str()
                && semver::Version::parse(name).is_ok()
            {
                children.pop();
            }

            for child in children {
                fs.remove_dir(
                    &child,
                    RemoveOptions {
                        recursive: true,
                        ignore_if_not_exists: true,
                    },
                )
                .await
                .ok();
            }

            anyhow::Ok(())
        })
        .detach();

        Self {
            mode,
            next_session_id: 0,
            downstream_client: None,
            breakpoint_store,
            worktree_store,
            sessions: Default::default(),
            adapter_options: Default::default(),
        }
    }

    #[cfg(feature = "test-support")]
    pub fn diagnostic_filesystem_for_test(&mut self, filesystem: Arc<dyn Fs>) -> Result<()> {
        let DapStoreMode::Local(local) = &mut self.mode else {
            anyhow::bail!("Diagnostic filesystem fixture requires a local store");
        };
        local.fs = filesystem;
        Ok(())
    }

    fn diagnostic_debug_authority(&self, worktree: &Entity<Worktree>, cx: &App) -> Result<()> {
        anyhow::ensure!(
            matches!(self.mode, DapStoreMode::Local(_))
                && worktree.read(cx).as_local().is_some()
                && self
                    .worktree_store
                    .read(cx)
                    .worktree_for_id(worktree.read(cx).id(), cx)
                    .is_some_and(|selected| selected == *worktree),
            "Diagnostic debugger preparation is unavailable on this project or execution host"
        );
        let trusted =
            crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx).and_then(|trusted| {
                trusted.read(cx).observed_worktree_trust(
                    &self.worktree_store,
                    worktree.read(cx).id(),
                    cx,
                )
            });
        anyhow::ensure!(
            trusted == Some(true)
                && !crate::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &self.worktree_store,
                    cx
                ),
            "Trust the selected project before diagnostic debugger preparation"
        );
        Ok(())
    }

    fn diagnostic_debug_binary(
        &self,
        definition: &DebugTaskDefinition,
        worktree: &Entity<Worktree>,
        cx: &mut App,
    ) -> Result<(
        Arc<dyn DebugAdapter>,
        DebugAdapterBinary,
        Option<DiagnosticDebugSettings>,
    )> {
        self.diagnostic_debug_authority(worktree, cx)?;
        anyhow::ensure!(
            definition.tcp_connection.is_none()
                && definition
                    .config
                    .get("request")
                    .is_some_and(|request| request == "launch")
                && diagnostic_debug_configuration_strings(&definition.config, 8, &mut (64 * 1024))
                && definition.config.to_string().len() <= 64 * 1024,
            "Diagnostic launch configuration is unresolved, excessive or requests a remote connection"
        );
        let adapter = DapRegistry::global(cx)
            .adapter(&definition.adapter)
            .context("Diagnostic adapter is not installed")?;
        let settings = ProjectSettings::get(
            Some(SettingsLocation {
                worktree_id: worktree.read(cx).id(),
                path: RelPath::empty(),
            }),
            cx,
        )
        .dap
        .get(&adapter.name());
        let settings_identity = settings.map(|settings| DiagnosticDebugSettings {
            custom_binary: match &settings.binary {
                DapBinary::Default => None,
                DapBinary::Custom(path) => Some(path.clone()),
            },
            arguments: settings.args.clone(),
            environment: settings.env.clone(),
        });
        let installed = settings.and_then(|settings| match &settings.binary {
            DapBinary::Custom(path) => {
                Some(worktree.read(cx).resolve_relative_path(PathBuf::from(path)))
            }
            DapBinary::Default => None,
        });
        let arguments = settings.and_then(|settings| settings.args.as_deref());
        let environment = settings.and_then(|settings| settings.env.as_ref());
        let binary = adapter.diagnostic_installed_binary(
            definition,
            &worktree.read(cx).abs_path(),
            installed.as_deref(),
            arguments,
            environment,
        )?;
        anyhow::ensure!(
            binary.connection.is_none()
                && binary.request_args.request
                    == dap::StartDebuggingRequestArgumentsRequest::Launch,
            "Diagnostic debugging requires an owned installed stdio launch"
        );
        let expected_directory = definition
            .config
            .get("cwd")
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| worktree.read(cx).abs_path().to_path_buf());
        anyhow::ensure!(
            binary.request_args.configuration.get("program") == definition.config.get("program")
                && binary
                    .request_args
                    .configuration
                    .get("cwd")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(|directory| Path::new(directory) == expected_directory)
                && binary.cwd.as_deref() == Some(worktree.read(cx).abs_path().as_ref()),
            "Diagnostic adapter changed the selected project program or working directory"
        );
        for key in binary.envs.keys().chain(
            binary
                .request_args
                .configuration
                .get("env")
                .and_then(serde_json::Value::as_object)
                .into_iter()
                .flat_map(|environment| environment.keys()),
        ) {
            anyhow::ensure!(
                !key.is_empty()
                    && key
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                    && !util::redact::should_redact(&key.to_ascii_uppercase())
                    && !key.starts_with("LD_")
                    && !key.starts_with("DYLD_")
                    && !matches!(
                        key.as_str(),
                        "BASH_ENV" | "ENV" | "PYTHONPATH" | "PYTHONHOME" | "NODE_OPTIONS"
                    ),
                "Diagnostic launches cannot introduce credentials or implicit execution hooks"
            );
        }
        anyhow::ensure!(
            binary.envs.len() <= 64
                && binary
                    .envs
                    .iter()
                    .all(|(key, value)| !key.contains(['=', '\0'])
                        && value.len() <= 4096
                        && !value.chars().any(char::is_control)),
            "Diagnostic adapter environment is invalid or excessive"
        );
        Ok((adapter, binary, settings_identity))
    }

    pub fn prepare_diagnostic_launch(
        &self,
        definition: DebugTaskDefinition,
        worktree: Entity<Worktree>,
        cx: &mut Context<Self>,
    ) -> Task<Result<DiagnosticDebugLaunch>> {
        if let Err(error) = self.diagnostic_debug_authority(&worktree, cx) {
            return Task::ready(Err(error));
        }
        let Some(local) = self.as_local() else {
            return Task::ready(Err(anyhow!(
                "Diagnostic launch preparation is unavailable on this host"
            )));
        };
        let filesystem = local.fs.clone();
        let store_id = cx.entity_id();
        let root = worktree.read(cx).abs_path();
        let scan_id = worktree.read(cx).scan_id();
        cx.spawn(async move |store, cx| {
            let canonical_root = filesystem.canonicalize(&root).await?;
            anyhow::ensure!(canonical_root == root.as_ref(), "Diagnostic project root is aliased");
            let mut directories = Vec::new();
            let mut project_program = None;
            for key in ["program", "cwd"] {
                let selected = definition.config.get(key).and_then(serde_json::Value::as_str)
                    .map(PathBuf::from).or_else(|| (key == "cwd").then(|| root.to_path_buf()))
                    .context("Select a fully resolved diagnostic program")?;
                let canonical = filesystem.canonicalize(&selected).await?;
                anyhow::ensure!(selected.is_absolute() && canonical == selected && selected.starts_with(&root),
                    "Diagnostic program and working directory must be contained canonical Project paths");
                let relative = selected.strip_prefix(&root)?;
                let relative = RelPath::new(relative, util::paths::PathStyle::local())?;
                cx.update(|cx| {
                    anyhow::ensure!(!worktree.read(cx).as_local().context("Diagnostic local worktree became unavailable")?.is_path_private(relative.as_ref()), "Diagnostic path is private");
                    Ok::<_, anyhow::Error>(())
                })?;
                if key == "program" {
                    project_program = Some(crate::ProjectPath {
                        worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
                        path: relative.into_owned().into(),
                    });
                }
                let metadata = filesystem.metadata(&selected).await?.context("Diagnostic input is missing")?;
                if key == "program" {
                    anyhow::ensure!(!metadata.is_dir && !metadata.is_fifo && !metadata.is_symlink
                        && metadata.is_executable && metadata.len <= 256 * 1024 * 1024,
                        "Diagnostic program must be a bounded regular native executable");
                }
                if key == "cwd" {
                    anyhow::ensure!(metadata.is_dir && !metadata.is_symlink, "Diagnostic working directory is not a contained directory");
                    directories.push((selected, metadata.inode));
                }
            }
            let root_metadata = filesystem.metadata(&root).await?.context("Diagnostic root is missing")?;
            directories.push((root.to_path_buf(), root_metadata.inode));
            let (adapter, binary, settings) = store.update(cx, |store, cx| store.diagnostic_debug_binary(&definition, &worktree, cx))??;
            let program = PathBuf::from(binary.request_args.configuration.get("program").and_then(serde_json::Value::as_str)
                .context("Diagnostic program is missing")?);
            let command = PathBuf::from(binary.command.as_ref().context("Diagnostic adapter binary is missing")?);
            let mut paths = vec![program, command];
            if let Some(debugserver) = binary.envs.get("LLDB_DEBUGSERVER_PATH") { paths.push(PathBuf::from(debugserver)); }
            let mut files = Vec::new();
            for path in paths {
                files.push(diagnostic_debug_file(filesystem.clone(), path, cx.background_executor().clone()).await?);
            }
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            anyhow::ensure!(definition.adapter.0.as_ref() != "CodeLLDB"
                || !files.first().is_some_and(|program| program.native_arm64)
                || binary.envs.contains_key("LLDB_DEBUGSERVER_PATH"),
                "Native ARM64 debugging requires an explicitly configured installed debugserver; Doctor will not create a launcher");
            store.update(cx, |store, cx| {
                let (current_adapter, current, current_settings) = store.diagnostic_debug_binary(&definition, &worktree, cx)?;
                anyhow::ensure!(Arc::ptr_eq(&adapter, &current_adapter) && current == binary && current_settings == settings && worktree.read(cx).scan_id() == scan_id,
                    "Diagnostic adapter, settings or project inputs changed during preparation");
                Ok::<_, anyhow::Error>(())
            })??;
            Ok(DiagnosticDebugLaunch { store_id, worktree, program: project_program.context("Diagnostic ProjectPath is missing")?, scan_id, adapter, definition, binary, settings, files, directories })
        })
    }

    pub fn revalidate_diagnostic_launch(
        &self,
        reviewed: &DiagnosticDebugLaunch,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if reviewed.store_id != cx.entity_id() {
            return Task::ready(Err(anyhow!("Diagnostic launch belongs to another project")));
        }
        let pending = self.prepare_diagnostic_launch(
            reviewed.definition.clone(),
            reviewed.worktree.clone(),
            cx,
        );
        let scan_id = reviewed.scan_id;
        let adapter = reviewed.adapter.clone();
        let binary = reviewed.binary.clone();
        let settings = reviewed.settings.clone();
        let files = reviewed.files.clone();
        let directories = reviewed.directories.clone();
        cx.spawn(async move |_, _| {
            let current = pending.await?;
            anyhow::ensure!(scan_id == current.scan_id && Arc::ptr_eq(&adapter, &current.adapter)
                && binary == current.binary && settings == current.settings && files == current.files && directories == current.directories,
                "Reviewed diagnostic executable, configuration or project inputs changed; review a new launch");
            Ok(())
        })
    }

    pub fn start_diagnostic_session(
        &mut self,
        reviewed: DiagnosticDebugLaunch,
        task_context: SharedTaskContext,
        authority: DiagnosticDebugAuthority,
        cx: &mut Context<Self>,
    ) -> Task<Result<DiagnosticDebugSession>> {
        let revalidation = self.revalidate_diagnostic_launch(&reviewed, cx);
        cx.spawn(async move |store, cx| {
            revalidation.await?;
            cx.update(|cx| authority(cx))?;
            store.update(cx, |store, cx| {
                store.diagnostic_debug_authority(&reviewed.worktree, cx)?;
                let (adapter, binary, settings) =
                    store.diagnostic_debug_binary(&reviewed.definition, &reviewed.worktree, cx)?;
                anyhow::ensure!(
                    Arc::ptr_eq(&adapter, &reviewed.adapter)
                        && binary == reviewed.binary
                        && settings == reviewed.settings
                        && reviewed
                            .worktree
                            .read(cx)
                            .as_local()
                            .context("Local worktree required")?
                            .scan_id()
                            == reviewed.scan_id,
                    "Diagnostic launch changed; review a new launch"
                );
                let session = store.new_session(
                    Some(reviewed.definition.label.clone()),
                    reviewed.definition.adapter.clone(),
                    task_context,
                    None,
                    SessionQuirks {
                        compact: adapter.compact_child_session(),
                        prefer_thread_name: adapter.prefer_thread_name(),
                    },
                    cx,
                );
                let session_id = session.read(cx).session_id();
                if let Err(error) =
                    session.update(cx, |session, _| session.restrict_diagnostic_collection())
                {
                    store
                        .shutdown_session(session_id, cx)
                        .detach_and_log_err(cx);
                    return Err(error);
                }
                let store_handle = cx.entity().downgrade();
                let pending = session.update(cx, |session, cx| {
                    session.boot(reviewed.binary, reviewed.worktree, store_handle, cx)
                });
                Ok(DiagnosticDebugSession {
                    session,
                    boot: pending,
                })
            })?
        })
    }

    pub fn get_debug_adapter_binary(
        &mut self,
        definition: DebugTaskDefinition,
        session_id: SessionId,
        worktree: &Entity<Worktree>,
        console: UnboundedSender<String>,
        cx: &mut Context<Self>,
    ) -> Task<Result<DebugAdapterBinary>> {
        #[cfg(feature = "jvm-tools")]
        if definition.adapter.0.as_ref() == "Java"
            && !self.groovy_java_debug_worktree_trusted(worktree, cx)
        {
            return Task::ready(Err(anyhow!("Trust the project before starting Java Debug")));
        }
        match &self.mode {
            DapStoreMode::Local(_) => {
                let Some(adapter) = DapRegistry::global(cx).adapter(&definition.adapter) else {
                    return Task::ready(Err(anyhow!("Failed to find a debug adapter")));
                };

                let settings_location = SettingsLocation {
                    worktree_id: worktree.read(cx).id(),
                    path: RelPath::empty(),
                };
                let dap_settings = ProjectSettings::get(Some(settings_location), cx)
                    .dap
                    .get(&adapter.name());
                let user_installed_path = dap_settings.and_then(|s| match &s.binary {
                    DapBinary::Default => None,
                    DapBinary::Custom(binary) => {
                        let path = PathBuf::from(binary);
                        Some(worktree.read(cx).resolve_relative_path(path))
                    }
                });
                let user_args = dap_settings.and_then(|s| s.args.clone());
                let user_env = dap_settings.and_then(|s| s.env.clone());

                let delegate = self.delegate(worktree, console, cx);

                let worktree = worktree.clone();
                cx.spawn(async move |this, cx| {
                    let mut binary = adapter
                        .get_binary(
                            &delegate,
                            &definition,
                            user_installed_path,
                            user_args,
                            user_env,
                            cx,
                        )
                        .await?;

                    #[cfg(feature = "jvm-tools")]
                    if definition.adapter.0.as_ref() == "Java"
                        && !this.update(cx, |store, cx| {
                            store.groovy_java_debug_worktree_trusted(&worktree, cx)
                        })?
                    {
                        anyhow::bail!("Trust the project before starting Java Debug");
                    }

                    let env = this
                        .update(cx, |this, cx| {
                            this.as_local()
                                .unwrap()
                                .environment
                                .update(cx, |environment, cx| {
                                    environment.worktree_environment(worktree, cx)
                                })
                        })?
                        .await;

                    if let Some(mut env) = env {
                        env.extend(std::mem::take(&mut binary.envs));
                        binary.envs = env;
                    }

                    Ok(binary)
                })
            }
            DapStoreMode::Remote(remote) => {
                let request = remote
                    .upstream_client
                    .request(proto::GetDebugAdapterBinary {
                        session_id: session_id.to_proto(),
                        project_id: remote.upstream_project_id,
                        worktree_id: worktree.read(cx).id().to_proto(),
                        definition: Some(definition.to_proto()),
                    });
                let remote = remote.remote_client.clone();

                cx.spawn(async move |_, cx| {
                    let response = request.await?;
                    let binary = DebugAdapterBinary::from_proto(response)?;

                    let port_forwarding;
                    let connection;
                    if let Some(c) = binary.connection {
                        let host = IpAddr::V4(Ipv4Addr::LOCALHOST);
                        let port;
                        if remote.read_with(cx, |remote, _cx| remote.shares_network_interface()) {
                            port = c.port;
                            port_forwarding = None;
                        } else {
                            port = dap::transport::TcpTransport::unused_port(host).await?;
                            port_forwarding = Some((port, c.host.to_string(), c.port));
                        }
                        connection = Some(TcpArguments {
                            port,
                            host,
                            timeout: c.timeout,
                        })
                    } else {
                        port_forwarding = None;
                        connection = None;
                    }

                    let command = remote.read_with(cx, |remote, _cx| {
                        if binary.command.is_none() {
                            if let Some(port_forwarding) = port_forwarding {
                                return remote
                                    .build_forward_ports_command(vec![port_forwarding])
                                    .map(Some);
                            }
                            return Ok(None);
                        }

                        remote
                            .build_command(
                                binary.command,
                                &binary.arguments,
                                &binary.envs,
                                binary.cwd.map(|path| path.display().to_string()),
                                port_forwarding,
                                Interactive::No,
                            )
                            .map(Some)
                    })?;

                    let (command, arguments, envs) = match command {
                        Some(command) => (Some(command.program), command.args, command.env),
                        None => (None, Vec::new(), HashMap::default()),
                    };

                    Ok(DebugAdapterBinary {
                        command,
                        arguments,
                        envs,
                        cwd: None,
                        connection,
                        request_args: binary.request_args,
                    })
                })
            }
            DapStoreMode::Collab => {
                Task::ready(Err(anyhow!("Debugging is not yet supported via collab")))
            }
        }
    }

    pub fn debug_scenario_for_build_task(
        &self,
        build: TaskTemplate,
        adapter: DebugAdapterName,
        label: SharedString,
        cx: &mut App,
    ) -> Task<Option<DebugScenario>> {
        let locators = DapRegistry::global(cx).locators();

        cx.background_spawn(async move {
            for locator in locators.values() {
                if let Some(scenario) = locator.create_scenario(&build, &label, &adapter).await {
                    return Some(scenario);
                }
            }
            None
        })
    }

    pub fn run_debug_locator(
        &mut self,
        locator_name: &str,
        build_command: SpawnInTerminal,
        cx: &mut Context<Self>,
    ) -> Task<Result<DebugRequest>> {
        match &self.mode {
            DapStoreMode::Local(_) => {
                // Pre-resolve args with existing environment.
                let locators = DapRegistry::global(cx).locators();
                let locator = locators.get(locator_name);
                let executor = cx.background_executor().clone();

                if let Some(locator) = locator.cloned() {
                    cx.background_spawn(async move {
                        let result = locator
                            .run(build_command.clone(), executor)
                            .await
                            .log_with_level(log::Level::Error);
                        if let Some(result) = result {
                            return Ok(result);
                        }

                        anyhow::bail!(
                            "None of the locators for task `{}` completed successfully",
                            build_command.label
                        )
                    })
                } else {
                    Task::ready(Err(anyhow!(
                        "Couldn't find any locator for task `{}`. Specify the `attach` or `launch` arguments in your debug scenario definition",
                        build_command.label
                    )))
                }
            }
            DapStoreMode::Remote(remote) => {
                let request = remote.upstream_client.request(proto::RunDebugLocators {
                    project_id: remote.upstream_project_id,
                    build_command: Some(build_command.to_proto()),
                    locator: locator_name.to_owned(),
                });
                cx.background_spawn(async move {
                    let response = request.await?;
                    DebugRequest::from_proto(response)
                })
            }
            DapStoreMode::Collab => {
                Task::ready(Err(anyhow!("Debugging is not yet supported via collab")))
            }
        }
    }

    fn as_local(&self) -> Option<&LocalDapStore> {
        match &self.mode {
            DapStoreMode::Local(local_dap_store) => Some(local_dap_store),
            _ => None,
        }
    }

    pub fn new_session(
        &mut self,
        label: Option<SharedString>,
        adapter: DebugAdapterName,
        task_context: SharedTaskContext,
        parent_session: Option<Entity<Session>>,
        quirks: SessionQuirks,
        cx: &mut Context<Self>,
    ) -> Entity<Session> {
        let session_id = SessionId(util::post_inc(&mut self.next_session_id));

        if let Some(session) = &parent_session {
            session.update(cx, |session, _| {
                session.add_child_session_id(session_id);
            });
        }

        let (remote_client, node_runtime, http_client) = match &self.mode {
            DapStoreMode::Local(_) => (None, None, None),
            DapStoreMode::Remote(remote_dap_store) => (
                Some(remote_dap_store.remote_client.clone()),
                Some(remote_dap_store.node_runtime.clone()),
                Some(remote_dap_store.http_client.clone()),
            ),
            DapStoreMode::Collab => (None, None, None),
        };
        let session = Session::new(
            self.breakpoint_store.clone(),
            session_id,
            parent_session,
            label,
            adapter,
            task_context,
            quirks,
            remote_client,
            node_runtime,
            http_client,
            cx,
        );

        self.sessions.insert(session_id, session.clone());
        cx.notify();

        cx.subscribe(&session, {
            move |this: &mut DapStore, _, event: &SessionStateEvent, cx| match event {
                SessionStateEvent::Shutdown => {
                    if this.sessions.contains_key(&session_id) {
                        this.shutdown_session(session_id, cx).detach_and_log_err(cx);
                    }
                }
                SessionStateEvent::Restart | SessionStateEvent::SpawnChildSession { .. } => {}
                SessionStateEvent::Running => {
                    cx.emit(DapStoreEvent::DebugClientStarted(session_id));
                }
            }
        })
        .detach();

        session
    }

    pub fn boot_session(
        &self,
        session: Entity<Session>,
        definition: DebugTaskDefinition,
        worktree: Entity<Worktree>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if let Err(error) = session.read(cx).ensure_experimental_access(cx) {
            return Task::ready(Err(error));
        }
        let dap_store = cx.weak_entity();
        let console = session.update(cx, |session, cx| session.console_output(cx));
        let session_id = session.read(cx).session_id();

        cx.spawn({
            let session = session.clone();
            async move |this, cx| {
                #[cfg(feature = "jvm-tools")]
                if definition.adapter.0.as_ref() == "Java"
                    && !this.update(cx, |store, cx| {
                        store.groovy_java_debug_worktree_trusted(&worktree, cx)
                    })?
                {
                    anyhow::bail!("Trust the project before starting Java Debug");
                }
                let binary = this
                    .update(cx, |this, cx| {
                        this.get_debug_adapter_binary(
                            definition.clone(),
                            session_id,
                            &worktree,
                            console,
                            cx,
                        )
                    })?
                    .await?;
                #[cfg(feature = "jvm-tools")]
                if definition.adapter.0.as_ref() == "Java"
                    && !this.update(cx, |store, cx| {
                        store.groovy_java_debug_worktree_trusted(&worktree, cx)
                    })?
                {
                    anyhow::bail!("Trust the project before starting Java Debug");
                }
                session.read_with(cx, |session, cx| session.ensure_experimental_access(cx))?;
                session
                    .update(cx, |session, cx| {
                        session.boot(binary, worktree, dap_store, cx)
                    })
                    .await
            }
        })
    }

    #[cfg(feature = "jvm-tools")]
    fn groovy_java_debug_worktree_trusted(
        &self,
        worktree: &Entity<Worktree>,
        cx: &mut Context<Self>,
    ) -> bool {
        let worktree_id = worktree.read(cx).id();
        if self
            .worktree_store
            .read(cx)
            .worktree_for_id(worktree_id, cx)
            .is_none()
        {
            return false;
        }
        let Some(trusted_worktrees) =
            crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
        else {
            return false;
        };
        trusted_worktrees.update(cx, |trusted_worktrees, cx| {
            trusted_worktrees.can_trust(&self.worktree_store, worktree_id, cx)
        }) && !crate::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
            &self.worktree_store,
            cx,
        )
    }

    pub fn session_by_id(
        &self,
        session_id: impl Borrow<SessionId>,
    ) -> Option<Entity<session::Session>> {
        let session_id = session_id.borrow();

        self.sessions.get(session_id).cloned()
    }
    pub fn sessions(&self) -> impl Iterator<Item = &Entity<Session>> {
        self.sessions.values()
    }

    pub fn capabilities_by_id(
        &self,
        session_id: impl Borrow<SessionId>,
        cx: &App,
    ) -> Option<Capabilities> {
        let session_id = session_id.borrow();
        self.sessions
            .get(session_id)
            .map(|client| client.read(cx).capabilities.clone())
    }

    pub fn breakpoint_store(&self) -> &Entity<BreakpointStore> {
        &self.breakpoint_store
    }

    pub fn worktree_store(&self) -> &Entity<WorktreeStore> {
        &self.worktree_store
    }

    #[allow(dead_code)]
    async fn handle_ignore_breakpoint_state(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::IgnoreBreakpointState>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let session_id = SessionId::from_proto(envelope.payload.session_id);

        this.update(&mut cx, |this, cx| {
            if let Some(session) = this.session_by_id(&session_id) {
                session.update(cx, |session, cx| {
                    session.set_ignore_breakpoints(envelope.payload.ignore, cx)
                })
            } else {
                Task::ready(HashMap::default())
            }
        })
        .await;

        Ok(())
    }

    fn delegate(
        &self,
        worktree: &Entity<Worktree>,
        console: UnboundedSender<String>,
        cx: &mut App,
    ) -> Arc<dyn DapDelegate> {
        let Some(local_store) = self.as_local() else {
            unimplemented!("Starting session on remote side");
        };

        Arc::new(DapAdapterDelegate::new(
            local_store.fs.clone(),
            worktree.read(cx).snapshot(),
            console,
            local_store.node_runtime.clone(),
            local_store.http_client.clone(),
            local_store.toolchain_store.clone(),
            local_store
                .environment
                .update(cx, |env, cx| env.worktree_environment(worktree.clone(), cx)),
            local_store.is_headless,
        ))
    }

    pub fn resolve_inline_value_locations(
        &self,
        session: Entity<Session>,
        stack_frame_id: StackFrameId,
        buffer_handle: Entity<Buffer>,
        inline_value_locations: Vec<dap::inline_value::InlineValueLocation>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Vec<InlayHint>>> {
        let snapshot = buffer_handle.read(cx).snapshot();
        let local_variables =
            session
                .read(cx)
                .variables_by_stack_frame_id(stack_frame_id, false, true);
        let global_variables =
            session
                .read(cx)
                .variables_by_stack_frame_id(stack_frame_id, true, false);

        fn format_value(mut value: String) -> String {
            const LIMIT: usize = 100;

            if let Some(index) = value.find("\n") {
                value.truncate(index);
                value.push_str("…");
            }

            if value.len() > LIMIT {
                let mut index = LIMIT;
                // If index isn't a char boundary truncate will cause a panic
                while !value.is_char_boundary(index) {
                    index -= 1;
                }
                value.truncate(index);
                value.push_str("…");
            }

            format!(": {}", value)
        }

        cx.spawn(async move |_, cx| {
            let mut inlay_hints = Vec::with_capacity(inline_value_locations.len());
            for inline_value_location in inline_value_locations.iter() {
                let point = snapshot.point_to_point_utf16(language::Point::new(
                    inline_value_location.row as u32,
                    inline_value_location.column as u32,
                ));
                let position = snapshot.anchor_after(point);

                match inline_value_location.lookup {
                    VariableLookupKind::Variable => {
                        let variable_search =
                            if inline_value_location.scope
                                == dap::inline_value::VariableScope::Local
                            {
                                local_variables.iter().chain(global_variables.iter()).find(
                                    |variable| variable.name == inline_value_location.variable_name,
                                )
                            } else {
                                global_variables.iter().find(|variable| {
                                    variable.name == inline_value_location.variable_name
                                })
                            };

                        let Some(variable) = variable_search else {
                            continue;
                        };

                        inlay_hints.push(InlayHint {
                            position,
                            label: InlayHintLabel::String(format_value(variable.value.clone())),
                            kind: Some(InlayHintKind::Type),
                            padding_left: false,
                            padding_right: false,
                            tooltip: None,
                            resolve_state: ResolveState::Resolved,
                        });
                    }
                    VariableLookupKind::Expression => {
                        let eval_task = session.read_with(cx, |session, _| {
                            session.state.request_dap(EvaluateCommand {
                                expression: inline_value_location.variable_name.clone(),
                                frame_id: Some(stack_frame_id),
                                source: None,
                                context: Some(EvaluateArgumentsContext::Variables),
                            })
                        });

                        if let Some(response) = eval_task.await.log_err() {
                            inlay_hints.push(InlayHint {
                                position,
                                label: InlayHintLabel::String(format_value(response.result)),
                                kind: Some(InlayHintKind::Type),
                                padding_left: false,
                                padding_right: false,
                                tooltip: None,
                                resolve_state: ResolveState::Resolved,
                            });
                        };
                    }
                };
            }

            Ok(inlay_hints)
        })
    }

    pub fn shutdown_sessions(&mut self, cx: &mut Context<Self>) -> Task<()> {
        let mut tasks = vec![];
        for session_id in self.sessions.keys().cloned().collect::<Vec<_>>() {
            tasks.push(self.shutdown_session(session_id, cx));
        }

        cx.background_executor().spawn(async move {
            futures::future::join_all(tasks).await;
        })
    }

    pub fn shutdown_session(
        &mut self,
        session_id: SessionId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let Some(session) = self.sessions.remove(&session_id) else {
            return Task::ready(Err(anyhow!("Could not find session: {:?}", session_id)));
        };

        let shutdown_children = session
            .read(cx)
            .child_session_ids()
            .iter()
            .map(|session_id| self.shutdown_session(*session_id, cx))
            .collect::<Vec<_>>();

        let shutdown_parent_task = if let Some(parent_session) = session
            .read(cx)
            .parent_id(cx)
            .and_then(|session_id| self.session_by_id(session_id))
        {
            let shutdown_id = parent_session.update(cx, |parent_session, _| {
                parent_session.remove_child_session_id(session_id);

                if parent_session.child_session_ids().is_empty() {
                    Some(parent_session.session_id())
                } else {
                    None
                }
            });

            shutdown_id.map(|session_id| self.shutdown_session(session_id, cx))
        } else {
            None
        };

        let shutdown_task = session.update(cx, |this, cx| this.shutdown(cx));

        cx.emit(DapStoreEvent::DebugClientShutdown(session_id));

        cx.background_spawn(async move {
            if !shutdown_children.is_empty() {
                let _ = join_all(shutdown_children).await;
            }

            shutdown_task.await;

            if let Some(parent_task) = shutdown_parent_task {
                parent_task.await?;
            }

            Ok(())
        })
    }

    pub fn shared(
        &mut self,
        project_id: u64,
        downstream_client: AnyProtoClient,
        _: &mut Context<Self>,
    ) {
        self.downstream_client = Some((downstream_client, project_id));
    }

    pub fn unshared(&mut self, cx: &mut Context<Self>) {
        self.downstream_client.take();

        cx.notify();
    }

    async fn handle_run_debug_locator(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::RunDebugLocators>,
        mut cx: AsyncApp,
    ) -> Result<proto::DebugRequest> {
        let task = envelope
            .payload
            .build_command
            .context("missing definition")?;
        let build_task = SpawnInTerminal::from_proto(task);
        let locator = envelope.payload.locator;
        let request = this
            .update(&mut cx, |this, cx| {
                this.run_debug_locator(&locator, build_task, cx)
            })
            .await?;

        Ok(request.to_proto())
    }

    async fn handle_get_debug_adapter_binary(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetDebugAdapterBinary>,
        mut cx: AsyncApp,
    ) -> Result<proto::DebugAdapterBinary> {
        let definition = DebugTaskDefinition::from_proto(
            envelope.payload.definition.context("missing definition")?,
        )?;
        let (tx, mut rx) = mpsc::unbounded();
        let session_id = envelope.payload.session_id;
        cx.spawn({
            let this = this.clone();
            async move |cx| {
                while let Some(message) = rx.next().await {
                    this.read_with(cx, |this, _| {
                        if let Some((downstream, project_id)) = this.downstream_client.clone() {
                            downstream
                                .send(proto::LogToDebugConsole {
                                    project_id,
                                    session_id,
                                    message,
                                })
                                .ok();
                        }
                    });
                }
            }
        })
        .detach();

        let worktree = this
            .update(&mut cx, |this, cx| {
                this.worktree_store
                    .read(cx)
                    .worktree_for_id(WorktreeId::from_proto(envelope.payload.worktree_id), cx)
            })
            .context("Failed to find worktree with a given ID")?;
        let binary = this
            .update(&mut cx, |this, cx| {
                this.get_debug_adapter_binary(
                    definition,
                    SessionId::from_proto(session_id),
                    &worktree,
                    tx,
                    cx,
                )
            })
            .await?;
        Ok(binary.to_proto())
    }

    async fn handle_log_to_debug_console(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::LogToDebugConsole>,
        mut cx: AsyncApp,
    ) -> Result<()> {
        let session_id = SessionId::from_proto(envelope.payload.session_id);
        this.update(&mut cx, |this, cx| {
            let Some(session) = this.sessions.get(&session_id) else {
                return;
            };
            session.update(cx, |session, cx| {
                session
                    .console_output(cx)
                    .unbounded_send(envelope.payload.message)
                    .ok();
            })
        });
        Ok(())
    }

    pub fn sync_adapter_options(
        &mut self,
        session: &Entity<Session>,
        cx: &App,
    ) -> Arc<PersistedAdapterOptions> {
        let session = session.read(cx);
        let adapter = session.adapter();
        let exceptions = session.exception_breakpoints();
        let exception_breakpoints = exceptions
            .map(|(exception, enabled)| {
                (
                    exception.filter.clone(),
                    PersistedExceptionBreakpoint { enabled: *enabled },
                )
            })
            .collect();
        let options = Arc::new(PersistedAdapterOptions {
            exception_breakpoints,
        });
        self.adapter_options.insert(adapter, options.clone());
        options
    }

    pub fn set_adapter_options(
        &mut self,
        adapter: DebugAdapterName,
        options: PersistedAdapterOptions,
    ) {
        self.adapter_options.insert(adapter, Arc::new(options));
    }

    pub fn adapter_options(&self, name: &str) -> Option<Arc<PersistedAdapterOptions>> {
        self.adapter_options.get(name).cloned()
    }

    pub fn all_adapter_options(&self) -> &BTreeMap<DebugAdapterName, Arc<PersistedAdapterOptions>> {
        &self.adapter_options
    }
}

#[derive(Clone)]
pub struct DapAdapterDelegate {
    fs: Arc<dyn Fs>,
    console: mpsc::UnboundedSender<String>,
    worktree: worktree::Snapshot,
    node_runtime: NodeRuntime,
    http_client: Arc<dyn HttpClient>,
    toolchain_store: Arc<dyn LanguageToolchainStore>,
    load_shell_env_task: Shared<Task<Option<HashMap<String, String>>>>,
    is_headless: bool,
}

impl DapAdapterDelegate {
    pub fn new(
        fs: Arc<dyn Fs>,
        worktree: worktree::Snapshot,
        status: mpsc::UnboundedSender<String>,
        node_runtime: NodeRuntime,
        http_client: Arc<dyn HttpClient>,
        toolchain_store: Arc<dyn LanguageToolchainStore>,
        load_shell_env_task: Shared<Task<Option<HashMap<String, String>>>>,
        is_headless: bool,
    ) -> Self {
        Self {
            fs,
            console: status,
            worktree,
            http_client,
            node_runtime,
            toolchain_store,
            load_shell_env_task,
            is_headless,
        }
    }
}

#[async_trait]
impl dap::adapters::DapDelegate for DapAdapterDelegate {
    fn worktree_id(&self) -> WorktreeId {
        self.worktree.id()
    }

    fn worktree_root_path(&self) -> &Path {
        self.worktree.abs_path()
    }
    fn http_client(&self) -> Arc<dyn HttpClient> {
        self.http_client.clone()
    }

    fn node_runtime(&self) -> NodeRuntime {
        self.node_runtime.clone()
    }

    fn fs(&self) -> Arc<dyn Fs> {
        self.fs.clone()
    }

    fn output_to_console(&self, msg: String) {
        self.console.unbounded_send(msg).ok();
    }

    #[cfg(not(target_os = "windows"))]
    async fn which(&self, command: &OsStr) -> Option<PathBuf> {
        let worktree_abs_path = self.worktree.abs_path();
        let shell_path = self.shell_env().await.get("PATH").cloned();
        which::which_in(command, shell_path.as_ref(), worktree_abs_path).ok()
    }

    #[cfg(target_os = "windows")]
    async fn which(&self, command: &OsStr) -> Option<PathBuf> {
        // On Windows, `PATH` is handled differently from Unix. Windows generally expects users to modify the `PATH` themselves,
        // and every program loads it directly from the system at startup.
        // There's also no concept of a default shell on Windows, and you can't really retrieve one, so trying to get shell environment variables
        // from a specific directory doesn’t make sense on Windows.
        which::which(command).ok()
    }

    async fn shell_env(&self) -> HashMap<String, String> {
        let task = self.load_shell_env_task.clone();
        task.await.unwrap_or_default()
    }

    fn toolchain_store(&self) -> Arc<dyn LanguageToolchainStore> {
        self.toolchain_store.clone()
    }

    async fn read_text_file(&self, path: &RelPath) -> Result<String> {
        let entry = self
            .worktree
            .entry_for_path(path)
            .with_context(|| format!("no worktree entry for path {path:?}"))?;
        let abs_path = self.worktree.absolutize(&entry.path);

        self.fs.load(&abs_path).await
    }

    fn is_headless(&self) -> bool {
        self.is_headless
    }
}
