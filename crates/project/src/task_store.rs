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
use task::{TaskContext, TaskVariables, VariableName};
use text::{BufferId, OffsetRangeExt};
use util::ResultExt;

use crate::{
    BasicContextProvider, Inventory, ProjectEnvironment, buffer_store::BufferStore,
    git_store::GitStore, worktree_store::WorktreeStore,
};

// platform-dependent warning
pub enum TaskStore {
    Functional(StoreState),
    Noop,
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
            use crate::source_coverage::SourceCoverageStatus;
            use std::sync::atomic::{AtomicBool, Ordering};
            anyhow::ensure!(
                self.is_local(),
                "Coverage Task capture requires authoritative local Terminal completion; remote host capture is unavailable"
            );
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
            crate::rust_coverage_provider::validate_capture_task_environment(&launch.env)?;
            let provider = self.rust_coverage_provider_store().clone();
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
                provider.subscribe_capture_buffer_changes(self.buffer_store(), cx)
            }));
            if let Some(trust) = crate::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
                let worktree_store = self.worktree_store().downgrade();
                subscriptions.push(cx.subscribe(&trust, {
                    let provider = provider.clone();
                    let token = token.clone();
                    let cancellation = cancellation.clone();
                    move |_, _, event, cx| {
                        let (changed_store, status) = match event {
                            crate::trusted_worktrees::TrustedWorktreesEvent::Restricted(
                                store,
                                _,
                            ) => (store, SourceCoverageStatus::Restricted),
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
                        }
                    }
                }));
            }
            Ok(Some(ProjectTaskArtifactCapture {
                inner: ProjectTaskArtifactCaptureKind::RustCoverage(
                    crate::rust_coverage_provider::RustCoverageTaskCapture {
                        provider,
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
                    },
                ),
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
                capture.path = Some(path.clone());
                capture.working_directory = Some(working_directory);
                capture.token.set(
                    capture
                        .provider
                        .read(cx)
                        .seal_source_revision(capture.token.get(), cx)?,
                );
                capture.admitted = true;
                capture.source_baseline =
                    Some(capture.provider.read(cx).begin_capture_source_baseline(
                        capture,
                        self.buffer_store(),
                        cx,
                    )?);
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
