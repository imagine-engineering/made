#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
use anyhow::Context as _;
use anyhow::Result;
use collections::HashMap;
use gpui::{App, AppContext as _, Context, Entity, Task, WeakEntity};

use async_channel::bounded;
use futures::{FutureExt, future::Shared};
use itertools::Itertools as _;
use language::LanguageName;
use remote::{Interactive, RemoteClient};
use settings::{Settings, SettingsLocation};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Arc,
};
use task::{
    Shell, ShellBuilder, ShellKind, SpawnInTerminal, StructuredTaskHandle, StructuredTerminalId,
};
use terminal::{
    TaskState, TaskStatus, Terminal, TerminalBuilder, insert_zed_terminal_env,
    terminal_settings::TerminalSettings,
};
use util::{
    command::new_std_command, get_default_system_shell, get_system_shell, maybe, rel_path::RelPath,
};

use crate::{Project, ProjectPath};

pub struct Terminals {
    pub(crate) local_handles: Vec<WeakEntity<terminal::Terminal>>,
}

pub struct PreparedNativeTerminalLaunch {
    _private: (),
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    project_id: gpui::EntityId,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    admission: crate::profile_results_store::NativeTerminalTaskAdmission,
}

impl PreparedNativeTerminalLaunch {
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    pub(crate) fn from_admission(
        project_id: gpui::EntityId,
        admission: crate::profile_results_store::NativeTerminalTaskAdmission,
    ) -> Self {
        Self {
            _private: (),
            project_id,
            admission,
        }
    }

    pub fn validate_launch(&self, launch: &SpawnInTerminal) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            self.admission.validate_launch(launch)
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let _launch = launch;
            anyhow::bail!(
                "Checked native terminal launch requires enabled local macOS Cargo support"
            )
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
pub(crate) struct NativeTerminalSpawnData {
    pub(crate) program: String,
    pub(crate) arguments: Vec<String>,
    pub(crate) working_directory: PathBuf,
    pub(crate) environment: terminal::NativeTerminalEnvironment,
}

pub struct NativeTaskEnvironmentRecipe {
    project_id: gpui::EntityId,
    working_directory: PathBuf,
    logical_working_directory: PathBuf,
    logical_environment: HashMap<String, String>,
    environment: HashMap<String, String>,
    worktree_id: settings::WorktreeId,
    worktree_root: Arc<Path>,
    shell: String,
    terminal_environment: HashMap<String, String>,
    detect_venv: settings::VenvSettings,
}

impl NativeTaskEnvironmentRecipe {
    pub fn working_directory(&self) -> &Path {
        &self.working_directory
    }

    pub fn environment(&self) -> &HashMap<String, String> {
        &self.environment
    }

    pub fn logical_environment(&self) -> &HashMap<String, String> {
        &self.logical_environment
    }
}

impl Project {
    pub fn active_entry_directory(&self, cx: &App) -> Option<PathBuf> {
        let entry_id = self.active_entry()?;
        let worktree = self.worktree_for_entry(entry_id, cx)?;
        let worktree = worktree.read(cx);
        let entry = worktree.entry_for_id(entry_id)?;

        let absolute_path = worktree.absolutize(entry.path.as_ref());
        if entry.is_dir() {
            Some(absolute_path)
        } else {
            absolute_path.parent().map(|p| p.to_path_buf())
        }
    }

    pub fn active_project_directory(&self, cx: &App) -> Option<Arc<Path>> {
        self.active_entry()
            .and_then(|entry_id| self.worktree_for_entry(entry_id, cx))
            .into_iter()
            .chain(self.worktrees(cx))
            .find_map(|tree| tree.read(cx).root_dir())
    }

    pub fn first_project_directory(&self, cx: &App) -> Option<PathBuf> {
        let worktree = self.worktrees(cx).next()?;
        let worktree = worktree.read(cx);
        if worktree.root_entry()?.is_dir() {
            Some(worktree.abs_path().to_path_buf())
        } else {
            None
        }
    }

    /// Artifact review can freeze this directory before terminal preparation instead of
    /// reselecting it if the project's active worktree changes.
    pub fn terminal_task_working_directory(
        &self,
        working_directory: Option<&Path>,
        cx: &App,
    ) -> Option<Arc<Path>> {
        working_directory
            .map(|directory| {
                explicit_terminal_task_working_directory(directory, self.remote_client.is_some())
            })
            .or_else(|| self.active_project_directory(cx))
    }

    pub fn prepare_native_task_environment(
        &mut self,
        launch: &SpawnInTerminal,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeTaskEnvironmentRecipe>>> {
        let directory = launch.cwd.as_deref().ok_or_else(|| {
            anyhow::anyhow!("Native Task environment requires an explicit host cwd")
        })?;
        self.prepare_native_task_environment_at(directory, &launch.env, cx)
    }

    pub fn revalidate_native_task_environment(
        &mut self,
        recipe: &NativeTaskEnvironmentRecipe,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            recipe.project_id == cx.entity_id(),
            "Native Task environment belongs to a different project"
        );
        let current = self.prepare_native_task_environment_at(
            &recipe.logical_working_directory,
            &recipe.logical_environment,
            cx,
        )?;
        let expected_directory = recipe.working_directory.clone();
        let expected_environment = recipe.environment.clone();
        let expected_worktree = recipe.worktree_id;
        let expected_root = recipe.worktree_root.clone();
        let expected_shell = recipe.shell.clone();
        let expected_terminal_environment = recipe.terminal_environment.clone();
        let expected_detect_venv = recipe.detect_venv.clone();
        Ok(cx.spawn(async move |_, _| {
            let current = current.await?;
            anyhow::ensure!(
                current.working_directory == expected_directory
                    && current.environment == expected_environment
                    && current.worktree_id == expected_worktree
                    && current.worktree_root == expected_root
                    && current.shell == expected_shell
                    && current.terminal_environment == expected_terminal_environment
                    && current.detect_venv == expected_detect_venv,
                "Native Task environment or activation context changed after preparation"
            );
            Ok(())
        }))
    }

    fn prepare_native_task_environment_at(
        &mut self,
        directory: &Path,
        logical_environment: &HashMap<String, String>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeTaskEnvironmentRecipe>>> {
        anyhow::ensure!(
            cfg!(target_os = "macos")
                && self.is_local()
                && !self.is_shared()
                && !self.isolated_recipe_test,
            "Checked native Task environments require a local macOS project"
        );
        anyhow::ensure!(
            directory.is_absolute()
                && directory.to_str().is_some_and(|directory| {
                    directory.len() <= 4096 && !directory.chars().any(char::is_control)
                }),
            "Native Task environment requires a bounded absolute UTF8 cwd"
        );
        terminal::validate_native_terminal_environment(logical_environment)?;
        let (worktree, relative_path) = self.find_worktree(directory, cx).ok_or_else(|| {
            anyhow::anyhow!("Native Task environment cwd is outside the current project")
        })?;
        anyhow::ensure!(
            worktree.read(cx).as_local().is_some(),
            "Native Task environment requires a local worktree"
        );
        let worktree_id = worktree.read(cx).id();
        let worktree_root = worktree.read(cx).abs_path();
        let settings_path = Some(directory.to_path_buf());
        let settings = self.terminal_settings(&settings_path, cx);
        terminal::validate_native_terminal_environment(&settings.env)?;
        if let Some(activation) = settings.detect_venv.as_option() {
            anyhow::ensure!(
                activation.venv_name.len() <= 4096
                    && activation.directories.len() <= 32
                    && activation.directories.iter().all(|directory| {
                        directory.to_str().is_some_and(|path| path.len() <= 4096)
                    }),
                "Native Task activation settings exceed their budget"
            );
        }
        let terminal_environment = settings.env.clone();
        let detect_venv = settings.detect_venv.clone();
        let shell = get_system_shell();
        anyhow::ensure!(
            shell.len() <= 4096 && !shell.chars().any(char::is_control),
            "Native Task shell exceeds its supported bounds"
        );
        let selected_toolchain = detect_venv.as_option().map(|_| {
            self.active_toolchain(
                ProjectPath {
                    worktree_id,
                    path: relative_path.clone(),
                },
                LanguageName::new_static("Python"),
                cx,
            )
        });
        let logical_working_directory = directory.to_path_buf();
        let logical_environment = logical_environment.clone();
        let filesystem = self.fs.clone();
        let directory: Arc<Path> = Arc::from(directory);
        let directory_environment =
            self.resolve_directory_environment(&shell, Some(directory.clone()), None, cx);
        let executor = cx.background_executor().clone();
        Ok(cx.spawn(async move |project, cx| {
            let preparation = async {
                let canonical = filesystem.canonicalize(&directory).await?;
                anyhow::ensure!(
                    canonical == directory.as_ref(),
                    "Native Task environment cwd must be a canonical host path"
                );
                let metadata = filesystem.metadata(&canonical).await?.ok_or_else(|| {
                    anyhow::anyhow!("Native Task environment cwd disappeared")
                })?;
                anyhow::ensure!(metadata.is_dir && !metadata.is_symlink, "Native Task environment cwd is not ordinary");
                if let Some(toolchain) = selected_toolchain {
                    anyhow::ensure!(
                        toolchain.await.is_none(),
                        "Checked native Tasks cannot yet materialize selected Python activation; disable terminal activation for this Task"
                    );
                }
                let directory_environment = directory_environment.await.ok_or_else(|| {
                    anyhow::anyhow!("Native Task directory environment is unavailable")
                })?;
                let environment = merge_native_task_environment(
                    directory_environment,
                    &terminal_environment,
                    &logical_environment,
                )?;
                if detect_venv.as_option().is_some() {
                    let current_toolchain = project.update(cx, |project, cx| {
                        project.active_toolchain(
                            ProjectPath { worktree_id, path: relative_path },
                            LanguageName::new_static("Python"),
                            cx,
                        )
                    })?;
                    anyhow::ensure!(
                        current_toolchain.await.is_none(),
                        "Native Task activation selection changed during preparation"
                    );
                }
                project.update(cx, |project, cx| {
                    anyhow::ensure!(project.is_local() && !project.is_shared() && !project.isolated_recipe_test, "Native Task environment route changed during preparation");
                    let (current_worktree, _) = project.find_worktree(&canonical, cx).ok_or_else(|| {
                        anyhow::anyhow!("Native Task environment worktree disappeared")
                    })?;
                    anyhow::ensure!(
                        current_worktree.read(cx).id() == worktree_id
                            && current_worktree.read(cx).abs_path() == worktree_root,
                        "Native Task environment worktree changed during preparation"
                    );
                    let current_settings_path = Some(canonical.clone());
                    let current_settings = project.terminal_settings(&current_settings_path, cx);
                    anyhow::ensure!(
                        current_settings.env == terminal_environment
                            && current_settings.detect_venv == detect_venv
                            && get_system_shell() == shell,
                        "Native Task environment settings changed during preparation"
                    );
                    let environment = terminal::complete_native_terminal_environment(&environment, cx.entity_id().as_u64(), cx)?;
                    Ok(NativeTaskEnvironmentRecipe {
                        project_id: cx.entity_id(),
                        working_directory: canonical,
                        logical_working_directory,
                        logical_environment,
                        environment,
                        worktree_id,
                        worktree_root,
                        shell,
                        terminal_environment,
                        detect_venv,
                    })
                })?
            };
            match futures::future::select(
                preparation.boxed_local(),
                executor.timer(std::time::Duration::from_secs(30)).boxed(),
            ).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(anyhow::anyhow!("Native Task environment preparation timed out")),
            }
        }))
    }

    pub fn create_terminal_task(
        &mut self,
        spawn_task: SpawnInTerminal,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        self.create_terminal_task_with_structured_handle(spawn_task, None, cx)
    }

    pub fn create_terminal_task_with_native_launch(
        &mut self,
        spawn_task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        native: Option<PreparedNativeTerminalLaunch>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        self.create_terminal_task_with_pending_launch(spawn_task, structured_task, native, None, cx)
    }

    pub fn create_terminal_task_with_pending_launch(
        &mut self,
        spawn_task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        native: Option<PreparedNativeTerminalLaunch>,
        pending_launch: Option<crate::task_store::PendingExperimentalTaskLaunch>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        let pending_launch = match pending_launch {
            Some(pending_launch) => {
                if let Err(error) = pending_launch.check(&spawn_task, cx) {
                    return Task::ready(Err(error));
                }
                Some(pending_launch)
            }
            None => match crate::task_store::PendingExperimentalTaskLaunch::new(&spawn_task, cx) {
                Ok(pending_launch) => pending_launch,
                Err(error) => return Task::ready(Err(error)),
            },
        };
        if let Err(error) =
            crate::task_store::validate_experimental_task_requirements(&spawn_task, cx)
        {
            return Task::ready(Err(error));
        }
        if self.remote_client.is_some() && !spawn_task.required_experimental_features.is_empty() {
            return Task::ready(Err(anyhow::anyhow!(
                "The remote Task host does not negotiate experimental requirement metadata"
            )));
        }

        let Some(native) = native else {
            return self.create_terminal_task_with_pending_structured_launch(
                spawn_task,
                structured_task,
                pending_launch,
                cx,
            );
        };
        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            let initial = || -> Result<()> {
                anyhow::ensure!(
                    self.is_local() && !self.is_shared() && native.project_id == cx.entity_id(),
                    "Native terminal launch belongs to a different host Project"
                );
                anyhow::ensure!(
                    !structured_task
                        .as_ref()
                        .is_some_and(|handle| handle.state().is_terminal()),
                    "The structured Task was cancelled before exact environment renewal"
                );
                native.validate_launch(&spawn_task)
            };
            if let Err(error) = initial() {
                return Task::ready(Err(error));
            }
            cx.spawn(async move |project, cx| {
                let original_project = project
                    .upgrade()
                    .context("Native terminal Project disappeared")?;
                let renewal = native.admission.renew_for_terminal(
                    original_project.clone(),
                    spawn_task.clone(),
                    cx,
                )?;
                let data = match futures::future::select(
                    renewal.boxed(),
                    cx.background_executor()
                        .timer(std::time::Duration::from_secs(30))
                        .boxed(),
                )
                .await
                {
                    futures::future::Either::Left((result, _)) => result?,
                    futures::future::Either::Right(_) => {
                        anyhow::bail!("Native terminal owner renewal timed out")
                    }
                };
                let (completion_tx, completion_rx) = bounded(1);
                let creation_handle = structured_task.clone();
                let builder = original_project
                    .update(cx, |project, cx| {
                        anyhow::ensure!(
                            project.is_local()
                                && !project.is_shared()
                                && native.project_id == cx.entity_id(),
                            "Native terminal Project route changed before direct launch"
                        );
                        anyhow::ensure!(
                            !creation_handle
                                .as_ref()
                                .is_some_and(|handle| handle.state().is_terminal()),
                            "The structured Task was cancelled before direct launch"
                        );
                        anyhow::ensure!(
                            spawn_task.command.as_deref() == Some(data.program.as_str())
                                && spawn_task.args == data.arguments
                                && spawn_task.cwd.as_deref()
                                    == Some(data.working_directory.as_path()),
                            "Native terminal child recipe changed after owner renewal"
                        );
                        crate::task_store::validate_experimental_task_requirements(
                            &spawn_task,
                            cx,
                        )?;
                        if let Some(pending_launch) = pending_launch.as_ref() {
                            pending_launch.check(&spawn_task, cx)?;
                        }
                        let mut settings_location = None;
                        if let Some((worktree, _)) =
                            project.find_worktree(&data.working_directory, cx)
                        {
                            settings_location = Some(SettingsLocation {
                                worktree_id: worktree.read(cx).id(),
                                path: RelPath::empty(),
                            });
                        }
                        let settings = TerminalSettings::get(settings_location, cx).clone();
                        let task = TaskState {
                            spawned_task: spawn_task,
                            status: TaskStatus::Running,
                            completion_rx,
                            structured_task: creation_handle,
                        };
                        drop(pending_launch);
                        Ok::<_, anyhow::Error>(TerminalBuilder::new_with_exact_native_environment(
                            data.working_directory,
                            task,
                            data.program,
                            data.arguments,
                            data.environment,
                            settings.cursor_shape,
                            settings.alternate_scroll,
                            settings.max_scroll_history_lines,
                            settings.path_hyperlink_regexes,
                            settings.path_hyperlink_timeout_ms,
                            cx.entity_id().as_u64(),
                            Some(completion_tx),
                            cx,
                            project.path_style(cx),
                        ))
                    })?
                    .await?;
                Ok(original_project.update(cx, |project, cx| {
                    project.register_task_terminal(builder, structured_task, cx)
                }))
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_spawn_task, _structured_task, _native, _pending_launch, _context) =
                (spawn_task, structured_task, native, pending_launch, cx);
            Task::ready(Err(anyhow::anyhow!(
                "Checked native terminal launch requires enabled local macOS Cargo support"
            )))
        }
    }

    pub fn create_terminal_task_with_structured_handle(
        &mut self,
        spawn_task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        self.create_terminal_task_with_pending_launch(spawn_task, structured_task, None, None, cx)
    }

    fn create_terminal_task_with_pending_structured_launch(
        &mut self,
        spawn_task: SpawnInTerminal,
        structured_task: Option<StructuredTaskHandle>,
        pending_launch: Option<crate::task_store::PendingExperimentalTaskLaunch>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        if let Err(error) =
            crate::task_store::validate_experimental_task_requirements(&spawn_task, cx)
        {
            return Task::ready(Err(error));
        }
        if self.remote_client.is_some() && !spawn_task.required_experimental_features.is_empty() {
            return Task::ready(Err(anyhow::anyhow!(
                "The remote Task host does not negotiate experimental requirement metadata"
            )));
        }

        if structured_task
            .as_ref()
            .is_some_and(|handle| handle.state().is_terminal())
        {
            return Task::ready(Err(anyhow::anyhow!(
                "The structured Task was cancelled before environment or terminal preparation"
            )));
        }
        let is_via_remote = self.remote_client.is_some();

        if spawn_task.windows_direct_program && (!cfg!(windows) || is_via_remote) {
            return Task::ready(Err(anyhow::anyhow!(
                "Literal Windows executable Tasks require a local Windows host"
            )));
        }

        let path = self.terminal_task_working_directory(spawn_task.cwd.as_deref(), cx);
        let path = if spawn_task.windows_direct_program && path.is_none() {
            match std::env::current_dir() {
                Ok(directory) => Some(Arc::from(directory)),
                Err(error) => return Task::ready(Err(error.into())),
            }
        } else {
            path
        };

        let mut settings_location = None;
        if let Some(path) = path.as_ref()
            && let Some((worktree, _)) = self.find_worktree(path, cx)
        {
            settings_location = Some(SettingsLocation {
                worktree_id: worktree.read(cx).id(),
                path: RelPath::empty(),
            });
        }
        let mut settings = TerminalSettings::get(settings_location, cx).clone();
        if self.isolated_recipe_test {
            settings.env.clear();
        }
        let detect_venv = !self.isolated_recipe_test && settings.detect_venv.as_option().is_some();

        let (completion_tx, completion_rx) = bounded(1);

        let local_path = if is_via_remote { None } else { path.clone() };
        let task_state = Some(TaskState {
            spawned_task: spawn_task.clone(),
            status: TaskStatus::Running,
            completion_rx,
            structured_task: structured_task.clone(),
        });
        let remote_client = self.remote_client.clone();
        let shell = match &remote_client {
            Some(remote_client) => remote_client
                .read(cx)
                .shell()
                .unwrap_or_else(get_default_system_shell),
            None => get_system_shell(),
        };
        let path_style = self.path_style(cx);
        let shell_kind = ShellKind::new(&shell, path_style.is_windows());
        let activation_shell_kind = if spawn_task.windows_direct_program {
            spawn_task.shell.shell_kind(true)
        } else {
            shell_kind
        };

        // Prepare a task for resolving the environment
        let env_task =
            self.resolve_directory_environment(&shell, path.clone(), remote_client.clone(), cx);

        // Scope the toolchain lookup to the worktree the terminal is being
        // spawned in. Previously this iterated the active editor's worktree
        // and then every visible worktree, so a Python toolchain persisted
        // for worktree A would leak into a terminal opened in worktree B and
        // inject (e.g.) `conda activate base` into a shell that has no
        // business with conda.
        let project_path_contexts: Vec<ProjectPath> = path
            .as_ref()
            .and_then(|p| self.find_worktree(p, cx))
            .map(|(worktree, relative_path)| ProjectPath {
                worktree_id: worktree.read(cx).id(),
                path: relative_path,
            })
            .into_iter()
            .collect();
        let toolchains = project_path_contexts
            .into_iter()
            .filter(|_| detect_venv)
            .map(|p| self.active_toolchain(p, LanguageName::new_static("Python"), cx))
            .collect::<Vec<_>>();
        let lang_registry = self.languages.clone();
        cx.spawn(async move |project, cx| {
            let mut env = env_task.await.unwrap_or_default();
            #[cfg(windows)]
            if spawn_task.windows_direct_program {
                env = util::shell_env::merge_task_environment_layers(&[
                    &env,
                    &settings.env,
                    &spawn_task.env,
                ])?;
            } else {
                env.extend(settings.env);
            }
            #[cfg(not(windows))]
            env.extend(settings.env);

            let activation_script = maybe!(async {
                for toolchain in toolchains {
                    let Some(toolchain) = toolchain.await else {
                        continue;
                    };
                    let language = lang_registry
                        .language_for_name(&toolchain.language_name.0)
                        .await
                        .ok();
                    let lister = language?.toolchain_lister()?;
                    let future = cx.update(|cx| {
                        lister.activation_script(&toolchain, activation_shell_kind, cx)
                    });
                    return Some(future.await);
                }
                None
            })
            .await
            .unwrap_or_default();

            if !spawn_task.windows_direct_program {
                env.extend(
                    spawn_task
                        .env
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone())),
                );
            }
            #[cfg(windows)]
            if spawn_task.windows_direct_program {
                cx.update(|cx| terminal::apply_task_runtime_environment(&mut env, cx))?;
                let capture_shell = spawn_task.shell.clone();
                let capture_activation = activation_script.clone();
                let capture_directory = path.clone().ok_or_else(|| {
                    anyhow::anyhow!("Literal Windows Tasks require a working directory")
                })?;
                env = cx
                    .background_spawn(async move {
                        util::shell_env::capture_task_environment(
                            &capture_shell,
                            &capture_activation,
                            &capture_directory,
                            &env,
                        )
                        .await
                    })
                    .await?;
            }

            let preparation_handle = structured_task.clone();
            let builder = project
                .update(cx, move |_, cx| {
                    crate::task_store::validate_experimental_task_requirements(&spawn_task, cx)?;
                    if let Some(pending_launch) = pending_launch.as_ref() {
                        pending_launch.check(&spawn_task, cx)?;
                    }
                    anyhow::ensure!(
                        !preparation_handle
                            .as_ref()
                            .is_some_and(|handle| handle.state().is_terminal()),
                        "The structured Task was cancelled before terminal creation"
                    );
                    let format_to_run = |spawn_task: &SpawnInTerminal| {
                        format_task_for_activation(
                            spawn_task,
                            shell_kind,
                            &shell,
                            path_style.is_windows(),
                        )
                    };

                    let (shell, env) = {
                        let requires_shell_activation =
                            !activation_script.is_empty() && !spawn_task.windows_direct_program;
                        let to_run = requires_shell_activation.then(|| format_to_run(&spawn_task));
                        match remote_client {
                            Some(remote_client) => match activation_script.clone() {
                                activation_script if requires_shell_activation => {
                                    let separator = shell_kind.sequential_commands_separator();
                                    let activation_script =
                                        activation_script.join(&format!("{separator} "));
                                    let to_run = to_run.expect("activation command was formatted");

                                    let arg = format!("{activation_script}{separator} {to_run}");
                                    let args = shell_kind.args_for_shell(true, arg);
                                    let shell = remote_client
                                        .read(cx)
                                        .shell()
                                        .unwrap_or_else(get_default_system_shell);

                                    create_remote_shell(
                                        Some((&shell, &args)),
                                        env,
                                        path,
                                        remote_client,
                                        cx,
                                    )?
                                }
                                _ => create_remote_shell(
                                    spawn_task
                                        .command
                                        .as_ref()
                                        .map(|command| (command, &spawn_task.args)),
                                    env,
                                    path,
                                    remote_client,
                                    cx,
                                )?,
                            },
                            None => match activation_script.clone() {
                                activation_script if requires_shell_activation => {
                                    let separator = shell_kind.sequential_commands_separator();
                                    let activation_script =
                                        activation_script.join(&format!("{separator} "));
                                    let to_run = to_run.expect("activation command was formatted");

                                    let arg = format!("{activation_script}{separator} {to_run}");
                                    let args = shell_kind.args_for_shell(true, arg);

                                    (
                                        Shell::WithArguments {
                                            program: shell,
                                            args,
                                            title_override: None,
                                        },
                                        env,
                                    )
                                }
                                _ => (
                                    if let Some(program) = spawn_task.command {
                                        Shell::WithArguments {
                                            program,
                                            args: spawn_task.args,
                                            title_override: None,
                                        }
                                    } else {
                                        Shell::System
                                    },
                                    env,
                                ),
                            },
                        }
                    };
                    drop(pending_launch);
                    anyhow::Ok(TerminalBuilder::new(
                        local_path.map(|path| path.to_path_buf()),
                        task_state,
                        shell,
                        env,
                        settings.cursor_shape,
                        settings.alternate_scroll,
                        settings.max_scroll_history_lines,
                        settings.path_hyperlink_regexes,
                        settings.path_hyperlink_timeout_ms,
                        is_via_remote,
                        cx.entity_id().as_u64(),
                        Some(completion_tx),
                        cx,
                        activation_script,
                        path_style,
                    ))
                })??
                .await?;
            project.update(cx, move |this, cx| {
                this.register_task_terminal(builder, structured_task, cx)
            })
        })
    }

    fn register_task_terminal(
        &mut self,
        builder: TerminalBuilder,
        structured_task: Option<StructuredTaskHandle>,
        cx: &mut Context<Self>,
    ) -> Entity<Terminal> {
        let terminal_handle = cx.new(|cx| builder.subscribe(cx));

        if let Some(structured_task) = structured_task {
            let weak_terminal = terminal_handle.downgrade();
            structured_task.bind_terminal(
                StructuredTerminalId(terminal_handle.entity_id().as_u64()),
                move |cx| {
                    weak_terminal
                        .update(cx, |terminal, _| terminal.kill_active_task())
                        .is_ok()
                },
                cx,
            );
        }

        self.terminals
            .local_handles
            .push(terminal_handle.downgrade());

        let id = terminal_handle.entity_id();
        cx.observe_release(&terminal_handle, move |project, _terminal, cx| {
            let handles = &mut project.terminals.local_handles;

            if let Some(index) = handles
                .iter()
                .position(|terminal| terminal.entity_id() == id)
            {
                handles.remove(index);
                cx.notify();
            }
        })
        .detach();

        terminal_handle
    }

    pub fn create_terminal_shell(
        &mut self,
        cwd: Option<PathBuf>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        self.create_terminal_shell_internal(cwd, false, cx)
    }

    /// Creates a local terminal even if the project is remote.
    /// In remote projects: opens in Zed's launch directory (bypasses SSH).
    /// In local projects: opens in the project directory (same as regular terminals).
    pub fn create_local_terminal(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        let working_directory = if self.remote_client.is_some() {
            // Remote project: don't use remote paths, let shell use Zed's cwd
            None
        } else {
            // Local project: use project directory like normal terminals
            self.active_project_directory(cx).map(|p| p.to_path_buf())
        };
        self.create_terminal_shell_internal(working_directory, true, cx)
    }

    /// Internal method for creating terminal shells.
    /// If force_local is true, creates a local terminal even if the project has a remote client.
    /// This allows "breaking out" to a local shell in remote projects.
    fn create_terminal_shell_internal(
        &mut self,
        cwd: Option<PathBuf>,
        force_local: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        if self.isolated_recipe_test && force_local {
            return Task::ready(Err(anyhow::anyhow!(
                "Local terminals are unavailable in an isolated recipe test"
            )));
        }
        let path = cwd.map(|p| Arc::from(&*p));
        let is_via_remote = !force_local && self.remote_client.is_some();

        let mut settings_location = None;
        if let Some(path) = path.as_ref()
            && let Some((worktree, _)) = self.find_worktree(path, cx)
        {
            settings_location = Some(SettingsLocation {
                worktree_id: worktree.read(cx).id(),
                path: RelPath::empty(),
            });
        }
        let mut settings = TerminalSettings::get(settings_location, cx).clone();
        if self.isolated_recipe_test {
            settings.env.clear();
        }
        let detect_venv = !self.isolated_recipe_test && settings.detect_venv.as_option().is_some();
        let local_path = if is_via_remote { None } else { path.clone() };

        // See create_terminal_task: scope the toolchain lookup to the
        // worktree the terminal is opened in, not the active editor's
        // worktree or other visible worktrees.
        let project_path_contexts: Vec<ProjectPath> = path
            .as_ref()
            .and_then(|p| self.find_worktree(p, cx))
            .map(|(worktree, relative_path)| ProjectPath {
                worktree_id: worktree.read(cx).id(),
                path: relative_path,
            })
            .into_iter()
            .collect();
        let toolchains = project_path_contexts
            .into_iter()
            .filter(|_| detect_venv)
            .map(|p| self.active_toolchain(p, LanguageName::new_static("Python"), cx))
            .collect::<Vec<_>>();
        let remote_client = if force_local {
            None
        } else {
            self.remote_client.clone()
        };
        let shell = match &remote_client {
            Some(remote_client) => remote_client
                .read(cx)
                .shell()
                .unwrap_or_else(get_default_system_shell),
            None => settings.shell.program(),
        };
        let env_shell = match &remote_client {
            Some(_) => shell.clone(),
            None => get_system_shell(),
        };

        let path_style = self.path_style(cx);

        // Prepare a task for resolving the environment
        let env_task =
            self.resolve_directory_environment(&env_shell, path.clone(), remote_client.clone(), cx);

        let lang_registry = self.languages.clone();
        cx.spawn(async move |project, cx| {
            let shell_kind = ShellKind::new(&shell, path_style.is_windows());
            let mut env = env_task.await.unwrap_or_default();
            env.extend(settings.env);

            let activation_script = maybe!(async {
                for toolchain in toolchains {
                    let Some(toolchain) = toolchain.await else {
                        continue;
                    };
                    let language = lang_registry
                        .language_for_name(&toolchain.language_name.0)
                        .await
                        .ok();
                    let lister = language?.toolchain_lister()?;
                    let future =
                        cx.update(|cx| lister.activation_script(&toolchain, shell_kind, cx));
                    return Some(future.await);
                }
                None
            })
            .await
            .unwrap_or_default();

            let builder = project
                .update(cx, move |_, cx| {
                    let (shell, env) = {
                        match remote_client {
                            Some(remote_client) => {
                                create_remote_shell(None, env, path, remote_client, cx)?
                            }
                            None => (settings.shell, env),
                        }
                    };
                    anyhow::Ok(TerminalBuilder::new(
                        local_path.map(|path| path.to_path_buf()),
                        None,
                        shell,
                        env,
                        settings.cursor_shape,
                        settings.alternate_scroll,
                        settings.max_scroll_history_lines,
                        settings.path_hyperlink_regexes,
                        settings.path_hyperlink_timeout_ms,
                        is_via_remote,
                        cx.entity_id().as_u64(),
                        None,
                        cx,
                        activation_script,
                        path_style,
                    ))
                })??
                .await?;
            project.update(cx, move |this, cx| {
                let terminal_handle = cx.new(|cx| builder.subscribe(cx));

                this.terminals
                    .local_handles
                    .push(terminal_handle.downgrade());

                let id = terminal_handle.entity_id();
                cx.observe_release(&terminal_handle, move |project, _terminal, cx| {
                    let handles = &mut project.terminals.local_handles;

                    if let Some(index) = handles
                        .iter()
                        .position(|terminal| terminal.entity_id() == id)
                    {
                        handles.remove(index);
                        cx.notify();
                    }
                })
                .detach();

                terminal_handle
            })
        })
    }

    pub fn clone_terminal(
        &mut self,
        terminal: &Entity<Terminal>,
        cx: &mut Context<'_, Project>,
        cwd: Option<PathBuf>,
    ) -> Task<Result<Entity<Terminal>>> {
        // We cannot clone the task's terminal, as it will effectively re-spawn the task, which might not be desirable.
        // For now, create a new shell instead.
        if terminal.read(cx).task().is_some() {
            return self.create_terminal_shell(cwd, cx);
        }
        let local_path = if self.is_via_remote_server() {
            None
        } else {
            cwd
        };

        let builder = terminal.read(cx).clone_builder(cx, local_path);
        cx.spawn(async |project, cx| {
            let terminal = builder.await?;
            project.update(cx, |project, cx| {
                let terminal_handle = cx.new(|cx| terminal.subscribe(cx));

                project
                    .terminals
                    .local_handles
                    .push(terminal_handle.downgrade());

                let id = terminal_handle.entity_id();
                cx.observe_release(&terminal_handle, move |project, _terminal, cx| {
                    let handles = &mut project.terminals.local_handles;

                    if let Some(index) = handles
                        .iter()
                        .position(|terminal| terminal.entity_id() == id)
                    {
                        handles.remove(index);
                        cx.notify();
                    }
                })
                .detach();

                terminal_handle
            })
        })
    }

    pub fn terminal_settings<'a>(
        &'a self,
        path: &'a Option<PathBuf>,
        cx: &'a App,
    ) -> &'a TerminalSettings {
        let mut settings_location = None;
        if let Some(path) = path.as_ref()
            && let Some((worktree, _)) = self.find_worktree(path, cx)
        {
            settings_location = Some(SettingsLocation {
                worktree_id: worktree.read(cx).id(),
                path: RelPath::empty(),
            });
        }
        TerminalSettings::get(settings_location, cx)
    }

    pub fn exec_in_shell(
        &self,
        command: String,
        cx: &mut Context<Self>,
    ) -> Task<Result<smol::process::Command>> {
        let command =
            self.exec_in_shell_at_directory(command, self.first_project_directory(cx), false, cx);
        cx.background_spawn(async move {
            let mut command = command.await?;
            util::set_pre_exec_to_start_new_session(&mut command);
            Ok(smol::process::Command::from(command))
        })
    }

    pub fn exec_read_only_probe_for_path(
        &self,
        executable: &Path,
        arguments: &[&str],
        directory: &ProjectPath,
        cx: &App,
    ) -> Result<std::process::Command> {
        let observation = self.doctor_observation(directory.worktree_id, cx);
        anyhow::ensure!(
            observation.supports_reviewed_probe()
                && observation.host == crate::DoctorExecutionHost::Local,
            "Dependency probe requires an available trusted local host"
        );
        anyhow::ensure!(
            executable.is_absolute() && directory.path.is_empty(),
            "Select an absolute vetted executable and a worktree root"
        );
        let worktree = self
            .worktree_for_id(directory.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Selected worktree is unavailable"))?;
        let worktree = worktree.read(cx);
        let root = worktree
            .root_dir()
            .ok_or_else(|| anyhow::anyhow!("Selected worktree is not a directory"))?;
        let mut command = new_std_command(executable);
        command
            .args(arguments)
            .current_dir(root.as_ref())
            .env_clear();
        // Availability checks must not source shell/direnv hooks or inherit project
        // loader/agent/credential variables merely to print an installed version.
        for name in [
            "HOME",
            "USER",
            "TMPDIR",
            "TMP",
            "TEMP",
            "SystemRoot",
            "USERPROFILE",
            "LANG",
            "LC_ALL",
        ] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        Ok(command)
    }

    pub fn exec_user_toolchain_command_for_path(
        &self,
        executable: &Path,
        arguments: &[&str],
        directory: &ProjectPath,
        user_home: &Path,
        manager_environment: &[(String, String)],
        cx: &App,
    ) -> Result<std::process::Command> {
        let mut command =
            self.exec_read_only_probe_for_path(executable, arguments, directory, cx)?;
        anyhow::ensure!(
            user_home.is_absolute(),
            "Toolchain operation requires an absolute user home"
        );
        // The caller consumes a concrete Doctor approval before spawning. Reuse
        // the direct Project command boundary without shell or project env hooks.
        command.current_dir(user_home).env("HOME", user_home);
        for (name, value) in manager_environment {
            command.env(name, value);
        }
        Ok(command)
    }

    pub fn exec_in_shell_for_path(
        &self,
        command: String,
        directory: ProjectPath,
        cx: &mut Context<Self>,
    ) -> Task<Result<std::process::Command>> {
        if !self
            .doctor_observation(directory.worktree_id, cx)
            .supports_reviewed_probe()
        {
            return Task::ready(Err(anyhow::anyhow!(
                "Selected execution scope is unavailable or unauthorized"
            )));
        }
        let Some(worktree) = self.worktree_for_id(directory.worktree_id, cx) else {
            return Task::ready(Err(anyhow::anyhow!("Selected worktree is unavailable")));
        };
        let worktree = worktree.read(cx);
        let Some(root) = worktree.root_dir() else {
            return Task::ready(Err(anyhow::anyhow!("Selected worktree is not a directory")));
        };
        if !directory.path.is_empty() {
            return Task::ready(Err(anyhow::anyhow!(
                "Dependency probes require the selected worktree root"
            )));
        }
        let path = worktree.absolutize(&directory.path);
        if !path.starts_with(root.as_ref()) {
            return Task::ready(Err(anyhow::anyhow!(
                "Selected directory is outside the worktree"
            )));
        }
        let task = self.exec_in_shell_at_directory(command, Some(path), true, cx);
        cx.spawn(async move |project, cx| {
            let command = task.await?;
            anyhow::ensure!(
                project.read_with(cx, |project, cx| {
                    project
                        .doctor_observation(directory.worktree_id, cx)
                        .supports_reviewed_probe()
                })?,
                "Selected execution authority changed while resolving its environment"
            );
            Ok(command)
        })
    }

    fn exec_in_shell_at_directory(
        &self,
        command: String,
        path: Option<PathBuf>,
        selected_directory: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<std::process::Command>> {
        let remote_client = self.remote_client.clone();
        let mut settings = self.terminal_settings(&path, cx).clone();
        if self.isolated_recipe_test {
            settings.env.clear();
        }
        let shell = remote_client
            .as_ref()
            .and_then(|remote_client| remote_client.read(cx).shell())
            .map(Shell::Program)
            .unwrap_or(Shell::System);
        let is_windows = self.path_style(cx).is_windows();
        let builder = ShellBuilder::new(&shell, is_windows).non_interactive();
        let (command, args) = builder.build(Some(command), &Vec::new());

        let env_task = self.resolve_directory_environment(
            &shell.program(),
            path.as_ref().map(|p| Arc::from(&**p)),
            remote_client.clone(),
            cx,
        );

        cx.spawn(async move |project, cx| {
            let mut env = env_task.await.unwrap_or_default();
            env.extend(settings.env);

            project.update(cx, move |_, cx| match remote_client {
                Some(remote_client) => {
                    let command_template = remote_client.read(cx).build_command(
                        Some(command),
                        &args,
                        &env,
                        selected_directory
                            .then(|| {
                                path.as_ref()
                                    .map(|path| path.to_string_lossy().into_owned())
                            })
                            .flatten(),
                        None,
                        Interactive::Yes,
                    )?;
                    let mut command = new_std_command(command_template.program);
                    command.args(command_template.args);
                    command.envs(command_template.env);
                    Ok(command)
                }
                None => {
                    let mut command = new_std_command(command);
                    command.args(args);
                    command.envs(env);
                    if let Some(path) = path {
                        command.current_dir(path);
                    }
                    Ok(command)
                }
            })?
        })
    }

    pub fn local_terminal_handles(&self) -> &Vec<WeakEntity<terminal::Terminal>> {
        &self.terminals.local_handles
    }

    fn resolve_directory_environment(
        &self,
        shell: &str,
        path: Option<Arc<Path>>,
        remote_client: Option<Entity<RemoteClient>>,
        cx: &mut App,
    ) -> Shared<Task<Option<HashMap<String, String>>>> {
        if let Some(path) = &path {
            let shell = Shell::Program(shell.to_string());
            self.environment
                .update(cx, |project_env, cx| match &remote_client {
                    Some(remote_client) => project_env.remote_directory_environment(
                        &shell,
                        path.clone(),
                        remote_client.clone(),
                        cx,
                    ),
                    None => project_env.local_directory_environment(&shell, path.clone(), cx),
                })
        } else {
            Task::ready(None).shared()
        }
    }
}

fn merge_native_task_environment(
    mut directory_environment: HashMap<String, String>,
    terminal_environment: &HashMap<String, String>,
    task_environment: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    terminal::validate_native_terminal_environment(&directory_environment)?;
    terminal::validate_native_terminal_environment(terminal_environment)?;
    terminal::validate_native_terminal_environment(task_environment)?;
    directory_environment.extend(
        terminal_environment
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    directory_environment.extend(
        task_environment
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    terminal::validate_native_terminal_environment(&directory_environment)?;
    Ok(directory_environment)
}

fn explicit_terminal_task_working_directory(directory: &Path, remote: bool) -> Arc<Path> {
    if remote {
        Arc::from(directory)
    } else {
        let directory = directory.to_string_lossy();
        let directory = shellexpand::tilde(&directory);
        Arc::from(Path::new(directory.as_ref()))
    }
}

fn create_remote_shell(
    spawn_command: Option<(&String, &Vec<String>)>,
    mut env: HashMap<String, String>,
    working_directory: Option<Arc<Path>>,
    remote_client: Entity<RemoteClient>,
    cx: &mut App,
) -> Result<(Shell, HashMap<String, String>)> {
    let isolated_recipe_test = matches!(remote_client.read(cx).connection_options(), remote::RemoteConnectionOptions::Docker(options) if options.isolated_recipe_test.is_some());
    if !isolated_recipe_test {
        insert_zed_terminal_env(&mut env, &release_channel::AppVersion::global(cx));
    }

    let (program, args) = match spawn_command {
        Some((program, args)) => (Some(program.clone()), args),
        None => (None, &Vec::new()),
    };

    let command = remote_client.read(cx).build_command(
        program,
        args.as_slice(),
        &env,
        working_directory.map(|path| path.display().to_string()),
        None,
        Interactive::Yes,
    )?;

    log::debug!("Connecting to a remote server: {:?}", command.program);
    let host = remote_client.read(cx).connection_options().display_name();

    Ok((
        Shell::WithArguments {
            program: command.program,
            args: command.args,
            title_override: Some(format!("{} — Terminal", host)),
        },
        command.env,
    ))
}

fn format_task_for_activation(
    spawn_task: &SpawnInTerminal,
    shell_kind: ShellKind,
    shell: &str,
    is_windows: bool,
) -> String {
    if let Some(command) = &spawn_task.command {
        let command = shell_kind.prepend_command_prefix(command);
        let command = shell_kind.try_quote_prefix_aware(&command);
        let args = spawn_task
            .args
            .iter()
            .enumerate()
            .filter_map(|(index, arg)| {
                quote_prepared_task_arg_for_activation(
                    spawn_task, shell_kind, arg, index, is_windows,
                )
            });

        command.into_iter().chain(args).join(" ")
    } else {
        // todo: this breaks for remotes to windows
        format!("exec {shell} -l")
    }
}

fn quote_prepared_task_arg_for_activation<'a>(
    spawn_task: &SpawnInTerminal,
    shell_kind: ShellKind,
    arg: &'a str,
    index: usize,
    is_windows: bool,
) -> Option<Cow<'a, str>> {
    if spawn_task.shell.shell_kind(is_windows) == ShellKind::Cmd
        && index >= 2
        && spawn_task
            .args
            .get(index - 2)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("/S"))
        && spawn_task
            .args
            .get(index - 1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("/C"))
    {
        // The /C argument is already a cmd command string from prepare_task_for_spawn.
        // Quoting it again for venv activation makes cmd see the quotes as literals.
        return quote_cmd_command_arg_for_outer_shell(arg, shell_kind).map(Cow::Owned);
    }

    shell_kind.try_quote(arg)
}

fn quote_cmd_command_arg_for_outer_shell(arg: &str, shell_kind: ShellKind) -> Option<String> {
    match shell_kind {
        ShellKind::PowerShell | ShellKind::Pwsh => Some(format!("'{}'", arg.replace('\'', "''"))),
        ShellKind::Cmd => Some(arg.to_string()),
        ShellKind::Posix
        | ShellKind::Csh
        | ShellKind::Tcsh
        | ShellKind::Fish
        | ShellKind::Nushell
        | ShellKind::Rc
        | ShellKind::Xonsh
        | ShellKind::Elvish => shell_kind.try_quote(arg).map(Cow::into_owned),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_task_environment_layer_precedence_is_complete_and_bounded() {
        fn assert_send<T: Send>() {}
        assert_send::<NativeTaskEnvironmentRecipe>();
        let directory = HashMap::from_iter([
            ("DIRECTORY_ONLY".into(), "directory".into()),
            ("PRECEDENCE".into(), "directory".into()),
        ]);
        let settings = HashMap::from_iter([
            ("SETTINGS_ONLY".into(), "settings".into()),
            ("PRECEDENCE".into(), "settings".into()),
        ]);
        let task = HashMap::from_iter([
            ("TASK_ONLY".into(), "task".into()),
            ("PRECEDENCE".into(), "task".into()),
        ]);
        assert_eq!(
            merge_native_task_environment(directory, &settings, &task).expect("known layers"),
            HashMap::from_iter([
                ("DIRECTORY_ONLY".into(), "directory".into()),
                ("SETTINGS_ONLY".into(), "settings".into()),
                ("TASK_ONLY".into(), "task".into()),
                ("PRECEDENCE".into(), "task".into()),
            ])
        );
        let invalid = HashMap::from_iter([("PRECEDENCE".into(), "x".repeat(4097))]);
        assert!(
            merge_native_task_environment(invalid, &settings, &task).is_err(),
            "later overrides do not hide an oversized input layer"
        );
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    async fn native_environment_project(cx: &mut gpui::TestAppContext) -> Entity<Project> {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings.set_user_settings(
                    r#"{"terminal":{"detect_venv":"off","env":{"PATH":"/settings/bin","SETTINGS_ONLY":"retained","PRECEDENCE":"settings"}}}"#,
                    cx,
                ).result().expect("native fixture settings");
            });
        });
        let filesystem = crate::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/native-first",
                serde_json::json!({"src":{"main.rs":"fn main() {}"}}),
            )
            .await;
        filesystem
            .insert_tree(
                "/native-second",
                serde_json::json!({"src":{"main.rs":"fn main() {}"}}),
            )
            .await;
        Project::test(
            filesystem,
            [Path::new("/native-first"), Path::new("/native-second")],
            cx,
        )
        .await
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_task_environment_uses_actual_cwd_settings_and_runtime(
        cx: &mut gpui::TestAppContext,
    ) {
        let project = native_environment_project(cx).await;
        let launch = SpawnInTerminal {
            cwd: Some(PathBuf::from("/native-second")),
            env: HashMap::from_iter([
                ("PATH".into(), "/task/bin".into()),
                ("PRECEDENCE".into(), "task".into()),
                ("TASK_ONLY".into(), "retained".into()),
            ]),
            ..Default::default()
        };
        let recipe = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("eligible preparation")
            .await
            .expect("complete native recipe");
        assert_eq!(recipe.working_directory(), Path::new("/native-second"));
        assert_eq!(recipe.logical_environment(), &launch.env);
        let environment = recipe.environment();
        assert_eq!(
            environment.get("PATH").map(String::as_str),
            Some("/task/bin")
        );
        assert_eq!(
            environment.get("PRECEDENCE").map(String::as_str),
            Some("task")
        );
        assert_eq!(
            environment.get("SETTINGS_ONLY").map(String::as_str),
            Some("retained")
        );
        assert_eq!(
            environment.get("TASK_ONLY").map(String::as_str),
            Some("retained")
        );
        assert_eq!(
            environment.get("TERM_PROGRAM").map(String::as_str),
            Some("zed")
        );
        assert_eq!(
            environment.get("TERM").map(String::as_str),
            Some("xterm-256color")
        );
        assert!(!environment.contains_key("SHLVL"));
        assert!(!environment.contains_key("ZED_PROFILE_TASK_TOKEN"));
        assert!(!environment.contains_key("ZED_PROFILE_BUILD_INVOCATION_ENV"));
        assert!(!environment.contains_key("ZED_PROFILE_NATIVE_INVOCATION_ENV"));
        let window_id = project.entity_id().as_u64().to_string();
        assert_eq!(environment.get("ALACRITTY_WINDOW_ID"), Some(&window_id));
        for (name, value) in std::env::vars() {
            if !launch.env.contains_key(&name)
                && !matches!(
                    name.as_str(),
                    "PATH"
                        | "PRECEDENCE"
                        | "SETTINGS_ONLY"
                        | "SHLVL"
                        | "LANG"
                        | "ALACRITTY_WINDOW_ID"
                        | "WINDOWID"
                        | "TERM"
                        | "COLORTERM"
                        | "TERM_PROGRAM"
                        | "TERM_PROGRAM_VERSION"
                        | "ZED_TERM"
                        | "XDG_ACTIVATION_TOKEN"
                        | "DESKTOP_STARTUP_ID"
                        | "ZED_PROFILE_TASK_TOKEN"
                        | "ZED_PROFILE_BUILD_INVOCATION_ENV"
                        | "ZED_PROFILE_NATIVE_INVOCATION_ENV"
                )
            {
                assert!(
                    environment.get(&name) == Some(&value),
                    "eligible inherited key retained"
                );
            }
        }
        project
            .update(cx, |project, cx| {
                project.revalidate_native_task_environment(&recipe, cx)
            })
            .expect("current recipe request")
            .await
            .expect("unchanged complete recipe");
        let first_source = project.read_with(cx, |project, cx| {
            project
                .find_project_path(Path::new("/native-first/src/main.rs"), cx)
                .expect("other worktree source")
        });
        project.update(cx, |project, cx| {
            project.set_active_path(Some(first_source), cx)
        });
        project
            .update(cx, |project, cx| {
                project.revalidate_native_task_environment(&recipe, cx)
            })
            .expect("explicit cwd request")
            .await
            .expect("active editor does not retarget explicit cwd");
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_task_environment_rejects_a_different_project_owner(
        cx: &mut gpui::TestAppContext,
    ) {
        let project = native_environment_project(cx).await;
        let launch = SpawnInTerminal {
            cwd: Some(PathBuf::from("/native-first")),
            ..Default::default()
        };
        let recipe = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("preparation")
            .await
            .expect("recipe");
        let foreign_project = native_environment_project(cx).await;
        let result = foreign_project.update(cx, |project, cx| {
            project.revalidate_native_task_environment(&recipe, cx)
        });
        assert!(
            result
                .err()
                .expect("foreign recipe rejected before preparation")
                .to_string()
                .contains("different project")
        );
        project
            .update(cx, |project, cx| {
                project.revalidate_native_task_environment(&recipe, cx)
            })
            .expect("original owner revalidation")
            .await
            .expect("original owner remains valid");
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_task_environment_rejects_changed_settings_and_runtime(
        cx: &mut gpui::TestAppContext,
    ) {
        let project = native_environment_project(cx).await;
        let launch = SpawnInTerminal {
            cwd: Some(PathBuf::from("/native-first")),
            ..Default::default()
        };
        let recipe = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("preparation")
            .await
            .expect("recipe");
        cx.update(|cx| cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
            settings.set_user_settings(r#"{"terminal":{"detect_venv":"off","env":{"PATH":"/changed/bin","SETTINGS_ONLY":"changed"}}}"#,cx).result().expect("changed settings");
        }));
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_native_task_environment(&recipe, cx))
                .expect("revalidation request")
                .await
                .is_err()
        );
        let fresh = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("fresh preparation")
            .await
            .expect("fresh recipe");
        assert_eq!(
            fresh.environment().get("PATH").map(String::as_str),
            Some("/changed/bin")
        );
        cx.update(|cx| cx.set_global(terminal::HeadlessTerminal(true)));
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_native_task_environment(&fresh, cx))
                .expect("runtime revalidation")
                .await
                .is_err()
        );
        let headless = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("headless preparation")
            .await
            .expect("headless recipe");
        assert!(!headless.environment().contains_key("ALACRITTY_WINDOW_ID"));
        project
            .update(cx, |project, cx| {
                project.revalidate_native_task_environment(&headless, cx)
            })
            .expect("headless revalidation")
            .await
            .expect("fresh headless recipe");
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_task_environment_rejects_activation_and_removed_worktree(
        cx: &mut gpui::TestAppContext,
    ) {
        let project = native_environment_project(cx).await;
        let launch = SpawnInTerminal {
            cwd: Some(PathBuf::from("/native-second")),
            ..Default::default()
        };
        let recipe = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("preparation")
            .await
            .expect("recipe");
        let second_source = project.read_with(cx, |project, cx| {
            project
                .find_project_path(Path::new("/native-second/src/main.rs"), cx)
                .expect("selected source")
        });
        let activation = project.update(cx, |project, cx| {
            project.activate_toolchain(
                ProjectPath::root_path(second_source.worktree_id),
                language::Toolchain {
                    name: "selected-python".into(),
                    path: "/native-second/.venv/bin/python".into(),
                    language_name: LanguageName::new_static("Python"),
                    as_json: serde_json::json!({}),
                },
                cx,
            )
        });
        assert!(
            activation.await.is_some(),
            "actual selected toolchain owner"
        );
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(
                        r#"{"terminal":{"detect_venv":{"on":{"directories":[".venv"]}}}}"#,
                        cx,
                    )
                    .result()
                    .expect("enable activation");
            })
        });
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_native_task_environment(&recipe, cx))
                .expect("activation revalidation")
                .await
                .is_err()
        );
        let refused = project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("activation preparation request")
            .await;
        assert!(
            refused
                .err()
                .expect("unsupported activation rejected")
                .to_string()
                .contains("activation")
        );
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(r#"{"terminal":{"detect_venv":"off"}}"#, cx)
                    .result()
                    .expect("explicit activation off");
            })
        });
        project
            .update(cx, |project, cx| {
                project.prepare_native_task_environment(&launch, cx)
            })
            .expect("fresh recovery")
            .await
            .expect("explicit no-activation recipe");
        project.update(cx, |project, cx| {
            project.remove_worktree(second_source.worktree_id, cx)
        });
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_native_task_environment(&recipe, cx))
                .is_err()
        );
    }

    use super::*;
    #[cfg(all(target_os = "macos", feature = "test-support"))]
    use gpui::BorrowAppContext as _;
    use pretty_assertions::assert_eq;

    #[test]
    fn terminal_task_working_directory_keeps_explicit_host_paths() {
        for directory in [
            "/host/workspace",
            "relative/workspace",
            "",
            "~/remote worktree",
        ] {
            let directory = Path::new(directory);
            assert_eq!(
                explicit_terminal_task_working_directory(directory, true).as_ref(),
                directory,
                "remote paths must not expand the client's home directory"
            );
        }
        assert_eq!(
            explicit_terminal_task_working_directory(Path::new("relative/workspace"), false)
                .as_ref(),
            Path::new("relative/workspace"),
            "ordinary Tasks retain existing relative cwd behavior"
        );
    }

    #[cfg(unix)]
    #[test]
    fn terminal_task_working_directory_expands_local_home_only() {
        let home = std::env::var_os("HOME").expect("Unix test host provides HOME");
        assert_eq!(
            explicit_terminal_task_working_directory(Path::new("~/local worktree"), false).as_ref(),
            Path::new(&home).join("local worktree")
        );
        assert_eq!(
            explicit_terminal_task_working_directory(Path::new("~/remote worktree"), true).as_ref(),
            Path::new("~/remote worktree")
        );
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn terminal_task_working_directory_freezes_the_actual_project_root(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
        });
        let fs = crate::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/first",
            serde_json::json!({"src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        fs.insert_tree(
            "/second",
            serde_json::json!({"src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let project = Project::test(fs, ["/first".as_ref(), "/second".as_ref()], cx).await;
        let first_root = project.read_with(cx, |project, cx| {
            project.terminal_task_working_directory(None, cx)
        });
        assert_eq!(first_root.as_deref(), Some(Path::new("/first")));
        let second_source = project.read_with(cx, |project, cx| {
            project
                .find_project_path(Path::new("/second/src/main.rs"), cx)
                .expect("second source")
        });
        project.update(cx, |project, cx| {
            project.set_active_path(Some(second_source), cx);
        });
        let frozen = project.read_with(cx, |project, cx| {
            assert_eq!(
                project.active_entry_directory(cx),
                Some(PathBuf::from("/second/src"))
            );
            project
                .terminal_task_working_directory(None, cx)
                .expect("selected Task project root")
        });
        assert_eq!(frozen.as_ref(), Path::new("/second"));
        let first_source = project.read_with(cx, |project, cx| {
            project
                .find_project_path(Path::new("/first/src/main.rs"), cx)
                .expect("first source")
        });
        project.update(cx, |project, cx| {
            project.set_active_path(Some(first_source), cx);
        });
        project.read_with(cx, |project, cx| {
            assert_eq!(
                project.terminal_task_working_directory(None, cx).as_deref(),
                Some(Path::new("/first"))
            );
            assert_eq!(
                project
                    .terminal_task_working_directory(Some(frozen.as_ref()), cx)
                    .as_deref(),
                Some(Path::new("/second")),
                "materialized Task cwd does not follow subsequent active worktree changes"
            );
        });
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    async fn terminal_task_working_directory_preserves_no_project_fallback(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
        });
        let fs = crate::FakeFs::new(cx.executor());
        let project = Project::test(fs, std::iter::empty::<&Path>(), cx).await;
        project.read_with(cx, |project, cx| {
            assert!(project.terminal_task_working_directory(None, cx).is_none());
            assert_eq!(
                project
                    .terminal_task_working_directory(Some(Path::new("")), cx)
                    .as_deref(),
                Some(Path::new("")),
                "explicit empty cwd retains ordinary Task semantics"
            );
        });
    }

    fn prepared_cmd_task(command_arg: &str) -> SpawnInTerminal {
        SpawnInTerminal {
            command: Some("cmd.exe".to_string()),
            args: vec!["/S".to_string(), "/C".to_string(), command_arg.to_string()],
            shell: Shell::Program("cmd.exe".to_string()),
            ..SpawnInTerminal::default()
        }
    }

    #[test]
    fn formats_prepared_cmd_task_for_powershell_activation() {
        let task = prepared_cmd_task("\"echo Hi there\"");

        assert_eq!(
            format_task_for_activation(&task, ShellKind::PowerShell, "powershell.exe", true),
            "&cmd.exe /S /C '\"echo Hi there\"'"
        );
    }

    #[test]
    fn formats_prepared_cmd_task_for_cmd_activation() {
        let task = prepared_cmd_task("\"echo Hi there\"");

        assert_eq!(
            format_task_for_activation(&task, ShellKind::Cmd, "cmd.exe", true),
            "cmd.exe /S /C \"echo Hi there\""
        );
    }

    #[test]
    fn formats_prepared_cmd_task_with_shell_args_for_activation() {
        let task = SpawnInTerminal {
            command: Some("cmd.exe".to_string()),
            args: vec![
                "/D".to_string(),
                "/S".to_string(),
                "/C".to_string(),
                "\"echo Hi there\"".to_string(),
            ],
            shell: Shell::WithArguments {
                program: "cmd.exe".to_string(),
                args: vec!["/D".to_string()],
                title_override: None,
            },
            ..SpawnInTerminal::default()
        };

        assert_eq!(
            format_task_for_activation(&task, ShellKind::PowerShell, "powershell.exe", true),
            "&cmd.exe /D /S /C '\"echo Hi there\"'"
        );
    }

    #[test]
    fn formats_prepared_cmd_task_with_single_quote_for_powershell_activation() {
        let task = prepared_cmd_task("\"echo It's fine\"");

        assert_eq!(
            format_task_for_activation(&task, ShellKind::PowerShell, "powershell.exe", true),
            "&cmd.exe /S /C '\"echo It''s fine\"'"
        );
    }

    #[test]
    fn formats_non_cmd_task_for_activation() {
        let task = SpawnInTerminal {
            command: Some("cargo".to_string()),
            args: vec!["test".to_string(), "some test".to_string()],
            shell: Shell::System,
            ..SpawnInTerminal::default()
        };

        assert_eq!(
            format_task_for_activation(&task, ShellKind::PowerShell, "powershell.exe", true),
            "&cargo test 'some test'"
        );
    }
}
