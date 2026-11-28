use std::{
    cell::Cell,
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context as _, Result, ensure};
use command_palette_hooks::CommandPaletteFilter;
use feature_flags::{
    FeatureFlagAppExt as _, FeatureFlagStore, RustNativeProfilingExperimentalFeatureFlag,
};
use futures::StreamExt as _;
use gpui::{
    App, AppContext as _, Entity, EventEmitter, FocusHandle, Focusable, PathPromptOptions, Render,
    Task, UniformListScrollHandle, WeakEntity, actions, relative, uniform_list,
};
use project::{
    DirectoryLister, Project, ProjectPath,
    profile_results::{ProfileHost, ProfileInput, ProfileResults, load_samply_profile},
    profile_results_store::RemoteProfileError,
};
use remote::{ConnectionState, RemoteClientEvent};
use ui::{Button, ButtonStyle, ContextMenu, PopoverMenu, Tab, TabPosition, Tooltip, prelude::*};
use util::ResultExt as _;
use workspace::{
    Item, Toast, Workspace, notifications::NotificationId, tasks::CompletedTaskArtifactGroup,
};

use crate::language_tool_tree::{
    self, LanguageToolNode, LanguageToolNodeId, LanguageToolProviderStatus, LanguageToolSnapshot,
    LanguageToolTreeHost, LanguageToolTreeOptions,
};

actions!(
    profiling,
    [
        /// Opens the native profile viewer for explicitly selected project artifacts.
        OpenNativeProfile,
        /// Zooms the native flamegraph to the selected call-tree frame.
        ZoomToSelectedFrame,
        /// Restores the complete native flamegraph.
        ResetProfileZoom,
        /// Switches between the native call tree and flamegraph.
        ToggleProfileView
    ]
);

pub fn init(cx: &mut App) {
    update_native_profile_command_filter(cx);
    cx.observe_flag::<RustNativeProfilingExperimentalFeatureFlag, _>(|_, cx| {
        update_native_profile_command_filter(cx)
    })
    .detach();
    cx.observe_new(|workspace: &mut Workspace, _, cx| {
        cx.observe_global::<FeatureFlagStore>(|workspace, cx| {
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                workspace.dismiss_notification(
                    &NotificationId::unique::<CompletedTaskArtifactGroup>(),
                    cx,
                );
            }
        })
        .detach();
        cx.subscribe(&cx.entity(), |workspace, _, event, cx| {
            if let workspace::Event::TaskArtifactGroupReady(group) = event {
                offer_task_profile(workspace, group.clone(), cx);
            }
        })
        .detach();
        workspace.register_action(|workspace, _: &OpenNativeProfile, window, cx| {
            if ProfileHost::local_import_host().is_none()
                || !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
            {
                return;
            }
            let project = workspace.project().clone();
            let workspace_handle = cx.entity().downgrade();
            let view = cx.new(|cx| ProfileView::new(project, workspace_handle, cx));
            workspace.add_item_to_active_pane(Box::new(view), None, true, window, cx);
        });
    })
    .detach();
}

fn update_native_profile_command_filter(cx: &mut App) {
    let enabled = ProfileHost::local_import_host().is_some()
        && cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>();
    CommandPaletteFilter::update_global(cx, |filter, _| {
        let actions = [
            std::any::TypeId::of::<OpenNativeProfile>(),
            std::any::TypeId::of::<ZoomToSelectedFrame>(),
            std::any::TypeId::of::<ResetProfileZoom>(),
            std::any::TypeId::of::<ToggleProfileView>(),
        ];
        if enabled {
            filter.show_action_types(&actions);
        } else {
            filter.hide_action_types(&actions);
        }
    });
}

pub fn open_diagnostic_profile_result(
    workspace: &mut Workspace,
    result: project::profile_results_store::DiagnosticProfileResult,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<Workspace>,
) -> Result<Entity<ProfileView>> {
    ensure!(
        cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
        "Native profiling requires the rust-native-profiling-experimental opt-in"
    );
    let project = workspace.project().clone();
    ensure!(
        project.read(cx).is_local()
            && !project.read(cx).is_shared()
            && !project.read(cx).is_isolated_recipe_test(),
        "Diagnostic profile host changed"
    );
    project
        .read(cx)
        .revalidate_diagnostic_profile_result(&result, cx)?;
    let (paths, result) = result.into_parts();
    let title = recording_title(&paths);
    let result_paths = paths.into_iter().collect::<collections::HashSet<_>>();
    ensure!(
        profile_paths_permitted(project.read(cx), &profile_path_groups(&result_paths), cx),
        "Diagnostic profile permission or trust changed"
    );
    let tree = profile_tree(&result, || false)?;
    let flame_rows = Arc::new(profile_flamegraph(&result, None, || false)?);
    let workspace_handle = cx.entity().downgrade();
    let view = cx.new(|cx| {
        let mut view = ProfileView::new(project, workspace_handle, cx);
        view.notice = "Imported explicitly selected bounded artifacts. Import does not prove the workload was captured or a repair verified. This view only.".into();
        view.recording_title = Some(title);
        view.result = Some(Arc::new(result));
        view.result_paths = result_paths;
        view.flame_rows = flame_rows;
        view.host.replace_snapshot(tree);
        view
    });
    workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
    Ok(view)
}

async fn check_task_versions(
    fs: &dyn project::Fs,
    versions: &[(PathBuf, project::Entry)],
) -> Result<()> {
    for (path, entry) in versions {
        let metadata = fs
            .metadata(path)
            .await
            .context("Cannot verify Task artifact version")?
            .context("Task artifact disappeared")?;
        ensure!(
            !metadata.is_dir
                && !metadata.is_fifo
                && metadata.inode == entry.inode
                && Some(metadata.mtime) == entry.mtime
                && metadata.len == entry.size,
            "Task artifact changed since successful completion"
        );
    }
    Ok(())
}

fn native_task_group_supported(group: &CompletedTaskArtifactGroup) -> bool {
    let declaration = group.group();
    declaration.format == "samply-0.13.1"
        && declaration.files.len() == 2
        && declaration.files.get("profile").is_some_and(|file| {
            file.max_bytes <= 32 * 1024 * 1024 && !file.path.ends_with(".syms.json")
        })
        && declaration.files.get("symbols").is_some_and(|file| {
            file.max_bytes <= 16 * 1024 * 1024 && file.path.ends_with(".syms.json")
        })
}

fn offer_task_profile(
    workspace: &mut Workspace,
    group: Rc<CompletedTaskArtifactGroup>,
    cx: &mut Context<Workspace>,
) {
    if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
        || ProfileHost::local_import_host().is_none()
        || !native_task_group_supported(&group)
        || !group.is_current(workspace.project(), cx)
    {
        return;
    }
    if cfg!(target_os = "linux") && !group.host_verified() {
        return;
    }
    if !workspace.project().read(cx).is_local() && !group.host_verified() {
        workspace.show_toast(Toast::new(NotificationId::unique::<CompletedTaskArtifactGroup>(),
            "Task-bound remote profile import needs host completion and file-version verification. Use Import Profile to select remote artifacts explicitly."), cx);
        return;
    }
    let Some(notice) =
        task_profile_import_notice(&group, workspace.project().read(cx), cx).log_err()
    else {
        return;
    };
    let handle = cx.entity().downgrade();
    workspace.show_toast(
        Toast::new(
            NotificationId::unique::<CompletedTaskArtifactGroup>(),
            notice,
        )
        .on_click("Import Profile", move |window, cx| {
            handle
                .update(cx, |workspace, cx| {
                    open_task_profile(workspace, group.clone(), window, cx)
                })
                .log_err();
        }),
        cx,
    );
}

fn task_profile_import_notice(
    group: &CompletedTaskArtifactGroup,
    project: &Project,
    cx: &App,
) -> Result<String> {
    let artifacts = ["profile", "symbols"]
        .into_iter()
        .map(|role| {
            let path = group.path(role).context("Task artifact role is missing")?;
            let path = project
                .absolute_path(path, cx)
                .context("Task artifact worktree is unavailable")?;
            Ok(format!("{role}: {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?
        .join("\n");
    let processing = if !project.is_local() {
        "Import transfers only normalized results to this window. Raw files stay on the host; no profile contents are uploaded to telemetry."
    } else {
        "Import processes these files locally without uploading them or sending profile contents to telemetry."
    };
    Ok(format!(
        "The profiling Task completed.\nSelected artifacts:\n{artifacts}\n{processing}\nRaw files may contain sensitive source paths, symbols and process metadata. Files stay on disk until you delete them through project file deletion; Clear or closing this view does not delete them."
    ))
}

fn open_task_profile(
    workspace: &mut Workspace,
    group: Rc<CompletedTaskArtifactGroup>,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let handle = cx.entity().downgrade();

    if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
        return;
    }
    if !group.is_current(workspace.project(), cx) {
        workspace.show_toast(
            Toast::new(
                NotificationId::unique::<CompletedTaskArtifactGroup>(),
                "These Task artifacts changed or their project session ended. Run the Task again.",
            ),
            cx,
        );
        return;
    }
    let view = workspace
        .item_of_type::<ProfileView>(cx)
        .unwrap_or_else(|| {
            cx.new(|cx| ProfileView::new(workspace.project().clone(), handle.clone(), cx))
        });
    let result = view.update(cx, |view, cx| view.import_task_group(group.clone(), cx));
    match result {
        Ok(()) => workspace.add_item_to_active_pane(Box::new(view), None, true, window, cx),
        Err(error) => workspace.show_toast(
            Toast::new(
                NotificationId::unique::<CompletedTaskArtifactGroup>(),
                format!("Task profile import could not start: {error}"),
            ),
            cx,
        ),
    }
}

#[derive(Clone, Debug)]
struct FlameBar {
    node: usize,
    left: f32,
    width: f32,
}

fn flamegraph_bar_at(bars: &[FlameBar], position: f32) -> Option<&FlameBar> {
    if !(0.0..1.0).contains(&position) {
        return None;
    }
    let index = bars
        .partition_point(|bar| bar.left <= position)
        .checked_sub(1)?;
    bars.get(index)
        .filter(|bar| position < bar.left + bar.width)
}

type ProfilePathGroups =
    collections::HashMap<project::WorktreeId, collections::HashSet<Arc<util::rel_path::RelPath>>>;

fn profile_path_groups(paths: &collections::HashSet<ProjectPath>) -> ProfilePathGroups {
    let mut groups = ProfilePathGroups::default();
    for path in paths {
        groups
            .entry(path.worktree_id)
            .or_default()
            .insert(path.path.clone());
    }
    groups
}

fn profile_paths_permitted(project: &Project, groups: &ProfilePathGroups, cx: &App) -> bool {
    if project.is_disconnected(cx)
        || project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
            &project.worktree_store(),
            cx,
        )
    {
        return false;
    }
    let mut checked = 0;
    for worktree in project.visible_worktrees(cx) {
        let worktree = worktree.read(cx);
        let Some(paths) = groups.get(&worktree.id()) else {
            continue;
        };
        let snapshot = worktree.snapshot();
        let permitted = |entry: &project::Entry| {
            entry.is_file() && !entry.is_private && !entry.is_fifo && !entry.is_external
        };
        // Dense imports avoid a separate tree search for every source while sparse imports avoid scanning unrelated files.
        let valid = if paths.len() >= snapshot.entry_count().div_ceil(2) {
            snapshot
                .files(true, 0)
                .filter(|entry| paths.contains(entry.path.as_ref()) && permitted(entry))
                .count()
                == paths.len()
        } else {
            paths
                .iter()
                .all(|path| snapshot.entry_for_path(path).is_some_and(permitted))
        };
        if !valid {
            return false;
        }
        checked += 1;
    }
    checked == groups.len()
}

pub struct ProfileView {
    project: Entity<Project>,
    workspace: WeakEntity<Workspace>,
    focus: FocusHandle,
    toolbar_focus: [FocusHandle; 10],
    pane_width: gpui::Pixels,
    flame_scroll: UniformListScrollHandle,
    hovered_bar: Option<(usize, usize)>,
    host: LanguageToolTreeHost,
    result: Option<Arc<ProfileResults>>,
    recording_title: Option<SharedString>,
    task_group: Option<Rc<CompletedTaskArtifactGroup>>,
    result_paths: collections::HashSet<ProjectPath>,
    importing_paths: collections::HashSet<ProjectPath>,
    flamegraph: bool,
    zoom_root: Option<usize>,
    zoom_generation: u64,
    zoom_cancellation: Arc<AtomicBool>,
    zoom_task: Task<()>,
    flame_rows: Arc<Vec<Vec<FlameBar>>>,
    #[cfg(test)]
    painted_flame_bars: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    import_application_time: Option<std::time::Duration>,
    #[cfg(test)]
    invalidation_time: Option<std::time::Duration>,
    #[cfg(test)]
    import_processing_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(test)]
    import_completion_gate: Option<(
        futures::channel::oneshot::Sender<(
            std::sync::Weak<ProfileResults>,
            std::sync::Weak<Vec<Vec<FlameBar>>>,
        )>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(test)]
    invalidation_path_count: usize,
    notice: String,
    generation: u64,
    cancellation: Arc<AtomicBool>,
    import: Task<()>,
    importing: bool,
    selecting: bool,
    _collaboration_session: Option<Task<()>>,
    _subscriptions: Vec<gpui::Subscription>,
}

impl ProfileView {
    fn new(
        project: Entity<Project>,
        workspace: WeakEntity<Workspace>,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus = cx.focus_handle().tab_index(0).tab_stop(true);
        let mut subscriptions = vec![cx.subscribe(&project, |this, _, event, cx| {
            #[cfg(test)]
            let invalidation_started = std::time::Instant::now();
            let revoked_path = if let project::Event::WorktreeUpdatedEntries(worktree_id, changes) =
                event
            {
                let project = this.project.read(cx);
                let paths = changes
                    .iter()
                    .filter_map(|(path, _, _)| {
                        let path = ProjectPath {
                            worktree_id: *worktree_id,
                            path: path.clone(),
                        };
                        (this.result_paths.contains(&path) || this.importing_paths.contains(&path))
                            .then_some(path.path)
                    })
                    .collect::<collections::HashSet<_>>();
                #[cfg(test)]
                { this.invalidation_path_count += paths.len(); }
                !paths.is_empty()
                    && (!project.is_local()
                        || !profile_paths_permitted(
                            project,
                            &collections::HashMap::from_iter([(*worktree_id, paths)]),
                            cx,
                        ))
            } else {
                false
            };
            let stale_task = this
                .task_group
                .as_ref()
                .is_some_and(|group| !group.is_current(&this.project, cx));
            if (revoked_path
                || stale_task
                || matches!(
                    event,
                    project::Event::WorktreeRemoved(_)
                        | project::Event::DisconnectedFromHost
                        | project::Event::DisconnectedFromRemote { .. }
                        | project::Event::RemoteIdChanged(_)
                        | project::Event::Reshared
                        | project::Event::HostReshared
                        | project::Event::UpstreamSessionChanged
                        | project::Event::Rejoined
                        | project::Event::Closed
                        | project::Event::WorktreePathsChanged { .. }
                ))
                && (this.result.is_some() || this.importing || this.selecting)
            {
                this.clear(cx);
                this.notice = if revoked_path {
                    "Profile cleared because a referenced file is no longer permitted or has changed"
                        .to_string()
                } else {
                    "Profile cleared because the project session or worktree changed".to_string()
                };
                cx.notify();
            }
            #[cfg(test)]
            if matches!(event, project::Event::WorktreeUpdatedEntries(..)) {
                this.invalidation_time = Some(
                    this.invalidation_time.unwrap_or_default() + invalidation_started.elapsed(),
                );
            }
        })];
        if let Some(remote) = project.read(cx).remote_client() {
            subscriptions.push(cx.subscribe(&remote, |this, _, event, cx| {
                if matches!(
                    event,
                    RemoteClientEvent::ConnectionStateChanged(
                        ConnectionState::Reconnecting | ConnectionState::Disconnected
                    ) | RemoteClientEvent::Disconnected { .. }
                ) && (this.result.is_some() || this.importing || this.selecting)
                {
                    this.clear(cx);
                    this.notice =
                        "Profile cleared because the remote project session changed".to_string();
                    cx.notify();
                }
            }));
        }
        if let Some(trust) = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(cx.subscribe(&trust, |this, _, _, cx| {
                if project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &this.project.read(cx).worktree_store(),
                    cx,
                ) {
                    this.clear(cx);
                    this.notice = "Profile cleared because project trust changed".to_string();
                    cx.notify();
                }
            }));
        }
        let collaboration_session = project.read(cx).is_via_collab().then(|| {
            let mut statuses = project.read(cx).client().status();
            let mut previous_status = *statuses.borrow();
            cx.spawn(async move |this, cx| {
                while let Some(status) = statuses.next().await {
                    if status == previous_status {
                        continue;
                    }
                    previous_status = status;
                    if this
                        .update(cx, |this, cx| {
                            if this.result.is_some() || this.importing || this.selecting {
                                this.clear(cx);
                                this.notice =
                                    "Profile cleared because the collaboration session changed"
                                        .to_string();
                                cx.notify();
                            }
                        })
                        .log_err()
                        .is_none()
                    {
                        break;
                    }
                }
            })
        });
        subscriptions.push(cx.observe_global::<FeatureFlagStore>(|this, cx| {
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                this.clear(cx);
                this.notice =
                    "Native profiling requires the rust-native-profiling-experimental opt-in"
                        .into();
            }
            cx.notify();
        }));
        let notice = if project.read(cx).is_local() {
            "Select a Samply 0.13.1 profile JSON and its symbol sidecar. Processing stays local; clearing this view does not delete your files."
        } else {
            "Select a Samply 0.13.1 profile JSON and its symbol sidecar on the project host. Import requires consent to transfer normalized results; raw files stay on the host."
        }.to_string();
        Self {
            project,
            workspace,
            host: LanguageToolTreeHost::with_focus_handle(focus.clone()),
            focus,
            toolbar_focus: std::array::from_fn(|_| cx.focus_handle()),
            pane_width: px(800.0),
            flame_scroll: UniformListScrollHandle::new(),
            hovered_bar: None,
            result: None,
            recording_title: None,
            task_group: None,
            result_paths: collections::HashSet::default(),
            importing_paths: collections::HashSet::default(),
            flamegraph: false,
            zoom_root: None,
            zoom_generation: 0,
            zoom_cancellation: Arc::new(AtomicBool::new(false)),
            zoom_task: Task::ready(()),
            flame_rows: Arc::new(Vec::new()),
            #[cfg(test)]
            painted_flame_bars: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            #[cfg(test)]
            import_application_time: None,
            #[cfg(test)]
            invalidation_time: None,
            #[cfg(test)]
            import_processing_gate: None,
            #[cfg(test)]
            import_completion_gate: None,
            #[cfg(test)]
            invalidation_path_count: 0,
            notice,
            generation: 0,
            cancellation: Arc::new(AtomicBool::new(false)),
            import: Task::ready(()),
            importing: false,
            selecting: false,
            _collaboration_session: collaboration_session,
            _subscriptions: subscriptions,
        }
    }

    fn cancel_import(&mut self, cx: &mut Context<Self>) {
        self.cancellation.store(true, Ordering::Relaxed);
        self.generation = self.generation.wrapping_add(1);
        self.import = Task::ready(());
        self.importing = false;
        self.selecting = false;
        self.importing_paths.clear();
        self.notice = "Import cancelled; the previous result is preserved".to_string();
        cx.notify();
    }

    fn clear(&mut self, cx: &mut Context<Self>) {
        self.cancellation.store(true, Ordering::Relaxed);
        self.generation = self.generation.wrapping_add(1);
        self.import = Task::ready(());
        self.importing = false;
        self.selecting = false;
        self.result = None;
        self.recording_title = None;
        self.task_group = None;
        self.result_paths.clear();
        self.importing_paths.clear();
        self.flame_rows = Arc::new(Vec::new());
        self.zoom_generation = self.zoom_generation.wrapping_add(1);
        self.zoom_cancellation.store(true, Ordering::Relaxed);
        self.zoom_task = Task::ready(());
        self.zoom_root = None;
        self.hovered_bar = None;
        self.host.replace_snapshot(LanguageToolSnapshot {
            roots: Vec::new(),
            status: LanguageToolProviderStatus::Empty(
                "Profile cleared; selected files remain on disk".to_string(),
            ),
        });
        self.notice = "Profile cleared from this view; user-owned artifact files were not deleted"
            .to_string();
        cx.notify();
    }

    fn prompt(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            self.notice =
                "Native profiling requires the rust-native-profiling-experimental opt-in".into();
            cx.notify();
            return;
        }
        let project = self.project.read(cx);
        if ProfileHost::local_import_host().is_none() {
            self.notice = "Native profile import requires a macOS or Linux host".to_string();
            cx.notify();
            return;
        }
        if project.is_disconnected(cx)
            || project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                &project.worktree_store(),
                cx,
            )
        {
            self.notice =
                "Connect to and trust the project before selecting profile artifacts".to_string();
            cx.notify();
            return;
        }
        let remote = !project.is_local();
        let Some(workspace) = self.workspace.upgrade() else {
            return;
        };
        self.cancel_import(cx);
        self.selecting = true;
        let selection_details = if remote {
            "Select profile JSON and its .syms.json companion. Raw files stay on the host; Clear does not delete them."
        } else {
            "Select profile JSON and its .syms.json companion. Import processes them locally without uploading them; Clear does not delete the files."
        };
        self.notice = selection_details.to_string();
        let generation = self.generation;
        let prompt = workspace.update(cx, |workspace, cx| {
            workspace.prompt_for_open_path(
                PathPromptOptions {
                    files: true,
                    directories: false,
                    multiple: true,
                    prompt: Some(selection_details.into()),
                },
                DirectoryLister::Project(self.project.clone()),
                window,
                cx,
            )
        });
        cx.spawn_in(window, async move |this, cx| {
            let selection = async {
                let paths = prompt.await?;
                if !this.read_with(cx, |this, _| this.generation == generation && this.selecting)? {
                    return anyhow::Ok(());
                }
                let Some(mut paths) = paths else {
                    this.update(cx, |this, cx| {
                        this.selecting = false;
                        this.notice = "Profile selection cancelled; the previous result is preserved".into();
                        cx.notify();
                    })?;
                    return Ok(());
                };
                if paths.len() == 1 {
                    let companion = this.update_in(cx, |this, window, cx| {
                        this.notice = "Now select the other profile artifact (JSON or .syms.json sidecar). Files stay on the project host; Clear does not delete them.".into();
                        cx.notify();
                        workspace.update(cx, |workspace, cx| {
                            workspace.prompt_for_open_path(
                                PathPromptOptions {
                                    files: true,
                                    directories: false,
                                    multiple: false,
                                    prompt: Some("Select the other profile artifact. Files stay on disk after Clear.".into()),
                                },
                                DirectoryLister::Project(this.project.clone()), window, cx,
                            )
                        })
                    })?;
                    let Some(companion) = companion.await? else {
                        this.update(cx, |this, cx| {
                            if this.generation == generation { this.cancel_import(cx); }
                        })?;
                        return Ok(());
                    };
                    paths.extend(companion);
                    if !this.read_with(cx, |this, _| this.generation == generation && this.selecting)? {
                        return Ok(());
                    }
                }
                if paths.len() != 2 {
                    this.update(cx, |this, cx| {
                        this.selecting = false;
                        this.notice = "Select exactly one profile JSON and one symbol sidecar".into();
                        cx.notify();
                    })?;
                    return Ok(());
                }
                let selected_artifacts = this.read_with(cx, |this, cx| {
                    this.permitted_artifact_paths(&paths, cx)
                })??;
                let selected_paths = paths.iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                let (message, details, import_button) = if remote {
                    (
                        "Import normalized profile results from this host?",
                        format!("Selected artifacts:\n{selected_paths}\n\nThe host will process these two files. Function labels, sample counts and permitted project-relative source paths will be transferred to this view. Raw files stay on the host. Results remain in memory and are cleared on disconnect or Clear; clearing does not delete the selected files. No profile contents are uploaded to telemetry."),
                        "Import Results",
                    )
                } else {
                    (
                        "Import these profile artifacts locally?",
                        format!("Selected artifacts:\n{selected_paths}\n\nThese files may contain sensitive source paths, symbols and process metadata. Zed processes them locally without uploading them or sending profile contents to telemetry. Results remain in memory and are cleared when you clear or close this view. The raw files stay on disk until you delete them through project file deletion; clearing or closing this view does not delete them."),
                        "Import Profile",
                    )
                };
                let consent = cx.update(|window, cx| window.prompt(
                    gpui::PromptLevel::Warning,
                    message,
                    Some(&details),
                    &[import_button, "Cancel"], cx,
                ))?;
                let answer = consent.await?;
                if !this.read_with(cx, |this, _| this.generation == generation && this.selecting)? {
                    return Ok(());
                }
                if answer != 0 {
                    this.update(cx, |this, cx| {
                        if this.generation == generation && this.selecting { this.cancel_import(cx); }
                    })?;
                    return Ok(());
                }
                this.update(cx, |this, cx| {
                    if this.generation != generation || !this.selecting { return; }
                    this.selecting = false;
                    let result = (|| {
                        ensure!(this.project.read(cx).is_local() != remote, "The profile host changed during selection");
                        ensure!(this.permitted_artifact_paths(&paths, cx)? == selected_artifacts, "The selected artifact worktree changed");
                        if remote {
                            this.import_paths_with_consent(paths, true, cx)
                        } else {
                            this.import_paths(paths, cx)
                        }
                    })();
                    if let Err(error) = result {
                        this.notice = error.to_string();
                        cx.notify();
                    }
                })?;
                Ok(())
            }
            .await;
            if selection.is_err() {
                this.update(cx, |this, cx| {
                    if this.generation == generation {
                        this.selecting = false;
                        this.notice = "Profile selection failed; the previous result is preserved".into();
                        cx.notify();
                    }
                }).log_err();
            }
        })
        .detach();
    }

    fn import_task_group(
        &mut self,
        group: Rc<CompletedTaskArtifactGroup>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        ensure!(
            native_task_group_supported(&group),
            "Unsupported native Task artifact group"
        );
        ensure!(
            group.is_current(&self.project, cx),
            "Task artifacts or their project session changed"
        );
        let remote = !self.project.read(cx).is_local();
        ensure!(
            !remote || group.host_verified(),
            "Task-bound remote import requires host verification"
        );
        let paths = ["profile", "symbols"]
            .into_iter()
            .map(|role| {
                let path = group.path(role).context("Task artifact role is missing")?;
                self.project
                    .read(cx)
                    .absolute_path(path, cx)
                    .context("Task artifact worktree is unavailable")
            })
            .collect::<Result<Vec<_>>>()?;
        self.import_artifacts(paths, remote, Some(group), cx)
    }

    fn import_paths(&mut self, paths: Vec<PathBuf>, cx: &mut Context<Self>) -> Result<()> {
        self.import_paths_with_consent(paths, false, cx)
    }

    fn import_paths_with_consent(
        &mut self,
        paths: Vec<PathBuf>,
        consent_to_transfer: bool,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.import_artifacts(paths, consent_to_transfer, None, cx)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn import_paths_with_consent_for_test(
        &mut self,
        paths: Vec<PathBuf>,
        consent_to_transfer: bool,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.import_paths_with_consent(paths, consent_to_transfer, cx)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn active_import_cancellation_for_test(&self) -> Option<Arc<AtomicBool>> {
        self.importing.then(|| self.cancellation.clone())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn is_selecting_for_test(&self) -> bool {
        self.selecting
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn sample_count_for_test(&self) -> Option<u64> {
        self.result.as_ref().map(|result| result.total_samples)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn notice_for_test(&self) -> &str {
        &self.notice
    }

    fn permitted_artifact_paths(&self, paths: &[PathBuf], cx: &App) -> Result<Vec<ProjectPath>> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native profiling opt-in was revoked before import"
        );
        let project = self.project.read(cx);
        ensure!(
            !project.is_disconnected(cx),
            "The profile host is disconnected"
        );
        ensure!(
            !project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                &project.worktree_store(),
                cx
            ),
            "Trust the project before importing profile artifacts"
        );
        let mut artifact_paths = Vec::new();
        for path in paths {
            let path = project
                .find_project_path(path, cx)
                .context("Profile inputs must be visible project files")?;
            ensure!(
                project
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == path.worktree_id),
                "Profile inputs must be in a visible worktree"
            );
            let entry = project
                .entry_for_path(&path, cx)
                .context("Profile input is missing")?;
            ensure!(
                entry.is_file() && !entry.is_private && !entry.is_fifo && !entry.is_external,
                "Profile inputs must be permitted project files"
            );
            artifact_paths.push(path);
        }
        Ok(artifact_paths)
    }

    fn import_artifacts(
        &mut self,
        mut paths: Vec<PathBuf>,
        consent_to_transfer: bool,
        task_group: Option<Rc<CompletedTaskArtifactGroup>>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native profiling opt-in was revoked before import"
        );
        let remote = !self.project.read(cx).is_local();
        let local_host = ProfileHost::local_import_host()
            .context("Native profile import requires a macOS or Linux host")?;
        ensure!(
            !self.project.read(cx).is_disconnected(cx),
            "The profile host is disconnected"
        );
        ensure!(
            paths.len() == 2,
            "Select exactly one profile JSON and one symbol sidecar"
        );
        let symbol_index = paths
            .iter()
            .position(|path| path.to_string_lossy().ends_with(".syms.json"))
            .context("Select an explicit .syms.json companion")?;
        let symbols_path = paths.remove(symbol_index);
        let profile_path = paths.pop().context("Select a profile JSON")?;
        ensure!(
            profile_path
                .extension()
                .is_some_and(|extension| extension == "json")
                && !profile_path.to_string_lossy().ends_with(".syms.json"),
            "Only uncompressed profile JSON is supported"
        );
        let project = self.project.read(cx);
        let artifact_paths = self.permitted_artifact_paths(&[profile_path, symbols_path], cx)?;
        let title = recording_title(&artifact_paths);
        let cancellation = Arc::new(AtomicBool::new(false));
        let remote_import = if remote {
            let inputs: [ProjectPath; 2] = artifact_paths
                .clone()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Select exactly two profile artifacts"))?;
            Some(if let Some(group) = task_group.as_ref() {
                ensure!(
                    group.host_verified(),
                    "Host did not verify this profile Task"
                );
                project.import_remote_task_profile(
                    inputs,
                    group.run_id().to_string(),
                    cancellation.clone(),
                    cx,
                )?
            } else {
                project.import_remote_profile(
                    inputs,
                    consent_to_transfer,
                    cancellation.clone(),
                    cx,
                )?
            })
        } else {
            None
        };
        let task_versions = task_group
            .as_ref()
            .filter(|_| !remote)
            .map(|group| {
                ["profile", "symbols"]
                    .into_iter()
                    .map(|role| {
                        let path = group.path(role).context("Task artifact role is missing")?;
                        let path = project
                            .absolute_path(path, cx)
                            .context("Task worktree is unavailable")?;
                        let entry = group
                            .entry(role)
                            .context("Task artifact version is missing")?;
                        Ok((path, entry.clone()))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let fs = (!remote).then(|| project.fs().clone());
        #[cfg(test)]
        let import_processing_gate = self.import_processing_gate.take();
        #[cfg(test)]
        let import_completion_gate = self.import_completion_gate.take();
        let worktrees = project
            .visible_worktrees(cx)
            .map(|worktree| {
                let worktree = worktree.read(cx);
                (worktree.id(), worktree.abs_path(), worktree.snapshot())
            })
            .collect::<Vec<_>>();
        let admitted_inputs = if remote {
            Vec::new()
        } else {
            artifact_paths
                .iter()
                .map(|path| {
                    let (_, _, snapshot) = worktrees
                        .iter()
                        .find(|(id, _, _)| *id == path.worktree_id)
                        .context("Profile worktree is unavailable")?;
                    ProfileInput::from_snapshot(snapshot, path.path.as_ref())
                })
                .collect::<Result<Vec<_>>>()?
        };
        self.cancellation.store(true, Ordering::Relaxed);
        self.cancellation = cancellation;
        self.selecting = false;
        let cancel = self.cancellation.clone();
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.importing = true;
        self.importing_paths = artifact_paths.iter().cloned().collect();
        let mut result_paths = self.importing_paths.clone();
        self.notice = if remote {
            "Importing normalized results from the project host…"
        } else {
            "Importing profile locally…"
        }
        .to_string();
        let parsing = cx.background_spawn(async move {
            if let Some(versions) = task_versions.as_ref() {
                check_task_versions(
                    fs.as_ref().context("Task filesystem unavailable")?.as_ref(),
                    versions,
                )
                .await?;
            }
            let result = if let Some(import) = remote_import {
                import.await?
            } else {
                load_samply_profile(
                    fs.as_ref()
                        .context("Local profile filesystem is unavailable")?
                        .as_ref(),
                    [
                        admitted_inputs.first().context("Missing profile input")?,
                        admitted_inputs.get(1).context("Missing symbol input")?,
                    ],
                    local_host,
                    &cancel,
                    |path, _line| {
                        worktrees.iter().find_map(|(id, root, snapshot)| {
                            let relative = path.strip_prefix(root).ok()?;
                            let relative = util::rel_path::RelPath::new(
                                relative,
                                util::paths::PathStyle::local(),
                            )
                            .ok()?;
                            let entry = snapshot.entry_for_path(relative.as_ref())?;
                            (entry.is_file()
                                && !entry.is_private
                                && !entry.is_fifo
                                && !entry.is_external)
                                .then(|| ProjectPath {
                                    worktree_id: *id,
                                    path: Arc::from(relative.as_ref()),
                                })
                        })
                    },
                )
                .await?
            };
            #[cfg(test)]
            if let Some((reached, resume)) = import_processing_gate {
                reached
                    .send(())
                    .expect("active import test must observe the processing boundary");
                resume
                    .await
                    .expect("active import test must release processing");
            }
            result_paths.extend(
                result
                    .frames
                    .iter()
                    .filter_map(|frame| frame.source.as_ref().map(|(path, _)| path.clone())),
            );
            if let Some(versions) = task_versions.as_ref() {
                check_task_versions(
                    fs.as_ref().context("Task filesystem unavailable")?.as_ref(),
                    versions,
                )
                .await?;
            }
            let tree = profile_tree(&result, || cancel.load(Ordering::Relaxed))?;
            let flame_rows = Arc::new(profile_flamegraph(&result, None, || {
                cancel.load(Ordering::Relaxed)
            })?);
            let path_groups = profile_path_groups(&result_paths);
            let result = Arc::new(result);
            #[cfg(test)]
            if let Some((reached, resume)) = import_completion_gate {
                reached
                    .send((Arc::downgrade(&result), Arc::downgrade(&flame_rows)))
                    .expect("completed import test must observe the model");
                resume
                    .await
                    .expect("completed import test must release processing");
            }
            anyhow::Ok((result, tree, flame_rows, result_paths, path_groups))
        });
        self.import = cx.spawn(async move |this, cx| {
            let result = parsing.await;
            this.update(cx, |this, cx| {
                #[cfg(test)]
                let application_started = std::time::Instant::now();
                if this.generation != generation || !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() { return; }
                this.importing = false;
                this.importing_paths.clear();
                if remote && result.as_ref().err().is_some_and(|error| matches!(
                    error.downcast_ref::<RemoteProfileError>(),
                    Some(RemoteProfileError::Disconnected | RemoteProfileError::Stale)
                )) {
                    this.clear(cx);
                    this.notice = "Profile cleared because the remote session changed; import again with consent".into();
                    cx.notify();
                    return;
                }
                // Remote error cleanup also sets cancellation; it must still update the UI.
                if result.is_ok() && this.cancellation.load(Ordering::Relaxed) {
                    this.notice = "Import cancelled; the previous result is preserved".into();
                    cx.notify();
                    return;
                }
                if task_group.as_ref().is_some_and(|group| !group.is_current(&this.project, cx)) {
                    this.notice = "Task profile rejected because its artifacts or session changed".into();
                    cx.notify();
                    return;
                }
                match result {
                    Ok((result, tree, flame_rows, result_paths, path_groups)) => {
                        let project = this.project.read(cx);
                        let permitted = profile_paths_permitted(project, &path_groups, cx);
                        if !permitted { this.notice = "Profile import rejected because project visibility or trust changed".to_string(); cx.notify(); return; }
                        this.zoom_generation = this.zoom_generation.wrapping_add(1);
                        this.zoom_cancellation.store(true, Ordering::Relaxed);
                        this.zoom_task = Task::ready(());
                        this.zoom_root = None;
                        this.hovered_bar = None;
                        this.flame_rows = flame_rows;
                        this.notice = "This view only.".to_string();
                        this.task_group = task_group.clone();
                        this.recording_title = Some(title);
                        this.result = Some(result);
                        this.result_paths = result_paths;
                        this.host.replace_snapshot(tree);
                    }
                    Err(error) => { this.notice = format!("Profile import failed: {error}. The previous result is preserved."); }
                }
                #[cfg(test)]
                { this.import_application_time = Some(application_started.elapsed()); }
                cx.notify();
            }).log_err();
        });
        cx.notify();
        Ok(())
    }

    fn selected_node_index(&self) -> Option<usize> {
        let index = self.host.selected()?.0.parse::<usize>().ok()?;
        self.result.as_ref()?.nodes.get(index)?;
        Some(index)
    }

    fn zoom_selected(&mut self, cx: &mut Context<Self>) {
        if let Some(index) = self.selected_node_index() {
            self.flamegraph = true;
            self.zoom(Some(index), cx);
            cx.notify();
        }
    }

    fn zoom(&mut self, root: Option<usize>, cx: &mut Context<Self>) {
        let Some(profile) = self.result.clone() else {
            return;
        };
        self.zoom_generation = self.zoom_generation.wrapping_add(1);
        let generation = self.zoom_generation;
        let input = profile.clone();
        self.zoom_cancellation.store(true, Ordering::Relaxed);
        let cancellation = Arc::new(AtomicBool::new(false));
        self.zoom_cancellation = cancellation.clone();
        let layout = cx.background_spawn(async move {
            profile_flamegraph(&input, root, || cancellation.load(Ordering::Relaxed))
        });
        self.zoom_task = cx.spawn(async move |this, cx| {
            let layout = layout.await;
            this.update(cx, |this, cx| {
                if this.zoom_generation != generation
                    || !this
                        .result
                        .as_ref()
                        .is_some_and(|current| Arc::ptr_eq(current, &profile))
                {
                    return;
                }
                match layout {
                    Ok(rows) => {
                        this.flame_rows = Arc::new(rows);
                        this.zoom_root = root;
                        this.hovered_bar = None;
                        this.reveal_flame_selection();
                    }
                    Err(_) => {
                        this.notice = "Cannot zoom this profile".to_string();
                    }
                }
                cx.notify();
            })
            .log_err();
        });
    }

    fn select_node(&mut self, index: usize) {
        self.expand_node_ancestors(index);
        self.host.select(LanguageToolNodeId(index.to_string()));
    }

    fn expand_node_ancestors(&mut self, index: usize) {
        let mut parent = self
            .result
            .as_ref()
            .and_then(|result| result.nodes.get(index))
            .and_then(|node| node.parent);
        let mut ancestors = Vec::new();
        while let Some(index) = parent {
            ancestors.push(LanguageToolNodeId(index.to_string()));
            parent = self
                .result
                .as_ref()
                .and_then(|result| result.nodes.get(index))
                .and_then(|node| node.parent);
        }
        self.host.expand_nodes(ancestors);
    }

    fn open_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.project.read(cx).is_disconnected(cx)
            || self
                .task_group
                .as_ref()
                .is_some_and(|group| !group.is_current(&self.project, cx))
        {
            return;
        }
        let source = self
            .host
            .selected()
            .and_then(|id| id.0.parse::<usize>().ok())
            .and_then(|index| self.result.as_ref()?.nodes.get(index))
            .and_then(|node| self.result.as_ref()?.frames.get(node.frame))
            .and_then(|frame| frame.source.clone());
        let Some((path, line)) = source else {
            return;
        };
        let project = self.project.read(cx);
        if project.entry_for_path(&path, cx).is_none_or(|entry| {
            !entry.is_file() || entry.is_private || entry.is_fifo || entry.is_external
        }) || !project
            .visible_worktrees(cx)
            .any(|worktree| worktree.read(cx).id() == path.worktree_id)
            || project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                &project.worktree_store(),
                cx,
            )
        {
            return;
        }
        if let Some(workspace) = self.workspace.upgrade() {
            let opening = workspace.update(cx, |workspace, cx| {
                workspace.open_path(path, None, true, window, cx)
            });
            cx.spawn_in(window, async move |_this, cx| {
                if let Some(editor) = opening.await?.downcast::<editor::Editor>() {
                    editor.update_in(cx, |editor, window, cx| {
                        let snapshot = editor.buffer().read(cx).snapshot(cx);
                        let point = snapshot.clip_point(
                            text::Point::new(line.saturating_sub(1), 0),
                            text::Bias::Left,
                        );
                        editor.change_selections(
                            editor::SelectionEffects::default(),
                            window,
                            cx,
                            |selections| selections.select_ranges([point..point]),
                        );
                    })?;
                }
                anyhow::Ok(())
            })
            .detach_and_log_err(cx);
        }
    }
}

fn profile_tree(
    profile: &ProfileResults,
    mut cancelled: impl FnMut() -> bool,
) -> Result<LanguageToolSnapshot> {
    ensure!(!cancelled(), "Profile projection was cancelled");
    let mut built = vec![None; profile.nodes.len()];
    for (index, node) in profile.nodes.iter().enumerate().rev() {
        ensure!(!cancelled(), "Profile projection was cancelled");
        let frame = profile
            .frames
            .get(node.frame)
            .context("Invalid profile frame")?;
        let children = node
            .children
            .iter()
            .map(|child| {
                ensure!(!cancelled(), "Profile projection was cancelled");
                built
                    .get_mut(*child)
                    .and_then(Option::take)
                    .context("Invalid profile child")
            })
            .collect::<Result<Vec<_>>>()?;
        let projected = LanguageToolNode {
            id: LanguageToolNodeId(index.to_string()),
            label: frame.label.clone(),
            secondary_label: None,
            icon: None,
            accessibility_label: format!(
                "{}, {} inclusive samples, {} of recording samples, {} self samples{}",
                frame.label,
                node.inclusive_samples,
                sample_share(node.inclusive_samples, profile.total_samples),
                node.self_samples,
                if frame.source.is_none() {
                    ", source unavailable"
                } else {
                    ""
                }
            ),
            children,
            enabled: true,
            activation_label: frame.source.as_ref().map(|_| "Open source".to_string()),
        };
        *built.get_mut(index).context("Invalid profile node")? = Some(projected);
    }
    let roots = profile
        .roots
        .iter()
        .map(|root| {
            ensure!(!cancelled(), "Profile projection was cancelled");
            built
                .get_mut(*root)
                .and_then(Option::take)
                .context("Invalid profile root")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LanguageToolSnapshot {
        roots,
        status: if profile.total_samples == 0 {
            LanguageToolProviderStatus::Empty("No samples in this profile".to_string())
        } else if profile.unresolved_frames > 0 || profile.unattributed_samples > 0 {
            LanguageToolProviderStatus::Partial(
                "Some frames are unresolved or samples have no stack".to_string(),
            )
        } else {
            LanguageToolProviderStatus::Current
        },
    })
}

fn profile_flamegraph(
    profile: &ProfileResults,
    root: Option<usize>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Vec<Vec<FlameBar>>> {
    ensure!(!cancelled(), "Profile projection was cancelled");
    let mut rows: Vec<Vec<FlameBar>> = Vec::new();
    let mut pending = Vec::new();
    let total = match root {
        Some(root) => {
            profile
                .nodes
                .get(root)
                .context("Invalid zoom root")?
                .inclusive_samples
        }
        None => profile.total_samples,
    }
    .max(1) as f64;
    let mut offset = 0u64;
    let roots = root
        .map(|root| vec![root])
        .unwrap_or_else(|| profile.roots.clone());
    for root in &roots {
        ensure!(!cancelled(), "Profile projection was cancelled");
        pending.push((*root, 0usize, offset));
        offset = offset
            .checked_add(
                profile
                    .nodes
                    .get(*root)
                    .context("Invalid flamegraph root")?
                    .inclusive_samples,
            )
            .context("Flamegraph weight overflow")?;
    }
    while let Some((index, depth, start)) = pending.pop() {
        ensure!(!cancelled(), "Profile projection was cancelled");
        let node = profile
            .nodes
            .get(index)
            .context("Invalid flamegraph node")?;
        ensure!(depth < 256, "Flamegraph depth limit exceeded");
        if rows.len() <= depth {
            rows.resize_with(depth + 1, Vec::new);
        }
        rows.get_mut(depth)
            .context("Invalid flamegraph depth")?
            .push(FlameBar {
                node: index,
                left: (start as f64 / total) as f32,
                width: (node.inclusive_samples as f64 / total) as f32,
            });
        let mut offset = start;
        for child in &node.children {
            ensure!(!cancelled(), "Profile projection was cancelled");
            pending.push((*child, depth + 1, offset));
            offset = offset
                .checked_add(
                    profile
                        .nodes
                        .get(*child)
                        .context("Invalid flamegraph child")?
                        .inclusive_samples,
                )
                .context("Flamegraph weight overflow")?;
        }
    }
    for row in &mut rows {
        ensure!(!cancelled(), "Profile projection was cancelled");
        row.sort_by(|left, right| left.left.total_cmp(&right.left));
    }
    ensure!(!cancelled(), "Profile projection was cancelled");
    Ok(rows)
}

impl Drop for ProfileView {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Relaxed);
        self.zoom_cancellation.store(true, Ordering::Relaxed);
    }
}
impl EventEmitter<()> for ProfileView {}
impl Focusable for ProfileView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}
impl Item for ProfileView {
    fn restart_blocker(&self, _: &App) -> Option<&'static str> {
        (self.importing || self.selecting)
            .then_some("Resolve the active profile collection/import before restarting")
    }

    type Event = ();
    fn to_item_events(_: &(), _: &mut dyn FnMut(workspace::item::ItemEvent)) {}
    fn tab_content_text(&self, _: usize, _: &App) -> SharedString {
        "Native Profile".into()
    }
    fn telemetry_event_text(&self) -> Option<&'static str> {
        None
    }
}

impl ProfileView {
    fn source_available(&self) -> bool {
        self.selected_node_index()
            .and_then(|index| self.result.as_ref()?.nodes.get(index))
            .and_then(|node| self.result.as_ref()?.frames.get(node.frame))
            .is_some_and(|frame| frame.source.is_some())
    }

    fn set_flamegraph(&mut self, flamegraph: bool, cx: &mut Context<Self>) {
        self.flamegraph = flamegraph;
        self.hovered_bar = None;
        if flamegraph {
            if self
                .selected_node_index()
                .is_some_and(|index| self.flame_depth(index).is_none())
            {
                self.zoom(None, cx);
            } else {
                self.reveal_flame_selection();
            }
        } else {
            self.host.reveal_selection(gpui::ScrollStrategy::Center);
        }
        cx.notify();
    }

    fn activate_flame_bar(
        &mut self,
        index: usize,
        click_count: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.select_node(index);
        self.hovered_bar = None;
        window.focus(&self.focus, cx);
        if click_count > 1 {
            self.zoom_selected(cx);
        }
        cx.notify();
    }

    fn displayed_samples(&self) -> u64 {
        let Some(profile) = &self.result else {
            return 0;
        };
        self.zoom_root
            .and_then(|index| profile.nodes.get(index))
            .map_or(profile.total_samples, |node| node.inclusive_samples)
    }

    fn status_counts(&self) -> Vec<(String, Color)> {
        let Some(profile) = &self.result else {
            return Vec::new();
        };
        let mut counts = vec![(sample_count(profile.total_samples), Color::Default)];
        if self.flamegraph && self.zoom_root.is_some() {
            counts.push((
                format!("{} in zoom", self.displayed_samples()),
                Color::Muted,
            ));
        }
        match profile.unresolved_frames {
            0 => {}
            1 => counts.push(("1 unresolved frame".into(), Color::Warning)),
            count => counts.push((format!("{count} unresolved frames"), Color::Warning)),
        }
        match profile.unattributed_samples {
            0 => {}
            1 => counts.push(("1 sample without a stack".into(), Color::Warning)),
            count => counts.push((format!("{count} samples without stacks"), Color::Warning)),
        }
        counts
    }

    fn accessible_status(&self, counts: &[(String, Color)]) -> String {
        counts
            .iter()
            .map(|(label, _)| label.as_str())
            .chain(std::iter::once(self.notice.as_str()))
            .collect::<Vec<_>>()
            .join(". ")
    }

    fn compact_toolbar(&self) -> bool {
        self.pane_width < px(700.0)
    }

    fn flame_depth(&self, index: usize) -> Option<usize> {
        let profile = self.result.as_ref()?;
        let mut depth = 0;
        let mut current = index;
        while Some(current) != self.zoom_root {
            match profile.nodes.get(current)?.parent {
                Some(parent) => current = parent,
                None => return self.zoom_root.is_none().then_some(depth),
            }
            depth += 1;
        }
        Some(depth)
    }

    fn reveal_flame_selection(&self) {
        if let Some(depth) = self
            .selected_node_index()
            .and_then(|index| self.flame_depth(index))
        {
            self.flame_scroll
                .scroll_to_item(depth, gpui::ScrollStrategy::Center);
        }
    }

    fn navigate_flamegraph(&mut self, key: &str) {
        let Some(profile) = &self.result else {
            return;
        };
        let first = self
            .flame_rows
            .first()
            .and_then(|row| row.first())
            .map(|bar| bar.node);
        let next = match (
            key,
            self.selected_node_index()
                .filter(|index| self.flame_depth(*index).is_some()),
        ) {
            ("home", _) | (_, None) => first,
            ("end", _) => self
                .flame_rows
                .last()
                .and_then(|row| row.last())
                .map(|bar| bar.node),
            (_, Some(index)) => {
                let Some(node) = profile.nodes.get(index) else {
                    return;
                };
                match key {
                    "up" => node
                        .parent
                        .filter(|&parent| self.flame_depth(parent).is_some()),
                    "down" => node.children.first().copied(),
                    "left" | "right" if Some(index) != self.zoom_root => {
                        let siblings = node
                            .parent
                            .and_then(|parent| profile.nodes.get(parent))
                            .map_or(&profile.roots, |parent| &parent.children);
                        let position = siblings.iter().position(|&sibling| sibling == index);
                        let position = position.and_then(|position| {
                            if key == "left" {
                                position.checked_sub(1)
                            } else {
                                Some(position + 1)
                            }
                        });
                        position
                            .and_then(|position| siblings.get(position))
                            .copied()
                    }
                    _ => None,
                }
            }
        };
        self.hovered_bar = None;
        if let Some(next) = next {
            self.select_node(next);
            self.reveal_flame_selection();
        }
    }

    fn move_control_focus(&self, forward: bool, window: &mut Window, cx: &mut App) {
        let compact = self.compact_toolbar();
        let enabled = [
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            true,
            true,
            self.flamegraph && self.selected_node_index().is_some(),
            self.flamegraph && self.zoom_root.is_some(),
            self.source_available(),
            self.importing,
            !compact,
            !compact && !self.flamegraph,
            compact,
        ];
        let controls = std::iter::once(&self.focus)
            .chain(
                self.toolbar_focus
                    .iter()
                    .zip(enabled)
                    .filter_map(|(handle, enabled)| enabled.then_some(handle)),
            )
            .collect::<Vec<_>>();
        let current = controls.iter().position(|handle| handle.is_focused(window));
        let next = current.and_then(|index| {
            if forward {
                controls.get(index + 1)
            } else {
                index.checked_sub(1).and_then(|index| controls.get(index))
            }
        });
        // Workspace toolbar groups can share tab indexes with an item's group.
        // Resolve this item's sequence before using window navigation at its edges.
        if let Some(handle) = next {
            window.focus(handle, cx);
        } else if forward {
            window.focus_next(cx);
        } else {
            window.focus_prev(cx);
        }
    }
}

fn sample_count(samples: u64) -> String {
    if samples == 1 {
        "1 sample".into()
    } else {
        format!("{samples} samples")
    }
}

fn recording_title(paths: &[ProjectPath]) -> SharedString {
    paths
        .first()
        .and_then(|path| path.path.file_stem())
        .map(|stem| {
            stem.chars()
                .map(|character| {
                    if character.is_control() {
                        ' '
                    } else {
                        character
                    }
                })
                .collect::<String>()
        })
        .filter(|title| !title.trim().is_empty())
        .map_or_else(|| "Native recording".into(), Into::into)
}

fn sample_share(samples: u64, total: u64) -> String {
    let percentage = if total == 0 {
        0.0
    } else {
        100.0 * samples as f64 / total as f64
    };
    format!("{percentage:.1}%")
}

fn frame_sample_summary(samples: u64, total: u64) -> String {
    format!(
        "{} · {} of displayed samples",
        sample_count(samples),
        sample_share(samples, total)
    )
}

#[derive(Clone, Copy)]
struct SampleColumns {
    count_width: gpui::Pixels,
    share_width: gpui::Pixels,
}

impl SampleColumns {
    fn render(
        self,
        count: impl Into<SharedString>,
        share: impl Into<SharedString>,
    ) -> impl IntoElement {
        h_flex()
            .flex_none()
            .gap_2()
            .pr_2()
            .child(
                h_flex()
                    .w(self.count_width)
                    .justify_end()
                    .overflow_hidden()
                    .child(Label::new(count).truncate().color(Color::Muted)),
            )
            .child(
                h_flex()
                    .w(self.share_width)
                    .justify_end()
                    .overflow_hidden()
                    .child(Label::new(share).truncate().color(Color::Muted)),
            )
    }
}

fn profile_tab(
    flamegraph: bool,
    active: bool,
    focus: &FocusHandle,
    other_tab: &FocusHandle,
    cx: &mut Context<ProfileView>,
) -> Tab {
    let (id, title, position, tab_index) = if flamegraph {
        ("flame-graph", "Flame Graph", TabPosition::First, 2)
    } else {
        ("call-tree", "Call Tree", TabPosition::Last, 3)
    };
    let focus = focus.clone().tab_index(tab_index).tab_stop(true);
    let clicked_focus = focus.clone();
    let other_tab = other_tab.clone();
    Tab::new(id)
        .position(position)
        .toggle_state(active)
        .track_focus(&focus)
        .role(gpui::Role::Tab)
        .aria_label(title)
        .aria_selected(active)
        .focus(|style| style.border_color(cx.theme().colors().border_focused))
        .on_click(cx.listener(move |view, _, window, cx| {
            view.set_flamegraph(flamegraph, cx);
            window.focus(&clicked_focus, cx);
        }))
        .on_key_down(
            cx.listener(move |view, event: &gpui::KeyDownEvent, window, cx| {
                match event.keystroke.key.as_str() {
                    "space" | "enter" => view.set_flamegraph(flamegraph, cx),
                    "left" | "right" => {
                        view.set_flamegraph(!flamegraph, cx);
                        window.focus(&other_tab, cx);
                    }
                    _ => return,
                }
                cx.stop_propagation();
            }),
        )
        .child(title)
        .when(active, |tab| {
            tab.child(
                div()
                    .absolute()
                    .bottom_0()
                    .left_0()
                    .right_0()
                    .h(px(2.0))
                    .bg(cx.theme().colors().text_accent),
            )
        })
}

impl Render for ProfileView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let [
            import_focus,
            flame_focus,
            tree_focus,
            zoom_focus,
            reset_focus,
            source_focus,
            cancel_focus,
            clear_focus,
            expand_focus,
            overflow_focus,
        ] = &self.toolbar_focus;
        let source_available = self.source_available();
        let compact = self.compact_toolbar();
        let displayed_samples = self.displayed_samples();
        let selected = self.selected_node_index();
        let click = cx.entity().downgrade();
        let toggle = click.clone();
        let measure = click.clone();
        let pane_width = self.pane_width;
        let columns = SampleColumns {
            count_width: if compact { px(64.0) } else { px(96.0) },
            share_width: if compact { px(56.0) } else { px(80.0) },
        };
        let content = if self.flamegraph {
            let rows = self.flame_rows.clone();
            let result = self.result.clone();
            let view = cx.entity().downgrade();
            let hovered = self.hovered_bar.map(|(_, index)| index);
            let label_width = (48.0 / f32::from(pane_width).max(1.0)).min(1.0);
            #[cfg(test)]
            let painted_bars = self.painted_flame_bars.clone();
            let row_count = rows.len();
            let render_rows = move |range: std::ops::Range<usize>, _: &mut Window, cx: &mut App| {
                let colors = cx.theme().colors();
                let status = cx.theme().status();
                let base = colors.editor_background;
                let palette = [
                    status.info,
                    status.modified,
                    status.created,
                    colors.text_accent,
                ]
                .map(|accent| base.blend(accent.opacity(0.32)));
                let selection = palette[0].blend(colors.text.opacity(0.18));
                let focus_outline = colors.text;
                range
                    .filter_map(|depth| rows.get(depth).map(|bars| (depth, bars)))
                    .map(|(depth, bars)| {
                        let measured_bounds = Rc::new(Cell::new(None));
                        let paint_bounds = measured_bounds.clone();
                        let paint_rows = rows.clone();
                        let hit_rows = rows.clone();
                        let hit_test = move |position: gpui::Pixels| {
                            let bounds: gpui::Bounds<gpui::Pixels> = measured_bounds.get()?;
                            let normalized = f32::from(position - bounds.origin.x)
                                / f32::from(bounds.size.width);
                            flamegraph_bar_at(hit_rows.get(depth)?, normalized).map(|bar| bar.node)
                        };
                        let click_hit = hit_test.clone();
                        let click_view = view.clone();
                        let hover_view = view.clone();
                        let leave_view = view.clone();
                        #[cfg(test)]
                        let painted_bars = painted_bars.clone();
                        div()
                            .relative()
                            .h_7()
                            .w_full()
                            .id(("native-flamegraph-row", depth))
                            .debug_selector(move || format!("native-flamegraph-row-{depth}"))
                            .role(gpui::Role::Group)
                            .aria_label("Flame graph row")
                            .child(
                                gpui::canvas(
                                    move |bounds, _, _| paint_bounds.set(Some(bounds)),
                                    move |bounds, _, window, _| {
                                        let Some(bars) = paint_rows.get(depth) else {
                                            return;
                                        };
                                        let mut pending: Option<gpui::Bounds<gpui::Pixels>> = None;
                                        for (position, bar) in
                                            bars.iter().enumerate().filter(|(_, bar)| {
                                                bar.width > 0.0 && bar.width < label_width
                                            })
                                        {
                                            let mut bar_bounds = gpui::Bounds::new(
                                                bounds.origin
                                                    + gpui::point(
                                                        bounds.size.width * bar.left,
                                                        px(1.0),
                                                    ),
                                                gpui::size(
                                                    bounds.size.width * bar.width,
                                                    bounds.size.height - px(2.0),
                                                ),
                                            );
                                            let is_selected = selected == Some(bar.node);
                                            let is_hovered = hovered == Some(bar.node);
                                            // Batch adjacent subpixel bars to bound paint work on very wide stacks.
                                            let batch = bar_bounds.size.width < px(1.0)
                                                && !is_selected
                                                && !is_hovered;
                                            if batch
                                                && let Some(run) = &mut pending
                                                && (run.right() - bar_bounds.left()).abs()
                                                    <= px(0.01)
                                            {
                                                run.size.width = bar_bounds.right() - run.left();
                                            } else {
                                                if let Some(run) = pending.take() {
                                                    window.paint_quad(gpui::fill(run, palette[0]));
                                                }
                                                if batch {
                                                    pending = Some(bar_bounds);
                                                } else {
                                                    if bar_bounds.size.width >= px(4.0) {
                                                        bar_bounds.size.width -= px(1.0);
                                                    }
                                                    let background = if is_selected {
                                                        selection
                                                    } else {
                                                        palette[position % palette.len()]
                                                    };
                                                    let background = if is_hovered {
                                                        background
                                                            .blend(focus_outline.opacity(0.10))
                                                    } else {
                                                        background
                                                    };
                                                    window.paint_quad(gpui::fill(
                                                        bar_bounds, background,
                                                    ));
                                                }
                                            }
                                            #[cfg(test)]
                                            painted_bars.fetch_add(1, Ordering::Relaxed);
                                        }
                                        if let Some(run) = pending {
                                            window.paint_quad(gpui::fill(run, palette[0]));
                                        }
                                    },
                                )
                                .absolute()
                                .size_full(),
                            )
                            .on_mouse_move(move |event, _, cx| {
                                let hovered =
                                    hit_test(event.position.x).map(|index| (depth, index));
                                hover_view
                                    .update(cx, |this, cx| {
                                        if this.hovered_bar != hovered {
                                            this.hovered_bar = hovered;
                                            cx.notify();
                                        }
                                    })
                                    .log_err();
                            })
                            .on_hover(move |inside, _, cx| {
                                if !*inside {
                                    leave_view
                                        .update(cx, |this, cx| {
                                            if this.hovered_bar.is_some_and(|(row, _)| row == depth)
                                            {
                                                this.hovered_bar = None;
                                                cx.notify();
                                            }
                                        })
                                        .log_err();
                                }
                            })
                            .on_click(move |event, window, cx| {
                                if let Some(index) = click_hit(event.position().x) {
                                    click_view
                                        .update(cx, |this, cx| {
                                            this.activate_flame_bar(
                                                index,
                                                event.click_count(),
                                                window,
                                                cx,
                                            );
                                        })
                                        .log_err();
                                }
                            })
                            .children(
                                bars.iter()
                                    .enumerate()
                                    .filter(|(_, bar)| bar.width >= label_width)
                                    .filter_map(|(position, bar)| {
                                        let result = result.as_ref()?;
                                        let node = result.nodes.get(bar.node)?;
                                        let frame = result.frames.get(node.frame)?;
                                        let view = view.clone();
                                        let index = bar.node;
                                        let is_selected = selected == Some(index);
                                        let background = if is_selected {
                                            selection
                                        } else {
                                            palette[position % palette.len()]
                                        };
                                        let background = if hovered == Some(index) {
                                            background.blend(focus_outline.opacity(0.10))
                                        } else {
                                            background
                                        };
                                        let summary = frame_sample_summary(
                                            node.inclusive_samples,
                                            displayed_samples,
                                        );
                                        Some(
                                            div()
                                                .id(index)
                                                .absolute()
                                                .left(relative(bar.left))
                                                .w(relative(bar.width))
                                                .h_full()
                                                .py(px(1.0))
                                                .pr(px(1.0))
                                                .overflow_hidden()
                                                .role(gpui::Role::Button)
                                                .aria_selected(is_selected)
                                                .aria_label(format!("{}, {summary}", frame.label))
                                                .when(is_selected, |bar| {
                                                    bar.aria_active_descendant()
                                                })
                                                .child(
                                                    h_flex()
                                                        .size_full()
                                                        .overflow_hidden()
                                                        .rounded_sm()
                                                        .border_1()
                                                        .border_color(if is_selected {
                                                            focus_outline
                                                        } else {
                                                            gpui::transparent_black()
                                                        })
                                                        .bg(background)
                                                        .px_2()
                                                        .child(
                                                            div()
                                                                .flex_1()
                                                                .min_w_0()
                                                                .truncate()
                                                                .child(frame.label.clone()),
                                                        ),
                                                )
                                                .on_click(move |event, window, cx| {
                                                    cx.stop_propagation();
                                                    view.update(cx, |this, cx| {
                                                        this.activate_flame_bar(
                                                            index,
                                                            event.click_count(),
                                                            window,
                                                            cx,
                                                        );
                                                    })
                                                    .log_err();
                                                }),
                                        )
                                    }),
                            )
                    })
                    .collect::<Vec<_>>()
            };
            uniform_list("native-flamegraph", row_count, render_rows)
                .track_scroll(&self.flame_scroll)
                .flex_1()
                .into_any_element()
        } else {
            let result = self.result.clone();
            let show_columns = result.is_some();
            v_flex()
                .flex_1()
                .min_h_0()
                .min_w_0()
                .when(show_columns, |tree| {
                    tree.child(
                        h_flex()
                            .flex_none()
                            .pl_2()
                            .h_7()
                            .border_b_1()
                            .border_color(cx.theme().colors().border)
                            .child(
                                div()
                                    .flex_1()
                                    .child(Label::new("Frame").color(Color::Muted)),
                            )
                            .child(columns.render("Samples", "Share")),
                    )
                })
                .child(language_tool_tree::language_tool_tree(
                    self.host.visible_rows_snapshot(),
                    self.host.selected().cloned(),
                    LanguageToolTreeOptions::with_row_end(pane_width * 0.32, move |row| {
                        let result = result.as_ref()?;
                        let node = result.nodes.get(row.id.0.parse::<usize>().ok()?)?;
                        Some(
                            columns
                                .render(
                                    node.inclusive_samples.to_string(),
                                    sample_share(node.inclusive_samples, result.total_samples),
                                )
                                .into_any_element(),
                        )
                    }),
                    self.host.scroll_handle().clone(),
                    move |id, count, window, cx| {
                        click
                            .update(cx, |this, cx| {
                                this.host.select(id);
                                window.focus(&this.focus, cx);
                                if count > 1 {
                                    this.open_selected(window, cx);
                                }
                                cx.notify();
                            })
                            .log_err();
                    },
                    move |id, _, cx| {
                        toggle
                            .update(cx, |this, cx| {
                                this.host.toggle(&id);
                                cx.notify();
                            })
                            .log_err();
                    },
                    |_, _, _, _| {},
                ))
                .into_any_element()
        };
        let frame_details = (|| {
            let index = self.hovered_bar.map(|(_, index)| index).or(selected)?;
            let result = self.result.as_ref()?;
            let node = result.nodes.get(index)?;
            let frame = result.frames.get(node.frame)?;
            let summary = if self.flamegraph {
                self.flame_depth(index)?;
                frame_sample_summary(node.inclusive_samples, displayed_samples)
            } else {
                format!(
                    "{} inclusive · {} of recording samples · {} self",
                    sample_count(node.inclusive_samples),
                    sample_share(node.inclusive_samples, result.total_samples),
                    sample_count(node.self_samples)
                )
            };
            Some((frame.label.clone(), summary))
        })();
        let overflow_view = cx.entity().downgrade();
        let counts = self.status_counts();
        let status = self.accessible_status(&counts);
        let actions = h_flex()
            .id("native-profile-actions")
            .role(gpui::Role::Toolbar)
            .aria_label("Profile view actions")
            .flex_wrap()
            .when_else(
                compact,
                |toolbar| toolbar.gap_x_1().gap_y_2(),
                |toolbar| toolbar.gap_2(),
            )
            .child(
                h_flex()
                    .gap_1()
                    .when(self.flamegraph, |actions| {
                        actions
                            .child(
                                Button::new(
                                    "zoom-selected",
                                    if compact { "Zoom" } else { "Zoom Selected" },
                                )
                                .aria_label("Zoom Selected")
                                .tooltip(Tooltip::text("Zoom Selected"))
                                .track_focus(
                                    &zoom_focus.clone().tab_index(4).tab_stop(selected.is_some()),
                                )
                                .disabled(selected.is_none())
                                .on_click(cx.listener(|this, _, _, cx| this.zoom_selected(cx))),
                            )
                            .child(
                                Button::new(
                                    "reset-zoom",
                                    if compact { "Reset" } else { "Reset Zoom" },
                                )
                                .aria_label("Reset Zoom")
                                .tooltip(Tooltip::text("Reset Zoom"))
                                .track_focus(
                                    &reset_focus
                                        .clone()
                                        .tab_index(5)
                                        .tab_stop(self.zoom_root.is_some()),
                                )
                                .disabled(self.zoom_root.is_none())
                                .on_click(cx.listener(|this, _, _, cx| this.zoom(None, cx))),
                            )
                    })
                    .child(
                        Button::new(
                            "open-source",
                            if compact { "Source" } else { "Open Source" },
                        )
                        .aria_label("Open Source")
                        .tooltip(Tooltip::text("Open Source"))
                        .track_focus(&source_focus.clone().tab_index(6).tab_stop(source_available))
                        .disabled(!source_available)
                        .on_click(
                            cx.listener(|this, _, window, cx| this.open_selected(window, cx)),
                        ),
                    ),
            )
            .when(self.importing, |toolbar| {
                toolbar.child(
                    Button::new("cancel-import", "Cancel Import")
                        .track_focus(&cancel_focus.clone().tab_index(7).tab_stop(true))
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.cancel_import(cx);
                            window.focus(&this.toolbar_focus[0], cx);
                        })),
                )
            })
            .when(!compact, |toolbar| {
                toolbar
                    .child(
                        Button::new("clear", "Clear")
                            .track_focus(&clear_focus.clone().tab_index(8).tab_stop(true))
                            .on_click(cx.listener(|this, _, _, cx| this.clear(cx))),
                    )
                    .when(!self.flamegraph, |toolbar| {
                        toolbar.child(
                            Button::new("expand", "Expand All")
                                .track_focus(&expand_focus.clone().tab_index(9).tab_stop(true))
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.host.expand_all();
                                    cx.notify();
                                })),
                        )
                    })
            })
            .when(compact, |toolbar| {
                let call_tree = !self.flamegraph;
                toolbar.child(
                    PopoverMenu::new("profile-actions")
                        .anchor(gpui::Anchor::TopRight)
                        .trigger_with_tooltip(
                            IconButton::new("profile-actions-trigger", IconName::Ellipsis)
                                .aria_label("More profile actions")
                                .track_focus(&overflow_focus.clone().tab_index(10).tab_stop(true)),
                            Tooltip::text("More profile actions"),
                        )
                        .menu(move |window, cx| {
                            let clear = overflow_view.clone();
                            let expand = overflow_view.clone();
                            Some(ContextMenu::build(window, cx, move |menu, _, _| {
                                let menu = menu.entry("Clear", None, move |_, cx| {
                                    clear.update(cx, |this, cx| this.clear(cx)).log_err();
                                });
                                if call_tree {
                                    menu.entry("Expand All", None, move |_, cx| {
                                        expand
                                            .update(cx, |this, cx| {
                                                this.host.expand_all();
                                                cx.notify();
                                            })
                                            .log_err();
                                    })
                                } else {
                                    menu
                                }
                            }))
                        }),
                )
            });
        let tabs = h_flex()
            .id("native-profile-tabs")
            .role(gpui::Role::TabList)
            .aria_label("Recording views")
            .flex_none()
            .child(profile_tab(
                true,
                self.flamegraph,
                flame_focus,
                tree_focus,
                cx,
            ))
            .child(profile_tab(
                false,
                !self.flamegraph,
                tree_focus,
                flame_focus,
                cx,
            ));
        v_flex()
            .id("native-profile-view")
            .role(gpui::Role::Group)
            .aria_label("Native profile")
            .relative()
            .size_full()
            .tab_group()
            .tab_index(0isize)
            .track_focus(&self.focus)
            .key_context("NativeProfile")
            .on_action(cx.listener(|this, _: &ZoomToSelectedFrame, _, cx| this.zoom_selected(cx)))
            .on_action(cx.listener(|this, _: &ResetProfileZoom, _, cx| this.zoom(None, cx)))
            .on_action(cx.listener(|this, _: &ToggleProfileView, _, cx| {
                this.set_flamegraph(!this.flamegraph, cx);
            }))
            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                let modifiers = event.keystroke.modifiers;
                if event.keystroke.key == "tab"
                    && !modifiers.control
                    && !modifiers.alt
                    && !modifiers.platform
                    && !modifiers.function
                {
                    this.move_control_focus(!modifiers.shift, window, cx);
                    cx.stop_propagation();
                    return;
                }
                if !this.focus.is_focused(window) {
                    return;
                }
                match event.keystroke.key.as_str() {
                    key @ ("up" | "down" | "left" | "right" | "home" | "end")
                        if this.flamegraph =>
                    {
                        this.navigate_flamegraph(key);
                    }
                    "up" => this.host.select_previous(),
                    "down" => this.host.select_next(),
                    "home" => this.host.select_first(),
                    "end" => this.host.select_last(),
                    "left" => this.host.select_parent(),
                    "right" => this.host.select_first_child(),
                    "space" if !this.flamegraph => this.host.toggle_selected(),
                    "enter" => this.open_selected(window, cx),
                    _ => return,
                }
                if !this.flamegraph {
                    this.host.reveal_selection(gpui::ScrollStrategy::Center);
                }
                cx.stop_propagation();
                cx.notify();
            }))
            .child(
                gpui::canvas(
                    move |bounds, window, cx| {
                        if bounds.size.width != pane_width {
                            window.defer(cx, move |_, cx| {
                                measure
                                    .update(cx, |this, cx| {
                                        this.pane_width = bounds.size.width;
                                        cx.notify();
                                    })
                                    .log_err();
                            });
                        }
                    },
                    |_, (), _, _| {},
                )
                .absolute()
                .size_full(),
            )
            .child(
                h_flex()
                    .w_full()
                    .min_w_0()
                    .gap_2()
                    .px_2()
                    .py_2()
                    .child(div().flex_1().min_w_0().when_some(
                        self.recording_title.clone(),
                        |heading, title| {
                            heading.child(
                                div()
                                    .id("native-recording-title")
                                    .role(gpui::Role::Heading)
                                    .aria_label(title.clone())
                                    .tooltip(Tooltip::text(title.clone()))
                                    .text_ui_lg(cx)
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .truncate()
                                    .child(title),
                            )
                        },
                    ))
                    .child(
                        Button::new("import", "Import Profile")
                            .disabled(!cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>())
                            .track_focus(&import_focus.clone().tab_index(1).tab_stop(true))
                            .style(ButtonStyle::Filled)
                            .on_click(cx.listener(|this, _, window, cx| this.prompt(window, cx))),
                    ),
            )
            .child(
                v_flex()
                    .id("native-profile-status")
                    .role(gpui::Role::Status)
                    .aria_label("Native profile status")
                    .aria_value(status)
                    .aria_live(gpui::accesskit::Live::Polite)
                    .px_2()
                    .pb_2()
                    .gap_1()
                    .when(!counts.is_empty(), |status| {
                        status.child(
                            h_flex().flex_wrap().gap_x_3().gap_y_1().children(
                                counts
                                    .into_iter()
                                    .enumerate()
                                    .map(|(index, (label, color))| {
                                        let label = Label::new(label).color(color);
                                        if index == 0 {
                                            label.weight(gpui::FontWeight::SEMIBOLD)
                                        } else {
                                            label
                                        }
                                    }),
                            ),
                        )
                    })
                    .child(Label::new(self.notice.clone()).color(Color::Muted)),
            )
            .child(
                h_flex()
                    .w_full()
                    .flex_wrap()
                    .justify_between()
                    .gap_1()
                    .pr_2()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .bg(cx.theme().colors().tab_bar_background)
                    .child(tabs)
                    .child(actions),
            )
            .child(content)
            .when(
                self.result.is_some() && (self.flamegraph || selected.is_some()),
                |view| {
                    view.child(
                        v_flex()
                            .w_full()
                            .px_2()
                            .py_1()
                            .border_t_1()
                            .border_color(cx.theme().colors().border)
                            .map(|details| match frame_details {
                                Some((label, summary)) => details
                                    .child(div().child(label))
                                    .child(Label::new(summary).color(Color::Muted)),
                                None => details.child(
                                    Label::new("Select a frame or use the arrow keys to explore.")
                                        .color(Color::Muted),
                                ),
                            }),
                    )
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use gpui::UpdateGlobal as _;

    fn set_native_profile_opt_in_for_test(enabled: bool, cx: &mut App) {
        settings::SettingsStore::update_global(cx, |store, _| {
            store.register_setting::<feature_flags::FeatureFlagsSettings>()
        });
        feature_flags::FeatureFlagStore::init(cx);
        settings::SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |content| {
                content.feature_flags.get_or_insert_default().insert(
                    "rust-native-profiling-experimental".into(),
                    if enabled { "on" } else { "off" }.into(),
                );
            })
        });
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_profile_opt_in_blocks_direct_import_and_retires_held_selection(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            settings::SettingsStore::update_global(cx, |store, _| {
                store.register_setting::<feature_flags::FeatureFlagsSettings>()
            });
            feature_flags::FeatureFlagStore::init(cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
            cx.update_flags(true, vec!["rust-native-profiling-experimental".into()]);
        });
        let project = Project::test(project::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let view = cx.new(|cx| ProfileView::new(project, workspace.downgrade(), cx));
        view.update_in(cx, |view, window, cx| {
            view.prompt(window, cx);
            assert!(!view.selecting);
            assert!(
                view.import_paths_with_consent(Vec::new(), true, cx)
                    .is_err(),
                "direct import cannot grant disabled permission"
            );
        });
        cx.update(|_, cx| set_native_profile_opt_in_for_test(true, cx));
        view.update_in(cx, |view, window, cx| view.prompt(window, cx));
        cx.run_until_parked();
        let (generation, cancellation) = view.read_with(cx, |view, _| {
            assert!(view.selecting);
            (view.generation, view.cancellation.clone())
        });
        cx.update(|_, cx| set_native_profile_opt_in_for_test(false, cx));
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(!view.selecting && !view.importing);
            assert!(view.generation > generation);
            assert!(cancellation.load(Ordering::Relaxed));
            assert!(view.result.is_none());
        });
        view.update_in(cx, |view, window, cx| {
            view.prompt(window, cx);
            assert!(!view.selecting);
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    use super::*;
    use project::profile_results::parse_samply_profile;

    const PROFILE: &[u8] = include_bytes!(
        "../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"
    );
    const SYMBOLS: &[u8] = include_bytes!(
        "../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json"
    );

    #[test]
    fn profile_view_projection_stops_after_work_has_started() {
        let profile = large_profile();
        let mut checkpoints = 0;
        let error = profile_tree(&profile, || {
            checkpoints += 1;
            checkpoints > 64
        })
        .expect_err("tree projection must stop midway");
        assert_eq!(error.to_string(), "Profile projection was cancelled");
        assert_eq!(checkpoints, 65);
        let mut checkpoints = 0;
        let error = profile_flamegraph(&profile, None, || {
            checkpoints += 1;
            checkpoints > 512
        })
        .expect_err("flamegraph projection must stop midway");
        assert_eq!(error.to_string(), "Profile projection was cancelled");
        assert_eq!(checkpoints, 513);
        assert!(profile_tree(&profile, || true).is_err());
        assert!(profile_flamegraph(&profile, None, || true).is_err());
    }

    #[test]
    fn profile_view_projects_call_tree_and_weighted_flamegraph() {
        let profile = parse_samply_profile(
            PROFILE,
            SYMBOLS,
            ProfileHost::MacOs,
            &AtomicBool::new(false),
            |_, _| None,
        )
        .expect("capture parses");
        let tree = profile_tree(&profile, || false).expect("call tree projects");
        assert_eq!(tree.roots.len(), profile.roots.len());
        let rows = profile_flamegraph(&profile, None, || false).expect("flamegraph projects");
        assert!(rows.len() > 1);
        assert_eq!(
            rows.iter().map(Vec::len).sum::<usize>(),
            profile.nodes.len()
        );
        for row in rows {
            for bar in &row {
                assert!(bar.left >= 0.0 && bar.width >= 0.0 && bar.left + bar.width <= 1.00001);
            }
            for pair in row.windows(2) {
                if let [left, right] = pair {
                    assert!(left.left + left.width <= right.left + 0.00001);
                }
            }
        }
    }

    fn large_profile() -> ProfileResults {
        use project::profile_results::{MAX_PROFILE_NODES, ProfileFrame, ProfileNode};
        let depth = 256;
        let roots = (0..MAX_PROFILE_NODES).step_by(depth).collect::<Vec<_>>();
        ProfileResults {
            frames: vec![ProfileFrame {
                label: "workload::frame".into(),
                source: None,
                resolved: true,
            }],
            nodes: (0..MAX_PROFILE_NODES)
                .map(|index| {
                    let is_root = index % depth == 0;
                    let is_leaf = (index + 1) % depth == 0 || index + 1 == MAX_PROFILE_NODES;
                    ProfileNode {
                        frame: 0,
                        parent: (!is_root).then(|| index - 1),
                        children: if is_leaf { Vec::new() } else { vec![index + 1] },
                        inclusive_samples: 1,
                        self_samples: u64::from(is_leaf),
                    }
                })
                .collect(),
            total_samples: roots.len() as u64,
            roots,
            unresolved_frames: 0,
            unattributed_samples: 0,
        }
    }

    #[test]
    fn profile_view_flamegraph_hit_testing_preserves_narrow_frames_and_gaps() {
        let bars = [
            FlameBar {
                node: 11,
                left: 0.1,
                width: 0.2,
            },
            FlameBar {
                node: 12,
                left: 0.5,
                width: 0.0001,
            },
        ];
        for position in [-0.1, 0.0, 0.4, 0.8, 1.0, f32::NAN] {
            assert!(flamegraph_bar_at(&bars, position).is_none());
        }
        assert_eq!(flamegraph_bar_at(&bars, 0.1).map(|bar| bar.node), Some(11));
        assert_eq!(
            flamegraph_bar_at(&bars, 0.50005).map(|bar| bar.node),
            Some(12)
        );
    }

    #[gpui::test]
    async fn profile_view_path_batches_reject_missing_private_and_directory_entries(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let mut settings = settings::SettingsStore::test(cx);
            settings.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files =
                    Some(settings::ExtendingVec(vec!["**/private.rs".into()]));
            });
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "a.rs":"", "b.rs":"", "c.rs":"", "d.rs":"", "e.rs":"", "f.rs":"",
                "private.rs":"", "directory":{}
            }),
        )
        .await;
        let project = Project::test(fs, ["/fixture".as_ref()], cx).await;
        cx.run_until_parked();
        let retained_groups = project.read_with(cx, |project, cx| {
            let paths = |names: &[&str]| {
                names
                    .iter()
                    .map(|name| {
                        project
                            .find_project_path(
                                std::path::Path::new(&format!("/fixture/{name}")),
                                cx,
                            )
                            .expect("path within worktree")
                    })
                    .collect::<collections::HashSet<_>>()
            };
            let permitted = |paths: &collections::HashSet<ProjectPath>| {
                profile_paths_permitted(project, &profile_path_groups(paths), cx)
            };
            assert!(permitted(&paths(&["a.rs"])));
            let dense = ["a.rs", "b.rs", "c.rs", "d.rs", "e.rs", "f.rs"];
            assert!(permitted(&paths(&dense)));
            for rejected in ["missing.rs", "private.rs", "directory"] {
                assert!(!permitted(&paths(&[rejected])));
                let mut names = dense.to_vec();
                names.push(rejected);
                assert!(!permitted(&paths(&names)));
            }
            let mut foreign = paths(&["a.rs"]);
            let mut foreign_path = foreign.iter().next().expect("fixture path").clone();
            foreign_path.worktree_id = project::WorktreeId::from_usize(999_999);
            foreign.insert(foreign_path);
            assert!(!permitted(&foreign));
            [
                profile_path_groups(&paths(&["a.rs"])),
                profile_path_groups(&paths(&dense)),
            ]
        });
        cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/a.rs".into()]));
                });
            });
        });
        cx.run_until_parked();
        project.read_with(cx, |project, cx| {
            for group in &retained_groups {
                assert!(
                    !profile_paths_permitted(project, group, cx),
                    "prepared paths must use current permissions"
                );
            }
        });
    }

    #[test]
    fn profile_view_large_call_tree_keeps_render_snapshots_bounded() {
        use project::profile_results::MAX_PROFILE_NODES;
        let profile = large_profile();
        let started = std::time::Instant::now();
        let tree = profile_tree(&profile, || false).expect("maximum supported call tree");
        let mut host = LanguageToolTreeHost::default();
        host.replace_snapshot(tree);
        host.expand_all();
        let projection_elapsed = started.elapsed();
        assert_eq!(host.visible_rows().len(), MAX_PROFILE_NODES);
        host.select_last();
        let rendered = host.visible_rows_snapshot();
        let started = std::time::Instant::now();
        for _ in 0..100 {
            let next_frame = host.visible_rows_snapshot();
            assert!(Arc::ptr_eq(&rendered, &next_frame));
            assert!(!host.can_expand_all());
        }
        eprintln!(
            "native 100000-node tree projection/expansion: {projection_elapsed:?}; 100 render snapshot acquisitions: {:?}",
            started.elapsed()
        );
        let final_row = rendered.last().expect("final row");
        assert_eq!(final_row.id.0, (MAX_PROFILE_NODES - 1).to_string());
        assert_eq!(final_row.label, "workload::frame");
        assert_eq!(host.visible_rows_in_range(99_975..100_000).len(), 25);
        host.collapse_all();
        assert_eq!(host.visible_rows().len(), profile.roots.len());
        assert_eq!(rendered.len(), MAX_PROFILE_NODES);
        host.expand_all();
        assert_eq!(host.selected(), Some(&final_row.id));
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn doctor_profile_native_view_uses_sealed_owner_result_and_revokes_private_inputs(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::UpdateGlobal as _;
        use project::Fs as _;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "capture.json": std::str::from_utf8(PROFILE).expect("profile"),
                "capture.syms.json": std::str::from_utf8(SYMBOLS).expect("symbols"),
                "untouched.rs": "original"
            }),
        )
        .await;
        let project = Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let paths = ["/fixture/capture.json", "/fixture/capture.syms.json"].map(|path| {
            project.read_with(cx, |owner, cx| {
                owner
                    .find_project_path(PathBuf::from(path).as_path(), cx)
                    .expect("Selected path")
            })
        });
        let plan = project
            .update(cx, |owner, cx| {
                owner.prepare_diagnostic_profile_import(paths, cx)
            })
            .expect("Sealed selected import");
        let result = project
            .update(cx, |owner, cx| {
                owner.import_diagnostic_profile(plan, Arc::new(AtomicBool::new(false)), cx)
            })
            .expect("Existing owner")
            .await
            .expect("Bounded normalization");
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let view = workspace.update_in(cx, |workspace, window, cx| {
            let view = open_diagnostic_profile_result(workspace, result, window, cx)
                .expect("Native result handoff");
            let focus = view.read(cx).focus.clone();
            window.focus(&focus, cx);
            assert_eq!(
                workspace
                    .active_item(cx)
                    .expect("Actual active item")
                    .item_id(),
                view.entity_id()
            );
            view
        });
        cx.run_until_parked();
        assert!(
            view.read_with(cx, |view, _| view
                .result
                .as_ref()
                .expect("Normalized result")
                .total_samples)
                > 0
        );
        assert!(view.read_with(cx, |view, _| view.notice.contains("does not prove")));
        cx.dispatch_action(ToggleProfileView);
        cx.run_until_parked();
        assert!(view.read_with(cx, |view, _| view.flamegraph));
        cx.update(|_, cx| {
            settings::SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/capture.json".into()]));
                })
            })
        });
        cx.run_until_parked();
        assert!(view.read_with(cx, |view, _| view.result.is_none()));
        assert!(view.read_with(cx, |view, _| view.result_paths.is_empty()
            && view.flame_rows.is_empty()));
        assert_eq!(
            fs.load(std::path::Path::new("/fixture/untouched.rs"))
                .await
                .expect("Preserved file"),
            "original"
        );
    }

    #[gpui::test]
    async fn profile_view_subpixel_flamegraph_selects_before_zoom(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({"workload.rs":"fn main() {}"}),
        )
        .await;
        let project = Project::test(fs, ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let mut profile = large_profile();
        profile.nodes.truncate(2_000);
        profile.roots = (0..profile.nodes.len()).collect();
        profile.total_samples = profile.nodes.len() as u64;
        for node in &mut profile.nodes {
            node.parent = None;
            node.children.clear();
            node.self_samples = 1;
        }
        let tree = profile_tree(&profile, || false).expect("flat call tree");
        let rows = profile_flamegraph(&profile, None, || false).expect("flat flamegraph");
        let view = cx.new(|cx| ProfileView::new(project.clone(), workspace.downgrade(), cx));
        view.update(cx, |view, _| {
            view.result = Some(Arc::new(profile));
            view.host.replace_snapshot(tree);
            view.flame_rows = Arc::new(rows);
            view.flamegraph = true;
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
            assert_eq!(
                workspace.active_item(cx).expect("active item").item_id(),
                view.entity_id()
            );
        });
        cx.simulate_resize(gpui::size(px(1200.0), px(720.0)));
        cx.run_until_parked();
        view.update(cx, |view, cx| {
            assert!(view.flamegraph && !view.flame_rows.is_empty());
            cx.notify();
        });
        cx.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
        let bounds = cx
            .debug_bounds("native-flamegraph-row-0")
            .expect("painted row");
        let position =
            bounds.origin + gpui::point(bounds.size.width * 0.50025, bounds.size.height / 2.0);
        cx.simulate_mouse_move(position, None, gpui::Modifiers::default());
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.hovered_bar, Some((0, 1_000)));
            assert!(view.selected_node_index().is_none());
        });
        cx.simulate_click(position, gpui::Modifiers::default());
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.selected_node_index(), Some(1_000));
            assert_eq!(view.hovered_bar, None);
            assert_eq!(view.zoom_root, None);
        });
        view.update(cx, |view, cx| view.zoom_selected(cx));
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.zoom_root, Some(1_000));
            let bar = view
                .flame_rows
                .first()
                .and_then(|row| row.first())
                .expect("zoomed frame");
            assert_eq!(bar.node, 1_000);
            assert_eq!(bar.left, 0.0);
            assert_eq!(bar.width, 1.0);
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn profile_view_large_flamegraph_paints_a_bounded_viewport(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let project = Project::test(project::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        for flat in [false, true] {
            let mut profile = large_profile();
            if flat {
                profile.roots = (0..profile.nodes.len()).collect();
                profile.total_samples = profile.nodes.len() as u64;
                for node in &mut profile.nodes {
                    node.parent = None;
                    node.children.clear();
                    node.self_samples = 1;
                }
            }
            let rows =
                profile_flamegraph(&profile, None, || false).expect("maximum supported flamegraph");
            assert_eq!(rows.len(), if flat { 1 } else { 256 });
            let view = cx.new(|cx| ProfileView::new(project.clone(), workspace.downgrade(), cx));
            view.update(cx, |view, _| {
                view.result = Some(Arc::new(profile));
                view.flame_rows = Arc::new(rows);
                view.flamegraph = true;
            });
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
            });
            cx.simulate_resize(gpui::size(px(1600.0), px(720.0)));
            cx.run_until_parked();
            view.read_with(cx, |view, _| {
                view.painted_flame_bars.store(0, Ordering::Relaxed);
            });
            let started = std::time::Instant::now();
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            let elapsed = started.elapsed();
            let root = cx
                .debug_bounds("native-flamegraph-row-0")
                .expect("first row painted");
            assert!(root.size.width > px(1000.0) && root.size.width <= px(1600.0));
            assert!(
                cx.debug_bounds("native-flamegraph-row-255").is_none(),
                "deep offscreen row must remain virtualized"
            );
            let painted = view.read_with(cx, |view, _| {
                view.painted_flame_bars.load(Ordering::Relaxed)
            });
            if flat {
                assert_eq!(painted, 100_000, "subpixel frames must not disappear");
            } else {
                assert!(
                    painted > 0 && painted < 100_000,
                    "only visible depths paint"
                );
            }
            eprintln!(
                "native-profile-flamegraph nodes=100000 flat={flat} painted={painted} foreground_us={}",
                elapsed.as_micros()
            );
            assert!(
                elapsed <= std::time::Duration::from_millis(250),
                "100,000-node flamegraph paint took {elapsed:?}"
            );
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn profile_view_deep_selection_expands_ancestors_in_one_batch(
        cx: &mut gpui::TestAppContext,
    ) {
        use project::profile_results::MAX_PROFILE_NODES;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let project = Project::test(project::FakeFs::new(cx.executor()), [], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let profile = large_profile();
        let collapsed_root = *profile.roots.last().expect("final root");
        let tree = profile_tree(&profile, || false).expect("profile tree");
        let view = cx.new(|cx| ProfileView::new(project, workspace.downgrade(), cx));
        view.update(cx, |view, _| {
            view.result = Some(Arc::new(profile));
            view.host.replace_snapshot(tree);
            view.host.expand_nodes(
                (0..collapsed_root).map(|index| LanguageToolNodeId(index.to_string())),
            );
            assert_eq!(view.host.visible_rows().len(), collapsed_root + 1);
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
        });
        cx.simulate_resize(gpui::size(px(1600.0), px(720.0)));
        cx.run_until_parked();
        let started = std::time::Instant::now();
        let (expansion_elapsed, selection_elapsed) = view.update(cx, |view, cx| {
            let expansion_started = std::time::Instant::now();
            view.expand_node_ancestors(MAX_PROFILE_NODES - 1);
            let expansion_elapsed = expansion_started.elapsed();
            let selection_started = std::time::Instant::now();
            view.host
                .select(LanguageToolNodeId((MAX_PROFILE_NODES - 1).to_string()));
            view.host.reveal_selection(gpui::ScrollStrategy::Bottom);
            cx.notify();
            (expansion_elapsed, selection_started.elapsed())
        });
        let draw_started = std::time::Instant::now();
        cx.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
        let elapsed = started.elapsed();
        let draw_elapsed = draw_started.elapsed();
        view.update(cx, |view, _| {
            assert_eq!(view.host.visible_rows().len(), MAX_PROFILE_NODES);
            assert_eq!(
                view.host.selected(),
                Some(&LanguageToolNodeId((MAX_PROFILE_NODES - 1).to_string()))
            );
            let rows = view.host.visible_rows_snapshot();
            view.select_node(MAX_PROFILE_NODES - 1);
            assert!(Arc::ptr_eq(&rows, &view.host.visible_rows_snapshot()));
            let scroll = view.host.scroll_handle().0.borrow();
            let measured = scroll.last_item_size.expect("painted tree");
            assert!(measured.item.height > px(0.0) && measured.item.height <= px(720.0));
            let bottom = measured.item.height - measured.contents.height;
            assert!((scroll.base_handle.offset().y - bottom).abs() <= px(1.0));
        });
        eprintln!(
            "native-profile-deep-selection nodes={MAX_PROFILE_NODES} expansion_us={} selection_reveal_us={} draw_us={} foreground_us={}",
            expansion_elapsed.as_micros(),
            selection_elapsed.as_micros(),
            draw_elapsed.as_micros(),
            elapsed.as_micros()
        );
        assert!(
            elapsed <= std::time::Duration::from_millis(250),
            "100,000-node deep selection and paint took {elapsed:?}"
        );
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_offers_successful_task_artifacts_and_rejects_changed_files(
        cx: &mut gpui::TestAppContext,
    ) {
        struct Terminal(Arc<project::FakeFs>);
        impl workspace::TerminalProvider for Terminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                handle: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<std::process::ExitStatus>>> {
                let terminal = Self(self.0.clone());
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
                _: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<std::process::ExitStatus>>> {
                let fs = self.0.clone();
                cx.background_spawn(async move {
                    fs.insert_tree(
                        "/fixture/target",
                        serde_json::json!({
                            "capture.json": std::str::from_utf8(PROFILE).expect("profile"),
                            "capture.syms.json": std::str::from_utf8(SYMBOLS).expect("symbols")
                        }),
                    )
                    .await;
                    Some(Ok(std::process::ExitStatus::default()))
                })
            }
        }
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/fixture", serde_json::json!({
            ".git": {}, ".gitignore": "target\n", "target": {},
            "workload.rs": include_str!("../../../crates/language_tools/test_fixtures/profiling/workload.rs")
        })).await;
        let project = Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let ready = Rc::new(std::cell::RefCell::new(None));
        let _subscription = cx.update(|_, cx| {
            cx.subscribe(&workspace, {
                let ready = ready.clone();
                move |_, event, _| {
                    if let workspace::Event::TaskArtifactGroupReady(group) = event {
                        *ready.borrow_mut() = Some(group.clone());
                    }
                }
            })
        });
        let template: task::TaskTemplate = serde_json::from_value(serde_json::json!({
            "label": "Native profile fixture", "command": "explicit-profiler", "cwd": "/fixture", "save": "none",
            "artifact_group": {"format":"samply-0.13.1", "files":{
                "profile":{"path":"target/capture.json", "kind":"data", "max_bytes":33554432},
                "symbols":{"path":"target/capture.syms.json", "kind":"data", "max_bytes":16777216}
            }}
        })).expect("declared profiling Task");
        let approve_capture = |cx: &mut gpui::VisualTestContext| {
            let (message, detail) = cx.pending_prompt().expect("Task capture consent is held");
            assert_eq!(message, "Capture a native profile on the project host?");
            assert!(detail.contains("/fixture/target/capture.json"));
            assert!(detail.contains("/fixture/target/capture.syms.json"));
            assert!(ready.borrow().is_none(), "No Task offer precedes consent");
            cx.simulate_prompt_answer("Capture");
            cx.run_until_parked();
        };
        for changed in [false, true] {
            let task = template
                .resolve_task("native-task", &task::TaskContext::default())
                .expect("resolved Task");
            workspace.update_in(cx, |workspace, window, cx| {
                workspace.set_terminal_provider(Terminal(fs.clone()));
                workspace.schedule_resolved_task(
                    project::TaskSourceKind::UserInput,
                    task,
                    true,
                    window,
                    cx,
                );
            });
            cx.run_until_parked();
            approve_capture(cx);
            let group = ready
                .borrow_mut()
                .take()
                .expect("actual successful completion");
            assert!(
                group.entry("profile").expect("indexed output").is_ignored,
                "refresh supports newly created ignored output"
            );
            let offered_notice = project
                .read_with(cx, |project, cx| {
                    task_profile_import_notice(&group, project, cx)
                })
                .expect("permitted completed Task artifacts");
            assert!(offered_notice.contains("profile: /fixture/target/capture.json"));
            assert!(offered_notice.contains("symbols: /fixture/target/capture.syms.json"));
            assert!(offered_notice.contains("locally without uploading"));
            assert!(offered_notice.contains("Clear or closing this view does not delete"));
            workspace.read_with(cx, |workspace, cx| {
                assert!(
                    workspace
                        .notification_ids()
                        .contains(&NotificationId::unique::<CompletedTaskArtifactGroup>())
                );
                if !changed {
                    assert!(
                        workspace.item_of_type::<ProfileView>(cx).is_none(),
                        "offer does not import without activation"
                    );
                }
            });
            if changed {
                let versions = ["profile", "symbols"]
                    .into_iter()
                    .map(|role| {
                        let path = group.path(role).expect("named artifact");
                        let absolute = project
                            .read_with(cx, |project, cx| project.absolute_path(path, cx))
                            .expect("artifact path");
                        (
                            absolute,
                            group.entry(role).expect("captured version").clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                fs.insert_tree(
                    "/fixture/target",
                    serde_json::json!({"capture.syms.json":"{}"}),
                )
                .await;
                assert!(
                    check_task_versions(fs.as_ref(), &versions).await.is_err(),
                    "filesystem version guard rejects changes before UI event processing"
                );
                cx.run_until_parked();
            }
            workspace.update_in(cx, |workspace, window, cx| {
                open_task_profile(workspace, group, window, cx)
            });
            cx.run_until_parked();
            let view = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.item_of_type::<ProfileView>(cx)
                })
                .expect("native viewer from activated offer");
            view.read_with(cx, |view, _| {
                if changed {
                    assert!(
                        view.result.is_none(),
                        "changed Task artifacts clear the retained result and cannot reimport"
                    );
                } else {
                    assert_eq!(
                        view.result
                            .as_ref()
                            .expect("Task profile imported")
                            .total_samples,
                        30
                    );
                }
            });
            if !changed {
                view.update(cx, |view, cx| {
                    view.import_paths(
                        vec![
                            "/fixture/target/capture.json".into(),
                            "/fixture/target/capture.syms.json".into(),
                        ],
                        cx,
                    )
                    .expect("explicit manual import before picker supersession");
                });
                cx.run_until_parked();
                view.read_with(cx, |view, _| {
                    assert!(view.result.is_some());
                    assert!(view.task_group.is_none());
                });
                view.update_in(cx, |view, window, cx| view.prompt(window, cx));
                assert!(cx.did_prompt_for_paths(), "actual held system picker");
                cx.run_until_parked();
                view.read_with(cx, |view, cx| {
                    assert!(view.selecting, "the profile picker is still outstanding");
                    assert!(view.restart_blocker(cx).is_some());
                });
                let replacement_task = template
                    .resolve_task("native-task-replacement", &task::TaskContext::default())
                    .expect("resolved replacement Task");
                workspace.update_in(cx, |workspace, window, cx| {
                    workspace.schedule_resolved_task(
                        project::TaskSourceKind::UserInput,
                        replacement_task,
                        true,
                        window,
                        cx,
                    );
                });
                cx.run_until_parked();
                approve_capture(cx);
                let replacement_group = ready
                    .borrow_mut()
                    .take()
                    .expect("replacement successful completion");
                view.read_with(cx, |view, cx| {
                    assert!(
                        view.selecting,
                        "Task completion must not resolve the picker"
                    );
                    assert!(view.restart_blocker(cx).is_some());
                    assert!(view.result.is_some(), "manual profile remains displayed");
                    assert!(view.task_group.is_none());
                });
                workspace.update_in(cx, |workspace, window, cx| {
                    open_task_profile(workspace, replacement_group.clone(), window, cx)
                });
                cx.run_until_parked();
                let (replacement_result, replacement_notice) = view.read_with(cx, |view, cx| {
                    assert!(!view.selecting, "Task import supersedes the held picker");
                    assert!(!view.importing, "replacement import completed");
                    assert!(view.restart_blocker(cx).is_none());
                    assert!(Rc::ptr_eq(
                        view.task_group.as_ref().expect("replacement Task binding"),
                        &replacement_group
                    ));
                    let result = view.result.clone().expect("replacement Task profile");
                    assert_eq!(result.total_samples, 30);
                    (result, view.notice.clone())
                });
                view.update_in(cx, |view, window, _| {
                    assert!(
                        view.focus.is_focused(window),
                        "Task offer focuses its viewer"
                    );
                });
                assert!(
                    cx.did_prompt_for_paths(),
                    "obsolete picker remains available until its response"
                );
                cx.simulate_path_prompt_response(|_| {
                    Some(vec![
                        "/fixture/target/capture.json".into(),
                        "/fixture/target/capture.syms.json".into(),
                    ])
                });
                cx.run_until_parked();
                view.read_with(cx, |view, cx| {
                    assert!(!view.selecting);
                    assert!(!view.importing);
                    assert!(view.restart_blocker(cx).is_none());
                    assert_eq!(view.notice, replacement_notice);
                    assert!(Arc::ptr_eq(
                        view.result.as_ref().expect("retained replacement result"),
                        &replacement_result
                    ));
                    assert!(Rc::ptr_eq(
                        view.task_group
                            .as_ref()
                            .expect("retained replacement binding"),
                        &replacement_group
                    ));
                });
                view.update_in(cx, |view, window, _| {
                    assert!(
                        view.focus.is_focused(window),
                        "obsolete picker cannot move focus"
                    );
                });
                view.update_in(cx, |view, window, cx| view.prompt(window, cx));
                assert!(cx.did_prompt_for_paths(), "actual native pair picker");
                let consent_replacement = template
                    .resolve_task(
                        "native-task-consent-replacement",
                        &task::TaskContext::default(),
                    )
                    .expect("resolved Task superseding held consent");
                workspace.update_in(cx, |workspace, window, cx| {
                    workspace.schedule_resolved_task(
                        project::TaskSourceKind::UserInput,
                        consent_replacement,
                        true,
                        window,
                        cx,
                    );
                });
                cx.run_until_parked();
                cx.simulate_path_prompt_response(|_| {
                    Some(vec![
                        "/fixture/target/capture.json".into(),
                        "/fixture/target/capture.syms.json".into(),
                    ])
                });
                cx.run_until_parked();
                assert!(
                    cx.has_pending_prompt(),
                    "actual local exact-pair consent is held"
                );
                view.read_with(cx, |view, _| {
                    assert!(view.selecting);
                    assert!(!view.importing);
                });
                approve_capture(cx);
                let consent_group = ready
                    .borrow_mut()
                    .take()
                    .expect("actual successful replacement completion");
                let (message, detail) = cx
                    .pending_prompt()
                    .expect("manual import consent remains behind completed Capture");
                assert_eq!(message, "Import these profile artifacts locally?");
                assert!(detail.contains("/fixture/target/capture.json"));
                assert!(detail.contains("/fixture/target/capture.syms.json"));
                workspace.update_in(cx, |workspace, window, cx| {
                    open_task_profile(workspace, consent_group.clone(), window, cx)
                });
                cx.run_until_parked();
                let (consent_result, consent_notice) = view.read_with(cx, |view, cx| {
                    assert!(!view.selecting);
                    assert!(!view.importing);
                    assert!(view.restart_blocker(cx).is_none());
                    assert!(Rc::ptr_eq(
                        view.task_group.as_ref().expect("current Task binding"),
                        &consent_group
                    ));
                    let result = view.result.clone().expect("Task replacement result");
                    assert_eq!(result.total_samples, 30);
                    (result, view.notice.clone())
                });
                cx.simulate_prompt_answer("Import Profile");
                cx.run_until_parked();
                view.read_with(cx, |view, cx| {
                    assert!(!view.selecting);
                    assert!(!view.importing);
                    assert!(view.restart_blocker(cx).is_none());
                    assert_eq!(view.notice, consent_notice);
                    assert!(Arc::ptr_eq(
                        view.result
                            .as_ref()
                            .expect("Task replacement retained after old consent"),
                        &consent_result
                    ));
                    assert!(Rc::ptr_eq(
                        view.task_group
                            .as_ref()
                            .expect("binding retained after old consent"),
                        &consent_group
                    ));
                });
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_large_import_applies_within_foreground_budget(
        cx: &mut gpui::TestAppContext,
    ) {
        check_large_profile_import(cx, false, None).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_mapped_import_applies_within_foreground_budget(
        cx: &mut gpui::TestAppContext,
    ) {
        check_large_profile_import(cx, true, None).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_large_denied_batch_releases_result(cx: &mut gpui::TestAppContext) {
        check_large_profile_import(cx, true, Some(LargeProfileLifecycle::DeniedBatch)).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_completed_import_disposal_rejects_stale_result(
        cx: &mut gpui::TestAppContext,
    ) {
        check_large_profile_import(
            cx,
            false,
            Some(LargeProfileLifecycle::CompletedImportDisposal),
        )
        .await;
    }

    #[cfg(target_os = "macos")]
    enum LargeProfileLifecycle {
        DeniedBatch,
        CompletedImportDisposal,
    }

    #[cfg(target_os = "macos")]
    async fn check_large_profile_import(
        cx: &mut gpui::TestAppContext,
        mapped: bool,
        lifecycle: Option<LargeProfileLifecycle>,
    ) {
        use project::Fs as _;
        use project::profile_results::MAX_PROFILE_NODES;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let count = MAX_PROFILE_NODES;
        let mut input: serde_json::Value =
            serde_json::from_slice(PROFILE).expect("profile fixture");
        let thread = &mut input["threads"][0];
        thread["stringArray"] = serde_json::json!(
            (0..count)
                .map(|index| format!("frame_{index}"))
                .collect::<Vec<_>>()
        );
        thread["frameTable"] = serde_json::json!({"length":count,"func":(0..count).collect::<Vec<_>>(),"address":(0..count).collect::<Vec<_>>()});
        thread["funcTable"] = serde_json::json!({"length":count,"name":(0..count).collect::<Vec<_>>(),"resource":vec![-1;count]});
        thread["stackTable"] = serde_json::json!({"length":count,"prefix":vec![serde_json::Value::Null;count],"frame":(0..count).collect::<Vec<_>>()});
        thread["samples"] = serde_json::json!({"length":count,"weightType":"samples","stack":(0..count).collect::<Vec<_>>(),"weight":vec![1;count]});
        let mut symbols: serde_json::Value =
            serde_json::from_slice(SYMBOLS).expect("symbol fixture");
        let mut files = serde_json::Map::new();
        if mapped {
            let source_count = count - 2;
            let mut strings = vec![serde_json::Value::from("workload::frame")];
            for index in 0..source_count {
                strings.push(serde_json::Value::from(format!("/fixture/s{index}.rs")));
                files.insert(
                    format!("s{index}.rs"),
                    serde_json::Value::from("fn first() {}\nfn second() {}\n"),
                );
            }
            symbols["string_table"] = serde_json::Value::Array(strings);
            let mut library = symbols["data"][0].clone();
            library["symbol_table"] =
                serde_json::json!((0..count).map(|index| serde_json::json!({
                "rva":index,"size":1,"symbol":0,
                "frames":[{"file":1 + index % source_count,"line":1 + index / source_count}]
            })).collect::<Vec<_>>());
            library["known_addresses"] =
                serde_json::json!((0..count).map(|index| [index, index]).collect::<Vec<_>>());
            symbols["data"] = serde_json::json!([library]);
            input["threads"][0]["stringArray"] = serde_json::json!(["fallback"]);
            input["threads"][0]["funcTable"]["name"] = serde_json::json!(vec![0; count]);
            input["threads"][0]["funcTable"]["resource"] = serde_json::json!(vec![0; count]);
        } else {
            symbols["string_table"] = serde_json::json!([]);
            symbols["data"] = serde_json::json!([]);
        }
        files.insert(
            "capture.json".into(),
            serde_json::Value::from(serde_json::to_string(&input).expect("large profile JSON")),
        );
        files.insert(
            "capture.syms.json".into(),
            serde_json::Value::from(serde_json::to_string(&symbols).expect("symbol table")),
        );
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/fixture", serde_json::Value::Object(files))
            .await;
        let project = Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        cx.run_until_parked();
        let view = cx.new(|cx| ProfileView::new(project, workspace.downgrade(), cx));
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
        });
        cx.simulate_resize(gpui::size(px(1200.0), px(720.0)));
        for replacement in [false, true] {
            if replacement {
                view.update(cx, |view, _| view.host.select_last());
            }
            view.update(cx, |view, cx| {
                view.import_application_time = None;
                view.import_paths(
                    vec![
                        "/fixture/capture.json".into(),
                        "/fixture/capture.syms.json".into(),
                    ],
                    cx,
                )
                .expect("admitted import");
            });
            cx.run_until_parked();
            let application = view.read_with(cx, |view, _| {
                let result = view.result.as_ref().expect(&view.notice);
                assert_eq!(result.nodes.len(), count);
                assert_eq!(
                    result
                        .frames
                        .iter()
                        .filter(|frame| frame.source.is_some())
                        .count(),
                    if mapped { count } else { 0 }
                );
                assert_eq!(view.result_paths.len(), if mapped { count } else { 2 });
                assert_eq!(result.total_samples, count as u64);
                assert_eq!(view.host.visible_rows().len(), count);
                if replacement {
                    assert_eq!(view.selected_node_index(), Some(count - 1));
                }
                view.import_application_time
                    .expect("completed application measurement")
            });
            let started = std::time::Instant::now();
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            let paint = started.elapsed();
            eprintln!(
                "native-profile-import nodes={count} mapped={mapped} replacement={replacement} application_us={} paint_us={}",
                application.as_micros(),
                paint.as_micros()
            );
            assert!(
                application + paint <= std::time::Duration::from_millis(250),
                "profile application {application:?} plus paint {paint:?} exceeded foreground budget"
            );
        }
        if let Some(lifecycle) = lifecycle {
            view.update_in(cx, |view, window, cx| {
                window.focus(&view.toolbar_focus[0], cx)
            });
            match lifecycle {
                LargeProfileLifecycle::DeniedBatch => {
                    let (project, released, released_flame_rows) = view.read_with(cx, |view, _| {
                        (
                            view.project.clone(),
                            Arc::downgrade(view.result.as_ref().expect("loaded mapped profile")),
                            Arc::downgrade(&view.flame_rows),
                        )
                    });
                    view.update(cx, |view, _| {
                        view.invalidation_time = None;
                        view.invalidation_path_count = 0;
                    });
                    cx.update(|_, cx| {
                        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                            store.update_user_settings(cx, |settings| {
                                settings.project.worktree.private_files =
                                    Some(settings::ExtendingVec(vec!["**/s*.rs".into()]));
                            });
                        });
                    });
                    cx.run_until_parked();
                    project.read_with(cx, |project, cx| {
                        let worktree = project
                            .visible_worktrees(cx)
                            .next()
                            .expect("mapped worktree")
                            .read(cx);
                        assert_eq!(
                            worktree
                                .snapshot()
                                .files(true, 0)
                                .filter(|entry| entry.is_private)
                                .count(),
                            count - 2
                        );
                    });
                    let application = view.read_with(cx, |view, _| {
                        assert!(view.result.is_none());
                        assert!(view.result_paths.is_empty());
                        assert!(view.flame_rows.is_empty());
                        assert!(view.host.visible_rows().is_empty());
                        assert!(view.notice.contains("no longer permitted"));
                        assert_eq!(
                            view.invalidation_path_count,
                            count - 2,
                            "the actual privacy event delivers the dense referenced batch"
                        );
                        view.invalidation_time
                            .expect("actual Project privacy callback")
                    });
                    let started = std::time::Instant::now();
                    cx.update(|window, cx| {
                        assert!(view.read(cx).toolbar_focus[0].is_focused(window));
                        window.refresh();
                        window.draw(cx).clear(cx);
                    });
                    let paint = started.elapsed();
                    assert!(
                        released.upgrade().is_none(),
                        "dense denial releases the normalized model"
                    );
                    assert!(
                        released_flame_rows.upgrade().is_none(),
                        "dense denial releases the flamegraph rows"
                    );
                    assert!(
                        fs.read_file_sync("/fixture/s0.rs").is_ok(),
                        "privacy revocation does not delete source"
                    );
                    eprintln!(
                        "native-profile-denied-batch paths={} application_us={} paint_us={}",
                        count - 2,
                        application.as_micros(),
                        paint.as_micros()
                    );
                    assert!(
                        application + paint <= std::time::Duration::from_millis(250),
                        "dense denial {application:?} plus paint {paint:?} exceeded foreground budget"
                    );
                }
                LargeProfileLifecycle::CompletedImportDisposal => {
                    for disposal in ["cancel", "replacement", "clear"] {
                        let previous = view.read_with(cx, |view, _| {
                            view.result.clone().expect("retained profile")
                        });
                        let (reached_sender, reached_receiver) =
                            futures::channel::oneshot::channel();
                        let (resume_sender, resume_receiver) = futures::channel::oneshot::channel();
                        let (cancellation, generation) = view.update(cx, |view, cx| {
                            view.import_completion_gate = Some((reached_sender, resume_receiver));
                            view.import_paths(
                                vec![
                                    "/fixture/capture.json".into(),
                                    "/fixture/capture.syms.json".into(),
                                ],
                                cx,
                            )
                            .expect("maximum-size replacement");
                            (view.cancellation.clone(), view.generation)
                        });
                        let (completed, completed_flame_rows) = reached_receiver
                            .await
                            .expect("actual parser and projections completed");
                        assert_eq!(
                            completed
                                .upgrade()
                                .expect("held complete model")
                                .nodes
                                .len(),
                            count
                        );
                        assert_eq!(
                            completed_flame_rows
                                .upgrade()
                                .expect("held complete flamegraph")
                                .iter()
                                .map(Vec::len)
                                .sum::<usize>(),
                            count
                        );
                        view.update(cx, |view, _| view.import_application_time = None);
                        let started = std::time::Instant::now();
                        view.update(cx, |view, cx| match disposal {
                            "cancel" => view.cancel_import(cx),
                            "replacement" => view
                                .import_paths(
                                    vec![
                                        "/fixture/capture.json".into(),
                                        "/fixture/capture.syms.json".into(),
                                    ],
                                    cx,
                                )
                                .expect("fresh replacement"),
                            "clear" => view.clear(cx),
                            _ => unreachable!("finite disposal scenario"),
                        });
                        let disposal_time = started.elapsed();
                        assert!(cancellation.load(Ordering::Relaxed));
                        if resume_sender.send(()).is_err() {
                            assert!(
                                cancellation.load(Ordering::Relaxed),
                                "only retired completed work may drop its gate"
                            );
                        }
                        cx.run_until_parked();
                        let application = view.read_with(cx, |view, _| {
                            assert_ne!(view.generation, generation);
                            assert!(!view.importing);
                            assert!(!view.selecting);
                            match disposal {
                                "cancel" => {
                                    assert!(Arc::ptr_eq(
                                        view.result
                                            .as_ref()
                                            .expect("cancel preserves prior result"),
                                        &previous
                                    ));
                                    assert!(view.notice.starts_with("Import cancelled"));
                                }
                                "replacement" => {
                                    let result =
                                        view.result.as_ref().expect("fresh import recovers");
                                    assert_eq!(result.nodes.len(), count);
                                    assert!(!Arc::ptr_eq(result, &previous));
                                }
                                "clear" => {
                                    assert!(view.result.is_none());
                                    assert!(view.result_paths.is_empty());
                                    assert!(view.flame_rows.is_empty());
                                    assert!(view.host.visible_rows().is_empty());
                                    assert!(view.notice.starts_with("Profile cleared"));
                                }
                                _ => unreachable!("finite disposal scenario"),
                            }
                            view.import_application_time.unwrap_or_default()
                        });
                        let started = std::time::Instant::now();
                        cx.update(|window, cx| {
                            assert!(
                                view.read(cx).toolbar_focus[0].is_focused(window),
                                "retired import cannot move focus"
                            );
                            window.refresh();
                            window.draw(cx).clear(cx);
                        });
                        let paint = started.elapsed();
                        assert!(
                            completed.upgrade().is_none(),
                            "retired completed model is released for {disposal}"
                        );
                        assert!(
                            completed_flame_rows.upgrade().is_none(),
                            "retired completed flamegraph rows are released for {disposal}"
                        );
                        eprintln!(
                            "native-profile-completed-disposal kind={disposal} nodes={count} disposal_us={} application_us={} paint_us={}",
                            disposal_time.as_micros(),
                            application.as_micros(),
                            paint.as_micros()
                        );
                        assert!(
                            disposal_time + application + paint
                                <= std::time::Duration::from_millis(250),
                            "completed disposal {disposal_time:?} and application {application:?} plus paint {paint:?} exceeded foreground budget"
                        );
                    }
                }
            }
            cx.deactivate_window();
            container.update_in(cx, |_, window, _| window.remove_window());
            return;
        }
        if mapped {
            let (project, previous) = view.read_with(cx, |view, _| {
                (
                    view.project.clone(),
                    view.result.clone().expect("loaded profile"),
                )
            });
            let (worktree_id, changes) = project.read_with(cx, |project, cx| {
                let worktree = project
                    .visible_worktrees(cx)
                    .next()
                    .expect("fixture worktree")
                    .read(cx);
                let changes = worktree
                    .snapshot()
                    .files(true, 0)
                    .map(|entry| (entry.path.clone(), entry.id, project::PathChange::Updated))
                    .collect::<Vec<_>>();
                assert_eq!(
                    changes.len(),
                    count,
                    "every referenced source and artifact is in the event"
                );
                (worktree.id(), Arc::from(changes))
            });
            view.update(cx, |view, _| view.invalidation_time = None);
            project.update(cx, |_, cx| {
                cx.emit(project::Event::WorktreeUpdatedEntries(worktree_id, changes))
            });
            cx.run_until_parked();
            let application = view.read_with(cx, |view, _| {
                assert!(Arc::ptr_eq(
                    view.result.as_ref().expect("permitted sources retained"),
                    &previous
                ));
                view.invalidation_time
                    .expect("actual project-event callback ran")
            });
            let started = std::time::Instant::now();
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            let paint = started.elapsed();
            eprintln!(
                "native-profile-file-events paths={count} application_us={} paint_us={}",
                application.as_micros(),
                paint.as_micros()
            );
            assert!(
                application + paint <= std::time::Duration::from_millis(250),
                "referenced-file event {application:?} plus paint {paint:?} exceeded foreground budget"
            );
            let released = Arc::downgrade(&previous);
            drop(previous);
            view.update(cx, |view, _| view.invalidation_time = None);
            fs.remove_file(
                std::path::Path::new("/fixture/s0.rs"),
                project::RemoveOptions::default(),
            )
            .await
            .expect("remove mapped source through filesystem");
            cx.run_until_parked();
            let application = view.read_with(cx, |view, _| {
                assert!(
                    view.result.is_none(),
                    "deleted source revokes the loaded profile"
                );
                assert!(view.result_paths.is_empty());
                assert!(view.flame_rows.is_empty());
                assert!(view.host.visible_rows().is_empty());
                assert!(view.notice.contains("no longer permitted"));
                view.invalidation_time
                    .expect("filesystem event reached native callback")
            });
            let started = std::time::Instant::now();
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            let paint = started.elapsed();
            assert!(
                released.upgrade().is_none(),
                "revoked profile model must be released"
            );
            eprintln!(
                "native-profile-revocation nodes={count} application_us={} paint_us={}",
                application.as_micros(),
                paint.as_micros()
            );
            assert!(
                application + paint <= std::time::Duration::from_millis(250),
                "source revocation {application:?} plus paint {paint:?} exceeded foreground budget"
            );

            fs.insert_file(
                "/fixture/s0.rs",
                "fn first() {}\nfn second() {}\n".as_bytes().to_vec(),
            )
            .await;
            cx.run_until_parked();
            view.update(cx, |view, cx| {
                view.import_paths(
                    vec![
                        "/fixture/capture.json".into(),
                        "/fixture/capture.syms.json".into(),
                    ],
                    cx,
                )
                .expect("reimport mapped profile after restoring source");
            });
            cx.run_until_parked();
            let previous = view.read_with(cx, |view, _| {
                view.result.clone().expect("mapped profile restored")
            });
            let released = Arc::downgrade(&previous);
            drop(previous);
            view.update(cx, |view, _| view.invalidation_time = None);
            cx.update(|_, cx| {
                cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                    store.update_user_settings(cx, |settings| {
                        settings.project.worktree.private_files =
                            Some(settings::ExtendingVec(vec!["**/s0.rs".to_string()]));
                    });
                });
            });
            cx.run_until_parked();
            let application = view.read_with(cx, |view, _| {
                assert!(
                    view.result.is_none(),
                    "private source revokes large profile"
                );
                assert!(view.result_paths.is_empty());
                assert!(view.flame_rows.is_empty());
                assert!(view.host.visible_rows().is_empty());
                assert!(view.notice.contains("no longer permitted"));
                view.invalidation_time
                    .expect("privacy event reached native callback")
            });
            let started = std::time::Instant::now();
            cx.update(|window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
            });
            let paint = started.elapsed();
            assert!(
                released.upgrade().is_none(),
                "privacy-revoked profile model must be released"
            );
            eprintln!(
                "native-profile-privacy-revocation nodes={count} application_us={} paint_us={}",
                application.as_micros(),
                paint.as_micros()
            );
            assert!(
                application + paint <= std::time::Duration::from_millis(250),
                "source privacy revocation {application:?} plus paint {paint:?} exceeded foreground budget"
            );
        }
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_toolbar_keyboard(cx: &mut gpui::TestAppContext) {
        use gpui::InputEvent as _;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/fixture", serde_json::json!({
            "capture.json": std::str::from_utf8(PROFILE).expect("profile UTF-8"),
            "capture.syms.json": std::str::from_utf8(SYMBOLS).expect("symbols UTF-8"),
            "workload.rs": include_str!("../../../crates/language_tools/test_fixtures/profiling/workload.rs")
        })).await;
        let project = Project::test(fs, ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let view = cx.new(|cx| ProfileView::new(project, workspace.downgrade(), cx));
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
            window.activate_a11y_for_test();
            let focus = view.read(cx).focus.clone();
            window.focus(&focus, cx);
        });
        cx.run_until_parked();
        let dispatch = |window: &mut gpui::Window, cx: &mut App, key: &str| {
            window.refresh();
            window.draw(cx).clear(cx);
            let keystroke = gpui::Keystroke::parse(key).expect("test keystroke");
            window.dispatch_event(
                gpui::KeyDownEvent {
                    keystroke: keystroke.clone(),
                    is_held: false,
                    prefer_character_input: false,
                }
                .to_platform_input(),
                cx,
            );
            window.dispatch_event(gpui::KeyUpEvent { keystroke }.to_platform_input(), cx);
        };
        let press = |cx: &mut gpui::VisualTestContext, key: &str| {
            cx.update(|window, cx| dispatch(window, cx, key));
            cx.run_until_parked();
        };
        cx.simulate_resize(gpui::size(px(1200.0), px(720.0)));
        cx.run_until_parked();
        let assert_accessibility_focus =
            |cx: &mut gpui::VisualTestContext, focus_field: &str, role: &str, label: &str| {
                for _ in 0..2 {
                    cx.update(|window, cx| {
                        window.draw(cx).clear(cx);
                        let tree = window.debug_a11y_tree_json().expect("active profile tree");
                        let tree: serde_json::Value =
                            serde_json::from_str(&tree).expect("valid accessibility tree");
                        let focused = tree[focus_field].as_str().unwrap_or_else(|| {
                            panic!("profile accessibility focus {focus_field}: {tree}")
                        });
                        let nodes = tree["nodes"].as_object().expect("profile tree nodes");
                        let node = nodes.get(focused).expect("focused profile control");
                        assert_eq!(node["aria"]["role"], role);
                        if focus_field == "active_descendant_focus" {
                            assert_eq!(node["aria"]["selected"], true);
                        } else if role == "Button" {
                            assert!(tree["active_descendant_focus"].is_null());
                            assert_eq!(node["aria"]["disabled"], false);
                        }
                        assert!(
                            node["aria"]["label"]
                                .as_str()
                                .is_some_and(|actual| actual.contains(label)),
                            "{node}"
                        );
                    });
                }
            };
        let mut previous_status_id = None;
        let mut assert_accessibility_status = |cx: &mut gpui::VisualTestContext| {
            let (notice, flamegraph, zoom_disabled, reset_disabled, importing) =
                view.read_with(cx, |view, _| {
                    (
                        view.accessible_status(&view.status_counts()),
                        view.flamegraph,
                        view.selected_node_index().is_none(),
                        view.zoom_root.is_none(),
                        view.importing,
                    )
                });
            for _ in 0..2 {
                cx.update(|window, cx| {
                    window.draw(cx).clear(cx);
                    let tree = window.debug_a11y_tree_json().expect("active profile tree");
                    let tree: serde_json::Value =
                        serde_json::from_str(&tree).expect("valid accessibility tree");
                    let status = tree["nodes"]
                        .as_object()
                        .expect("profile tree nodes")
                        .values()
                        .find(|node| node["aria"]["role"] == "Status")
                        .expect("native profile status");
                    assert_eq!(status["aria"]["value"], notice, "{status}");
                    assert_eq!(status["aria"]["live"], "Polite", "{status}");
                    assert_eq!(status["aria"]["label"], "Native profile status");
                    if let Some(previous_id) = &previous_status_id {
                        assert_eq!(&status["accesskit_id"], previous_id);
                    }
                    previous_status_id = Some(status["accesskit_id"].clone());
                    for (label, disabled) in [
                        ("Zoom Selected", zoom_disabled),
                        ("Reset Zoom", reset_disabled),
                    ] {
                        let control = tree["nodes"]
                            .as_object()
                            .expect("profile nodes")
                            .values()
                            .find(|node| {
                                node["aria"]["role"] == "Button" && node["aria"]["label"] == label
                            });
                        assert_eq!(control.is_some(), flamegraph);
                        if let Some(control) = control {
                            assert_eq!(control["aria"]["disabled"], disabled, "{control}");
                        }
                    }
                    for (label, selected) in
                        [("Flame Graph", flamegraph), ("Call Tree", !flamegraph)]
                    {
                        let tab = tree["nodes"]
                            .as_object()
                            .expect("profile nodes")
                            .values()
                            .find(|node| {
                                node["aria"]["role"] == "Tab" && node["aria"]["label"] == label
                            })
                            .expect("profile tab");
                        assert_eq!(tab["aria"]["selected"], selected, "{tab}");
                    }
                    let cancel = tree["nodes"]
                        .as_object()
                        .expect("profile nodes")
                        .values()
                        .find(|node| node["aria"]["label"] == "Cancel Import");
                    assert_eq!(cancel.is_some(), importing);
                    if let Some(cancel) = cancel {
                        assert_eq!(cancel["aria"]["disabled"], false);
                    }
                });
            }
        };
        assert_accessibility_status(cx);
        press(cx, "tab");
        assert_accessibility_focus(cx, "gpui_focus", "Button", "Import Profile");
        press(cx, "enter");
        assert!(cx.did_prompt_for_paths(), "Tab must reach Import Profile");
        cx.simulate_path_prompt_response(|_| None);
        cx.run_until_parked();
        press(cx, "tab");
        press(cx, "space");
        assert!(
            view.read_with(cx, |view, _| view.flamegraph),
            "Space activates the mode button"
        );
        press(cx, "shift-tab");
        press(cx, "enter");
        assert!(
            cx.did_prompt_for_paths(),
            "Shift-Tab returns to Import Profile"
        );
        cx.simulate_path_prompt_response(|_| None);
        cx.run_until_parked();
        press(cx, "tab");
        press(cx, "tab");
        press(cx, "space");
        press(cx, "tab");
        press(cx, "enter");
        assert!(
            view.read_with(cx, |view, _| view.notice.starts_with("Profile cleared")),
            "disabled controls are skipped and Clear is activated"
        );
        for _ in 0..3 {
            press(cx, "shift-tab");
        }
        press(cx, "enter");
        assert!(cx.did_prompt_for_paths());
        cx.simulate_path_prompt_response(|_| {
            Some(vec![
                "/fixture/capture.json".into(),
                "/fixture/capture.syms.json".into(),
            ])
        });
        cx.run_until_parked();
        assert!(
            cx.has_pending_prompt(),
            "keyboard import presents the selected pair before reading"
        );
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        let (source_node, source_line, node_count) = view.read_with(cx, |view, _| {
            let result = view.result.as_ref().expect("keyboard import completed");
            assert_eq!(view.recording_title.as_deref(), Some("capture"));
            let (index, line) = result
                .nodes
                .iter()
                .enumerate()
                .find_map(|(index, node)| {
                    let frame = result.frames.get(node.frame)?;
                    (frame.label == "workload::leaf")
                        .then(|| frame.source.as_ref().map(|(_, line)| (index, *line)))
                        .flatten()
                })
                .expect("mapped leaf");
            (index, line, result.nodes.len())
        });
        assert_accessibility_status(cx);
        for _ in 0..4 {
            press(cx, "tab");
        }
        press(cx, "enter");
        assert_eq!(
            view.read_with(cx, |view, _| view.host.visible_rows().len()),
            node_count,
            "Expand All activates from the toolbar"
        );
        for _ in 0..5 {
            press(cx, "shift-tab");
        }
        for _ in 0..node_count {
            press(cx, "down");
            if view.read_with(cx, |view, _| view.selected_node_index()) == Some(source_node) {
                break;
            }
        }
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_node_index()),
            Some(source_node)
        );
        assert_accessibility_focus(cx, "gpui_focus", "Group", "Native profile");
        assert_accessibility_focus(
            cx,
            "active_descendant_focus",
            "TreeItem",
            "of recording samples",
        );
        press(cx, "tab");
        press(cx, "tab");
        press(cx, "space");
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_node_index()),
            Some(source_node)
        );
        assert_accessibility_focus(cx, "gpui_focus", "Tab", "Flame Graph");
        press(cx, "right");
        assert_accessibility_focus(cx, "gpui_focus", "Tab", "Call Tree");
        assert!(!view.read_with(cx, |view, _| view.flamegraph));
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_node_index()),
            Some(source_node)
        );
        press(cx, "left");
        assert!(view.read_with(cx, |view, _| view.flamegraph));
        press(cx, "shift-tab");
        press(cx, "shift-tab");
        press(cx, "home");
        for _ in 0..node_count {
            if view.read_with(cx, |view, _| view.selected_node_index()) == Some(source_node) {
                break;
            }
            press(cx, "down");
        }
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_node_index()),
            Some(source_node)
        );
        assert_accessibility_focus(
            cx,
            "active_descendant_focus",
            "Button",
            "of displayed samples",
        );
        for _ in 0..4 {
            press(cx, "tab");
        }
        press(cx, "space");
        assert!(
            view.read_with(cx, |view, _| view.zoom_root.is_some()),
            "Zoom Selected activates from the toolbar"
        );
        assert_accessibility_status(cx);
        press(cx, "tab");
        press(cx, "enter");
        assert!(
            !view.read_with(cx, |view, _| view.zoom_root.is_some()),
            "Reset Zoom activates from the toolbar"
        );
        assert_accessibility_status(cx);
        press(cx, "shift-tab");
        press(cx, "shift-tab");
        press(cx, "enter");
        assert!(!view.read_with(cx, |view, _| view.flamegraph));
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_node_index()),
            Some(source_node)
        );
        press(cx, "tab");
        assert_accessibility_focus(cx, "gpui_focus", "Button", "Open Source");
        press(cx, "enter");
        let editor = workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_item_as::<editor::Editor>(cx)
            })
            .expect("keyboard Open Source activates the editor");
        assert_eq!(
            editor.update(cx, |editor, cx| editor
                .selections
                .newest::<text::Point>(&editor.display_snapshot(cx))
                .head()
                .row),
            source_line - 1
        );
        workspace.update_in(cx, |workspace, window, cx| {
            assert!(workspace.activate_item(&view, true, true, window, cx));
        });
        cx.run_until_parked();
        let previous = view.read_with(cx, |view, _| view.result.clone().expect("retained profile"));
        let (reached_sender, reached_receiver) = futures::channel::oneshot::channel();
        let (resume_sender, resume_receiver) = futures::channel::oneshot::channel();
        let active_cancellation = view.update(cx, |view, cx| {
            view.import_processing_gate = Some((reached_sender, resume_receiver));
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
            .expect("active replacement import");
            view.cancellation.clone()
        });
        reached_receiver
            .await
            .expect("background import reached its processing boundary");
        assert_accessibility_status(cx);
        cx.update(|window, cx| {
            for _ in 0..5 {
                dispatch(window, cx, "tab");
            }
            dispatch(window, cx, "enter");
            assert!(
                !view.read(cx).importing,
                "Cancel Import activates after profile parsing"
            );
        });
        assert!(active_cancellation.load(Ordering::Relaxed));
        if resume_sender.send(()).is_err() {
            assert!(
                active_cancellation.load(Ordering::Relaxed),
                "only a cancelled import may release its background task"
            );
        }
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(Arc::ptr_eq(
                view.result
                    .as_ref()
                    .expect("active cancellation retains result"),
                &previous
            ));
            assert!(view.notice.starts_with("Import cancelled"));
        });
        assert_accessibility_status(cx);
        view.update_in(cx, |view, window, cx| window.focus(&view.focus, cx));
        let cancellation = cx.update(|window, cx| {
            let cancellation = view.update(cx, |view, cx| {
                view.import_paths(
                    vec![
                        "/fixture/capture.json".into(),
                        "/fixture/capture.syms.json".into(),
                    ],
                    cx,
                )
                .expect("replacement queued");
                assert!(view.importing);
                view.cancellation.clone()
            });
            for _ in 0..5 {
                dispatch(window, cx, "tab");
            }
            dispatch(window, cx, "enter");
            assert!(
                !view.read(cx).importing,
                "Cancel Import activates before queued work completes"
            );
            cancellation
        });
        cx.run_until_parked();
        assert!(cancellation.load(Ordering::Relaxed));
        view.read_with(cx, |view, _| {
            assert!(Arc::ptr_eq(
                view.result.as_ref().expect("cancel retains result"),
                &previous
            ));
            assert!(view.notice.starts_with("Import cancelled"));
        });
        assert_accessibility_focus(cx, "gpui_focus", "Button", "Import Profile");
        for _ in 0..4 {
            press(cx, "tab");
        }
        press(cx, "space");
        assert!(view.read_with(cx, |view, _| view.result.is_none()));
        assert_accessibility_status(cx);
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_real_watcher_clears_deleted_source(cx: &mut gpui::TestAppContext) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let directory = tempfile::TempDir::new().expect("profile fixture directory");
        let root = std::fs::canonicalize(directory.path()).expect("canonical fixture directory");
        let source = root.join("workload.rs");
        let profile = root.join("capture.json");
        let symbols = root.join("capture.syms.json");
        std::fs::write(
            &source,
            include_bytes!("../../../crates/language_tools/test_fixtures/profiling/workload.rs"),
        )
        .expect("source fixture");
        std::fs::write(&profile, PROFILE).expect("profile fixture");
        let mapped_symbols = std::str::from_utf8(SYMBOLS)
            .expect("symbol fixture UTF-8")
            .replace("/fixture/workload.rs", &source.to_string_lossy());
        std::fs::write(&symbols, mapped_symbols).expect("symbol fixture");

        let fs = Arc::new(fs::RealFs::new(None, cx.executor()));
        let project = Project::test(fs, [root.as_path()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        cx.run_until_parked();
        let view = cx.new(|cx| ProfileView::new(project, workspace.downgrade(), cx));
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
        });
        view.update(cx, |view, cx| {
            view.import_paths(vec![profile, symbols], cx)
                .expect("admitted real-filesystem import");
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view.read_with(cx, |view, _| view.result.is_none()) {
            assert!(
                std::time::Instant::now() < deadline,
                "profile import timed out"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        view.read_with(cx, |view, _| {
            assert!(
                view.result_paths
                    .iter()
                    .any(|path| path.path.as_ref() == util::rel_path::rel_path("workload.rs")),
                "imported profile must retain the source permission dependency"
            );
        });

        std::fs::remove_file(&source).expect("delete referenced source on disk");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view.read_with(cx, |view, _| view.result.is_some()) {
            assert!(
                std::time::Instant::now() < deadline,
                "real watcher did not clear the profile"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        view.read_with(cx, |view, _| {
            assert!(
                view.notice
                    .contains("referenced file is no longer permitted")
            );
            assert!(view.flame_rows.is_empty());
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_local_import_disclosure_combined_selection(
        cx: &mut gpui::TestAppContext,
    ) {
        local_import_disclosure_scenario(cx, false).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_local_import_disclosure_companion_selection(
        cx: &mut gpui::TestAppContext,
    ) {
        local_import_disclosure_scenario(cx, true).await;
    }

    #[cfg(target_os = "macos")]
    async fn local_import_disclosure_scenario(cx: &mut gpui::TestAppContext, companion: bool) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
            project::trusted_worktrees::init(collections::HashMap::default(), cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree("/fixture", serde_json::json!({
            "capture.json": std::str::from_utf8(PROFILE).expect("profile"),
            "capture.syms.json": std::str::from_utf8(SYMBOLS).expect("symbols"),
            "replacement.json": "invalid-json-private-payload",
            "replacement.syms.json": std::str::from_utf8(SYMBOLS).expect("symbols"),
            "workload.rs": include_str!("../../../crates/language_tools/test_fixtures/profiling/workload.rs")
        })).await;
        let project =
            Project::test_with_worktree_trust(fs.clone(), ["/fixture".as_ref()], cx).await;
        let worktrees = project.read_with(cx, |project, _| project.worktree_store());
        let worktree = project.read_with(cx, |project, cx| {
            project
                .visible_worktrees(cx)
                .next()
                .expect("visible fixture")
                .read(cx)
                .id()
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust global")
        });
        trust.update(cx, |trust, cx| {
            trust.restrict(
                worktrees.downgrade(),
                collections::HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                    worktree,
                )]),
                cx,
            );
        });
        assert!(
            project.read_with(cx, |project, cx| {
                project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &project.worktree_store(),
                    cx,
                )
            }),
            "the fixture begins with actual registered restriction"
        );
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktrees,
                collections::HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                    worktree,
                )]),
                cx,
            );
        });
        cx.run_until_parked();
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let view = cx.new(|cx| ProfileView::new(project.clone(), workspace.downgrade(), cx));
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(view.clone()), None, true, window, cx);
        });
        cx.run_until_parked();
        view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
        })
        .expect("existing profile");
        cx.run_until_parked();
        let previous = view.read_with(cx, |view, _| view.result.clone().expect("existing result"));
        assert_eq!(previous.total_samples, 30);
        let select_pair = |cx: &mut gpui::VisualTestContext| {
            view.update_in(cx, |view, window, cx| view.prompt(window, cx));
            assert!(cx.did_prompt_for_paths());
            cx.simulate_path_prompt_response(|_| {
                Some(if companion {
                    vec!["/fixture/replacement.json".into()]
                } else {
                    vec![
                        "/fixture/replacement.json".into(),
                        "/fixture/replacement.syms.json".into(),
                    ]
                })
            });
            cx.run_until_parked();
            if companion {
                assert!(cx.did_prompt_for_paths());
                assert!(
                    !cx.has_pending_prompt(),
                    "one artifact is not a complete pair"
                );
                view.read_with(cx, |view, _| {
                    assert!(view.selecting);
                    assert!(!view.importing);
                });
                cx.simulate_path_prompt_response(|_| {
                    Some(vec!["/fixture/replacement.syms.json".into()])
                });
                cx.run_until_parked();
            }
            let (message, details) = cx.pending_prompt().expect("actual held local disclosure");
            assert_eq!(message, "Import these profile artifacts locally?");
            assert!(details.contains("/fixture/replacement.json"));
            assert!(details.contains("/fixture/replacement.syms.json"));
            assert!(details.contains("locally without uploading"));
            assert!(details.contains("profile contents to telemetry"));
            assert!(details.contains("raw files stay on disk"));
            assert!(details.contains("project file deletion"));
            assert!(
                !details.contains("invalid-json-private-payload"),
                "review never echoes file contents"
            );
            view.read_with(cx, |view, _| {
                assert!(view.selecting);
                assert!(!view.importing);
            });
        };
        let (processing_sender, mut processing_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        view.update(cx, |view, _| {
            view.import_processing_gate = Some((processing_sender, release_receiver))
        });
        select_pair(cx);
        view.read_with(cx, |view, _| {
            assert!(Arc::ptr_eq(
                view.result.as_ref().expect("preserved result"),
                &previous
            ));
            assert!(
                view.import_processing_gate.is_some(),
                "no local import admitted before review"
            );
        });
        assert!(
            processing_receiver
                .try_recv()
                .expect("processing gate pending")
                .is_none()
        );
        cx.simulate_prompt_answer("Cancel");
        cx.run_until_parked();
        view.update(cx, |view, _| {
            assert!(!view.selecting);
            assert!(!view.importing);
            assert!(Arc::ptr_eq(
                view.result.as_ref().expect("cancel preserves result"),
                &previous
            ));
            assert!(view.notice.contains("previous result is preserved"));
            assert!(
                view.import_processing_gate.take().is_some(),
                "cancel does not admit parsing"
            );
        });
        drop(release_sender);
        assert_eq!(
            fs.read_file_sync("/fixture/replacement.json")
                .expect("raw profile retained"),
            b"invalid-json-private-payload"
        );
        assert_eq!(
            fs.read_file_sync("/fixture/replacement.syms.json")
                .expect("raw sidecar retained"),
            SYMBOLS
        );
        assert_eq!(
            fs.read_file_sync("/fixture/capture.json")
                .expect("prior raw profile retained"),
            PROFILE
        );
        select_pair(cx);
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(!view.selecting);
            assert!(!view.importing);
            assert!(Arc::ptr_eq(
                view.result
                    .as_ref()
                    .expect("invalid import preserves result"),
                &previous
            ));
            assert!(!view.notice.contains("invalid-json-private-payload"));
            assert!(view.notice.starts_with("Profile import failed:"));
            assert!(
                view.notice.contains("Malformed profile JSON"),
                "accepted malformed data reaches the bounded importer"
            );
        });
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "replacement.json": std::str::from_utf8(PROFILE).expect("valid replacement")
            }),
        )
        .await;
        cx.run_until_parked();
        select_pair(cx);
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/replacement.json".into()]));
                });
            })
        });
        cx.run_until_parked();
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(!view.importing);
            assert!(!view.selecting);
            assert!(Arc::ptr_eq(
                view.result
                    .as_ref()
                    .expect("private replacement preserves prior"),
                &previous
            ));
            assert!(view.notice.contains("permitted project files"));
        });
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None
                });
            })
        });
        cx.run_until_parked();
        select_pair(cx);
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            let fresh = view
                .result
                .as_ref()
                .expect("fresh disclosed import recovers");
            assert_eq!(fresh.total_samples, 30);
            assert!(!Arc::ptr_eq(fresh, &previous));
            assert!(!view.selecting);
            assert!(!view.importing);
        });
        select_pair(cx);
        trust.update(cx, |trust, cx| {
            trust.restrict(
                worktrees.downgrade(),
                collections::HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                    worktree,
                )]),
                cx,
            );
        });
        cx.run_until_parked();
        let restricted_notice = view.read_with(cx, |view, _| {
            assert!(view.result.is_none());
            assert!(!view.selecting);
            assert!(!view.importing);
            assert!(view.notice.contains("trust changed"));
            view.notice.clone()
        });
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(view.result.is_none());
            assert!(!view.importing);
            assert_eq!(
                view.notice, restricted_notice,
                "stale consent cannot replace restriction"
            );
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktrees,
                collections::HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                    worktree,
                )]),
                cx,
            );
        });
        cx.run_until_parked();
        select_pair(cx);
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        assert_eq!(
            view.read_with(cx, |view, _| view
                .result
                .as_ref()
                .map(|result| result.total_samples)),
            Some(30)
        );
        select_pair(cx);
        project.update(cx, |project, cx| project.close(cx));
        cx.run_until_parked();
        let closed_notice = view.read_with(cx, |view, _| {
            assert!(view.result.is_none());
            assert!(!view.selecting);
            assert!(!view.importing);
            assert!(view.notice.contains("project session"));
            view.notice.clone()
        });
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(view.result.is_none());
            assert!(!view.importing);
            assert_eq!(
                view.notice, closed_notice,
                "closed-project consent cannot republish"
            );
        });
        assert_eq!(
            fs.read_file_sync("/fixture/replacement.json")
                .expect("raw profile retained after close"),
            PROFILE
        );
        assert_eq!(
            fs.read_file_sync("/fixture/replacement.syms.json")
                .expect("raw sidecar retained after close"),
            SYMBOLS
        );
        cx.deactivate_window();
        assert!(
            cx.windows().is_empty(),
            "Project close removes the fixture window"
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn profile_view_imports_locally_and_clears_session_results(
        cx: &mut gpui::TestAppContext,
    ) {
        use project::Fs as _;
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            set_native_profile_opt_in_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            init(cx);
        });
        let fs = project::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/fixture",
            serde_json::json!({
                "capture.json": std::str::from_utf8(PROFILE).expect("profile UTF-8"),
                "capture.syms.json": std::str::from_utf8(SYMBOLS).expect("symbols UTF-8").replace("/fixture/workload.rs", "/fixture-alias/workload.rs"),
                "workload.rs": include_str!("../../../crates/language_tools/test_fixtures/profiling/workload.rs")
            }),
        )
        .await;
        fs.create_symlink(std::path::Path::new("/fixture-alias"), "/fixture".into())
            .await
            .expect("source alias");
        let project = Project::test(fs.clone(), ["/fixture".as_ref()], cx).await;
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        cx.run_until_parked();
        workspace.update_in(cx, |workspace, window, cx| {
            window.focus(&workspace.focus_handle(cx), cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(OpenNativeProfile);
        cx.run_until_parked();
        let view = workspace
            .read_with(cx, |workspace, cx| {
                workspace.item_of_type::<ProfileView>(cx)
            })
            .expect("registered action opens the native view");
        view.update_in(cx, |view, window, cx| view.prompt(window, cx));
        assert!(cx.did_prompt_for_paths());
        cx.simulate_path_prompt_response(|options| {
            let prompt = options.prompt.as_deref().expect("production import prompt");
            assert!(prompt.contains("locally without uploading"));
            assert!(prompt.contains("Clear does not delete"));
            Some(vec!["/fixture/capture.json".into()])
        });
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(view.selecting);
            assert!(!view.importing);
            assert!(view.result.is_none());
        });
        assert!(cx.did_prompt_for_paths());
        cx.simulate_path_prompt_response(|options| {
            let prompt = options
                .prompt
                .as_deref()
                .expect("production companion prompt");
            assert!(prompt.contains("Files stay on disk after Clear"));
            Some(vec!["/fixture/capture.syms.json".into()])
        });
        cx.run_until_parked();
        let (message, details) = cx.pending_prompt().expect("local exact-pair disclosure");
        assert_eq!(message, "Import these profile artifacts locally?");
        assert!(details.contains("/fixture/capture.json"));
        assert!(details.contains("/fixture/capture.syms.json"));
        assert!(details.contains("locally without uploading"));
        assert!(details.contains("raw files stay on disk"));
        view.read_with(cx, |view, _| {
            assert!(view.selecting);
            assert!(!view.importing);
            assert!(
                view.result.is_none(),
                "files must not load before the disclosure is accepted"
            );
        });
        cx.simulate_prompt_answer("Import Profile");
        cx.run_until_parked();
        view.update_in(cx, |view, window, cx| {
            window.focus(&view.focus, cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.simulate_keystrokes("down");
        view.update(cx, |view, _cx| {
            assert!(
                view.host.selected().is_some(),
                "keyboard navigation selects a native row"
            );
            let result = view.result.as_ref().expect("background import completed");
            assert_eq!(result.total_samples, 30);
            assert!(result.frames.iter().any(|frame| frame.source.is_some()));
            assert!(!view.flame_rows.is_empty());
        });
        let (source_node, source_line) = view.read_with(cx, |view, _| {
            let result = view.result.as_ref().expect("profile result");
            result
                .nodes
                .iter()
                .enumerate()
                .find_map(|(index, node)| {
                    let frame = result.frames.get(node.frame)?;
                    (frame.label == "workload::leaf")
                        .then(|| frame.source.as_ref().map(|(_, line)| (index, *line)))
                        .flatten()
                })
                .expect("mapped workload leaf")
        });
        view.update_in(cx, |view, window, cx| {
            view.select_node(source_node);
            window.focus(&view.focus, cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(ZoomToSelectedFrame);
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(view.flamegraph && view.zoom_root.is_some());
            assert_eq!(view.selected_node_index(), Some(source_node));
        });
        cx.dispatch_action(ResetProfileZoom);
        cx.run_until_parked();
        view.read_with(cx, |view, _| assert!(!view.zoom_root.is_some()));
        cx.dispatch_action(ToggleProfileView);
        cx.run_until_parked();
        view.read_with(cx, |view, _| assert!(!view.flamegraph));
        cx.dispatch_action(ZoomToSelectedFrame);
        cx.run_until_parked();
        view.update_in(cx, |view, window, cx| view.open_selected(window, cx));
        cx.run_until_parked();
        let editor = workspace
            .read_with(cx, |workspace, cx| {
                workspace.active_item_as::<editor::Editor>(cx)
            })
            .expect("source navigation opens an editor");
        assert_eq!(
            editor.update(cx, |editor, cx| editor
                .selections
                .newest::<text::Point>(&editor.display_snapshot(cx))
                .head()
                .row),
            source_line - 1
        );
        view.read_with(cx, |view, _| {
            assert!(view.zoom_root.is_some());
            assert_eq!(
                view.flame_rows
                    .first()
                    .and_then(|row| row.first())
                    .expect("zoomed root")
                    .width,
                1.0
            );
        });
        let previous = view.read_with(cx, |view, _| view.result.clone().expect("imported result"));
        let (reached_sender, reached_receiver) = futures::channel::oneshot::channel();
        let (resume_sender, resume_receiver) = futures::channel::oneshot::channel();
        let cancellation = view.update(cx, |view, cx| {
            view.import_processing_gate = Some((reached_sender, resume_receiver));
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
            .expect("active replacement import");
            view.cancellation.clone()
        });
        reached_receiver
            .await
            .expect("background import reached its processing boundary");
        view.update(cx, |view, cx| view.cancel_import(cx));
        assert!(cancellation.load(Ordering::Relaxed));
        if resume_sender.send(()).is_err() {
            assert!(
                cancellation.load(Ordering::Relaxed),
                "only a cancelled import may release its background task"
            );
        }
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(!view.importing);
            assert!(Arc::ptr_eq(
                view.result.as_ref().expect("previous result retained"),
                &previous
            ));
        });
        view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
            .expect("replacement import");
            view.cancel_import(cx);
        });
        cx.run_until_parked();
        fs.insert_tree("/fixture", serde_json::json!({"capture.json": "{"}))
            .await;
        cx.run_until_parked();
        view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
            .expect("malformed import starts asynchronously");
        });
        cx.run_until_parked();
        view.update(cx, |view, cx| {
            assert!(Arc::ptr_eq(
                view.result.as_ref().expect("previous result retained"),
                &previous
            ));
            assert!(view.notice.starts_with("Profile import failed:"));
            view.clear(cx);
            assert!(view.result.is_none());
            assert!(view.flame_rows.is_empty());
            let project = view.project.read(cx);
            let artifact = project
                .find_project_path(std::path::Path::new("/fixture/capture.json"), cx)
                .expect("artifact remains visible");
            assert!(project.entry_for_path(&artifact, cx).is_some());
        });
        fs.insert_tree("/fixture", serde_json::json!({"capture.json": std::str::from_utf8(PROFILE).expect("profile UTF-8")})).await;
        cx.run_until_parked();
        view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
        })
        .expect("reimport after clear");
        cx.run_until_parked();
        assert!(view.read_with(cx, |view, _| view.result.is_some()));
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/workload.rs".to_string()]));
                });
            });
        });
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(
                view.result.is_none(),
                "newly private source clears retained profile"
            );
            assert!(view.result_paths.is_empty());
            assert!(view.notice.contains("no longer permitted"));
        });
        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None;
                });
            });
        });
        cx.run_until_parked();
        view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
        })
        .expect("reimport after restoring visibility");
        cx.run_until_parked();
        assert!(view.read_with(cx, |view, _| view.result.is_some()));
        view.update_in(cx, |view, window, cx| view.prompt(window, cx));
        cx.simulate_path_prompt_response(|_| Some(vec!["/fixture/capture.json".into()]));
        cx.run_until_parked();
        assert!(cx.did_prompt_for_paths());
        view.update(cx, |view, cx| view.clear(cx));
        cx.simulate_path_prompt_response(|_| Some(vec!["/fixture/capture.syms.json".into()]));
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert!(
                view.result.is_none(),
                "late picker selection must not import after Clear"
            );
            assert!(!view.importing);
            assert!(!view.selecting);
        });
        view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
        })
        .expect("reimport after stale picker rejection");
        cx.run_until_parked();
        assert!(view.read_with(cx, |view, _| view.result.is_some()));
        let projection_profile =
            view.read_with(cx, |view, _| view.result.clone().expect("loaded profile"));
        let transient = cx.new(|cx| ProfileView::new(project.clone(), workspace.downgrade(), cx));
        let (first_zoom, second_zoom) = transient.update(cx, |view, cx| {
            view.result = Some(projection_profile.clone());
            view.zoom(Some(0), cx);
            let first_zoom = view.zoom_cancellation.clone();
            view.zoom(None, cx);
            let second_zoom = view.zoom_cancellation.clone();
            assert!(
                first_zoom.load(Ordering::Relaxed),
                "replacement zoom cancels its predecessor"
            );
            view.clear(cx);
            (first_zoom, second_zoom)
        });
        cx.run_until_parked();
        assert!(first_zoom.load(Ordering::Relaxed));
        assert!(
            second_zoom.load(Ordering::Relaxed),
            "Clear cancels zoom projection"
        );
        transient.read_with(cx, |view, _| {
            assert!(view.result.is_none());
            assert!(
                view.flame_rows.is_empty(),
                "cancelled zoom cannot restore cleared rows"
            );
        });
        let released_zoom = transient.update(cx, |view, cx| {
            view.result = Some(projection_profile);
            view.zoom(None, cx);
            view.zoom_cancellation.clone()
        });
        cx.update(|_, _| drop(transient));
        cx.run_until_parked();
        assert!(
            released_zoom.load(Ordering::Relaxed),
            "release cancels zoom projection"
        );
        let cancellation = view.update(cx, |view, cx| {
            view.import_paths(
                vec![
                    "/fixture/capture.json".into(),
                    "/fixture/capture.syms.json".into(),
                ],
                cx,
            )
            .expect("in-flight replacement");
            view.cancellation.clone()
        });
        project.update(cx, |project, cx| {
            let worktree = project
                .visible_worktrees(cx)
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id();
            project.remove_worktree(worktree, cx);
        });
        cx.run_until_parked();
        assert!(cancellation.load(Ordering::Relaxed));
        view.read_with(cx, |view, _| {
            assert!(view.result.is_none());
            assert!(!view.importing);
            assert!(view.notice.contains("worktree changed"));
        });
        let transient = cx.new(|cx| ProfileView::new(project.clone(), workspace.downgrade(), cx));
        let cancellation = transient.read_with(cx, |view, _| view.cancellation.clone());
        cx.update(|_, _| drop(transient));
        cx.run_until_parked();
        assert!(
            cancellation.load(Ordering::Relaxed),
            "view release cancels background parsing"
        );
    }
}
