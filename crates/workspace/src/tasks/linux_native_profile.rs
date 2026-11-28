use std::os::unix::fs::MetadataExt as _;

use project::task_store::PendingExperimentalTaskLaunch;
use project::terminals::{LinuxNativeCargoTask, NativeProfileTaskDestination};

use super::*;

pub(super) struct LinuxNativeCargoProfileLaunch {
    pub run_id: uuid::Uuid,
    pub build: SpawnInTerminal,
    pub report: task::TaskStdoutReport,
    pub pending: Option<PendingExperimentalTaskLaunch>,
    pub artifact_current: Arc<AtomicBool>,
    pub artifact_subscriptions: Vec<gpui::Subscription>,
}

impl Workspace {
    pub(super) fn schedule_linux_native_cargo_profile(
        &mut self,
        mut launch: LinuxNativeCargoProfileLaunch,
        build_handle: Option<StructuredTaskHandle>,
        on_complete: Option<TaskCompletionHandler>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let review = self.begin_native_profile_review(
            native_capture_phase_task_id(&launch.build.id),
            launch.run_id,
            launch.artifact_current.clone(),
            None,
            cx.try_global::<settings::SettingsStore>()
                .and_then(|settings| settings.observed_revision()),
            cx,
        );
        let prerequisite_lock = self.task_admission_prerequisite_lock.clone();
        // A terminal rerun would replay a single authenticated stage without the capture review.
        launch.build.show_rerun = false;
        let subscriptions = review
            .as_ref()
            .ok()
            .map(|review| watch_linux_profile_review(self, review, &launch.build, cx));
        let task = cx.spawn_in(window, async move |workspace, cx| {
            let _subscriptions = subscriptions;
            let result = async {
                let review = review?;
                let revoked = review.revoked.clone();
                let pipeline = run_linux_native_cargo_profile(
                    &workspace,
                    &review,
                    launch,
                    build_handle,
                    prerequisite_lock,
                    cx,
                );
                let result = match futures::future::select(Box::pin(pipeline), revoked).await {
                    futures::future::Either::Left((result, _)) => result,
                    futures::future::Either::Right(_) => Err(crate::TaskLaunchCancelled.into()),
                };
                workspace.update(cx, |workspace, cx| {
                    if result.is_err()
                        && let Some(pending) = workspace.pending_native_profile_review_mut(&review)
                        && let Some(handle) = pending.supersession_handle.as_ref()
                    {
                        handle.cancel(cx);
                    }
                    workspace.retire_native_profile_review(&review);
                })?;
                result
            }
            .await;
            let status = match result {
                Ok(()) => ScheduledTaskResult::Success,
                Err(error) if error.is::<crate::TaskLaunchCancelled>() => {
                    ScheduledTaskResult::Cancelled
                }
                Err(error) => {
                    log::warn!("Native Linux Cargo profiling failed: {error:#}");
                    if let Err(error) = workspace.update(cx, |workspace, cx| {
                        workspace.show_toast(
                            Toast::new(
                                NotificationId::unique::<TaskInvocationContext>(),
                                format!("Native Linux Cargo profiling failed: {error:#}"),
                            ),
                            cx,
                        );
                    }) {
                        log::debug!("Native Linux Cargo profile workspace closed: {error:#}");
                    }
                    ScheduledTaskResult::Failure
                }
            };
            if let Some(on_complete) = on_complete {
                on_complete(status, cx);
            }
        });
        self.scheduled_tasks.push(task);
    }
}

fn watch_linux_profile_review(
    workspace: &Workspace,
    review: &NativeProfileReviewLease,
    build: &SpawnInTerminal,
    cx: &mut Context<Workspace>,
) -> Vec<gpui::Subscription> {
    let cancel = {
        let task_id = review.task_id.clone();
        let run_id = review.run_id;
        move |workspace: &mut Workspace, cx: &mut Context<Workspace>| {
            if let Some(review) = workspace.pending_native_profile_reviews.get(&task_id)
                && review.run_id == run_id
            {
                revoke_native_profile_review(&review.current, &review.revocation);
                if let Some(handle) = review.supersession_handle.as_ref() {
                    handle.cancel(cx);
                }
            }
        }
    };
    let mut subscriptions = vec![
        cx.subscribe(&workspace.project, {
            let cancel = cancel.clone();
            move |workspace, _, event, cx| {
                if matches!(
                    event,
                    project::Event::RemoteIdChanged(_)
                        | project::Event::WorktreeRemoved(_)
                        | project::Event::WorktreePathsChanged { .. }
                        | project::Event::Closed
                ) {
                    cancel(workspace, cx);
                }
            }
        }),
        cx.observe_global::<feature_flags::FeatureFlagStore>({
            let cancel = cancel.clone();
            let required_features = build.required_experimental_features.clone();
            move |workspace, cx| {
                if required_features
                    .iter()
                    .any(|feature| !cx.has_flag_by_name(feature))
                {
                    cancel(workspace, cx);
                }
            }
        }),
        cx.observe_global::<settings::SettingsStore>({
            let cancel = cancel.clone();
            let revision = review.settings_revision;
            move |workspace, cx| {
                if cx.global::<settings::SettingsStore>().observed_revision() != revision {
                    cancel(workspace, cx);
                }
            }
        }),
    ];
    if let Some(trust) = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
        subscriptions.push(cx.subscribe(&trust, move |workspace, _, _, cx| {
            if project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                &workspace.project.read(cx).worktree_store(),
                cx,
            ) {
                cancel(workspace, cx);
            }
        }));
    }
    subscriptions
}

async fn run_linux_native_cargo_profile(
    workspace: &WeakEntity<Workspace>,
    review: &NativeProfileReviewLease,
    launch: LinuxNativeCargoProfileLaunch,
    build_handle: Option<StructuredTaskHandle>,
    prerequisite_lock: Rc<futures::lock::Mutex<()>>,
    cx: &mut AsyncWindowContext,
) -> Result<()> {
    let LinuxNativeCargoProfileLaunch {
        run_id,
        build,
        report,
        pending,
        artifact_current,
        artifact_subscriptions,
    } = launch;
    if let Some(retirement) = &review.predecessor_retirement {
        retirement.clone().await;
    }
    let continuation = report
        .native_profile
        .context("Cargo Profile preset is missing")?;
    let group = continuation.bound_artifact_group(&run_id.to_string())?;
    let presentation = (build.reveal, build.reveal_target);
    let (project, directory) = workspace.read_with(cx, |workspace, cx| {
        check_linux_native_profile(workspace, review, cx)?;
        let project = workspace.project.read(cx);
        let directory = project
            .terminal_task_working_directory(build.cwd.as_deref(), cx)
            .context("Native Cargo Profile needs an absolute project directory")?;
        Ok::<_, anyhow::Error>((workspace.project.clone(), directory.to_path_buf()))
    })??;
    let runner = cx
        .background_spawn(async {
            let executable = std::env::current_exe()?;
            let parent = executable
                .parent()
                .context("Copper installation has no libexec directory")?;
            let runner = parent.join("native-cargo-runtime-observer");
            anyhow::ensure!(
                parent.ends_with("libexec") && runner.is_file(),
                "Install the Linux Copper package with its native Cargo runtime observer"
            );
            Ok::<_, anyhow::Error>(runner)
        })
        .await?;
    let build = {
        let _guard = prerequisite_lock.lock().await;
        Workspace::save_for_task_checked(workspace, build.save, cx).await?;
        let handle = workspace.update(cx, |workspace, cx| {
            check_linux_native_profile(workspace, review, cx)?;
            if let Some(pending) = pending.as_ref() {
                pending.check(&build, cx)?;
            }
            let handle =
                build_handle.unwrap_or_else(|| StructuredTaskHandle::new(build.id.clone()));
            register_linux_profile_phase_handle(workspace, review, &handle, cx)?;
            Ok::<_, anyhow::Error>(handle)
        })??;
        project.update(cx, |project, cx| {
            project.start_selected_linux_native_cargo_build_task(
                run_id,
                build,
                Some(handle),
                report.max_bytes,
                cx,
            )
        })?
    };
    let workload = await_linux_profile_phase(workspace, build.await?, presentation, cx).await?;
    let handle = next_linux_profile_phase(workspace, review, cx)?;
    let runtime = project
        .update(cx, |project, cx| {
            project.start_selected_linux_native_cargo_runtime_task(
                workload,
                runner,
                continuation.workload_arguments,
                Some(handle),
                cx,
            )
        })?
        .await?;
    let runtime = await_linux_profile_phase(workspace, runtime, presentation, cx).await?;

    let paths = workspace.read_with(cx, |workspace, cx| {
        check_linux_native_profile(workspace, review, cx)?;
        native_profile_capture_disclosure_paths(
            workspace,
            &group,
            Some(&directory),
            &review.current,
            cx,
        )
    })??;
    let destination = cx
        .background_spawn({
            let paths = paths.clone();
            async move { NativeProfileTaskDestination::open(paths) }
        })
        .await
        .context("Prepare an existing project directory for the profile recording and symbols")?;
    confirm_native_profile_capture(
        workspace,
        &group,
        Some(&directory),
        &review.current,
        &review.revoked,
        cx,
    )
    .await?;
    let handle = next_linux_profile_phase(workspace, review, cx)?;
    let capture = project
        .update(cx, |project, cx| {
            project.start_selected_linux_native_cargo_profile_task(
                runtime,
                continuation.collector_path.into(),
                Some(handle),
                cx,
            )
        })?
        .await?;
    let profile = await_linux_profile_phase(workspace, capture, presentation, cx).await?;
    workspace.read_with(cx, |workspace, cx| {
        check_linux_native_profile(workspace, review, cx)?;
        anyhow::ensure!(
            native_profile_capture_disclosure_paths(
                workspace,
                &group,
                Some(&directory),
                &review.current,
                cx
            )? == paths,
            "Native Cargo Profile output paths changed"
        );
        Ok::<_, anyhow::Error>(())
    })??;
    let retained = cx
        .background_spawn({
            let current = review.current.clone();
            let artifact_current = artifact_current.clone();
            async move {
                profile.retain(destination, || {
                    current.load(Ordering::Acquire) && artifact_current.load(Ordering::Acquire)
                })
            }
        })
        .await?;
    workspace
        .update(cx, |workspace, cx| {
            workspace.refresh_task_artifacts(group.files.values(), Some(&directory), cx)
        })??
        .await?;
    workspace.update(cx, |workspace, cx| {
        check_linux_native_profile(workspace, review, cx)?;
        for (role, metadata) in ["profile", "symbols"].into_iter().zip(retained) {
            let project = workspace.project.read(cx);
            let artifact = group
                .files
                .get(role)
                .context("Native Cargo Profile output role is missing")?;
            let path = resolve_task_artifact(project, artifact, Some(&directory), cx)?;
            let entry = project
                .entry_for_path(&path, cx)
                .context("Native Cargo Profile output is missing")?;
            let modified = metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?;
            anyhow::ensure!(
                entry.inode == metadata.ino()
                    && entry.size == metadata.len()
                    && entry.mtime
                        == Some(fs::MTime::from_seconds_and_nanos(
                            modified.as_secs(),
                            modified.subsec_nanos()
                        )),
                "Native Cargo Profile output changed after retention"
            );
        }
        workspace.publish_task_artifact_group(
            run_id,
            group,
            Some(&directory),
            true,
            artifact_current,
            artifact_subscriptions,
            cx,
        )
    })?
}

fn check_linux_native_profile(
    workspace: &Workspace,
    review: &NativeProfileReviewLease,
    cx: &gpui::App,
) -> Result<()> {
    if !workspace.native_profile_review_is_current(review, cx) {
        return Err(crate::TaskLaunchCancelled.into());
    }
    let project = workspace.project.read(cx);
    anyhow::ensure!(
        project.is_local() && !project.is_shared() && !project.is_disconnected(cx),
        "Native Linux Cargo Profile requires an unshared connected local project"
    );
    Ok(())
}

fn next_linux_profile_phase(
    workspace: &WeakEntity<Workspace>,
    review: &NativeProfileReviewLease,
    cx: &mut AsyncWindowContext,
) -> Result<StructuredTaskHandle> {
    workspace.update(cx, |workspace, cx| {
        check_linux_native_profile(workspace, review, cx)?;
        let handle = StructuredTaskHandle::new(review.task_id.clone());
        register_linux_profile_phase_handle(workspace, review, &handle, cx)?;
        Ok(handle)
    })?
}

fn register_linux_profile_phase_handle(
    workspace: &mut Workspace,
    review: &NativeProfileReviewLease,
    handle: &StructuredTaskHandle,
    cx: &mut gpui::App,
) -> Result<()> {
    let pending = workspace
        .pending_native_profile_review_mut(review)
        .context("The Cargo Profile review ended")?;
    pending.supersession_handle = Some(handle.clone());
    let current = review.current.clone();
    let revocation = review.revocation.clone();
    handle.subscribe(cx, move |event, _| {
        if matches!(event.state, task::StructuredTaskState::Cancelled { .. }) {
            revoke_native_profile_review(&current, &revocation);
        }
    });
    Ok(())
}

async fn await_linux_profile_phase<Output: 'static>(
    workspace: &WeakEntity<Workspace>,
    phase: LinuxNativeCargoTask<Output>,
    (reveal, target): (task::RevealStrategy, task::RevealTarget),
    cx: &mut AsyncWindowContext,
) -> Result<Output> {
    workspace
        .update_in(cx, |workspace, window, cx| {
            let provider = workspace
                .terminal_provider
                .as_ref()
                .context("Task terminal is unavailable")?;
            Ok::<_, anyhow::Error>(provider.attach_task_terminal(
                phase.terminal.clone(),
                reveal,
                target,
                window,
                cx,
            ))
        })??
        .await?;
    let ordinary = phase
        .terminal
        .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
    let (output, status) = futures::future::join(phase.completion, ordinary).await;
    let output = output?;
    anyhow::ensure!(
        status.is_some_and(|status| status.success()),
        "Native Cargo Profile Task did not complete successfully"
    );
    Ok(output)
}
