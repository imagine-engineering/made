use std::time::Duration;

use project::{
    ProjectPath, cargo_workspace_store::HostCargoProfileSelection,
    profile_results_store::RemoteProfileError,
};
use remote::RemoteClientEvent;

use super::*;

impl Workspace {
    pub fn schedule_remote_linux_cargo_profile(
        &mut self,
        selection: HostCargoProfileSelection,
        save: SaveStrategy,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let workspace = cx.entity().downgrade();
        let project = self.project.clone();
        let remote = project
            .read(cx)
            .remote_client()
            .context("The Linux host disconnected")?;
        let id = uuid::Uuid::new_v4();
        let review = self.begin_native_profile_review(
            TaskId("remote-linux-cargo-profile".into()),
            id,
            Arc::new(AtomicBool::new(true)),
            None,
            cx.try_global::<settings::SettingsStore>()
                .and_then(|settings| settings.observed_revision()),
            cx,
        )?;
        let invalidate = {
            let current = review.current.clone();
            let artifact_current = review.artifact_current.clone();
            let revocation = review.revocation.clone();
            move || {
                artifact_current.store(false, Ordering::Relaxed);
                revoke_native_profile_review(&current, &revocation);
            }
        };
        let mut subscriptions = vec![
            cx.subscribe(&remote, {
                let invalidate = invalidate.clone();
                move |_, _, event, _| {
                    if matches!(event, RemoteClientEvent::Disconnected { .. } | RemoteClientEvent::ConnectionStateChanged(_)) {
                        invalidate();
                    }
                }
            }),
            cx.subscribe(&project, {
                let invalidate = invalidate.clone();
                move |_, _, event, _| {
                    if matches!(event, project::Event::RemoteIdChanged(_) | project::Event::WorktreeRemoved(_) | project::Event::WorktreePathsChanged { .. } | project::Event::Closed) {
                        invalidate();
                    }
                }
            }),
            cx.observe_global::<feature_flags::FeatureFlagStore>({
                let invalidate = invalidate.clone();
                move |_, cx| {
                    if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
                        || !cx.has_flag::<feature_flags::RustNamedCargoPresetsExperimentalFeatureFlag>()
                    {
                        invalidate();
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
                    invalidate();
                }
            }));
        }
        let settings = cx.observe_global::<settings::SettingsStore>({
            let revision = review.settings_revision;
            let current = review.current.clone();
            let revocation = review.revocation.clone();
            move |_, cx| {
                if cx.global::<settings::SettingsStore>().observed_revision() != revision {
                    revoke_native_profile_review(&current, &revocation);
                }
            }
        });
        let notice = NotificationId::named(format!("remote-linux-cargo-profile-{id}").into());
        self.show_toast(
            remote_profile_progress(
                &review,
                notice.clone(),
                "Preparing native profile on the Linux host",
            ),
            cx,
        );
        let task = cx.spawn_in(window, async move |_, cx| {
            let _settings = settings;
            let revoked = review.revoked.clone();
            let pipeline = run_remote_linux_cargo_profile(
                &workspace,
                &project,
                &review,
                selection,
                save,
                (&notice, subscriptions),
                cx,
            );
            let result = match futures::future::select(pipeline.boxed_local(), revoked).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => Err(crate::TaskLaunchCancelled.into()),
            };
            if let Err(error) = workspace.update(cx, |workspace, cx| {
                workspace.dismiss_toast(&notice, cx);
                workspace.retire_native_profile_review(&review);
                if let Err(error) = result
                    && !error.is::<crate::TaskLaunchCancelled>()
                    && error.downcast_ref::<RemoteProfileError>()
                        != Some(&RemoteProfileError::Cancelled)
                {
                    workspace.show_toast(
                        Toast::new(notice, format!("Native Linux profiling failed: {error:#}")),
                        cx,
                    );
                }
            }) {
                log::debug!("Remote Linux profile owner closed: {error:#}");
            }
        });
        self.scheduled_tasks.push(task);
        Ok(())
    }
}

fn remote_profile_progress(
    review: &NativeProfileReviewLease,
    notice: NotificationId,
    label: &'static str,
) -> Toast {
    let current = review.current.clone();
    let revocation = review.revocation.clone();
    Toast::new(notice, label).on_click("Cancel", move |_, _| {
        revoke_native_profile_review(&current, &revocation);
    })
}

async fn run_remote_linux_cargo_profile(
    workspace: &WeakEntity<Workspace>,
    project: &Entity<project::Project>,
    review: &NativeProfileReviewLease,
    selection: HostCargoProfileSelection,
    save: SaveStrategy,
    presentation: (&NotificationId, Vec<gpui::Subscription>),
    cx: &mut AsyncWindowContext,
) -> Result<()> {
    let (notice, subscriptions) = presentation;
    if let Some(predecessor) = &review.predecessor_retirement {
        predecessor.clone().await;
    }
    Workspace::save_for_task_checked(workspace, save, cx).await?;
    ensure_review_current(workspace, review, cx)?;
    let prepared = project
        .read_with(cx, |project, cx| {
            project.prepare_remote_linux_cargo_profile(selection, review.run_id, cx)
        })?
        .await?;
    ensure_review_current(workspace, review, cx)?;
    let [profile, symbols] = prepared.paths();
    let group = remote_profile_group(profile, symbols)?;
    let directory = project
        .read_with(cx, |project, cx| {
            project.absolute_path(&ProjectPath::root_path(profile.worktree_id), cx)
        })
        .context("The Linux profile worktree is unavailable")?;
    workspace.update(cx, |workspace, cx| workspace.dismiss_toast(notice, cx))?;
    confirm_native_profile_capture(
        workspace,
        &group,
        Some(&directory),
        &review.current,
        &review.revoked,
        cx,
    )
    .await?;
    ensure_review_current(workspace, review, cx)?;
    workspace.update(cx, |workspace, cx| {
        workspace.show_toast(
            remote_profile_progress(
                review,
                notice.clone(),
                "Capturing native profile on the Linux host",
            ),
            cx,
        )
    })?;
    let completed = prepared.capture().await?;
    ensure_review_current(workspace, review, cx)?;
    wait_for_remote_profile_entries(workspace, project, &completed.paths, cx).await?;
    workspace.update(cx, |workspace, cx| {
        anyhow::ensure!(
            workspace.native_profile_review_is_current(review, cx),
            crate::TaskLaunchCancelled
        );
        workspace.publish_task_artifact_group(
            completed.run_id,
            group,
            Some(&directory),
            true,
            review.artifact_current.clone(),
            subscriptions,
            cx,
        )
    })?
}

fn ensure_review_current(
    workspace: &WeakEntity<Workspace>,
    review: &NativeProfileReviewLease,
    cx: &AsyncWindowContext,
) -> Result<()> {
    anyhow::ensure!(
        workspace.read_with(cx, |workspace, cx| workspace
            .native_profile_review_is_current(review, cx))?,
        crate::TaskLaunchCancelled
    );
    Ok(())
}

fn remote_profile_group(profile: &ProjectPath, symbols: &ProjectPath) -> Result<TaskArtifactGroup> {
    let group = TaskArtifactGroup {
        format: "samply-0.13.1".into(),
        files: [
            ("profile", profile, task::MAX_TASK_ARTIFACT_BYTES),
            ("symbols", symbols, 16 * 1024 * 1024),
        ]
        .into_iter()
        .map(|(role, path, max_bytes)| {
            (
                role.into(),
                TaskArtifact {
                    path: path.path.as_unix_str().into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes,
                },
            )
        })
        .collect(),
    };
    group.with_resolved_files(group.files.clone())
}

async fn wait_for_remote_profile_entries(
    workspace: &WeakEntity<Workspace>,
    project: &Entity<project::Project>,
    paths: &[ProjectPath; 2],
    cx: &mut AsyncWindowContext,
) -> Result<()> {
    use gpui::FutureExt as _;
    let (mut changed, mut changes) = futures::channel::mpsc::channel(1);
    let _subscription = workspace.update(cx, |_, cx| {
        cx.subscribe(project, move |_, _, event, _| {
            if matches!(event, project::Event::WorktreeUpdatedEntries(..))
                && let Err(error) = changed.try_send(())
            {
                log::trace!("Remote profile worktree update coalesced: {error}");
            }
        })
    })?;
    let indexed = async {
        while !project.read_with(cx, |project, cx| {
            paths
                .iter()
                .all(|path| project.entry_for_path(path, cx).is_some())
        }) {
            changes
                .next()
                .await
                .context("The remote worktree stopped indexing")?;
        }
        anyhow::Ok(())
    };
    indexed
        .with_timeout(Duration::from_secs(10), cx.background_executor())
        .await
        .context("The Linux profile files did not appear in the worktree")??;
    Ok(())
}
