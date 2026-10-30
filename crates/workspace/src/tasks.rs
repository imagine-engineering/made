use feature_flags::{
    FeatureFlagAppExt as _, RustCoverageExperimentalFeatureFlag,
    RustNativeProfilingExperimentalFeatureFlag,
};
use std::{
    path::{Path, PathBuf},
    process::ExitStatus,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context as _, Result, anyhow, bail};
use collections::{BTreeMap, HashSet};
use futures::{FutureExt as _, StreamExt as _};
use gpui::{AppContext, AsyncWindowContext, Context, Entity, Task, TaskExt, WeakEntity};
use language::Buffer;
use project::{TaskSourceKind, WorktreeId};
use remote::ConnectionState;
use task::{
    DebugScenario, ResolvedTask, SaveStrategy, SharedTaskContext, SpawnInTerminal,
    StructuredTaskHandle, TaskArtifact, TaskArtifactGroup, TaskContext, TaskHook, TaskId,
    TaskTemplate, TaskVariables, VariableName,
};
use ui::Window;
use util::TryFutureExt;

use crate::{SaveIntent, Toast, Workspace, notifications::NotificationId};

#[cfg(any(test, feature = "test-support"))]
fn native_capture_diagnostic_category(error: &anyhow::Error) -> &'static str {
    if error.is::<crate::TaskLaunchCancelled>() {
        return "cancelled";
    }
    if error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::TimedOut)
    }) {
        return "io-timeout";
    }
    let message: String = error.to_string().chars().take(1024).collect();
    if message.contains("timed out") || message.contains("registration expired") {
        "deadline-message"
    } else {
        "refused"
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduledTaskResult {
    Success,
    Failure,
    SpawnFailed,
    Cancelled,
}

#[derive(Clone)]
pub struct TaskInvocationContext {
    run_id: uuid::Uuid,
    task_id: TaskId,
    command: Option<String>,
    arguments: Vec<String>,
    working_directory: Option<PathBuf>,
    artifact_directory: Option<PathBuf>,
    artifact: Option<TaskArtifact>,
    artifact_group: Option<TaskArtifactGroup>,
}

impl TaskInvocationContext {
    fn from_launch(
        run_id: uuid::Uuid,
        launch: &SpawnInTerminal,
        artifact: Option<TaskArtifact>,
        artifact_group: Option<TaskArtifactGroup>,
    ) -> Self {
        Self {
            run_id,
            task_id: launch.id.clone(),
            command: launch.command.clone(),
            arguments: launch.args.clone(),
            working_directory: launch.cwd.clone(),
            artifact_directory: launch.cwd.clone(),
            artifact,
            artifact_group,
        }
    }

    pub fn run_id(&self) -> uuid::Uuid {
        self.run_id
    }

    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    pub fn command(&self) -> Option<&str> {
        self.command.as_deref()
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    pub fn working_directory(&self) -> Option<&Path> {
        self.working_directory.as_deref()
    }

    pub fn artifact_directory(&self) -> Option<&Path> {
        self.artifact_directory.as_deref()
    }

    pub fn artifact(&self) -> Option<&TaskArtifact> {
        self.artifact.as_ref()
    }

    pub fn artifact_group(&self) -> Option<&TaskArtifactGroup> {
        self.artifact_group.as_ref()
    }
}

#[cfg(feature = "jvm-tools")]
fn groovy_jvm_worktree_trusted(
    project: &Entity<project::Project>,
    worktree_id: WorktreeId,
    cx: &mut gpui::App,
) -> bool {
    let worktree_store = project.read(cx).worktree_store();
    if worktree_store
        .read(cx)
        .worktree_for_id(worktree_id, cx)
        .is_none()
    {
        return false;
    }
    let Some(trusted_worktrees) = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
    else {
        return false;
    };
    trusted_worktrees.update(cx, |trusted_worktrees, cx| {
        trusted_worktrees.can_trust(&worktree_store, worktree_id, cx)
    }) && !project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
        &worktree_store,
        cx,
    )
}

#[cfg(feature = "jvm-tools")]
fn groovy_jvm_operation_block_reason(
    project: &Entity<project::Project>,
    worktree_id: WorktreeId,
    resolved_task: &ResolvedTask,
    cx: &mut gpui::App,
) -> Option<&'static str> {
    if !groovy_jvm_worktree_trusted(project, worktree_id, cx) {
        return Some("Trust the project before running a JVM Task");
    }
    let project = project.read(cx);
    if project.is_remote() || project.remote_connection_state(cx).is_some() {
        return Some(
            "Discovered build operations require a local host; remote execution is not qualified",
        );
    }
    let lsp_store = project.lsp_store();
    let Some((snapshot, current)) = lsp_store.read(cx).jdtls_project_snapshot(worktree_id) else {
        return Some(
            "Build operation metadata is unavailable; refresh and select the operation again",
        );
    };
    let task_store = project.task_store();
    let Some(inventory) = task_store.read(cx).task_inventory() else {
        return Some("The build Task inventory is unavailable; reopen the project");
    };
    if !inventory.read(cx).jvm_operation_is_current(
        worktree_id,
        resolved_task,
        snapshot,
        current,
        cx,
    ) {
        return Some(
            "The build operation or project files changed; refresh and select the operation again",
        );
    }
    None
}

#[cfg(feature = "jvm-tools")]
fn groovy_jvm_task_block_reason(
    project: &Entity<project::Project>,
    worktree_id: WorktreeId,
    build_task: Option<&ResolvedTask>,
    cx: &mut gpui::App,
) -> Option<&'static str> {
    if !groovy_jvm_worktree_trusted(project, worktree_id, cx) {
        return Some("Trust the project before running a JVM Task");
    }
    let Some(resolved_task) = build_task else {
        return None;
    };
    let task_store = project.read(cx).task_store();
    let Some(inventory) = task_store.read(cx).task_inventory().cloned() else {
        return Some("The JVM build project is no longer available");
    };
    if !inventory
        .read(cx)
        .groovy_jvm_build_task_is_current(worktree_id, resolved_task, cx)
    {
        return Some("The JVM build Task no longer matches the current project wrapper");
    }
    None
}

pub struct CompletedTaskArtifactGroup {
    run_id: uuid::Uuid,
    group: task::TaskArtifactGroup,
    files: BTreeMap<String, (project::ProjectPath, project::Entry)>,
    workspace: WeakEntity<Workspace>,
    project_id: gpui::EntityId,
    host_verified: bool,
    current: Arc<AtomicBool>,
    _subscriptions: Vec<gpui::Subscription>,
}

impl CompletedTaskArtifactGroup {
    pub fn run_id(&self) -> uuid::Uuid {
        self.run_id
    }

    pub fn group(&self) -> &task::TaskArtifactGroup {
        &self.group
    }

    pub fn path(&self, role: &str) -> Option<&project::ProjectPath> {
        self.files.get(role).map(|(path, _)| path)
    }

    pub fn entry(&self, role: &str) -> Option<&project::Entry> {
        self.files.get(role).map(|(_, entry)| entry)
    }

    pub fn host_verified(&self) -> bool {
        self.host_verified
    }

    pub fn is_current(&self, project: &Entity<project::Project>, cx: &gpui::App) -> bool {
        if project.entity_id() != self.project_id {
            return false;
        }
        let project = project.read(cx);
        self.current.load(Ordering::Relaxed)
            && self.workspace.upgrade().is_some()
            && !project.is_disconnected(cx)
            && structured_task_connection_error(project.remote_connection_state(cx)).is_none()
            && !project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                &project.worktree_store(),
                cx,
            )
            && self.files.values().all(|(path, expected)| {
                project
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == path.worktree_id)
                    && project.entry_for_path(path, cx).is_some_and(|entry| {
                        entry.is_file()
                            && !entry.is_private
                            && !entry.is_external
                            && !entry.is_fifo
                            && entry.id == expected.id
                            && entry.inode == expected.inode
                            && entry.mtime == expected.mtime
                            && entry.size == expected.size
                            && entry.canonical_path == expected.canonical_path
                    })
            })
    }
}

type NativeProfileRetirement = futures::future::Shared<futures::future::BoxFuture<'static, ()>>;
type NativeProfileRevocation =
    Arc<parking_lot::Mutex<Option<futures::channel::oneshot::Sender<()>>>>;

fn revoke_native_profile_review(current: &AtomicBool, revocation: &NativeProfileRevocation) {
    current.store(false, Ordering::Relaxed);
    if let Some(sender) = revocation.lock().take()
        && sender.send(()).is_err()
    {
        log::trace!("Native capture revocation has no pending waiter");
    }
}

const MAX_NATIVE_PROFILE_PROMPT_RECEIPTS: usize = 32;

struct NativeProfilePromptPermit(Arc<std::sync::atomic::AtomicUsize>);

impl NativeProfilePromptPermit {
    fn reserve(slots: Arc<std::sync::atomic::AtomicUsize>) -> Result<Self> {
        slots.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
            used.checked_add(1).filter(|count| *count <= MAX_NATIVE_PROFILE_PROMPT_RECEIPTS)
        }).map_err(|_| anyhow!("Dismiss an existing native capture prompt before opening another; at most 32 unanswered prompts can be retained"))?;
        Ok(Self(slots))
    }
}

impl Drop for NativeProfilePromptPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

const MAX_NATIVE_PROFILE_RETIREMENT_LEASES: usize = 32;

struct NativeProfileRetirementPermit(Arc<std::sync::atomic::AtomicUsize>);

impl NativeProfileRetirementPermit {
    fn reserve(slots: Arc<std::sync::atomic::AtomicUsize>) -> Result<Arc<Self>> {
        slots.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
            used.checked_add(1).filter(|count| *count <= MAX_NATIVE_PROFILE_RETIREMENT_LEASES)
        }).map_err(|_| anyhow!("Wait for cancelled native Tasks to retire before starting another; at most 32 native Task retirement leases can be retained"))?;
        Ok(Arc::new(Self(slots)))
    }
}

impl Drop for NativeProfileRetirementPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

struct NativeProfileRetirementSignal {
    sender: Option<futures::channel::oneshot::Sender<()>>,
    permit: Option<Arc<NativeProfileRetirementPermit>>,
}

impl Drop for NativeProfileRetirementSignal {
    fn drop(&mut self) {
        drop(self.permit.take());
        if let Some(sender) = self.sender.take()
            && sender.send(()).is_err()
        {
            log::trace!("Native Task retirement has no pending replacement");
        }
    }
}

type NativeProfileRetirementLease = Arc<NativeProfileRetirementSignal>;

struct NativeProfileRetirementOwner<T> {
    owned: Option<T>,
    retirement: Option<NativeProfileRetirementLease>,
}

impl<T> Drop for NativeProfileRetirementOwner<T> {
    fn drop(&mut self) {
        drop(self.owned.take());
        drop(self.retirement.take());
    }
}

struct NativeProfileRetirementFuture<F> {
    future: Option<std::pin::Pin<Box<F>>>,
    retirement: Option<NativeProfileRetirementLease>,
}

impl<F> Drop for NativeProfileRetirementFuture<F> {
    fn drop(&mut self) {
        // A GPUI Task handle can schedule destruction; only the actual future releases this lease.
        drop(self.future.take());
        drop(self.retirement.take());
    }
}

impl<F: std::future::Future> std::future::Future for NativeProfileRetirementFuture<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        let Some(future) = this.future.as_mut() else {
            return std::task::Poll::Pending;
        };
        match future.as_mut().poll(context) {
            std::task::Poll::Ready(result) => {
                drop(this.future.take());
                drop(this.retirement.take());
                std::task::Poll::Ready(result)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

fn spawn_native_profile_retiring<R: 'static>(
    cx: &AsyncWindowContext,
    retirement: Option<NativeProfileRetirementLease>,
    work: impl AsyncFnOnce(&mut AsyncWindowContext) -> R + 'static,
) -> Task<R> {
    let mut context = cx.clone();
    let future = NativeProfileRetirementFuture {
        future: Some(Box::pin(async move { work(&mut context).await })),
        retirement,
    };
    cx.spawn(async move |_| future.await)
}

fn native_profile_retiring_preparation(
    preparation: crate::TaskLaunchPreparation,
    retirement: Option<NativeProfileRetirementLease>,
) -> crate::TaskLaunchPreparation {
    let owner = NativeProfileRetirementOwner {
        owned: Some(preparation),
        retirement,
    };
    Box::new(move |launch, cx| {
        let mut owner = owner;
        let Some(preparation) = owner.owned.take() else {
            return Task::ready(Err(anyhow!("Native Task preparation was already retired")));
        };
        let mut waiting = NativeProfileRetirementOwner {
            owned: Some(preparation(launch, cx)),
            retirement: owner.retirement.take(),
        };
        drop(owner);
        cx.spawn(async move |_| {
            let prepared = waiting
                .owned
                .as_mut()
                .context("Native Task preparation disappeared")?
                .await?;
            let retirement = waiting.retirement.take();
            drop(waiting);
            let owner = NativeProfileRetirementOwner {
                owned: Some(prepared.completion),
                retirement,
            };
            let completion: crate::TaskLaunchCompletion = Box::new(move |spawned, cx| {
                let mut owner = owner;
                let Some(completion) = owner.owned.take() else {
                    return Task::ready(Some(Err(anyhow!(
                        "Native Task completion was already retired"
                    ))));
                };
                let completed = match completion {
                    Some(completion) => completion(spawned, cx),
                    None => spawned,
                };
                let mut waiting = NativeProfileRetirementOwner {
                    owned: Some(completed),
                    retirement: owner.retirement.take(),
                };
                drop(owner);
                cx.spawn(async move |_| {
                    let result = match waiting.owned.as_mut() {
                        Some(completed) => completed.await,
                        None => Some(Err(anyhow!("Native Task completion disappeared"))),
                    };
                    drop(waiting);
                    result
                })
            });
            Ok(crate::PreparedTaskLaunch {
                launch: prepared.launch,
                completion: Some(completion),
                native_terminal: prepared.native_terminal,
                pending_launch: prepared.pending_launch,
            })
        })
    })
}

pub(crate) struct PendingNativeProfileReview {
    supersession_handle: Option<StructuredTaskHandle>,
    retirement: NativeProfileRetirement,
    revocation: NativeProfileRevocation,
    run_id: uuid::Uuid,
    current: Arc<AtomicBool>,
    artifact_current: Arc<AtomicBool>,
    project_id: gpui::EntityId,
    settings_revision: Option<u64>,
}

struct NativeProfileReviewLease {
    retirement_permit: Arc<NativeProfileRetirementPermit>,
    predecessor_retirement: Option<NativeProfileRetirement>,
    retirement: Option<futures::channel::oneshot::Sender<()>>,
    revoked: NativeProfileRetirement,
    revocation: NativeProfileRevocation,
    task_id: TaskId,
    run_id: uuid::Uuid,
    project_id: gpui::EntityId,
    current: Arc<AtomicBool>,
    artifact_current: Arc<AtomicBool>,
    structured_handle: Option<StructuredTaskHandle>,
    settings_revision: Option<u64>,
}

impl Drop for NativeProfileReviewLease {
    fn drop(&mut self) {
        // Dropping a scheduled future must retire its held answer without revoking a launched Task.
        revoke_native_profile_review(&self.current, &self.revocation);
        if let Some(sender) = self.retirement.take()
            && sender.send(()).is_err()
        {
            log::trace!("Unprepared native Task retirement has no pending replacement");
        }
    }
}

impl Drop for PendingNativeProfileReview {
    fn drop(&mut self) {
        revoke_native_profile_review(&self.current, &self.revocation);
    }
}

fn native_build_structured_handle(
    supplied: Option<StructuredTaskHandle>,
    build_task_id: &TaskId,
    report: Option<&task::TaskStdoutReport>,
) -> Option<StructuredTaskHandle> {
    supplied.or_else(|| {
        report
            .filter(|report| report.native_profile.is_some())
            .map(|_| StructuredTaskHandle::new(build_task_id.clone()))
    })
}

fn observe_native_build_review_lifecycle(
    handle: &StructuredTaskHandle,
    current: Arc<AtomicBool>,
    cx: &mut gpui::App,
) {
    handle.subscribe(cx, move |event, _| {
        if matches!(
            event.state,
            task::StructuredTaskState::Cancelled { .. }
                | task::StructuredTaskState::SpawnError { .. }
                | task::StructuredTaskState::Completed { success: false, .. }
        ) {
            current.store(false, Ordering::Relaxed);
        }
    });
}

fn native_capture_phase_task_id(build_task_id: &TaskId) -> TaskId {
    TaskId(format!("{}:native-capture", build_task_id.0))
}

type TaskCompletionHandler = Box<dyn FnOnce(ScheduledTaskResult, &mut AsyncWindowContext)>;
type TaskAdmissionHandler =
    Box<dyn FnOnce(&TaskInvocationContext, &mut Workspace, &mut Context<Workspace>) -> Result<()>>;
type TaskAdmissionPreflight = Box<dyn FnOnce(&Workspace, &gpui::App) -> bool>;

pub fn resolve_task_reference(
    tasks: Vec<(TaskSourceKind, TaskTemplate)>,
    label: &str,
    task_context: &TaskContext,
) -> Result<(TaskSourceKind, ResolvedTask)> {
    let mut matches = tasks
        .into_iter()
        .filter(|(_, template)| template.label == label);
    let (source, template) = matches
        .next()
        .with_context(|| format!("pre-launch task `{label}` was not found"))?;
    if matches.next().is_some() {
        bail!("pre-launch task reference `{label}` is ambiguous");
    }
    let resolved = template
        .resolve_task(&source.to_id_base(), task_context)
        .with_context(|| format!("pre-launch task `{label}` could not be resolved"))?;
    Ok((source, resolved))
}

impl Workspace {
    fn begin_native_profile_review(
        &mut self,
        task_id: TaskId,
        run_id: uuid::Uuid,
        artifact_current: Arc<AtomicBool>,
        structured_handle: Option<StructuredTaskHandle>,
        settings_revision: Option<u64>,
        cx: &mut gpui::App,
    ) -> Result<NativeProfileReviewLease> {
        let project_id = self.project.entity_id();
        let current_revision = cx
            .try_global::<settings::SettingsStore>()
            .and_then(|settings| settings.observed_revision());
        let retirement_permit =
            NativeProfileRetirementPermit::reserve(self.native_profile_retirement_slots.clone())?;
        let previous = self.pending_native_profile_reviews.remove(&task_id);
        if let Some(previous) = previous.as_ref() {
            revoke_native_profile_review(&previous.current, &previous.revocation);
            if let Some(handle) = previous.supersession_handle.as_ref() {
                handle.cancel(cx);
            }
        }
        let predecessor_retirement = previous
            .as_ref()
            .filter(|previous| previous.retirement.peek().is_none())
            .map(|previous| previous.retirement.clone());
        drop(previous);
        self.pending_native_profile_reviews.retain(|_, review| {
            review.current.load(Ordering::Relaxed)
                && review.artifact_current.load(Ordering::Relaxed)
                && review.project_id == project_id
                && review.settings_revision == current_revision
        });
        anyhow::ensure!(
            self.pending_native_profile_reviews.len() < 16
                || self.pending_native_profile_reviews.contains_key(&task_id),
            "Finish or cancel a pending native capture review before starting another"
        );
        let current = Arc::new(AtomicBool::new(true));
        let (retirement_sender, retirement_receiver) = futures::channel::oneshot::channel();
        let predecessor = predecessor_retirement.clone();
        let retained_permit = retirement_permit.clone();
        let retirement = async move {
            if let Some(predecessor) = predecessor {
                predecessor.await;
            }
            if retirement_receiver.await.is_err() {
                log::trace!("Native Task retirement owner was dropped");
            }
            drop(retained_permit);
        }
        .boxed()
        .shared();
        // Receiver-only drivers free finished chain allocations even without another Rerun.
        cx.foreground_executor().spawn(retirement.clone()).detach();
        let (revocation_sender, revocation_receiver) = futures::channel::oneshot::channel();
        let revocation = Arc::new(parking_lot::Mutex::new(Some(revocation_sender)));
        let revoked = async move {
            if revocation_receiver.await.is_err() {
                log::trace!("Native review revocation owner was dropped");
            }
        }
        .boxed()
        .shared();
        self.pending_native_profile_reviews.insert(
            task_id.clone(),
            PendingNativeProfileReview {
                supersession_handle: structured_handle.clone(),
                retirement,
                revocation: revocation.clone(),
                run_id,
                current: current.clone(),
                artifact_current: artifact_current.clone(),
                project_id,
                settings_revision,
            },
        );
        if let Some(handle) = structured_handle.as_ref() {
            let current = current.clone();
            let revocation = revocation.clone();
            handle.subscribe(cx, move |event, _| {
                if event.state.is_terminal() {
                    revoke_native_profile_review(&current, &revocation);
                }
            });
        }
        Ok(NativeProfileReviewLease {
            retirement_permit,
            predecessor_retirement,
            retirement: Some(retirement_sender),
            revoked,
            revocation,
            task_id,
            run_id,
            project_id: self.project.entity_id(),
            current,
            artifact_current,
            structured_handle,
            settings_revision,
        })
    }

    fn attach_native_build_supersession_handle(
        &mut self,
        review: &NativeProfileReviewLease,
        handle: &StructuredTaskHandle,
        cx: &mut gpui::App,
    ) {
        if let Some(pending) = self
            .pending_native_profile_reviews
            .get_mut(&review.task_id)
            .filter(|pending| {
                pending.run_id == review.run_id && Arc::ptr_eq(&pending.current, &review.current)
            })
        {
            pending.supersession_handle = Some(handle.clone());
        }
        let current = review.current.clone();
        let revocation = review.revocation.clone();
        handle.subscribe(cx, move |event, _| {
            if matches!(
                event.state,
                task::StructuredTaskState::Cancelled { .. }
                    | task::StructuredTaskState::SpawnError { .. }
                    | task::StructuredTaskState::Completed { success: false, .. }
            ) {
                revoke_native_profile_review(&current, &revocation);
            }
        });
    }

    fn native_profile_review_is_current(
        &self,
        review: &NativeProfileReviewLease,
        cx: &gpui::App,
    ) -> bool {
        cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
            && review.current.load(Ordering::Relaxed)
            && review.artifact_current.load(Ordering::Relaxed)
            && self.project.entity_id() == review.project_id
            && cx
                .try_global::<settings::SettingsStore>()
                .and_then(|settings| settings.observed_revision())
                == review.settings_revision
            && !review
                .structured_handle
                .as_ref()
                .is_some_and(|handle| handle.state().is_terminal())
            && self
                .pending_native_profile_reviews
                .get(&review.task_id)
                .is_some_and(|pending| {
                    pending.run_id == review.run_id
                        && Arc::ptr_eq(&pending.current, &review.current)
                })
    }

    fn retire_native_profile_review(&mut self, review: &NativeProfileReviewLease) {
        if self
            .pending_native_profile_reviews
            .get(&review.task_id)
            .is_some_and(|pending| {
                pending.run_id == review.run_id && Arc::ptr_eq(&pending.current, &review.current)
            })
        {
            self.pending_native_profile_reviews.remove(&review.task_id);
        }
        revoke_native_profile_review(&review.current, &review.revocation);
    }

    fn consume_native_profile_review(
        &mut self,
        review: &NativeProfileReviewLease,
        cx: &gpui::App,
    ) -> bool {
        let current = self.native_profile_review_is_current(review, cx);
        self.retire_native_profile_review(review);
        current
    }

    pub fn schedule_task_reference_with_completion(
        &mut self,
        label: String,
        worktree_id: WorktreeId,
        task_context: TaskContext,
        omit_history: bool,
        on_complete: impl FnOnce(ScheduledTaskResult, &mut AsyncWindowContext) + 'static,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let inventory = self
            .project
            .read(cx)
            .task_store()
            .read(cx)
            .task_inventory()
            .cloned();
        let task_list = inventory.map(|inventory| {
            inventory
                .read(cx)
                .list_tasks(None, None, Some(worktree_id), cx)
        });
        let workspace = cx.entity().downgrade();
        let mut on_complete = Some(Box::new(on_complete) as TaskCompletionHandler);
        let task = cx.spawn_in(window, async move |_workspace, cx| {
            let resolution = match task_list {
                Some(task_list) => resolve_task_reference(task_list.await, &label, &task_context),
                None => Err(anyhow::anyhow!("task inventory is unavailable")),
            };
            match resolution {
                Ok((source, resolved)) => {
                    let Some(completion) = on_complete.take() else {
                        log::error!("Pre-launch completion handler is unavailable");
                        return;
                    };
                    if let Err(error) = workspace.update_in(cx, |workspace, window, cx| {
                        workspace.schedule_resolved_task_with_completion(
                            source,
                            resolved,
                            omit_history,
                            completion,
                            window,
                            cx,
                        );
                    }) {
                        log::debug!(
                            "Workspace closed before pre-launch task scheduling: {error:#}"
                        );
                    }
                }
                Err(error) => {
                    log::error!("Cargo pre-launch task could not start: {error:#}");
                    if let Err(update_error) = workspace.update(cx, |workspace, cx| {
                        let id = NotificationId::unique::<ResolvedTask>();
                        workspace.show_toast(
                            Toast::new(id, format!("Pre-launch task could not start: {error}")),
                            cx,
                        );
                    }) {
                        log::debug!(
                            "Workspace closed before pre-launch error display: {update_error:#}"
                        );
                    }
                    if let Some(on_complete) = on_complete.take() {
                        on_complete(ScheduledTaskResult::SpawnFailed, cx);
                    }
                }
            }
        });
        self.scheduled_tasks.push(task);
    }

    pub fn schedule_task(
        self: &mut Workspace,
        task_source_kind: TaskSourceKind,
        task_to_resolve: &TaskTemplate,
        task_cx: &TaskContext,
        omit_history: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match self.project.read(cx).remote_connection_state(cx) {
            None | Some(ConnectionState::Connected) => {}
            Some(
                ConnectionState::Connecting
                | ConnectionState::Disconnected
                | ConnectionState::HeartbeatMissed
                | ConnectionState::Reconnecting,
            ) => {
                log::warn!("Cannot schedule tasks when disconnected from a remote host");
                return;
            }
        }

        if let Some(spawn_in_terminal) =
            task_to_resolve.resolve_task(&task_source_kind.to_id_base(), task_cx)
        {
            self.schedule_resolved_task(
                task_source_kind,
                spawn_in_terminal,
                omit_history,
                window,
                cx,
            );
        }
    }

    #[cfg(feature = "jvm-tools")]
    pub fn run_jvm_operation(
        &mut self,
        receipt: project::jvm_project::JvmBuildOperationReceipt,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.terminal_provider.is_some(),
            "The Task terminal is unavailable; reopen the workspace and check terminal setup"
        );
        anyhow::ensure!(
            groovy_jvm_worktree_trusted(&self.project, receipt.worktree_id, cx),
            "Trust the project before running a JVM Task"
        );
        let project = self.project.read(cx);
        anyhow::ensure!(
            !project.is_remote() && project.remote_connection_state(cx).is_none(),
            "Discovered build operations require a local host; remote execution is not qualified"
        );
        let lsp_store = project.lsp_store();
        let (snapshot, current) = lsp_store
            .read(cx)
            .jdtls_project_snapshot(receipt.worktree_id)
            .map(|(snapshot, current)| (snapshot.clone(), current))
            .context(
                "Build operation metadata is unavailable; refresh and select the operation again",
            )?;
        let task_store = project.task_store();
        let inventory = task_store
            .read(cx)
            .task_inventory()
            .cloned()
            .context("The build Task inventory is unavailable; reopen the project")?;
        let (source, resolved) = inventory.update(cx, |inventory, cx| {
            inventory.register_jvm_operation(receipt, &snapshot, current, cx)
        })?;
        self.schedule_resolved_task(source, resolved, false, window, cx);
        Ok(())
    }

    pub fn schedule_resolved_task(
        self: &mut Workspace,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            None,
            None,
            None,
            None,
            None,
            window,
            cx,
        );
    }

    pub fn schedule_resolved_task_with_structured_handle(
        &mut self,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> StructuredTaskHandle {
        let handle = StructuredTaskHandle::new(resolved_task.resolved.id.clone());
        if let Some(message) =
            structured_task_connection_error(self.project.read(cx).remote_connection_state(cx))
        {
            handle.mark_spawn_error(message, cx);
            return handle;
        }
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            Some(handle.clone()),
            None,
            None,
            None,
            None,
            window,
            cx,
        );
        handle
    }

    pub fn schedule_resolved_task_with_structured_handle_and_completion(
        &mut self,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        on_complete: impl FnOnce(ScheduledTaskResult, &mut AsyncWindowContext) + 'static,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> StructuredTaskHandle {
        let handle = StructuredTaskHandle::new(resolved_task.resolved.id.clone());
        if let Some(message) =
            structured_task_connection_error(self.project.read(cx).remote_connection_state(cx))
        {
            handle.mark_spawn_error(message, cx);
            cx.spawn_in(window, async move |_, cx| {
                on_complete(ScheduledTaskResult::SpawnFailed, cx);
            })
            .detach();
            return handle;
        }
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            Some(handle.clone()),
            Some(Box::new(on_complete)),
            None,
            None,
            None,
            window,
            cx,
        );
        handle
    }

    pub fn schedule_resolved_task_with_completion(
        self: &mut Workspace,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        on_complete: impl FnOnce(ScheduledTaskResult, &mut AsyncWindowContext) + 'static,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            None,
            Some(Box::new(on_complete)),
            None,
            None,
            None,
            window,
            cx,
        );
    }

    pub fn schedule_resolved_task_with_admission_and_completion(
        &mut self,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        admit: impl FnOnce(&mut Workspace, &mut Context<Workspace>) -> Result<()> + 'static,
        on_complete: impl FnOnce(ScheduledTaskResult, &mut AsyncWindowContext) + 'static,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            None,
            Some(Box::new(on_complete)),
            Some(Box::new(move |_, workspace, cx| admit(workspace, cx))),
            None,
            None,
            window,
            cx,
        );
    }

    pub fn schedule_resolved_task_with_admission_preflight_and_completion(
        &mut self,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        is_owned: impl FnOnce(&Workspace, &gpui::App) -> bool + 'static,
        admit: impl FnOnce(&mut Workspace, &mut Context<Workspace>) -> Result<()> + 'static,
        on_complete: impl FnOnce(ScheduledTaskResult, &mut AsyncWindowContext) + 'static,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            None,
            Some(Box::new(on_complete)),
            Some(Box::new(move |_, workspace, cx| admit(workspace, cx))),
            Some(Box::new(is_owned)),
            None,
            window,
            cx,
        );
    }

    pub fn schedule_resolved_task_with_invocation_admission_preflight_and_completion(
        &mut self,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        is_owned: impl FnOnce(&Workspace, &gpui::App) -> bool + 'static,
        admit: impl FnOnce(
            &TaskInvocationContext,
            &mut Workspace,
            &mut Context<Workspace>,
        ) -> Result<()>
        + 'static,
        on_complete: impl FnOnce(ScheduledTaskResult, &mut AsyncWindowContext) + 'static,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        self.schedule_resolved_task_internal(
            task_source_kind,
            resolved_task,
            omit_history,
            None,
            Some(Box::new(on_complete)),
            Some(Box::new(admit)),
            Some(Box::new(is_owned)),
            None,
            window,
            cx,
        );
    }

    fn schedule_selected_native_capture(
        &mut self,
        source: TaskSourceKind,
        plan: project::profile_results_store::PreparedNativeCapturePlan,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let resolved = plan.resolved_task().clone();
        let handle = StructuredTaskHandle::new(resolved.id.clone());
        let notification_id =
            NotificationId::named(format!("native-capture-{}", uuid::Uuid::new_v4()).into());
        let workspace = cx.entity().downgrade();
        self.show_toast(
            Toast::new(notification_id.clone(), "Native profile capture").on_click("Cancel", {
                let workspace = workspace.clone();
                let notification_id = notification_id.clone();
                let handle = handle.clone();
                move |_, cx| {
                    handle.cancel(cx);
                    if let Err(error) = workspace.update(cx, |workspace, cx| {
                        workspace.dismiss_toast(&notification_id, cx)
                    }) {
                        log::debug!("Native capture cancellation notice owner expired: {error:#}");
                    }
                }
            }),
            cx,
        );
        self.schedule_resolved_task_internal(
            source,
            resolved,
            true,
            Some(handle),
            Some(Box::new(move |_, cx| {
                if let Err(error) = workspace.update(cx, |workspace, cx| {
                    workspace.dismiss_toast(&notification_id, cx)
                }) {
                    log::debug!("Native capture completion notice owner expired: {error:#}");
                }
            })),
            None,
            None,
            Some(plan),
            window,
            cx,
        );
    }

    fn schedule_resolved_task_internal(
        self: &mut Workspace,
        task_source_kind: TaskSourceKind,
        resolved_task: ResolvedTask,
        omit_history: bool,
        structured_handle: Option<StructuredTaskHandle>,
        on_complete: Option<TaskCompletionHandler>,
        admit: Option<TaskAdmissionHandler>,
        admission_preflight: Option<TaskAdmissionPreflight>,
        selected_native_capture: Option<project::profile_results_store::PreparedNativeCapturePlan>,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) {
        let task_run_id = uuid::Uuid::new_v4();
        let resolved_task = match resolved_task.bind_artifact_invocation(&task_run_id.to_string()) {
            Ok(resolved_task) => resolved_task,
            Err(error) => {
                log::warn!("Task artifact output binding rejected: {error:#}");
                let message = "Task artifact output binding is invalid; review the artifact declaration and output argument";
                self.show_toast(
                    Toast::new(NotificationId::unique::<TaskInvocationContext>(), message),
                    cx,
                );
                if let Some(handle) = structured_handle {
                    handle.mark_spawn_error(message, cx);
                }
                if let Some(on_complete) = on_complete {
                    cx.spawn_in(window, async move |_, cx| {
                        on_complete(ScheduledTaskResult::SpawnFailed, cx);
                    })
                    .detach();
                }
                return;
            }
        };
        #[cfg(feature = "jvm-tools")]
        let groovy_jvm_build_worktree = task_source_kind.groovy_jvm_build_worktree();
        #[cfg(feature = "jvm-tools")]
        let groovy_jvm_task_worktree = task_source_kind.groovy_jvm_task_worktree();
        #[cfg(feature = "jvm-tools")]
        let groovy_jvm_build_task = groovy_jvm_build_worktree.map(|_| resolved_task.clone());
        #[cfg(feature = "jvm-tools")]
        let groovy_jvm_operation_worktree = task_source_kind.groovy_jvm_operation_worktree();
        #[cfg(feature = "jvm-tools")]
        let groovy_jvm_operation_task =
            groovy_jvm_operation_worktree.map(|_| resolved_task.clone());
        #[cfg(feature = "jvm-tools")]
        if let Some(message) = groovy_jvm_task_worktree.and_then(|worktree_id| {
            if let Some(operation) = groovy_jvm_operation_task.as_ref() {
                if task_source_kind != TaskSourceKind::groovy_jvm_operation(worktree_id) {
                    return Some(
                        "The build operation source changed; select the current operation again",
                    );
                }
                return groovy_jvm_operation_block_reason(
                    &self.project,
                    worktree_id,
                    operation,
                    cx,
                );
            }
            groovy_jvm_task_block_reason(
                &self.project,
                worktree_id,
                groovy_jvm_build_task.as_ref(),
                cx,
            )
        }) {
            self.show_toast(
                Toast::new(NotificationId::unique::<TaskSourceKind>(), message),
                cx,
            );
            if let Some(handle) = structured_handle {
                handle.mark_spawn_error(message, cx);
            }
            if let Some(on_complete) = on_complete {
                cx.spawn_in(window, async move |_, cx| {
                    on_complete(ScheduledTaskResult::SpawnFailed, cx);
                })
                .detach();
            }
            return;
        }
        let native_build = resolved_task.stdout_report().cloned();
        let mut experimental_admission =
            || -> Result<Option<project::task_store::PendingExperimentalTaskLaunch>> {
                ensure_generated_task_policy(&task_source_kind, resolved_task.original_task())?;
                project::task_store::validate_experimental_task_requirements(
                    &resolved_task.resolved,
                    cx,
                )?;
                let native = native_build
                    .as_ref()
                    .is_some_and(|report| report.native_profile.is_some())
                    || resolved_task
                        .resolved_artifact_group()
                        .is_some_and(native_profile_group);
                anyhow::ensure!(
                    !native || cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
                    "Native Rust profiling requires local experimental opt-in"
                );
                let coverage = resolved_task
                    .resolved_artifact_group()
                    .is_some_and(|group| group.format == "llvm-cov-json-v1");
                anyhow::ensure!(
                    !coverage || cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                    "Rust coverage requires local experimental opt-in"
                );
                project::task_store::PendingExperimentalTaskLaunch::new(&resolved_task.resolved, cx)
            };
        let pending_launch = match experimental_admission() {
            Ok(pending_launch) => pending_launch,
            Err(error) => {
                self.show_toast(
                    Toast::new(
                        NotificationId::unique::<TaskInvocationContext>(),
                        error.to_string(),
                    ),
                    cx,
                );
                if let Some(handle) = structured_handle.as_ref() {
                    handle.mark_spawn_error(error.to_string(), cx);
                }
                if let Some(on_complete) = on_complete {
                    cx.spawn_in(window, async move |_, cx| {
                        on_complete(ScheduledTaskResult::SpawnFailed, cx);
                    })
                    .detach();
                }
                return;
            }
        };
        #[cfg(any(test, feature = "test-support"))]
        let native_capture_diagnostic_scope = selected_native_capture.is_some();
        #[cfg(any(test, feature = "test-support"))]
        let native_diagnostic_scope = native_capture_diagnostic_scope
            || native_build
                .as_ref()
                .is_some_and(|report| report.native_profile.is_some());
        if native_build
            .as_ref()
            .is_some_and(|report| report.native_profile.is_none())
        {
            let message =
                "Observed Cargo Build Tasks without a native continuation are not supported yet";
            self.show_toast(
                Toast::new(NotificationId::unique::<TaskInvocationContext>(), message),
                cx,
            );
            if let Some(handle) = structured_handle.as_ref() {
                handle.mark_spawn_error(message, cx);
            }
            if let Some(on_complete) = on_complete {
                cx.spawn_in(window, async move |_, cx| {
                    on_complete(ScheduledTaskResult::SpawnFailed, cx);
                })
                .detach();
            }
            return;
        }
        let structured_handle = native_build_structured_handle(
            structured_handle,
            &resolved_task.id,
            native_build.as_ref(),
        );
        let native_build_source = native_build.as_ref().map(|_| task_source_kind.clone());
        let declared_artifact_directory = resolved_task.artifact_directory().map(Path::to_path_buf);
        let artifact_directory_is_explicit = declared_artifact_directory.is_some();
        let spawn_in_terminal = resolved_task.resolved.clone();
        let task_artifact = resolved_task.resolved_artifact().cloned();
        let task_artifact_group = resolved_task.resolved_artifact_group().cloned();
        let task_artifact_directory = Rc::new(std::cell::RefCell::new(
            declared_artifact_directory
                .clone()
                .or_else(|| spawn_in_terminal.cwd.clone()),
        ));
        if !omit_history {
            if let Some(debugger_provider) = self.debugger_provider.as_ref() {
                debugger_provider.task_scheduled(cx);
            }

            self.project().update(cx, |project, cx| {
                if let Some(task_inventory) =
                    project.task_store().read(cx).task_inventory().cloned()
                {
                    task_inventory.update(cx, |inventory, _| {
                        inventory.task_scheduled(task_source_kind, resolved_task);
                    })
                }
            });
        }

        if self.terminal_provider.is_some() {
            let mut artifact_capture = match task_artifact_group.as_ref() {
                Some(group) => match self.project.update(cx, |project, cx| {
                    project.begin_task_artifact_capture(
                        task_run_id.to_string(),
                        &spawn_in_terminal,
                        group,
                        cx,
                    )
                }) {
                    Ok(capture) => capture,
                    Err(error) => {
                        log::warn!("Task artifact capture was rejected: {error:#}");
                        self.show_toast(
                            Toast::new(
                                NotificationId::unique::<TaskInvocationContext>(),
                                format!("Task artifact capture unavailable: {error}"),
                            ),
                            cx,
                        );
                        if let Some(handle) = structured_handle.as_ref() {
                            handle.mark_spawn_error("Task artifact capture unavailable", cx);
                        }
                        if let Some(on_complete) = on_complete {
                            cx.spawn_in(window, async move |_, cx| {
                                on_complete(ScheduledTaskResult::SpawnFailed, cx);
                            })
                            .detach();
                        }
                        return;
                    }
                },
                None => None,
            };
            let structured_handle = if artifact_capture.is_some() {
                structured_handle
                    .or_else(|| Some(StructuredTaskHandle::new(spawn_in_terminal.id.clone())))
            } else {
                structured_handle
            };
            let artifact_current = Arc::new(AtomicBool::new(true));
            let mut artifact_subscriptions = Vec::new();
            if let Some(feature) = experimental_capture_feature(
                native_build.as_ref(),
                task_artifact_group.as_ref(),
                &spawn_in_terminal.required_experimental_features,
            ) {
                let native = feature == "rust-native-profiling-experimental";
                let handle = structured_handle.clone();
                let current = artifact_current.clone();
                artifact_subscriptions.push(cx.observe_global::<feature_flags::FeatureFlagStore>(
                    move |workspace, cx| {
                        let enabled = cx.has_flag_by_name(feature);
                        if !enabled {
                            current.store(false, Ordering::Relaxed);
                            if let Some(handle) = handle.as_ref() {
                                handle.cancel(cx);
                            }
                            if native {
                                for review in workspace.pending_native_profile_reviews.values() {
                                    revoke_native_profile_review(
                                        &review.current,
                                        &review.revocation,
                                    );
                                    if let Some(handle) = review.supersession_handle.as_ref() {
                                        handle.cancel(cx);
                                    }
                                }
                            }
                        }
                    },
                ));
            }
            if task_artifact.is_some() || task_artifact_group.is_some() || native_build.is_some() {
                if let Some(remote) = self.project.read(cx).remote_client() {
                    artifact_subscriptions.push(cx.subscribe(&remote, {
                        let artifact_current = artifact_current.clone();
                        move |_, _, event, _| {
                            if matches!(
                                event,
                                remote::RemoteClientEvent::ConnectionStateChanged(
                                    ConnectionState::Reconnecting | ConnectionState::Disconnected
                                ) | remote::RemoteClientEvent::Disconnected { .. }
                            ) {
                                artifact_current.store(false, Ordering::Relaxed);
                            }
                        }
                    }));
                }
                artifact_subscriptions.push(cx.subscribe(&self.project, {
                    let artifact_current = artifact_current.clone();
                    move |_, _, event, _| {
                        if matches!(
                            event,
                            project::Event::DisconnectedFromHost
                                | project::Event::DisconnectedFromRemote { .. }
                                | project::Event::RemoteIdChanged(_)
                                | project::Event::WorktreeRemoved(_)
                                | project::Event::WorktreePathsChanged { .. }
                                | project::Event::Closed
                        ) {
                            artifact_current.store(false, Ordering::Relaxed);
                        }
                    }
                }));
                if let Some(trust) =
                    project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                {
                    artifact_subscriptions.push(cx.subscribe(&trust, {
                        let artifact_current = artifact_current.clone();
                        move |workspace, _, _, cx| {
                            if project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                                &workspace.project.read(cx).worktree_store(), cx,
                            ) {
                                artifact_current.store(false, Ordering::Relaxed);
                            }
                        }
                    }));
                }
            }
            let mut native_review = match task_artifact_group
                .as_ref()
                .filter(|group| native_profile_group(group))
                .map(|_| {
                    self.begin_native_profile_review(
                        spawn_in_terminal.id.clone(),
                        task_run_id,
                        artifact_current.clone(),
                        structured_handle.clone(),
                        cx.try_global::<settings::SettingsStore>()
                            .and_then(|settings| settings.observed_revision()),
                        cx,
                    )
                })
                .transpose()
            {
                Ok(review) => review,
                Err(error) => {
                    self.show_toast(
                        Toast::new(
                            NotificationId::unique::<TaskInvocationContext>(),
                            error.to_string(),
                        ),
                        cx,
                    );
                    if let Some(handle) = structured_handle.as_ref() {
                        handle.mark_spawn_error(error.to_string(), cx);
                    }
                    if let Some(on_complete) = on_complete {
                        cx.spawn_in(window, async move |_, cx| {
                            on_complete(ScheduledTaskResult::SpawnFailed, cx)
                        })
                        .detach();
                    }
                    return;
                }
            };
            let mut build_review = match native_build
                .as_ref()
                .map(|_| {
                    self.begin_native_profile_review(
                        native_capture_phase_task_id(&spawn_in_terminal.id),
                        task_run_id,
                        artifact_current.clone(),
                        None,
                        cx.try_global::<settings::SettingsStore>()
                            .and_then(|settings| settings.observed_revision()),
                        cx,
                    )
                })
                .transpose()
            {
                Ok(review) => review,
                Err(error) => {
                    self.show_toast(
                        Toast::new(
                            NotificationId::unique::<TaskInvocationContext>(),
                            error.to_string(),
                        ),
                        cx,
                    );
                    if let Some(handle) = structured_handle.as_ref() {
                        handle.mark_spawn_error(error.to_string(), cx);
                    }
                    if let Some(on_complete) = on_complete {
                        cx.spawn_in(window, async move |_, cx| {
                            on_complete(ScheduledTaskResult::SpawnFailed, cx)
                        })
                        .detach();
                    }
                    return;
                }
            };
            if let Some(review) = build_review.as_ref()
                && let Some(handle) = structured_handle.as_ref()
            {
                observe_native_build_review_lifecycle(handle, review.current.clone(), cx);
                self.attach_native_build_supersession_handle(review, handle, cx);
            }
            let retirement = build_review
                .as_mut()
                .or(native_review.as_mut())
                .and_then(|review| {
                    review.retirement.take().map(|sender| {
                        Arc::new(NativeProfileRetirementSignal {
                            sender: Some(sender),
                            permit: Some(review.retirement_permit.clone()),
                        })
                    })
                });
            let admission_lock = (admit.is_some()
                || artifact_capture.is_some()
                || native_review.is_some()
                || native_build.is_some())
            .then(|| self.task_admission_prerequisite_lock.clone());
            let task = cx.spawn_in(window, async move |workspace, cx| {
                // An older prerequisite save must finish before a newer capture seals its source.
                let prerequisite_guard = match admission_lock.as_ref() {
                    Some(lock) => Some(lock.lock().await),
                    None => None,
                };
                if let Some(is_owned) = admission_preflight {
                    let owned = match workspace.update(cx, |workspace, cx| is_owned(workspace, cx)) {
                        Ok(owned) => owned,
                        Err(error) => {
                            log::debug!("Task prerequisite owner expired: {error:#}");
                            false
                        }
                    };
                    if !owned {
                        drop(prerequisite_guard);
                        if let Some(on_complete) = on_complete {
                            on_complete(ScheduledTaskResult::Cancelled, cx);
                        }
                        return;
                    }
                }
                if artifact_capture.as_ref().is_some_and(|capture| {
                    !artifact_current.load(Ordering::Relaxed)
                        || !workspace.read_with(cx, |_, cx| capture.is_current(cx)).unwrap_or_else(|error| {
                            log::debug!("Task artifact capture owner expired: {error:#}");
                            false
                        })
                }) {
                    drop(prerequisite_guard);
                    if let Some(handle) = structured_handle.as_ref()
                        && let Err(error) = cx.update(|_, cx| {
                            handle.mark_cancelled(handle.state().terminal_id(), false, cx);
                        })
                    {
                        log::debug!("Cancelled artifact capture could not update its Task: {error:#}");
                    }
                    if let Some(on_complete) = on_complete {
                        on_complete(ScheduledTaskResult::Cancelled, cx);
                    }
                    return;
                }
                if native_review.as_ref().is_some_and(|review| {
                    !workspace.read_with(cx, |workspace, cx| workspace.native_profile_review_is_current(review, cx))
                        .unwrap_or_else(|error| {
                            log::debug!("Native Task prerequisite owner expired: {error:#}");
                            false
                        })
                }) {
                    drop(prerequisite_guard);
                    if let Some(handle) = structured_handle.as_ref()
                        && let Err(error) = cx.update(|_, cx| handle.mark_cancelled(handle.state().terminal_id(), false, cx))
                    {
                        log::debug!("Native Task cancellation could not update its state: {error:#}");
                    }
                    if let Some(on_complete) = on_complete {
                        on_complete(ScheduledTaskResult::Cancelled, cx);
                    }
                    return;
                }
                if artifact_capture.is_some() || native_review.is_some() || native_build.is_some() {
                    if let Err(error) = Self::save_for_task_checked(
                        &workspace, spawn_in_terminal.save, cx,
                    ).await {
                        log::warn!("Task artifact prerequisite save failed: {error:#}");
                        if let Some(handle) = structured_handle.as_ref()
                            && let Err(error) = cx.update(|_, cx| {
                                handle.mark_spawn_error("Task prerequisite save did not complete", cx);
                            })
                        {
                            log::debug!("Artifact save failure could not update its Task: {error:#}");
                        }
                        if let Err(error) = workspace.update(cx, |workspace, cx| {
                            workspace.show_toast(Toast::new(
                                NotificationId::unique::<TaskInvocationContext>(),
                                "Task prerequisite save did not complete; artifact capture was not started",
                            ), cx);
                        }) {
                            log::debug!("Artifact save failure notice could not be shown: {error:#}");
                        }
                        if let Some(on_complete) = on_complete {
                            on_complete(ScheduledTaskResult::SpawnFailed, cx);
                        }
                        return;
                    }
                } else {
                    Self::save_for_task(&workspace, spawn_in_terminal.save, cx).await;
                }

                #[cfg(feature = "jvm-tools")]
                if let Some(message) = groovy_jvm_task_worktree.and_then(|worktree_id| {
                    workspace
                        .update(cx, |workspace, cx| {
                            if let Some(operation)=groovy_jvm_operation_task.as_ref() {
                                return groovy_jvm_operation_block_reason(&workspace.project,worktree_id,operation,cx);
                            }
                            groovy_jvm_task_block_reason(
                                &workspace.project,
                                worktree_id,
                                groovy_jvm_build_task.as_ref(),
                                cx,
                            )
                        })
                        .unwrap_or(Some("The JVM build project is no longer available"))
                }) {
                    if let Some(handle) = structured_handle.as_ref() {
                        if let Err(error) = cx.update(|_, cx| handle.mark_spawn_error(message, cx)) {
                            log::debug!("JVM build could not report blocked trust: {error:#}");
                        }
                    }
                    if let Err(error) = workspace.update(cx, |workspace, cx| {
                        workspace.show_toast(
                            Toast::new(NotificationId::unique::<TaskSourceKind>(), message),
                            cx,
                        );
                    }) {
                        log::debug!("JVM build trust notice could not be shown: {error:#}");
                    }
                    if let Some(on_complete) = on_complete {
                        on_complete(ScheduledTaskResult::SpawnFailed, cx);
                    }
                    return;
                }

                if structured_handle
                    .as_ref()
                    .is_some_and(|handle| handle.state().is_terminal())
                {
                    if let Some(on_complete) = on_complete {
                        on_complete(ScheduledTaskResult::Cancelled,cx);
                    }
                    return;
                }

                let remote_profile = task_artifact_group.as_ref().is_some_and(native_profile_group)
                    && workspace.read_with(cx, |workspace, cx| {
                        workspace.project.read(cx).remote_client().is_some()
                    }).unwrap_or(false);
                let allow_unsupported_remote_profile =
                    selected_native_capture.is_none() && native_build.is_none();
                let local_native_verified = Arc::new(AtomicBool::new(false));
                let host_prepared = Arc::new(AtomicBool::new(false));
                let mut launch = spawn_in_terminal;
                let spawn_task = workspace.update_in(cx, |workspace, window, cx| {
                    project::task_store::validate_experimental_task_requirements(&launch, cx)?;
                    if let Some(pending_launch) = pending_launch.as_ref() {
                        pending_launch.check(&launch, cx)?;
                    }
                    if let Some(capture) = artifact_capture.as_mut() {
                        anyhow::ensure!(
                            artifact_current.load(Ordering::Relaxed) && capture.is_current(cx),
                            "Task artifact capture was superseded or its project changed"
                        );
                        let working_directory = workspace.project.read(cx)
                            .terminal_task_working_directory(launch.cwd.as_deref(), cx)
                            .ok_or_else(|| anyhow!("Task artifact capture needs a project working directory"))?;
                        anyhow::ensure!(working_directory.is_absolute(),
                            "Task artifact capture needs an absolute working directory");
                        launch.cwd = Some(working_directory.to_path_buf());
                        workspace.project.read(cx).admit_task_artifact_capture(capture, &launch, cx)?;
                        if !artifact_directory_is_explicit {
                            *task_artifact_directory.borrow_mut() = launch.cwd.clone();
                        }
                    }
                    if let Some(admit) = admit {
                        let mut invocation = TaskInvocationContext::from_launch(
                            task_run_id,
                            &launch,
                            task_artifact.clone(),
                            task_artifact_group.clone(),
                        );
                        invocation.artifact_directory = task_artifact_directory.borrow().clone();
                        admit(&invocation, workspace, cx)?;
                    }
                    let build_preparation = native_build.map(|report| {
                        let project = workspace.project.downgrade();
                        let workspace = workspace.weak_handle();
                        let source = native_build_source;
                        let review = build_review;
                        let retirement = retirement.clone();
                        Box::new(move|mut launch:SpawnInTerminal,cx:&mut AsyncWindowContext| {
                            let owner_retirement = retirement.clone();
                            spawn_native_profile_retiring(cx, retirement, async move|cx| {
                                let review = review.context("Selected Build review is unavailable")?;
                                if let Some(predecessor) = review.predecessor_retirement.as_ref() {
                                    match futures::future::select(predecessor.clone(), review.revoked.clone()).await {
                                        futures::future::Either::Left(_) => {},
                                        futures::future::Either::Right(_) => return Err(anyhow::Error::new(crate::TaskLaunchCancelled)),
                                    }
                                }
                                if !workspace.read_with(cx,|workspace,cx|workspace.native_profile_review_is_current(&review,cx))? {
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                let directory = project.read_with(cx,|project,cx|project.terminal_task_working_directory(launch.cwd.as_deref(),cx))?
                                    .context("Selected Build requires a project working directory")?;
                                anyhow::ensure!(directory.is_absolute(),"Selected Build requires an absolute working directory");
                                launch.cwd = Some(directory.to_path_buf());
                                let continuation = report.native_profile.context("Selected native preparation continuation is unavailable")?;
                                let prepared = project.update(cx,|project,cx|match report.format {
                                    task::TaskStdoutReportFormat::CargoBuildJson => project.prepare_selected_native_cargo_build_task(task_run_id,launch,report.max_bytes,cx),
                                    task::TaskStdoutReportFormat::CargoRuntimePreparationJson => project.prepare_selected_native_cargo_runtime_task(task_run_id,launch,report.max_bytes,continuation.clone(),cx),
                                })??;
                                let mut prepared = prepared.await?;
                                if !workspace.read_with(cx,|workspace,cx|workspace.native_profile_review_is_current(&review,cx))? {
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                let native_terminal = prepared.take_native_terminal_launch()?;
                                let launch = prepared.launch().clone();
                                let source = source.context("Selected Build source is unavailable")?;
                                let completion = Box::new(move|spawned:Task<Option<Result<ExitStatus>>>,cx:&mut AsyncWindowContext| {
                                    spawn_native_profile_retiring(cx, owner_retirement, async move|cx| {
                                        let result = spawned.await;
                                        if result.as_ref().is_some_and(|result|result.as_ref().is_ok_and(|status|status.success())) {
                                            let transition = async {
                                                anyhow::ensure!(workspace.read_with(cx,|workspace,cx|workspace.native_profile_review_is_current(&review,cx))?,
                                                    "Selected Build was superseded before capture");
                                                let finished = project.update(cx,|project,cx|project.finish_selected_native_cargo_build_task(prepared,cx))??;
                                                let workload = finished.await?;
                                                let planned = project.update(cx,|project,cx|project.prepare_native_capture_plan(workload,continuation,cx))??;
                                                let plan = planned.await?;
                                                workspace.update_in(cx,|workspace,window,cx| {
                                                    anyhow::ensure!(workspace.consume_native_profile_review(&review,cx),
                                                        "Selected Build was superseded before capture");
                                                    workspace.schedule_selected_native_capture(source,plan,window,cx);
                                                    Ok::<_,anyhow::Error>(())
                                                })??;
                                                Ok::<_,anyhow::Error>(())
                                            }.await;
                                            if let Err(error) = transition {
                                                log::warn!("Native capture did not start after Build: {error:#}");
                                                #[cfg(feature = "test-support")]
                                                eprintln!("native selected capture did not start after Build: {error:#}");
                                                if let Err(window_error) = workspace.update(cx,|workspace,cx|workspace.show_toast(Toast::new(
                                                    NotificationId::unique::<TaskInvocationContext>(),format!("Build completed; native capture was not started: {error}")),cx))
                                                { log::debug!("Native capture transition notice could not be shown: {window_error:#}"); }
                                            }
                                        }
                                        result
                                    })
                                }) as crate::TaskLaunchCompletion;
                                Ok(crate::PreparedTaskLaunch{launch,completion:Some(completion),native_terminal,pending_launch:None})
                            })
                        }) as crate::TaskLaunchPreparation
                    });
                    let native_preparation = native_review.map(|review| {
                        let workspace = workspace.weak_handle();
                        let group = task_artifact_group.clone();
                        let current = artifact_current.clone();
                        let host_prepared = host_prepared.clone();
                        let local_native_verified = local_native_verified.clone();
                        let working_directory = task_artifact_directory.clone();
                        let retirement = retirement.clone();
                        Box::new(move |mut launch: SpawnInTerminal, cx: &mut AsyncWindowContext| {
                            let owner_retirement = retirement.clone();
                            spawn_native_profile_retiring(cx, retirement, async move |cx| {
                                let current_review = workspace.read_with(cx, |workspace, cx| {
                                    workspace.native_profile_review_is_current(&review, cx)
                                })?;
                                if !current_review {
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                let directory = workspace.read_with(cx, |workspace, cx| {
                                    workspace.project.read(cx).terminal_task_working_directory(launch.cwd.as_deref(), cx)
                                })?.ok_or_else(|| anyhow!("Native capture needs a project working directory"))?;
                                anyhow::ensure!(directory.is_absolute(), "Native capture needs an absolute working directory");
                                launch.cwd = Some(directory.to_path_buf());
                                let group = group.as_ref().ok_or_else(|| anyhow!("Missing native capture group"))?;
                                let native_artifact_directory = if artifact_directory_is_explicit {
                                    working_directory.borrow().clone()
                                } else {
                                    launch.cwd.clone()
                                };
                                anyhow::ensure!(!remote_profile || !artifact_directory_is_explicit,
                                    "An explicit native artifact directory requires host-issued remote authority");
                                confirm_native_profile_capture(&workspace, group, native_artifact_directory.as_deref(), &review.current, &review.revoked, cx).await?;
                                if !workspace.read_with(cx, |workspace, cx| workspace.native_profile_review_is_current(&review,cx))? {
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                let mut selected_completion = None;
                                let mut native_terminal = None;
                                if let Some(plan) = selected_native_capture {
                                    #[cfg(any(test, feature = "test-support"))]
                                    eprintln!("native selected capture stage=post-consent-admission-start");
                                    let prepared = workspace.update(cx,|workspace,cx|workspace.project.update(cx,|project,cx| {
                                        project.admit_native_capture_task(plan,task_run_id,launch.clone(),group.clone(),declared_artifact_directory.clone(),cx)
                                    }))??;
                                    let mut prepared = prepared.await?;
                                    if !workspace.read_with(cx,|workspace,cx|workspace.native_profile_review_is_current(&review,cx))? {
                                        return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                    }
                                    native_terminal = prepared.take_native_terminal_launch()?;
                                    #[cfg(any(test, feature = "test-support"))]
                                    eprintln!("native selected capture stage=terminal-handoff-ready");
                                    launch = prepared.launch().clone();
                                    let project = workspace.read_with(cx,|workspace,_|workspace.project.downgrade())?;
                                    let current = current.clone();
                                    let workspace = workspace.clone();
                                    selected_completion = Some(Box::new(move|spawned:Task<Option<Result<ExitStatus>>>,cx:&mut AsyncWindowContext| {
                                        spawn_native_profile_retiring(cx, owner_retirement, async move|cx| {
                                            let result = spawned.await;
                                            #[cfg(any(test, feature = "test-support"))]
                                            eprintln!("native selected capture stage=terminal-completed success={}", result.as_ref().is_some_and(|result|result.as_ref().is_ok_and(|status|status.success())));
                                            if result.as_ref().is_some_and(|result|result.as_ref().is_ok_and(|status|status.success())) {
                                                let finished = match project.update(cx,|project,cx|project.finish_native_capture_task(prepared,cx)) {
                                                    Ok(Ok(finished)) => finished.await,
                                                    Ok(Err(error)) | Err(error) => Err(error),
                                                };
                                                if finished.is_ok() && current.load(Ordering::Relaxed) {
                                                    local_native_verified.store(true, Ordering::Relaxed);
                                                }
                                                if let Err(error) = finished {
                                                    #[cfg(any(test, feature = "test-support"))]
                                                    eprintln!("native selected capture stage=pair-verification-refused category={}", native_capture_diagnostic_category(&error));
                                                    current.store(false,Ordering::Relaxed);
                                                    log::warn!("Native capture pair was not verified: {error:#}");
                                                    if let Err(window_error) = workspace.update(cx,|workspace,cx|workspace.show_toast(Toast::new(
                                                        NotificationId::unique::<TaskInvocationContext>(),format!("Capture completed; profile files were not verified: {error}")),cx))
                                                    { log::debug!("Native capture verification notice could not be shown: {window_error:#}"); }
                                                }
                                            }
                                            result
                                        })
                                    }) as crate::TaskLaunchCompletion);
                                }
                                if remote_profile {
                                    anyhow::ensure!(launch.command.as_deref().is_some_and(profile_task_command_supported),
                                        "Native capture Tasks require a simple executable command");
                                    let preparation = workspace.update(cx, |workspace,cx| {
                                        let project = workspace.project.read(cx);
                                        let paths = pending_remote_profile_paths(project,group,native_artifact_directory.as_deref(),cx)?;
                                        project.prepare_remote_profile_task(task_run_id.to_string(),paths,cx)
                                    })?;
                                    let binding = match preparation {
                                        Ok(preparation) => preparation.await,
                                        Err(error) => Err(error),
                                    };
                                    let binding = match binding {
                                        Ok(binding) => Some(binding),
                                        Err(error) if allow_unsupported_remote_profile
                                            && matches!(error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                                                Some(project::profile_results_store::RemoteProfileError::Unsupported)) => None,
                                        Err(error) => return Err(error),
                                    };
                                    if !current.load(Ordering::Relaxed)
                                        || !workspace.read_with(cx,|workspace,cx| workspace.native_profile_review_is_current(&review,cx))?
                                    {
                                        if binding.is_some() {
                                            let cancel = workspace.update(cx,|workspace,cx| workspace.project.read(cx)
                                                .cancel_remote_profile_task(task_run_id.to_string(),cx))??;
                                            cancel.await?;
                                        }
                                        return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                    }
                                    if let Some(binding) = binding {
                                        launch = wrap_remote_profile_task(launch,binding);
                                        host_prepared.store(true,Ordering::Relaxed);
                                    } else {
                                        workspace.update(cx, |workspace,cx| {
                                            workspace.show_toast(Toast::new(
                                                NotificationId::unique::<CompletedTaskArtifactGroup>(),
                                                "This host cannot verify a native profile. Your Task will run unchanged; no profile import will be offered.",
                                            ),cx);
                                        })?;
                                    }
                                }
                                if !workspace.update(cx,|workspace,cx| workspace.consume_native_profile_review(&review,cx))? {
                                    if host_prepared.load(Ordering::Relaxed) {
                                        let cancel = workspace.update(cx,|workspace,cx| workspace.project.read(cx)
                                            .cancel_remote_profile_task(task_run_id.to_string(),cx))??;
                                        cancel.await?;
                                    }
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                if !artifact_directory_is_explicit {
                                    *working_directory.borrow_mut() = launch.cwd.clone();
                                }
                                Ok(crate::PreparedTaskLaunch{launch,completion:selected_completion,native_terminal,pending_launch:None})
                            })
                        }) as crate::TaskLaunchPreparation
                    });
                    let preparation=build_preparation.or(native_preparation).or_else(|| artifact_capture.take().map(|capture| {
                        let project=workspace.project.downgrade();
                        let current=artifact_current.clone();
                        Box::new(move|launch:SpawnInTerminal,cx:&mut AsyncWindowContext| {
                            cx.spawn(async move|cx| {
                                if !current.load(Ordering::Relaxed)
                                    || !project.read_with(cx,|_,cx|capture.is_current(cx))?
                                {
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                let prepared = project.update(cx, |project, cx| project.prepare_task_artifact_capture_sources(capture, cx))??;
                                let capture = prepared.await?;
                                if !current.load(Ordering::Relaxed) {
                                    return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
                                }
                                project.read_with(cx,|project,cx|project.revalidate_task_artifact_capture(&capture,&launch,cx))??;
                                let completion=Box::new(move|spawned:Task<Option<Result<ExitStatus>>>,cx:&mut AsyncWindowContext| {
                                    match project.update(cx,|project,cx|project.observe_task_artifact_capture(capture,spawned,cx)) {
                                        Ok(observed)=>observed,
                                        Err(error)=>Task::ready(Some(Err(error))),
                                    }
                                }) as crate::TaskLaunchCompletion;
                                Ok(crate::PreparedTaskLaunch{launch,completion:Some(completion),native_terminal:None,pending_launch:None})
                            })
                        }) as crate::TaskLaunchPreparation
                    }));
                    let preparation = match (preparation, retirement) {
                        (Some(preparation), Some(retirement)) => Some(
                            native_profile_retiring_preparation(preparation, Some(retirement)),
                        ),
                        (preparation, _) => preparation,
                    };
                    let preparation = match (preparation, pending_launch) {
                        (preparation, Some(pending_launch)) => Some(Box::new(
                            move |launch: SpawnInTerminal, cx: &mut AsyncWindowContext| {
                                cx.spawn(async move |cx| {
                                    cx.update(|_, cx| pending_launch.check(&launch, cx))??;
                                    let mut prepared = match preparation {
                                        Some(prepare) => prepare(launch, cx).await?,
                                        None => crate::PreparedTaskLaunch {
                                            launch,
                                            completion: None,
                                            native_terminal: None,
                                            pending_launch: None,
                                        },
                                    };
                                    cx.update(|_, cx| pending_launch.check(&prepared.launch, cx))??;
                                    prepared.pending_launch = Some(pending_launch);
                                    Ok(prepared)
                                })
                            },
                        ) as crate::TaskLaunchPreparation),
                        (preparation, None) => preparation,
                    };
                    let spawned = workspace.terminal_provider.as_ref().map(|terminal_provider| {
                        match preparation {
                            Some(prepare) => terminal_provider.spawn_prepared(launch,structured_handle.clone(),prepare,window,cx),
                            None => if let Some(handle) = structured_handle.clone() {
                                terminal_provider.spawn_structured(launch,handle,window,cx)
                            } else {
                                terminal_provider.spawn(launch,window,cx)
                            },
                        }
                    });
                    Ok::<_, anyhow::Error>(spawned)
                });
                let spawn_task = match spawn_task {
                    Ok(Ok(task)) => task,
                    Ok(Err(error)) | Err(error) => {
                        #[cfg(any(test, feature = "test-support"))]
                        if native_diagnostic_scope {
                            if native_capture_diagnostic_scope {
                                eprintln!("native selected capture stage=prelaunch-admission-refused category={}", native_capture_diagnostic_category(&error));
                            } else {
                                let detail: String = error.to_string().chars().take(1024).collect();
                                eprintln!("native selected prelaunch admission rejected: {detail}");
                            }
                        }
                        log::warn!("Task admission rejected before terminal launch: {error:#}");
                        None
                    }
                };
                drop(prerequisite_guard);
                if let Some(spawn_task) = spawn_task {
                    let res = cx.background_spawn(spawn_task).await;
                    let mut host_verified = local_native_verified.load(Ordering::Relaxed);
                    let result = match res {
                        Some(Ok(status)) => {
                            if let Some(handle) = structured_handle.as_ref() {
                                if let Err(error) = cx.update(|_, cx| {
                                    handle.mark_completed(status, cx);
                                }) {
                                    log::debug!("Structured task window closed: {error:#}");
                                }
                            }
                            if status.success() {
                                log::debug!("Task spawn succeeded");
                                ScheduledTaskResult::Success
                            } else {
                                log::debug!("Task spawn failed, code: {:?}", status.code());
                                ScheduledTaskResult::Failure
                            }
                        }
                        Some(Err(e)) => {
                            #[cfg(any(test, feature = "test-support"))]
                            if native_diagnostic_scope {
                                if native_capture_diagnostic_scope {
                                    eprintln!("native selected capture stage=preparation-or-spawn-refused category={}", native_capture_diagnostic_category(&e));
                                } else {
                                    let detail: String = e.to_string().chars().take(1024).collect();
                                    eprintln!("native selected preparation or terminal spawn failed: {detail}");
                                }
                            }
                            let detail=e.to_string();
                            #[cfg(feature = "jvm-tools")]
                            let detail=if groovy_jvm_operation_worktree.is_some() {
                                "Build task failed to start; check the local wrapper, Java setup and terminal permissions, then retry".to_string()
                            } else {detail};
                            if let Some(handle) = structured_handle.as_ref() {
                                if let Err(error) = cx.update(|_, cx| {
                                    handle.mark_spawn_error(detail.clone(), cx);
                                }) {
                                    log::debug!("Structured task window closed: {error:#}");
                                }
                            }
                            let message=format!("Task spawn failed: {detail}");
                            #[cfg(feature = "jvm-tools")]
                            if groovy_jvm_operation_worktree.is_some() {
                                log::error!("{message}");
                            } else {
                                log::error!("Task spawn failed: {e:#}");
                            }
                            #[cfg(not(feature = "jvm-tools"))]
                            log::error!("Task spawn failed: {e:#}");
                            if let Err(error) = workspace.update(cx, |w, cx| {
                                let id = NotificationId::unique::<ResolvedTask>();
                                w.show_toast(Toast::new(id, message.clone()), cx);
                            }) {
                                log::debug!("Task error toast could not be shown: {error:#}");
                            }
                            ScheduledTaskResult::SpawnFailed
                        }
                        None => {
                            if let Some(handle) = structured_handle.as_ref()
                                && let Err(error) = cx.update(|_, cx| {
                                    handle.mark_cancelled(handle.state().terminal_id(), false, cx);
                                })
                            {
                                log::debug!("Structured task window closed: {error:#}");
                            }
                            log::debug!("Task spawn got cancelled");
                            ScheduledTaskResult::Cancelled
                        }
                    };
                    if host_prepared.load(Ordering::Relaxed) {
                        if result == ScheduledTaskResult::Success
                            && artifact_current.load(Ordering::Relaxed)
                            && !structured_handle.as_ref().is_some_and(|handle| {
                                matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
                            })
                        {
                            let proof = workspace.update(cx, |workspace, cx| {
                                workspace.project.read(cx).get_remote_profile_task(
                                    task_run_id.to_string(), cx,
                                )
                            });
                            let proof = match proof {
                                Ok(Ok(task)) => {
                                    let deadline = cx.background_executor().timer(
                                        std::time::Duration::from_secs(30),
                                    );
                                    match futures::future::select(Box::pin(task), Box::pin(deadline)).await {
                                        futures::future::Either::Left((result, _)) => result,
                                        futures::future::Either::Right(_) => {
                                            Err(anyhow!("Host profile completion timed out"))
                                        }
                                    }
                                }
                                Ok(Err(error)) | Err(error) => Err(error),
                            };
                            match proof {
                                Ok(()) => host_verified = true,
                                Err(error) => {
                                    log::warn!("Remote profile Task was not certified: {error:#}");
                                    if let Err(error) = workspace.update(cx, |workspace, cx| {
                                        workspace.show_toast(Toast::new(
                                            NotificationId::unique::<CompletedTaskArtifactGroup>(),
                                            format!("Native Task profile is unavailable: {error}"),
                                        ), cx);
                                    }) {
                                        log::debug!("Native Task profile notice could not be shown: {error:#}");
                                    }
                                }
                            }
                        }
                        if host_verified {
                            let mut indexed = false;
                            for _ in 0..100 {
                                if !artifact_current.load(Ordering::Relaxed)
                                    || structured_handle.as_ref().is_some_and(|handle| {
                                        matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
                                    })
                                {
                                    break;
                                }
                                indexed = workspace.update(cx, |workspace, cx| {
                                    let project = workspace.project.read(cx);
                                    task_artifact_group.as_ref().is_some_and(|group| {
                                        group.files.values().all(|artifact| {
                                            resolve_task_artifact(
                                                project,
                                                artifact,
                                                task_artifact_directory.borrow().as_deref(),
                                                cx,
                                            ).is_ok()
                                        })
                                    })
                                }).unwrap_or(false);
                                if indexed {
                                    break;
                                }
                                cx.background_executor()
                                    .timer(std::time::Duration::from_millis(100))
                                    .await;
                            }
                            if !indexed {
                                host_verified = false;
                                if artifact_current.load(Ordering::Relaxed)
                                    && !structured_handle.as_ref().is_some_and(|handle| {
                                        matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
                                    })
                                {
                                    log::warn!("Verified profile files did not reach the client worktree");
                                    if let Err(error) = workspace.update(cx, |workspace, cx| {
                                        workspace.show_toast(Toast::new(
                                            NotificationId::unique::<CompletedTaskArtifactGroup>(),
                                            "The profile completed on the host, but its files are not visible in this project window. Run the Task again after the worktree refreshes.",
                                        ), cx);
                                    }) {
                                        log::debug!("Native Task profile notice could not be shown: {error:#}");
                                    }
                                }
                            }
                        }
                        if !host_verified {
                            if let Ok(Ok(cancel)) = workspace.update(cx, |workspace, cx| {
                                workspace.project.read(cx).cancel_remote_profile_task(
                                    task_run_id.to_string(), cx,
                                )
                            }) {
                                if let Err(error) = cx.update(|_, cx| cancel.detach_and_log_err(cx)) {
                                    log::debug!("Profile Task cancellation could not be detached: {error:#}");
                                }
                            }
                        }
                    }
                    let local_data_artifact = task_artifact
                        .as_ref()
                        .is_some_and(|artifact| artifact.kind == task::TaskArtifactKind::Data)
                        && workspace
                            .read_with(cx, |workspace, cx| workspace.project.read(cx).is_local())
                            .unwrap_or(false);
                    if result == ScheduledTaskResult::Success
                        && artifact_current.load(Ordering::Relaxed)
                        && !structured_handle.as_ref().is_some_and(|handle| {
                            matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
                        })
                        && (task_artifact.as_ref().is_some_and(|artifact| artifact.kind != task::TaskArtifactKind::Data)
                            || local_data_artifact
                            || (task_artifact_group.is_some()
                                && (!remote_profile || host_verified)))
                    {
                        let prepared = match workspace.update(cx, |workspace, cx| {
                            let artifacts = task_artifact.iter().chain(task_artifact_group.iter().flat_map(|group| group.files.values()));
                            workspace.refresh_task_artifacts(artifacts, task_artifact_directory.borrow().as_deref(), cx)
                        }) {
                            Ok(Ok(preparation)) => preparation.await,
                            Ok(Err(error)) | Err(error) => Err(error),
                        };
                        if let Err(error) = workspace.update_in(cx, |workspace, window, cx| {
                            let result = prepared.and_then(|buffers| {
                                if !artifact_current.load(Ordering::Relaxed)
                                    || structured_handle.as_ref().is_some_and(|handle| matches!(handle.state(), task::StructuredTaskState::Cancelled { .. }))
                                {
                                    bail!("Task was cancelled or its project session changed during artifact refresh");
                                }
                                if let Some(group) = task_artifact_group {
                                    workspace.publish_task_artifact_group(
                                        task_run_id, group, task_artifact_directory.borrow().as_deref(),
                                        host_verified, artifact_current, artifact_subscriptions, cx,
                                    )?;
                                } else if let Some(artifact) = task_artifact.filter(|artifact| artifact.kind != task::TaskArtifactKind::Data) {
                                    workspace.open_task_artifact(artifact, task_artifact_directory.borrow().clone(), artifact_current, artifact_subscriptions, buffers, window, cx);
                                }
                                Ok(())
                            });
                            if let Err(error) = result {
                                workspace.show_toast(Toast::new(
                                    NotificationId::unique::<CompletedTaskArtifactGroup>(),
                                    format!("Task artifacts are not ready: {error}"),
                                ), cx);
                            }
                        }) {
                            log::debug!("Task artifacts could not be opened after workspace close: {error:#}");
                        }
                    }
                    if let Some(on_complete) = on_complete {
                        on_complete(result, cx);
                    }
                } else {
                    if host_prepared.load(Ordering::Relaxed) {
                        if let Ok(Ok(cancel)) = workspace.update(cx, |workspace, cx| {
                            workspace.project.read(cx).cancel_remote_profile_task(
                                task_run_id.to_string(), cx,
                            )
                        }) {
                            if let Err(error) = cx.update(|_, cx| cancel.detach_and_log_err(cx)) {
                                log::debug!("Profile Task cancellation could not be detached: {error:#}");
                            }
                        }
                    }
                    if let Some(handle) = structured_handle.as_ref()
                        && let Err(error) = cx.update(|_, cx| {
                            handle.mark_cancelled(handle.state().terminal_id(), false, cx);
                        })
                    {
                        log::debug!("Structured task window closed: {error:#}");
                    }
                    if let Some(on_complete) = on_complete {
                        on_complete(ScheduledTaskResult::Cancelled, cx);
                    }
                }
            });
            self.scheduled_tasks.push(task);
        } else if let Some(handle) = structured_handle {
            handle.mark_spawn_error("No terminal provider is available", cx);
        }
    }

    fn refresh_task_artifacts<'a>(
        &self,
        artifacts: impl Iterator<Item = &'a TaskArtifact>,
        working_directory: Option<&std::path::Path>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<Vec<Entity<language::Buffer>>>>> {
        let project = self.project.read(cx);
        let is_local = project.is_local();
        if project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
            &project.worktree_store(),
            cx,
        ) {
            bail!("Trust the project before reading Task artifacts");
        }
        let paths = artifacts
            .filter(|artifact| is_local || artifact.kind != task::TaskArtifactKind::Data)
            .map(|artifact| {
                let absolute = working_directory
                    .map(|directory| directory.join(&artifact.path))
                    .unwrap_or_else(|| artifact.path.clone().into());
                let path = project
                    .find_project_path(&absolute, cx)
                    .context("Task artifact is outside the project")?;
                if !project
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == path.worktree_id)
                {
                    bail!("Task artifact worktree is not visible");
                }
                Ok((path, absolute, artifact.max_bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let fs = project.fs().clone();
        let project = self.project.clone();
        Ok(cx.spawn(async move |_, cx| {
            let deadline = cx
                .background_executor()
                .timer(std::time::Duration::from_secs(10));
            let preparation = async {
                if !is_local {
                    let mut buffers = Vec::new();
                    for (path, _, limit) in paths {
                        let limit = usize::try_from(limit)?;
                        let buffer = project
                            .update(cx, |project, cx| {
                                project.open_buffer_with_limit(path, limit, cx)
                            })
                            .await?;
                        buffers.push(buffer);
                    }
                    return Ok(buffers);
                }
                let mut versions = Vec::new();
                for (path, absolute, limit) in paths {
                    let metadata = fs
                        .metadata(&absolute)
                        .await?
                        .context("Task artifact was not created")?;
                    if metadata.is_dir || metadata.is_fifo || metadata.len > limit {
                        bail!("Task artifact is not a bounded regular file");
                    }
                    versions.push((path, metadata));
                }
                let refreshes = project.read_with(cx, |project, cx| {
                    versions
                        .iter()
                        .map(|(path, _)| {
                            let worktree = project
                                .worktree_for_id(path.worktree_id, cx)
                                .context("Task worktree was removed")?;
                            let worktree = worktree
                                .read(cx)
                                .as_local()
                                .context("Task worktree is no longer local")?;
                            Ok(worktree.refresh_entries_for_paths(vec![path.path.clone()]))
                        })
                        .collect::<Result<Vec<_>>>()
                })?;
                for mut refresh in refreshes {
                    refresh
                        .next()
                        .await
                        .context("Task artifact refresh was cancelled")?;
                }
                project.read_with(cx, |project, cx| {
                    for (path, metadata) in versions {
                        let entry = project
                            .entry_for_path(&path, cx)
                            .context("Task artifact is not visible after refresh")?;
                        if !entry.is_file()
                            || entry.is_private
                            || entry.is_external
                            || entry.is_fifo
                            || entry.inode != metadata.inode
                            || entry.mtime != Some(metadata.mtime)
                            || entry.size != metadata.len
                        {
                            bail!("Task artifact changed or is not permitted after refresh");
                        }
                    }
                    Ok(())
                })?;
                Ok(Vec::new())
            };
            match futures::future::select(Box::pin(preparation), Box::pin(deadline)).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => {
                    Err(anyhow!("Task artifact refresh timed out"))
                }
            }
        }))
    }

    fn publish_task_artifact_group(
        &mut self,
        run_id: uuid::Uuid,
        group: task::TaskArtifactGroup,
        working_directory: Option<&std::path::Path>,
        host_verified: bool,
        current: Arc<AtomicBool>,
        mut subscriptions: Vec<gpui::Subscription>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let project = self.project.read(cx);
        let mut files = BTreeMap::new();
        for (role, artifact) in &group.files {
            let path = resolve_task_artifact(project, artifact, working_directory, cx)?;
            let entry = project
                .entry_for_path(&path, cx)
                .ok_or_else(|| anyhow!("Task artifact is no longer available"))?;
            if entry.is_external || entry.is_fifo {
                bail!("Task artifact is not a permitted regular file");
            }
            files.insert(role.clone(), (path, entry.clone()));
        }
        let paths: HashSet<_> = files.values().map(|(path, _)| path.clone()).collect();
        subscriptions.push(cx.subscribe(&self.project, {
            let current = current.clone();
            move |_, _, event, _| {
                if let project::Event::WorktreeUpdatedEntries(worktree_id, changes) = event
                    && changes.iter().any(|(path, _, _)| {
                        paths.contains(&project::ProjectPath {
                            worktree_id: *worktree_id,
                            path: path.clone(),
                        })
                    })
                {
                    current.store(false, Ordering::Relaxed);
                }
            }
        }));
        cx.emit(crate::Event::TaskArtifactGroupReady(Rc::new(
            CompletedTaskArtifactGroup {
                run_id,
                group,
                files,
                workspace: cx.entity().downgrade(),
                project_id: self.project.entity_id(),
                host_verified,
                current,
                _subscriptions: subscriptions,
            },
        )));
        Ok(())
    }

    fn open_task_artifact(
        &mut self,
        artifact: TaskArtifact,
        task_working_directory: Option<std::path::PathBuf>,
        current: Arc<AtomicBool>,
        subscriptions: Vec<gpui::Subscription>,
        buffers: Vec<Entity<language::Buffer>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match resolve_task_artifact(
            self.project.read(cx),
            &artifact,
            task_working_directory.as_deref(),
            cx,
        ) {
            Ok(project_path) => {
                let version = self
                    .project
                    .read(cx)
                    .entry_for_path(&project_path, cx)
                    .cloned();
                let max_bytes = match usize::try_from(artifact.max_bytes) {
                    Ok(max_bytes) => max_bytes,
                    Err(error) => {
                        self.show_toast(
                            Toast::new(
                                NotificationId::unique::<TaskArtifact>(),
                                format!("Task artifact byte limit is invalid: {error}"),
                            ),
                            cx,
                        );
                        return;
                    }
                };
                let buffer = match buffers.into_iter().next() {
                    Some(buffer) => Task::ready(Ok(buffer)),
                    None => self.project.update(cx, |project, cx| {
                        project.open_buffer_with_limit(project_path.clone(), max_bytes, cx)
                    }),
                };
                let loading = cx.spawn_in(window, {
                    let project_path = project_path.clone();
                    async move |workspace, cx| {
                        let buffer = buffer.await?;
                        let item = workspace
                            .update_in(cx, |workspace, window, cx| {
                                workspace.load_path(project_path, window, cx)
                            })?
                            .await?;
                        anyhow::Ok((buffer, item))
                    }
                });
                let pane = self
                    .last_active_center_pane
                    .clone()
                    .unwrap_or_else(|| self.active_pane().downgrade());
                cx.spawn_in(window, async move |workspace, cx| {
                    let _subscriptions = subscriptions;
                    let result = async {
                        let (buffer, (entry_id, build_item)) = loading.await?;
                        workspace.update_in(cx, |workspace, window, cx| {
                            if buffer.read(cx).is_dirty() || buffer.read(cx).len() > max_bytes {
                                bail!("Task artifact buffer changed while opening");
                            }
                            if !current.load(Ordering::Relaxed) {
                                bail!("Task project session changed while loading its artifact");
                            }
                            let current_path = resolve_task_artifact(
                                workspace.project.read(cx),
                                &artifact,
                                task_working_directory.as_deref(),
                                cx,
                            )?;
                            if current_path != project_path {
                                bail!("Task artifact changed while loading");
                            }
                            let unchanged = version
                                .as_ref()
                                .zip(workspace.project.read(cx).entry_for_path(&current_path, cx))
                                .is_some_and(|(before, after)| {
                                    before.id == after.id
                                        && before.inode == after.inode
                                        && before.mtime == after.mtime
                                        && before.size == after.size
                                        && before.canonical_path == after.canonical_path
                                });
                            if !unchanged {
                                bail!("Task artifact version changed while loading");
                            }
                            pane.update(cx, |pane, cx| {
                                pane.open_item(
                                    entry_id,
                                    project_path,
                                    true,
                                    false,
                                    true,
                                    None,
                                    window,
                                    cx,
                                    build_item,
                                )
                            })?;
                            Ok(())
                        })?
                    }
                    .await;
                    if let Err(error) = result {
                        workspace.update(cx, |workspace, cx| {
                            workspace.show_toast(
                                Toast::new(
                                    NotificationId::unique::<TaskArtifact>(),
                                    format!("Task artifact could not be opened: {error}"),
                                ),
                                cx,
                            );
                        })?;
                    }
                    anyhow::Ok(())
                })
                .detach_and_log_err(cx);
            }
            Err(error) => self.show_toast(
                Toast::new(
                    NotificationId::unique::<TaskArtifact>(),
                    format!("Task artifact could not be opened: {error}"),
                ),
                cx,
            ),
        }
    }

    pub async fn save_for_task(
        workspace: &WeakEntity<Self>,
        save_strategy: SaveStrategy,
        cx: &mut AsyncWindowContext,
    ) {
        let save_action = match save_strategy {
            SaveStrategy::All => {
                let save_all = workspace.update_in(cx, |workspace, window, cx| {
                    let task = workspace.save_all_internal(SaveIntent::SaveAll, true, window, cx);
                    cx.background_spawn(async { task.await.map(|_| ()) })
                });
                save_all.ok()
            }
            SaveStrategy::Current => {
                let save_current = workspace.update_in(cx, |workspace, window, cx| {
                    workspace.save_active_item(SaveIntent::SaveAll, window, cx)
                });
                save_current.ok()
            }
            SaveStrategy::None => None,
        };
        if let Some(save_action) = save_action {
            save_action.log_err().await;
        }
    }

    async fn save_for_task_checked(
        workspace: &WeakEntity<Self>,
        save_strategy: SaveStrategy,
        cx: &mut AsyncWindowContext,
    ) -> Result<()> {
        match save_strategy {
            SaveStrategy::All => {
                let save = workspace.update_in(cx, |workspace, window, cx| {
                    workspace.save_all_internal(SaveIntent::SaveAll, true, window, cx)
                })?;
                anyhow::ensure!(save.await?, "Task prerequisite save was cancelled");
                Ok(())
            }
            SaveStrategy::Current => {
                let save = workspace.update_in(cx, |workspace, window, cx| {
                    workspace.save_active_item(SaveIntent::SaveAll, window, cx)
                })?;
                save.await
            }
            SaveStrategy::None => Ok(()),
        }
    }

    pub fn start_diagnostic_debug_session(
        &mut self,
        reviewed: project::debugger::dap_store::DiagnosticDebugLaunch,
        task_context: SharedTaskContext,
        authority: project::debugger::dap_store::DiagnosticDebugAuthority,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<project::debugger::dap_store::DiagnosticDebugSession>> {
        match self.debugger_provider.as_ref() {
            Some(provider) => {
                provider.start_diagnostic_session(reviewed, task_context, authority, window, cx)
            }
            None => Task::ready(Err(anyhow::anyhow!(
                "Open the existing Debugger panel before launching a diagnostic session"
            ))),
        }
    }

    pub fn start_debug_session(
        &mut self,
        scenario: DebugScenario,
        task_context: SharedTaskContext,
        active_buffer: Option<Entity<Buffer>>,
        worktree_id: Option<WorktreeId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        #[cfg(feature = "jvm-tools")]
        let mut worktree_id = worktree_id;
        #[cfg(feature = "jvm-tools")]
        if scenario.adapter.as_ref() == "Java" {
            worktree_id = worktree_id.or_else(|| {
                active_buffer
                    .as_ref()
                    .and_then(|buffer| buffer.read(cx).file())
                    .map(|file| file.worktree_id(cx))
            });
            if worktree_id.is_none() {
                let mut worktrees = self.project.read(cx).visible_worktrees(cx);
                let first = worktrees.next();
                if worktrees.next().is_none() {
                    worktree_id = first.map(|worktree| worktree.read(cx).id());
                }
            }
            let message = match worktree_id {
                None => Some("Select a single project worktree before starting Java Debug"),
                Some(id) if !groovy_jvm_worktree_trusted(&self.project, id, cx) => {
                    Some("Trust the project before starting Java Debug")
                }
                Some(_) => None,
            };
            if let Some(message) = message {
                self.show_toast(
                    Toast::new(NotificationId::unique::<DebugScenario>(), message),
                    cx,
                );
                return;
            }
        }
        if let Some(provider) = self.debugger_provider.as_mut() {
            provider.start_session(
                scenario,
                task_context,
                active_buffer,
                worktree_id,
                window,
                cx,
            )
        }
    }

    pub fn spawn_in_terminal(
        self: &mut Workspace,
        spawn_in_terminal: SpawnInTerminal,
        window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Task<Option<Result<ExitStatus>>> {
        if let Some(terminal_provider) = self.terminal_provider.as_ref() {
            terminal_provider.spawn(spawn_in_terminal, window, cx)
        } else {
            Task::ready(None)
        }
    }

    pub fn run_create_worktree_tasks(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let project = self.project().clone();
        let hooks = HashSet::from_iter([TaskHook::CreateWorktree]);

        let worktree_tasks: Vec<(WorktreeId, TaskContext, Vec<TaskTemplate>)> = {
            let project = project.read(cx);
            let task_store = project.task_store();
            let Some(inventory) = task_store.read(cx).task_inventory().cloned() else {
                return;
            };

            let git_store = project.git_store().read(cx);

            let mut worktree_tasks = Vec::new();
            for worktree in project.worktrees(cx) {
                let worktree = worktree.read(cx);
                let worktree_id = worktree.id();
                let worktree_abs_path = worktree.abs_path();

                let templates: Vec<TaskTemplate> = inventory
                    .read(cx)
                    .templates_with_hooks(&hooks, worktree_id)
                    .into_iter()
                    .map(|(_, template)| template)
                    .collect();

                if templates.is_empty() {
                    continue;
                }

                let mut task_variables = TaskVariables::default();
                task_variables.insert(
                    VariableName::WorktreeRoot,
                    worktree_abs_path.to_string_lossy().into_owned(),
                );

                if let Some(path) = git_store.original_repo_path_for_worktree(worktree_id, cx) {
                    task_variables.insert(
                        VariableName::MainGitWorktree,
                        path.to_string_lossy().into_owned(),
                    );
                }

                let task_context = TaskContext {
                    cwd: Some(worktree_abs_path.to_path_buf()),
                    task_variables,
                    project_env: Default::default(),
                };

                worktree_tasks.push((worktree_id, task_context, templates));
            }
            worktree_tasks
        };

        if worktree_tasks.is_empty() {
            return;
        }

        let task = cx.spawn_in(window, async move |workspace, cx| {
            let mut tasks = Vec::new();
            for (worktree_id, task_context, templates) in worktree_tasks {
                let id_base = format!("worktree_setup_{worktree_id}");

                tasks.push(cx.spawn({
                    let workspace = workspace.clone();
                    async move |cx| {
                        for task_template in templates {
                            let Some(resolved) =
                                task_template.resolve_task(&id_base, &task_context)
                            else {
                                continue;
                            };

                            let status = workspace.update_in(cx, |workspace, window, cx| {
                                workspace.spawn_in_terminal(resolved.resolved, window, cx)
                            })?;

                            if let Some(result) = status.await {
                                match result {
                                    Ok(exit_status) if !exit_status.success() => {
                                        log::error!(
                                            "Git worktree setup task failed with status: {:?}",
                                            exit_status.code()
                                        );
                                        break;
                                    }
                                    Err(error) => {
                                        log::error!("Git worktree setup task error: {error:#}");
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        anyhow::Ok(())
                    }
                }));
            }

            futures::future::join_all(tasks).await;
            anyhow::Ok(())
        });
        task.detach_and_log_err(cx);
    }
}

pub fn resolve_task_artifact(
    project: &project::Project,
    artifact: &TaskArtifact,
    task_working_directory: Option<&std::path::Path>,
    cx: &gpui::App,
) -> Result<project::ProjectPath> {
    if project.is_disconnected(cx)
        || structured_task_connection_error(project.remote_connection_state(cx)).is_some()
    {
        bail!("The project host is disconnected");
    }
    if project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
        &project.worktree_store(),
        cx,
    ) {
        bail!("Trust the project before opening task artifacts");
    }
    let candidate = task_working_directory
        .map(|working_directory| working_directory.join(&artifact.path))
        .unwrap_or_else(|| artifact.path.clone().into());
    let project_path = project
        .find_project_path(&candidate, cx)
        .ok_or_else(|| anyhow!("{} is not visible in the project", artifact.path))?;
    if !project
        .visible_worktrees(cx)
        .any(|worktree| worktree.read(cx).id() == project_path.worktree_id)
    {
        bail!("Task artifact worktree is not visible");
    }
    let entry = project
        .entry_for_path(&project_path, cx)
        .ok_or_else(|| anyhow!("{} was not created", artifact.path))?;
    if !entry.is_file() || entry.is_private || entry.is_external || entry.is_fifo {
        bail!("Task artifact is not a permitted file");
    }
    if entry.size > artifact.max_bytes {
        bail!(
            "{} exceeds the declared {}-byte limit",
            artifact.path,
            artifact.max_bytes
        );
    }
    Ok(project_path)
}

fn native_profile_capture_disclosure_paths(
    workspace: &Workspace,
    group: &TaskArtifactGroup,
    working_directory: Option<&Path>,
    current: &AtomicBool,
    cx: &gpui::App,
) -> Result<[PathBuf; 2]> {
    anyhow::ensure!(
        current.load(Ordering::Relaxed),
        "The native capture invocation changed"
    );
    let project = workspace.project.read(cx);
    anyhow::ensure!(
        project.supports_terminal(cx)
            && !project.is_read_only(cx)
            && structured_task_connection_error(project.remote_connection_state(cx)).is_none(),
        "The native capture host is unavailable"
    );
    anyhow::ensure!(
        !project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
            &project.worktree_store(),
            cx
        ),
        "Trust the project before native capture"
    );
    let paths = pending_remote_profile_paths(project, group, working_directory, cx)?;
    anyhow::ensure!(
        paths[0].worktree_id == paths[1].worktree_id && paths[0] != paths[1],
        "Native capture artifacts must be distinct files in one visible worktree"
    );
    let absolute = |path: &project::ProjectPath| -> Result<PathBuf> {
        use settings::Settings as _;
        let location = settings::SettingsLocation {
            worktree_id: path.worktree_id,
            path: &path.path,
        };
        let settings = project::WorktreeSettings::get(Some(location), cx);
        anyhow::ensure!(
            !settings.is_path_private(&path.path) && !settings.is_path_excluded(&path.path),
            "The native capture path is private or excluded"
        );
        anyhow::ensure!(
            project
                .entry_for_path(path, cx)
                .is_none_or(|entry| !entry.is_private && !entry.is_external && !entry.is_fifo),
            "The native capture path is not permitted"
        );
        let absolute = project
            .absolute_path(path, cx)
            .ok_or_else(|| anyhow!("The native capture worktree changed"))?;
        let text = absolute
            .to_str()
            .ok_or_else(|| anyhow!("Native capture paths must be UTF-8"))?;
        anyhow::ensure!(
            absolute.is_absolute()
                && text.len() <= 16 * 1024
                && !text.chars().any(char::is_control),
            "Native capture paths cannot be shown safely"
        );
        Ok(absolute)
    };
    Ok([absolute(&paths[0])?, absolute(&paths[1])?])
}

async fn confirm_native_profile_capture(
    workspace: &WeakEntity<Workspace>,
    group: &TaskArtifactGroup,
    working_directory: Option<&Path>,
    current: &AtomicBool,
    revoked: &NativeProfileRetirement,
    cx: &mut AsyncWindowContext,
) -> Result<()> {
    let paths = workspace.read_with(cx, |workspace, cx| {
        native_profile_capture_disclosure_paths(workspace, group, working_directory, current, cx)
    })??;
    let detail = format!(
        "Profile: {}\nSymbols: {}\n\nProfiles can contain function names, source and library paths, and workload details. Processing stays local to the authoritative host; profile contents are not uploaded or sent as telemetry. Only an explicitly consented remote import transfers a bounded, filtered view. These two raw files remain on that host after the viewer is closed. Delete them through the project file controls when no longer needed. Closing or clearing the viewer releases its session-only result; it does not delete these files. This approval applies only to this capture.",
        paths[0].display(),
        paths[1].display()
    );
    let permit = workspace.read_with(cx, |workspace, _| {
        NativeProfilePromptPermit::reserve(workspace.native_profile_prompt_slots.clone())
    })??;
    let answer = cx.prompt(
        gpui::PromptLevel::Warning,
        "Capture a native profile on the project host?",
        Some(&detail),
        &["Cancel", "Capture"],
    );
    let mut owner = NativeProfileRetirementOwner {
        owned: Some((answer, permit)),
        retirement: None,
    };
    let answer = cx.spawn(async move |_| match owner.owned.as_mut() {
        Some((answer, _)) => answer.await.map_err(anyhow::Error::from),
        None => Err(anyhow!("Native capture prompt owner disappeared")),
    });
    let answer = match futures::future::select(answer, revoked.clone()).await {
        futures::future::Either::Left((answer, _)) => answer?,
        futures::future::Either::Right((_, answer)) => {
            // Keep the platform answer receiver alive without retaining launch authority.
            cx.spawn(async move |_| {
                if answer.await.is_err() {
                    log::trace!("Obsolete native capture prompt was dismissed");
                }
            })
            .detach();
            return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
        }
    };
    if answer != 1 || !current.load(Ordering::Relaxed) {
        return Err(anyhow::Error::new(crate::TaskLaunchCancelled));
    }
    let current_paths = workspace.read_with(cx, |workspace, cx| {
        native_profile_capture_disclosure_paths(workspace, group, working_directory, current, cx)
    })??;
    anyhow::ensure!(
        current_paths == paths,
        "The native capture paths changed during approval"
    );
    Ok(())
}

fn ensure_generated_task_policy(
    source: &TaskSourceKind,
    template: &task::TaskTemplate,
) -> Result<()> {
    if matches!(source, TaskSourceKind::Language { name } if name.as_ref() == "Cargo" || name.as_ref() == "Rust Tests")
    {
        anyhow::ensure!(
            template.experimental_policy_version == task::EXPERIMENTAL_TASK_POLICY_VERSION,
            "This saved generated Rust Task predates experimental admission; refresh and replan it before running"
        );
    }
    Ok(())
}

fn experimental_capture_feature(
    report: Option<&task::TaskStdoutReport>,
    group: Option<&TaskArtifactGroup>,
    required: &[String],
) -> Option<&'static str> {
    if report.is_some_and(|report| report.native_profile.is_some())
        || group.is_some_and(native_profile_group)
    {
        Some("rust-native-profiling-experimental")
    } else if group.is_some_and(|group| group.format == "llvm-cov-json-v1")
        || required
            .iter()
            .any(|feature| feature == "rust-coverage-experimental")
    {
        Some("rust-coverage-experimental")
    } else {
        None
    }
}

fn native_profile_group(group: &task::TaskArtifactGroup) -> bool {
    group.format == "samply-0.13.1"
        && group.files.len() == 2
        && group.files.get("profile").is_some_and(|file| {
            file.kind == task::TaskArtifactKind::Data
                && file.max_bytes <= 32 * 1024 * 1024
                && file.path.ends_with(".json")
                && !file.path.ends_with(".syms.json")
        })
        && group.files.get("symbols").is_some_and(|file| {
            file.kind == task::TaskArtifactKind::Data
                && file.max_bytes <= 16 * 1024 * 1024
                && file.path.ends_with(".syms.json")
        })
}

fn profile_task_command_supported(command: &str) -> bool {
    !command.is_empty()
        && !command.starts_with('~')
        && !command.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || matches!(
                    character,
                    ';' | '|' | '&' | '<' | '>' | '`' | '$' | '\\' | '"' | '\''
                )
        })
}

fn pending_remote_profile_paths(
    project: &project::Project,
    group: &task::TaskArtifactGroup,
    working_directory: Option<&std::path::Path>,
    cx: &gpui::App,
) -> Result<[project::ProjectPath; 2]> {
    if !native_profile_group(group) {
        bail!("Unsupported profile Task artifact group");
    }
    let working_directory = working_directory.ok_or_else(|| {
        anyhow!("Remote native profiling Tasks need an explicit project working directory")
    })?;
    if !working_directory.is_absolute() {
        bail!("Remote native profiling Tasks need an absolute project working directory");
    }
    let path = |role| -> Result<project::ProjectPath> {
        let artifact = group
            .files
            .get(role)
            .ok_or_else(|| anyhow!("Missing {role} artifact"))?;
        let candidate = working_directory.join(&artifact.path);
        let path = project
            .find_project_path(&candidate, cx)
            .ok_or_else(|| anyhow!("{role} output is outside a visible worktree"))?;
        if !project
            .visible_worktrees(cx)
            .any(|worktree| worktree.read(cx).id() == path.worktree_id)
        {
            bail!("{role} output worktree is not visible");
        }
        Ok(path)
    };
    Ok([path("profile")?, path("symbols")?])
}

fn wrap_remote_profile_task(
    mut original: SpawnInTerminal,
    binding: project::profile_results_store::ProfileTaskBinding,
) -> SpawnInTerminal {
    let Some(command) = original.command.take() else {
        return original;
    };
    original.preserve_command_label = true;
    original.literal_prefix_args = 5 + original.literal_prefix_args.min(original.args.len());
    original.command = Some(binding.helper_path);
    let arguments = std::iter::once("observe-task".to_string())
        .chain([
            "--socket".to_string(),
            binding.socket_path,
            "--".to_string(),
            command,
        ])
        .chain(original.args)
        .collect();
    original.args = arguments;
    original
        .env
        .insert("ZED_PROFILE_TASK_TOKEN".into(), binding.token);
    original
}

fn structured_task_connection_error(state: Option<ConnectionState>) -> Option<&'static str> {
    match state {
        None | Some(ConnectionState::Connected) => None,
        Some(ConnectionState::Connecting) => Some("The project host is still connecting"),
        Some(
            ConnectionState::Disconnected
            | ConnectionState::HeartbeatMissed
            | ConnectionState::Reconnecting,
        ) => Some("The project host is disconnected"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TerminalProvider,
        item::test::{TestItem, TestProjectItem},
        register_serializable_item,
    };
    use gpui::{App, BorrowAppContext as _, TestAppContext};
    use parking_lot::Mutex;
    use project::{FakeFs, Fs as _, Project, TaskSourceKind};
    use serde_json::json;
    use std::sync::Arc;
    use task::{TaskArtifactKind, TaskTemplate};

    #[test]
    fn remote_profile_launch_keeps_the_displayed_command_and_original_task() {
        let original = SpawnInTerminal {
            command: Some("cargo".into()),
            args: vec!["build".into(), "--release".into()],
            command_label: "cargo build --release".into(),
            cwd: Some("/fixture".into()),
            ..SpawnInTerminal::default()
        };
        let launch = wrap_remote_profile_task(
            original.clone(),
            project::profile_results_store::ProfileTaskBinding {
                helper_path: "/host/zed-remote-server".into(),
                socket_path: "/private/socket".into(),
                token: "42".repeat(32),
            },
        );
        assert_eq!(original.command.as_deref(), Some("cargo"));
        assert!(!original.env.contains_key("ZED_PROFILE_TASK_TOKEN"));
        assert_eq!(launch.command.as_deref(), Some("/host/zed-remote-server"));
        assert_eq!(
            launch.args,
            [
                "observe-task",
                "--socket",
                "/private/socket",
                "--",
                "cargo",
                "build",
                "--release"
            ]
        );
        assert_eq!(launch.command_label, original.command_label);
        assert!(launch.preserve_command_label);
        assert_eq!(launch.cwd, original.cwd);
        let run_id = uuid::Uuid::new_v4();
        let invocation = TaskInvocationContext::from_launch(run_id, &launch, None, None);
        assert_eq!(invocation.run_id(), run_id);
        assert_eq!(invocation.task_id(), &launch.id);
        assert_eq!(invocation.command(), launch.command.as_deref());
        assert_eq!(invocation.arguments(), launch.args.as_slice());
        assert_eq!(invocation.working_directory(), launch.cwd.as_deref());
        assert!(invocation.artifact().is_none());
        assert!(invocation.artifact_group().is_none());
        assert_eq!(
            launch.env.get("ZED_PROFILE_TASK_TOKEN"),
            Some(&"42".repeat(32))
        );
        assert!(profile_task_command_supported("/usr/bin/cargo"));
        assert!(!profile_task_command_supported("cargo nextest"));
        assert!(!profile_task_command_supported("~/bin/cargo"));
    }

    #[test]
    fn remote_profile_launch_keeps_literal_prefix_bounded() {
        for (original_prefix, expected_prefix) in [(0, 5), (1, 6), (3, 8), (usize::MAX, 8)] {
            let original = SpawnInTerminal {
                command: Some("/host/bin/collector".into()),
                args: vec!["record".into(), "data with spaces".into(), "$HOME".into()],
                literal_prefix_args: original_prefix,
                command_label: "Reviewed capture".into(),
                cwd: Some("/host/workspace".into()),
                env: collections::HashMap::from_iter([("WORKLOAD_ENV".into(), "preserved".into())]),
                ..SpawnInTerminal::default()
            };
            let launch = wrap_remote_profile_task(
                original.clone(),
                project::profile_results_store::ProfileTaskBinding {
                    helper_path: "/host/zed-remote-server".into(),
                    socket_path: "/private/observer".into(),
                    token: "42".repeat(32),
                },
            );
            assert_eq!(launch.literal_prefix_args, expected_prefix);
            assert!(launch.literal_prefix_args <= launch.args.len());
            assert_eq!(launch.args.get(5..), Some(original.args.as_slice()));
            assert_eq!(launch.command_label, original.command_label);
            assert!(launch.preserve_command_label);
            assert_eq!(launch.cwd, original.cwd);
            assert_eq!(launch.id, original.id);
            assert_eq!(launch.save, original.save);
            assert_eq!(
                launch.env.get("WORKLOAD_ENV"),
                original.env.get("WORKLOAD_ENV")
            );
            assert!(!original.env.contains_key("ZED_PROFILE_TASK_TOKEN"));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn remote_profile_launch_preserves_literal_arguments_at_actual_shell_helper(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let received = cx
            .background_spawn(async move {
                let directory = tempfile::tempdir()?;
                smol::fs::write(
                    directory.path().join("observe-task"),
                    b"test \"$1\" = --socket || exit 31\nreceipt=$2\ntest \"$3\" = -- || exit 32\nshift 3\nprintf '%s\\0' \"$@\" > \"$receipt\"\n",
                )
                .await?;
                let mut received = Vec::new();
                for shell in [
                    "/bin/sh",
                    "/bin/bash",
                    #[cfg(target_os = "macos")]
                    "/bin/zsh",
                ] {
                    for mode in ["literal", "legacy-tail", "oversized"] {
                        let receipt = directory.path().join(format!(
                            "{}-{mode}.argv",
                            Path::new(shell)
                                .file_name()
                                .and_then(|name| name.to_str())
                                .context("receiver shell name")?
                        ));
                        let sentinel = directory.path().join("must-not-exist");
                        let (arguments, prefix, expected) = match mode {
                            "literal" => {
                                let arguments = vec![
                                    "profile with spaces.json".to_string(),
                                    "profile with spaces.syms.json".to_string(),
                                    "$ZED_FILE".to_string(),
                                    "${ZED_FILE}".to_string(),
                                    "$HOME".to_string(),
                                    "日本語 ' \" ; & | < >".to_string(),
                                    "$(touch \"$RUST_PROFILE_EXPANSION_SENTINEL\")".to_string(),
                                    String::new(),
                                ];
                                (arguments.clone(), arguments.len(), arguments)
                            }
                            "legacy-tail" => (
                                vec![
                                    "$RUST_PROFILE_LITERAL_VALUE".into(),
                                    "$RUST_PROFILE_LEGACY_VALUE".into(),
                                ],
                                1,
                                vec![
                                    "$RUST_PROFILE_LITERAL_VALUE".into(),
                                    "expanded-legacy".into(),
                                ],
                            ),
                            "oversized" => (
                                vec!["$HOME".into(), "argument with spaces".into()],
                                usize::MAX,
                                vec!["$HOME".into(), "argument with spaces".into()],
                            ),
                            _ => anyhow::bail!("unexpected receiver mode"),
                        };
                        let original = SpawnInTerminal {
                            command: Some("/host/bin/collector".into()),
                            args: arguments,
                            literal_prefix_args: prefix,
                            command_label: "Reviewed native capture".into(),
                            cwd: Some(directory.path().to_path_buf()),
                            env: collections::HashMap::from_iter([
                                ("RUST_PROFILE_LITERAL_VALUE".into(), "must-stay-literal".into()),
                                ("RUST_PROFILE_LEGACY_VALUE".into(), "expanded-legacy".into()),
                                (
                                    "RUST_PROFILE_EXPANSION_SENTINEL".into(),
                                    sentinel.to_string_lossy().into_owned(),
                                ),
                            ]),
                            ..SpawnInTerminal::default()
                        };
                        let launch = wrap_remote_profile_task(
                            original,
                            project::profile_results_store::ProfileTaskBinding {
                                helper_path: "/bin/sh".into(),
                                socket_path: receipt.to_string_lossy().into_owned(),
                                token: "42".repeat(32),
                            },
                        );
                        let (program, arguments) = util::shell_builder::ShellBuilder::new(
                            &task::Shell::Program(shell.into()),
                            false,
                        )
                        .build_with_literal_prefix(
                            launch.command.clone(),
                            &launch.args,
                            launch.literal_prefix_args,
                        );
                        let mut command = std::process::Command::new(program);
                        command.args(arguments).envs(launch.env).current_dir(
                            launch.cwd.context("receiver cwd")?,
                        );
                        let output = util::process::Child::spawn(
                            command,
                            std::process::Stdio::null(),
                            std::process::Stdio::piped(),
                            std::process::Stdio::piped(),
                        )?
                        .output()
                        .await?;
                        anyhow::ensure!(output.status.success(), "{shell}/{mode} helper failed");
                        let actual = smol::fs::read(receipt).await?;
                        let expected = std::iter::once("/host/bin/collector".to_string())
                            .chain(expected)
                            .flat_map(|argument| {
                                argument.into_bytes().into_iter().chain(std::iter::once(0))
                            })
                            .collect::<Vec<_>>();
                        anyhow::ensure!(
                            smol::fs::metadata(&sentinel).await.is_err_and(|error| {
                                error.kind() == std::io::ErrorKind::NotFound
                            }),
                            "shell substitution created the injection sentinel"
                        );
                        received.push((shell.to_string(), mode.to_string(), actual, expected));
                    }
                }
                anyhow::Ok(received)
            })
            .await
            .expect("actual shell/helper receiver completes");
        for (shell, mode, actual, expected) in received {
            assert_eq!(actual, expected, "{shell}/{mode} exact helper-bound argv");
            println!(
                "native-profile-wrapper shell={shell} mode={mode} argv_bytes={}",
                actual.len()
            );
        }
    }

    struct Fixture {
        fs: Arc<FakeFs>,
        workspace: Entity<Workspace>,
        item: Entity<TestItem>,
        task: ResolvedTask,
        dirty_before_spawn: Arc<Mutex<Option<bool>>>,
    }

    #[cfg(feature = "jvm-tools")]
    struct RecordingOperationTerminal(Arc<Mutex<Vec<SpawnInTerminal>>>, bool);

    #[cfg(feature = "jvm-tools")]
    impl TerminalProvider for RecordingOperationTerminal {
        fn spawn(
            &self,
            task: SpawnInTerminal,
            _window: &mut Window,
            _cx: &mut App,
        ) -> Task<Option<Result<ExitStatus>>> {
            self.0.lock().push(task);
            if self.1 {
                Task::ready(Some(Err(anyhow::anyhow!("private-canary terminal setup"))))
            } else {
                Task::ready(Some(Ok(ExitStatus::default())))
            }
        }
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn discovered_jvm_tasks_recheck_live_provider_after_save_and_before_terminal(
        cx: &mut TestAppContext,
    ) {
        use futures::StreamExt as _;
        use language::{FakeLspAdapter, Language, LanguageConfig, LanguageMatcher};
        use project::jvm_project::{JvmBuildOperation, JvmBuildOperationReceipt};
        cx.update(|cx| {
            if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
            }
        });
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        fixture
            .fs
            .insert_tree(
                "/root/application",
                json!({"pom.xml":"<project/>","Main.java":"class Main {}"}),
            )
            .await;
        cx.update(|_, cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let project = fixture
            .workspace
            .read_with(cx, |workspace, _| workspace.project.clone());
        let (store, lsp_store, languages, worktree_id) = project.read_with(cx, |project, cx| {
            let store = project.worktree_store();
            let id = store
                .read(cx)
                .worktrees()
                .next()
                .expect("worktree")
                .read(cx)
                .id();
            (store, project.lsp_store(), project.languages().clone(), id)
        });
        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx).expect("trust")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &store,
                HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        languages.add(Arc::new(Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        )));
        let mut servers=languages.register_fake_lsp("Java",FakeLspAdapter{name:"jdtls",capabilities:lsp::ServerCapabilities{
            execute_command_provider:Some(lsp::ExecuteCommandOptions{commands:["java.project.getAll","java.project.getSettings","groovy.maven.importedProjects","groovy.gradle.importedProjects","groovy.maven.projects"].into_iter().map(str::to_owned).collect(),..Default::default()}),..Default::default()},
            initializer:Some(Box::new(|server| {server.set_request_handler::<lsp::request::ExecuteCommand,_,_>(|params,_|async move {
                match params.command.as_str() {
                    "java.project.getAll"=>Ok(Some(json!(["file:///root/application/"]))),
                    "java.project.getSettings"=>Ok(Some(json!({"org.eclipse.jdt.ls.core.sourcePaths":[],"org.eclipse.jdt.ls.core.classpathEntries":[]}))),
                    "groovy.maven.importedProjects"|"groovy.gradle.importedProjects"=>Ok(Some(json!({"version":1,"rootUri":"file:///root/","projects":if params.command=="groovy.maven.importedProjects" {vec![json!({"projectUri":"file:///root/application/","javaNature":true})]}else{vec![]},"partial":false,"fullSynchronizationKnown":false}))),
                    "groovy.maven.projects"=>Ok(Some(json!({"version":1,"rootUri":"file:///root/","cachedOnly":true,"reactorOrderKnown":false,"unavailablePomUris":[],
                        "lifecycles":[{"id":"default","phases":["verify"]}],"projects":[{"pomUri":"file:///root/application/pom.xml","groupId":"fixture","artifactId":"application","version":"1","packaging":"jar","moduleUris":[],"profiles":{},"inheritedProfileDefinitionsKnown":false,"plugins":[]}]}))),
                    _=>anyhow::bail!("Unexpected operation fixture command"),
                }
            });})),..Default::default()});
        let (_buffer, _open) = project
            .update(cx, |project, cx| {
                project.open_local_buffer_with_lsp("/root/application/Main.java", cx)
            })
            .await
            .expect("source");
        let _provider = servers.next().await.expect("live provider");
        let refresh = |cx: &mut gpui::VisualTestContext| {
            cx.run_until_parked();
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
            cx.run_until_parked();
            for _ in 0..20 {
                if lsp_store.read_with(cx, |store, _| {
                    store
                        .jdtls_project_snapshot(worktree_id)
                        .is_some_and(|(_, current)| current)
                }) {
                    return;
                }
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(300));
                cx.run_until_parked();
            }
            panic!("operation fixture did not obtain current metadata");
        };
        refresh(cx);
        let current = |cx: &mut gpui::VisualTestContext| {
            lsp_store.read_with(cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("live metadata");
                assert!(current);
                let receipt = JvmBuildOperationReceipt::new(
                    snapshot,
                    current,
                    snapshot.projects[0].module_root.clone(),
                    vec![JvmBuildOperation::MavenPhase {
                        lifecycle: "default".into(),
                        phase: "verify".into(),
                    }],
                    util::paths::PathStyle::Unix,
                )
                .expect("current receipt");
                (receipt, snapshot.clone())
            })
        };
        let inventory = project.read_with(cx, |project, cx| {
            project
                .task_store()
                .read(cx)
                .task_inventory()
                .expect("inventory")
                .clone()
        });
        let spawns = Arc::new(Mutex::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(RecordingOperationTerminal(spawns.clone(), false))
        });
        let (receipt, snapshot) = current(cx);
        fixture
            .workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.run_jvm_operation(receipt.clone(), window, cx)
            })
            .expect("current Task registration");
        cx.run_until_parked();
        assert_eq!(spawns.lock().len(), 1);
        assert_eq!(
            spawns.lock()[0].args,
            ["--offline", "--file", "application/pom.xml", "verify"]
        );
        assert_eq!(spawns.lock()[0].literal_prefix_args, 4);
        let (source, resolved) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(receipt, &snapshot, true, cx)
            })
            .expect("held receipt");
        let outcome = Arc::new(Mutex::new(None));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                resolved.clone(),
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
            lsp_store.update(cx, |store, cx| store.refresh_jdtls_project(worktree_id, cx));
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(
            spawns.lock().len(),
            1,
            "save-boundary invalidation prevents terminal spawn"
        );
        refresh(cx);
        let (replacement, snapshot) = current(cx);
        let (_, replacement_task) = inventory
            .update(cx, |inventory, cx| {
                inventory.register_jvm_operation(replacement.clone(), &snapshot, true, cx)
            })
            .expect("fresh generation");
        assert_ne!(resolved.id, replacement_task.id);
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(source.clone(), resolved.clone(), false, window, cx)
        });
        cx.run_until_parked();
        assert_eq!(
            spawns.lock().len(),
            1,
            "history cannot revive a retired provider generation"
        );
        fixture
            .workspace
            .update_in(cx, |workspace, window, cx| {
                workspace.run_jvm_operation(replacement.clone(), window, cx)
            })
            .expect("recover from refresh");
        cx.run_until_parked();
        assert_eq!(spawns.lock().len(), 2);
        let mut altered = replacement_task.clone();
        altered.resolved.literal_prefix_args = 0;
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(source.clone(), altered, false, window, cx)
        });
        cx.run_until_parked();
        assert_eq!(spawns.lock().len(), 2, "changed quote boundary denied");
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(RecordingOperationTerminal(spawns.clone(), true));
        });
        let handle = fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_structured_handle(
                source.clone(),
                replacement_task.clone(),
                false,
                window,
                cx,
            )
        });
        cx.run_until_parked();
        match handle.state() {
            task::StructuredTaskState::SpawnError { message } => {
                assert!(message.contains("wrapper") && message.contains("retry"));
                assert!(!message.contains("canary") && !message.contains("terminal setup"));
            }
            state => panic!("expected sanitized spawn failure, got {state:?}"),
        }
        assert_eq!(spawns.lock().len(), 3);
        trust.update(cx, |trust, cx| {
            trust.restrict(
                store.downgrade(),
                HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(source, replacement_task, false, window, cx)
        });
        cx.run_until_parked();
        assert_eq!(spawns.lock().len(), 3, "revoked trust denies held action");
    }

    #[cfg(feature = "jvm-tools")]
    struct RecordingDebuggerProvider(Arc<Mutex<Vec<Option<WorktreeId>>>>);

    #[cfg(feature = "jvm-tools")]
    impl crate::DebuggerProvider for RecordingDebuggerProvider {
        fn start_session(
            &self,
            _definition: DebugScenario,
            _task_context: SharedTaskContext,
            _active_buffer: Option<Entity<Buffer>>,
            worktree_id: Option<WorktreeId>,
            _window: &mut Window,
            _cx: &mut App,
        ) {
            self.0.lock().push(worktree_id);
        }

        fn spawn_task_or_modal(
            &self,
            _workspace: &mut Workspace,
            _action: &crate::Spawn,
            _window: &mut Window,
            _cx: &mut Context<Workspace>,
        ) {
        }

        fn task_scheduled(&self, _cx: &mut App) {}

        fn debug_scenario_scheduled(&self, _cx: &mut App) {}

        fn debug_scenario_scheduled_last(&self, _cx: &App) -> bool {
            false
        }

        fn active_thread_state(
            &self,
            _cx: &App,
        ) -> Option<project::debugger::session::ThreadStatus> {
            None
        }
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn java_debug_requires_a_current_trusted_worktree(cx: &mut TestAppContext) {
        cx.update(|cx| {
            if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
            }
        });
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = fixture.workspace.read_with(cx, |workspace, cx| {
            let store = workspace.project.read(cx).worktree_store();
            let id = store
                .read(cx)
                .worktrees()
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id();
            (store, id)
        });
        let starts = Arc::new(Mutex::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_debugger_provider(RecordingDebuggerProvider(starts.clone()));
        });
        let scenario = DebugScenario {
            adapter: "Java".into(),
            label: "Java main".into(),
            build: None,
            config: json!({"request": "launch", "mainClass": "evidence.Main"}),
            tcp_connection: None,
        };
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.start_debug_session(
                scenario.clone(),
                SharedTaskContext::default(),
                None,
                None,
                window,
                cx,
            );
        });
        assert!(starts.lock().is_empty());

        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("trust initialized")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.start_debug_session(
                scenario.clone(),
                SharedTaskContext::default(),
                None,
                None,
                window,
                cx,
            );
        });
        assert_eq!(*starts.lock(), vec![Some(worktree_id)]);

        fixture.workspace.update(cx, |workspace, cx| {
            workspace
                .project
                .update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
        });
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.start_debug_session(
                scenario,
                SharedTaskContext::default(),
                None,
                Some(worktree_id),
                window,
                cx,
            );
        });
        assert_eq!(*starts.lock(), vec![Some(worktree_id)]);
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_save_all(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::All).await;
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(
                TaskSourceKind::UserInput,
                fixture.task,
                false,
                window,
                cx,
            );
        });
        cx.executor().run_until_parked();

        assert_eq!(*fixture.dirty_before_spawn.lock(), Some(false));
        assert!(cx.read(|cx| !fixture.item.read(cx).is_dirty));
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_save_current(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        // Add a second inactive dirty item
        let inactive = add_test_item(&fixture.workspace, "file2.txt", false, cx);
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(
                TaskSourceKind::UserInput,
                fixture.task,
                false,
                window,
                cx,
            );
        });
        cx.executor().run_until_parked();

        // The active item (fixture.item) should be saved
        assert_eq!(*fixture.dirty_before_spawn.lock(), Some(false));
        assert!(cx.read(|cx| !fixture.item.read(cx).is_dirty));
        // The inactive item should not be saved
        assert!(cx.read(|cx| inactive.read(cx).is_dirty));
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_save_none(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(
                TaskSourceKind::UserInput,
                fixture.task,
                false,
                window,
                cx,
            );
        });
        cx.executor().run_until_parked();

        assert_eq!(*fixture.dirty_before_spawn.lock(), Some(true));
        assert!(cx.read(|cx| fixture.item.read(cx).is_dirty));
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_with_completion_reports_success(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        let task_result = Arc::new(Mutex::new(None));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                TaskSourceKind::UserInput,
                fixture.task,
                false,
                {
                    let task_result = task_result.clone();
                    move |result, _| {
                        *task_result.lock() = Some(result);
                    }
                },
                window,
                cx,
            );
        });
        cx.executor().run_until_parked();

        assert_eq!(*task_result.lock(), Some(ScheduledTaskResult::Success));
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_structured_handle_and_completion_share_result(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        let task_result = Arc::new(Mutex::new(None));
        let handle = fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_structured_handle_and_completion(
                TaskSourceKind::UserInput,
                fixture.task,
                false,
                {
                    let task_result = task_result.clone();
                    move |result, _| *task_result.lock() = Some(result)
                },
                window,
                cx,
            )
        });
        cx.run_until_parked();
        assert_eq!(*task_result.lock(), Some(ScheduledTaskResult::Success));
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::Completed { success: true, .. }
        ));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn experimental_pending_task_prerequisites_require_fresh_opt_in_operation(
        cx: &mut TestAppContext,
    ) {
        struct ActualPreparedTerminal(Entity<Project>);
        impl TerminalProvider for ActualPreparedTerminal {
            fn spawn(
                &self,
                _: SpawnInTerminal,
                _: &mut Window,
                _: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                panic!("marked Task must retain its original pending preparation");
            }

            fn spawn_prepared(
                &self,
                launch: SpawnInTerminal,
                handle: Option<StructuredTaskHandle>,
                prepare: crate::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let project = self.0.clone();
                window.spawn(cx, async move |cx| {
                    let prepared = match prepare(launch, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) => return Some(Err(error)),
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
                    let completed =
                        terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                    completed.await.map(Ok)
                })
            }
        }
        cx.executor().allow_parking();
        for feature in [
            "rust-named-cargo-presets-experimental",
            "rust-exact-doctest-actions-experimental",
        ] {
            let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
            let directory = tempfile::tempdir().expect("owned actual child directory");
            let marker = directory.path().join("executed");
            let mut resolved = fixture.task;
            resolved.resolved.command = Some("/bin/sh".into());
            resolved.resolved.args = vec![
                "-c".into(),
                "printf launched > \"$PENDING_TASK_MARKER\"".into(),
            ];
            resolved.resolved.literal_prefix_args = 2;
            resolved.resolved.cwd = Some(directory.path().to_path_buf());
            resolved.resolved.env.insert(
                "PENDING_TASK_MARKER".into(),
                marker.to_str().expect("fixture UTF-8").into(),
            );
            resolved.resolved.required_experimental_features = vec![feature.into()];
            let outcomes = Arc::new(Mutex::new(Vec::new()));
            let lock = fixture.workspace.read_with(cx, |workspace, _| {
                workspace.task_admission_prerequisite_lock.clone()
            });
            fixture.workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(ActualPreparedTerminal(workspace.project.clone()))
            });
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
            let held = lock.lock().await;
            let schedule = |cx: &mut gpui::VisualTestContext| {
                fixture.workspace.update_in(cx, |workspace, window, cx| {
                    workspace.schedule_resolved_task_with_admission_preflight_and_completion(
                        TaskSourceKind::UserInput,
                        resolved.clone(),
                        true,
                        |_, _| true,
                        {
                            let item = fixture.item.clone();
                            move |_, cx| {
                                assert!(
                                    !item.read(cx).is_dirty,
                                    "actual save precedes fresh admission"
                                );
                                Ok(())
                            }
                        },
                        {
                            let outcomes = outcomes.clone();
                            move |outcome, _| outcomes.lock().push(outcome)
                        },
                        window,
                        cx,
                    );
                });
            };
            schedule(cx);
            cx.run_until_parked();
            assert!(fixture.item.read_with(cx, |item, _| item.is_dirty));
            assert!(!marker.exists());
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(feature, false, cx)
            });
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
            drop(held);
            cx.run_until_parked();
            assert_eq!(*outcomes.lock(), [ScheduledTaskResult::Cancelled]);
            assert!(
                !marker.exists(),
                "re-enabled flag cannot rejuvenate original save-waiting Task"
            );
            schedule(cx);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while outcomes.lock().len() != 2 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "actual fresh Task did not complete"
                );
                cx.background_executor
                    .timer(std::time::Duration::from_millis(10))
                    .await;
            }
            assert_eq!(
                *outcomes.lock(),
                [ScheduledTaskResult::Cancelled, ScheduledTaskResult::Success]
            );
            assert_eq!(
                std::fs::read_to_string(marker).expect("actual child marker"),
                "launched"
            );
        }
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_admission_runs_after_save_and_can_reject(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::All).await;
        let admitted = Arc::new(Mutex::new(false));
        let result = Arc::new(Mutex::new(None));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_admission_and_completion(
                TaskSourceKind::UserInput,
                fixture.task,
                false,
                {
                    let item = fixture.item.clone();
                    let admitted = admitted.clone();
                    move |_, cx| {
                        assert!(
                            !item.read(cx).is_dirty,
                            "configured save finishes before admission"
                        );
                        *admitted.lock() = true;
                        bail!("controlled expired Task admission")
                    }
                },
                {
                    let result = result.clone();
                    move |completion, _| {
                        *result.lock() = Some(completion);
                    }
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert!(*admitted.lock());
        assert_eq!(
            *fixture.dirty_before_spawn.lock(),
            None,
            "rejection does not launch the terminal"
        );
        assert_eq!(*result.lock(), Some(ScheduledTaskResult::Cancelled));
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_admission_gates_prerequisites_not_terminal_execution(
        cx: &mut TestAppContext,
    ) {
        use futures::channel::oneshot;
        use std::cell::RefCell;

        type Completion = oneshot::Sender<()>;
        struct HeldTerminal(Rc<RefCell<Vec<Option<Completion>>>>);
        impl TerminalProvider for HeldTerminal {
            fn spawn(
                &self,
                _: SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let (release, completion) = oneshot::channel();
                self.0.borrow_mut().push(Some(release));
                cx.background_spawn(async move {
                    completion.await.expect("controlled terminal completion");
                    Some(Ok(ExitStatus::default()))
                })
            }
        }
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let terminals = Rc::new(RefCell::new(Vec::new()));
        let results = Arc::new(Mutex::new(Vec::new()));
        let admissions = Arc::new(Mutex::new(0));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(HeldTerminal(terminals.clone()));
        });
        let lock = fixture.workspace.read_with(cx, |workspace, _| {
            workspace.task_admission_prerequisite_lock.clone()
        });
        let held_prerequisites = lock.lock().await;
        let schedule = |owned, cx: &mut gpui::VisualTestContext| {
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace.schedule_resolved_task_with_admission_preflight_and_completion(
                    TaskSourceKind::UserInput,
                    fixture.task.clone(),
                    false,
                    move |_, _| owned,
                    {
                        let item = fixture.item.clone();
                        let admissions = admissions.clone();
                        move |_, cx| {
                            assert!(!item.read(cx).is_dirty, "admission follows configured save");
                            *admissions.lock() += 1;
                            Ok(())
                        }
                    },
                    {
                        let results = results.clone();
                        move |result, _| results.lock().push(result)
                    },
                    window,
                    cx,
                );
            });
        };
        schedule(true, cx);
        cx.run_until_parked();
        fixture.item.read_with(cx, |item, _| {
            assert!(item.is_dirty);
            assert_eq!(
                item.save_count, 0,
                "held prerequisite gate prevents the save itself"
            );
        });
        assert_eq!(*admissions.lock(), 0);
        assert!(terminals.borrow().is_empty());
        drop(held_prerequisites);
        cx.run_until_parked();
        assert_eq!(terminals.borrow().len(), 1);
        assert!(results.lock().is_empty(), "first terminal remains held");
        fixture.item.update(cx, |item, _| item.is_dirty = true);
        schedule(true, cx);
        cx.run_until_parked();
        assert_eq!(
            terminals.borrow().len(),
            2,
            "next Task must launch while first terminal execution is still held"
        );
        assert_eq!(*admissions.lock(), 2);
        assert_eq!(fixture.item.read_with(cx, |item, _| item.save_count), 2);
        assert!(results.lock().is_empty());
        fixture.item.update(cx, |item, _| item.is_dirty = true);
        schedule(false, cx);
        cx.run_until_parked();
        fixture.item.read_with(cx, |item, _| {
            assert!(
                item.is_dirty,
                "obsolete admission must not save a later source state"
            );
            assert_eq!(item.save_count, 2);
        });
        assert_eq!(
            *admissions.lock(),
            2,
            "obsolete Task does not consume post-save admission"
        );
        assert_eq!(terminals.borrow().len(), 2);
        assert_eq!(*results.lock(), [ScheduledTaskResult::Cancelled]);
        for index in 0..2 {
            terminals
                .borrow_mut()
                .get_mut(index)
                .expect("held terminal")
                .take()
                .expect("release each terminal once")
                .send(())
                .expect("Task is awaiting terminal completion");
            cx.run_until_parked();
            assert_eq!(results.lock().len(), index + 2);
        }
        assert_eq!(
            *results.lock(),
            [
                ScheduledTaskResult::Cancelled,
                ScheduledTaskResult::Success,
                ScheduledTaskResult::Success
            ]
        );
    }

    #[cfg(feature = "jvm-tools")]
    #[gpui::test]
    async fn groovy_build_waits_for_worktree_trust_before_spawning(cx: &mut TestAppContext) {
        cx.update(|cx| {
            if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
            }
        });
        let (mut fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        fixture.task = TaskTemplate {
            label: "Build Maven project".to_string(),
            command: "./mvnw".to_string(),
            args: vec!["package".to_string()],
            cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
            ..TaskTemplate::default()
        }
        .resolve_task(
            "Groovy JVM build",
            &TaskContext {
                cwd: Some("/root".into()),
                task_variables: {
                    let mut variables = TaskVariables::default();
                    variables.insert(VariableName::WorktreeRoot, "/root".to_string());
                    variables
                },
                ..TaskContext::default()
            },
        )
        .expect("Maven build task");
        cx.update(|_, cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = fixture.workspace.read_with(cx, |workspace, cx| {
            let worktree_store = workspace.project.read(cx).worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("fixture worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        let source = TaskSourceKind::Worktree {
            id: worktree_id,
            directory_in_worktree: util::rel_path::RelPath::empty_arc(),
            id_base: std::borrow::Cow::Borrowed("Groovy JVM build"),
        };
        let outcome = Arc::new(Mutex::new(None));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                fixture.task.clone(),
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);

        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("trust initialized")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                fixture.task.clone(),
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::Success));
        assert_eq!(*fixture.dirty_before_spawn.lock(), Some(true));

        let selected_test_source = TaskSourceKind::groovy_jvm_test(worktree_id);
        let selected_test = TaskTemplate {
            label: "Run GreetingTest".to_string(),
            command: "./mvnw".to_string(),
            args: vec![
                "-Dtest=GreetingTest#messageIsAvailable".to_string(),
                "test".to_string(),
            ],
            cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
            ..TaskTemplate::default()
        }
        .resolve_task(
            &selected_test_source.to_id_base(),
            &TaskContext {
                cwd: Some("/root".into()),
                task_variables: {
                    let mut variables = TaskVariables::default();
                    variables.insert(VariableName::WorktreeRoot, "/root".to_string());
                    variables
                },
                ..TaskContext::default()
            },
        )
        .expect("selected Java test Task");
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                selected_test_source,
                selected_test,
                true,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::Success));
        let last_task_id = fixture.workspace.read_with(cx, |workspace, cx| {
            workspace
                .project
                .read(cx)
                .task_store()
                .read(cx)
                .task_inventory()
                .and_then(|inventory| inventory.read(cx).last_scheduled_task(None))
                .map(|(_, task)| task.id)
        });
        assert_eq!(last_task_id, Some(fixture.task.id.clone()));

        let current_task = fixture.task.clone();
        fixture.task.resolved.args.push("-DskipTests".into());
        *fixture.dirty_before_spawn.lock() = None;
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                fixture.task.clone(),
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);

        fixture.task = current_task.clone();
        fixture.task.resolved.cwd = Some("/other-project".into());
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                fixture.task.clone(),
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);
        fixture.task = current_task;

        cx.update(|_, cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/mvnw".into()]));
                });
            });
        });
        cx.run_until_parked();
        fixture.workspace.read_with(cx, |workspace, cx| {
            let worktree = workspace
                .project
                .read(cx)
                .worktree_store()
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("JVM worktree");
            assert!(
                worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(util::rel_path::rel_path("mvnw"))
                    .expect("Maven wrapper")
                    .is_private
            );
        });
        *fixture.dirty_before_spawn.lock() = None;
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                fixture.task.clone(),
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);

        let project = fixture
            .workspace
            .read_with(cx, |workspace, _| workspace.project.clone());
        project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
        *fixture.dirty_before_spawn.lock() = None;
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source,
                fixture.task,
                false,
                {
                    let outcome = outcome.clone();
                    move |result, _| *outcome.lock() = Some(result)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*outcome.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    struct ProcessTerminalProvider;

    #[cfg(all(feature = "jvm-tools", unix))]
    impl TerminalProvider for ProcessTerminalProvider {
        fn spawn(
            &self,
            task: SpawnInTerminal,
            _window: &mut Window,
            cx: &mut App,
        ) -> Task<Option<Result<ExitStatus>>> {
            cx.background_spawn(async move {
                let result: Result<ExitStatus> = async {
                    let command = task.command.context("The Task has no executable")?;
                    let mut process = smol::process::Command::new(command);
                    process.args(task.args);
                    if let Some(cwd) = task.cwd {
                        process.current_dir(cwd);
                    }
                    Ok(process.status().await?)
                }
                .await;
                Some(result)
            })
        }
    }

    #[cfg(all(feature = "jvm-tools", unix))]
    #[gpui::test]
    async fn groovy_build_task_reports_host_process_outcomes(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt as _;

        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
            }
        });
        let root = tempfile::tempdir().expect("temporary host worktree");
        std::fs::write(root.path().join("pom.xml"), "<project/>").expect("write host manifest");
        let wrapper = root.path().join("mvnw");
        std::fs::write(
            &wrapper,
            "#!/bin/sh\npwd > task-cwd.txt\nprintf '%s\\n' \"$*\" > task-args.txt\n",
        )
        .expect("write host wrapper");
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755))
            .expect("make host wrapper executable");

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            root.path(),
            json!({ "pom.xml": "<project/>", "mvnw": "#!/bin/sh\n" }),
        )
        .await;
        let project = Project::test(fs, [root.path()], cx).await;
        let (workspace, cx) =
            cx.add_window_view(|window, cx| Workspace::test_new(project, window, cx));
        workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(ProcessTerminalProvider);
        });
        cx.update(|_, cx| project::trusted_worktrees::init(collections::HashMap::default(), cx));
        let (worktree_store, worktree_id) = workspace.read_with(cx, |workspace, cx| {
            let worktree_store = workspace.project.read(cx).worktree_store();
            let worktree_id = worktree_store
                .read(cx)
                .worktrees()
                .next()
                .expect("host worktree")
                .read(cx)
                .id();
            (worktree_store, worktree_id)
        });
        let source = TaskSourceKind::groovy_jvm_build(worktree_id);
        let mut variables = TaskVariables::default();
        variables.insert(
            VariableName::WorktreeRoot,
            root.path().to_string_lossy().into_owned(),
        );
        let resolved = TaskTemplate {
            label: "Build Maven project".to_string(),
            command: "./mvnw".to_string(),
            args: vec!["package".to_string()],
            cwd: Some("$ZED_WORKTREE_ROOT".to_string()),
            ..TaskTemplate::default()
        }
        .resolve_task(
            &source.to_id_base(),
            &TaskContext {
                cwd: Some(root.path().to_path_buf()),
                task_variables: variables,
                ..TaskContext::default()
            },
        )
        .expect("detected Maven Task");

        let (sender, receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                resolved.clone(),
                false,
                move |result, _| sender.send(result).expect("blocked Task receiver"),
                window,
                cx,
            );
        });
        assert_eq!(
            receiver.await.expect("blocked Task result"),
            ScheduledTaskResult::SpawnFailed
        );
        assert!(!root.path().join("task-cwd.txt").exists());

        let trust = cx.read(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("trust initialized")
        });
        trust.update(cx, |trust, cx| {
            trust.trust(
                &worktree_store,
                HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(worktree_id)]),
                cx,
            )
        });
        let (sender, receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                resolved.clone(),
                false,
                move |result, _| sender.send(result).expect("host Task receiver"),
                window,
                cx,
            );
        });
        assert_eq!(
            receiver.await.expect("host Task result"),
            ScheduledTaskResult::Success
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join("task-cwd.txt"))
                .expect("host working directory")
                .trim(),
            root.path()
                .canonicalize()
                .expect("canonical host worktree")
                .to_string_lossy()
        );
        assert_eq!(
            std::fs::read_to_string(root.path().join("task-args.txt"))
                .expect("host arguments")
                .trim(),
            "package"
        );

        std::fs::write(&wrapper, "#!/bin/sh\nexit 7\n").expect("write failing wrapper");
        let (sender, receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source.clone(),
                resolved.clone(),
                false,
                move |result, _| sender.send(result).expect("failing Task receiver"),
                window,
                cx,
            );
        });
        assert_eq!(
            receiver.await.expect("failing Task result"),
            ScheduledTaskResult::Failure
        );

        std::fs::remove_file(wrapper).expect("remove host wrapper");
        let (sender, receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                source,
                resolved,
                false,
                move |result, _| sender.send(result).expect("missing Task receiver"),
                window,
                cx,
            );
        });
        assert_eq!(
            receiver.await.expect("missing Task result"),
            ScheduledTaskResult::SpawnFailed
        );
    }

    #[gpui::test]
    async fn profile_artifact_opening_is_bounded_to_visible_project_files(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/root",
            json!({
                "target": {
                    "profile.svg": "<svg></svg>",
                    "oversized.svg": "0123456789abcdef",
                }
            }),
        )
        .await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let artifact = TaskArtifact {
            path: "target/profile.svg".to_string(),
            kind: TaskArtifactKind::Svg,
            max_bytes: 1024,
        };
        let project_path = project
            .read_with(cx, |project, cx| {
                resolve_task_artifact(project, &artifact, Some("/root".as_ref()), cx)
            })
            .expect("visible bounded artifact should resolve");
        assert_eq!(project_path.path.as_unix_str(), "target/profile.svg");

        let missing = TaskArtifact {
            path: "target/missing.svg".to_string(),
            kind: TaskArtifactKind::Svg,
            max_bytes: 1024,
        };
        assert!(project.read_with(cx, |project, cx| {
            resolve_task_artifact(project, &missing, Some("/root".as_ref()), cx).is_err()
        }));
        let oversized = TaskArtifact {
            path: "target/oversized.svg".to_string(),
            max_bytes: 8,
            ..artifact
        };
        assert!(project.read_with(cx, |project, cx| {
            resolve_task_artifact(project, &oversized, Some("/root".as_ref()), cx).is_err()
        }));
    }

    struct ArtifactModel(project::ProjectPath);

    struct ArtifactLoadGate(Option<futures::channel::oneshot::Receiver<()>>);
    impl gpui::Global for ArtifactLoadGate {}

    impl project::ProjectItem for ArtifactModel {
        fn try_open(
            _project: &Entity<Project>,
            path: &project::ProjectPath,
            cx: &mut App,
        ) -> Option<Task<Result<Entity<Self>>>> {
            if cx.has_global::<ArtifactLoadGate>()
                && let Some(receiver) = cx.global_mut::<ArtifactLoadGate>().0.take()
            {
                let path = path.clone();
                return Some(cx.spawn(async move |cx| {
                    receiver.await?;
                    Ok(cx.new(|_| Self(path)))
                }));
            }
            Some(Task::ready(Ok(cx.new(|_| Self(path.clone())))))
        }

        fn entry_id(&self, _: &App) -> Option<project::ProjectEntryId> {
            None
        }
        fn project_path(&self, _: &App) -> Option<project::ProjectPath> {
            Some(self.0.clone())
        }
        fn is_dirty(&self) -> bool {
            false
        }
    }

    struct ArtifactView {
        focus: gpui::FocusHandle,
        path: project::ProjectPath,
    }

    impl crate::Item for ArtifactView {
        type Event = ();
        fn tab_content_text(&self, _: usize, _: &App) -> gpui::SharedString {
            "Task artifact".into()
        }
    }
    impl gpui::EventEmitter<()> for ArtifactView {}
    impl gpui::Focusable for ArtifactView {
        fn focus_handle(&self, _: &App) -> gpui::FocusHandle {
            self.focus.clone()
        }
    }
    impl gpui::Render for ArtifactView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl gpui::IntoElement {
            gpui::Empty
        }
    }
    impl crate::ProjectItem for ArtifactView {
        type Item = ArtifactModel;
        fn for_project_item(
            _: Entity<Project>,
            _: Option<&crate::Pane>,
            item: Entity<ArtifactModel>,
            _: &mut Window,
            cx: &mut Context<Self>,
        ) -> Self {
            Self {
                focus: cx.focus_handle(),
                path: item.read(cx).0.clone(),
            }
        }
    }

    struct ArtifactTerminalProvider(
        Mutex<Option<futures::channel::oneshot::Receiver<Option<Result<ExitStatus>>>>>,
    );

    impl TerminalProvider for ArtifactTerminalProvider {
        fn spawn(
            &self,
            _: task::SpawnInTerminal,
            _: &mut Window,
            cx: &mut App,
        ) -> Task<Option<Result<ExitStatus>>> {
            let receiver = self.0.lock().take().expect("one explicit spawn");
            cx.background_spawn(async move { receiver.await.expect("test terminal completion") })
        }
    }

    #[gpui::test]
    async fn profile_artifact_lifecycle_opens_only_permitted_success(cx: &mut TestAppContext) {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt as _;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt as _;

        for scenario in [
            "success",
            "failure",
            "cancelled",
            "spawn-error",
            "disconnect",
            "restricted",
            "removed",
            "missing",
            "oversized",
            "private",
            "external",
            "data",
            "late-success",
            "late-disconnect",
            "late-change",
        ] {
            cx.update(|cx| {
                if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                    cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
                }
            });
            let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
            cx.update(|_, cx| {
                crate::register_project_item::<ArtifactView>(cx);
                project::trusted_worktrees::init(collections::HashMap::default(), cx);
            });
            let (sender, receiver) = futures::channel::oneshot::channel();
            let load_sender = if scenario.starts_with("late-") {
                let (sender, receiver) = futures::channel::oneshot::channel();
                cx.update(|_, cx| cx.set_global(ArtifactLoadGate(Some(receiver))));
                Some(sender)
            } else {
                None
            };
            let path = match scenario {
                "missing" => "missing.svg",
                "private" => ".env.svg",
                "external" => "external.svg",
                "data" => "artifact.json",
                _ => "artifact.svg",
            };
            if scenario == "external" {
                fixture
                    .fs
                    .insert_tree("/outside", json!({"artifact.svg":"<svg></svg>"}))
                    .await;
                fixture
                    .fs
                    .create_symlink(
                        std::path::Path::new("/root/external.svg"),
                        "/outside/artifact.svg".into(),
                    )
                    .await
                    .expect("external artifact symlink");
                cx.run_until_parked();
            }
            let task = TaskTemplate {
                label: format!("profile artifact {scenario}"),
                command: "fixture-profiler".into(),
                cwd: Some("/root".into()),
                save: SaveStrategy::None,
                artifact: Some(TaskArtifact {
                    path: path.into(),
                    kind: if scenario == "data" {
                        TaskArtifactKind::Data
                    } else {
                        TaskArtifactKind::Svg
                    },
                    max_bytes: if scenario == "oversized" { 1 } else { 1024 },
                }),
                ..TaskTemplate::default()
            }
            .resolve_task("profile-lifecycle", &TaskContext::default())
            .unwrap_or_else(|| panic!("explicit artifact task: {scenario}"));
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace.terminal_provider = Some(Box::new(ArtifactTerminalProvider(Mutex::new(
                    Some(receiver),
                ))));
                workspace.schedule_resolved_task(TaskSourceKind::UserInput, task, true, window, cx);
            });
            cx.run_until_parked();
            if scenario == "disconnect" {
                let project = fixture
                    .workspace
                    .read_with(cx, |workspace, _| workspace.project.clone());
                project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost));
                cx.run_until_parked();
            }
            if matches!(scenario, "restricted" | "removed") {
                let project = fixture
                    .workspace
                    .read_with(cx, |workspace, _| workspace.project.clone());
                let (store, worktree_id) = project.read_with(cx, |project, cx| {
                    (
                        project.worktree_store(),
                        project
                            .visible_worktrees(cx)
                            .next()
                            .expect("fixture worktree")
                            .read(cx)
                            .id(),
                    )
                });
                if scenario == "removed" {
                    project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx));
                } else {
                    cx.update(|_, cx| {
                        let trust =
                            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                                .expect("trust initialized");
                        trust.update(cx, |trust, cx| {
                            trust.restrict(
                                store.downgrade(),
                                HashSet::from_iter([
                                    project::trusted_worktrees::PathTrust::Worktree(worktree_id),
                                ]),
                                cx,
                            )
                        });
                    });
                }
                cx.run_until_parked();
            }
            let outcome = match scenario {
                "failure" => Some(Ok(ExitStatus::from_raw(256))),
                "cancelled" => None,
                "spawn-error" => Some(Err(anyhow!("fixture spawn failure"))),
                _ => Some(Ok(ExitStatus::default())),
            };
            sender.send(outcome).expect("terminal waiter exists");
            cx.run_until_parked();
            if let Some(sender) = load_sender {
                fixture.workspace.read_with(cx, |workspace, cx| {
                    assert!(workspace.item_of_type::<ArtifactView>(cx).is_none());
                    assert!(
                        cx.global::<ArtifactLoadGate>().0.is_none(),
                        "loading reached gate: {scenario}"
                    );
                });
                if scenario == "late-disconnect" {
                    let project = fixture
                        .workspace
                        .read_with(cx, |workspace, _| workspace.project.clone());
                    project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost));
                } else if scenario == "late-change" {
                    fixture
                        .fs
                        .insert_tree("/root", json!({"artifact.svg":"<svg>changed</svg>"}))
                        .await;
                }
                cx.run_until_parked();
                sender.send(()).expect("pending artifact load");
                cx.run_until_parked();
            }
            fixture.workspace.read_with(cx, |workspace, cx| {
                let opened = workspace.item_of_type::<ArtifactView>(cx);
                assert_eq!(
                    opened.is_some(),
                    matches!(scenario, "success" | "late-success"),
                    "{scenario}"
                );
                if let Some(opened) = opened {
                    assert_eq!(opened.read(cx).path.path.as_unix_str(), "artifact.svg");
                }
            });
        }
    }

    #[gpui::test]
    async fn profile_artifact_lifecycle_opens_fresh_ignored_output(cx: &mut TestAppContext) {
        struct Producer(Arc<FakeFs>);
        impl TerminalProvider for Producer {
            fn spawn(
                &self,
                _: task::SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let fs = self.0.clone();
                cx.background_spawn(async move {
                    fs.insert_tree("/root/target", json!({"new.svg": "<svg></svg>"}))
                        .await;
                    Some(Ok(ExitStatus::default()))
                })
            }
        }
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        fixture
            .fs
            .insert_tree(
                "/root",
                json!({".git": {}, ".gitignore":"target\n", "target":{}}),
            )
            .await;
        cx.update(|_, cx| crate::register_project_item::<ArtifactView>(cx));
        cx.run_until_parked();
        let task = TaskTemplate {
            label: "fresh profile output".into(),
            command: "fixture-profiler".into(),
            cwd: Some("/root".into()),
            save: SaveStrategy::None,
            artifact: Some(TaskArtifact {
                path: "target/new.svg".into(),
                kind: TaskArtifactKind::Svg,
                max_bytes: 1024,
            }),
            ..TaskTemplate::default()
        }
        .resolve_task("fresh-profile", &TaskContext::default())
        .expect("explicit artifact Task");
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.set_terminal_provider(Producer(fixture.fs.clone()));
            workspace.schedule_resolved_task(TaskSourceKind::UserInput, task, true, window, cx);
        });
        cx.run_until_parked();
        fixture.workspace.read_with(cx, |workspace, cx| {
            let opened = workspace
                .item_of_type::<ArtifactView>(cx)
                .expect("fresh Task output opened");
            let path = &opened.read(cx).path;
            assert_eq!(path.path.as_unix_str(), "target/new.svg");
            assert!(
                workspace
                    .project
                    .read(cx)
                    .entry_for_path(path, cx)
                    .expect("indexed artifact")
                    .is_ignored
            );
        });
    }

    #[gpui::test]
    async fn task_artifact_group_completion_binds_session_and_versions(cx: &mut TestAppContext) {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt as _;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt as _;

        let mut run_ids = HashSet::default();
        for scenario in [
            "success",
            "failure",
            "cancelled",
            "spawn-error",
            "disconnect",
            "missing",
            "oversized",
            "changed-after",
            "disconnect-after",
        ] {
            let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
            let ready = Rc::new(std::cell::RefCell::new(Vec::new()));
            let _subscription = cx.update(|_, cx| {
                cx.subscribe(&fixture.workspace, {
                    let ready = ready.clone();
                    move |_, event, _| {
                        if let crate::Event::TaskArtifactGroupReady(group) = event {
                            ready.borrow_mut().push(group.clone());
                        }
                    }
                })
            });
            let project = fixture
                .workspace
                .read_with(cx, |workspace, _| workspace.project.clone());
            let group = task::TaskArtifactGroup {
                format: "samply-0.13.1".into(),
                files: BTreeMap::from_iter([
                    (
                        "profile".into(),
                        TaskArtifact {
                            path: "artifact.json".into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: if scenario == "oversized" { 1 } else { 1024 },
                        },
                    ),
                    (
                        "symbols".into(),
                        TaskArtifact {
                            path: if scenario == "missing" {
                                "missing.json"
                            } else {
                                "symbols.json"
                            }
                            .into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 1024,
                        },
                    ),
                ]),
            };
            let task = TaskTemplate {
                label: "native profile".into(),
                command: "fixture-profiler".into(),
                cwd: Some("/root".into()),
                save: SaveStrategy::None,
                artifact_group: Some(group),
                ..TaskTemplate::default()
            }
            .resolve_task("native-profile", &TaskContext::default())
            .expect("explicit group");
            let (sender, receiver) = futures::channel::oneshot::channel();
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace.terminal_provider = Some(Box::new(ArtifactTerminalProvider(Mutex::new(
                    Some(receiver),
                ))));
                workspace.schedule_resolved_task(TaskSourceKind::UserInput, task, true, window, cx);
            });
            cx.run_until_parked();
            assert!(ready.borrow().is_empty(), "no offer before completion");
            if scenario == "disconnect" {
                project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost));
                cx.run_until_parked();
            }
            sender
                .send(match scenario {
                    "failure" => Some(Ok(ExitStatus::from_raw(256))),
                    "cancelled" => None,
                    "spawn-error" => Some(Err(anyhow!("fixture failure"))),
                    _ => Some(Ok(ExitStatus::default())),
                })
                .expect("terminal waiting");
            cx.run_until_parked();
            let expected = matches!(scenario, "success" | "changed-after" | "disconnect-after");
            assert_eq!(ready.borrow().len(), usize::from(expected), "{scenario}");
            if !expected {
                continue;
            }
            let group = ready.borrow()[0].clone();
            assert!(
                run_ids.insert(group.run_id()),
                "each execution has a distinct identity"
            );
            assert_eq!(group.group().format, "samply-0.13.1");
            assert_eq!(
                group.path("symbols").expect("sidecar").path.as_unix_str(),
                "symbols.json"
            );
            assert!(cx.read(|cx| group.is_current(&project, cx)), "{scenario}");
            if scenario == "disconnect-after" {
                project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost));
                cx.run_until_parked();
                assert!(!cx.read(|cx| group.is_current(&project, cx)));
            }
            if scenario == "changed-after" {
                fixture
                    .fs
                    .insert_tree("/root", json!({"symbols.json": "{\"changed\":true}"}))
                    .await;
                cx.run_until_parked();
                assert!(!cx.read(|cx| group.is_current(&project, cx)));
                fixture
                    .fs
                    .insert_tree("/root", json!({"symbols.json": "{}"}))
                    .await;
                cx.run_until_parked();
                assert!(
                    !cx.read(|cx| group.is_current(&project, cx)),
                    "restoration cannot revive an offer"
                );
            }
        }
    }

    #[test]
    fn cargo_pre_launch_task_reference_is_unique_and_uses_existing_task_resolution() {
        let context = TaskContext::default();
        let template = TaskTemplate {
            label: "Generate bindings".to_string(),
            command: "generator".to_string(),
            args: vec!["--checked".to_string()],
            ..TaskTemplate::default()
        };
        let (source, resolved) = resolve_task_reference(
            vec![(TaskSourceKind::UserInput, template.clone())],
            "Generate bindings",
            &context,
        )
        .expect("an exact existing Tasks label should resolve");
        assert_eq!(source, TaskSourceKind::UserInput);
        assert_eq!(resolved.resolved.command.as_deref(), Some("generator"));
        assert_eq!(resolved.resolved.args, vec!["--checked"]);

        assert!(
            resolve_task_reference(Vec::new(), "Generate bindings", &context)
                .expect_err("a missing reference must be isolated")
                .to_string()
                .contains("not found")
        );
        assert!(
            resolve_task_reference(
                vec![
                    (TaskSourceKind::UserInput, template.clone()),
                    (TaskSourceKind::UserInput, template),
                ],
                "Generate bindings",
                &context,
            )
            .expect_err("an ambiguous reference must not select arbitrarily")
            .to_string()
            .contains("ambiguous")
        );
    }

    fn resolved_binding_fixture(save: SaveStrategy) -> ResolvedTask {
        let template = TaskTemplate {
            label: "invocation capture".into(),
            command: "fixture-collector".into(),
            args: vec![
                "--output".into(),
                "target/report.json".into(),
                "retained".into(),
            ],
            cwd: Some("/root".into()),
            save,
            allow_concurrent_runs: true,
            artifact: Some(TaskArtifact {
                path: "target/report.json".into(),
                kind: TaskArtifactKind::Data,
                max_bytes: 1024,
            }),
            artifact_argument_binding: Some(task::TaskArtifactArgumentBinding {
                argument_index: 1,
            }),
            ..Default::default()
        };
        template
            .resolve_task("invocation-fixture", &task::TaskContext::default())
            .expect("valid invocation binding fixture")
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_artifact_invocation_rebinds_cached_and_original_routes(
        cx: &mut TestAppContext,
    ) {
        use futures::channel::oneshot;
        use std::{cell::RefCell, rc::Rc};
        type Release = oneshot::Sender<()>;
        struct HeldCapture {
            launches: Rc<RefCell<Vec<SpawnInTerminal>>>,
            releases: Rc<RefCell<Vec<Option<Release>>>>,
            fs: Arc<FakeFs>,
        }
        impl TerminalProvider for HeldCapture {
            fn spawn(
                &self,
                launch: SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let artifact_path = launch.args.get(1).expect("bound output argument").clone();
                let (release, completion) = oneshot::channel();
                self.launches.borrow_mut().push(launch);
                self.releases.borrow_mut().push(Some(release));
                let fs = self.fs.clone();
                cx.background_spawn(async move {
                    completion.await.expect("controlled terminal completion");
                    let artifact = std::path::Path::new(&artifact_path);
                    assert_eq!(artifact.parent(), Some(std::path::Path::new("target")));
                    let file_name = artifact
                        .file_name()
                        .and_then(|name| name.to_str())
                        .expect("portable bound filename");
                    fs.insert_tree("/root/target", json!({(file_name): "{}"}))
                        .await;
                    Some(Ok(ExitStatus::default()))
                })
            }
        }
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let launches = Rc::new(RefCell::new(Vec::new()));
        let releases = Rc::new(RefCell::new(Vec::new()));
        let admissions = Rc::new(RefCell::new(Vec::<TaskInvocationContext>::new()));
        let results = Rc::new(RefCell::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(HeldCapture {
                launches: launches.clone(),
                releases: releases.clone(),
                fs: fixture.fs.clone(),
            })
        });
        let fresh = resolved_binding_fixture(SaveStrategy::Current);
        let original_template = fresh.original_task().clone();
        let task_id = fresh.id.clone();
        let mut candidate = fresh;
        for route in 0..4 {
            fixture.item.update(cx, |item, _| item.is_dirty = true);
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace
                    .schedule_resolved_task_with_invocation_admission_preflight_and_completion(
                        TaskSourceKind::UserInput,
                        candidate.clone(),
                        false,
                        |_, _| true,
                        {
                            let item = fixture.item.clone();
                            let admissions = admissions.clone();
                            move |invocation, _, cx| {
                                assert!(
                                    !item.read(cx).is_dirty,
                                    "save precedes invocation admission"
                                );
                                assert_eq!(item.read(cx).save_count, route + 1);
                                admissions.borrow_mut().push(invocation.clone());
                                Ok(())
                            }
                        },
                        {
                            let results = results.clone();
                            move |result, _| results.borrow_mut().push(result)
                        },
                        window,
                        cx,
                    );
            });
            cx.run_until_parked();
            assert_eq!(launches.borrow().len(), route + 1);
            assert_eq!(admissions.borrow().len(), route + 1);
            assert!(
                results.borrow().is_empty(),
                "prior terminal does not hold prerequisite gate"
            );
            let inventory = fixture.workspace.read_with(cx, |workspace, cx| {
                workspace
                    .project
                    .read(cx)
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .cloned()
                    .expect("production history owner")
            });
            let (_, cached) = inventory.read_with(cx, |inventory, _| {
                inventory
                    .last_scheduled_task(None)
                    .expect("scheduled task in history")
            });
            assert_eq!(cached.id, task_id);
            let artifact = cached.resolved_artifact().expect("bound history artifact");
            assert_eq!(cached.resolved.args.get(1), Some(&artifact.path));
            assert_eq!(
                admissions
                    .borrow()
                    .last()
                    .expect("post-save admission")
                    .artifact(),
                Some(artifact)
            );
            assert_eq!(cached.original_task(), &original_template);
            candidate = match route {
                0 => cached,
                1 => cached.clone(),
                _ => cached
                    .original_task()
                    .resolve_task("invocation-fixture", &task::TaskContext::default())
                    .expect("original-template route"),
            };
        }
        {
            let admitted_invocations = admissions.borrow();
            let recorded_launches = launches.borrow();
            for (index, (invocation, launch)) in admitted_invocations
                .iter()
                .zip(recorded_launches.iter())
                .enumerate()
            {
                assert_eq!(invocation.task_id(), &task_id);
                assert_eq!(launch.id, task_id);
                assert_eq!(invocation.command(), launch.command.as_deref());
                assert_eq!(invocation.arguments(), launch.args.as_slice());
                assert_eq!(invocation.working_directory(), launch.cwd.as_deref());
                let artifact = invocation.artifact().expect("bound declaration");
                assert_eq!(launch.args.get(1), Some(&artifact.path));
                assert_eq!(launch.args.first().map(String::as_str), Some("--output"));
                assert_eq!(launch.args.get(2).map(String::as_str), Some("retained"));
                assert!(invocation.artifact_group().is_none());
                for older in admitted_invocations.iter().take(index) {
                    assert_ne!(older.run_id(), invocation.run_id());
                    assert_ne!(
                        older.artifact().expect("prior declaration").path,
                        artifact.path
                    );
                }
            }
        }
        for index in 0..4 {
            releases
                .borrow_mut()
                .get_mut(index)
                .expect("held run")
                .take()
                .expect("single release")
                .send(())
                .expect("terminal remains pending");
            cx.run_until_parked();
            assert_eq!(results.borrow().len(), index + 1);
            let path = admissions
                .borrow()
                .get(index)
                .expect("completed invocation")
                .artifact()
                .expect("bound output")
                .path
                .clone();
            assert_eq!(
                fixture
                    .fs
                    .load(Path::new("/root").join(path).as_path())
                    .await
                    .expect("collector wrote declared output"),
                "{}"
            );
        }
        assert_eq!(*results.borrow(), vec![ScheduledTaskResult::Success; 4]);
    }

    fn resolved_group_binding_fixture(save: SaveStrategy) -> ResolvedTask {
        let template = TaskTemplate {
            label: "invocation capture".into(),
            command: "fixture-collector".into(),
            args: vec![
                "--output".into(),
                "target/report.json".into(),
                "retained".into(),
            ],
            cwd: Some("/root".into()),
            save,
            allow_concurrent_runs: true,
            artifact_group: Some(task::TaskArtifactGroup {
                format: "paired-data-v1".into(),
                files: collections::BTreeMap::from_iter([
                    (
                        "profile".into(),
                        TaskArtifact {
                            path: "target/report.json".into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 1024,
                        },
                    ),
                    (
                        "symbols".into(),
                        TaskArtifact {
                            path: "target/report.syms.json".into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 2048,
                        },
                    ),
                ]),
            }),
            artifact_group_argument_binding: Some(task::TaskArtifactGroupArgumentBinding {
                argument_index: 1,
                role: "profile".into(),
                related: vec![task::TaskArtifactRelatedArgumentBinding {
                    role: "symbols".into(),
                    extension: "syms.json".into(),
                    argument_index: None,
                }],
            }),
            ..Default::default()
        };
        template
            .resolve_task("invocation-fixture", &task::TaskContext::default())
            .expect("valid invocation binding fixture")
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_artifact_group_invocation_rebinds_cached_and_original_routes(
        cx: &mut TestAppContext,
    ) {
        use futures::channel::oneshot;
        use std::{cell::RefCell, rc::Rc};
        type Release = oneshot::Sender<()>;
        struct HeldCapture {
            launches: Rc<RefCell<Vec<SpawnInTerminal>>>,
            releases: Rc<RefCell<Vec<Option<Release>>>>,
            fs: Arc<FakeFs>,
        }
        impl TerminalProvider for HeldCapture {
            fn spawn(
                &self,
                launch: SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let artifact_path = launch.args.get(1).expect("bound output argument").clone();
                let (release, completion) = oneshot::channel();
                self.launches.borrow_mut().push(launch);
                self.releases.borrow_mut().push(Some(release));
                let fs = self.fs.clone();
                cx.background_spawn(async move {
                    completion.await.expect("controlled terminal completion");
                    let artifact = std::path::Path::new(&artifact_path);
                    assert_eq!(artifact.parent(), Some(std::path::Path::new("target")));
                    let file_name = artifact
                        .file_name()
                        .and_then(|name| name.to_str())
                        .expect("portable bound filename");
                    let symbols = artifact.with_extension("syms.json");
                    let symbols_name = symbols
                        .file_name()
                        .and_then(|name| name.to_str())
                        .expect("portable companion filename");
                    fs.insert_tree(
                        "/root/target",
                        json!({(file_name): "{}", (symbols_name): "symbols"}),
                    )
                    .await;
                    Some(Ok(ExitStatus::default()))
                })
            }
        }
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let launches = Rc::new(RefCell::new(Vec::new()));
        let releases = Rc::new(RefCell::new(Vec::new()));
        let admissions = Rc::new(RefCell::new(Vec::<TaskInvocationContext>::new()));
        let results = Rc::new(RefCell::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(HeldCapture {
                launches: launches.clone(),
                releases: releases.clone(),
                fs: fixture.fs.clone(),
            })
        });
        let fresh = resolved_group_binding_fixture(SaveStrategy::Current);
        let original_template = fresh.original_task().clone();
        let task_id = fresh.id.clone();
        let mut candidate = fresh;
        for route in 0..4 {
            fixture.item.update(cx, |item, _| item.is_dirty = true);
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace
                    .schedule_resolved_task_with_invocation_admission_preflight_and_completion(
                        TaskSourceKind::UserInput,
                        candidate.clone(),
                        false,
                        |_, _| true,
                        {
                            let item = fixture.item.clone();
                            let admissions = admissions.clone();
                            move |invocation, _, cx| {
                                assert!(
                                    !item.read(cx).is_dirty,
                                    "save precedes invocation admission"
                                );
                                assert_eq!(item.read(cx).save_count, route + 1);
                                admissions.borrow_mut().push(invocation.clone());
                                Ok(())
                            }
                        },
                        {
                            let results = results.clone();
                            move |result, _| results.borrow_mut().push(result)
                        },
                        window,
                        cx,
                    );
            });
            cx.run_until_parked();
            assert_eq!(launches.borrow().len(), route + 1);
            assert_eq!(admissions.borrow().len(), route + 1);
            assert!(
                results.borrow().is_empty(),
                "prior terminal does not hold prerequisite gate"
            );
            let inventory = fixture.workspace.read_with(cx, |workspace, cx| {
                workspace
                    .project
                    .read(cx)
                    .task_store()
                    .read(cx)
                    .task_inventory()
                    .cloned()
                    .expect("production history owner")
            });
            let (_, cached) = inventory.read_with(cx, |inventory, _| {
                inventory
                    .last_scheduled_task(None)
                    .expect("scheduled task in history")
            });
            assert_eq!(cached.id, task_id);
            let group = cached
                .resolved_artifact_group()
                .expect("bound history group");
            let artifact = group.files.get("profile").expect("profile");
            let symbols = group.files.get("symbols").expect("symbols");
            assert_eq!(
                Path::new(&artifact.path).with_extension("syms.json"),
                Path::new(&symbols.path)
            );
            assert_eq!(cached.resolved.args.get(1), Some(&artifact.path));
            assert_eq!(
                admissions
                    .borrow()
                    .last()
                    .expect("post-save admission")
                    .artifact_group(),
                Some(group)
            );
            assert_eq!(cached.original_task(), &original_template);
            candidate = match route {
                0 => cached,
                1 => cached.clone(),
                _ => cached
                    .original_task()
                    .resolve_task("invocation-fixture", &task::TaskContext::default())
                    .expect("original-template route"),
            };
        }
        {
            let admitted_invocations = admissions.borrow();
            let recorded_launches = launches.borrow();
            for (index, (invocation, launch)) in admitted_invocations
                .iter()
                .zip(recorded_launches.iter())
                .enumerate()
            {
                assert_eq!(invocation.task_id(), &task_id);
                assert_eq!(launch.id, task_id);
                assert_eq!(invocation.command(), launch.command.as_deref());
                assert_eq!(invocation.arguments(), launch.args.as_slice());
                assert_eq!(invocation.working_directory(), launch.cwd.as_deref());
                assert!(invocation.artifact().is_none());
                let group = invocation
                    .artifact_group()
                    .expect("bound group declaration");
                let artifact = group.files.get("profile").expect("profile");
                let symbols = group.files.get("symbols").expect("symbols");
                assert_eq!(
                    Path::new(&artifact.path).with_extension("syms.json"),
                    Path::new(&symbols.path)
                );
                assert_eq!(launch.args.get(1), Some(&artifact.path));
                assert_eq!(launch.args.first().map(String::as_str), Some("--output"));
                assert_eq!(launch.args.get(2).map(String::as_str), Some("retained"));
                assert_eq!(group.files.len(), 2);
                for older in admitted_invocations.iter().take(index) {
                    assert_ne!(older.run_id(), invocation.run_id());
                    assert_ne!(
                        older
                            .artifact_group()
                            .expect("prior declaration")
                            .files
                            .get("profile")
                            .expect("prior profile")
                            .path,
                        artifact.path
                    );
                }
            }
        }
        for index in 0..4 {
            releases
                .borrow_mut()
                .get_mut(index)
                .expect("held run")
                .take()
                .expect("single release")
                .send(())
                .expect("terminal remains pending");
            cx.run_until_parked();
            assert_eq!(results.borrow().len(), index + 1);
            let path = admissions
                .borrow()
                .get(index)
                .expect("completed invocation")
                .artifact_group()
                .expect("bound output group")
                .files
                .get("profile")
                .expect("profile")
                .path
                .clone();
            let symbols = Path::new(&path).with_extension("syms.json");
            assert_eq!(
                fixture
                    .fs
                    .load(Path::new("/root").join(symbols).as_path())
                    .await
                    .expect("collector wrote declared companion"),
                "symbols"
            );
            assert_eq!(
                fixture
                    .fs
                    .load(Path::new("/root").join(path).as_path())
                    .await
                    .expect("collector wrote declared output"),
                "{}"
            );
        }
        assert_eq!(*results.borrow(), vec![ScheduledTaskResult::Success; 4]);
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_artifact_invocation_rejects_changed_binding_before_save(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let mut invalid = resolved_binding_fixture(SaveStrategy::Current);
        *invalid.resolved.args.get_mut(1).expect("selected argument") =
            "target/foreign.json".into();
        let result = Arc::new(Mutex::new(None));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_invocation_admission_preflight_and_completion(
                TaskSourceKind::UserInput,
                invalid,
                false,
                |_, _| true,
                |_, _, _| panic!("invalid binding must not reach admission"),
                {
                    let result = result.clone();
                    move |outcome, _| *result.lock() = Some(outcome)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*result.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);
        fixture.item.read_with(cx, |item, _| {
            assert!(item.is_dirty);
            assert_eq!(item.save_count, 0);
        });
        fixture.workspace.read_with(cx, |workspace, cx| {
            let task_store = workspace.project.read(cx).task_store();
            let inventory = task_store.read(cx).task_inventory().expect("history owner");
            assert!(inventory.read(cx).last_scheduled_task(None).is_none());
        });
        let mut invalid = resolved_binding_fixture(SaveStrategy::Current);
        *invalid.resolved.args.get_mut(1).expect("selected argument") =
            "target/foreign.json".into();
        let handle = fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_structured_handle(
                TaskSourceKind::UserInput,
                invalid,
                false,
                window,
                cx,
            )
        });
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::SpawnError { .. }
        ));
        cx.run_until_parked();
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);
        fixture.item.read_with(cx, |item, _| {
            assert!(item.is_dirty);
            assert_eq!(item.save_count, 0);
        });
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_artifact_group_invocation_rejects_changed_binding_before_save(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let mut invalid = resolved_group_binding_fixture(SaveStrategy::Current);
        *invalid.resolved.args.get_mut(1).expect("selected argument") =
            "target/foreign.json".into();
        let result = Arc::new(Mutex::new(None));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_invocation_admission_preflight_and_completion(
                TaskSourceKind::UserInput,
                invalid,
                false,
                |_, _| true,
                |_, _, _| panic!("invalid binding must not reach admission"),
                {
                    let result = result.clone();
                    move |outcome, _| *result.lock() = Some(outcome)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*result.lock(), Some(ScheduledTaskResult::SpawnFailed));
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);
        fixture.item.read_with(cx, |item, _| {
            assert!(item.is_dirty);
            assert_eq!(item.save_count, 0);
        });
        fixture.workspace.read_with(cx, |workspace, cx| {
            let task_store = workspace.project.read(cx).task_store();
            let inventory = task_store.read(cx).task_inventory().expect("history owner");
            assert!(inventory.read(cx).last_scheduled_task(None).is_none());
        });
        let mut invalid = resolved_group_binding_fixture(SaveStrategy::Current);
        *invalid.resolved.args.get_mut(1).expect("selected argument") =
            "target/foreign.json".into();
        let handle = fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_structured_handle(
                TaskSourceKind::UserInput,
                invalid,
                false,
                window,
                cx,
            )
        });
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::SpawnError { .. }
        ));
        cx.run_until_parked();
        assert_eq!(*fixture.dirty_before_spawn.lock(), None);
        fixture.item.read_with(cx, |item, _| {
            assert!(item.is_dirty);
            assert_eq!(item.save_count, 0);
        });
    }

    #[gpui::test]
    async fn test_schedule_resolved_task_artifact_invocation_keeps_unbound_tasks_ungated(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let lock = fixture.workspace.read_with(cx, |workspace, _| {
            workspace.task_admission_prerequisite_lock.clone()
        });
        let held_prerequisites = lock.lock().await;
        let result = Arc::new(Mutex::new(None));
        let original = fixture.task.clone();
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_completion(
                TaskSourceKind::UserInput,
                fixture.task.clone(),
                false,
                {
                    let result = result.clone();
                    move |outcome, _| *result.lock() = Some(outcome)
                },
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(*result.lock(), Some(ScheduledTaskResult::Success));
        assert_eq!(*fixture.dirty_before_spawn.lock(), Some(false));
        fixture.item.read_with(cx, |item, _| {
            assert!(!item.is_dirty);
            assert_eq!(item.save_count, 1);
        });
        fixture.workspace.read_with(cx, |workspace, cx| {
            let task_store = workspace.project.read(cx).task_store();
            let inventory = task_store.read(cx).task_inventory().expect("history owner");
            let (_, cached) = inventory
                .read(cx)
                .last_scheduled_task(None)
                .expect("ordinary history");
            assert_eq!(cached.id, original.id);
            assert_eq!(cached.resolved, original.resolved);
            assert_eq!(cached.original_task(), original.original_task());
            assert!(cached.resolved_artifact().is_none());
            assert!(cached.resolved_artifact_group().is_none());
        });
        drop(held_prerequisites);
    }

    async fn create_fixture(
        cx: &mut TestAppContext,
        save_strategy: SaveStrategy,
    ) -> (Fixture, &mut gpui::VisualTestContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            register_serializable_item::<TestItem>(cx);
        });
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/root",
            json!({ "file.txt": "dirty", "artifact.svg": "<svg></svg>", "artifact.json": "{}", "symbols.json": "{}", ".env.svg": "private", "pom.xml": "<project/>", "mvnw": "#!/bin/sh\n" }),
        )
        .await;
        let project = Project::test(fs.clone(), ["/root".as_ref()], cx).await;
        let (workspace, cx) =
            cx.add_window_view(|window, cx| Workspace::test_new(project.clone(), window, cx));

        // Add a dirty item to the workspace
        let item = add_test_item(&workspace, "file.txt", true, cx);

        let template = TaskTemplate {
            label: "test".to_string(),
            command: "echo".to_string(),
            save: save_strategy,
            ..Default::default()
        };
        let task = template
            .resolve_task("test", &task::TaskContext::default())
            .unwrap();
        let dirty_before_spawn: Arc<Mutex<Option<bool>>> = Arc::default();
        let terminal_provider = Box::new(TestTerminalProvider {
            item: item.clone(),
            dirty_before_spawn: dirty_before_spawn.clone(),
        });
        workspace.update(cx, |workspace, _| {
            workspace.terminal_provider = Some(terminal_provider);
        });
        let fixture = Fixture {
            fs,
            workspace,
            item,
            task,
            dirty_before_spawn,
        };
        (fixture, cx)
    }

    fn add_test_item(
        workspace: &Entity<Workspace>,
        name: &str,
        active: bool,
        cx: &mut gpui::VisualTestContext,
    ) -> Entity<TestItem> {
        let item = cx.new(|cx| {
            TestItem::new(cx)
                .with_dirty(true)
                .with_project_items(&[TestProjectItem::new(1, name, cx)])
        });
        workspace.update_in(cx, |workspace, window, cx| {
            let pane = workspace.active_pane().clone();
            workspace.add_item(pane, Box::new(item.clone()), None, true, active, window, cx);
        });
        item
    }

    #[gpui::test]
    async fn test_save_for_task_all(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::All).await;
        let workspace = fixture.workspace.downgrade();
        cx.run_until_parked();

        assert!(cx.read(|cx| fixture.item.read(cx).is_dirty));
        fixture.workspace.update_in(cx, |_workspace, window, cx| {
            cx.spawn_in(window, {
                let workspace = workspace.clone();
                async move |_this, cx| {
                    Workspace::save_for_task(&workspace, SaveStrategy::All, cx).await;
                }
            })
            .detach();
        });
        cx.run_until_parked();
        assert!(cx.read(|cx| !fixture.item.read(cx).is_dirty));
    }

    #[gpui::test]
    async fn test_save_for_task_none(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        let workspace = fixture.workspace.downgrade();
        cx.run_until_parked();

        assert!(cx.read(|cx| fixture.item.read(cx).is_dirty));
        fixture.workspace.update_in(cx, |_workspace, window, cx| {
            cx.spawn_in(window, {
                let workspace = workspace.clone();
                async move |_this, cx| {
                    Workspace::save_for_task(&workspace, SaveStrategy::None, cx).await;
                }
            })
            .detach();
        });
        cx.run_until_parked();
        assert!(cx.read(|cx| fixture.item.read(cx).is_dirty));
    }

    #[gpui::test]
    async fn test_save_for_task_current(cx: &mut TestAppContext) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        let inactive = add_test_item(&fixture.workspace, "file2.txt", false, cx);
        let workspace = fixture.workspace.downgrade();
        cx.run_until_parked();

        assert!(cx.read(|cx| fixture.item.read(cx).is_dirty));
        assert!(cx.read(|cx| inactive.read(cx).is_dirty));
        fixture.workspace.update_in(cx, |_workspace, window, cx| {
            cx.spawn_in(window, {
                let workspace = workspace.clone();
                async move |_this, cx| {
                    Workspace::save_for_task(&workspace, SaveStrategy::Current, cx).await;
                }
            })
            .detach();
        });
        cx.run_until_parked();
        assert!(cx.read(|cx| !fixture.item.read(cx).is_dirty));
        assert!(cx.read(|cx| inactive.read(cx).is_dirty));
    }

    #[gpui::test]
    async fn native_capture_scheduler_unanchored_home_cwd_discloses_frozen_directory(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let project = fixture
            .workspace
            .read_with(cx, |workspace, _| workspace.project.clone());
        let home = project
            .read_with(cx, |project, cx| {
                project.terminal_task_working_directory(Some(Path::new("~")), cx)
            })
            .expect("Actual expanded home directory")
            .to_path_buf();
        fixture
            .fs
            .insert_tree(&home, json!({"fixture.txt": "home fixture"}))
            .await;
        project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(&home, true, cx)
            })
            .await
            .expect("Visible fake home worktree");
        cx.run_until_parked();
        let mut template = native_capture_guard_fixture().original_task().clone();
        template.cwd = Some("~".into());
        let resolved = template
            .resolve_task("native-home-cwd", &TaskContext::default())
            .expect("Unanchored native Task");
        assert!(resolved.artifact_directory().is_none());
        assert_eq!(resolved.resolved.cwd.as_deref(), Some(Path::new("~")));
        let launches = Arc::new(Mutex::new(Vec::new()));
        let outcomes = Arc::new(Mutex::new(Vec::new()));
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.set_terminal_provider(PreparedReceiver(launches.clone()));
            let outcomes = outcomes.clone();
            workspace.schedule_resolved_task_with_completion(
                TaskSourceKind::UserInput,
                resolved,
                true,
                move |result, _| outcomes.lock().push(result),
                window,
                cx,
            );
        });
        cx.run_until_parked();
        let detail = cx.pending_prompt().expect("Frozen home consent").1;
        assert!(detail.contains(&format!("{}/capture-", home.display())));
        assert!(!detail.contains("~/capture-"));
        assert!(launches.lock().is_empty());
        cx.simulate_prompt_answer("Capture");
        cx.run_until_parked();
        let recorded = launches.lock();
        assert_eq!(recorded.len(), 1);
        let launch = recorded.first().expect("Prepared invocation");
        assert_eq!(launch.cwd.as_deref(), Some(home.as_path()));
        assert!(
            detail.contains(
                &home
                    .join(launch.args.first().expect("Bound output"))
                    .display()
                    .to_string()
            )
        );
        assert_eq!(
            outcomes.lock().as_slice(),
            &[ScheduledTaskResult::SpawnFailed]
        );
    }

    #[gpui::test]
    async fn native_capture_scheduler_separate_artifact_directory_holds_exact_pair_and_execution_cwd(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::Current).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        fixture
            .fs
            .insert_tree(
                "/root/package",
                json!({"package-only.txt": "relative input"}),
            )
            .await;
        cx.run_until_parked();
        let mut template = native_capture_guard_fixture().original_task().clone();
        template.cwd = Some("/root/package".into());
        template.save = SaveStrategy::Current;
        let unanchored = template
            .resolve_task("native-two-directories", &TaskContext::default())
            .expect("Execution directory Task");
        let resolved = unanchored
            .with_artifact_directory(Path::new("/root"))
            .expect("Literal artifact anchor data");
        let launches = Arc::new(Mutex::new(Vec::new()));
        let outcomes = Arc::new(Mutex::new(Vec::new()));
        let admitted = Arc::new(Mutex::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(PreparedReceiver(launches.clone()))
        });
        let schedule = |cx: &mut gpui::VisualTestContext| {
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                let outcomes = outcomes.clone();
                let admitted = admitted.clone();
                let item = fixture.item.clone();
                workspace
                    .schedule_resolved_task_with_invocation_admission_preflight_and_completion(
                        TaskSourceKind::UserInput,
                        resolved.clone(),
                        true,
                        |_, _| true,
                        move |invocation, _, cx| {
                            assert!(
                                !item.read(cx).is_dirty,
                                "Configured save precedes admission"
                            );
                            assert_eq!(
                                invocation.working_directory(),
                                Some(Path::new("/root/package"))
                            );
                            assert_eq!(invocation.artifact_directory(), Some(Path::new("/root")));
                            admitted.lock().push(invocation.clone());
                            Ok(())
                        },
                        move |result, _| outcomes.lock().push(result),
                        window,
                        cx,
                    );
            });
            cx.run_until_parked();
        };
        schedule(cx);
        let cancelled_detail = cx.pending_prompt().expect("Held anchored pair").1;
        assert!(cancelled_detail.contains("/root/capture-"));
        assert!(!cancelled_detail.contains("/root/package/capture-"));
        assert!(launches.lock().is_empty());
        cx.simulate_prompt_answer("Cancel");
        cx.run_until_parked();
        assert_eq!(
            outcomes.lock().as_slice(),
            &[ScheduledTaskResult::Cancelled]
        );
        assert!(launches.lock().is_empty());
        schedule(cx);
        let fresh_detail = cx
            .pending_prompt()
            .expect("Fresh independently bound pair")
            .1;
        assert_ne!(fresh_detail, cancelled_detail);
        assert!(launches.lock().is_empty());
        cx.simulate_prompt_answer("Capture");
        cx.run_until_parked();
        let recorded = launches.lock();
        assert_eq!(recorded.len(), 1);
        let launch = recorded.first().expect("Prepared anchored launch");
        assert_eq!(launch.cwd.as_deref(), Some(Path::new("/root/package")));
        let output = launch.args.first().expect("Absolute bound output");
        assert!(Path::new(output).is_absolute());
        assert_eq!(Path::new(output).parent(), Some(Path::new("/root")));
        assert!(fresh_detail.contains(output));
        let companion = Path::new(output).with_extension("syms.json");
        assert!(fresh_detail.contains(companion.to_str().expect("Literal companion")));
        assert_eq!(
            outcomes.lock().as_slice(),
            &[
                ScheduledTaskResult::Cancelled,
                ScheduledTaskResult::SpawnFailed
            ]
        );
        assert_eq!(admitted.lock().len(), 2);
    }

    #[gpui::test]
    async fn native_capture_scheduler_held_consent_cancel_supersession_and_fresh_launch(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let launches = Arc::new(Mutex::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(PreparedReceiver(launches.clone()))
        });
        let resolved = native_capture_guard_fixture();
        let outcomes = Arc::new(Mutex::new(Vec::new()));
        let schedule = |cx: &mut gpui::VisualTestContext| {
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                let outcomes = outcomes.clone();
                workspace.schedule_resolved_task_with_completion(
                    TaskSourceKind::UserInput,
                    resolved.clone(),
                    true,
                    move |result, _| outcomes.lock().push(result),
                    window,
                    cx,
                );
            });
            cx.run_until_parked();
        };
        schedule(cx);
        let (message, detail) = cx.pending_prompt().expect("Held Cancel prompt");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert!(detail.contains("/root/capture-"));
        assert!(detail.contains(".syms.json"));
        for term in [
            "not uploaded",
            "telemetry",
            "raw files remain",
            "Delete them",
            "session-only",
        ] {
            assert!(detail.contains(term), "{term}");
        }
        assert!(launches.lock().is_empty());
        cx.simulate_prompt_answer("Cancel");
        cx.run_until_parked();
        assert_eq!(
            outcomes.lock().as_slice(),
            &[ScheduledTaskResult::Cancelled]
        );
        assert!(launches.lock().is_empty());
        schedule(cx);
        let old_detail = cx.pending_prompt().expect("Old held approval").1;
        schedule(cx);
        assert!(launches.lock().is_empty());
        cx.simulate_prompt_answer("Capture");
        cx.run_until_parked();
        assert_eq!(
            outcomes.lock().as_slice(),
            &[
                ScheduledTaskResult::Cancelled,
                ScheduledTaskResult::Cancelled
            ]
        );
        assert!(launches.lock().is_empty(), "Late old approval cannot spawn");
        let fresh_detail = cx
            .pending_prompt()
            .expect("Replacement has its own prompt")
            .1;
        assert_ne!(
            fresh_detail, old_detail,
            "Every invocation has fresh exact paths"
        );
        cx.simulate_prompt_answer("Capture");
        cx.run_until_parked();
        assert_eq!(launches.lock().len(), 1);
        assert_eq!(
            outcomes.lock().as_slice(),
            &[
                ScheduledTaskResult::Cancelled,
                ScheduledTaskResult::Cancelled,
                ScheduledTaskResult::SpawnFailed
            ]
        );
        let launch = launches.lock()[0].clone();
        assert_eq!(launch.id, resolved.id);
        assert_eq!(launch.cwd.as_deref(), Some(Path::new("/root")));
        assert!(fresh_detail.contains(&format!("/root/{}", launch.args[0])));
        assert!(!cx.has_pending_prompt());
        let handle = fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task_with_structured_handle(
                TaskSourceKind::UserInput,
                resolved,
                true,
                window,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(cx.has_pending_prompt());
        cx.update(|_, cx| assert!(handle.cancel(cx)));
        cx.simulate_prompt_answer("Capture");
        cx.run_until_parked();
        assert_eq!(
            launches.lock().len(),
            1,
            "A cancelled structured Task cannot use its late approval"
        );
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));
    }

    #[gpui::test]
    fn native_selected_build_preserves_one_existing_structured_cancellation_owner(
        cx: &mut TestAppContext,
    ) {
        let template = TaskTemplate {
            label: "selected native prerequisite Build".into(),
            command: "/usr/bin/cargo".into(),
            args: vec!["build".into(), "--message-format=json".into()],
            literal_arguments: true,
            stdout_report: Some(task::TaskStdoutReport {
                format: task::TaskStdoutReportFormat::CargoBuildJson,
                max_bytes: 1024 * 1024,
                native_profile: Some(task::TaskNativeProfileContinuation {
                    collector_path: "/usr/local/bin/samply".into(),
                    profile_path: "target/native-profile.json".into(),
                    workload_arguments: vec!["literal$argument".into()],
                }),
            }),
            ..TaskTemplate::default()
        };
        let resolved = template
            .resolve_task(
                &TaskSourceKind::UserInput.to_id_base(),
                &TaskContext::default(),
            )
            .expect("validated declarative intent; no producer or capture credential");
        let report = resolved.stdout_report().expect("Build report declaration");
        let generated = native_build_structured_handle(None, &resolved.id, Some(report))
            .expect("ordinary selected-action/Rerun entry creates a genuine Build handle");
        assert_eq!(generated.task_id(), resolved.id);
        assert!(matches!(
            generated.state(),
            task::StructuredTaskState::Queued
        ));
        cx.update(|cx| assert!(generated.cancel(cx)));
        assert!(matches!(
            generated.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));

        let supplied = StructuredTaskHandle::new(resolved.id.clone());
        let retained =
            native_build_structured_handle(Some(supplied.clone()), &resolved.id, Some(report))
                .expect("explicit structured entry retains its existing owner");
        cx.update(|cx| assert!(supplied.cancel(cx)));
        assert!(
            matches!(
                retained.state(),
                task::StructuredTaskState::Cancelled { .. }
            ),
            "the supplied and scheduler handle share the same cancellation state"
        );
        assert_eq!(retained.task_id(), supplied.task_id());

        assert!(
            native_build_structured_handle(None, &resolved.id, None).is_none(),
            "ordinary Task entry must not acquire a new handle"
        );
        let report_only = task::TaskStdoutReport {
            format: task::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: 1024 * 1024,
            native_profile: None,
        };
        assert!(
            native_build_structured_handle(None, &resolved.id, Some(&report_only)).is_none(),
            "unsupported generic stdout metadata does not activate a native Build owner"
        );
        let ordinary_supplied = StructuredTaskHandle::new(TaskId("ordinary explicit owner".into()));
        let ordinary_retained =
            native_build_structured_handle(Some(ordinary_supplied.clone()), &resolved.id, None)
                .expect("ordinary structured entry remains intact");
        cx.update(|cx| assert!(ordinary_retained.cancel(cx)));
        assert!(matches!(
            ordinary_supplied.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn native_selected_retirement_drops_owned_receipt_before_signaling() {
        struct ReceiptDrop(Arc<AtomicBool>);
        impl Drop for ReceiptDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = futures::channel::oneshot::channel();
        let owner = NativeProfileRetirementOwner {
            owned: Some(ReceiptDrop(dropped.clone())),
            retirement: Some(Arc::new(NativeProfileRetirementSignal {
                sender: Some(sender),
                permit: Some(
                    NativeProfileRetirementPermit::reserve(Arc::new(
                        std::sync::atomic::AtomicUsize::new(0),
                    ))
                    .expect("private retirement permit"),
                ),
            })),
        };
        assert!(!dropped.load(Ordering::Relaxed));
        drop(owner);
        assert!(dropped.load(Ordering::Relaxed));
        assert!(
            receiver
                .now_or_never()
                .expect("retired after opaque receipt drop")
                .is_ok()
        );
        let slots = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let permits: Vec<_> = (0..MAX_NATIVE_PROFILE_PROMPT_RECEIPTS)
            .map(|_| {
                NativeProfilePromptPermit::reserve(slots.clone())
                    .expect("bounded unanswered prompt")
            })
            .collect();
        assert!(NativeProfilePromptPermit::reserve(slots.clone()).is_err());
        assert_eq!(
            slots.load(Ordering::Relaxed),
            MAX_NATIVE_PROFILE_PROMPT_RECEIPTS
        );
        drop(permits);
        assert_eq!(slots.load(Ordering::Relaxed), 0);
        drop(NativeProfilePromptPermit::reserve(slots.clone()).expect("fresh answer capacity"));
        assert_eq!(slots.load(Ordering::Relaxed), 0);
        let phase_slots = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let phase_permits: Vec<_> = (0..MAX_NATIVE_PROFILE_RETIREMENT_LEASES)
            .map(|_| {
                NativeProfileRetirementPermit::reserve(phase_slots.clone())
                    .expect("inclusive bounded phase lease")
            })
            .collect();
        assert!(NativeProfileRetirementPermit::reserve(phase_slots.clone()).is_err());
        assert_eq!(
            phase_slots.load(Ordering::Relaxed),
            MAX_NATIVE_PROFILE_RETIREMENT_LEASES
        );
        drop(phase_permits);
        assert_eq!(phase_slots.load(Ordering::Relaxed), 0);
        drop(
            NativeProfileRetirementPermit::reserve(phase_slots.clone())
                .expect("actual phase retirement permits recovery"),
        );
        assert_eq!(phase_slots.load(Ordering::Relaxed), 0);
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn native_selected_retirement_waits_parked_spawned_future_destruction(
        cx: &mut TestAppContext,
    ) {
        struct ReceiptDrop(Arc<AtomicBool>);
        impl Drop for ReceiptDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let (sender, mut receiver) = futures::channel::oneshot::channel();
        let receipt = ReceiptDrop(dropped.clone());
        let original = async move {
            let _receipt = receipt;
            futures::future::pending::<()>().await;
        };
        let future = NativeProfileRetirementFuture {
            future: Some(Box::pin(original)),
            retirement: Some(Arc::new(NativeProfileRetirementSignal {
                sender: Some(sender),
                permit: Some(
                    NativeProfileRetirementPermit::reserve(Arc::new(
                        std::sync::atomic::AtomicUsize::new(0),
                    ))
                    .expect("private retirement permit"),
                ),
            })),
        };
        let task = cx.foreground_executor().spawn(future);
        cx.run_until_parked();
        assert!(!dropped.load(Ordering::Relaxed));
        assert!((&mut receiver).now_or_never().is_none());
        drop(task);
        if !dropped.load(Ordering::Relaxed) {
            assert!(
                (&mut receiver).now_or_never().is_none(),
                "dropping the GPUI Task handle cannot retire still-owned original future fields"
            );
        }
        cx.run_until_parked();
        assert!(dropped.load(Ordering::Relaxed));
        assert!(
            receiver
                .now_or_never()
                .expect("actual parked future destroyed")
                .is_ok()
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn native_selected_retirement_supersession_waits_owned_completion_and_chains_reruns(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::process::ExitStatusExt as _;
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        fixture.workspace.update(cx, |workspace, cx| {
            let revision = cx
                .try_global::<settings::SettingsStore>()
                .and_then(|settings| settings.observed_revision());
            let build_id = TaskId("retiring-native-build".into());
            let phase_id = native_capture_phase_task_id(&build_id);
            let older_handle = StructuredTaskHandle::new(build_id.clone());
            let cancelled = Arc::new(AtomicBool::new(false));
            let cancellation = cancelled.clone();
            assert!(older_handle.bind_terminal(
                task::StructuredTerminalId(1),
                move |_| {
                    cancellation.store(true, Ordering::Relaxed);
                    true
                },
                cx
            ));
            let mut older = workspace
                .begin_native_profile_review(
                    phase_id.clone(),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("older Build phase");
            workspace.attach_native_build_supersession_handle(&older, &older_handle, cx);
            let old_owner = NativeProfileRetirementOwner {
                owned: Some(Arc::new(AtomicBool::new(true))),
                retirement: older.retirement.take().map(|sender| {
                    Arc::new(NativeProfileRetirementSignal {
                        sender: Some(sender),
                        permit: Some(older.retirement_permit.clone()),
                    })
                }),
            };
            let middle_handle = StructuredTaskHandle::new(build_id.clone());
            let middle = workspace
                .begin_native_profile_review(
                    phase_id.clone(),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("middle Build phase");
            workspace.attach_native_build_supersession_handle(&middle, &middle_handle, cx);
            assert!(
                cancelled.load(Ordering::Relaxed),
                "supersession requests actual bound Task cancellation"
            );
            assert!(matches!(
                older_handle.state(),
                task::StructuredTaskState::Cancelled {
                    termination_confirmed: false,
                    ..
                }
            ));
            assert!(
                middle
                    .predecessor_retirement
                    .as_ref()
                    .expect("predecessor receipt")
                    .clone()
                    .now_or_never()
                    .is_none(),
                "Cancelled status is not an opaque-owner retirement receipt"
            );
            let latest_handle = StructuredTaskHandle::new(build_id);
            let latest = workspace
                .begin_native_profile_review(
                    phase_id,
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("latest Build phase");
            workspace.attach_native_build_supersession_handle(&latest, &latest_handle, cx);
            drop(middle);
            assert!(
                latest
                    .predecessor_retirement
                    .as_ref()
                    .expect("transitive retirement")
                    .clone()
                    .now_or_never()
                    .is_none(),
                "rapid third Rerun cannot bypass the first live owner's reservation"
            );
            drop(old_owner);
            assert!(
                latest
                    .predecessor_retirement
                    .as_ref()
                    .expect("transitive retirement")
                    .clone()
                    .now_or_never()
                    .is_some()
            );
            let latest_cancelled = Arc::new(AtomicBool::new(false));
            let cancellation = latest_cancelled.clone();
            assert!(latest_handle.bind_terminal(
                task::StructuredTerminalId(3),
                move |_| {
                    cancellation.store(true, Ordering::Relaxed);
                    true
                },
                cx,
            ));
            assert!(matches!(
                latest_handle.state(),
                task::StructuredTaskState::Running {
                    terminal_id: Some(task::StructuredTerminalId(3)),
                }
            ));
            assert!(latest_handle.mark_completed(ExitStatus::from_raw(0), cx));
            assert!(
                workspace.native_profile_review_is_current(&latest, cx),
                "completed successful Build remains eligible for Capture handoff"
            );
            let retained_current = latest.current.clone();
            let existing_used = workspace
                .native_profile_retirement_slots
                .load(Ordering::Relaxed);
            let fillers: Vec<_> = (existing_used..MAX_NATIVE_PROFILE_RETIREMENT_LEASES)
                .map(|_| {
                    NativeProfileRetirementPermit::reserve(
                        workspace.native_profile_retirement_slots.clone(),
                    )
                    .expect("fill remaining actual phase capacity")
                })
                .collect();
            assert!(
                workspace
                    .begin_native_profile_review(
                        latest.task_id.clone(),
                        uuid::Uuid::new_v4(),
                        Arc::new(AtomicBool::new(true)),
                        None,
                        revision,
                        cx
                    )
                    .is_err()
            );
            assert!(
                retained_current.load(Ordering::Relaxed),
                "capacity refusal must preserve current working generation"
            );
            assert!(workspace.native_profile_review_is_current(&latest, cx));
            drop(fillers);
            assert!(!latest_cancelled.load(Ordering::Relaxed));
            assert!(workspace.consume_native_profile_review(&latest, cx));
            assert!(matches!(
                latest_handle.state(),
                task::StructuredTaskState::Completed { success: true, .. }
            ));
        });
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn native_selected_retirement_preserves_fifo_answers_and_bounds_same_task_prompts(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let launches = Arc::new(Mutex::new(Vec::new()));
        fixture.workspace.update(cx, |workspace, _| {
            workspace.set_terminal_provider(PreparedReceiver(launches.clone()));
        });
        for _ in 0..=MAX_NATIVE_PROFILE_PROMPT_RECEIPTS {
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace.schedule_resolved_task(
                    TaskSourceKind::UserInput,
                    native_capture_guard_fixture(),
                    true,
                    window,
                    cx,
                );
            });
            cx.run_until_parked();
            assert!(launches.lock().is_empty());
        }
        assert_eq!(
            fixture.workspace.read_with(cx, |workspace, _| workspace
                .native_profile_prompt_slots
                .load(Ordering::Relaxed)),
            MAX_NATIVE_PROFILE_PROMPT_RECEIPTS
        );
        for _ in 0..MAX_NATIVE_PROFILE_PROMPT_RECEIPTS {
            assert!(cx.has_pending_prompt());
            cx.simulate_prompt_answer("Capture");
            cx.run_until_parked();
            assert!(
                launches.lock().is_empty(),
                "late superseded FIFO answers cannot launch"
            );
        }
        assert!(!cx.has_pending_prompt());
        assert_eq!(
            fixture.workspace.read_with(cx, |workspace, _| workspace
                .native_profile_prompt_slots
                .load(Ordering::Relaxed)),
            0
        );
        fixture.workspace.update_in(cx, |workspace, window, cx| {
            workspace.schedule_resolved_task(
                TaskSourceKind::UserInput,
                native_capture_guard_fixture(),
                true,
                window,
                cx,
            );
        });
        cx.run_until_parked();
        assert!(launches.lock().is_empty());
        assert!(cx.has_pending_prompt());
        cx.simulate_prompt_answer("Capture");
        cx.run_until_parked();
        assert_eq!(
            launches.lock().len(),
            1,
            "dismissed bounded answers permit fresh approval"
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn native_selected_phase_review_rejects_late_old_build_without_rewriting_completed_state(
        cx: &mut TestAppContext,
    ) {
        use std::os::unix::process::ExitStatusExt as _;
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        fixture.workspace.update(cx, |workspace, cx| {
            let revision = cx
                .try_global::<settings::SettingsStore>()
                .and_then(|settings| settings.observed_revision());
            let build_id = TaskId("selected-build-intent".into());
            let phase_id = native_capture_phase_task_id(&build_id);
            assert_ne!(phase_id, build_id);
            let older_handle = StructuredTaskHandle::new(build_id.clone());
            assert!(older_handle.bind_terminal(task::StructuredTerminalId(11), |_| true, cx));
            assert!(matches!(
                older_handle.state(),
                task::StructuredTaskState::Running {
                    terminal_id: Some(task::StructuredTerminalId(11)),
                }
            ));
            let older = workspace
                .begin_native_profile_review(
                    phase_id.clone(),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("older Build lease");
            observe_native_build_review_lifecycle(&older_handle, older.current.clone(), cx);
            let newer_handle = StructuredTaskHandle::new(build_id.clone());
            assert!(newer_handle.bind_terminal(task::StructuredTerminalId(12), |_| true, cx));
            assert!(matches!(
                newer_handle.state(),
                task::StructuredTaskState::Running {
                    terminal_id: Some(task::StructuredTerminalId(12)),
                }
            ));
            let newer = workspace
                .begin_native_profile_review(
                    phase_id.clone(),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("newer Build lease");
            observe_native_build_review_lifecycle(&newer_handle, newer.current.clone(), cx);
            assert!(!workspace.native_profile_review_is_current(&older, cx));
            assert!(newer_handle.mark_completed(ExitStatus::from_raw(0), cx));
            assert!(
                workspace.native_profile_review_is_current(&newer, cx),
                "real successful Build completion leaves only its transition lease current"
            );
            assert!(workspace.consume_native_profile_review(&newer, cx));
            let capture_handle = StructuredTaskHandle::new(phase_id.clone());
            let capture = workspace
                .begin_native_profile_review(
                    phase_id.clone(),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    Some(capture_handle.clone()),
                    revision,
                    cx,
                )
                .expect("newer separate held Capture review");
            assert!(older_handle.mark_completed(ExitStatus::from_raw(0), cx));
            assert!(
                !workspace.consume_native_profile_review(&older, cx),
                "late old successful Build cannot consume or replace the newer Capture review"
            );
            assert!(workspace.native_profile_review_is_current(&capture, cx));
            assert_eq!(workspace.pending_native_profile_reviews.len(), 1);
            assert!(matches!(
                capture_handle.state(),
                task::StructuredTaskState::Queued
            ));
            assert!(matches!(
                newer_handle.state(),
                task::StructuredTaskState::Completed { success: true, .. }
            ));
            assert!(capture_handle.cancel(cx));
            assert!(!workspace.native_profile_review_is_current(&capture, cx));
            assert!(
                matches!(
                    newer_handle.state(),
                    task::StructuredTaskState::Completed { success: true, .. }
                ),
                "Capture cancellation cannot rewrite the completed Build handle"
            );
            for failed in [false, true] {
                let handle = StructuredTaskHandle::new(build_id.clone());
                let review = workspace
                    .begin_native_profile_review(
                        phase_id.clone(),
                        uuid::Uuid::new_v4(),
                        Arc::new(AtomicBool::new(true)),
                        None,
                        revision,
                        cx,
                    )
                    .expect("fresh Build recovery");
                observe_native_build_review_lifecycle(&handle, review.current.clone(), cx);
                assert!(workspace.native_profile_review_is_current(&review, cx));
                if failed {
                    assert!(handle.bind_terminal(task::StructuredTerminalId(13), |_| true, cx));
                    assert!(matches!(
                        handle.state(),
                        task::StructuredTaskState::Running {
                            terminal_id: Some(task::StructuredTerminalId(13)),
                        }
                    ));
                    assert!(handle.mark_completed(ExitStatus::from_raw(7 << 8), cx));
                } else {
                    assert!(handle.cancel(cx));
                }
                assert!(
                    !workspace.native_profile_review_is_current(&review, cx),
                    "real failure/cancel retires the transition lease without an answer"
                );
            }
            let fresh = workspace
                .begin_native_profile_review(
                    phase_id,
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("fresh current recovery");
            assert!(workspace.native_profile_review_is_current(&fresh, cx));
            assert_eq!(workspace.pending_native_profile_reviews.len(), 1);
        });
    }

    #[gpui::test]
    async fn native_capture_pending_reviews_are_bounded_and_same_task_replacement_retires_only_old_answer(
        cx: &mut TestAppContext,
    ) {
        let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        fixture.workspace.update(cx, |workspace, cx| {
            let revision = cx
                .try_global::<settings::SettingsStore>()
                .and_then(|settings| settings.observed_revision());
            let mut held = Vec::new();
            for index in 0..16 {
                held.push(
                    workspace
                        .begin_native_profile_review(
                            TaskId(format!("pending-{index}")),
                            uuid::Uuid::new_v4(),
                            Arc::new(AtomicBool::new(true)),
                            None,
                            revision,
                            cx,
                        )
                        .expect("Bounded current review"),
                );
            }
            let rejected = workspace.begin_native_profile_review(
                TaskId("seventeenth".into()),
                uuid::Uuid::new_v4(),
                Arc::new(AtomicBool::new(true)),
                None,
                revision,
                cx,
            );
            assert!(rejected.is_err());
            assert_eq!(workspace.pending_native_profile_reviews.len(), 16);
            let replacement = workspace
                .begin_native_profile_review(
                    held[0].task_id.clone(),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("Same Task replacement allowed at cap");
            assert!(!workspace.native_profile_review_is_current(&held[0], cx));
            assert!(workspace.native_profile_review_is_current(&replacement, cx));
            workspace.retire_native_profile_review(&held[0]);
            assert!(
                workspace.native_profile_review_is_current(&replacement, cx),
                "Old cleanup cannot remove replacement"
            );
            assert!(workspace.consume_native_profile_review(&replacement, cx));
            assert_eq!(workspace.pending_native_profile_reviews.len(), 15);
            let recovered = workspace
                .begin_native_profile_review(
                    TaskId("recovered".into()),
                    uuid::Uuid::new_v4(),
                    Arc::new(AtomicBool::new(true)),
                    None,
                    revision,
                    cx,
                )
                .expect("Consumed review frees its slot");
            assert!(workspace.native_profile_review_is_current(&recovered, cx));
            assert!(
                held.iter()
                    .skip(1)
                    .all(|review| workspace.native_profile_review_is_current(review, cx))
            );
        });
    }

    #[test]
    fn legacy_generated_task_policy_refuses_unadmitted_history_and_preserves_authored_tasks() {
        let legacy: task::TaskTemplate =
            serde_json::from_str(r#"{"label":"old task","command":"cargo"}"#)
                .expect("legacy history");
        for name in ["Cargo", "Rust Tests"] {
            let source = TaskSourceKind::Language { name: name.into() };
            assert!(ensure_generated_task_policy(&source, &legacy).is_err());
            let current = task::TaskTemplate {
                experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
                ..legacy.clone()
            };
            assert!(ensure_generated_task_policy(&source, &current).is_ok());
            let restored: task::TaskTemplate =
                serde_json::from_str(&serde_json::to_string(&current).expect("current history"))
                    .expect("restore current history");
            assert!(ensure_generated_task_policy(&source, &restored).is_ok());
        }
        assert!(ensure_generated_task_policy(&TaskSourceKind::UserInput, &legacy).is_ok());
        assert!(
            ensure_generated_task_policy(
                &TaskSourceKind::Language {
                    name: "ordinary-language".into()
                },
                &legacy
            )
            .is_ok()
        );
        assert!(
            serde_json::to_value(&legacy)
                .expect("unchanged ordinary history")
                .get("experimental_policy_version")
                .is_none()
        );
    }

    #[test]
    fn experimental_capture_subscription_classifies_only_owned_workflows() {
        let ordinary_report = task::TaskStdoutReport {
            format: task::TaskStdoutReportFormat::CargoBuildJson,
            max_bytes: 1024,
            native_profile: None,
        };
        let mut group = TaskArtifactGroup {
            format: "ordinary-report-v1".into(),
            files: Default::default(),
        };
        assert_eq!(
            experimental_capture_feature(Some(&ordinary_report), Some(&group), &[]),
            None
        );
        group.format = "llvm-cov-json-v1".into();
        assert_eq!(
            experimental_capture_feature(Some(&ordinary_report), Some(&group), &[]),
            Some("rust-coverage-experimental")
        );
        group.format = "samply-0.13.1".into();
        group.files = [
            (
                "profile".into(),
                task::TaskArtifact {
                    path: "profile.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 32 * 1024 * 1024,
                },
            ),
            (
                "symbols".into(),
                task::TaskArtifact {
                    path: "profile.syms.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 16 * 1024 * 1024,
                },
            ),
        ]
        .into();
        assert_eq!(
            experimental_capture_feature(None, Some(&group), &[]),
            Some("rust-native-profiling-experimental")
        );
        assert_eq!(
            experimental_capture_feature(
                None,
                None,
                &["rust-named-cargo-presets-experimental".into()]
            ),
            None
        );
        assert_eq!(
            experimental_capture_feature(None, None, &["rust-coverage-experimental".into()]),
            Some("rust-coverage-experimental")
        );
        let checked = task::TaskStdoutReport {
            native_profile: Some(task::TaskNativeProfileContinuation {
                collector_path: "/collector".into(),
                profile_path: "profile.json".into(),
                workload_arguments: Vec::new(),
            }),
            ..ordinary_report
        };
        assert_eq!(
            experimental_capture_feature(Some(&checked), None, &[]),
            Some("rust-native-profiling-experimental")
        );
    }

    #[gpui::test]
    async fn ordinary_artifact_task_survives_flag_updates_and_experimental_task_retires(
        cx: &mut TestAppContext,
    ) {
        struct HeldOrdinary {
            started: Arc<AtomicBool>,
            release: Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
        }
        impl TerminalProvider for HeldOrdinary {
            fn spawn_prepared(
                &self,
                task: SpawnInTerminal,
                handle: Option<StructuredTaskHandle>,
                prepare: crate::TaskLaunchPreparation,
                window: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let held = HeldOrdinary {
                    started: self.started.clone(),
                    release: Mutex::new(self.release.lock().take()),
                };
                window.spawn(cx, async move |cx| {
                    let prepared = match prepare(task, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) => return Some(Err(error)),
                    };
                    let result = match cx.update(|window, cx| {
                        if let Some(pending_launch) = prepared.pending_launch.as_ref() {
                            pending_launch.check(&prepared.launch, cx)?;
                        }
                        drop(prepared.pending_launch);
                        Ok::<_, anyhow::Error>(match handle {
                            Some(handle) => {
                                held.spawn_structured(prepared.launch, handle, window, cx)
                            }
                            None => held.spawn(prepared.launch, window, cx),
                        })
                    }) {
                        Ok(Ok(result)) => result,
                        Ok(Err(error)) | Err(error) => return Some(Err(error)),
                    };
                    match prepared.completion {
                        Some(completion) => completion(result, cx).await,
                        None => result.await,
                    }
                })
            }

            fn spawn_structured(
                &self,
                task: SpawnInTerminal,
                handle: StructuredTaskHandle,
                window: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                if handle.state().is_terminal() {
                    return Task::ready(None);
                }
                handle.mark_running(None, cx);
                let (cancel, cancelled) = futures::channel::oneshot::channel();
                let cancel = Arc::new(Mutex::new(Some(cancel)));
                handle.subscribe(cx, move |event, _| {
                    if matches!(event.state, task::StructuredTaskState::Cancelled { .. })
                        && let Some(cancel) = cancel.lock().take()
                        && cancel.send(()).is_err()
                    {
                        log::debug!("Held provider already retired before cancellation");
                    }
                });
                let held = self.spawn(task, window, cx);
                cx.spawn(
                    async move |_| match futures::future::select(held, cancelled).await {
                        futures::future::Either::Left((result, _)) => result,
                        futures::future::Either::Right((cancelled, held)) => {
                            cancelled.expect("original structured cancellation owner");
                            drop(held);
                            None
                        }
                    },
                )
            }

            fn spawn(
                &self,
                _: SpawnInTerminal,
                _: &mut Window,
                cx: &mut App,
            ) -> Task<Option<Result<ExitStatus>>> {
                let released = self
                    .release
                    .lock()
                    .take()
                    .expect("one owned ordinary launch");
                self.started.store(true, Ordering::Relaxed);
                cx.spawn(async move |_| {
                    released.await.expect("release original held task");
                    Some(Err(anyhow::anyhow!(
                        "Ordinary provider fixture failed after release"
                    )))
                })
            }
        }
        for experimental in [false, true] {
            let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-coverage-experimental",
                    true,
                    cx,
                )
            });
            let (release, released) = futures::channel::oneshot::channel();
            let started = Arc::new(AtomicBool::new(false));
            fixture.workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(HeldOrdinary {
                    started: started.clone(),
                    release: Mutex::new(Some(released)),
                })
            });
            let template = task::TaskTemplate {
                label: "held ordinary artifact".into(),
                command: "echo".into(),
                args: vec!["ordinary".into()],
                save: SaveStrategy::None,
                required_experimental_features: if experimental {
                    vec!["rust-coverage-experimental".into()]
                } else {
                    Vec::new()
                },
                artifact_group: Some(TaskArtifactGroup {
                    format: "ordinary-report-v1".into(),
                    files: std::collections::BTreeMap::from_iter([(
                        "report".into(),
                        task::TaskArtifact {
                            path: "ordinary.json".into(),
                            kind: task::TaskArtifactKind::Data,
                            max_bytes: 1024,
                        },
                    )]),
                }),
                ..Default::default()
            };
            let resolved = template
                .resolve_task("flag-lifecycle", &TaskContext::default())
                .expect("ordinary artifact intent");
            let outcomes = Arc::new(Mutex::new(Vec::new()));
            let handle = fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace.schedule_resolved_task_with_structured_handle_and_completion(
                    TaskSourceKind::UserInput,
                    resolved,
                    true,
                    {
                        let outcomes = outcomes.clone();
                        move |outcome, _| outcomes.lock().push(outcome)
                    },
                    window,
                    cx,
                )
            });
            cx.run_until_parked();
            assert!(
                started.load(Ordering::Relaxed),
                "existing ordinary provider owns the held launch"
            );
            assert!(outcomes.lock().is_empty());
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-coverage-experimental",
                    false,
                    cx,
                )
            });
            cx.run_until_parked();
            if experimental {
                assert!(matches!(
                    handle.state(),
                    task::StructuredTaskState::Cancelled { .. }
                ));
                assert_eq!(
                    outcomes.lock().as_slice(),
                    &[ScheduledTaskResult::Cancelled]
                );
                assert!(
                    release.send(()).is_err(),
                    "revocation drops original held future"
                );
            } else {
                assert!(!matches!(
                    handle.state(),
                    task::StructuredTaskState::Cancelled { .. }
                ));
                assert!(
                    outcomes.lock().is_empty(),
                    "ordinary artifact task stays held across flag update"
                );
                release
                    .send(())
                    .expect("ordinary task retained its original future");
                cx.run_until_parked();
                assert_eq!(
                    outcomes.lock().as_slice(),
                    &[ScheduledTaskResult::SpawnFailed],
                    "no synthetic success is supplied"
                );
            }
        }
    }

    struct PreparedReceiver(Arc<Mutex<Vec<SpawnInTerminal>>>);
    impl TerminalProvider for PreparedReceiver {
        fn spawn(
            &self,
            _: SpawnInTerminal,
            _: &mut Window,
            _: &mut App,
        ) -> Task<Option<Result<ExitStatus>>> {
            panic!("Native capture must use the eligibility preparation route");
        }
        fn spawn_prepared(
            &self,
            launch: SpawnInTerminal,
            handle: Option<StructuredTaskHandle>,
            prepare: crate::TaskLaunchPreparation,
            window: &mut Window,
            cx: &mut App,
        ) -> Task<Option<Result<ExitStatus>>> {
            let launches = self.0.clone();
            window.spawn(cx, async move |cx| {
                let prepared = match prepare(launch, cx).await {
                    Ok(prepared) => prepared,
                    Err(error) if error.is::<crate::TaskLaunchCancelled>() => return None,
                    Err(error) => return Some(Err(error)),
                };
                assert!(
                    !handle
                        .as_ref()
                        .is_some_and(|handle| handle.state().is_terminal())
                );
                launches.lock().push(prepared.launch);
                Some(Err(anyhow!(
                    "Controlled receiver stops before any child spawn"
                )))
            })
        }
    }
    fn native_capture_guard_fixture() -> ResolvedTask {
        let template = TaskTemplate {
            label: "Held native capture".into(),
            command: "fixture-profiler".into(),
            cwd: Some("/root".into()),
            save: SaveStrategy::None,
            allow_concurrent_runs: true,
            args: vec!["capture.json".into()],
            literal_arguments: true,
            artifact_group: Some(TaskArtifactGroup {
                format: "samply-0.13.1".into(),
                files: BTreeMap::from_iter([
                    (
                        "profile".into(),
                        TaskArtifact {
                            path: "capture.json".into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 1024,
                        },
                    ),
                    (
                        "symbols".into(),
                        TaskArtifact {
                            path: "capture.syms.json".into(),
                            kind: TaskArtifactKind::Data,
                            max_bytes: 1024,
                        },
                    ),
                ]),
            }),
            artifact_group_argument_binding: Some(task::TaskArtifactGroupArgumentBinding {
                argument_index: 0,
                role: "profile".into(),
                related: vec![task::TaskArtifactRelatedArgumentBinding {
                    role: "symbols".into(),
                    extension: "syms.json".into(),
                    argument_index: None,
                }],
            }),
            ..Default::default()
        };
        template
            .resolve_task("held-native", &TaskContext::default())
            .expect("Valid native Task")
    }

    #[gpui::test]
    async fn native_capture_scheduler_held_consent_rechecks_trust_settings_worktree_and_session(
        cx: &mut TestAppContext,
    ) {
        for scenario in [
            "trust",
            "settings",
            "removed",
            "session-event",
            "experimental-opt-in",
        ] {
            cx.update(|cx| {
                if cx.has_global::<project::trusted_worktrees::TrustedWorktrees>() {
                    cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>();
                }
            });
            let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-native-profiling-experimental",
                    true,
                    cx,
                )
            });
            cx.update(|_, cx| {
                project::trusted_worktrees::init(collections::HashMap::default(), cx)
            });
            let project = fixture
                .workspace
                .read_with(cx, |workspace, _| workspace.project.clone());
            let (store, worktree_id) = project.read_with(cx, |project, cx| {
                let store = project.worktree_store();
                let id = store
                    .read(cx)
                    .worktrees()
                    .next()
                    .expect("Owned fixture worktree")
                    .read(cx)
                    .id();
                (store, id)
            });
            let trust = cx.read(|cx| {
                project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                    .expect("Trust owner")
            });
            trust.update(cx, |trust, cx| {
                trust.trust(
                    &store,
                    HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                        worktree_id,
                    )]),
                    cx,
                )
            });
            let launches = Arc::new(Mutex::new(Vec::new()));
            fixture.workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(PreparedReceiver(launches.clone()))
            });
            let resolved = native_capture_guard_fixture();
            let outcomes = Arc::new(Mutex::new(Vec::new()));
            let schedule = |cx: &mut gpui::VisualTestContext| {
                fixture.workspace.update_in(cx, |workspace, window, cx| {
                    let outcomes = outcomes.clone();
                    workspace.schedule_resolved_task_with_completion(
                        TaskSourceKind::UserInput,
                        resolved.clone(),
                        true,
                        move |result, _| outcomes.lock().push(result),
                        window,
                        cx,
                    );
                });
                cx.run_until_parked();
            };
            schedule(cx);
            assert!(cx.has_pending_prompt(), "{scenario}");
            assert!(launches.lock().is_empty());
            let current = fixture.workspace.read_with(cx, |workspace, _| {
                workspace
                    .pending_native_profile_reviews
                    .get(&resolved.id)
                    .expect("Own pending review")
                    .current
                    .clone()
            });
            assert!(current.load(Ordering::Relaxed));
            match scenario {
                "trust" => trust.update(cx, |trust, cx| {
                    trust.restrict(
                        store.downgrade(),
                        HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                            worktree_id,
                        )]),
                        cx,
                    )
                }),
                "settings" => cx.update(|_, cx| {
                    cx.update_global(|settings: &mut settings::SettingsStore, cx| {
                        settings
                            .set_server_settings("{}", cx)
                            .expect("Advance actual settings owner")
                    })
                }),
                "experimental-opt-in" => cx.update(|_, cx| {
                    project::task_store::set_experimental_feature_for_test(
                        "rust-native-profiling-experimental",
                        false,
                        cx,
                    )
                }),
                "removed" => {
                    project.update(cx, |project, cx| project.remove_worktree(worktree_id, cx))
                }
                "session-event" => {
                    project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost))
                }
                _ => panic!("Unknown guard scenario"),
            }
            cx.run_until_parked();
            cx.simulate_prompt_answer("Capture");
            cx.run_until_parked();
            assert!(
                launches.lock().is_empty(),
                "Late {scenario} approval cannot launch"
            );
            assert!(
                !current.load(Ordering::Relaxed),
                "The expired review must retire: {scenario}"
            );
            let blocked = match scenario {
                "trust" | "removed" => ScheduledTaskResult::SpawnFailed,
                "settings" | "session-event" | "experimental-opt-in" => {
                    ScheduledTaskResult::Cancelled
                }
                _ => panic!("Unknown guard scenario"),
            };
            assert_eq!(outcomes.lock().as_slice(), &[blocked]);
            assert!(!cx.has_pending_prompt());
            if matches!(scenario, "trust" | "settings" | "experimental-opt-in") {
                if scenario == "experimental-opt-in" {
                    cx.update(|_, cx| {
                        project::task_store::set_experimental_feature_for_test(
                            "rust-native-profiling-experimental",
                            true,
                            cx,
                        )
                    });
                }
                if scenario == "trust" {
                    trust.update(cx, |trust, cx| {
                        trust.trust(
                            &store,
                            HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                                worktree_id,
                            )]),
                            cx,
                        )
                    });
                }
                schedule(cx);
                assert!(cx.has_pending_prompt(), "Fresh {scenario} review");
                assert!(launches.lock().is_empty());
                cx.simulate_prompt_answer("Capture");
                cx.run_until_parked();
                assert_eq!(
                    launches.lock().len(),
                    1,
                    "Fresh {scenario} recovers independently"
                );
                assert_eq!(
                    outcomes.lock().as_slice(),
                    &[blocked, ScheduledTaskResult::SpawnFailed]
                );
            }
        }
    }

    #[gpui::test]
    async fn native_capture_expired_unanswered_reviews_free_capacity_without_reusing_late_approval(
        cx: &mut TestAppContext,
    ) {
        for scenario in ["cancel", "settings", "session-event"] {
            let (fixture, cx) = create_fixture(cx, SaveStrategy::None).await;
            cx.update(|_, cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-native-profiling-experimental",
                    true,
                    cx,
                )
            });
            let launches = Arc::new(Mutex::new(Vec::new()));
            fixture.workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(PreparedReceiver(launches.clone()))
            });
            let mut handles = Vec::new();
            for index in 0..16 {
                let mut original = native_capture_guard_fixture().original_task().clone();
                original.label = format!("Held {scenario} {index}");
                let task = original
                    .resolve_task(&format!("{scenario}-{index}"), &TaskContext::default())
                    .expect("Distinct bounded Task");
                handles.push(fixture.workspace.update_in(cx, |workspace, window, cx| {
                    workspace.schedule_resolved_task_with_structured_handle(
                        TaskSourceKind::UserInput,
                        task,
                        true,
                        window,
                        cx,
                    )
                }));
            }
            cx.run_until_parked();
            let prior = fixture.workspace.read_with(cx, |workspace, _| {
                assert_eq!(workspace.pending_native_profile_reviews.len(), 16);
                workspace
                    .pending_native_profile_reviews
                    .values()
                    .map(|review| review.current.clone())
                    .collect::<Vec<_>>()
            });
            assert!(prior.iter().all(|flag| flag.load(Ordering::Relaxed)));
            assert!(launches.lock().is_empty());
            assert!(cx.has_pending_prompt());
            match scenario {
                "cancel" => cx.update(|_, cx| {
                    for handle in &handles {
                        assert!(handle.cancel(cx));
                    }
                }),
                "settings" => cx.update(|_, cx| {
                    cx.update_global(|settings: &mut settings::SettingsStore, cx| {
                        settings
                            .set_server_settings("{}", cx)
                            .expect("Advance actual settings owner")
                    })
                }),
                "session-event" => {
                    let project = fixture
                        .workspace
                        .read_with(cx, |workspace, _| workspace.project.clone());
                    project.update(cx, |_, cx| cx.emit(project::Event::DisconnectedFromHost));
                }
                _ => panic!("Unknown expiry"),
            }
            cx.run_until_parked();
            if scenario == "cancel" {
                assert!(
                    prior.iter().all(|flag| !flag.load(Ordering::Relaxed)),
                    "Lifecycle retires cancelled review without an answer"
                );
            }
            let mut original = native_capture_guard_fixture().original_task().clone();
            original.label = format!("Fresh {scenario}");
            let fresh = original
                .resolve_task(&format!("fresh-{scenario}"), &TaskContext::default())
                .expect("Fresh Task");
            let fresh_id = fresh.id.clone();
            fixture.workspace.update_in(cx, |workspace, window, cx| {
                workspace.schedule_resolved_task(TaskSourceKind::UserInput, fresh, true, window, cx)
            });
            cx.run_until_parked();
            fixture.workspace.read_with(cx, |workspace, _| {
                assert_eq!(
                    workspace.pending_native_profile_reviews.len(),
                    1,
                    "Expired unanswered entries cannot consume capacity: {scenario}"
                );
                assert!(
                    workspace
                        .pending_native_profile_reviews
                        .get(&fresh_id)
                        .expect("Fresh review")
                        .current
                        .load(Ordering::Relaxed)
                );
            });
            assert!(launches.lock().is_empty());
            for _ in 0..16 {
                cx.simulate_prompt_answer("Capture");
                cx.run_until_parked();
                assert!(
                    launches.lock().is_empty(),
                    "Late {scenario} answers never launch"
                );
            }
            assert!(prior.iter().all(|flag| !flag.load(Ordering::Relaxed)));
            assert!(cx.has_pending_prompt());
            cx.simulate_prompt_answer("Capture");
            cx.run_until_parked();
            assert_eq!(launches.lock().len(), 1);
            assert_eq!(launches.lock()[0].id, fresh_id);
            assert!(!cx.has_pending_prompt());
        }
    }

    struct TestTerminalProvider {
        item: Entity<TestItem>,
        dirty_before_spawn: Arc<Mutex<Option<bool>>>,
    }

    impl TerminalProvider for TestTerminalProvider {
        fn spawn(
            &self,
            _task: task::SpawnInTerminal,
            _window: &mut ui::Window,
            cx: &mut App,
        ) -> Task<Option<Result<ExitStatus>>> {
            *self.dirty_before_spawn.lock() = Some(cx.read_entity(&self.item, |e, _| e.is_dirty));
            Task::ready(Some(Ok(ExitStatus::default())))
        }
    }
}
