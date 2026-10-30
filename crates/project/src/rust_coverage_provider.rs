use feature_flags::{FeatureFlagAppExt as _, RustCoverageExperimentalFeatureFlag};
#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    path::Path,
    rc::Rc,
    sync::{Arc, atomic::AtomicBool},
};

use anyhow::{Context as _, Result, bail};
use futures::StreamExt as _;
use gpui::{App, AppContext as _, AsyncApp, Context, Entity, Task};
use rpc::{AnyProtoClient, ErrorCodeExt as _, ErrorExt as _, TypedEnvelope, proto};
use serde_json::Value;

use crate::{
    ProjectItem as _, ProjectPath,
    source_coverage::{
        MAX_SOURCE_COVERAGE_FILES, MAX_SOURCE_COVERAGE_RANGES, MAX_SOURCE_COVERAGE_RANGES_PER_FILE,
        SourceCoverageFile, SourceCoveragePoint, SourceCoverageProviderId, SourceCoverageRange,
        SourceCoverageRequestVersion, SourceCoverageSnapshot, SourceCoverageStatus,
        SourceCoverageStore, SourceCoverageUnsupportedHost, snapshot_from_proto, snapshot_to_proto,
    },
    trusted_worktrees::TrustedWorktrees,
    worktree_store::WorktreeStore,
};

pub const RUST_COVERAGE_PROVIDER_ID: &str = "cargo-llvm-cov";
pub const MAX_RUST_COVERAGE_ARTIFACT_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const COVERAGE_TASK_EXPIRED_DIAGNOSTIC: &str =
    "Coverage Task expired because the project session or trust changed";
const MAX_RUST_COVERAGE_EXPORTS: usize = 16;
#[cfg(any(test, feature = "test-support"))]
static LOCAL_RUST_COVERAGE_INTERPRET_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(any(test, feature = "test-support"))]
pub fn local_rust_coverage_interpret_calls_for_test() -> u64 {
    LOCAL_RUST_COVERAGE_INTERPRET_CALLS.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustCoverageRunToken {
    project_generation: u64,
    generation: u64,
    request_version: Option<SourceCoverageRequestVersion>,
}

pub struct RustCoverageArtifactProvider {
    project_generation: u64,
    generation: u64,
}

impl RustCoverageArtifactProvider {
    pub fn new(project_generation: u64) -> Self {
        Self {
            project_generation,
            generation: 0,
        }
    }

    pub fn begin_run(&mut self) -> RustCoverageRunToken {
        self.generation = self.generation.wrapping_add(1);
        RustCoverageRunToken {
            project_generation: self.project_generation,
            generation: self.generation,
            request_version: None,
        }
    }

    pub fn cancel(&mut self, token: RustCoverageRunToken) {
        if self.is_run_current(token) {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    pub fn is_run_current(&self, token: RustCoverageRunToken) -> bool {
        token.project_generation == self.project_generation && token.generation == self.generation
    }

    pub fn parse_artifact(
        &self,
        token: RustCoverageRunToken,
        contents: &[u8],
        resolve_path: impl FnMut(&Path) -> Option<ProjectPath>,
    ) -> Result<SourceCoverageSnapshot> {
        if !self.is_run_current(token) {
            bail!("Rust coverage artifact belongs to a stale or cancelled run");
        }
        parse_coverage_artifact(
            self.project_generation,
            token.generation,
            contents,
            resolve_path,
        )
    }
}

// Parsing returns report facts; publication or producing-Task authority is checked by its caller.
fn parse_coverage_artifact(
    project_generation: u64,
    generation: u64,
    contents: &[u8],
    mut resolve_path: impl FnMut(&Path) -> Option<ProjectPath>,
) -> Result<SourceCoverageSnapshot> {
    if contents.len() > MAX_RUST_COVERAGE_ARTIFACT_BYTES {
        bail!("Rust coverage artifact exceeds the supported byte limit");
    }
    let root: Value = serde_json::from_slice(contents).context("malformed coverage JSON")?;
    if root.get("type").and_then(Value::as_str) != Some("llvm.coverage.json.export") {
        bail!("unsupported Rust coverage artifact type");
    }
    let version = root
        .get("version")
        .and_then(Value::as_str)
        .context("coverage artifact is missing its schema version")?;
    if !matches!(
        version
            .split('.')
            .next()
            .and_then(|major| major.parse::<u32>().ok()),
        Some(2 | 3)
    ) {
        bail!("unsupported LLVM coverage export schema version {version}");
    }
    let collector_version = root
        .get("cargo_llvm_cov")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("version"))
        .and_then(Value::as_str)
        .filter(|version| !version.is_empty() && version.len() <= 64)
        .context("coverage artifact was not produced by a supported cargo-llvm-cov")?;
    let _ = collector_version;

    let data = root
        .get("data")
        .and_then(Value::as_array)
        .context("coverage artifact is missing data")?;
    let mut files = Vec::new();
    let mut truncated = data.len() > MAX_RUST_COVERAGE_EXPORTS;
    let mut partial_count = 0usize;
    let mut remaining_ranges = MAX_SOURCE_COVERAGE_RANGES;
    for export in data.iter().take(MAX_RUST_COVERAGE_EXPORTS) {
        let Some(export_files) = export.get("files").and_then(Value::as_array) else {
            partial_count += 1;
            continue;
        };
        for file in export_files {
            if files.len() >= MAX_SOURCE_COVERAGE_FILES || remaining_ranges == 0 {
                truncated = true;
                break;
            }
            let Some(filename) = file.get("filename").and_then(Value::as_str) else {
                partial_count += 1;
                continue;
            };
            let Some(path) = resolve_path(Path::new(filename)) else {
                partial_count += 1;
                continue;
            };
            let segments = file.get("segments").and_then(Value::as_array);
            if segments.is_none() {
                partial_count += 1;
            }
            let segments = segments.map(Vec::as_slice).unwrap_or_default();
            let range_limit = remaining_ranges.min(MAX_SOURCE_COVERAGE_RANGES_PER_FILE);
            let mut ranges = Vec::new();
            let mut file_truncated = false;
            for (index, segment) in segments.iter().enumerate() {
                if ranges.len() >= range_limit {
                    file_truncated = true;
                    truncated = true;
                    break;
                }
                let Some(segment) = segment.as_array() else {
                    partial_count += 1;
                    continue;
                };
                let Some(line) = segment.first().and_then(Value::as_u64) else {
                    partial_count += 1;
                    continue;
                };
                let Some(column) = segment.get(1).and_then(Value::as_u64) else {
                    partial_count += 1;
                    continue;
                };
                let Some(hit_count) = segment.get(2).and_then(Value::as_u64) else {
                    partial_count += 1;
                    continue;
                };
                if segment.get(3).and_then(Value::as_bool) != Some(true) {
                    continue;
                }
                let Ok(line) = u32::try_from(line.saturating_sub(1)) else {
                    partial_count += 1;
                    continue;
                };
                let Ok(column) = u32::try_from(column.saturating_sub(1)) else {
                    partial_count += 1;
                    continue;
                };
                let next = segments.get(index + 1).and_then(Value::as_array);
                let end_line = next
                    .and_then(|next| next.first())
                    .and_then(Value::as_u64)
                    .and_then(|line| u32::try_from(line.saturating_sub(1)).ok())
                    .unwrap_or(line);
                let end_column = next
                    .and_then(|next| next.get(1))
                    .and_then(Value::as_u64)
                    .and_then(|column| u32::try_from(column.saturating_sub(1)).ok())
                    .unwrap_or_else(|| column.saturating_add(1));
                ranges.push(SourceCoverageRange {
                    start: SourceCoveragePoint { line, column },
                    end: SourceCoveragePoint {
                        line: end_line,
                        column: end_column,
                    },
                    hit_count,
                });
            }
            remaining_ranges = remaining_ranges.saturating_sub(ranges.len());
            let lines = file.get("summary").and_then(|summary| summary.get("lines"));
            let line_counts = lines.and_then(|lines| {
                let total = u32::try_from(lines.get("count")?.as_u64()?).ok()?;
                let covered = u32::try_from(lines.get("covered")?.as_u64()?).ok()?;
                Some((total, covered))
            });
            let (total, covered) = match line_counts {
                Some((total, covered)) if covered <= total => (total, covered),
                Some((total, _)) => {
                    partial_count += 1;
                    (total, total)
                }
                None => {
                    partial_count += 1;
                    (0, 0)
                }
            };
            files.push(SourceCoverageFile {
                path,
                ranges,
                covered_lines: covered,
                uncovered_lines: total.saturating_sub(covered),
                truncated: file_truncated,
            });
        }
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(SourceCoverageSnapshot {
        project_generation: project_generation,
        provider_id: SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.to_string()),
        generation: generation,
        status: if partial_count > 0 || truncated {
            SourceCoverageStatus::Partial
        } else if files.is_empty() {
            SourceCoverageStatus::Empty
        } else {
            SourceCoverageStatus::Current
        },
        files,
        truncated,
        diagnostic: (partial_count > 0).then(|| {
            format!("Coverage report contains {partial_count} malformed or non-visible issue(s)")
        }),
    })
}

pub(crate) struct RustCoverageTaskCapture {
    pub(crate) provider: Entity<RustCoverageProviderStore>,
    pub(crate) token: Rc<Cell<RustCoverageRunToken>>,
    pub(crate) run_id: String,
    pub(crate) task_id: task::TaskId,
    pub(crate) artifact: task::TaskArtifact,
    pub(crate) path: Option<ProjectPath>,
    pub(crate) admitted_output: Rc<RefCell<Option<ProjectPath>>>,
    pub(crate) command: Option<String>,
    pub(crate) arguments: Vec<String>,
    pub(crate) environment: collections::HashMap<String, String>,
    pub(crate) working_directory: Option<std::path::PathBuf>,
    pub(crate) admitted: bool,
    pub(crate) source_baseline: Option<CoverageMappedSourceBaselineState>,
    pub(crate) cancellation: Arc<AtomicBool>,
    pub(crate) stop: async_channel::Sender<()>,
    pub(crate) _subscriptions: Vec<gpui::Subscription>,
    pub(crate) armed: bool,
}

impl RustCoverageTaskCapture {
    pub(crate) fn is_current(&self, cx: &App) -> bool {
        !self.cancellation.load(Ordering::Relaxed)
            && self.provider.read(cx).is_run_current(self.token.get(), cx)
    }
}

impl Drop for RustCoverageTaskCapture {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.store(true, Ordering::Relaxed);
        }
        self.stop.close();
    }
}

pub(crate) fn validate_capture_task_environment(
    environment: &collections::HashMap<String, String>,
) -> Result<()> {
    anyhow::ensure!(
        environment.len() <= 1024,
        "Coverage Task environment exceeds its entry bound"
    );
    let mut bytes = 0usize;
    for (key, value) in environment {
        anyhow::ensure!(
            !key.is_empty()
                && key.len() <= 4096
                && !key.contains(['=', '\0'])
                && value.len() <= 65536
                && !value.contains('\0'),
            "Coverage Task environment has an unsupported entry"
        );
        bytes = bytes
            .checked_add(key.len())
            .and_then(|bytes| bytes.checked_add(value.len()))
            .context("Coverage Task environment size overflow")?;
        anyhow::ensure!(
            bytes <= 256 * 1024,
            "Coverage Task environment exceeds its byte bound"
        );
    }
    Ok(())
}

pub(crate) enum CoverageMappedSourceBaselineState {
    Pending(Task<Result<Arc<CoverageMappedSourceBaseline>>>),
    Ready(Arc<CoverageMappedSourceBaseline>),
}

struct CoverageMappedBuffer {
    path: ProjectPath,
    buffer: gpui::WeakEntity<language::Buffer>,
    version: clock::Global,
    dirty: bool,
}

pub(crate) struct CoverageMappedSourceBaseline {
    buffer_store: gpui::WeakEntity<crate::buffer_store::BufferStore>,
    buffers: Vec<CoverageMappedBuffer>,
    roots: Vec<worktree::Snapshot>,
    output: ProjectPath,
}

impl CoverageMappedSourceBaseline {
    fn validate_buffers(&self, snapshot: &SourceCoverageSnapshot, cx: &App) -> Result<()> {
        let changed = || anyhow::Error::from(crate::source_coverage::SourceCoverageSourceChanged);
        let buffer_store = self.buffer_store.upgrade().ok_or_else(changed)?;
        for admitted in &self.buffers {
            if !snapshot.files.iter().any(|file| file.path == admitted.path) {
                continue;
            }
            if admitted.dirty {
                return Err(changed());
            }
            let Some(buffer) = admitted.buffer.upgrade() else {
                continue;
            };
            if buffer.read(cx).is_dirty()
                || buffer.read(cx).version() != admitted.version
                || !buffer_store
                    .read(cx)
                    .buffers()
                    .any(|current| current == buffer)
                || buffer.read(cx).project_path(cx).as_ref() != Some(&admitted.path)
            {
                return Err(changed());
            }
        }
        for buffer in buffer_store.read(cx).buffers() {
            if buffer
                .read(cx)
                .project_path(cx)
                .is_some_and(|path| snapshot.files.iter().any(|file| file.path == path))
                && buffer.read(cx).is_dirty()
            {
                return Err(changed());
            }
        }
        Ok(())
    }

    async fn validate_reported_disk_sources(
        &self,
        reported_paths: &[ProjectPath],
        filesystem: &dyn fs::Fs,
        cancellation: &AtomicBool,
    ) -> Result<()> {
        let changed = || anyhow::Error::from(crate::source_coverage::SourceCoverageSourceChanged);
        anyhow::ensure!(
            reported_paths.len() <= MAX_SOURCE_COVERAGE_FILES,
            "Coverage mapped sources exceed the report file bound"
        );
        for path in reported_paths {
            if cancellation.load(Ordering::Relaxed) {
                return Err(changed());
            }
            let root = self
                .roots
                .iter()
                .find(|root| root.id() == path.worktree_id)
                .ok_or_else(changed)?;
            let root_entry = root.root_entry().ok_or_else(changed)?;
            let entry = root.entry_for_path(&path.path).ok_or_else(changed)?;
            if *path == self.output
                || !root_entry.is_dir()
                || !entry.is_file()
                || entry.is_ignored
                || entry.is_private
                || entry.is_external
                || entry.is_fifo
            {
                return Err(changed());
            }
            let absolute_path = root.abs_path().join(path.path.as_std_path());
            let before = filesystem
                .metadata(&absolute_path)
                .await
                .map_err(|_| changed())?
                .ok_or_else(changed)?;
            if before.inode != entry.inode
                || Some(before.mtime) != entry.mtime
                || before.len != entry.size
                || before.is_symlink
                || before.is_dir
                || before.is_fifo
            {
                return Err(changed());
            }
            // The report does not establish which bytes the compiler consumed.
            // Confined opening authenticates the retained root and file identity,
            // without scanning or retaining unrelated workspace source bodies.
            let mut reader = filesystem
                .open_sync_confined_file(
                    root.abs_path().as_ref(),
                    path.path.as_std_path(),
                    root_entry.inode,
                    entry.inode,
                )
                .await
                .map_err(|_| changed())?;
            let mut accessibility = [0u8; 1];
            std::io::Read::read(reader.as_mut(), &mut accessibility).map_err(|_| changed())?;
            drop(reader);
            let after = filesystem
                .metadata(&absolute_path)
                .await
                .map_err(|_| changed())?
                .ok_or_else(changed)?;
            if cancellation.load(Ordering::Relaxed)
                || after.inode != before.inode
                || after.mtime != before.mtime
                || after.len != before.len
                || after.is_symlink != before.is_symlink
                || after.is_dir != before.is_dir
                || after.is_fifo != before.is_fifo
            {
                return Err(changed());
            }
        }
        Ok(())
    }
}

struct ParsedLocalCoverageArtifact {
    snapshot: SourceCoverageSnapshot,
    artifact_path: ProjectPath,
    expected_entry: worktree::Entry,
    worktree: Entity<worktree::Worktree>,
    sealed: Option<SealedCoverageArtifact>,
}

struct SealedCoverageArtifact {
    source_baseline: Arc<CoverageMappedSourceBaseline>,
    entry: worktree::Entry,
    input: Arc<crate::profile_results::ProfileInput>,
    cancellation: Arc<AtomicBool>,
}

pub struct RustCoverageProviderStore {
    mode: RustCoverageProviderStoreMode,
    worktree_store: Entity<WorktreeStore>,
    source_coverage_store: Entity<SourceCoverageStore>,
    provider: RustCoverageArtifactProvider,
    experimental_subscription: Option<gpui::Subscription>,
    #[cfg(any(test, feature = "test-support"))]
    remote_reply_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    host_preparation_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    local_parse_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
}

enum RustCoverageProviderStoreMode {
    Local,
    Remote {
        project_id: u64,
        client: AnyProtoClient,
    },
}

impl RustCoverageProviderStore {
    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_interpret_artifact);
    }

    pub fn local(
        worktree_store: Entity<WorktreeStore>,
        source_coverage_store: Entity<SourceCoverageStore>,
        project_generation: u64,
    ) -> Self {
        Self {
            mode: RustCoverageProviderStoreMode::Local,
            worktree_store,
            source_coverage_store,
            provider: RustCoverageArtifactProvider::new(project_generation),
            experimental_subscription: None,
            #[cfg(any(test, feature = "test-support"))]
            remote_reply_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            local_parse_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            host_preparation_gate: None,
        }
    }

    pub fn remote(
        project_id: u64,
        client: AnyProtoClient,
        worktree_store: Entity<WorktreeStore>,
        source_coverage_store: Entity<SourceCoverageStore>,
        project_generation: u64,
    ) -> Self {
        Self {
            mode: RustCoverageProviderStoreMode::Remote { project_id, client },
            worktree_store,
            source_coverage_store,
            provider: RustCoverageArtifactProvider::new(project_generation),
            experimental_subscription: None,
            #[cfg(any(test, feature = "test-support"))]
            remote_reply_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            local_parse_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            host_preparation_gate: None,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_remote_reply_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        self.remote_reply_gate = Some((started_sender, release_receiver));
        (started_receiver, release_sender)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_host_preparation_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    ) {
        let (started, receiver) = futures::channel::oneshot::channel();
        let (release, released) = futures::channel::oneshot::channel();
        self.host_preparation_gate = Some((started, released));
        (receiver, release)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_local_parse_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        self.local_parse_gate = Some((started_sender, release_receiver));
        (started_receiver, release_sender)
    }

    pub fn interpret_artifact(
        &mut self,
        artifact_path: ProjectPath,
        max_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        let token = match self.begin_run(cx) {
            Ok(token) => token,
            Err(error) => return Task::ready(Err(error)),
        };
        self.interpret_artifact_for_run(token, artifact_path, max_bytes, cx)
    }

    fn ensure_experimental_admission(&mut self, cx: &mut Context<Self>) -> Result<()> {
        if !cx.has_flag::<RustCoverageExperimentalFeatureFlag>() {
            return Err(crate::source_coverage::SourceCoverageUnsupportedHost.into());
        }
        if self.experimental_subscription.is_none() {
            self.experimental_subscription = Some(
                cx.observe_global::<feature_flags::FeatureFlagStore>(|this, cx| {
                    if !cx.has_flag::<RustCoverageExperimentalFeatureFlag>() {
                        this.provider.generation = this.provider.generation.wrapping_add(1);
                    }
                }),
            );
        }
        Ok(())
    }

    pub fn begin_run(&mut self, cx: &mut Context<Self>) -> Result<RustCoverageRunToken> {
        self.ensure_experimental_admission(cx)?;
        let request_version = self.source_coverage_store.read(cx).request_version()?;
        let mut token = self.provider.begin_run();
        token.request_version = Some(request_version);
        Ok(token)
    }

    pub(crate) fn begin_task_capture_run(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Result<RustCoverageRunToken> {
        let token = self.begin_run(cx)?;
        let publication = self.source_coverage_store.update(cx, |store, cx| {
            store.publish(
                SourceCoverageSnapshot {
                    project_generation: token.project_generation,
                    provider_id: SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.into()),
                    generation: token.generation,
                    status: SourceCoverageStatus::Loading,
                    files: Vec::new(),
                    truncated: false,
                    diagnostic: None,
                },
                cx,
            )
        });
        if let Err(error) = publication {
            self.cancel_run(token);
            return Err(error);
        }
        Ok(token)
    }

    pub fn is_run_current(&self, token: RustCoverageRunToken, cx: &App) -> bool {
        cx.has_flag::<RustCoverageExperimentalFeatureFlag>()
            && self.provider.is_run_current(token)
            && token.request_version.is_some_and(|version| {
                self.source_coverage_store
                    .read(cx)
                    .is_request_current(version)
            })
    }

    pub fn owns_run(&self, token: RustCoverageRunToken) -> bool {
        self.provider.is_run_current(token)
    }

    pub fn seal_source_revision(
        &self,
        mut token: RustCoverageRunToken,
        cx: &App,
    ) -> Result<RustCoverageRunToken> {
        if !self.is_run_current(token, cx) {
            bail!("Rust coverage Task expired while saving source");
        }
        token.request_version = Some(self.source_coverage_store.read(cx).request_version()?);
        Ok(token)
    }

    fn task_capture_current(&self, capture: &RustCoverageTaskCapture, cx: &App) -> bool {
        !capture.cancellation.load(Ordering::Relaxed)
            && self.is_run_current(capture.token.get(), cx)
    }

    pub(crate) fn finish_expired_task_capture(
        &mut self,
        capture: &RustCoverageTaskCapture,
        cx: &mut Context<Self>,
    ) -> bool {
        if capture.provider.entity_id() != cx.entity_id() {
            return true;
        }
        if self.task_capture_current(capture, cx) {
            return false;
        }
        self.finish_task_capture_status(
            capture.token.get(),
            SourceCoverageStatus::Stale,
            COVERAGE_TASK_EXPIRED_DIAGNOSTIC,
            cx,
        );
        true
    }

    pub(crate) fn revalidate_task_capture_before_launch(
        &self,
        capture: &RustCoverageTaskCapture,
        cx: &App,
    ) -> Result<()> {
        anyhow::ensure!(
            capture.admitted && self.task_capture_current(capture, cx),
            "Coverage Task capture expired before launch"
        );
        let admitted_version = capture
            .token
            .get()
            .request_version
            .context("Coverage Task source admission is missing")?;
        if self.source_coverage_store.read(cx).request_version()? != admitted_version {
            return Err(crate::source_coverage::SourceCoverageSourceChanged.into());
        }
        Ok(())
    }

    pub(crate) fn is_task_capture_source_revision_current(
        &self,
        capture: &RustCoverageTaskCapture,
        cx: &App,
    ) -> bool {
        self.task_capture_request_is_current(
            capture.token.get(),
            capture.admitted_output.borrow().as_ref(),
            cx,
        )
    }

    fn task_capture_request_is_current(
        &self,
        token: RustCoverageRunToken,
        admitted_output: Option<&ProjectPath>,
        cx: &App,
    ) -> bool {
        token.request_version.is_some_and(|admitted| {
            let coverage = self.source_coverage_store.read(cx);
            match admitted_output {
                Some(output) => coverage.is_request_current_except_output(admitted, output),
                None => coverage
                    .request_version()
                    .is_ok_and(|current| current == admitted),
            }
        })
    }

    pub fn finish_cancelled_run(&mut self, token: RustCoverageRunToken, cx: &mut Context<Self>) {
        self.finish_task_capture_status(
            token,
            SourceCoverageStatus::Cancelled,
            "Coverage Task capture was cancelled",
            cx,
        );
    }

    pub(crate) fn finish_task_capture_status(
        &mut self,
        token: RustCoverageRunToken,
        status: SourceCoverageStatus,
        diagnostic: &'static str,
        cx: &mut Context<Self>,
    ) {
        self.finish_task_capture_status_with_output(token, status, diagnostic, None, cx);
    }

    pub(crate) fn finish_task_capture_status_with_output(
        &mut self,
        token: RustCoverageRunToken,
        status: SourceCoverageStatus,
        diagnostic: &'static str,
        admitted_output: Option<&ProjectPath>,
        cx: &mut Context<Self>,
    ) {
        if !self.owns_run(token) {
            return;
        }
        let retained_status = self
            .source_coverage_store
            .read(cx)
            .cached_snapshots()
            .find(|snapshot| snapshot.provider_id.0 == RUST_COVERAGE_PROVIDER_ID)
            .map(|snapshot| snapshot.status)
            .filter(|status| {
                matches!(
                    status,
                    SourceCoverageStatus::Disconnected | SourceCoverageStatus::Restricted
                )
            });
        if retained_status.is_some() {
            self.cancel_run(token);
            return;
        }
        // Only a capture's validated output is expected to change after admission.
        let cancelled_source_changed = status == SourceCoverageStatus::Cancelled
            && !self.task_capture_request_is_current(token, admitted_output, cx);
        let (status, diagnostic) = if (!self.is_run_current(token, cx) || cancelled_source_changed)
            && !matches!(
                status,
                SourceCoverageStatus::Restricted | SourceCoverageStatus::Disconnected
            ) {
            (
                SourceCoverageStatus::Stale,
                COVERAGE_TASK_EXPIRED_DIAGNOSTIC,
            )
        } else {
            (status, diagnostic)
        };
        if let Err(error) = self.source_coverage_store.update(cx, |store, cx| {
            store.mark_provider_status(
                SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.into()),
                status,
                Some(diagnostic.into()),
                cx,
            )
        }) {
            log::error!("Coverage capture terminal state could not be published: {error:#}");
        }
        self.cancel_run(token);
    }

    pub(crate) fn subscribe_capture_buffer_changes(
        &mut self,
        buffer_store: &Entity<crate::buffer_store::BufferStore>,
        cx: &mut Context<Self>,
    ) -> gpui::Subscription {
        let subscriptions = Rc::new(RefCell::new(Vec::new()));
        let buffers = buffer_store.read(cx).buffers().collect::<Vec<_>>();
        for buffer in buffers {
            subscriptions.borrow_mut().push((
                buffer.downgrade(),
                Self::subscribe_capture_buffer(&buffer, cx),
            ));
        }
        let weak_subscriptions = Rc::downgrade(&subscriptions);
        let outer = cx.subscribe(buffer_store, move |_, _, event, cx| {
            let Some(subscriptions) = weak_subscriptions.upgrade() else {
                return;
            };
            let mut subscriptions = subscriptions.borrow_mut();
            subscriptions.retain(|(buffer, _)| buffer.upgrade().is_some());
            if let crate::buffer_store::BufferStoreEvent::BufferAdded(buffer) = event {
                subscriptions.push((
                    buffer.downgrade(),
                    Self::subscribe_capture_buffer(buffer, cx),
                ));
            }
        });
        gpui::Subscription::new(move || {
            // Removing the outer callback must release the subscriber-set borrow before children.
            drop(outer);
            drop(subscriptions);
        })
    }

    fn subscribe_capture_buffer(
        buffer: &Entity<language::Buffer>,
        cx: &mut Context<Self>,
    ) -> gpui::Subscription {
        let mut version = buffer.read(cx).version();
        cx.subscribe(buffer, move |store, buffer, event, cx| {
            if matches!(
                event,
                language::BufferEvent::Edited { .. } | language::BufferEvent::Reloaded
            ) {
                let current_version = buffer.read(cx).version();
                if current_version != version {
                    version = current_version;
                    if let Some(path) = buffer.read(cx).project_path(cx) {
                        store.source_coverage_store.update(cx, |coverage, cx| {
                            coverage.invalidate_paths(&[path], cx);
                        });
                    }
                }
            }
        })
    }

    pub(crate) fn begin_capture_source_baseline(
        &self,
        capture: &RustCoverageTaskCapture,
        buffer_store: &Entity<crate::buffer_store::BufferStore>,
        cx: &App,
    ) -> Result<CoverageMappedSourceBaselineState> {
        anyhow::ensure!(
            capture.admitted && self.task_capture_current(capture, cx),
            "Coverage Task source admission expired"
        );
        let output = capture
            .path
            .as_ref()
            .context("Coverage Task output admission is missing")?;
        let buffers = buffer_store
            .read(cx)
            .buffers()
            .filter_map(|buffer| {
                let path = buffer.read(cx).project_path(cx)?;
                Some(CoverageMappedBuffer {
                    path,
                    buffer: buffer.downgrade(),
                    version: buffer.read(cx).version(),
                    dirty: buffer.read(cx).is_dirty(),
                })
            })
            .collect();
        let roots = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|tree| {
                let worktree = tree.read(cx);
                anyhow::ensure!(
                    worktree.is_local(),
                    "Coverage Task sources are not on their host"
                );
                Ok(worktree.snapshot())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CoverageMappedSourceBaselineState::Pending(Task::ready(Ok(
            Arc::new(CoverageMappedSourceBaseline {
                buffer_store: buffer_store.downgrade(),
                buffers,
                roots,
                output: output.clone(),
            }),
        ))))
    }

    pub(crate) fn prepare_capture_sources(
        &mut self,
        mut capture: RustCoverageTaskCapture,
        before_launch: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<RustCoverageTaskCapture>> {
        cx.spawn(async move |this, cx| {
            let result = async {
                let baseline = match capture
                    .source_baseline
                    .take()
                    .context("Coverage source admission is missing")?
                {
                    CoverageMappedSourceBaselineState::Pending(task) => task.await?,
                    CoverageMappedSourceBaselineState::Ready(baseline) => baseline,
                };
                this.read_with(cx, |store, cx| {
                    if before_launch {
                        store.revalidate_task_capture_before_launch(&capture, cx)?;
                    } else {
                        anyhow::ensure!(
                            store.task_capture_current(&capture, cx),
                            COVERAGE_TASK_EXPIRED_DIAGNOSTIC
                        );
                    }
                    Ok::<_, anyhow::Error>(())
                })??;
                capture.source_baseline = Some(CoverageMappedSourceBaselineState::Ready(baseline));
                Ok::<_, anyhow::Error>(())
            }
            .await;
            if let Err(error) = result {
                this.update(cx, |store, cx| {
                    store.finish_task_capture_status(
                        capture.token.get(),
                        SourceCoverageStatus::Stale,
                        "Coverage Task source admission could not be retained",
                        cx,
                    )
                })?;
                capture.armed = false;
                drop(capture);
                return Err(error);
            }
            Ok(capture)
        })
    }

    pub(crate) fn complete_local_task_capture(
        &mut self,
        capture: RustCoverageTaskCapture,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        let prepared = self.prepare_capture_sources(capture, false, cx);
        cx.spawn(async move |this, cx| {
            let capture = prepared.await?;
            this.update(cx, |store, cx| {
                store.complete_local_task_capture_with_sources(capture, cx)
            })?
            .await
        })
    }

    fn complete_local_task_capture_with_sources(
        &mut self,
        mut capture: RustCoverageTaskCapture,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        if self.finish_expired_task_capture(&capture, cx) {
            capture.armed = false;
            return Task::ready(Err(anyhow::anyhow!(COVERAGE_TASK_EXPIRED_DIAGNOSTIC)));
        }
        if !capture.admitted {
            self.finish_task_capture_status(
                capture.token.get(),
                SourceCoverageStatus::Stale,
                "Coverage Task admission expired",
                cx,
            );
            return Task::ready(Err(anyhow::anyhow!(
                "Coverage Task capture was not admitted or became stale"
            )));
        }
        let Some(artifact_path) = capture.path.clone() else {
            self.finish_task_capture_status(
                capture.token.get(),
                SourceCoverageStatus::Stale,
                "Coverage Task artifact was not admitted",
                cx,
            );
            return Task::ready(Err(anyhow::anyhow!(
                "Coverage capture has no admitted artifact"
            )));
        };
        let worktree = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .find(|worktree| worktree.read(cx).id() == artifact_path.worktree_id);
        let Some(worktree) = worktree else {
            self.finish_task_capture_status(
                capture.token.get(),
                SourceCoverageStatus::Stale,
                "Coverage worktree is no longer visible",
                cx,
            );
            return Task::ready(Err(anyhow::anyhow!("Coverage worktree is unavailable")));
        };
        let refresh = match worktree.update(cx, |worktree, cx| {
            worktree
                .as_local()
                .context("Coverage capture is not on its host")
                .map(|local| local.refresh_entry(artifact_path.path.clone(), None, cx))
        }) {
            Ok(refresh) => refresh,
            Err(error) => return Task::ready(Err(error)),
        };
        cx.spawn(async move |this, cx| {
            let result = async {
                let entry = refresh.await?.context("Coverage Task created no report")?;
                let (filesystem, input) = this.read_with(cx, |store, cx| {
                    anyhow::ensure!(
                        store.task_capture_current(&capture, cx),
                        "Coverage Task capture became stale"
                    );
                    store.validate_local_artifact_entry(&artifact_path, &entry, cx)?;
                    let worktree = store
                        .worktree_store
                        .read(cx)
                        .visible_worktrees(cx)
                        .find(|worktree| worktree.read(cx).id() == artifact_path.worktree_id)
                        .context("Coverage worktree is unavailable")?;
                    let input = crate::profile_results::ProfileInput::from_snapshot(
                        &worktree.read(cx).snapshot(),
                        artifact_path.path.as_ref(),
                    )?;
                    let filesystem = store
                        .worktree_store
                        .read(cx)
                        .fs()
                        .context("Coverage filesystem is unavailable")?;
                    Ok::<_, anyhow::Error>((filesystem, input))
                })??;
                let max_bytes = usize::try_from(capture.artifact.max_bytes)
                    .context("Coverage Task byte limit is invalid")?;
                let cancellation = capture.cancellation.clone();
                let fingerprint = cx
                    .background_spawn(async move {
                        let fingerprint = input
                            .read_with_fingerprint(filesystem.as_ref(), max_bytes, &cancellation)
                            .await?
                            .1;
                        Ok::<_, anyhow::Error>((input.with_fingerprint(fingerprint), cancellation))
                    })
                    .await?;
                let interpretation = this.update(cx, |store, cx| {
                    anyhow::ensure!(
                        store.task_capture_current(&capture, cx),
                        "Coverage Task capture became stale"
                    );
                    store.validate_local_artifact_entry(&artifact_path, &entry, cx)?;
                    Ok::<_, anyhow::Error>(store.interpret_local_with_seal(
                        capture.token.get(),
                        artifact_path.clone(),
                        max_bytes,
                        Some(SealedCoverageArtifact {
                            source_baseline: match capture.source_baseline.as_ref() {
                                Some(CoverageMappedSourceBaselineState::Ready(baseline)) => {
                                    baseline.clone()
                                }
                                _ => bail!("Coverage Task source baseline was not prepared"),
                            },
                            entry,
                            input: Arc::new(fingerprint.0),
                            cancellation: fingerprint.1,
                        }),
                        cx,
                    ))
                })??;
                interpretation.await
            }
            .await;
            if let Err(error) = &result {
                this.update(cx, |store, cx| {
                    if store.finish_expired_task_capture(&capture, cx) {
                        return;
                    }
                    let status =
                        if error.is::<crate::source_coverage::SourceCoverageSourceChanged>() {
                            SourceCoverageStatus::Stale
                        } else {
                            SourceCoverageStatus::Error
                        };
                    store.finish_task_capture_status(
                        capture.token.get(),
                        status,
                        if status == SourceCoverageStatus::Stale {
                            "Coverage Task or source became stale"
                        } else {
                            "Coverage Task report could not be sealed and interpreted"
                        },
                        cx,
                    );
                })?;
            }
            capture.armed = false;
            drop(capture);
            result
        })
    }

    pub fn cancel_run(&mut self, token: RustCoverageRunToken) {
        self.provider.cancel(token);
    }

    pub fn interpret_artifact_for_run(
        &mut self,
        token: RustCoverageRunToken,
        artifact_path: ProjectPath,
        max_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        if !self.is_run_current(token, cx) {
            return Task::ready(Err(anyhow::anyhow!(
                "Rust coverage Task belongs to a stale or cancelled run"
            )));
        }
        match &self.mode {
            RustCoverageProviderStoreMode::Local => {
                self.interpret_local(token, artifact_path, max_bytes, cx)
            }
            RustCoverageProviderStoreMode::Remote { project_id, client } => {
                let project_id = *project_id;
                let client = client.clone();
                let source_coverage_store = self.source_coverage_store.clone();
                let request_version = match token.request_version {
                    Some(version) => version,
                    None => {
                        return Task::ready(Err(anyhow::anyhow!(
                            "Rust coverage admission is missing"
                        )));
                    }
                };
                #[cfg(any(test, feature = "test-support"))]
                let reply_gate = self.remote_reply_gate.take();
                cx.spawn(async move |this, cx| {
                    let response = client
                        .request(proto::InterpretRustCoverageArtifact {
                            project_id,
                            artifact_path: Some(artifact_path.to_proto()),
                            max_bytes: u64::try_from(max_bytes).unwrap_or(u64::MAX),
                        })
                        .await
                        .map_err(map_remote_coverage_error)?;
                    #[cfg(any(test, feature = "test-support"))]
                    if let Some((started, release)) = reply_gate {
                        started.send(()).map_err(|_| {
                            anyhow::anyhow!("coverage reply start receiver was dropped")
                        })?;
                        release
                            .await
                            .context("coverage reply release was dropped")?;
                    }
                    let snapshot = snapshot_from_proto(response)?;
                    this.read_with(cx, |store, _| {
                        if !store.owns_run(token) {
                            bail!("Rust coverage Task completed after cancellation");
                        }
                        Ok(())
                    })??;
                    source_coverage_store.update(cx, |store, cx| {
                        store.replace_remote(snapshot, request_version, cx)
                    })
                })
            }
        }
    }

    fn validate_local_artifact_entry(
        &self,
        artifact_path: &ProjectPath,
        expected: &worktree::Entry,
        cx: &App,
    ) -> Result<()> {
        if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx) {
            bail!("Rust coverage is unavailable for restricted worktrees");
        }
        let worktree = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .find(|worktree| worktree.read(cx).id() == artifact_path.worktree_id)
            .context("coverage artifact worktree is no longer visible")?;
        let worktree = worktree.read(cx);
        anyhow::ensure!(
            worktree.as_local().is_some(),
            "coverage artifact is not on its host"
        );
        let snapshot = worktree.snapshot();
        let entry = snapshot
            .entry_for_path(artifact_path.path.as_ref())
            .context("coverage artifact is no longer visible")?;
        anyhow::ensure!(
            entry.is_file() && !entry.is_private && !entry.is_external && !entry.is_fifo,
            "coverage artifact is not a permitted regular file"
        );
        anyhow::ensure!(
            entry.id == expected.id
                && entry.path == expected.path
                && entry.inode == expected.inode
                && entry.mtime == expected.mtime
                && entry.size == expected.size
                && entry.canonical_path == expected.canonical_path,
            "coverage artifact changed after Task completion"
        );
        Ok(())
    }

    fn interpret_local(
        &mut self,
        token: RustCoverageRunToken,
        artifact_path: ProjectPath,
        max_bytes: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        self.interpret_local_with_seal(token, artifact_path, max_bytes, None, cx)
    }

    fn interpret_local_with_seal(
        &mut self,
        token: RustCoverageRunToken,
        artifact_path: ProjectPath,
        max_bytes: usize,
        sealed: Option<SealedCoverageArtifact>,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        let max_bytes = max_bytes.min(MAX_RUST_COVERAGE_ARTIFACT_BYTES);
        let parsed =
            self.read_local_artifact_with_seal(token, artifact_path, max_bytes, sealed, cx);
        self.publish_local_artifact_for_run(token, max_bytes, parsed, cx)
    }

    fn publish_local_artifact_for_run(
        &self,
        token: RustCoverageRunToken,
        max_bytes: usize,
        parsed: Task<Result<ParsedLocalCoverageArtifact>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        let source_coverage_store = self.source_coverage_store.clone();
        cx.spawn(async move |this, cx| {
            let ParsedLocalCoverageArtifact {
                snapshot,
                artifact_path,
                expected_entry,
                worktree,
                sealed,
            } = parsed.await?;
            this.read_with(cx, |store, cx| {
                if !store.is_run_current(token, cx) {
                    bail!("Rust coverage artifact completed after cancellation");
                }
                Ok(())
            })??;
            if let Some(sealed) = sealed.as_ref() {
                let filesystem = this.read_with(cx, |store, cx| {
                    store
                        .worktree_store
                        .read(cx)
                        .fs()
                        .context("Coverage filesystem is unavailable")
                })??;
                this.read_with(cx, |_, cx| {
                    sealed.source_baseline.validate_buffers(&snapshot, cx)
                })??;
                let input = sealed.input.clone();
                let cancellation = sealed.cancellation.clone();
                let source_baseline = sealed.source_baseline.clone();
                let reported_paths = snapshot
                    .files
                    .iter()
                    .map(|file| file.path.clone())
                    .collect::<Vec<_>>();
                let executor = cx.background_executor().clone();
                cx.background_spawn(async move {
                    let verification = async {
                        input
                            .read_with_fingerprint(filesystem.as_ref(), max_bytes, &cancellation)
                            .await?;
                        source_baseline
                            .validate_reported_disk_sources(
                                &reported_paths,
                                filesystem.as_ref(),
                                &cancellation,
                            )
                            .await
                    };
                    match futures::future::select(
                        Box::pin(verification),
                        Box::pin(executor.timer(std::time::Duration::from_secs(30))),
                    )
                    .await
                    {
                        futures::future::Either::Left((result, _)) => result,
                        futures::future::Either::Right(_) => {
                            bail!("Coverage source publication verification timed out")
                        }
                    }
                })
                .await?;
            }
            let refreshed_entry = worktree
                .update(cx, |worktree, cx| {
                    worktree
                        .as_local()
                        .context("coverage artifact is no longer on its host")
                        .map(|local| local.refresh_entry(artifact_path.path.clone(), None, cx))
                })?
                .await?
                .context("coverage artifact disappeared during interpretation")?;
            anyhow::ensure!(
                refreshed_entry.id == expected_entry.id
                    && refreshed_entry.inode == expected_entry.inode
                    && refreshed_entry.mtime == expected_entry.mtime
                    && refreshed_entry.size == expected_entry.size
                    && refreshed_entry.canonical_path == expected_entry.canonical_path,
                "coverage artifact changed during interpretation"
            );
            this.read_with(cx, |store, cx| {
                if !store.is_run_current(token, cx) {
                    bail!("Rust coverage artifact completed after cancellation");
                }
                store.validate_local_artifact_entry(&artifact_path, &expected_entry, cx)?;
                if let Some(sealed) = sealed.as_ref() {
                    sealed.source_baseline.validate_buffers(&snapshot, cx)?;
                }
                Ok::<_, anyhow::Error>(())
            })??;
            source_coverage_store.update(cx, |store, cx| {
                store.validate_snapshot_version(
                    &snapshot,
                    token
                        .request_version
                        .context("Rust coverage admission is missing")?,
                )?;
                store.publish(snapshot.clone(), cx)
            })?;
            Ok(snapshot)
        })
    }

    fn read_local_artifact_with_seal(
        &mut self,
        token: RustCoverageRunToken,
        artifact_path: ProjectPath,
        max_bytes: usize,
        sealed: Option<SealedCoverageArtifact>,
        cx: &mut Context<Self>,
    ) -> Task<Result<ParsedLocalCoverageArtifact>> {
        #[cfg(any(test, feature = "test-support"))]
        LOCAL_RUST_COVERAGE_INTERPRET_CALLS.fetch_add(1, Ordering::Relaxed);
        let max_bytes = max_bytes.min(MAX_RUST_COVERAGE_ARTIFACT_BYTES);
        let Some(worktree) = self
            .worktree_store
            .read(cx)
            .worktree_for_id(artifact_path.worktree_id, cx)
        else {
            return Task::ready(Err(anyhow::anyhow!(
                "coverage artifact worktree is not visible"
            )));
        };
        let roots = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| {
                worktree.read_with(cx, |worktree, _| {
                    (worktree.id(), worktree.abs_path().to_path_buf())
                })
            })
            .collect::<Vec<_>>();
        let refresh = match worktree.update(cx, |worktree, cx| {
            worktree
                .as_local()
                .context("coverage artifact is not on its host")
                .map(|local| local.refresh_entry(artifact_path.path.clone(), None, cx))
        }) {
            Ok(refresh) => refresh,
            Err(error) => return Task::ready(Err(error)),
        };
        #[cfg(any(test, feature = "test-support"))]
        let parse_gate = self.local_parse_gate.take();
        let project_generation = token.project_generation;
        let generation = token.generation;
        cx.spawn(async move |this, cx| {
            let expected_entry = refresh
                .await?
                .context("coverage artifact is missing after Task completion")?;
            this.read_with(cx, |store, cx| {
                if !store.is_run_current(token, cx) {
                    bail!("Rust coverage artifact completed after cancellation");
                }
                store.validate_local_artifact_entry(&artifact_path, &expected_entry, cx)
            })??;
            anyhow::ensure!(
                usize::try_from(expected_entry.size).unwrap_or(usize::MAX) <= max_bytes,
                "coverage artifact exceeds the declared byte limit"
            );
            let loaded_text = if let Some(sealed) = sealed.as_ref() {
                anyhow::ensure!(
                    expected_entry == sealed.entry,
                    "Sealed coverage entry changed"
                );
                let filesystem = this.read_with(cx, |store, cx| {
                    store
                        .worktree_store
                        .read(cx)
                        .fs()
                        .context("Coverage filesystem is unavailable")
                })??;
                let input = sealed.input.clone();
                let cancellation = sealed.cancellation.clone();
                cx.background_spawn(async move {
                    let (bytes, _) = input
                        .read_with_fingerprint(filesystem.as_ref(), max_bytes, &cancellation)
                        .await?;
                    worktree::decode_file_bytes_with_limit(bytes, max_bytes)
                        .map(|decoded| decoded.0)
                })
                .await?
            } else {
                let loaded = worktree
                    .update(cx, |worktree, cx| {
                        worktree.load_file_with_limit(artifact_path.path.as_ref(), max_bytes, cx)
                    })
                    .await?;
                anyhow::ensure!(
                    loaded.file.entry_id == Some(expected_entry.id)
                        && loaded.file.path == expected_entry.path
                        && loaded.file.disk_state.mtime() == expected_entry.mtime
                        && loaded.file.disk_state.size() == Some(expected_entry.size)
                        && loaded.file.is_local
                        && !loaded.file.is_private,
                    "coverage artifact changed while loading"
                );
                loaded.text
            };
            this.read_with(cx, |store, cx| {
                if !store.is_run_current(token, cx) {
                    bail!("Rust coverage artifact completed after cancellation");
                }
                store.validate_local_artifact_entry(&artifact_path, &expected_entry, cx)
            })??;
            let parse = cx.background_spawn(async move {
                parse_coverage_artifact(
                    project_generation,
                    generation,
                    loaded_text.as_bytes(),
                    |absolute_path| {
                        roots.iter().find_map(|(worktree_id, root)| {
                            let relative = absolute_path.strip_prefix(root).ok()?;
                            let path = util::rel_path::RelPath::new(
                                relative,
                                util::paths::PathStyle::local(),
                            )
                            .ok()?;
                            Some(ProjectPath {
                                worktree_id: *worktree_id,
                                path: path.as_ref().into(),
                            })
                        })
                    },
                )
            });
            let snapshot = parse.await?;
            #[cfg(any(test, feature = "test-support"))]
            if let Some((started, release)) = parse_gate {
                started.send(()).map_err(|_| {
                    anyhow::anyhow!("coverage local parse start receiver was dropped")
                })?;
                release
                    .await
                    .context("coverage local parse release was dropped")?;
            }
            Ok(ParsedLocalCoverageArtifact {
                snapshot,
                artifact_path,
                expected_entry,
                worktree,
                sealed,
            })
        })
    }

    async fn handle_interpret_artifact(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::InterpretRustCoverageArtifact>,
        mut cx: AsyncApp,
    ) -> Result<proto::SourceCoverageResponse> {
        let admission_generation = this
            .update(&mut cx, |store, cx| {
                store.ensure_experimental_admission(cx)?;
                Ok::<_, anyhow::Error>(store.provider.generation)
            })
            .map_err(tag_remote_coverage_unsupported)?;

        let request = envelope.payload;
        let artifact_path = ProjectPath::from_proto(
            request
                .artifact_path
                .context("missing Rust coverage artifact path")?,
        )
        .context("invalid Rust coverage artifact path")?;
        let max_bytes = usize::try_from(request.max_bytes)
            .unwrap_or(usize::MAX)
            .min(MAX_RUST_COVERAGE_ARTIFACT_BYTES);
        let remote = this.read_with(&cx, |store, _| {
            matches!(&store.mode, RustCoverageProviderStoreMode::Remote { .. })
        });
        if remote {
            this.read_with(&cx, |store, cx| {
                if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx) {
                    bail!("Rust coverage is unavailable for restricted worktrees");
                }
                let visible = store
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .find(|worktree| worktree.read(cx).id() == artifact_path.worktree_id)
                    .is_some_and(|worktree| {
                        worktree
                            .read(cx)
                            .snapshot()
                            .entry_for_path(artifact_path.path.as_ref())
                            .is_some_and(|entry| {
                                entry.is_file()
                                    && !entry.is_private
                                    && !entry.is_fifo
                                    && !entry.is_external
                            })
                    });
                if !visible {
                    bail!("Rust coverage artifact is not visible to the host");
                }
                Ok::<_, anyhow::Error>(())
            })?;
        } else {
            let mut refreshed = this.read_with(&cx, |store, cx| {
                if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx) {
                    bail!("Rust coverage is unavailable for restricted worktrees");
                }
                let worktree = store
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .find(|worktree| worktree.read(cx).id() == artifact_path.worktree_id)
                    .context("Rust coverage artifact worktree is not visible")?;
                let worktree = worktree.read(cx);
                let local = worktree
                    .as_local()
                    .context("Rust coverage artifact worktree is not local to the host")?;
                Ok::<_, anyhow::Error>(
                    local.refresh_entries_for_paths(vec![artifact_path.path.clone()]),
                )
            })?;
            refreshed
                .next()
                .await
                .context("Rust coverage artifact refresh was cancelled")?;
        }
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release)) =
            this.update(&mut cx, |store, _| store.host_preparation_gate.take())
        {
            started.send(()).map_err(|_| {
                anyhow::anyhow!("coverage host preparation start receiver was dropped")
            })?;
            release
                .await
                .context("coverage host preparation release was dropped")?;
        }
        let artifact_path_for_verification = artifact_path.clone();
        let (token, interpretation) = this
            .update(&mut cx, |store, cx| {
                anyhow::ensure!(
                    store.provider.generation == admission_generation,
                    "Rust coverage host opt-in was revoked during preparation"
                );
                let token = store.begin_run(cx)?;
                Ok::<_, anyhow::Error>((
                    token,
                    store.interpret_artifact_for_run(token, artifact_path, max_bytes, cx),
                ))
            })
            .map_err(tag_remote_coverage_unsupported)?;
        let mut snapshot = interpretation
            .await
            .map_err(tag_remote_coverage_unsupported)?;
        this.read_with(&cx, |store, cx| {
            anyhow::ensure!(
                store.is_run_current(token, cx),
                "Rust coverage host interpretation was revoked"
            );
            if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx) {
                bail!("Rust coverage is unavailable for restricted worktrees");
            }
            let visible_worktrees = store
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| (worktree.read(cx).id(), worktree))
                .collect::<HashMap<_, _>>();
            let is_visible_file = |path: &ProjectPath| {
                visible_worktrees
                    .get(&path.worktree_id)
                    .is_some_and(|worktree| {
                        worktree
                            .read(cx)
                            .snapshot()
                            .entry_for_path(path.path.as_ref())
                            .is_some_and(|entry| {
                                entry.is_file()
                                    && !entry.is_private
                                    && !entry.is_fifo
                                    && !entry.is_external
                            })
                    })
            };
            if !is_visible_file(&artifact_path_for_verification) {
                bail!("Rust coverage artifact is no longer visible");
            }
            let previous_count = snapshot.files.len();
            snapshot.files.retain(|file| is_visible_file(&file.path));
            snapshot.truncated |= snapshot.files.len() != previous_count;
            Ok::<_, anyhow::Error>(())
        })?;
        Ok(snapshot_to_proto(&snapshot, None))
    }
}

fn tag_remote_coverage_unsupported(error: anyhow::Error) -> anyhow::Error {
    if error.is::<SourceCoverageUnsupportedHost>() {
        proto::ErrorCode::Internal
            .with_tag("reason", "rust_tools_unsupported")
            .message(error.to_string())
            .into()
    } else {
        error
    }
}

fn map_remote_coverage_error(error: anyhow::Error) -> anyhow::Error {
    let unsupported = error
        .downcast_ref::<proto::RpcError>()
        .is_some_and(|error| {
            error.error_tag("reason") == Some("rust_tools_unsupported")
                || error.raw_message().contains("was not handled")
                || error
                    .raw_message()
                    .starts_with("no handler registered for ")
        });
    if unsupported {
        SourceCoverageUnsupportedHost.into()
    } else {
        anyhow::anyhow!("The Rust coverage host rejected the request: {error}")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn resolve(path: &Path) -> Option<ProjectPath> {
        let relative = path.strip_prefix("/workspace").ok()?;
        Some(ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(
                RelPath::new(relative, util::paths::PathStyle::Unix)
                    .ok()?
                    .as_ref(),
            ),
        })
    }

    #[test]
    fn rust_coverage_provider_read_publication_split_parser_facts_are_not_admission() {
        let contents = include_bytes!("../test_data/source_coverage/llvm3-run.json");
        let mut provider = RustCoverageArtifactProvider::new(17);
        let obsolete = provider.begin_run();
        let current = provider.begin_run();
        let facts = parse_coverage_artifact(17, obsolete.generation, contents, resolve)
            .expect("bounded parser returns data independently of publication ownership");
        assert_eq!(facts.project_generation, 17);
        assert_eq!(facts.generation, obsolete.generation);
        assert_eq!(facts.status, SourceCoverageStatus::Current);
        assert_eq!(facts.files.len(), 1);
        assert!(
            provider
                .parse_artifact(obsolete, contents, resolve)
                .is_err()
        );
        assert!(provider.is_run_current(current));
        assert_eq!(
            provider
                .parse_artifact(current, contents, resolve)
                .expect("current parser wrapper keeps its admission check")
                .files,
            facts.files
        );
    }

    #[gpui::test]
    async fn rust_coverage_experimental_opt_in_revokes_original_run_and_preserves_generic_coverage(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                false,
                cx,
            )
        });
        let filesystem = crate::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"src": {"lib.rs": "fn fixture() {}"}}),
            )
            .await;
        let project = crate::Project::test(filesystem, ["/workspace".as_ref()], cx).await;
        let (provider, coverage) = project.read_with(cx, |project, _| {
            (
                project.rust_coverage_provider_store().clone(),
                project.source_coverage_store().clone(),
            )
        });
        assert!(
            provider
                .update(cx, |store, cx| store.begin_run(cx))
                .is_err()
        );
        coverage
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        project_generation: 1,
                        provider_id: SourceCoverageProviderId("ordinary-language".into()),
                        generation: 1,
                        status: SourceCoverageStatus::Current,
                        files: Vec::new(),
                        truncated: false,
                        diagnostic: None,
                    },
                    cx,
                )
            })
            .expect("generic coverage remains available without Rust opt-in");
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });
        cx.run_until_parked();
        let token = provider
            .update(cx, |store, cx| store.begin_task_capture_run(cx))
            .expect("explicit local opt-in admits run");
        assert!(provider.read_with(cx, |store, cx| store.is_run_current(token, cx)));
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                false,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(!provider.read_with(cx, |store, cx| store.is_run_current(token, cx)));
        assert_eq!(
            coverage.read_with(cx, |store, _| store
                .cached_snapshots()
                .find(|snapshot| snapshot.provider_id.0 == RUST_COVERAGE_PROVIDER_ID)
                .expect("revoked Rust snapshot")
                .status),
            SourceCoverageStatus::Stale
        );
        assert_eq!(
            coverage
                .update(cx, |store, cx| store.snapshot(
                    SourceCoverageProviderId("ordinary-language".into()),
                    cx
                ))
                .await
                .expect("generic snapshot survives")
                .status,
            SourceCoverageStatus::Current
        );
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(
            !provider.read_with(cx, |store, cx| store.is_run_current(token, cx)),
            "re-enabling cannot revive an old run"
        );
        let fresh = provider
            .update(cx, |store, cx| store.begin_run(cx))
            .expect("fresh run");
        assert!(provider.read_with(cx, |store, cx| store.is_run_current(fresh, cx)));
    }

    #[gpui::test]
    async fn rust_coverage_provider_read_publication_split_reads_without_publication(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
        let fs = crate::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/workspace",
            serde_json::json!({
                "src": {"bin": {"fixture.rs": "fn main() {}"}},
                "target": {"zed-coverage.json": contents},
            }),
        )
        .await;
        let project = crate::Project::test(fs, ["/workspace".as_ref()], cx).await;
        let (provider, coverage, artifact) = project.read_with(cx, |project, cx| {
            (
                project.rust_coverage_provider_store().clone(),
                project.source_coverage_store().clone(),
                project
                    .find_project_path("/workspace/target/zed-coverage.json", cx)
                    .expect("coverage artifact"),
            )
        });
        let token = provider
            .update(cx, |provider, cx| provider.begin_run(cx))
            .expect("local publication admission");
        let parsed = provider
            .update(cx, |provider, cx| {
                provider.read_local_artifact_with_seal(token, artifact, contents.len(), None, cx)
            })
            .await
            .expect("actual bounded load and parse");
        assert_eq!(parsed.snapshot.status, SourceCoverageStatus::Current);
        assert_eq!(parsed.snapshot.files.len(), 1);
        coverage.read_with(cx, |coverage, _| {
            assert!(
                coverage.selected_snapshot().is_none(),
                "reading is not publishing"
            )
        });
        let published = provider
            .update(cx, |provider, cx| {
                provider.publish_local_artifact_for_run(
                    token,
                    contents.len(),
                    Task::ready(Ok(parsed)),
                    cx,
                )
            })
            .await
            .expect("current publication owns the final validation and state change");
        assert_eq!(published.status, SourceCoverageStatus::Current);
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(coverage.selected_snapshot(), Some(&published))
        });
        project.update(cx, |project, cx| project.close(cx));
    }

    #[gpui::test]
    async fn rust_coverage_provider_read_publication_split_revalidates_before_publication(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        for change in ["superseded", "cancelled", "session", "source", "report"] {
            let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
            let fs = crate::FakeFs::new(cx.executor());
            fs.insert_tree(
                "/workspace",
                serde_json::json!({
                    "src": {"bin": {"fixture.rs": "fn main() {}"}},
                    "target": {"zed-coverage.json": contents},
                }),
            )
            .await;
            let project = crate::Project::test(fs.clone(), ["/workspace".as_ref()], cx).await;
            let (provider, coverage, artifact, source) = project.read_with(cx, |project, cx| {
                (
                    project.rust_coverage_provider_store().clone(),
                    project.source_coverage_store().clone(),
                    project
                        .find_project_path("/workspace/target/zed-coverage.json", cx)
                        .expect("coverage artifact"),
                    project
                        .find_project_path("/workspace/src/bin/fixture.rs", cx)
                        .expect("reported source"),
                )
            });
            let baseline = provider
                .update(cx, |provider, cx| {
                    provider.interpret_artifact(artifact.clone(), contents.len(), cx)
                })
                .await
                .expect("previous current report");
            let token = provider
                .update(cx, |provider, cx| provider.begin_run(cx))
                .expect("new publication admission");
            let parsed = provider
                .update(cx, |provider, cx| {
                    provider.read_local_artifact_with_seal(
                        token,
                        artifact.clone(),
                        contents.len(),
                        None,
                        cx,
                    )
                })
                .await
                .expect("actual report is parsed before transition");
            assert_eq!(parsed.snapshot.files, baseline.files);
            coverage.read_with(cx, |coverage, _| {
                assert_eq!(coverage.selected_snapshot(), Some(&baseline), "{change}")
            });
            match change {
                "superseded" => {
                    provider
                        .update(cx, |provider, cx| provider.begin_run(cx))
                        .expect("newer publication owner");
                }
                "cancelled" => provider.update(cx, |provider, _| provider.cancel_run(token)),
                "session" => coverage.update(cx, |coverage, cx| {
                    coverage.invalidate(SourceCoverageStatus::Stale, cx)
                }),
                "source" => coverage.update(cx, |coverage, cx| {
                    coverage.invalidate_paths(std::slice::from_ref(&source), cx)
                }),
                "report" => {
                    fs.pause_events();
                    let changed = contents.replace("\"covered\": 3", "\"covered\": 2");
                    assert_ne!(changed, contents);
                    assert_eq!(changed.len(), contents.len());
                    fs.insert_file(
                        "/workspace/target/zed-coverage.json",
                        changed.as_bytes().to_vec(),
                    )
                    .await;
                    assert!(
                        fs.buffered_event_count() > 0,
                        "watcher delivery remains held"
                    );
                }
                _ => unreachable!("fixed publication transition"),
            }
            let result = provider
                .update(cx, |provider, cx| {
                    provider.publish_local_artifact_for_run(
                        token,
                        contents.len(),
                        Task::ready(Ok(parsed)),
                        cx,
                    )
                })
                .await;
            if change == "source" {
                assert!(
                    result
                        .expect_err("changed source cannot publish read facts")
                        .is::<crate::source_coverage::SourceCoverageSourceChanged>()
                );
            } else {
                assert!(result.is_err(), "{change} cannot publish read facts");
            }
            coverage.read_with(cx, |coverage, _| {
                let retained = coverage
                    .selected_snapshot()
                    .expect("retained previous report");
                assert_eq!(retained.generation, baseline.generation, "{change}");
                assert_eq!(retained.files, baseline.files, "{change}");
            });
            if change == "report" {
                fs.unpause_events_and_flush();
            }
            let recovered = provider
                .update(cx, |provider, cx| {
                    provider.interpret_artifact(artifact.clone(), contents.len(), cx)
                })
                .await
                .expect("fresh admission recovers after refused publication");
            assert_eq!(recovered.status, SourceCoverageStatus::Current);
            assert_eq!(recovered.files.len(), 1);
            assert!(recovered.generation > baseline.generation);
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[test]
    fn rust_coverage_provider_admission_rejects_obsolete_runs_without_cancelling_current() {
        let mut provider = RustCoverageArtifactProvider::new(7);
        let obsolete = provider.begin_run();
        let current = provider.begin_run();
        assert!(!provider.is_run_current(obsolete));
        assert!(provider.is_run_current(current));
        provider.cancel(obsolete);
        assert!(provider.is_run_current(current));
        let artifact = include_bytes!("../test_data/source_coverage/llvm3-run.json");
        assert!(
            provider
                .parse_artifact(obsolete, artifact, resolve)
                .is_err()
        );
        assert!(provider.parse_artifact(current, artifact, resolve).is_ok());
        provider.cancel(current);
        assert!(!provider.is_run_current(current));
        assert!(provider.parse_artifact(current, artifact, resolve).is_err());
    }

    #[gpui::test]
    async fn rust_coverage_provider_public_capture_drop_retires_only_owned_loading(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = crate::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/workspace",
            serde_json::json!({"src": {"fixture.rs": "fn main() {}", "unrelated.rs": "fn unrelated() {}"}, "target": {"other.json": "before"}}),
        )
        .await;
        let project = crate::Project::test(fs.clone(), ["/workspace".as_ref()], cx).await;
        let coverage = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let template = task::TaskTemplate {
            label: "capture API fixture".into(),
            command: "cargo".into(),
            args: vec![
                "llvm-cov".into(),
                "--output-path".into(),
                "target/coverage.json".into(),
            ],
            cwd: Some("/workspace".into()),
            env: collections::HashMap::from_iter([(
                "COVERAGE_EXPLICIT_ENV".into(),
                "unchanged".into(),
            )]),
            artifact_group: Some(task::TaskArtifactGroup {
                format: "llvm-cov-json-v1".into(),
                files: collections::BTreeMap::from_iter([(
                    "report".into(),
                    task::TaskArtifact {
                        path: "target/coverage.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 1024,
                    },
                )]),
            }),
            artifact_group_argument_binding: Some(task::TaskArtifactGroupArgumentBinding {
                role: "report".into(),
                argument_index: 2,
                related: Vec::new(),
            }),
            ..task::TaskTemplate::default()
        };
        let fixed = template
            .resolve_task("capture-api", &task::TaskContext::default())
            .expect("actual fixed-path template");
        let initial_snapshot =
            coverage.read_with(cx, |store, _| store.selected_snapshot().cloned());
        assert!(
            project
                .update(cx, |project, cx| project.begin_task_artifact_capture(
                    "00000000-0000-4000-8000-000000000001".into(),
                    &fixed.resolved,
                    fixed.resolved_artifact_group().expect("fixed report"),
                    cx
                ))
                .is_err(),
            "a successful unrelated fixed-path Task cannot claim capture ownership"
        );
        coverage.read_with(cx, |store, _| {
            assert_eq!(store.selected_snapshot(), initial_snapshot.as_ref())
        });
        let begin = |run_id: &str,
                     working_directory: Option<&str>,
                     cx: &mut gpui::TestAppContext| {
            let mut template = template.clone();
            template.cwd = working_directory.map(str::to_owned);
            let task = template
                .resolve_task("capture-api", &task::TaskContext::default())
                .expect("actual template resolution")
                .bind_artifact_invocation(run_id)
                .expect("actual checked UUID binding");
            let group = task
                .resolved_artifact_group()
                .expect("coverage group")
                .clone();
            let capture = project
                .update(cx, |project, cx| {
                    project.begin_task_artifact_capture(run_id.into(), &task.resolved, &group, cx)
                })
                .expect("public capture admission")
                .expect("coverage intent");
            (capture, task.resolved, group)
        };
        let (old, _, _) = begin(
            "00000000-0000-4000-8000-000000000001",
            Some("/workspace"),
            cx,
        );
        let (new, _, _) = begin(
            "00000000-0000-4000-8000-000000000002",
            Some("/workspace"),
            cx,
        );
        let before = coverage.read_with(cx, |store, _| {
            store.selected_snapshot().expect("new Loading").clone()
        });
        assert_eq!(before.status, SourceCoverageStatus::Loading);
        drop(old);
        cx.run_until_parked();
        coverage.read_with(cx, |store, _| {
            assert_eq!(store.selected_snapshot(), Some(&before))
        });
        drop(new);
        cx.run_until_parked();
        coverage.read_with(cx, |store, _| {
            let snapshot = store
                .selected_snapshot()
                .expect("owned cancellation publication");
            assert_eq!(snapshot.status, SourceCoverageStatus::Cancelled);
            assert_eq!(snapshot.generation, before.generation);
            assert!(snapshot.files.is_empty());
        });
        let (mut observed_capture, observed_launch, _) = begin(
            "00000000-0000-4000-8000-000000000008",
            Some("/workspace"),
            cx,
        );
        project
            .read_with(cx, |project, cx| {
                project.admit_task_artifact_capture(&mut observed_capture, &observed_launch, cx)
            })
            .expect("admit observed cancellation");
        let observed = project
            .update(cx, |project, cx| {
                project.observe_task_artifact_capture(observed_capture, gpui::Task::ready(None), cx)
            })
            .await;
        assert!(
            observed.is_none(),
            "canonical cancellation result stays absent"
        );
        cx.run_until_parked();
        coverage.read_with(cx, |store, _| {
            let cancelled = store.selected_snapshot().expect("observed cancellation");
            assert_eq!(cancelled.status, SourceCoverageStatus::Cancelled);
            assert!(cancelled.generation > before.generation);
            assert!(cancelled.files.is_empty());
        });
        let mut serial = 32u64;
        for change in [
            "report",
            "source",
            "unrelated",
            "non-output",
            "edit-restore",
        ] {
            for observed_none in [false, true] {
                serial += 1;
                let run_id = format!("00000000-0000-4000-8000-{serial:012x}");
                let (mut capture, launch, group) = begin(&run_id, Some("/workspace"), cx);
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &launch, cx)
                    })
                    .expect("actual immutable report admission");
                let report = group.files.get("report").expect("single admitted report");
                let output = project.read_with(cx, |project, cx| {
                    project
                        .find_project_path(Path::new("/workspace").join(&report.path), cx)
                        .expect("actual bound output ProjectPath")
                });
                let admitted = coverage.read_with(cx, |coverage, _| {
                    coverage.request_version().expect("admitted source epoch")
                });
                fs.pause_events();
                fs.insert_file(
                    Path::new("/workspace").join(&report.path),
                    b"actual report output".to_vec(),
                )
                .await;
                coverage.update(cx, |coverage, cx| {
                    coverage.invalidate_paths(std::slice::from_ref(&output), cx)
                });
                if matches!(change, "source" | "unrelated" | "edit-restore") {
                    let source = project.read_with(cx, |project, cx| {
                        project
                            .find_project_path(
                                if change == "unrelated" {
                                    "/workspace/src/unrelated.rs"
                                } else {
                                    "/workspace/src/fixture.rs"
                                },
                                cx,
                            )
                            .expect("indexed actual source")
                    });
                    let buffer = project
                        .update(cx, |project, cx| project.open_buffer(source, cx))
                        .await
                        .expect("actual source Buffer");
                    buffer.update(cx, |buffer, cx| {
                        let original = buffer.text();
                        assert!(buffer.edit([(0..0, "// held edit\n")], None, cx).is_some());
                        if change == "edit-restore" {
                            assert!(buffer.edit([(0..13, "")], None, cx).is_some());
                            assert_eq!(buffer.text(), original);
                        }
                    });
                } else if change == "non-output" {
                    let other = project.read_with(cx, |project, cx| {
                        project
                            .find_project_path("/workspace/target/other.json", cx)
                            .expect("non-output artifact path")
                    });
                    coverage.update(cx, |coverage, cx| {
                        coverage.invalidate_paths(std::slice::from_ref(&other), cx)
                    });
                }
                cx.run_until_parked();
                coverage.read_with(cx, |coverage, _| {
                    assert_ne!(
                        coverage.request_version().expect("changed journal"),
                        admitted
                    );
                });
                if observed_none {
                    assert!(
                        project
                            .update(cx, |project, cx| {
                                project.observe_task_artifact_capture(
                                    capture,
                                    Task::ready(None),
                                    cx,
                                )
                            })
                            .await
                            .is_none()
                    );
                } else {
                    drop(capture);
                }
                cx.run_until_parked();
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage.selected_snapshot().expect("owned terminal state");
                    assert_eq!(
                        snapshot.status,
                        if change == "report" {
                            SourceCoverageStatus::Cancelled
                        } else {
                            SourceCoverageStatus::Stale
                        },
                        "{change}/{observed_none}"
                    );
                    assert!(snapshot.files.is_empty());
                });
                fs.unpause_events_and_flush();
                cx.run_until_parked();
            }
        }
        let provider = project.read_with(cx, |project, _| {
            project.rust_coverage_provider_store().clone()
        });
        let token = provider
            .update(cx, |provider, cx| provider.begin_task_capture_run(cx))
            .expect("public token-only admission");
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(
                coverage
                    .selected_snapshot()
                    .expect("new token-only Loading")
                    .status,
                SourceCoverageStatus::Loading
            );
        });
        let source = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/workspace/target/other.json", cx)
                .expect("journal path for strict public cancellation")
        });
        coverage.update(cx, |coverage, cx| {
            coverage.invalidate_paths(std::slice::from_ref(&source), cx)
        });
        provider.update(cx, |provider, cx| provider.finish_cancelled_run(token, cx));
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(
                coverage
                    .selected_snapshot()
                    .expect("strict token state")
                    .status,
                SourceCoverageStatus::Stale
            );
        });
        let (mut changed_environment_capture, mut changed_environment_launch, _) = begin(
            "00000000-0000-4000-8000-000000000005",
            Some("/workspace"),
            cx,
        );
        coverage.read_with(cx, |store, _| {
            let recovered = store.selected_snapshot().expect("fresh capture Loading");
            assert_eq!(recovered.status, SourceCoverageStatus::Loading);
            assert!(recovered.generation > before.generation);
            assert!(recovered.files.is_empty());
        });
        changed_environment_launch
            .env
            .insert("COVERAGE_EXPLICIT_ENV".into(), "changed".into());
        assert!(
            project
                .read_with(cx, |project, cx| project.admit_task_artifact_capture(
                    &mut changed_environment_capture,
                    &changed_environment_launch,
                    cx
                ))
                .is_err(),
            "post-save admission cannot substitute explicit environment overrides"
        );
        drop(changed_environment_capture);
        cx.run_until_parked();
        let (mut changed, mut changed_launch, _) = begin(
            "00000000-0000-4000-8000-000000000003",
            Some("/workspace"),
            cx,
        );
        changed_launch.args.push("--unexpected".into());
        assert!(
            project
                .read_with(cx, |project, cx| project.admit_task_artifact_capture(
                    &mut changed,
                    &changed_launch,
                    cx
                ))
                .is_err()
        );
        drop(changed);
        cx.run_until_parked();
        coverage.read_with(cx, |store, _| {
            assert_eq!(
                store
                    .selected_snapshot()
                    .expect("rejected admission")
                    .status,
                SourceCoverageStatus::Cancelled
            )
        });
        let (mut capture, mut deferred_cwd, _) =
            begin("00000000-0000-4000-8000-000000000004", None, cx);
        deferred_cwd.cwd = Some("/workspace".into());
        project
            .read_with(cx, |project, cx| {
                project.admit_task_artifact_capture(&mut capture, &deferred_cwd, cx)
            })
            .expect("post-save scheduler freezes actual cwd");
        project
            .read_with(cx, |project, cx| {
                project.revalidate_task_artifact_capture(&capture, &deferred_cwd, cx)
            })
            .expect("unchanged admitted invocation remains eligible without resealing");
        let mut changed_environment = deferred_cwd.clone();
        changed_environment
            .env
            .insert("COVERAGE_EXPLICIT_ENV".into(), "changed".into());
        assert!(
            project
                .read_with(cx, |project, cx| project.revalidate_task_artifact_capture(
                    &capture,
                    &changed_environment,
                    cx
                ))
                .is_err(),
            "queued capture cannot substitute explicit environment overrides"
        );
        let mut changed_invocation = deferred_cwd.clone();
        changed_invocation
            .args
            .push("--changed-after-admission".into());
        assert!(
            project
                .read_with(cx, |project, cx| project.revalidate_task_artifact_capture(
                    &capture,
                    &changed_invocation,
                    cx
                ))
                .is_err(),
            "actual eligibility cannot substitute admitted argv"
        );
        changed_invocation
            .args
            .pop()
            .expect("remove changed argument");
        changed_invocation.cwd = Some("/workspace/src".into());
        assert!(
            project
                .read_with(cx, |project, cx| project.revalidate_task_artifact_capture(
                    &capture,
                    &changed_invocation,
                    cx
                ))
                .is_err(),
            "actual eligibility cannot retarget admitted cwd"
        );
        let source = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/workspace/src/fixture.rs", cx)
                .expect("indexed source")
        });
        coverage.update(cx, |coverage, cx| {
            coverage.invalidate_paths(std::slice::from_ref(&source), cx)
        });
        assert!(
            project.read_with(cx, |_, cx| capture.is_current(cx)),
            "a source journal change alone preserves the provider/session identity"
        );
        let error = project
            .read_with(cx, |project, cx| {
                project.revalidate_task_artifact_capture(&capture, &deferred_cwd, cx)
            })
            .expect_err("changed indexed source denies launch without resealing");
        assert!(error.is::<crate::source_coverage::SourceCoverageSourceChanged>());
        assert!(
            project
                .read_with(cx, |project, cx| project.admit_task_artifact_capture(
                    &mut capture,
                    &deferred_cwd,
                    cx
                ))
                .is_err(),
            "immutable admitted context cannot be resealed"
        );
        drop(capture);
        cx.run_until_parked();
        coverage.read_with(cx, |store, _| {
            assert_eq!(
                store.selected_snapshot().expect("admitted drop").status,
                SourceCoverageStatus::Stale
            )
        });
        for (run_id, status) in [
            (
                "00000000-0000-4000-8000-000000000006",
                SourceCoverageStatus::Restricted,
            ),
            (
                "00000000-0000-4000-8000-000000000007",
                SourceCoverageStatus::Disconnected,
            ),
        ] {
            let (capture, _, _) = begin(run_id, Some("/workspace"), cx);
            coverage
                .update(cx, |store, cx| {
                    store.mark_provider_status(
                        SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.into()),
                        status,
                        None,
                        cx,
                    )
                })
                .expect("lifecycle status before cancellation");
            let lifecycle = coverage.read_with(cx, |store, _| {
                store
                    .selected_snapshot()
                    .expect("lifecycle snapshot")
                    .clone()
            });
            drop(capture);
            cx.run_until_parked();
            coverage.read_with(cx, |store, _| {
                assert_eq!(
                    store.selected_snapshot(),
                    Some(&lifecycle),
                    "capture cancellation must preserve restricted/disconnected precedence"
                );
            });
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_public_capture_expiry_precedes_process_outcome_and_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let statuses = cx.background_spawn(async {
            let successful = smol::process::Command::new("/bin/sh")
                .args(["-c", "exit 0"])
                .status()
                .await?;
            let failed = smol::process::Command::new("/bin/sh")
                .args(["-c", "exit 7"])
                .status()
                .await?;
            Ok::<_, anyhow::Error>((successful, failed))
        });
        let (successful, failed) = match futures::future::select(
            statuses,
            cx.background_executor
                .timer(std::time::Duration::from_secs(10)),
        )
        .await
        {
            futures::future::Either::Left((result, _)) => result.expect("actual child statuses"),
            futures::future::Either::Right(_) => panic!("controlled child statuses timed out"),
        };
        let run_counter = std::cell::Cell::new(0_u32);
        for expiry in [
            "session",
            "journal-overflow",
            "obsolete",
            "restricted",
            "disconnected",
            "post-parse",
        ] {
            for outcome in ["success", "failure", "spawn-error", "none"] {
                if expiry == "post-parse" && outcome != "success" {
                    continue;
                }
                let filesystem = crate::FakeFs::new(cx.executor());
                filesystem
                    .insert_tree(
                        "/workspace",
                        serde_json::json!({
                            "src": {"bin": {"fixture.rs": "fn main() {}"}}, "target": {}
                        }),
                    )
                    .await;
                let project =
                    crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
                let (provider, coverage, worktree_id) = project.read_with(cx, |project, cx| {
                    (
                        project.rust_coverage_provider_store().clone(),
                        project.source_coverage_store().clone(),
                        project
                            .find_project_path("/workspace/src/bin/fixture.rs", cx)
                            .expect("actual source")
                            .worktree_id,
                    )
                });
                let template = task::TaskTemplate {
                    label: "capture expiry fixture".into(),
                    command: "cargo".into(),
                    args: vec![
                        "llvm-cov".into(),
                        "--output-path".into(),
                        "target/coverage.json".into(),
                    ],
                    cwd: Some("/workspace".into()),
                    artifact_group: Some(task::TaskArtifactGroup {
                        format: "llvm-cov-json-v1".into(),
                        files: collections::BTreeMap::from_iter([(
                            "report".into(),
                            task::TaskArtifact {
                                path: "target/coverage.json".into(),
                                kind: task::TaskArtifactKind::Data,
                                max_bytes: MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64,
                            },
                        )]),
                    }),
                    artifact_group_argument_binding: Some(task::TaskArtifactGroupArgumentBinding {
                        role: "report".into(),
                        argument_index: 2,
                        related: Vec::new(),
                    }),
                    ..Default::default()
                };
                let begin = |cx: &mut gpui::TestAppContext| {
                    let serial = run_counter
                        .get()
                        .checked_add(1)
                        .expect("bounded fixture run count");
                    run_counter.set(serial);
                    let run_id = format!("00000000-0000-4000-8000-{serial:012x}");
                    let task = template
                        .resolve_task("capture-expiry", &task::TaskContext::default())
                        .expect("declarative Task")
                        .bind_artifact_invocation(&run_id)
                        .expect("scheduler UUID binding");
                    let group = task.resolved_artifact_group().expect("report group");
                    let mut capture = project
                        .update(cx, |project, cx| {
                            project.begin_task_artifact_capture(run_id, &task.resolved, group, cx)
                        })
                        .expect("public capture owner")
                        .expect("supported coverage group");
                    project
                        .read_with(cx, |project, cx| {
                            project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                        })
                        .expect("genuine post-save public admission");
                    (capture, task)
                };
                let (capture, task) = begin(cx);
                let parse_gate = if expiry == "post-parse" {
                    let path = &task
                        .resolved_artifact_group()
                        .expect("report")
                        .files
                        .get("report")
                        .expect("role")
                        .path;
                    filesystem
                        .insert_file(
                            std::path::Path::new("/workspace").join(path),
                            include_bytes!("../test_data/source_coverage/llvm3-run.json").to_vec(),
                        )
                        .await;
                    Some(
                        provider
                            .update(cx, |provider, _| provider.gate_next_local_parse_for_test()),
                    )
                } else {
                    None
                };
                let newer = if expiry == "obsolete" {
                    Some(begin(cx).0)
                } else {
                    None
                };
                let expire = |cx: &mut gpui::TestAppContext| {
                    coverage.update(cx, |coverage, cx| {
                        if expiry == "journal-overflow" {
                            let changes = (0..=MAX_SOURCE_COVERAGE_FILES)
                                .map(|index| ProjectPath {
                                    worktree_id,
                                    path: Arc::from(
                                        RelPath::from_unix_str(&format!("journal-{index}.rs"))
                                            .expect("journal path"),
                                    ),
                                })
                                .collect::<Vec<_>>();
                            coverage.invalidate_paths(&changes, cx);
                        } else {
                            coverage.invalidate(SourceCoverageStatus::Stale, cx);
                        }
                        if matches!(expiry, "restricted" | "disconnected") {
                            coverage
                                .mark_provider_status(
                                    SourceCoverageProviderId(RUST_COVERAGE_PROVIDER_ID.into()),
                                    if expiry == "restricted" {
                                        SourceCoverageStatus::Restricted
                                    } else {
                                        SourceCoverageStatus::Disconnected
                                    },
                                    Some("retained exact lifecycle snapshot".into()),
                                    cx,
                                )
                                .expect("actual retained lifecycle state");
                        }
                    });
                };
                if !matches!(expiry, "obsolete" | "post-parse") {
                    expire(cx);
                }
                if expiry != "post-parse" {
                    assert!(
                        project.read_with(cx, |_, cx| !capture.is_current(cx)),
                        "actual source-session/request version or run replacement must expire admission"
                    );
                }
                let retained = coverage.read_with(cx, |coverage, _| {
                    coverage
                        .selected_snapshot()
                        .expect("owned Loading or lifecycle snapshot")
                        .clone()
                });
                let result = match outcome {
                    "success" => Some(Ok(successful)),
                    "failure" => Some(Ok(failed)),
                    "spawn-error" => Some(Err(anyhow::anyhow!("actual Task launch failed"))),
                    "none" => None,
                    _ => unreachable!("fixed process outcomes"),
                };
                let observed = project.update(cx, |project, cx| {
                    project.observe_task_artifact_capture(capture, Task::ready(result), cx)
                });
                if let Some((started, release)) = parse_gate {
                    assert!(
                        matches!(
                            futures::future::select(
                                started,
                                cx.background_executor
                                    .timer(std::time::Duration::from_secs(10))
                            )
                            .await,
                            futures::future::Either::Left((Ok(()), _))
                        ),
                        "actual report must parse before held session expiry"
                    );
                    expire(cx);
                    release.send(()).expect("release actual held parser");
                }
                let observed = observed.await;
                match outcome {
                    "success" => assert_eq!(
                        observed
                            .expect("same actual successful exit")
                            .expect("no status rewrite"),
                        successful
                    ),
                    "failure" => assert_eq!(
                        observed
                            .expect("same actual failed exit")
                            .expect("no status rewrite"),
                        failed
                    ),
                    "spawn-error" => assert_eq!(
                        observed
                            .expect("same launch error")
                            .expect_err("launch stays failed")
                            .to_string(),
                        "actual Task launch failed"
                    ),
                    "none" => assert!(observed.is_none()),
                    _ => unreachable!("fixed process outcomes"),
                }
                cx.run_until_parked();
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage
                        .selected_snapshot()
                        .expect("retained terminal snapshot");
                    if matches!(expiry, "obsolete" | "restricted" | "disconnected") {
                        assert_eq!(
                            snapshot, &retained,
                            "obsolete result cannot mutate newer or protected lifecycle snapshot"
                        );
                    } else {
                        assert_eq!(
                            snapshot.status,
                            SourceCoverageStatus::Stale,
                            "{expiry}/{outcome}"
                        );
                        assert_eq!(
                            snapshot.diagnostic.as_deref(),
                            Some(COVERAGE_TASK_EXPIRED_DIAGNOSTIC)
                        );
                        assert!(snapshot.files.is_empty());
                    }
                });
                drop(newer);
                cx.run_until_parked();
            }
        }
    }

    #[gpui::test]
    async fn rust_coverage_provider_task_admission_preserves_source_and_session_freshness(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = crate::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/workspace",
            serde_json::json!({
                "src": {"bin": {"fixture.rs": "fn main() {}"}},
                "target": {"zed-coverage.json": include_str!("../test_data/source_coverage/llvm3-run.json")}
            }),
        )
        .await;
        let project = crate::Project::test(fs, ["/workspace".as_ref()], cx).await;
        let (provider, coverage, artifact, source) = project.read_with(cx, |project, cx| {
            (
                project.rust_coverage_provider_store().clone(),
                project.source_coverage_store().clone(),
                project
                    .find_project_path("/workspace/target/zed-coverage.json", cx)
                    .expect("coverage artifact"),
                project
                    .find_project_path("/workspace/src/bin/fixture.rs", cx)
                    .expect("coverage source"),
            )
        });
        let token = provider
            .update(cx, |provider, cx| provider.begin_run(cx))
            .expect("Task admission");
        coverage.update(cx, |coverage, cx| {
            coverage.invalidate_paths(std::slice::from_ref(&source), cx)
        });
        let interpretation = provider.update(cx, |provider, cx| {
            provider.interpret_artifact_for_run(
                token,
                artifact.clone(),
                MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        assert!(
            interpretation.await.is_err(),
            "a changed reported source invalidates Task admission"
        );
        coverage.read_with(cx, |coverage, _| {
            assert!(coverage.selected_snapshot().is_none())
        });

        let token = provider
            .update(cx, |provider, cx| provider.begin_run(cx))
            .expect("fresh Task admission");
        coverage.update(cx, |coverage, cx| {
            coverage.invalidate_paths(std::slice::from_ref(&artifact), cx)
        });
        let interpretation = provider.update(cx, |provider, cx| {
            provider.interpret_artifact_for_run(
                token,
                artifact.clone(),
                MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        assert_eq!(
            interpretation
                .await
                .expect("artifact refresh does not change covered source")
                .status,
            SourceCoverageStatus::Current
        );

        let token = provider
            .update(cx, |provider, cx| provider.begin_run(cx))
            .expect("session-bound Task admission");
        coverage.update(cx, |coverage, cx| {
            coverage.invalidate(SourceCoverageStatus::Stale, cx)
        });
        provider.read_with(cx, |provider, cx| {
            assert!(!provider.is_run_current(token, cx))
        });
        let interpretation = provider.update(cx, |provider, cx| {
            provider.interpret_artifact_for_run(
                token,
                artifact.clone(),
                MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        assert!(
            interpretation.await.is_err(),
            "an expired session cannot begin artifact interpretation"
        );

        let token = provider
            .update(cx, |provider, cx| provider.begin_run(cx))
            .expect("admission before close");
        project.update(cx, |project, cx| project.close(cx));
        provider.read_with(cx, |provider, cx| {
            assert!(!provider.is_run_current(token, cx))
        });
        assert!(
            provider
                .update(cx, |provider, cx| provider.begin_run(cx))
                .is_err()
        );
        coverage.read_with(cx, |coverage, _| {
            assert_eq!(
                coverage
                    .selected_snapshot()
                    .expect("retained closed snapshot")
                    .status,
                SourceCoverageStatus::Disconnected
            )
        });
    }

    fn mapped_baseline_fixture_task(report_bytes: usize) -> (String, task::ResolvedTask) {
        let run_id = uuid::Uuid::new_v4().to_string();
        let task = task::TaskTemplate {
            label: "baseline admission boundary".into(),
            command: "cargo".into(),
            args: vec![
                "llvm-cov".into(),
                "--output-path".into(),
                "report.json".into(),
            ],
            cwd: Some("/workspace".into()),
            artifact_group: Some(task::TaskArtifactGroup {
                format: "llvm-cov-json-v1".into(),
                files: collections::BTreeMap::from_iter([(
                    "report".into(),
                    task::TaskArtifact {
                        path: "report.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: report_bytes as u64,
                    },
                )]),
            }),
            artifact_group_argument_binding: Some(task::TaskArtifactGroupArgumentBinding {
                role: "report".into(),
                argument_index: 2,
                related: Vec::new(),
            }),
            ..task::TaskTemplate::default()
        }
        .resolve_task("baseline-boundary", &task::TaskContext::default())
        .expect("actual coverage template")
        .bind_artifact_invocation(&run_id)
        .expect("actual unique bound report");
        (run_id, task)
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_public_source_epoch_precedes_unsuccessful_outcome(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let unavailable = tempfile::TempDir::new().expect("owned missing-program parent");
        let missing_program = unavailable.path().join("unavailable-coverage-child");
        assert!(!missing_program.exists());
        for change in [None, Some(true), Some(false)] {
            let mapped = change.unwrap_or(true);
            for spawn_error in [false, true] {
                let filesystem = crate::FakeFs::new(cx.executor());
                filesystem
                    .insert_tree(
                        "/workspace",
                        serde_json::json!({"src": {"bin": {
                            "fixture.rs": "fn main() {}",
                            "unrelated.rs": "fn unrelated() {}"
                        }}}),
                    )
                    .await;
                let project = crate::Project::test(filesystem, ["/workspace".as_ref()], cx).await;
                let (coverage, changed_path) = project.read_with(cx, |project, cx| {
                    (
                        project.source_coverage_store().clone(),
                        project
                            .find_project_path(
                                if mapped {
                                    "/workspace/src/bin/fixture.rs"
                                } else {
                                    "/workspace/src/bin/unrelated.rs"
                                },
                                cx,
                            )
                            .expect("actual visible source"),
                    )
                });
                let changed_buffer = project
                    .update(cx, |project, cx| project.open_buffer(changed_path, cx))
                    .await
                    .expect("admitted source Buffer");
                let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
                let (run_id, task) = mapped_baseline_fixture_task(contents.len());
                let mut capture = project
                    .update(cx, |project, cx| {
                        project.begin_task_artifact_capture(
                            run_id,
                            &task.resolved,
                            task.resolved_artifact_group().expect("UUID report binding"),
                            cx,
                        )
                    })
                    .expect("public begin")
                    .expect("capture owner");
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                    })
                    .expect("post-save public admission");
                let capture = project
                    .update(cx, |project, cx| {
                        project.prepare_task_artifact_capture_sources(capture, cx)
                    })
                    .expect("source preparation")
                    .await
                    .expect("retained baseline");
                let admitted = coverage.read_with(cx, |coverage, _| {
                    coverage.request_version().expect("admitted journal")
                });
                if change.is_some() {
                    changed_buffer.update(cx, |buffer, cx| {
                        assert!(
                            buffer
                                .edit([(0..0, "// actual held Buffer edit\n")], None, cx)
                                .is_some()
                        );
                    });
                }
                cx.run_until_parked();
                let changed = coverage.read_with(cx, |coverage, _| {
                    coverage.request_version().expect("changed journal")
                });
                assert_eq!(
                    changed == admitted,
                    change.is_none(),
                    "only the actual edit changes the source journal"
                );
                assert!(
                    project.read_with(cx, |_, cx| capture.is_current(cx)),
                    "one path edit does not manufacture session expiry"
                );
                let outcome = if spawn_error {
                    let missing_program = missing_program.clone();
                    let actual = cx
                        .background_spawn(async move {
                            smol::process::Command::new(missing_program).status().await
                        })
                        .await
                        .expect_err("actual OS spawn failure");
                    Err(anyhow::Error::from(actual))
                } else {
                    let actual = cx
                        .background_spawn(async {
                            smol::process::Command::new("/bin/sh")
                                .args(["-c", "exit 7"])
                                .status()
                                .await
                        })
                        .await
                        .expect("actual failed child status");
                    assert_eq!(actual.code(), Some(7));
                    Ok(actual)
                };
                let expected_error = outcome.as_ref().err().map(ToString::to_string);
                let expected_exit = outcome.as_ref().ok().copied();
                let observed = project
                    .update(cx, |project, cx| {
                        project.observe_task_artifact_capture(
                            capture,
                            Task::ready(Some(outcome)),
                            cx,
                        )
                    })
                    .await
                    .expect("same terminal outcome");
                match observed {
                    Ok(status) => assert_eq!(Some(status), expected_exit),
                    Err(error) => assert_eq!(Some(error.to_string()), expected_error),
                }
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage
                        .selected_snapshot()
                        .expect("terminal coverage state");
                    assert_eq!(
                        snapshot.status,
                        if change.is_some() {
                            SourceCoverageStatus::Stale
                        } else {
                            SourceCoverageStatus::Error
                        },
                        "change={change:?}, spawn_error={spawn_error}"
                    );
                    assert!(snapshot.files.is_empty());
                });
                drop(changed_buffer);
                project.update(cx, |project, cx| project.close(cx));
                cx.run_until_parked();
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_mapped_source_baseline_requires_readable_admitted_sources(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let actual_status = cx
            .background_spawn(async {
                smol::process::Command::new("/bin/sh")
                    .args(["-c", "exit 0"])
                    .status()
                    .await
            })
            .await
            .expect("actual controlled child exit");
        for boundary in ["missing", "symlink", "unadmitted-ignored"] {
            let filesystem = crate::FakeFs::new(cx.executor());
            filesystem
                .insert_tree(
                    "/workspace",
                    serde_json::json!({
                        ".git": {}, ".gitignore": "target/\n",
                        "src": {"bin": {"fixture.rs": "fn main() {}"}},
                        "target": {"included.rs": "fn main() {}"}
                    }),
                )
                .await;
            filesystem
                .insert_tree("/outside", serde_json::json!({"input.rs": "private input"}))
                .await;
            let project =
                crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
            let (coverage, worktree) = project.read_with(cx, |project, cx| {
                (
                    project.source_coverage_store().clone(),
                    project
                        .worktrees(cx)
                        .next()
                        .expect("actual source worktree"),
                )
            });
            let source_relative = if boundary == "unadmitted-ignored" {
                "target/included.rs"
            } else {
                "src/bin/fixture.rs"
            };
            let relative = util::rel_path::RelPath::from_unix_str(source_relative)
                .expect("source relative path")
                .into_arc();
            let entry = worktree
                .update(cx, |worktree, cx| {
                    worktree
                        .as_local()
                        .expect("actual local source owner")
                        .refresh_entry(relative.clone(), None, cx)
                })
                .await
                .expect("actual entry refresh")
                .expect("actual source entry");
            assert_eq!(entry.is_ignored, boundary == "unadmitted-ignored");
            let contents = include_str!("../test_data/source_coverage/llvm3-run.json")
                .replace("src/bin/fixture.rs", source_relative);
            let source_path = std::path::Path::new("/workspace").join(source_relative);
            filesystem.pause_events();
            if boundary != "unadmitted-ignored" {
                filesystem
                    .remove_file(&source_path, fs::RemoveOptions::default())
                    .await
                    .expect("actual source removed before baseline read");
                if boundary == "symlink" {
                    filesystem
                        .insert_symlink(&source_path, "/outside/input.rs".into())
                        .await;
                }
            }
            for recovery in [false, true] {
                if recovery {
                    if boundary == "unadmitted-ignored" {
                        filesystem
                            .write(Path::new("/workspace/.gitignore"), b"")
                            .await
                            .expect("remove source ignore rule");
                    } else {
                        if boundary == "symlink" {
                            filesystem
                                .rename(
                                    &source_path,
                                    Path::new("/outside/retired-input-link"),
                                    fs::RenameOptions::default(),
                                )
                                .await
                                .expect("retire actual symlink without following its target");
                            assert_eq!(
                                filesystem
                                    .read_link(Path::new("/outside/retired-input-link"))
                                    .await
                                    .expect("retired link identity"),
                                Path::new("/outside/input.rs")
                            );
                        }
                        filesystem
                            .insert_file(&source_path, b"fn main() {}".to_vec())
                            .await;
                    }
                    let metadata = filesystem
                        .metadata(&source_path)
                        .await
                        .expect("recovered source metadata")
                        .expect("recovered source");
                    filesystem.unpause_events_and_flush();
                    if boundary == "unadmitted-ignored" {
                        let ignore_path = util::rel_path::RelPath::from_unix_str(".gitignore")
                            .expect("actual ignore rule path")
                            .into_arc();
                        worktree
                            .update(cx, |worktree, cx| {
                                worktree
                                    .as_local()
                                    .expect("actual local ignore owner")
                                    .refresh_entry(ignore_path, None, cx)
                            })
                            .await
                            .expect("actual updated ignore rule refresh")
                            .expect("actual ignore rule remains");
                    }
                    worktree
                        .update(cx, |worktree, cx| {
                            worktree
                                .as_local()
                                .expect("actual local source owner")
                                .refresh_entry(relative.clone(), None, cx)
                        })
                        .await
                        .expect("actual recovered source refresh")
                        .expect("actual recovered source entry");
                    cx.condition(&worktree, |worktree, _| {
                        worktree.entry_for_path(&relative).is_some_and(|entry| {
                            entry.inode == metadata.inode
                                && entry.mtime == Some(metadata.mtime)
                                && !entry.is_ignored
                        })
                    })
                    .await;
                    filesystem.pause_events();
                }
                let (run_id, task) = mapped_baseline_fixture_task(contents.len());
                let group = task.resolved_artifact_group().expect("actual group");
                let report = group.files.get("report").expect("declared exact report");
                let mut capture = project
                    .update(cx, |project, cx| {
                        project.begin_task_artifact_capture(run_id, &task.resolved, group, cx)
                    })
                    .expect("actual public begin")
                    .expect("coverage intent");
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                    })
                    .expect("public post-save admission");
                let capture = project.update(cx, |project, cx|
                    project.prepare_task_artifact_capture_sources(capture, cx))
                    .expect("public baseline preparation").await.expect("baseline records unreadable sources without classifying all files as source");
                filesystem
                    .insert_file(
                        std::path::Path::new("/workspace").join(&report.path),
                        contents.as_bytes().to_vec(),
                    )
                    .await;
                let observed = project
                    .update(cx, |project, cx| {
                        project.observe_task_artifact_capture(
                            capture,
                            Task::ready(Some(Ok(actual_status))),
                            cx,
                        )
                    })
                    .await;
                assert_eq!(
                    observed
                        .expect("unchanged actual completion")
                        .expect("same successful exit"),
                    actual_status
                );
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage.selected_snapshot().expect("owned report state");
                    assert_eq!(
                        snapshot.status,
                        if recovery {
                            SourceCoverageStatus::Current
                        } else {
                            SourceCoverageStatus::Stale
                        },
                        "{boundary}"
                    );
                    assert_eq!(snapshot.files.len(), usize::from(recovery), "{boundary}");
                });
            }
            filesystem.unpause_events_and_flush();
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_mapped_source_baseline_admits_large_workspaces_and_rejects_mapped_drift(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
        let actual_status = cx
            .background_spawn(async {
                smol::process::Command::new("/bin/sh")
                    .args(["-c", "exit 0"])
                    .status()
                    .await
            })
            .await
            .expect("actual controlled child exit");
        for scale in ["files", "bytes"] {
            let filesystem = crate::FakeFs::new(cx.executor());
            filesystem
                .insert_tree(
                    "/workspace",
                    serde_json::json!({"src": {"bin": {"fixture.rs": "fn main() {}"}}}),
                )
                .await;
            if scale == "files" {
                for index in 0..=MAX_SOURCE_COVERAGE_FILES {
                    filesystem
                        .insert_file(format!("/workspace/padding-{index}"), Vec::new())
                        .await;
                }
            } else {
                filesystem
                    .insert_file(
                        "/workspace/unreported-asset",
                        vec![b'x'; MAX_RUST_COVERAGE_ARTIFACT_BYTES + 1],
                    )
                    .await;
            }
            let project =
                crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
            let (coverage, worktree) = project.read_with(cx, |project, cx| {
                (
                    project.source_coverage_store().clone(),
                    project.worktrees(cx).next().expect("actual large worktree"),
                )
            });
            if scale == "files" {
                worktree.read_with(cx, |worktree, _| {
                    assert!(
                        worktree.snapshot().files(false, 0).count() > MAX_SOURCE_COVERAGE_FILES
                    );
                });
            } else {
                assert!(
                    filesystem
                        .metadata(Path::new("/workspace/unreported-asset"))
                        .await
                        .expect("actual unreported asset metadata")
                        .expect("actual asset")
                        .len
                        > MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64
                );
            }
            let source_path = Path::new("/workspace/src/bin/fixture.rs");
            let source_relative = util::rel_path::RelPath::from_unix_str("src/bin/fixture.rs")
                .expect("mapped source path")
                .into_arc();
            for phase in ["current", "changed", "fresh"] {
                if phase == "fresh" {
                    let metadata = filesystem
                        .metadata(source_path)
                        .await
                        .expect("actual changed source metadata")
                        .expect("actual source");
                    filesystem.unpause_events_and_flush();
                    worktree
                        .update(cx, |worktree, cx| {
                            worktree
                                .as_local()
                                .expect("actual local source owner")
                                .refresh_entry(source_relative.clone(), None, cx)
                        })
                        .await
                        .expect("actual fresh mapped source refresh")
                        .expect("actual mapped source remains");
                    cx.condition(&worktree, |worktree, _| {
                        worktree
                            .entry_for_path(&source_relative)
                            .is_some_and(|entry| {
                                entry.inode == metadata.inode && entry.mtime == Some(metadata.mtime)
                            })
                    })
                    .await;
                }
                let (run_id, task) = mapped_baseline_fixture_task(contents.len());
                let group = task.resolved_artifact_group().expect("actual bound group");
                let mut capture = project
                    .update(cx, |project, cx| {
                        project.begin_task_artifact_capture(run_id, &task.resolved, group, cx)
                    })
                    .expect("actual public begin")
                    .expect("capture intent");
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                    })
                    .expect("unrelated workspace size is not a report admission bound");
                let capture = project
                    .update(cx, |project, cx| {
                        project.prepare_task_artifact_capture_sources(capture, cx)
                    })
                    .expect("actual public preparation")
                    .await
                    .expect("shared initial source identities");
                filesystem.pause_events();
                if phase == "changed" {
                    let before = filesystem
                        .metadata(source_path)
                        .await
                        .expect("actual initial mapped identity")
                        .expect("mapped source");
                    filesystem
                        .write(source_path, b"fn main(){ }")
                        .await
                        .expect("same-size mapped source mutation with watcher held");
                    let after = filesystem
                        .metadata(source_path)
                        .await
                        .expect("actual changed mapped identity")
                        .expect("mapped source remains");
                    assert_eq!(before.inode, after.inode);
                    assert_eq!(before.len, after.len);
                    assert_ne!(before.mtime, after.mtime);
                }
                filesystem
                    .insert_file(
                        Path::new("/workspace").join(
                            &group
                                .files
                                .get("report")
                                .expect("exact declared report")
                                .path,
                        ),
                        contents.as_bytes().to_vec(),
                    )
                    .await;
                let observed = project
                    .update(cx, |project, cx| {
                        project.observe_task_artifact_capture(
                            capture,
                            Task::ready(Some(Ok(actual_status))),
                            cx,
                        )
                    })
                    .await;
                assert_eq!(
                    observed
                        .expect("actual completion")
                        .expect("same actual successful exit"),
                    actual_status
                );
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage
                        .selected_snapshot()
                        .expect("large-workspace result");
                    let current = phase != "changed";
                    assert_eq!(
                        snapshot.status,
                        if current {
                            SourceCoverageStatus::Current
                        } else {
                            SourceCoverageStatus::Stale
                        },
                        "{scale}/{phase}"
                    );
                    assert_eq!(
                        snapshot.files.len(),
                        usize::from(current),
                        "{scale}/{phase}"
                    );
                });
            }
            filesystem.unpause_events_and_flush();
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_mapped_source_baseline_retires_buffer_subscriptions(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let filesystem = crate::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"src": {"bin": {
                    "fixture.rs": "fn main() {}", "unrelated.rs": "fn unrelated() {}"
                }}}),
            )
            .await;
        let project = crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
        let (coverage, source, unrelated) = project.read_with(cx, |project, cx| {
            (
                project.source_coverage_store().clone(),
                project
                    .find_project_path("/workspace/src/bin/fixture.rs", cx)
                    .expect("mapped source"),
                project
                    .find_project_path("/workspace/src/bin/unrelated.rs", cx)
                    .expect("unrelated source"),
            )
        });
        let existing = project
            .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
            .await
            .expect("existing Buffer before capture");
        let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
        let (run_id, task) = mapped_baseline_fixture_task(contents.len());
        let mut capture = project
            .update(cx, |project, cx| {
                project.begin_task_artifact_capture(
                    run_id,
                    &task.resolved,
                    task.resolved_artifact_group().expect("bound report"),
                    cx,
                )
            })
            .expect("public begin")
            .expect("capture intent");
        project
            .read_with(cx, |project, cx| {
                project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
            })
            .expect("public admission");
        cx.run_until_parked();
        drop(capture);
        cx.run_until_parked();
        let retired = coverage
            .read_with(cx, |coverage, _| coverage.request_version())
            .expect("source journal after normal capture Drop");
        existing.update(cx, |buffer, cx| {
            assert!(
                buffer
                    .edit([(0..0, "// retired existing Buffer edit\n")], None, cx)
                    .is_some()
            );
        });
        let new_buffer = project
            .update(cx, |project, cx| project.open_buffer(unrelated, cx))
            .await
            .expect("new Buffer after capture Drop");
        new_buffer.update(cx, |buffer, cx| {
            assert!(
                buffer
                    .edit([(0..0, "// retired new Buffer edit\n")], None, cx)
                    .is_some()
            );
        });
        let weak_existing = existing.downgrade();
        let weak_new = new_buffer.downgrade();
        drop(existing);
        drop(new_buffer);
        cx.run_until_parked();
        assert!(weak_existing.upgrade().is_none());
        assert!(weak_new.upgrade().is_none());
        assert_eq!(
            coverage
                .read_with(cx, |coverage, _| coverage.request_version())
                .expect("retired journal remains available"),
            retired,
            "neither retained child nor new Buffer callback survives capture Drop"
        );
        let actual_status = cx
            .background_spawn(async {
                smol::process::Command::new("/bin/sh")
                    .args(["-c", "exit 0"])
                    .status()
                    .await
            })
            .await
            .expect("actual controlled child exit");
        for dirty in [true, false] {
            let existing = project
                .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
                .await
                .expect("fresh mapped Buffer");
            existing.read_with(cx, |buffer, _| assert!(!buffer.is_dirty()));
            let (run_id, task) = mapped_baseline_fixture_task(contents.len());
            let group = task.resolved_artifact_group().expect("fresh bound report");
            let mut capture = project
                .update(cx, |project, cx| {
                    project.begin_task_artifact_capture(run_id, &task.resolved, group, cx)
                })
                .expect("fresh public begin")
                .expect("fresh capture");
            project
                .read_with(cx, |project, cx| {
                    project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                })
                .expect("fresh public admission");
            let capture = project
                .update(cx, |project, cx| {
                    project.prepare_task_artifact_capture_sources(capture, cx)
                })
                .expect("source preparation")
                .await
                .expect("fresh source baseline");
            filesystem.pause_events();
            filesystem
                .insert_file(
                    Path::new("/workspace")
                        .join(&group.files.get("report").expect("exact report role").path),
                    contents.as_bytes().to_vec(),
                )
                .await;
            let before = coverage
                .read_with(cx, |coverage, _| coverage.request_version())
                .expect("fresh admitted journal");
            if dirty {
                existing.update(cx, |buffer, cx| {
                    assert!(
                        buffer
                            .edit([(0..0, "// current capture edit\n")], None, cx)
                            .is_some()
                    );
                });
            }
            cx.run_until_parked();
            assert_eq!(
                coverage
                    .read_with(cx, |coverage, _| coverage.request_version())
                    .expect("current journal")
                    == before,
                !dirty,
                "fresh capture reattaches actual Buffer path observations"
            );
            let observed = project.update(cx, |project, cx| {
                project.observe_task_artifact_capture(
                    capture,
                    Task::ready(Some(Ok(actual_status))),
                    cx,
                )
            });
            assert_eq!(
                observed.await.expect("completion").expect("actual status"),
                actual_status
            );
            coverage.read_with(cx, |coverage, _| {
                let snapshot = coverage
                    .selected_snapshot()
                    .expect("terminal coverage state");
                assert_eq!(
                    snapshot.status,
                    if dirty {
                        SourceCoverageStatus::Stale
                    } else {
                        SourceCoverageStatus::Current
                    }
                );
                assert_eq!(snapshot.files.len(), usize::from(!dirty));
            });
            cx.run_until_parked();
            let completed = coverage
                .read_with(cx, |coverage, _| coverage.request_version())
                .expect("completed journal");
            existing.update(cx, |buffer, cx| {
                assert!(
                    buffer
                        .edit([(0..0, "// after observer retirement\n")], None, cx)
                        .is_some()
                );
            });
            drop(existing);
            cx.run_until_parked();
            assert_eq!(
                coverage
                    .read_with(cx, |coverage, _| coverage.request_version())
                    .expect("retained journal"),
                completed,
                "actual observer retirement removes live child Buffer subscriptions"
            );
            filesystem.unpause_events_and_flush();
            cx.run_until_parked();
        }
        project.update(cx, |project, cx| project.close(cx));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_mapped_source_baseline_journals_closed_new_buffer_edits(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
        let actual_status = cx
            .background_spawn(async {
                smol::process::Command::new("/bin/sh")
                    .args(["-c", "exit 0"])
                    .status()
                    .await
            })
            .await
            .expect("actual controlled child exit");
        for change in ["clean-mapped", "dirty-mapped", "dirty-unrelated"] {
            let filesystem = crate::FakeFs::new(cx.executor());
            filesystem
                .insert_tree(
                    "/workspace",
                    serde_json::json!({"src": {"bin": {
                        "fixture.rs": "fn main() {}",
                        "unrelated.rs": "fn unrelated() {}"
                    }}}),
                )
                .await;
            let project =
                crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
            let (coverage, source, worktree) = project.read_with(cx, |project, cx| {
                assert_eq!(project.buffer_store().read(cx).buffers().count(), 0);
                (
                    project.source_coverage_store().clone(),
                    project
                        .find_project_path("/workspace/src/bin/fixture.rs", cx)
                        .expect("mapped source"),
                    project
                        .worktrees(cx)
                        .next()
                        .expect("actual source worktree"),
                )
            });
            for recovery in [false, true] {
                let saved_buffer = if recovery {
                    let buffer = project
                        .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
                        .await
                        .expect("fresh clean mapped Buffer after old closure");
                    buffer.update(cx, |buffer, cx| {
                        assert!(!buffer.is_dirty());
                        assert!(
                            buffer
                                .edit([(0..0, "// saved recovery\n")], None, cx)
                                .is_some()
                        );
                    });
                    project
                        .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
                        .await
                        .expect("actual save before fresh source admission");
                    let metadata = filesystem
                        .metadata(Path::new("/workspace/src/bin/fixture.rs"))
                        .await
                        .expect("saved source metadata")
                        .expect("saved source exists");
                    cx.condition(&worktree, |worktree, _| {
                        worktree.entry_for_path(&source.path).is_some_and(|entry| {
                            entry.inode == metadata.inode && entry.mtime == Some(metadata.mtime)
                        })
                    })
                    .await;
                    Some(buffer)
                } else {
                    None
                };
                let (run_id, task) = mapped_baseline_fixture_task(contents.len());
                let group = task.resolved_artifact_group().expect("actual bound group");
                let mut capture = project
                    .update(cx, |project, cx| {
                        project.begin_task_artifact_capture(run_id, &task.resolved, group, cx)
                    })
                    .expect("public begin")
                    .expect("capture intent");
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                    })
                    .expect("actual post-save admission");
                let capture = project
                    .update(cx, |project, cx| {
                        project.prepare_task_artifact_capture_sources(capture, cx)
                    })
                    .expect("public source preparation")
                    .await
                    .expect("retained source baseline");
                filesystem.pause_events();
                filesystem
                    .insert_file(
                        Path::new("/workspace")
                            .join(&group.files.get("report").expect("exact report role").path),
                        contents.as_bytes().to_vec(),
                    )
                    .await;
                let gate = if recovery {
                    None
                } else {
                    Some(
                        project
                            .read_with(cx, |project, _| {
                                project.rust_coverage_provider_store().clone()
                            })
                            .update(cx, |provider, _| provider.gate_next_local_parse_for_test()),
                    )
                };
                let observed = project.update(cx, |project, cx| {
                    project.observe_task_artifact_capture(
                        capture,
                        Task::ready(Some(Ok(actual_status))),
                        cx,
                    )
                });
                if let Some((started, release)) = gate {
                    assert!(
                        matches!(
                            futures::future::select(
                                started,
                                cx.background_executor
                                    .timer(std::time::Duration::from_secs(10))
                            )
                            .await,
                            futures::future::Either::Left((Ok(()), _))
                        ),
                        "actual parsed report is held"
                    );
                    let path = if change == "dirty-unrelated" {
                        project
                            .read_with(cx, |project, cx| {
                                project.find_project_path("/workspace/src/bin/unrelated.rs", cx)
                            })
                            .expect("unrelated visible source")
                    } else {
                        source.clone()
                    };
                    let disk_path = Path::new("/workspace").join(path.path.as_std_path());
                    let before_disk = filesystem
                        .metadata(&disk_path)
                        .await
                        .expect("disk metadata")
                        .expect("actual source");
                    let before_journal = coverage
                        .read_with(cx, |coverage, _| coverage.request_version())
                        .expect("current journal");
                    let buffer = project
                        .update(cx, |project, cx| project.open_buffer(path, cx))
                        .await
                        .expect("new Buffer opened after report parse");
                    let weak_buffer = buffer.downgrade();
                    buffer.update(cx, |buffer, cx| {
                        assert!(!buffer.is_dirty());
                        if change != "clean-mapped" {
                            assert!(
                                buffer
                                    .edit([(0..0, "// discarded unsaved change\n")], None, cx)
                                    .is_some()
                            );
                            assert!(buffer.is_dirty());
                        }
                    });
                    drop(buffer);
                    cx.run_until_parked();
                    assert!(
                        weak_buffer.upgrade().is_none(),
                        "actual edited Buffer is released before publication"
                    );
                    project.read_with(cx, |project, cx| {
                        assert_eq!(project.buffer_store().read(cx).buffers().count(), 0)
                    });
                    let after_journal = coverage
                        .read_with(cx, |coverage, _| coverage.request_version())
                        .expect("retained journal");
                    assert_eq!(
                        before_journal == after_journal,
                        change == "clean-mapped",
                        "only actual Buffer edits record exact paths"
                    );
                    let after_disk = filesystem
                        .metadata(&disk_path)
                        .await
                        .expect("unchanged disk metadata")
                        .expect("actual source remains");
                    assert_eq!(
                        (before_disk.inode, before_disk.mtime, before_disk.len),
                        (after_disk.inode, after_disk.mtime, after_disk.len),
                        "discarded unsaved edits never change disk identity"
                    );
                    release
                        .send(())
                        .expect("release genuine parsed report after Buffer closure");
                }
                assert_eq!(
                    observed
                        .await
                        .expect("unchanged actual completion")
                        .expect("same successful exit"),
                    actual_status
                );
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage.selected_snapshot().expect("owned report state");
                    let current = recovery || change != "dirty-mapped";
                    assert_eq!(
                        snapshot.status,
                        if current {
                            SourceCoverageStatus::Current
                        } else {
                            SourceCoverageStatus::Stale
                        },
                        "{change}/{recovery}: {:?}",
                        snapshot.diagnostic
                    );
                    assert_eq!(
                        snapshot.files.len(),
                        usize::from(current),
                        "{change}/{recovery}"
                    );
                });
                filesystem.unpause_events_and_flush();
                drop(saved_buffer);
                cx.run_until_parked();
            }
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_mapped_source_baseline_checks_buffers_opened_after_admission(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
        let actual_status = cx
            .background_spawn(async {
                smol::process::Command::new("/bin/sh")
                    .args(["-c", "exit 0"])
                    .status()
                    .await
            })
            .await
            .expect("actual controlled child exit");
        for change in [
            "clean-new",
            "clean-new-noop-reload",
            "dirty-new",
            "late-dirty-new",
        ] {
            let filesystem = crate::FakeFs::new(cx.executor());
            filesystem
                .insert_tree(
                    "/workspace",
                    serde_json::json!({"src": {"bin": {"fixture.rs": "fn main() {}"}}}),
                )
                .await;
            let project =
                crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
            let (coverage, source, worktree) = project.read_with(cx, |project, cx| {
                assert_eq!(project.buffer_store().read(cx).buffers().count(), 0);
                (
                    project.source_coverage_store().clone(),
                    project
                        .find_project_path("/workspace/src/bin/fixture.rs", cx)
                        .expect("mapped source"),
                    project
                        .worktrees(cx)
                        .next()
                        .expect("actual source worktree"),
                )
            });
            let mut buffer: Option<Entity<language::Buffer>> = None;
            for recovery in [false, true] {
                if recovery {
                    let buffer = buffer.as_ref().expect("actual new Buffer retained");
                    project
                        .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
                        .await
                        .expect("actual source save before fresh admission");
                    let metadata = filesystem
                        .metadata(Path::new("/workspace/src/bin/fixture.rs"))
                        .await
                        .expect("actual fresh source metadata")
                        .expect("actual source");
                    cx.condition(&worktree, |worktree, _| {
                        worktree.entry_for_path(&source.path).is_some_and(|entry| {
                            entry.inode == metadata.inode && entry.mtime == Some(metadata.mtime)
                        })
                    })
                    .await;
                }
                let (run_id, task) = mapped_baseline_fixture_task(contents.len());
                let group = task.resolved_artifact_group().expect("actual bound group");
                let mut capture = project
                    .update(cx, |project, cx| {
                        project.begin_task_artifact_capture(run_id, &task.resolved, group, cx)
                    })
                    .expect("actual public begin")
                    .expect("capture intent");
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                    })
                    .expect("actual post-save admission");
                let capture = project
                    .update(cx, |project, cx| {
                        project.prepare_task_artifact_capture_sources(capture, cx)
                    })
                    .expect("actual public preparation")
                    .await
                    .expect("initial source baseline before opening Buffer");
                let metadata = filesystem
                    .metadata(Path::new("/workspace/src/bin/fixture.rs"))
                    .await
                    .expect("actual initial source metadata")
                    .expect("mapped source");
                if !recovery && change != "late-dirty-new" {
                    buffer = Some(
                        project
                            .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
                            .await
                            .expect("actual Buffer opened after admission"),
                    );
                    let buffer = buffer.as_ref().expect("new Buffer");
                    buffer.read_with(cx, |buffer, _| assert!(!buffer.is_dirty()));
                    if change == "clean-new-noop-reload" {
                        let version = buffer.read_with(cx, |buffer, _| buffer.version());
                        let transaction = buffer
                            .update(cx, |buffer, cx| buffer.reload(cx))
                            .await
                            .expect("actual unchanged source reload completes");
                        assert!(transaction.is_none(), "no source edits in actual reload");
                        buffer.read_with(cx, |buffer, _| {
                            assert_eq!(buffer.version(), version);
                            assert!(!buffer.is_dirty());
                        });
                    }
                    if change == "dirty-new" {
                        buffer.update(cx, |buffer, cx| {
                            assert!(
                                buffer
                                    .edit([(0..0, "// unsaved new Buffer\n")], None, cx)
                                    .is_some()
                            );
                            assert!(buffer.is_dirty());
                        });
                    }
                }
                filesystem
                    .insert_file(
                        Path::new("/workspace")
                            .join(&group.files.get("report").expect("exact report role").path),
                        contents.as_bytes().to_vec(),
                    )
                    .await;
                let parse_gate = if !recovery && change == "late-dirty-new" {
                    Some(
                        project
                            .read_with(cx, |project, _| {
                                project.rust_coverage_provider_store().clone()
                            })
                            .update(cx, |provider, _| provider.gate_next_local_parse_for_test()),
                    )
                } else {
                    None
                };
                let observed = project.update(cx, |project, cx| {
                    project.observe_task_artifact_capture(
                        capture,
                        Task::ready(Some(Ok(actual_status))),
                        cx,
                    )
                });
                if let Some((started, release)) = parse_gate {
                    assert!(
                        matches!(
                            futures::future::select(
                                started,
                                cx.background_executor
                                    .timer(std::time::Duration::from_secs(10))
                            )
                            .await,
                            futures::future::Either::Left((Ok(()), _))
                        ),
                        "actual parsed report awaits publication"
                    );
                    buffer = Some(
                        project
                            .update(cx, |project, cx| project.open_buffer(source.clone(), cx))
                            .await
                            .expect("new Buffer opened after report parse"),
                    );
                    buffer
                        .as_ref()
                        .expect("actual new late Buffer")
                        .update(cx, |buffer, cx| {
                            assert!(!buffer.is_dirty());
                            assert!(
                                buffer
                                    .edit([(0..0, "// unsaved late Buffer\n")], None, cx)
                                    .is_some()
                            );
                            assert!(buffer.is_dirty());
                        });
                    release.send(()).expect("release genuine parsed report");
                }
                assert_eq!(
                    observed
                        .await
                        .expect("actual completion")
                        .expect("same actual successful exit"),
                    actual_status
                );
                let after = filesystem
                    .metadata(Path::new("/workspace/src/bin/fixture.rs"))
                    .await
                    .expect("actual disk source remains")
                    .expect("mapped source");
                assert_eq!(after.inode, metadata.inode);
                assert_eq!(
                    after.mtime, metadata.mtime,
                    "unsaved Buffer changes do not alter disk metadata"
                );
                assert_eq!(after.len, metadata.len);
                coverage.read_with(cx, |coverage, _| {
                    let snapshot = coverage
                        .selected_snapshot()
                        .expect("owned new Buffer result");
                    let current =
                        recovery || matches!(change, "clean-new" | "clean-new-noop-reload");
                    assert_eq!(
                        snapshot.status,
                        if current {
                            SourceCoverageStatus::Current
                        } else {
                            SourceCoverageStatus::Stale
                        },
                        "{change}/{recovery}: {:?}",
                        snapshot.diagnostic
                    );
                    assert_eq!(
                        snapshot.files.len(),
                        usize::from(current),
                        "{change}/{recovery}"
                    );
                });
            }
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_coverage_provider_mapped_source_baseline_rejects_held_disk_edit_and_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let actual_status = cx
            .background_spawn(async {
                smol::process::Command::new("/bin/sh")
                    .args(["-c", "exit 0"])
                    .status()
                    .await
            })
            .await
            .expect("actual controlled child exit");
        assert!(actual_status.success());
        for change in [
            "same-size-source",
            "unrelated-output",
            "root-report",
            "dirty-buffer",
            "closed-buffer",
            "clean-closed-buffer",
            "late-parse",
        ] {
            let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
            let filesystem = crate::FakeFs::new(cx.executor());
            filesystem
                .insert_tree(
                    "/workspace",
                    serde_json::json!({
                        "src": {"bin": {"fixture.rs": "fn main() {}"}},
                        "target": {"other-output": "before"}
                    }),
                )
                .await;
            let project =
                crate::Project::test(filesystem.clone(), ["/workspace".as_ref()], cx).await;
            let coverage =
                project.read_with(cx, |project, _| project.source_coverage_store().clone());
            let mut buffer = if matches!(
                change,
                "dirty-buffer" | "closed-buffer" | "clean-closed-buffer"
            ) {
                let path = project.read_with(cx, |project, cx| {
                    project
                        .find_project_path("/workspace/src/bin/fixture.rs", cx)
                        .expect("source path")
                });
                Some(
                    project
                        .update(cx, |project, cx| project.open_buffer(path, cx))
                        .await
                        .expect("actual source Buffer"),
                )
            } else {
                None
            };
            let report_path = if change == "root-report" {
                "report.json"
            } else {
                "target/report.json"
            };
            let template = task::TaskTemplate {
                label: "mapped source baseline fixture".into(),
                command: "cargo".into(),
                args: vec![
                    "llvm-cov".into(),
                    "--output-path".into(),
                    report_path.into(),
                ],
                cwd: Some("/workspace".into()),
                artifact_group: Some(task::TaskArtifactGroup {
                    format: "llvm-cov-json-v1".into(),
                    files: collections::BTreeMap::from_iter([(
                        "report".into(),
                        task::TaskArtifact {
                            path: report_path.into(),
                            kind: task::TaskArtifactKind::Data,
                            max_bytes: contents.len() as u64,
                        },
                    )]),
                }),
                artifact_group_argument_binding: Some(task::TaskArtifactGroupArgumentBinding {
                    role: "report".into(),
                    argument_index: 2,
                    related: Vec::new(),
                }),
                ..task::TaskTemplate::default()
            };
            for recovery in [false, true] {
                if recovery && matches!(change, "closed-buffer" | "clean-closed-buffer") {
                    let path = project.read_with(cx, |project, cx| {
                        project
                            .find_project_path("/workspace/src/bin/fixture.rs", cx)
                            .expect("reopened source path")
                    });
                    buffer = Some(
                        project
                            .update(cx, |project, cx| project.open_buffer(path, cx))
                            .await
                            .expect("fresh source Buffer"),
                    );
                }
                if recovery && let Some(buffer) = buffer.as_ref() {
                    project
                        .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
                        .await
                        .expect("actual fresh configured source save");
                }
                let run_id = uuid::Uuid::new_v4().to_string();
                let task = template
                    .resolve_task("mapped-source", &task::TaskContext::default())
                    .expect("actual template resolution")
                    .bind_artifact_invocation(&run_id)
                    .expect("unique actual invocation");
                let group = task.resolved_artifact_group().expect("bound group").clone();
                let mut capture = project
                    .update(cx, |project, cx| {
                        project.begin_task_artifact_capture(run_id, &task.resolved, &group, cx)
                    })
                    .expect("public begin")
                    .expect("coverage capture");
                project
                    .read_with(cx, |project, cx| {
                        project.admit_task_artifact_capture(&mut capture, &task.resolved, cx)
                    })
                    .expect("post-save admission");
                let capture = project
                    .update(cx, |project, cx| {
                        project.prepare_task_artifact_capture_sources(capture, cx)
                    })
                    .expect("actual source preparation")
                    .await
                    .expect("baseline before child completion");
                filesystem.pause_events();
                let source = Path::new("/workspace/src/bin/fixture.rs");
                if !recovery && matches!(change, "closed-buffer" | "clean-closed-buffer") {
                    let buffer = buffer.take().expect("actual admitted source Buffer");
                    if change == "closed-buffer" {
                        buffer.update(cx, |buffer, cx| {
                            assert!(
                                buffer
                                    .edit([(0..0, "// dirty before close\n")], None, cx)
                                    .is_some()
                            );
                            assert!(buffer.is_dirty());
                        });
                    } else {
                        buffer.read_with(cx, |buffer, _| assert!(!buffer.is_dirty()));
                    }
                    drop(buffer);
                } else if !recovery && let Some(buffer) = buffer.as_ref() {
                    buffer.update(cx, |buffer, cx| {
                        assert!(buffer.edit([(0..0, "// dirty\n")], None, cx).is_some());
                        assert!(buffer.is_dirty());
                    });
                } else if !recovery && !matches!(change, "unrelated-output" | "late-parse") {
                    let before = filesystem
                        .metadata(source)
                        .await
                        .expect("source metadata")
                        .expect("source exists");
                    filesystem
                        .write(source, b"fn main(){ }")
                        .await
                        .expect("actual same-size source write");
                    let after = filesystem
                        .metadata(source)
                        .await
                        .expect("changed metadata")
                        .expect("changed source");
                    assert_eq!(before.inode, after.inode, "same actual source inode");
                    assert_eq!(before.len, after.len, "same actual source length");
                    assert_ne!(before.mtime, after.mtime);
                } else {
                    filesystem
                        .write(Path::new("/workspace/target/other-output"), b"after!")
                        .await
                        .expect("unrelated output write");
                }
                filesystem
                    .insert_file(
                        &format!(
                            "/workspace/{}",
                            group
                                .files
                                .get("report")
                                .expect("declared report role")
                                .path
                        ),
                        contents.as_bytes().to_vec(),
                    )
                    .await;
                assert!(
                    filesystem.buffered_event_count() > 0,
                    "actual watcher delivery held"
                );
                let parse_gate = if !recovery && change == "late-parse" {
                    Some(
                        project
                            .read_with(cx, |project, _| {
                                project.rust_coverage_provider_store().clone()
                            })
                            .update(cx, |provider, _| provider.gate_next_local_parse_for_test()),
                    )
                } else {
                    None
                };
                let observed = project.update(cx, |project, cx| {
                    project.observe_task_artifact_capture(
                        capture,
                        Task::ready(Some(Ok(actual_status))),
                        cx,
                    )
                });
                if let Some((started, release)) = parse_gate {
                    assert!(
                        matches!(
                            futures::future::select(
                                started,
                                cx.background_executor
                                    .timer(std::time::Duration::from_secs(10))
                            )
                            .await,
                            futures::future::Either::Left((Ok(()), _))
                        ),
                        "actual report parser reached held publication"
                    );
                    filesystem
                        .write(source, b"fn main(){ }")
                        .await
                        .expect("actual post-parse source write");
                    release.send(()).expect("release genuine parsed report");
                }
                let observed = observed.await;
                assert_eq!(
                    observed
                        .expect("actual process result")
                        .expect("unchanged successful exit"),
                    actual_status
                );
                coverage.read_with(cx, |store, _| {
                    let snapshot = store.selected_snapshot().expect("current owned status");
                    if !recovery && !matches!(change, "unrelated-output" | "clean-closed-buffer") {
                        assert_eq!(snapshot.status, SourceCoverageStatus::Stale, "{change}");
                        assert!(
                            snapshot.files.is_empty(),
                            "no obsolete current source facts"
                        );
                    } else {
                        assert_eq!(
                            snapshot.status,
                            SourceCoverageStatus::Current,
                            "{change}: {:?}",
                            snapshot.diagnostic
                        );
                        assert_eq!(snapshot.files.len(), 1);
                    }
                });
                filesystem.unpause_events_and_flush();
                cx.run_until_parked();
            }
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[gpui::test]
    async fn rust_coverage_provider_task_artifact_replacement_rejects_changes_after_parse(
        cx: &mut gpui::TestAppContext,
    ) {
        use fs::Fs as _;

        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        for change in [
            "rewrite",
            "replacement",
            "missing",
            "grow",
            "private",
            "external",
        ] {
            let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
            let fs = crate::FakeFs::new(cx.executor());
            fs.insert_tree(
                "/workspace",
                serde_json::json!({
                    "src": {"bin": {"fixture.rs": "fn main() {}"}},
                    "target": {"zed-coverage.json": contents},
                    ".env": contents,
                }),
            )
            .await;
            fs.insert_tree("/outside", serde_json::json!({"report.json": contents}))
                .await;
            let project = crate::Project::test(fs.clone(), ["/workspace".as_ref()], cx).await;
            let (provider, coverage, artifact) = project.read_with(cx, |project, cx| {
                (
                    project.rust_coverage_provider_store().clone(),
                    project.source_coverage_store().clone(),
                    project
                        .find_project_path("/workspace/target/zed-coverage.json", cx)
                        .expect("coverage artifact"),
                )
            });
            let baseline = provider
                .update(cx, |provider, cx| {
                    provider.interpret_artifact(artifact.clone(), contents.len(), cx)
                })
                .await
                .expect("unchanged exact-byte-limit report");
            assert_eq!(baseline.status, SourceCoverageStatus::Current);
            assert_eq!(baseline.files.len(), 1);
            let (started, release) =
                provider.update(cx, |provider, _| provider.gate_next_local_parse_for_test());
            let pending = provider.update(cx, |provider, cx| {
                provider.interpret_artifact(artifact.clone(), contents.len(), cx)
            });
            assert!(
                matches!(
                    futures::future::select(
                        started,
                        cx.background_executor
                            .timer(std::time::Duration::from_secs(10)),
                    )
                    .await,
                    futures::future::Either::Left((Ok(()), _))
                ),
                "actual bounded load and parse must complete before artifact mutation"
            );
            fs.pause_events();
            let path = Path::new("/workspace/target/zed-coverage.json");
            match change {
                "rewrite" => {
                    let before = fs
                        .metadata(path)
                        .await
                        .expect("artifact metadata")
                        .expect("artifact exists");
                    let changed = contents.replace("\"covered\": 3", "\"covered\": 2");
                    assert_ne!(changed, contents, "rewrite changes report contents");
                    assert_eq!(changed.len(), contents.len(), "same-size rewrite");
                    fs.write(path, changed.as_bytes())
                        .await
                        .expect("rewrite same inode");
                    let after = fs
                        .metadata(path)
                        .await
                        .expect("rewritten metadata")
                        .expect("rewritten artifact");
                    assert_eq!(
                        after.inode, before.inode,
                        "version guard must detect same inode"
                    );
                    assert_ne!(after.mtime, before.mtime);
                }
                "replacement" => {
                    fs.remove_file(path, fs::RemoveOptions::default())
                        .await
                        .expect("remove parsed artifact");
                    fs.insert_file(path, contents.as_bytes().to_vec()).await;
                }
                "missing" => fs
                    .remove_file(path, fs::RemoveOptions::default())
                    .await
                    .expect("remove parsed artifact"),
                "grow" => {
                    fs.insert_file(path, format!("{contents} ").into_bytes())
                        .await
                }
                "private" => fs.insert_symlink(path, "/workspace/.env".into()).await,
                "external" => fs.insert_symlink(path, "/outside/report.json".into()).await,
                _ => unreachable!("fixed artifact transition"),
            }
            assert!(
                fs.buffered_event_count() > 0,
                "watcher delivery remains held"
            );
            release
                .send(())
                .expect("release parsed artifact before explicit final refresh");
            assert!(
                pending.await.is_err(),
                "{change} must not publish parsed obsolete bytes"
            );
            coverage.read_with(cx, |coverage, _| {
                assert_eq!(coverage.selected_snapshot(), Some(&baseline), "{change}");
            });
            let recovery = Path::new("/workspace/target/recovery-report.json");
            fs.insert_file(recovery, contents.as_bytes().to_vec()).await;
            fs.rename(
                recovery,
                path,
                fs::RenameOptions {
                    overwrite: true,
                    ..Default::default()
                },
            )
            .await
            .expect("replace changed entry before fresh recovery");
            fs.unpause_events_and_flush();
            let recovered = provider
                .update(cx, |provider, cx| {
                    provider.interpret_artifact(artifact.clone(), contents.len(), cx)
                })
                .await
                .expect("fresh successful report recovers");
            assert_eq!(recovered.status, SourceCoverageStatus::Current);
            assert_eq!(recovered.files, baseline.files);
            assert!(recovered.generation > baseline.generation);
            project.update(cx, |project, cx| project.close(cx));
        }
    }

    #[gpui::test]
    async fn rust_coverage_provider_task_artifact_replacement_loads_fresh_output_with_byte_limit(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let fs = crate::FakeFs::new(cx.executor());
        fs.insert_tree(
            "/workspace",
            serde_json::json!({
                ".git": {},
                ".gitignore": "target\n",
                "src": {"bin": {"fixture.rs": "fn main() {}"}},
                "target": {},
            }),
        )
        .await;
        let project = crate::Project::test(fs.clone(), ["/workspace".as_ref()], cx).await;
        let (provider, coverage, worktree_id) = project.read_with(cx, |project, cx| {
            (
                project.rust_coverage_provider_store().clone(),
                project.source_coverage_store().clone(),
                project
                    .worktrees(cx)
                    .next()
                    .expect("artifact worktree")
                    .read(cx)
                    .id(),
            )
        });
        let contents = include_str!("../test_data/source_coverage/llvm3-run.json");
        fs.pause_events();
        fs.insert_file(
            "/workspace/target/new-coverage.json",
            contents.as_bytes().to_vec(),
        )
        .await;
        let artifact = ProjectPath {
            worktree_id,
            path: util::rel_path::rel_path("target/new-coverage.json").into(),
        };
        assert!(
            provider
                .update(cx, |provider, cx| {
                    provider.interpret_artifact(artifact.clone(), contents.len() - 1, cx)
                })
                .await
                .is_err(),
            "declared limit must reject before oversized interpretation"
        );
        coverage.read_with(cx, |coverage, _| {
            assert!(coverage.selected_snapshot().is_none())
        });
        let snapshot = provider
            .update(cx, |provider, cx| {
                provider.interpret_artifact(artifact.clone(), contents.len(), cx)
            })
            .await
            .expect("fresh output is explicitly refreshed and bounded");
        assert_eq!(snapshot.status, SourceCoverageStatus::Current);
        assert_eq!(snapshot.files.len(), 1);
        project.read_with(cx, |project, cx| {
            let worktree = project
                .worktree_for_id(worktree_id, cx)
                .expect("artifact worktree");
            assert!(
                worktree
                    .read(cx)
                    .entry_for_path(&artifact.path)
                    .expect("fresh artifact entry")
                    .is_ignored,
                "actual repository ignore rules apply to freshly loaded Task output"
            );
        });
        fs.unpause_events_and_flush();
        project.update(cx, |project, cx| project.close(cx));
    }

    #[test]
    fn rust_coverage_provider_validates_supported_artifacts_paths_and_cancellation() {
        let mut provider = RustCoverageArtifactProvider::new(5);
        let token = provider.begin_run();
        let artifact = br#"{
          "type":"llvm.coverage.json.export","version":"2.0.1",
          "cargo_llvm_cov":{"version":"0.9.0","manifest_path":"/workspace/Cargo.toml"},
          "data":[{"files":[
            {"filename":"/workspace/src/lib.rs","segments":[[1,1,3,true,true,false],[2,1,0,true,true,false]],"summary":{"lines":{"count":2,"covered":1}}},
            {"filename":"/private/cache.rs","segments":[],"summary":{"lines":{"count":1,"covered":0}}},
            {"segments":[]}
          ]}]
        }"#;
        let snapshot = provider
            .parse_artifact(token, artifact, resolve)
            .expect("supported artifact should parse");
        assert_eq!(snapshot.files.len(), 1);
        assert_eq!(snapshot.status, SourceCoverageStatus::Partial);
        assert_eq!(snapshot.files[0].ranges.len(), 2);
        provider.cancel(token);
        assert!(provider.parse_artifact(token, artifact, resolve).is_err());
    }

    #[test]
    fn rust_coverage_provider_rejects_malformed_unsupported_and_oversized_artifacts() {
        let mut provider = RustCoverageArtifactProvider::new(1);
        let token = provider.begin_run();
        assert!(provider.parse_artifact(token, b"{}", resolve).is_err());
        let unsupported = br#"{
            "type":"llvm.coverage.json.export","version":"4.0.0",
            "cargo_llvm_cov":{"version":"0.9.0"},"data":[]
        }"#;
        assert!(
            provider
                .parse_artifact(token, unsupported, resolve)
                .is_err()
        );
        assert!(
            provider
                .parse_artifact(
                    token,
                    &vec![b' '; MAX_RUST_COVERAGE_ARTIFACT_BYTES + 1],
                    resolve
                )
                .is_err()
        );
    }

    #[test]
    fn rust_coverage_provider_marks_omitted_exports_partial() {
        let mut provider = RustCoverageArtifactProvider::new(1);
        let token = provider.begin_run();
        let exports: Vec<_> = (0..17)
            .map(|index| {
                serde_json::json!({
                    "files": [{
                        "filename": format!("/workspace/src/file_{index}.rs"),
                        "segments": [[1, 1, 1, true, true, false]],
                        "summary": {"lines": {"count": 1, "covered": 1}}
                    }]
                })
            })
            .collect();
        let artifact = serde_json::to_vec(&serde_json::json!({
            "type": "llvm.coverage.json.export",
            "version": "2.0.1",
            "cargo_llvm_cov": {"version": "0.9.0"},
            "data": exports
        }))
        .expect("fixture should serialize");

        let snapshot = provider
            .parse_artifact(token, &artifact, resolve)
            .expect("bounded artifact should parse");
        assert_eq!(snapshot.files.len(), 16);
        assert!(snapshot.truncated);
        assert_eq!(snapshot.status, SourceCoverageStatus::Partial);
        assert!(snapshot.files.iter().all(|file| !file.truncated));
    }
}
