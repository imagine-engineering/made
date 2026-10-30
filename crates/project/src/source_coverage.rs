use feature_flags::{FeatureFlagAppExt as _, RustCoverageExperimentalFeatureFlag};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, bail};
use client::Client;
use futures::StreamExt;
use gpui::{App, AsyncApp, Context, Entity, EventEmitter, Subscription, Task};
use rpc::{AnyProtoClient, TypedEnvelope, proto};
use util::ResultExt;

use crate::{
    ProjectPath, WorktreeId,
    trusted_worktrees::{TrustedWorktrees, TrustedWorktreesEvent},
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};

pub const SOURCE_COVERAGE_PROTOCOL_VERSION: u32 = 1;
pub const MAX_SOURCE_COVERAGE_FILES: usize = 4_096;
pub const MAX_SOURCE_COVERAGE_RANGES: usize = 100_000;
pub const MAX_SOURCE_COVERAGE_RANGES_PER_FILE: usize = 10_000;
pub const MAX_SOURCE_COVERAGE_PROVIDER_BYTES: usize = 256;
pub const MAX_SOURCE_COVERAGE_DIAGNOSTIC_BYTES: usize = 4 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SourceCoverageProviderId(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceCoverageStatus {
    Loading,
    Current,
    Empty,
    Partial,
    Stale,
    Error,
    Restricted,
    Disconnected,
    Mismatch,
    Cancelled,
}

#[derive(Debug)]
pub struct SourceCoverageUnsupportedHost;

impl std::fmt::Display for SourceCoverageUnsupportedHost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("The authoritative host does not support source coverage")
    }
}

impl std::error::Error for SourceCoverageUnsupportedHost {}

#[derive(Debug)]
pub struct SourceCoverageSourceChanged;

impl std::fmt::Display for SourceCoverageSourceChanged {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Source coverage completed after its source changed")
    }
}

impl std::error::Error for SourceCoverageSourceChanged {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceCoveragePoint {
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceCoverageRange {
    pub start: SourceCoveragePoint,
    pub end: SourceCoveragePoint,
    pub hit_count: u64,
}

impl SourceCoverageRange {
    pub fn is_covered(&self) -> bool {
        self.hit_count > 0
    }

    fn is_valid(&self) -> bool {
        (self.start.line, self.start.column) <= (self.end.line, self.end.column)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceCoverageFile {
    pub path: ProjectPath,
    pub ranges: Vec<SourceCoverageRange>,
    pub covered_lines: u32,
    pub uncovered_lines: u32,
    pub truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceCoverageSnapshot {
    pub project_generation: u64,
    pub provider_id: SourceCoverageProviderId,
    pub generation: u64,
    pub status: SourceCoverageStatus,
    pub files: Vec<SourceCoverageFile>,
    pub truncated: bool,
    pub diagnostic: Option<String>,
}

impl SourceCoverageSnapshot {
    pub fn bounded(mut self) -> Result<Self> {
        if self.provider_id.0.trim().is_empty()
            || self.provider_id.0.len() > MAX_SOURCE_COVERAGE_PROVIDER_BYTES
        {
            bail!("source coverage provider identity is empty or too long");
        }
        if let Some(diagnostic) = &mut self.diagnostic {
            truncate_utf8(diagnostic, MAX_SOURCE_COVERAGE_DIAGNOSTIC_BYTES);
        }
        self.files.sort_by(|left, right| left.path.cmp(&right.path));
        self.files.dedup_by(|left, right| left.path == right.path);
        if self.files.len() > MAX_SOURCE_COVERAGE_FILES {
            self.files.truncate(MAX_SOURCE_COVERAGE_FILES);
            self.truncated = true;
        }
        let mut remaining_ranges = MAX_SOURCE_COVERAGE_RANGES;
        let mut malformed = 0usize;
        for file in &mut self.files {
            let original_len = file.ranges.len();
            file.ranges.retain(SourceCoverageRange::is_valid);
            malformed += original_len.saturating_sub(file.ranges.len());
            file.ranges.sort_by_key(|range| {
                (
                    range.start.line,
                    range.start.column,
                    range.end.line,
                    range.end.column,
                )
            });
            let limit = remaining_ranges.min(MAX_SOURCE_COVERAGE_RANGES_PER_FILE);
            if file.ranges.len() > limit {
                file.ranges.truncate(limit);
                file.truncated = true;
                self.truncated = true;
            }
            remaining_ranges = remaining_ranges.saturating_sub(file.ranges.len());
        }
        if malformed > 0 {
            if self.status != SourceCoverageStatus::Cancelled {
                self.status = SourceCoverageStatus::Partial;
            }
            self.diagnostic = Some(format!(
                "Ignored {malformed} malformed source coverage range(s)"
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct SourceCoverageState {
    project_generation: u64,
    providers: BTreeMap<SourceCoverageProviderId, SourceCoverageSnapshot>,
}

impl SourceCoverageState {
    pub fn new(project_generation: u64) -> Self {
        Self {
            project_generation,
            providers: BTreeMap::new(),
        }
    }

    pub fn project_generation(&self) -> u64 {
        self.project_generation
    }

    pub fn provider(
        &self,
        provider_id: &SourceCoverageProviderId,
    ) -> Option<&SourceCoverageSnapshot> {
        self.providers.get(provider_id)
    }

    pub fn publish(&mut self, snapshot: SourceCoverageSnapshot) -> Result<()> {
        let snapshot = snapshot.bounded()?;
        if snapshot.project_generation != self.project_generation {
            bail!("source coverage project generation is stale");
        }
        if self
            .providers
            .get(&snapshot.provider_id)
            .is_some_and(|current| snapshot.generation < current.generation)
        {
            bail!("source coverage provider generation is stale");
        }
        self.providers
            .insert(snapshot.provider_id.clone(), snapshot);
        Ok(())
    }

    pub fn mark_provider_status(
        &mut self,
        provider_id: SourceCoverageProviderId,
        status: SourceCoverageStatus,
        diagnostic: Option<String>,
    ) -> Result<()> {
        let snapshot = self
            .provider(&provider_id)
            .cloned()
            .map(|mut snapshot| {
                snapshot.status = status;
                snapshot.diagnostic = diagnostic.clone();
                snapshot
            })
            .unwrap_or(SourceCoverageSnapshot {
                project_generation: self.project_generation(),
                provider_id,
                generation: 0,
                status,
                files: Vec::new(),
                truncated: false,
                diagnostic,
            });
        self.publish(snapshot)
    }
}

#[derive(Clone, Debug)]
pub enum SourceCoverageStoreEvent {
    Changed(SourceCoverageProviderId),
}

impl EventEmitter<SourceCoverageStoreEvent> for SourceCoverageStore {}

#[derive(Default)]
struct SourceCoverageChanges {
    revision: u64,
    paths: HashMap<ProjectPath, u64>,
}

impl SourceCoverageChanges {
    fn record(&mut self, paths: &[ProjectPath]) -> bool {
        if paths.is_empty() {
            return false;
        }
        self.revision = self.revision.wrapping_add(1);
        let mut expired = self.revision == 0;
        if expired {
            self.paths.clear();
        }
        for path in paths {
            if self.paths.len() == MAX_SOURCE_COVERAGE_FILES && !self.paths.contains_key(path) {
                // Old replies may name an evicted path, so eviction expires their session.
                self.paths.clear();
                expired = true;
            }
            self.paths.insert(path.clone(), self.revision);
        }
        expired
    }

    fn changed_since(&self, path: &ProjectPath, revision: u64) -> bool {
        self.paths
            .get(path)
            .is_some_and(|changed| *changed > revision)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SourceCoverageRequestVersion {
    session_generation: u64,
    source_revision: u64,
}

pub struct SourceCoverageStore {
    mode: SourceCoverageStoreMode,
    worktree_store: Entity<WorktreeStore>,
    state: SourceCoverageState,
    selected_provider: Option<SourceCoverageProviderId>,
    disconnected: bool,
    request_generation: u64,
    // Rust revocation must not expire requests belonging to unrelated coverage providers.
    rust_experimental_generation: u64,
    session_generation: u64,
    source_changes: SourceCoverageChanges,
    active_requests: HashMap<SourceCoverageProviderId, u64>,
    _collaboration_session: Option<Task<()>>,
    _worktree_trust: Option<Subscription>,
    _worktree_entries: Option<Subscription>,
    _experimental_subscription: Subscription,
    #[cfg(any(test, feature = "test-support"))]
    snapshot_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    assembled_response_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    )>,
}

enum SourceCoverageStoreMode {
    Local,
    Remote {
        project_id: u64,
        client: AnyProtoClient,
    },
}

impl SourceCoverageStore {
    pub fn cached_snapshots(&self) -> impl Iterator<Item = &SourceCoverageSnapshot> {
        self.state.providers.values()
    }

    pub fn selected_snapshot(&self) -> Option<&SourceCoverageSnapshot> {
        self.selected_provider
            .as_ref()
            .and_then(|provider| self.state.provider(provider))
            .or_else(|| self.cached_snapshots().next())
    }

    pub fn select_provider(
        &mut self,
        provider: SourceCoverageProviderId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if self.state.provider(&provider).is_none() {
            bail!("unknown source coverage provider");
        }
        self.selected_provider = Some(provider.clone());
        cx.emit(SourceCoverageStoreEvent::Changed(provider));
        Ok(())
    }

    pub fn invalidate_paths(&mut self, paths: &[ProjectPath], cx: &mut Context<Self>) {
        if self.source_changes.record(paths) {
            if matches!(self.mode, SourceCoverageStoreMode::Remote { .. }) {
                self.invalidate_remote_session(cx);
            } else {
                self.session_generation = self.session_generation.wrapping_add(1);
            }
        }
        let providers = self
            .cached_snapshots()
            .filter(|snapshot| {
                matches!(
                    snapshot.status,
                    SourceCoverageStatus::Current | SourceCoverageStatus::Partial
                ) && snapshot.files.iter().any(|file| paths.contains(&file.path))
            })
            .map(|snapshot| snapshot.provider_id.clone())
            .collect::<Vec<_>>();
        for provider in providers {
            if let Some(snapshot) = self.state.providers.get_mut(&provider) {
                snapshot.status = SourceCoverageStatus::Stale;
                if matches!(self.mode, SourceCoverageStoreMode::Remote { .. }) {
                    snapshot.files.retain(|file| !paths.contains(&file.path));
                    snapshot.truncated = true;
                }
                snapshot.diagnostic =
                    Some("Source changed after coverage was collected. Run coverage again.".into());
            }
            cx.emit(SourceCoverageStoreEvent::Changed(provider));
        }
    }

    pub fn invalidate(&mut self, status: SourceCoverageStatus, cx: &mut Context<Self>) {
        if matches!(self.mode, SourceCoverageStoreMode::Local) {
            self.session_generation = self.session_generation.wrapping_add(1);
        }
        let mut changed = Vec::new();
        for snapshot in self.state.providers.values_mut() {
            if matches!(
                snapshot.status,
                SourceCoverageStatus::Current | SourceCoverageStatus::Partial
            ) {
                snapshot.status = status;
                changed.push(snapshot.provider_id.clone());
            }
        }
        for provider in changed {
            cx.emit(SourceCoverageStoreEvent::Changed(provider));
        }
    }

    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_get_source_coverage);
    }

    pub fn local(
        worktree_store: Entity<WorktreeStore>,
        project_generation: u64,
        cx: &mut Context<Self>,
    ) -> Self {
        let worktree_entries = Self::subscribe_worktree_entries(&worktree_store, cx);
        Self {
            mode: SourceCoverageStoreMode::Local,
            worktree_store,
            state: SourceCoverageState::new(project_generation),
            selected_provider: None,
            disconnected: false,
            request_generation: 0,
            rust_experimental_generation: 0,
            session_generation: 0,
            source_changes: SourceCoverageChanges::default(),
            active_requests: HashMap::new(),
            _collaboration_session: None,
            _worktree_trust: None,
            _worktree_entries: Some(worktree_entries),
            _experimental_subscription: cx.observe_global::<feature_flags::FeatureFlagStore>(
                |store, cx| {
                    if !cx.has_flag::<RustCoverageExperimentalFeatureFlag>() {
                        store.rust_experimental_generation =
                            store.rust_experimental_generation.wrapping_add(1);
                        let id = SourceCoverageProviderId("cargo-llvm-cov".into());
                        store.active_requests.remove(&id);
                        if store.state.provider(&id).is_some() {
                            if let Err(error) = store.mark_provider_status(
                                id,
                                SourceCoverageStatus::Stale,
                                Some("Rust coverage experimental opt-in was disabled".into()),
                                cx,
                            ) {
                                log::error!(
                                    "Rust coverage revocation could not be published: {error:#}"
                                );
                            }
                        }
                    }
                },
            ),
            #[cfg(any(test, feature = "test-support"))]
            snapshot_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            assembled_response_gate: None,
        }
    }

    fn subscribe_worktree_entries(
        worktree_store: &Entity<WorktreeStore>,
        cx: &mut Context<Self>,
    ) -> Subscription {
        cx.subscribe(worktree_store, |store, _, event, cx| match event {
            WorktreeStoreEvent::WorktreeUpdatedEntries(worktree_id, changes) => {
                let paths = changes
                    .iter()
                    .map(|(path, _, _)| ProjectPath {
                        worktree_id: *worktree_id,
                        path: path.clone(),
                    })
                    .collect::<Vec<_>>();
                store.invalidate_paths(&paths, cx);
            }
            WorktreeStoreEvent::WorktreeAdded(..)
            | WorktreeStoreEvent::WorktreeRemoved(..)
            | WorktreeStoreEvent::WorktreeReleased(..)
            | WorktreeStoreEvent::WorktreeOrderChanged
            | WorktreeStoreEvent::WorktreeDeletedEntry(..) => {
                if matches!(store.mode, SourceCoverageStoreMode::Remote { .. }) {
                    store.source_changes.paths.clear();
                    store.invalidate_remote_session(cx);
                } else {
                    store.invalidate(SourceCoverageStatus::Stale, cx);
                }
            }
            _ => {}
        })
    }

    pub fn remote(
        project_id: u64,
        client: AnyProtoClient,
        worktree_store: Entity<WorktreeStore>,
        project_generation: u64,
        cx: &mut Context<Self>,
    ) -> Self {
        let worktree_entries = Self::subscribe_worktree_entries(&worktree_store, cx);
        let worktree_trust = TrustedWorktrees::try_get_global(cx).map(|trust| {
            cx.subscribe(&trust, |store, _, event, cx| {
                let worktree_store = match event {
                    TrustedWorktreesEvent::Trusted(worktree_store, _)
                    | TrustedWorktreesEvent::Restricted(worktree_store, _) => worktree_store,
                };
                if *worktree_store != store.worktree_store.downgrade() {
                    return;
                }
                store.session_generation = store.session_generation.wrapping_add(1);
                store.active_requests.clear();
                let status = if store.disconnected {
                    SourceCoverageStatus::Disconnected
                } else if matches!(event, TrustedWorktreesEvent::Restricted(..)) {
                    SourceCoverageStatus::Restricted
                } else {
                    SourceCoverageStatus::Stale
                };
                for snapshot in store.state.providers.values_mut() {
                    snapshot.files.clear();
                    snapshot.truncated = true;
                    snapshot.status = status;
                    snapshot.diagnostic = Some("Worktree trust changed; refresh coverage".into());
                    cx.emit(SourceCoverageStoreEvent::Changed(
                        snapshot.provider_id.clone(),
                    ));
                }
            })
        });
        Self {
            mode: SourceCoverageStoreMode::Remote { project_id, client },
            worktree_store,
            state: SourceCoverageState::new(project_generation),
            selected_provider: None,
            disconnected: false,
            request_generation: 0,
            rust_experimental_generation: 0,
            session_generation: 0,
            source_changes: SourceCoverageChanges::default(),
            active_requests: HashMap::new(),
            _collaboration_session: None,
            _worktree_trust: worktree_trust,
            _worktree_entries: Some(worktree_entries),
            _experimental_subscription: cx.observe_global::<feature_flags::FeatureFlagStore>(
                |store, cx| {
                    if !cx.has_flag::<RustCoverageExperimentalFeatureFlag>() {
                        store.rust_experimental_generation =
                            store.rust_experimental_generation.wrapping_add(1);
                        let id = SourceCoverageProviderId("cargo-llvm-cov".into());
                        store.active_requests.remove(&id);
                        if store.state.provider(&id).is_some() {
                            if let Err(error) = store.mark_provider_status(
                                id,
                                SourceCoverageStatus::Stale,
                                Some("Rust coverage experimental opt-in was disabled".into()),
                                cx,
                            ) {
                                log::error!(
                                    "Rust coverage revocation could not be published: {error:#}"
                                );
                            }
                        }
                    }
                },
            ),
            #[cfg(any(test, feature = "test-support"))]
            snapshot_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            assembled_response_gate: None,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_snapshot_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
        self.snapshot_gate = Some((started_sender, release_receiver, completed_sender));
        (started_receiver, release_sender, completed_receiver)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_assembled_response_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
        self.assembled_response_gate = Some((started_sender, release_receiver, completed_sender));
        (started_receiver, release_sender, completed_receiver)
    }

    pub(crate) fn watch_collaboration_session(
        &mut self,
        client: &Arc<Client>,
        cx: &mut Context<Self>,
    ) {
        let mut statuses = client.status();
        let mut previous_status = *statuses.borrow();
        self._collaboration_session = Some(cx.spawn(async move |this, cx| {
            while let Some(status) = statuses.next().await {
                if status == previous_status {
                    continue;
                }
                previous_status = status;
                if this
                    .update(cx, |store, cx| {
                        if status.is_connected() {
                            store.reconnected_to_host(cx);
                        } else {
                            store.disconnected_from_host(cx);
                        }
                    })
                    .log_err()
                    .is_none()
                {
                    break;
                }
            }
        }));
    }

    pub(crate) fn invalidate_remote_session(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, SourceCoverageStoreMode::Remote { .. }) {
            return;
        }
        self.session_generation = self.session_generation.wrapping_add(1);
        self.active_requests.clear();
        self.invalidate(SourceCoverageStatus::Stale, cx);
    }

    pub(crate) fn remote_request_version(&self) -> Result<SourceCoverageRequestVersion> {
        if !matches!(self.mode, SourceCoverageStoreMode::Remote { .. }) {
            bail!("remote source coverage requires a remote store");
        }
        self.request_version()
    }

    pub(crate) fn request_version(&self) -> Result<SourceCoverageRequestVersion> {
        if self.disconnected {
            bail!("The source coverage host is disconnected");
        }
        Ok(SourceCoverageRequestVersion {
            session_generation: self.session_generation,
            source_revision: self.source_changes.revision,
        })
    }

    pub(crate) fn is_request_current(&self, request_version: SourceCoverageRequestVersion) -> bool {
        !self.disconnected && self.session_generation == request_version.session_generation
    }

    #[cfg(feature = "rust-coverage")]
    pub(crate) fn is_request_current_except_output(
        &self,
        request_version: SourceCoverageRequestVersion,
        output: &ProjectPath,
    ) -> bool {
        self.is_request_current(request_version)
            && request_version.source_revision <= self.source_changes.revision
            && self.source_changes.paths.iter().all(|(path, revision)| {
                *revision <= request_version.source_revision || path == output
            })
    }

    pub(crate) fn validate_snapshot_version(
        &self,
        snapshot: &SourceCoverageSnapshot,
        request_version: SourceCoverageRequestVersion,
    ) -> Result<()> {
        if !self.is_request_current(request_version) {
            bail!("Source coverage completed for an expired project session");
        }
        if snapshot.files.iter().any(|file| {
            self.source_changes
                .changed_since(&file.path, request_version.source_revision)
        }) {
            return Err(SourceCoverageSourceChanged.into());
        }
        Ok(())
    }

    pub fn disconnected_from_host(&mut self, cx: &mut Context<Self>) {
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        self.session_generation = self.session_generation.wrapping_add(1);
        self.active_requests.clear();
        for snapshot in self.state.providers.values_mut() {
            snapshot.status = SourceCoverageStatus::Disconnected;
            snapshot.diagnostic = Some("The source coverage host is disconnected".into());
            cx.emit(SourceCoverageStoreEvent::Changed(
                snapshot.provider_id.clone(),
            ));
        }
    }

    pub fn reconnected_to_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, SourceCoverageStoreMode::Remote { .. }) || !self.disconnected {
            return;
        }
        self.disconnected = false;
        for snapshot in self.state.providers.values_mut() {
            snapshot.status = SourceCoverageStatus::Stale;
            snapshot.diagnostic = Some("Connection restored; refresh coverage".into());
            cx.emit(SourceCoverageStoreEvent::Changed(
                snapshot.provider_id.clone(),
            ));
        }
    }

    pub fn publish(
        &mut self,
        snapshot: SourceCoverageSnapshot,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if snapshot.provider_id.0 == "cargo-llvm-cov" {
            anyhow::ensure!(
                cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                "Rust coverage requires local experimental opt-in"
            );
        }
        if !matches!(self.mode, SourceCoverageStoreMode::Local) {
            bail!("only the authoritative project host can publish source coverage");
        }
        let provider_id = snapshot.provider_id.clone();
        self.state.publish(snapshot)?;
        self.selected_provider.get_or_insert(provider_id.clone());
        cx.emit(SourceCoverageStoreEvent::Changed(provider_id));
        Ok(())
    }

    pub fn mark_provider_status(
        &mut self,
        provider_id: SourceCoverageProviderId,
        status: SourceCoverageStatus,
        diagnostic: Option<String>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.state
            .mark_provider_status(provider_id.clone(), status, diagnostic)?;
        cx.emit(SourceCoverageStoreEvent::Changed(provider_id));
        Ok(())
    }

    fn filter_remote_snapshot(&self, snapshot: &mut SourceCoverageSnapshot, cx: &App) {
        if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx) {
            snapshot.files.clear();
            snapshot.truncated = true;
            snapshot.status = SourceCoverageStatus::Restricted;
        } else {
            let visible_worktrees = self
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| (worktree.read(cx).id(), worktree))
                .collect::<HashMap<_, _>>();
            let previous_count = snapshot.files.len();
            snapshot.files.retain(|file| {
                visible_worktrees
                    .get(&file.path.worktree_id)
                    .is_some_and(|worktree| {
                        worktree
                            .read(cx)
                            .snapshot()
                            .entry_for_path(file.path.path.as_ref())
                            .is_some_and(|entry| {
                                entry.is_file()
                                    && !entry.is_private
                                    && !entry.is_fifo
                                    && !entry.is_external
                            })
                    })
            });
            snapshot.truncated |= snapshot.files.len() != previous_count;
        }
    }

    #[cfg(feature = "rust-coverage")]
    pub(crate) fn replace_remote(
        &mut self,
        mut snapshot: SourceCoverageSnapshot,
        request_version: SourceCoverageRequestVersion,
        cx: &mut Context<Self>,
    ) -> Result<SourceCoverageSnapshot> {
        if !matches!(self.mode, SourceCoverageStoreMode::Remote { .. }) {
            bail!("remote source coverage replacement requires a remote store");
        }
        self.validate_snapshot_version(&snapshot, request_version)?;
        self.filter_remote_snapshot(&mut snapshot, cx);
        if self.state.project_generation != snapshot.project_generation {
            self.state = SourceCoverageState::new(snapshot.project_generation);
        }
        let provider_id = snapshot.provider_id.clone();
        self.state.publish(snapshot.clone())?;
        self.selected_provider.get_or_insert(provider_id.clone());
        cx.emit(SourceCoverageStoreEvent::Changed(provider_id));
        Ok(snapshot)
    }

    pub fn snapshot(
        &mut self,
        provider_id: SourceCoverageProviderId,
        cx: &mut Context<Self>,
    ) -> Task<Result<SourceCoverageSnapshot>> {
        if provider_id.0 == "cargo-llvm-cov"
            && !cx.has_flag::<RustCoverageExperimentalFeatureFlag>()
        {
            return Task::ready(Err(anyhow!(
                "Rust coverage requires local experimental opt-in"
            )));
        }
        if self.disconnected {
            return Task::ready(Err(anyhow!("The source coverage host is disconnected")));
        }
        match &self.mode {
            SourceCoverageStoreMode::Local => Task::ready(
                self.state
                    .provider(&provider_id)
                    .cloned()
                    .context("unknown source coverage provider"),
            ),
            SourceCoverageStoreMode::Remote { project_id, client } => {
                let project_id = *project_id;
                let client = client.clone();
                self.request_generation = self.request_generation.wrapping_add(1);
                let generation = self.request_generation;
                self.active_requests.insert(provider_id.clone(), generation);
                let request_version = match self.remote_request_version() {
                    Ok(version) => version,
                    Err(error) => return Task::ready(Err(error)),
                };
                let worktree_ids = self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .map(|worktree| worktree.read(cx).id().to_proto())
                    .collect();
                cx.spawn(async move |this, cx| {
                    let response = client
                        .request(proto::GetSourceCoverage {
                            project_id,
                            provider_id: provider_id.0.clone(),
                            worktree_ids,
                        })
                        .await;
                    this.update(cx, |store, cx| {
                        if (provider_id.0 == "cargo-llvm-cov"
                            && !cx.has_flag::<RustCoverageExperimentalFeatureFlag>())
                            || store.disconnected
                            || store.active_requests.get(&provider_id) != Some(&generation)
                        {
                            bail!("stale source coverage refresh");
                        }
                        store.active_requests.remove(&provider_id);
                        let mut snapshot = snapshot_from_proto(
                            response.map_err(map_remote_source_coverage_error)?,
                        )?;
                        if snapshot.provider_id != provider_id {
                            bail!("source coverage provider response identity mismatch");
                        }
                        store.validate_snapshot_version(&snapshot, request_version)?;
                        store.filter_remote_snapshot(&mut snapshot, cx);
                        if store.state.project_generation != snapshot.project_generation {
                            store.state = SourceCoverageState::new(snapshot.project_generation);
                        }
                        store.state.publish(snapshot.clone())?;
                        store
                            .selected_provider
                            .get_or_insert(snapshot.provider_id.clone());
                        cx.emit(SourceCoverageStoreEvent::Changed(
                            snapshot.provider_id.clone(),
                        ));
                        Ok::<SourceCoverageSnapshot, anyhow::Error>(snapshot)
                    })?
                })
            }
        }
    }

    async fn handle_get_source_coverage(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetSourceCoverage>,
        cx: AsyncApp,
    ) -> Result<proto::SourceCoverageResponse> {
        #[cfg(any(test, feature = "test-support"))]
        let mut cx = cx;
        let request = envelope.payload;
        let rust_admission = this.read_with(&cx, |store, cx| {
            if request.provider_id == "cargo-llvm-cov" {
                anyhow::ensure!(
                    cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                    "Rust coverage requires local host experimental opt-in"
                );
                Ok::<_, anyhow::Error>(Some(store.rust_experimental_generation))
            } else {
                Ok(None)
            }
        })?;
        let allowed_worktrees = request
            .worktree_ids
            .into_iter()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        #[cfg(any(test, feature = "test-support"))]
        let gate_completed = if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.snapshot_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow!("source coverage snapshot start receiver was dropped"))?;
            release
                .await
                .map_err(|_| anyhow!("source coverage snapshot release was dropped"))?;
            Some(completed)
        } else {
            None
        };
        let mut response = this.read_with(&cx, |store, cx| {
            anyhow::ensure!(
                request.provider_id != "cargo-llvm-cov"
                    || cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                "Rust coverage requires local host experimental opt-in"
            );
            if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx) {
                bail!("source coverage is unavailable for restricted worktrees");
            }
            let visible_worktrees = store
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| (worktree.read(cx).id(), worktree))
                .collect::<HashMap<_, _>>();
            let snapshot = store
                .state
                .provider(&SourceCoverageProviderId(request.provider_id))
                .context("unknown source coverage provider")?;
            Ok(snapshot_to_proto_filtered(snapshot, |file| {
                allowed_worktrees.contains(&file.path.worktree_id)
                    && visible_worktrees
                        .get(&file.path.worktree_id)
                        .is_some_and(|worktree| {
                            worktree
                                .read(cx)
                                .snapshot()
                                .entry_for_path(file.path.path.as_ref())
                                .is_some_and(|entry| {
                                    entry.is_file()
                                        && !entry.is_private
                                        && !entry.is_fifo
                                        && !entry.is_external
                                })
                        })
            }))
        })?;
        #[cfg(any(test, feature = "test-support"))]
        if let Some(completed) = gate_completed {
            completed
                .send(())
                .map_err(|_| anyhow!("source coverage snapshot completion receiver was dropped"))?;
        }
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.assembled_response_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow!("source coverage response start receiver was dropped"))?;
            release
                .await
                .map_err(|_| anyhow!("source coverage response release was dropped"))?;
            completed
                .send(())
                .map_err(|_| anyhow!("source coverage response completion receiver was dropped"))?;
        }
        this.read_with(&cx, |store, cx| {
            anyhow::ensure!(
                rust_admission
                    .is_none_or(|generation| generation == store.rust_experimental_generation),
                "Rust coverage host opt-in was revoked while assembling the response"
            );
            anyhow::ensure!(
                response.provider_id != "cargo-llvm-cov"
                    || cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
                "Rust coverage requires local host experimental opt-in"
            );
            if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx) {
                response.files.clear();
                response.truncated = true;
                response.status = status_to_proto(SourceCoverageStatus::Restricted);
                return Ok(response);
            }
            let visible_worktrees = store
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| (worktree.read(cx).id(), worktree))
                .collect::<HashMap<_, _>>();
            let previous_count = response.files.len();
            response.files = response.files.into_iter().try_fold(
                Vec::with_capacity(previous_count),
                |mut files, file| {
                    let path = ProjectPath::from_proto(
                        file.path
                            .clone()
                            .context("missing source coverage file path")?,
                    )
                    .context("invalid source coverage file path")?;
                    if allowed_worktrees.contains(&path.worktree_id)
                        && visible_worktrees
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
                    {
                        files.push(file);
                    }
                    Ok::<_, anyhow::Error>(files)
                },
            )?;
            response.truncated |= response.files.len() != previous_count;
            Ok(response)
        })
    }
}

fn map_remote_source_coverage_error(error: anyhow::Error) -> anyhow::Error {
    let unsupported = error
        .downcast_ref::<proto::RpcError>()
        .is_some_and(|error| {
            error.raw_message().contains("was not handled")
                || error
                    .raw_message()
                    .starts_with("no handler registered for ")
        });
    if unsupported {
        SourceCoverageUnsupportedHost.into()
    } else {
        error
    }
}

pub fn snapshot_to_proto(
    snapshot: &SourceCoverageSnapshot,
    allowed_worktrees: Option<&HashSet<WorktreeId>>,
) -> proto::SourceCoverageResponse {
    snapshot_to_proto_filtered(snapshot, |file| {
        allowed_worktrees.is_none_or(|allowed| allowed.contains(&file.path.worktree_id))
    })
}

fn snapshot_to_proto_filtered(
    snapshot: &SourceCoverageSnapshot,
    mut include: impl FnMut(&SourceCoverageFile) -> bool,
) -> proto::SourceCoverageResponse {
    let mut filtered = false;
    let files = snapshot
        .files
        .iter()
        .filter_map(|file| {
            if !include(file) {
                filtered = true;
                return None;
            }
            Some(proto::SourceCoverageFile {
                path: Some(file.path.to_proto()),
                ranges: file
                    .ranges
                    .iter()
                    .map(|range| proto::SourceCoverageRange {
                        start_line: range.start.line,
                        start_column: range.start.column,
                        end_line: range.end.line,
                        end_column: range.end.column,
                        hit_count: range.hit_count,
                    })
                    .collect(),
                covered_lines: file.covered_lines,
                uncovered_lines: file.uncovered_lines,
                truncated: file.truncated,
            })
        })
        .collect();
    proto::SourceCoverageResponse {
        project_generation: snapshot.project_generation,
        provider_id: snapshot.provider_id.0.clone(),
        generation: snapshot.generation,
        status: status_to_proto(snapshot.status),
        files,
        truncated: snapshot.truncated || filtered,
        diagnostic: snapshot
            .diagnostic
            .as_ref()
            .map(|_| "Coverage provider details are kept on the project host.".to_owned()),
    }
}

pub fn snapshot_from_proto(
    response: proto::SourceCoverageResponse,
) -> Result<SourceCoverageSnapshot> {
    SourceCoverageSnapshot {
        project_generation: response.project_generation,
        provider_id: SourceCoverageProviderId(response.provider_id),
        generation: response.generation,
        status: status_from_proto(response.status),
        files: response
            .files
            .into_iter()
            .map(|file| {
                Ok(SourceCoverageFile {
                    path: ProjectPath::from_proto(
                        file.path.context("missing source coverage file path")?,
                    )
                    .context("invalid source coverage file path")?,
                    ranges: file
                        .ranges
                        .into_iter()
                        .map(|range| SourceCoverageRange {
                            start: SourceCoveragePoint {
                                line: range.start_line,
                                column: range.start_column,
                            },
                            end: SourceCoveragePoint {
                                line: range.end_line,
                                column: range.end_column,
                            },
                            hit_count: range.hit_count,
                        })
                        .collect(),
                    covered_lines: file.covered_lines,
                    uncovered_lines: file.uncovered_lines,
                    truncated: file.truncated,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        truncated: response.truncated,
        diagnostic: response.diagnostic,
    }
    .bounded()
}

fn status_to_proto(status: SourceCoverageStatus) -> i32 {
    match status {
        SourceCoverageStatus::Loading => 0,
        SourceCoverageStatus::Current => 1,
        SourceCoverageStatus::Empty => 2,
        SourceCoverageStatus::Partial => 3,
        SourceCoverageStatus::Stale => 4,
        SourceCoverageStatus::Error => 5,
        SourceCoverageStatus::Restricted => 6,
        SourceCoverageStatus::Disconnected => 7,
        SourceCoverageStatus::Mismatch => 8,
        SourceCoverageStatus::Cancelled => 9,
    }
}

fn status_from_proto(status: i32) -> SourceCoverageStatus {
    match status {
        1 => SourceCoverageStatus::Current,
        2 => SourceCoverageStatus::Empty,
        3 => SourceCoverageStatus::Partial,
        4 => SourceCoverageStatus::Stale,
        5 => SourceCoverageStatus::Error,
        6 => SourceCoverageStatus::Restricted,
        7 => SourceCoverageStatus::Disconnected,
        8 => SourceCoverageStatus::Mismatch,
        9 => SourceCoverageStatus::Cancelled,
        _ => SourceCoverageStatus::Loading,
    }
}

fn truncate_utf8(value: &mut String, limit: usize) {
    while value.len() > limit {
        value.pop();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;

    fn path(value: &str, worktree: usize) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(worktree),
            path: Arc::from(RelPath::from_unix_str(value).expect("valid fixture path")),
        }
    }

    fn snapshot() -> SourceCoverageSnapshot {
        SourceCoverageSnapshot {
            project_generation: 7,
            provider_id: SourceCoverageProviderId("fake-language".to_string()),
            generation: 2,
            status: SourceCoverageStatus::Current,
            files: vec![SourceCoverageFile {
                path: path("src/lib.rs", 1),
                ranges: vec![
                    SourceCoverageRange {
                        start: SourceCoveragePoint { line: 1, column: 0 },
                        end: SourceCoveragePoint { line: 1, column: 8 },
                        hit_count: 4,
                    },
                    SourceCoverageRange {
                        start: SourceCoveragePoint { line: 4, column: 0 },
                        end: SourceCoveragePoint { line: 3, column: 0 },
                        hit_count: 0,
                    },
                ],
                covered_lines: 1,
                uncovered_lines: 1,
                truncated: false,
            }],
            truncated: false,
            diagnostic: None,
        }
    }

    #[test]
    fn source_coverage_changes_distinguish_artifacts_and_restoration() {
        let mut changes = SourceCoverageChanges::default();
        let source = path("src/lib.rs", 1);
        assert!(!changes.record(&[]));
        assert_eq!(changes.revision, 0);
        assert!(!changes.record(std::slice::from_ref(&source)));
        let before_artifact = changes.revision;
        assert!(!changes.record(&[path("coverage.json", 1)]));
        assert!(!changes.changed_since(&source, before_artifact));
        assert!(!changes.record(std::slice::from_ref(&source)));
        assert!(changes.changed_since(&source, before_artifact));
        let before_restoration = changes.revision;
        assert!(!changes.record(std::slice::from_ref(&source)));
        assert!(changes.changed_since(&source, before_restoration));
        assert!(!changes.changed_since(&path("src/lib.rs", 2), before_artifact));
    }

    #[test]
    fn source_coverage_changes_expire_on_overflow_and_revision_wrap() {
        let mut changes = SourceCoverageChanges::default();
        let paths = (0..MAX_SOURCE_COVERAGE_FILES)
            .map(|index| path(&format!("source-{index}.rs"), 1))
            .collect::<Vec<_>>();
        assert!(!changes.record(&paths));
        assert_eq!(changes.paths.len(), MAX_SOURCE_COVERAGE_FILES);
        assert!(!changes.record(&[path("source-0.rs", 1)]));
        assert_eq!(changes.paths.len(), MAX_SOURCE_COVERAGE_FILES);
        assert!(changes.record(&[path("additional.rs", 1)]));
        assert_eq!(changes.paths.len(), 1);
        changes.revision = u64::MAX;
        assert!(changes.record(&[path("after-wrap.rs", 1)]));
        assert_eq!(changes.revision, 0);
        assert_eq!(changes.paths.len(), 1);
    }

    #[test]
    fn source_coverage_bounds_malformed_input_and_stale_generations() {
        let mut state = SourceCoverageState::new(7);
        state.publish(snapshot()).expect("snapshot should publish");
        let current = state
            .provider(&SourceCoverageProviderId("fake-language".to_string()))
            .expect("provider should exist");
        assert_eq!(current.files[0].ranges.len(), 1);
        assert_eq!(current.status, SourceCoverageStatus::Partial);
        let mut stale = snapshot();
        stale.generation = 1;
        assert!(state.publish(stale).is_err());

        state
            .mark_provider_status(
                SourceCoverageProviderId("fake-language".to_string()),
                SourceCoverageStatus::Stale,
                Some("cancelled".to_string()),
            )
            .expect("lifecycle transition should retain bounded facts");
        let stale = state
            .provider(&SourceCoverageProviderId("fake-language".to_string()))
            .expect("provider should remain available");
        assert_eq!(stale.status, SourceCoverageStatus::Stale);
        assert_eq!(stale.files.len(), 1);
        assert_eq!(stale.diagnostic.as_deref(), Some("cancelled"));
    }

    #[test]
    fn source_coverage_remote_round_trip_filters_hidden_worktrees() {
        let mut snapshot = snapshot().bounded().expect("snapshot should bound");
        snapshot.files.push(SourceCoverageFile {
            path: path("private.rs", 2),
            ranges: Vec::new(),
            covered_lines: 0,
            uncovered_lines: 0,
            truncated: false,
        });
        let proto = snapshot_to_proto(&snapshot, Some(&HashSet::from([WorktreeId::from_usize(1)])));
        assert_eq!(proto.files.len(), 1);
        assert!(proto.truncated);
        let decoded = snapshot_from_proto(proto).expect("filtered snapshot should decode");
        assert_eq!(decoded.files[0].path, path("src/lib.rs", 1));
    }

    #[test]
    fn source_coverage_status_wire_values_preserve_cancellation_and_unknown_fallback() {
        for (status, value) in [
            (SourceCoverageStatus::Loading, 0),
            (SourceCoverageStatus::Current, 1),
            (SourceCoverageStatus::Empty, 2),
            (SourceCoverageStatus::Partial, 3),
            (SourceCoverageStatus::Stale, 4),
            (SourceCoverageStatus::Error, 5),
            (SourceCoverageStatus::Restricted, 6),
            (SourceCoverageStatus::Disconnected, 7),
            (SourceCoverageStatus::Mismatch, 8),
            (SourceCoverageStatus::Cancelled, 9),
        ] {
            assert_eq!(status_to_proto(status), value);
            assert_eq!(status_from_proto(value), status);
        }
        for unknown in [-1, 10, i32::MAX] {
            assert_eq!(status_from_proto(unknown), SourceCoverageStatus::Loading);
        }
        let mut cancelled = snapshot();
        cancelled.status = SourceCoverageStatus::Cancelled;
        cancelled.diagnostic = Some("/private/host TOKEN=secret".into());
        let bounded = cancelled
            .clone()
            .bounded()
            .expect("bounded cancelled facts");
        assert_eq!(bounded.status, SourceCoverageStatus::Cancelled);
        assert_eq!(
            bounded
                .files
                .first()
                .expect("cancelled source")
                .ranges
                .len(),
            1
        );
        let response = snapshot_to_proto(&cancelled, None);
        assert_eq!(response.status, 9);
        assert_eq!(response.files.len(), 1);
        assert_eq!(
            response.diagnostic.as_deref(),
            Some("Coverage provider details are kept on the project host.")
        );
        let decoded = snapshot_from_proto(response).expect("cancelled response");
        assert_eq!(decoded.status, SourceCoverageStatus::Cancelled);
        assert_eq!(decoded.files, bounded.files);
        assert!(!format!("{decoded:?}").contains("TOKEN=secret"));
        assert!(!format!("{decoded:?}").contains("/private/host"));
    }

    #[test]
    fn source_coverage_remote_response_redacts_host_diagnostic() {
        let mut snapshot = snapshot().bounded().expect("bounded coverage");
        snapshot.diagnostic = Some("/Users/host/private TOKEN=secret".into());
        let response = snapshot_to_proto(&snapshot, None);
        assert_eq!(
            response.diagnostic.as_deref(),
            Some("Coverage provider details are kept on the project host.")
        );
        assert!(!format!("{response:?}").contains("/Users/host"));
        assert!(!format!("{response:?}").contains("TOKEN=secret"));
    }
}
