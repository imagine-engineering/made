use std::collections::{BTreeMap, HashSet, VecDeque};

use anyhow::{Context as _, Result, anyhow, bail};
use feature_flags::FeatureFlagAppExt as _;
use futures::future::{AbortHandle, Abortable};
use gpui::{AsyncApp, Context, Entity, EventEmitter, Task};
use rpc::{AnyProtoClient, ErrorCodeExt as _, ErrorExt as _, TypedEnvelope, proto};
use task::{StructuredTaskHandle, StructuredTaskLifecycleEvent, StructuredTaskState};

use crate::{ProjectPath, WorktreeId, worktree_store::WorktreeStore};

pub const STRUCTURED_EXECUTION_PROTOCOL_VERSION: u32 = 1;
pub const MAX_STRUCTURED_NODES: usize = 10_000;
pub const MAX_STRUCTURED_EVENTS: usize = 50_000;
pub const MAX_STRUCTURED_MESSAGE_BYTES: usize = 64 * 1024;
pub const MAX_STRUCTURED_COMPLETED_RUNS: usize = 20;
pub const MAX_STRUCTURED_PAGE_SIZE: usize = 256;
pub const MAX_STRUCTURED_EVENT_CHUNK_SIZE: usize = 512;
const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_LABEL_BYTES: usize = 4 * 1024;
const MAX_DIAGNOSTIC_BYTES: usize = 4 * 1024;
const UNCONFIRMED_CANCELLATION: &str = "Cancellation requested; host termination was not confirmed";

#[derive(Debug)]
pub struct StructuredExecutionUnsupportedHost;

impl std::fmt::Display for StructuredExecutionUnsupportedHost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("The project host does not support structured execution results")
    }
}

impl std::error::Error for StructuredExecutionUnsupportedHost {}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StructuredProviderId(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StructuredNodeId(pub String);

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StructuredRunId(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiscoveryGeneration(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredNodeKind {
    Provider,
    Suite,
    Group,
    Case,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredProviderStatus {
    Loading,
    Current,
    Empty,
    Partial,
    Stale,
    Error,
    Restricted,
    Disconnected,
    Mismatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredDebugCapability {
    Supported,
    Unsupported(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredNode {
    pub experimental_feature: Option<String>,
    pub experimental_feature_enabled: Option<bool>,
    pub debug_capability: Option<StructuredDebugCapability>,
    pub id: StructuredNodeId,
    pub parent_id: Option<StructuredNodeId>,
    pub label: String,
    pub kind: StructuredNodeKind,
    pub path: Option<ProjectPath>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredNodeState {
    Queued,
    Running,
    Passed,
    Failed,
    Skipped,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredRunPhase {
    Queued,
    Running,
    Completed,
    Cancelled,
    SpawnError,
}

impl StructuredRunPhase {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::SpawnError)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredExecutionEvent {
    pub sequence: u64,
    pub node_id: StructuredNodeId,
    pub state: StructuredNodeState,
    pub duration_millis: Option<u64>,
    pub message: Option<String>,
    pub location: Option<ProjectPath>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StructuredExecutionSummary {
    pub total: u32,
    pub queued: u32,
    pub running: u32,
    pub passed: u32,
    pub failed: u32,
    pub skipped: u32,
    pub cancelled: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredRun {
    pub id: StructuredRunId,
    pub discovery_generation: DiscoveryGeneration,
    pub phase: StructuredRunPhase,
    pub scope_node_ids: Vec<StructuredNodeId>,
    pub summary: StructuredExecutionSummary,
    pub events: Vec<StructuredExecutionEvent>,
    pub next_sequence: u64,
    pub truncated: bool,
    pub diagnostic: Option<String>,
    message_bytes: usize,
    latest_states: BTreeMap<StructuredNodeId, StructuredNodeState>,
}

impl StructuredRun {
    pub fn new(
        id: StructuredRunId,
        discovery_generation: DiscoveryGeneration,
        scope_node_ids: Vec<StructuredNodeId>,
    ) -> Self {
        Self {
            id,
            discovery_generation,
            phase: StructuredRunPhase::Queued,
            scope_node_ids,
            summary: StructuredExecutionSummary::default(),
            events: Vec::new(),
            next_sequence: 0,
            truncated: false,
            diagnostic: None,
            message_bytes: 0,
            latest_states: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredProviderSnapshot {
    pub provider_id: StructuredProviderId,
    pub discovery_generation: DiscoveryGeneration,
    pub status: StructuredProviderStatus,
    pub nodes: Vec<StructuredNode>,
    pub partial: bool,
    pub diagnostic: Option<String>,
    pub current_run: Option<StructuredRun>,
    pub last_complete_run: Option<StructuredRun>,
    pub completed_runs: VecDeque<StructuredRun>,
}

impl StructuredProviderSnapshot {
    pub fn project_path_for_node(&self, identifier: &StructuredNodeId) -> Option<&ProjectPath> {
        let mut current = identifier;
        for _ in 0..64 {
            let node = self.nodes.iter().find(|node| &node.id == current)?;
            if let Some(path) = &node.path {
                return Some(path);
            }
            current = node.parent_id.as_ref()?;
        }
        None
    }

    pub fn discovery(
        provider_id: StructuredProviderId,
        discovery_generation: DiscoveryGeneration,
        status: StructuredProviderStatus,
        nodes: Vec<StructuredNode>,
    ) -> Self {
        Self {
            provider_id,
            discovery_generation,
            status,
            nodes,
            partial: false,
            diagnostic: None,
            current_run: None,
            last_complete_run: None,
            completed_runs: VecDeque::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructuredExecutionLimits {
    pub nodes: usize,
    pub events_per_run: usize,
    pub message_bytes_per_run: usize,
    pub completed_runs: usize,
}

impl Default for StructuredExecutionLimits {
    fn default() -> Self {
        Self {
            nodes: MAX_STRUCTURED_NODES,
            events_per_run: MAX_STRUCTURED_EVENTS,
            message_bytes_per_run: MAX_STRUCTURED_MESSAGE_BYTES,
            completed_runs: MAX_STRUCTURED_COMPLETED_RUNS,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredApplyOutcome {
    Applied,
    Duplicate,
    Truncated,
}

#[derive(Clone, Debug)]
pub struct StructuredExecutionState {
    project_generation: u64,
    limits: StructuredExecutionLimits,
    providers: BTreeMap<StructuredProviderId, StructuredProviderSnapshot>,
    provider_node_ids: BTreeMap<StructuredProviderId, HashSet<StructuredNodeId>>,
}

impl StructuredExecutionState {
    pub fn new(project_generation: u64) -> Self {
        Self::with_limits(project_generation, StructuredExecutionLimits::default())
    }

    pub fn with_limits(project_generation: u64, limits: StructuredExecutionLimits) -> Self {
        Self {
            project_generation,
            limits,
            providers: BTreeMap::new(),
            provider_node_ids: BTreeMap::new(),
        }
    }

    pub fn project_generation(&self) -> u64 {
        self.project_generation
    }

    pub fn replace_project_generation(&mut self, project_generation: u64) {
        if self.project_generation != project_generation {
            self.project_generation = project_generation;
            self.providers.clear();
            self.provider_node_ids.clear();
        }
    }

    pub fn providers(&self) -> impl Iterator<Item = &StructuredProviderSnapshot> {
        self.providers.values()
    }

    pub fn provider(
        &self,
        provider_id: &StructuredProviderId,
    ) -> Option<&StructuredProviderSnapshot> {
        self.providers.get(provider_id)
    }

    pub fn apply_discovery(
        &mut self,
        project_generation: u64,
        mut snapshot: StructuredProviderSnapshot,
        visible_worktrees: Option<&HashSet<WorktreeId>>,
    ) -> Result<StructuredApplyOutcome> {
        self.require_project_generation(project_generation)?;
        validate_identifier("provider ID", &snapshot.provider_id.0)?;
        let (nodes, was_filtered, diagnostic) = bounded_nodes(
            std::mem::take(&mut snapshot.nodes),
            visible_worktrees,
            self.limits.nodes,
        )?;
        snapshot.nodes = nodes;
        snapshot.partial |= was_filtered;
        if snapshot.partial && snapshot.status == StructuredProviderStatus::Current {
            snapshot.status = StructuredProviderStatus::Partial;
        }
        if snapshot.diagnostic.is_none() {
            snapshot.diagnostic = diagnostic;
        } else {
            snapshot.diagnostic = snapshot
                .diagnostic
                .map(|value| bounded_text(&value, MAX_DIAGNOSTIC_BYTES));
        }
        if let Some(existing) = self.providers.get(&snapshot.provider_id) {
            if snapshot.discovery_generation < existing.discovery_generation {
                bail!("stale structured discovery generation");
            }
            if snapshot.discovery_generation == existing.discovery_generation {
                if existing.status == snapshot.status
                    && existing.nodes == snapshot.nodes
                    && existing.partial == snapshot.partial
                    && existing.diagnostic == snapshot.diagnostic
                {
                    return Ok(StructuredApplyOutcome::Duplicate);
                }
                if existing.status == snapshot.status
                    && existing.partial == snapshot.partial
                    && existing.diagnostic == snapshot.diagnostic
                    && existing.current_run == snapshot.current_run
                    && existing.last_complete_run == snapshot.last_complete_run
                    && existing.completed_runs == snapshot.completed_runs
                    && existing.nodes.len() == snapshot.nodes.len()
                    && existing
                        .nodes
                        .iter()
                        .zip(&snapshot.nodes)
                        .all(|(existing, incoming)| {
                            nodes_match_except_experimental_availability(existing, incoming)
                        })
                {
                    let retained = self
                        .providers
                        .get_mut(&snapshot.provider_id)
                        .context("structured provider disappeared during availability renewal")?;
                    for (retained, incoming) in retained.nodes.iter_mut().zip(&snapshot.nodes) {
                        retained.experimental_feature_enabled =
                            incoming.experimental_feature_enabled;
                    }
                    return Ok(StructuredApplyOutcome::Applied);
                }
                bail!("conflicting structured discovery generation");
            }
            snapshot.last_complete_run = existing.last_complete_run.clone();
            snapshot.completed_runs = existing.completed_runs.clone();
        }
        snapshot.current_run = None;
        let outcome = if was_filtered {
            StructuredApplyOutcome::Truncated
        } else {
            StructuredApplyOutcome::Applied
        };
        let provider_id = snapshot.provider_id.clone();
        self.provider_node_ids.insert(
            provider_id.clone(),
            snapshot.nodes.iter().map(|node| node.id.clone()).collect(),
        );
        self.providers.insert(provider_id, snapshot);
        Ok(outcome)
    }

    pub fn begin_run(
        &mut self,
        project_generation: u64,
        provider_id: &StructuredProviderId,
        mut run: StructuredRun,
    ) -> Result<StructuredApplyOutcome> {
        self.require_project_generation(project_generation)?;
        validate_identifier("run ID", &run.id.0)?;
        let limits = self.limits;
        let provider = self
            .providers
            .get_mut(provider_id)
            .context("unknown structured execution provider")?;
        if run.discovery_generation != provider.discovery_generation {
            bail!("run uses a stale structured discovery generation");
        }
        let known_nodes = self
            .provider_node_ids
            .get(provider_id)
            .context("structured provider node index is missing")?;
        run.scope_node_ids.retain(|id| known_nodes.contains(id));
        run.scope_node_ids.sort();
        run.scope_node_ids.dedup();
        if run.scope_node_ids.is_empty() {
            bail!("structured run has no visible scope nodes");
        }
        if let Some(current) = provider.current_run.as_ref() {
            if current.id == run.id {
                return Ok(StructuredApplyOutcome::Duplicate);
            }
        }
        if let Some(mut superseded) = provider.current_run.take()
            && !superseded.phase.is_terminal()
        {
            superseded.phase = StructuredRunPhase::Cancelled;
            superseded.diagnostic = Some("Superseded by a newer structured run".to_string());
            retain_completed_run(provider, superseded, limits.completed_runs);
        }
        run.events.clear();
        run.next_sequence = 0;
        run.summary = StructuredExecutionSummary::default();
        run.message_bytes = 0;
        run.latest_states.clear();
        provider.current_run = Some(run);
        Ok(StructuredApplyOutcome::Applied)
    }

    pub fn set_run_phase(
        &mut self,
        project_generation: u64,
        provider_id: &StructuredProviderId,
        run_id: &StructuredRunId,
        phase: StructuredRunPhase,
        diagnostic: Option<&str>,
    ) -> Result<StructuredApplyOutcome> {
        self.require_project_generation(project_generation)?;
        let limits = self.limits;
        let provider = self
            .providers
            .get_mut(provider_id)
            .context("unknown structured execution provider")?;
        let run = provider
            .current_run
            .as_mut()
            .filter(|run| &run.id == run_id)
            .context("stale or unknown structured run")?;
        if run.phase == phase {
            return Ok(StructuredApplyOutcome::Duplicate);
        }
        if run.phase.is_terminal() {
            bail!("structured run is already terminal");
        }
        run.phase = phase;
        run.diagnostic = diagnostic.map(|value| bounded_text(value, MAX_DIAGNOSTIC_BYTES));
        if phase.is_terminal() {
            let completed = run.clone();
            if phase == StructuredRunPhase::Completed {
                provider.last_complete_run = Some(completed.clone());
            }
            retain_completed_run(provider, completed, limits.completed_runs);
        }
        Ok(StructuredApplyOutcome::Applied)
    }

    pub fn apply_event(
        &mut self,
        project_generation: u64,
        provider_id: &StructuredProviderId,
        run_id: &StructuredRunId,
        mut event: StructuredExecutionEvent,
        visible_worktrees: Option<&HashSet<WorktreeId>>,
    ) -> Result<StructuredApplyOutcome> {
        self.require_project_generation(project_generation)?;
        validate_identifier("node ID", &event.node_id.0)?;
        let limits = self.limits;
        let known_nodes = self
            .provider_node_ids
            .get(provider_id)
            .context("structured provider node index is missing")?;
        if !known_nodes.contains(&event.node_id) {
            bail!("structured event references an unknown node");
        }
        let provider = self
            .providers
            .get_mut(provider_id)
            .context("unknown structured execution provider")?;
        let run = provider
            .current_run
            .as_mut()
            .filter(|run| &run.id == run_id)
            .context("stale or unknown structured run")?;
        if run.discovery_generation != provider.discovery_generation {
            bail!("structured event uses a stale discovery generation");
        }
        if event.sequence < run.next_sequence {
            return if run.events.iter().any(|existing| existing == &event) {
                Ok(StructuredApplyOutcome::Duplicate)
            } else {
                bail!("conflicting duplicate structured event")
            };
        }
        if event.sequence > run.next_sequence {
            bail!("structured event sequence gap");
        }
        run.next_sequence = run.next_sequence.saturating_add(1);
        if event
            .location
            .as_ref()
            .is_some_and(|path| !path_is_visible(path, visible_worktrees))
        {
            run.truncated = true;
            return Ok(StructuredApplyOutcome::Truncated);
        }
        if let Some(message) = event.message.take() {
            let original_message_bytes = message.len();
            let remaining = limits
                .message_bytes_per_run
                .saturating_sub(run.message_bytes);
            if remaining == 0 {
                run.truncated = true;
            } else {
                let message = bounded_text(&message, remaining.min(MAX_DIAGNOSTIC_BYTES));
                run.message_bytes = run.message_bytes.saturating_add(message.len());
                if message.len() < original_message_bytes
                    || run.message_bytes >= limits.message_bytes_per_run
                {
                    run.truncated = true;
                }
                event.message = Some(message);
            }
        }
        if run.events.len() >= limits.events_per_run {
            run.truncated = true;
            return Ok(StructuredApplyOutcome::Truncated);
        }
        update_summary(
            &mut run.summary,
            run.latest_states.insert(event.node_id.clone(), event.state),
            event.state,
        );
        run.events.push(event);
        Ok(if run.truncated {
            StructuredApplyOutcome::Truncated
        } else {
            StructuredApplyOutcome::Applied
        })
    }

    fn require_project_generation(&self, project_generation: u64) -> Result<()> {
        if project_generation != self.project_generation {
            bail!("stale or cross-project structured execution generation");
        }
        Ok(())
    }

    fn replace_remote_snapshot(
        &mut self,
        project_generation: u64,
        snapshot: StructuredProviderSnapshot,
    ) {
        self.replace_project_generation(project_generation);
        let provider_id = snapshot.provider_id.clone();
        self.provider_node_ids.insert(
            provider_id.clone(),
            snapshot.nodes.iter().map(|node| node.id.clone()).collect(),
        );
        self.providers.insert(provider_id, snapshot);
    }
}

fn nodes_match_except_experimental_availability(
    existing: &StructuredNode,
    incoming: &StructuredNode,
) -> bool {
    existing.id == incoming.id
        && existing.parent_id == incoming.parent_id
        && existing.label == incoming.label
        && existing.kind == incoming.kind
        && existing.path == incoming.path
        && existing.debug_capability == incoming.debug_capability
        && existing.experimental_feature == incoming.experimental_feature
        && (existing.experimental_feature_enabled == incoming.experimental_feature_enabled
            || existing
                .experimental_feature
                .as_deref()
                .is_some_and(|feature| {
                    task::validate_experimental_feature_requirement(feature).is_ok()
                }))
}

fn retain_completed_run(
    provider: &mut StructuredProviderSnapshot,
    run: StructuredRun,
    retention: usize,
) {
    provider.completed_runs.push_back(run);
    while provider.completed_runs.len() > retention {
        provider.completed_runs.pop_front();
    }
}

fn summarize_events(events: &[StructuredExecutionEvent]) -> StructuredExecutionSummary {
    let mut latest = BTreeMap::<&StructuredNodeId, StructuredNodeState>::new();
    for event in events {
        latest.insert(&event.node_id, event.state);
    }
    let mut summary = StructuredExecutionSummary {
        total: latest.len().try_into().unwrap_or(u32::MAX),
        ..StructuredExecutionSummary::default()
    };
    for state in latest.into_values() {
        match state {
            StructuredNodeState::Queued => summary.queued = summary.queued.saturating_add(1),
            StructuredNodeState::Running => summary.running = summary.running.saturating_add(1),
            StructuredNodeState::Passed => summary.passed = summary.passed.saturating_add(1),
            StructuredNodeState::Failed => summary.failed = summary.failed.saturating_add(1),
            StructuredNodeState::Skipped => summary.skipped = summary.skipped.saturating_add(1),
            StructuredNodeState::Cancelled => {
                summary.cancelled = summary.cancelled.saturating_add(1)
            }
        }
    }
    summary
}

fn update_summary(
    summary: &mut StructuredExecutionSummary,
    previous: Option<StructuredNodeState>,
    current: StructuredNodeState,
) {
    if previous == Some(current) {
        return;
    }
    if let Some(previous) = previous {
        update_summary_count(summary, previous, false);
    } else {
        summary.total = summary.total.saturating_add(1);
    }
    update_summary_count(summary, current, true);
}

fn update_summary_count(
    summary: &mut StructuredExecutionSummary,
    state: StructuredNodeState,
    increment: bool,
) {
    let count = match state {
        StructuredNodeState::Queued => &mut summary.queued,
        StructuredNodeState::Running => &mut summary.running,
        StructuredNodeState::Passed => &mut summary.passed,
        StructuredNodeState::Failed => &mut summary.failed,
        StructuredNodeState::Skipped => &mut summary.skipped,
        StructuredNodeState::Cancelled => &mut summary.cancelled,
    };
    *count = if increment {
        count.saturating_add(1)
    } else {
        count.saturating_sub(1)
    };
}

fn rebuild_run_summary(run: &mut StructuredRun) {
    run.latest_states.clear();
    for event in &run.events {
        run.latest_states.insert(event.node_id.clone(), event.state);
    }
    run.summary = summarize_events(&run.events);
}

fn bounded_nodes(
    nodes: Vec<StructuredNode>,
    visible_worktrees: Option<&HashSet<WorktreeId>>,
    limit: usize,
) -> Result<(Vec<StructuredNode>, bool, Option<String>)> {
    let mut result = Vec::new();
    let mut retained_ids = HashSet::new();
    let mut seen_ids = HashSet::new();
    let mut partial = false;
    let mut diagnostic = None;
    for mut node in nodes {
        validate_identifier("node ID", &node.id.0)?;
        if let Some(feature) = node.experimental_feature.as_deref() {
            task::validate_experimental_feature_requirement(feature)?;
        }
        if !seen_ids.insert(node.id.clone()) {
            bail!("duplicate structured node ID");
        }
        node.label = bounded_text(&node.label, MAX_LABEL_BYTES);
        if let Some(StructuredDebugCapability::Unsupported(reason)) = &mut node.debug_capability {
            if reason.trim().is_empty() {
                bail!("unsupported Debug capability has an empty reason");
            }
            *reason = bounded_text(reason, MAX_DIAGNOSTIC_BYTES);
        }
        let parent_visible = node
            .parent_id
            .as_ref()
            .is_none_or(|parent| retained_ids.contains(parent));
        let path_visible = node
            .path
            .as_ref()
            .is_none_or(|path| path_is_visible(path, visible_worktrees));
        if !parent_visible || !path_visible {
            partial = true;
            diagnostic.get_or_insert_with(|| {
                "Some structured result nodes were filtered or had missing parents".to_string()
            });
            continue;
        }
        if result.len() >= limit {
            partial = true;
            diagnostic.get_or_insert_with(|| {
                format!("Structured result nodes were truncated at {limit} entries")
            });
            continue;
        }
        retained_ids.insert(node.id.clone());
        result.push(node);
    }
    Ok((result, partial, diagnostic))
}

fn filter_nodes(
    nodes: &[StructuredNode],
    visible_worktrees: Option<&HashSet<WorktreeId>>,
) -> Result<Vec<StructuredNode>> {
    bounded_nodes(nodes.to_vec(), visible_worktrees, MAX_STRUCTURED_NODES)
        .map(|(nodes, _, _)| nodes)
}

fn filter_nodes_for_peer(
    nodes: &[StructuredNode],
    visible_worktrees: Option<&HashSet<WorktreeId>>,
    include_path: &dyn Fn(&ProjectPath) -> bool,
) -> Result<Vec<StructuredNode>> {
    let nodes = filter_nodes(nodes, visible_worktrees)?;
    let mut retained_ids = HashSet::new();
    Ok(nodes
        .into_iter()
        .filter(|node| {
            if !node
                .parent_id
                .as_ref()
                .is_none_or(|parent| retained_ids.contains(parent))
                || !node.path.as_ref().is_none_or(include_path)
            {
                return false;
            }
            retained_ids.insert(node.id.clone());
            true
        })
        .collect())
}

fn path_is_visible(path: &ProjectPath, visible_worktrees: Option<&HashSet<WorktreeId>>) -> bool {
    visible_worktrees.is_none_or(|worktrees| worktrees.contains(&path.worktree_id))
}

fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > MAX_IDENTIFIER_BYTES {
        bail!("{label} is empty or exceeds {MAX_IDENTIFIER_BYTES} bytes");
    }
    Ok(())
}

fn bounded_text(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes.min(value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

#[derive(Clone, Debug)]
pub enum StructuredExecutionStoreEvent {
    Changed(StructuredProviderId),
    ProjectGenerationChanged(u64),
}

impl EventEmitter<StructuredExecutionStoreEvent> for StructuredExecutionStore {}

enum StructuredExecutionStoreMode {
    Local {
        shared: Option<(u64, AnyProtoClient)>,
    },
    Remote {
        project_id: u64,
        client: AnyProtoClient,
    },
}

pub struct StructuredExecutionStore {
    mode: StructuredExecutionStoreMode,
    worktree_store: Entity<WorktreeStore>,
    state: StructuredExecutionState,
    disconnected: bool,
    refresh_generation: u64,
    active_refreshes: BTreeMap<StructuredProviderId, (u64, AbortHandle)>,
    #[cfg(any(test, feature = "test-support"))]
    snapshot_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    events_gate: Option<(
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

impl StructuredExecutionStore {
    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_get_snapshot);
        client.add_entity_request_handler(Self::handle_get_events);
    }

    pub fn local(worktree_store: Entity<WorktreeStore>, project_generation: u64) -> Self {
        Self {
            mode: StructuredExecutionStoreMode::Local { shared: None },
            worktree_store,
            state: StructuredExecutionState::new(project_generation),
            disconnected: false,
            refresh_generation: 0,
            active_refreshes: BTreeMap::new(),
            #[cfg(any(test, feature = "test-support"))]
            snapshot_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            events_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            assembled_response_gate: None,
        }
    }

    pub fn remote(
        project_id: u64,
        client: AnyProtoClient,
        worktree_store: Entity<WorktreeStore>,
    ) -> Self {
        Self {
            mode: StructuredExecutionStoreMode::Remote { project_id, client },
            worktree_store,
            state: StructuredExecutionState::new(0),
            disconnected: false,
            refresh_generation: 0,
            active_refreshes: BTreeMap::new(),
            #[cfg(any(test, feature = "test-support"))]
            snapshot_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            events_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            assembled_response_gate: None,
        }
    }

    pub fn state(&self) -> &StructuredExecutionState {
        &self.state
    }

    pub fn has_active_work(&self) -> bool {
        !self.active_refreshes.is_empty()
            || self.state.providers().any(|provider| {
                provider
                    .current_run
                    .as_ref()
                    .is_some_and(|run| !run.phase.is_terminal())
            })
    }

    pub fn state_mut(&mut self) -> &mut StructuredExecutionState {
        &mut self.state
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
    pub fn gate_next_events_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
        self.events_gate = Some((started_sender, release_receiver, completed_sender));
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

    pub fn shared(&mut self, project_id: u64, client: AnyProtoClient) {
        if let StructuredExecutionStoreMode::Local { shared } = &mut self.mode {
            *shared = Some((project_id, client));
        }
    }

    pub fn unshared(&mut self) {
        if let StructuredExecutionStoreMode::Local { shared } = &mut self.mode {
            *shared = None;
        }
    }

    pub fn observe_task_handle(
        &mut self,
        provider_id: StructuredProviderId,
        run_id: StructuredRunId,
        handle: &StructuredTaskHandle,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if !matches!(self.mode, StructuredExecutionStoreMode::Local { .. }) {
            bail!("structured task lifecycle must be observed by the authoritative project host");
        }
        let project_generation = self.state.project_generation();
        let store = cx.entity().downgrade();
        handle.subscribe(cx, move |event, cx| {
            let event = event.clone();
            let store = store.clone();
            let provider_id = provider_id.clone();
            let run_id = run_id.clone();
            cx.defer(move |cx| {
                let update = store.update(cx, |store, cx| {
                    match apply_task_lifecycle(
                        &mut store.state,
                        project_generation,
                        &provider_id,
                        &run_id,
                        &event,
                    ) {
                        Ok(_) => {
                            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()))
                        }
                        Err(error) => {
                            log::warn!("Rejected structured task lifecycle event: {error:#}")
                        }
                    }
                });
                if let Err(error) = update {
                    log::debug!("Structured execution store was dropped: {error:#}");
                }
            });
        });
        Ok(())
    }

    pub fn apply_task_lifecycle(
        &mut self,
        provider_id: &StructuredProviderId,
        run_id: &StructuredRunId,
        event: &StructuredTaskLifecycleEvent,
        cx: &mut Context<Self>,
    ) -> Result<StructuredApplyOutcome> {
        if !matches!(self.mode, StructuredExecutionStoreMode::Local { .. }) {
            bail!("structured task lifecycle must be applied by the authoritative project host");
        }
        let project_generation = self.state.project_generation();
        let outcome = apply_task_lifecycle(
            &mut self.state,
            project_generation,
            provider_id,
            run_id,
            event,
        )?;
        cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
        Ok(outcome)
    }

    pub fn reconnected_to_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, StructuredExecutionStoreMode::Remote { .. }) || !self.disconnected {
            return;
        }
        self.disconnected = false;
        for provider in self.state.providers.values_mut() {
            provider.status = StructuredProviderStatus::Stale;
            provider.diagnostic = Some("Connection restored; refresh results".into());
            cx.emit(StructuredExecutionStoreEvent::Changed(
                provider.provider_id.clone(),
            ));
        }
    }

    pub fn disconnected_from_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, StructuredExecutionStoreMode::Remote { .. }) {
            return;
        }
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        for (_, (_, abort)) in std::mem::take(&mut self.active_refreshes) {
            abort.abort();
        }
        for provider in self.state.providers.values_mut() {
            provider.status = StructuredProviderStatus::Disconnected;
            provider.diagnostic = Some("The structured execution host is disconnected".into());
            cx.emit(StructuredExecutionStoreEvent::Changed(
                provider.provider_id.clone(),
            ));
        }
    }

    pub fn refresh_provider(
        &mut self,
        provider_id: StructuredProviderId,
        cx: &mut Context<Self>,
    ) -> Task<Result<StructuredProviderSnapshot>> {
        if self.disconnected {
            return Task::ready(Err(anyhow!(
                "The structured execution host is disconnected"
            )));
        }
        match &self.mode {
            StructuredExecutionStoreMode::Local { .. } => Task::ready(
                self.state
                    .provider(&provider_id)
                    .cloned()
                    .context("unknown structured execution provider"),
            ),
            StructuredExecutionStoreMode::Remote { project_id, client } => {
                let project_id = *project_id;
                let client = client.clone();
                let worktree_ids = self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .map(|worktree| worktree.read(cx).id().to_proto())
                    .collect::<Vec<_>>();
                self.refresh_generation = self.refresh_generation.wrapping_add(1);
                let generation = self.refresh_generation;
                let (abort, registration) = AbortHandle::new_pair();
                if let Some((_, previous)) = self
                    .active_refreshes
                    .insert(provider_id.clone(), (generation, abort))
                {
                    previous.abort();
                }
                cx.spawn(async move |this, cx| {
                    let result = Abortable::new(
                        fetch_remote_provider(
                            project_id,
                            client,
                            provider_id.clone(),
                            worktree_ids,
                        ),
                        registration,
                    )
                    .await
                    .map_err(|_| {
                        anyhow!("Structured result refresh was cancelled or the host disconnected")
                    })?;
                    this.update(cx, |store, cx| {
                        if store.disconnected
                            || store
                                .active_refreshes
                                .get(&provider_id)
                                .map(|(current, _)| *current)
                                != Some(generation)
                        {
                            bail!("stale structured result refresh");
                        }
                        store.active_refreshes.remove(&provider_id);
                        let (project_generation, snapshot) = result?;
                        store
                            .state
                            .replace_remote_snapshot(project_generation, snapshot.clone());
                        cx.emit(StructuredExecutionStoreEvent::Changed(provider_id));
                        Ok(snapshot)
                    })?
                })
            }
        }
    }

    async fn handle_get_snapshot(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetStructuredExecutionSnapshot>,
        mut cx: AsyncApp,
    ) -> Result<proto::StructuredExecutionSnapshotPage> {
        #[cfg(any(test, feature = "test-support"))]
        let completed = if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.snapshot_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow!("structured snapshot start receiver was dropped"))?;
            release
                .await
                .map_err(|_| anyhow!("structured snapshot release was dropped"))?;
            Some(completed)
        } else {
            None
        };
        let request = envelope.payload;
        let provider_id = StructuredProviderId(request.provider_id.clone());
        let needs_remote_refresh = this.read_with(&cx, |store, _| {
            matches!(&store.mode, StructuredExecutionStoreMode::Remote { .. })
                && store.state.provider(&provider_id).is_none()
        });
        if needs_remote_refresh {
            this.update(&mut cx, |store, cx| store.refresh_provider(provider_id, cx))
                .await
                .map_err(tag_remote_structured_unsupported)?;
        }
        let allowed_worktrees = request
            .worktree_ids
            .iter()
            .cloned()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        let assembled = this.read_with(&cx, |store, cx| {
            store.snapshot_page_for_peer(&request, &allowed_worktrees, cx)
        });
        #[cfg(any(test, feature = "test-support"))]
        if let Some(completed) = completed
            && completed.send(()).is_err()
        {
            log::debug!("structured snapshot completion receiver was dropped");
        }
        assembled?;
        #[cfg(any(test, feature = "test-support"))]
        let assembled_completed = if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.assembled_response_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow!("structured response start receiver was dropped"))?;
            release
                .await
                .map_err(|_| anyhow!("structured response release was dropped"))?;
            Some(completed)
        } else {
            None
        };
        // Source visibility can change after a page is assembled, so check it again immediately before replying.
        let result = this.read_with(&cx, |store, cx| {
            store.snapshot_page_for_peer(&request, &allowed_worktrees, cx)
        });
        #[cfg(any(test, feature = "test-support"))]
        if let Some(completed) = assembled_completed
            && completed.send(()).is_err()
        {
            log::debug!("structured response completion receiver was dropped");
        }
        result
    }

    fn snapshot_page_for_peer(
        &self,
        request: &proto::GetStructuredExecutionSnapshot,
        allowed_worktrees: &HashSet<WorktreeId>,
        cx: &gpui::App,
    ) -> Result<proto::StructuredExecutionSnapshotPage> {
        let worktrees = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| (worktree.read(cx).id(), worktree))
            .collect::<std::collections::HashMap<_, _>>();
        let include_path = |path: &ProjectPath| {
            allowed_worktrees.contains(&path.worktree_id)
                && worktrees.get(&path.worktree_id).is_some_and(|worktree| {
                    worktree
                        .read(cx)
                        .snapshot()
                        .entry_for_path(path.path.as_ref())
                        .is_some_and(|entry| {
                            !entry.is_private && !entry.is_fifo && !entry.is_external
                        })
                })
        };
        let mut page = snapshot_page_to_proto_filtered(
            &self.state,
            &StructuredProviderId(request.provider_id.clone()),
            DiscoveryGeneration(request.discovery_generation),
            request.page_start as usize,
            request.page_size as usize,
            Some(allowed_worktrees),
            Some(&include_path),
        )?;
        if matches!(&self.mode, StructuredExecutionStoreMode::Local { .. }) {
            for node in &mut page.nodes {
                if let Some(feature) = node.experimental_feature.as_deref() {
                    node.experimental_feature_enabled = Some(
                        task::validate_experimental_feature_requirement(feature).is_ok()
                            && cx.has_flag_by_name(feature),
                    );
                }
            }
        }
        redact_remote_diagnostics(&mut page);
        Ok(page)
    }

    async fn handle_get_events(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetStructuredExecutionEvents>,
        cx: AsyncApp,
    ) -> Result<proto::StructuredExecutionEventChunk> {
        #[cfg(any(test, feature = "test-support"))]
        let mut cx = cx;
        #[cfg(any(test, feature = "test-support"))]
        let completed = if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.events_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow!("structured events start receiver was dropped"))?;
            release
                .await
                .map_err(|_| anyhow!("structured events release was dropped"))?;
            Some(completed)
        } else {
            None
        };
        let request = envelope.payload;
        let allowed_worktrees = request
            .worktree_ids
            .into_iter()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        let result = this.read_with(&cx, |store, cx| {
            let worktrees = store
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| (worktree.read(cx).id(), worktree))
                .collect::<std::collections::HashMap<_, _>>();
            let include_path = |path: &ProjectPath| {
                allowed_worktrees.contains(&path.worktree_id)
                    && worktrees.get(&path.worktree_id).is_some_and(|worktree| {
                        worktree
                            .read(cx)
                            .snapshot()
                            .entry_for_path(path.path.as_ref())
                            .is_some_and(|entry| {
                                !entry.is_private && !entry.is_fifo && !entry.is_external
                            })
                    })
            };
            let mut chunk = event_chunk_to_proto(
                &store.state,
                &StructuredProviderId(request.provider_id),
                &StructuredRunId(request.run_id),
                request.start_sequence,
                request.chunk_size as usize,
                Some(&allowed_worktrees),
                Some(&include_path),
            )?;
            for event in &mut chunk.events {
                if event.message.is_some() {
                    event.message =
                        Some("Structured event details remain on the project host.".into());
                }
            }
            Ok(chunk)
        });
        #[cfg(any(test, feature = "test-support"))]
        if let Some(completed) = completed
            && completed.send(()).is_err()
        {
            log::debug!("structured events completion receiver was dropped");
        }
        result
    }
}

fn apply_task_lifecycle(
    state: &mut StructuredExecutionState,
    project_generation: u64,
    provider_id: &StructuredProviderId,
    run_id: &StructuredRunId,
    event: &StructuredTaskLifecycleEvent,
) -> Result<StructuredApplyOutcome> {
    let single_case = state.provider(provider_id).and_then(|provider| {
        let run = provider
            .current_run
            .as_ref()
            .filter(|run| &run.id == run_id)?;
        let node_id = run
            .scope_node_ids
            .first()
            .filter(|_| run.scope_node_ids.len() == 1)?;
        provider
            .nodes
            .iter()
            .find(|node| &node.id == node_id && node.kind == StructuredNodeKind::Case)
            .map(|node| (node.id.clone(), run.next_sequence))
    });
    let (phase, diagnostic) = match &event.state {
        StructuredTaskState::Queued => (StructuredRunPhase::Queued, None),
        StructuredTaskState::Running { .. } => (StructuredRunPhase::Running, None),
        StructuredTaskState::Completed {
            exit_code, success, ..
        } => (
            StructuredRunPhase::Completed,
            (!success).then(|| match exit_code {
                Some(exit_code) => format!("Task exited with status {exit_code}"),
                None => "Task exited unsuccessfully without a status code".to_string(),
            }),
        ),
        StructuredTaskState::SpawnError { message } => {
            (StructuredRunPhase::SpawnError, Some(message.clone()))
        }
        StructuredTaskState::Cancelled {
            termination_confirmed,
            ..
        } => (
            StructuredRunPhase::Cancelled,
            (!termination_confirmed).then(|| UNCONFIRMED_CANCELLATION.to_string()),
        ),
    };
    let single_case_event = single_case.map(|(node_id, sequence)| {
        let node_state = match &event.state {
            StructuredTaskState::Queued => StructuredNodeState::Queued,
            StructuredTaskState::Running { .. } => StructuredNodeState::Running,
            StructuredTaskState::Completed { success: true, .. } => StructuredNodeState::Passed,
            StructuredTaskState::Completed { success: false, .. }
            | StructuredTaskState::SpawnError { .. } => StructuredNodeState::Failed,
            StructuredTaskState::Cancelled { .. } => StructuredNodeState::Cancelled,
        };
        StructuredExecutionEvent {
            sequence,
            node_id,
            state: node_state,
            duration_millis: None,
            message: diagnostic.clone(),
            location: None,
        }
    });
    if phase.is_terminal() {
        let current_phase = state
            .provider(provider_id)
            .and_then(|provider| provider.current_run.as_ref())
            .filter(|run| &run.id == run_id)
            .map(|run| run.phase)
            .context("stale or unknown structured run")?;
        if current_phase.is_terminal() {
            if current_phase == StructuredRunPhase::Cancelled
                && matches!(
                    &event.state,
                    StructuredTaskState::Cancelled {
                        termination_confirmed: true,
                        ..
                    }
                )
            {
                return confirm_cancelled_run(state, project_generation, provider_id, run_id);
            }
            bail!("structured run is already terminal");
        }
        if let Some(single_case_event) = single_case_event {
            state.apply_event(
                project_generation,
                provider_id,
                run_id,
                single_case_event,
                None,
            )?;
        }
        return state.set_run_phase(
            project_generation,
            provider_id,
            run_id,
            phase,
            diagnostic.as_deref(),
        );
    }
    let outcome = state.set_run_phase(
        project_generation,
        provider_id,
        run_id,
        phase,
        diagnostic.as_deref(),
    )?;
    if let Some(single_case_event) = single_case_event {
        state.apply_event(
            project_generation,
            provider_id,
            run_id,
            single_case_event,
            None,
        )?;
    }
    Ok(outcome)
}

fn confirm_cancelled_run(
    state: &mut StructuredExecutionState,
    project_generation: u64,
    provider_id: &StructuredProviderId,
    run_id: &StructuredRunId,
) -> Result<StructuredApplyOutcome> {
    state.require_project_generation(project_generation)?;
    let provider = state
        .providers
        .get_mut(provider_id)
        .context("unknown structured execution provider")?;
    let run = provider
        .current_run
        .as_mut()
        .filter(|run| &run.id == run_id)
        .context("stale or unknown structured run")?;
    if run.phase != StructuredRunPhase::Cancelled {
        bail!("structured run is not cancelled");
    }
    if !clear_unconfirmed_cancellation(run) {
        return Ok(StructuredApplyOutcome::Duplicate);
    }
    if let Some(retained) = provider
        .completed_runs
        .iter_mut()
        .find(|retained| &retained.id == run_id)
    {
        clear_unconfirmed_cancellation(retained);
    }
    Ok(StructuredApplyOutcome::Applied)
}

fn clear_unconfirmed_cancellation(run: &mut StructuredRun) -> bool {
    if run.diagnostic.as_deref() != Some(UNCONFIRMED_CANCELLATION) {
        return false;
    }
    run.diagnostic = None;
    for event in &mut run.events {
        if event.state == StructuredNodeState::Cancelled
            && event.message.as_deref() == Some(UNCONFIRMED_CANCELLATION)
        {
            event.message = None;
            run.message_bytes = run
                .message_bytes
                .saturating_sub(UNCONFIRMED_CANCELLATION.len());
        }
    }
    true
}

pub fn snapshot_page_to_proto(
    state: &StructuredExecutionState,
    provider_id: &StructuredProviderId,
    requested_generation: DiscoveryGeneration,
    page_start: usize,
    requested_page_size: usize,
    visible_worktrees: Option<&HashSet<WorktreeId>>,
) -> Result<proto::StructuredExecutionSnapshotPage> {
    snapshot_page_to_proto_filtered(
        state,
        provider_id,
        requested_generation,
        page_start,
        requested_page_size,
        visible_worktrees,
        None,
    )
}

fn snapshot_page_to_proto_filtered(
    state: &StructuredExecutionState,
    provider_id: &StructuredProviderId,
    requested_generation: DiscoveryGeneration,
    page_start: usize,
    requested_page_size: usize,
    visible_worktrees: Option<&HashSet<WorktreeId>>,
    include_path: Option<&dyn Fn(&ProjectPath) -> bool>,
) -> Result<proto::StructuredExecutionSnapshotPage> {
    let provider = state
        .provider(provider_id)
        .context("unknown structured execution provider")?;
    if requested_generation.0 != 0 && requested_generation != provider.discovery_generation {
        bail!("stale structured discovery generation");
    }
    let filtered_nodes;
    let nodes = if let Some(include_path) = include_path {
        filtered_nodes = filter_nodes_for_peer(&provider.nodes, visible_worktrees, include_path)?;
        filtered_nodes.as_slice()
    } else if visible_worktrees.is_some() {
        filtered_nodes = filter_nodes(&provider.nodes, visible_worktrees)?;
        filtered_nodes.as_slice()
    } else {
        provider.nodes.as_slice()
    };
    let partial = provider.partial || nodes.len() != provider.nodes.len();
    let status = if partial && provider.status == StructuredProviderStatus::Current {
        StructuredProviderStatus::Partial
    } else {
        provider.status
    };
    let page_size = requested_page_size.clamp(1, MAX_STRUCTURED_PAGE_SIZE);
    let page_end = page_start.saturating_add(page_size).min(nodes.len());
    let page = nodes
        .get(page_start..page_end)
        .unwrap_or_default()
        .iter()
        .map(node_to_proto)
        .collect();
    Ok(proto::StructuredExecutionSnapshotPage {
        protocol_version: STRUCTURED_EXECUTION_PROTOCOL_VERSION,
        project_generation: state.project_generation(),
        provider_id: provider.provider_id.0.clone(),
        discovery_generation: provider.discovery_generation.0,
        status: provider_status_to_proto(status),
        nodes: page,
        next_page_start: if page_end < nodes.len() {
            page_end.try_into().unwrap_or(u32::MAX)
        } else {
            0
        },
        partial,
        diagnostic: provider
            .diagnostic
            .as_deref()
            .map(|value| bounded_text(value, MAX_DIAGNOSTIC_BYTES)),
        current_run: if page_start == 0 {
            provider
                .current_run
                .as_ref()
                .filter(|run| run_has_visible_scope(run, nodes))
                .map(|run| run_to_proto(run, &nodes, visible_worktrees, include_path))
        } else {
            None
        },
        last_complete_run: if page_start == 0 {
            provider
                .last_complete_run
                .as_ref()
                .filter(|run| run_has_visible_scope(run, nodes))
                .map(|run| run_to_proto(run, &nodes, visible_worktrees, include_path))
        } else {
            None
        },
    })
}

fn redact_remote_diagnostics(page: &mut proto::StructuredExecutionSnapshotPage) {
    if page.diagnostic.is_some() {
        page.diagnostic = Some("Structured provider details remain on the project host.".into());
    }
    for run in [&mut page.current_run, &mut page.last_complete_run]
        .into_iter()
        .flatten()
    {
        if run.diagnostic.is_some() {
            run.diagnostic = Some("Structured run details remain on the project host.".into());
        }
    }
}

fn event_chunk_to_proto(
    state: &StructuredExecutionState,
    provider_id: &StructuredProviderId,
    run_id: &StructuredRunId,
    start_sequence: u64,
    requested_chunk_size: usize,
    visible_worktrees: Option<&HashSet<WorktreeId>>,
    include_path: Option<&dyn Fn(&ProjectPath) -> bool>,
) -> Result<proto::StructuredExecutionEventChunk> {
    let provider = state
        .provider(provider_id)
        .context("unknown structured execution provider")?;
    let run = provider
        .current_run
        .iter()
        .chain(provider.last_complete_run.iter())
        .chain(provider.completed_runs.iter())
        .find(|run| &run.id == run_id)
        .context("unknown structured run")?;
    let visible_nodes = if let Some(include_path) = include_path {
        filter_nodes_for_peer(&provider.nodes, visible_worktrees, include_path)?
    } else {
        filter_nodes(&provider.nodes, visible_worktrees)?
    }
    .into_iter()
    .map(|node| node.id)
    .collect::<HashSet<_>>();
    if !run
        .scope_node_ids
        .iter()
        .any(|node_id| visible_nodes.contains(node_id))
    {
        bail!("unknown structured run");
    }
    let chunk_size = requested_chunk_size.clamp(1, MAX_STRUCTURED_EVENT_CHUNK_SIZE);
    let mut events = Vec::new();
    let mut next_sequence = start_sequence;
    for event in run
        .events
        .iter()
        .filter(|event| event.sequence >= start_sequence)
    {
        next_sequence = event.sequence.saturating_add(1);
        if visible_nodes.contains(&event.node_id)
            && event.location.as_ref().is_none_or(|path| {
                path_is_visible(path, visible_worktrees)
                    && include_path.is_none_or(|include_path| include_path(path))
            })
        {
            events.push(event_to_proto(event));
            if events.len() >= chunk_size {
                break;
            }
        }
    }
    let complete = next_sequence >= run.next_sequence;
    Ok(proto::StructuredExecutionEventChunk {
        protocol_version: STRUCTURED_EXECUTION_PROTOCOL_VERSION,
        project_generation: state.project_generation(),
        provider_id: provider.provider_id.0.clone(),
        run_id: run.id.0.clone(),
        discovery_generation: run.discovery_generation.0,
        events,
        next_sequence,
        complete,
        truncated: run.truncated,
    })
}

fn tag_remote_structured_unsupported(error: anyhow::Error) -> anyhow::Error {
    if error.is::<StructuredExecutionUnsupportedHost>() {
        proto::ErrorCode::Internal
            .with_tag("reason", "rust_tools_unsupported")
            .message(error.to_string())
            .into()
    } else {
        error
    }
}

fn map_remote_structured_error(error: anyhow::Error) -> anyhow::Error {
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
        StructuredExecutionUnsupportedHost.into()
    } else {
        error
    }
}

async fn fetch_remote_provider(
    project_id: u64,
    client: AnyProtoClient,
    provider_id: StructuredProviderId,
    worktree_ids: Vec<u64>,
) -> Result<(u64, StructuredProviderSnapshot)> {
    let mut page_start = 0;
    let mut nodes = Vec::new();
    let mut first_page: Option<proto::StructuredExecutionSnapshotPage> = None;
    loop {
        let page = client
            .request(proto::GetStructuredExecutionSnapshot {
                project_id,
                provider_id: provider_id.0.clone(),
                discovery_generation: first_page
                    .as_ref()
                    .map_or(0, |page| page.discovery_generation),
                page_start,
                page_size: MAX_STRUCTURED_PAGE_SIZE as u32,
                worktree_ids: worktree_ids.clone(),
            })
            .await
            .map_err(map_remote_structured_error)?;
        validate_protocol(page.protocol_version)?;
        if page.provider_id != provider_id.0 {
            bail!("structured provider response identity mismatch");
        }
        if page.nodes.len() > MAX_STRUCTURED_PAGE_SIZE {
            bail!("structured provider exceeded the page limit");
        }
        let retained_nodes = nodes.len().saturating_add(page.nodes.len());
        if retained_nodes > MAX_STRUCTURED_NODES {
            bail!("structured provider exceeded the node limit");
        }
        if page.next_page_start != 0
            && (page.nodes.is_empty()
                || page.next_page_start as usize != retained_nodes
                || retained_nodes == MAX_STRUCTURED_NODES)
        {
            bail!("structured provider returned an invalid continuation");
        }
        if let Some(first) = first_page.as_ref() {
            let first: &proto::StructuredExecutionSnapshotPage = first;
            if first.project_generation != page.project_generation
                || first.discovery_generation != page.discovery_generation
            {
                bail!("structured provider page generation mismatch");
            }
        } else {
            first_page = Some(page.clone());
        }
        nodes.extend(
            page.nodes
                .into_iter()
                .map(node_from_proto)
                .collect::<Result<Vec<_>>>()?,
        );
        if page.next_page_start == 0 {
            break;
        }
        if page.next_page_start <= page_start {
            bail!("structured provider page did not advance");
        }
        page_start = page.next_page_start;
    }
    let first = first_page.context("structured provider returned no page")?;
    let mut snapshot = StructuredProviderSnapshot {
        provider_id,
        discovery_generation: DiscoveryGeneration(first.discovery_generation),
        status: provider_status_from_proto(first.status)?,
        nodes,
        partial: first.partial,
        diagnostic: first
            .diagnostic
            .map(|value| bounded_text(&value, MAX_DIAGNOSTIC_BYTES)),
        current_run: first.current_run.map(run_from_proto).transpose()?,
        last_complete_run: first.last_complete_run.map(run_from_proto).transpose()?,
        completed_runs: VecDeque::new(),
    };
    if let Some(run) = snapshot.current_run.as_mut() {
        if run.discovery_generation != snapshot.discovery_generation {
            bail!("structured current run generation mismatch");
        }
        run.events = fetch_remote_events(
            project_id,
            first.project_generation,
            &client,
            &snapshot.provider_id,
            run,
            &worktree_ids,
        )
        .await?;
        run.next_sequence = run
            .events
            .last()
            .map_or(0, |event| event.sequence.saturating_add(1));
        rebuild_run_summary(run);
    }
    if let Some(run) = snapshot.last_complete_run.as_mut() {
        if run.discovery_generation > snapshot.discovery_generation {
            bail!("structured retained run generation mismatch");
        }
        run.events = fetch_remote_events(
            project_id,
            first.project_generation,
            &client,
            &snapshot.provider_id,
            run,
            &worktree_ids,
        )
        .await?;
        run.next_sequence = run
            .events
            .last()
            .map_or(0, |event| event.sequence.saturating_add(1));
        rebuild_run_summary(run);
    }
    Ok((first.project_generation, snapshot))
}

async fn fetch_remote_events(
    project_id: u64,
    project_generation: u64,
    client: &AnyProtoClient,
    provider_id: &StructuredProviderId,
    run: &StructuredRun,
    worktree_ids: &[u64],
) -> Result<Vec<StructuredExecutionEvent>> {
    let mut start_sequence = 0;
    let mut events = Vec::new();
    loop {
        let chunk = client
            .request(proto::GetStructuredExecutionEvents {
                project_id,
                provider_id: provider_id.0.clone(),
                run_id: run.id.0.clone(),
                start_sequence,
                chunk_size: MAX_STRUCTURED_EVENT_CHUNK_SIZE as u32,
                worktree_ids: worktree_ids.to_vec(),
            })
            .await
            .map_err(map_remote_structured_error)?;
        validate_protocol(chunk.protocol_version)?;
        if chunk.provider_id != provider_id.0 || chunk.run_id != run.id.0 {
            bail!("structured event response identity mismatch");
        }
        if chunk.project_generation != project_generation
            || chunk.discovery_generation != run.discovery_generation.0
        {
            bail!("structured event response generation mismatch");
        }
        if chunk.events.len() > MAX_STRUCTURED_EVENT_CHUNK_SIZE {
            bail!("structured event response exceeded the chunk limit");
        }
        let retained_events = events.len().saturating_add(chunk.events.len());
        if retained_events > MAX_STRUCTURED_EVENTS {
            bail!("structured event response exceeded the event limit");
        }
        if !chunk.complete && (chunk.events.is_empty() || retained_events == MAX_STRUCTURED_EVENTS)
        {
            bail!("structured event response returned an invalid continuation");
        }
        events.extend(
            chunk
                .events
                .into_iter()
                .map(event_from_proto)
                .collect::<Result<Vec<_>>>()?,
        );
        if chunk.complete {
            break;
        }
        if chunk.next_sequence <= start_sequence {
            bail!("structured event chunk did not advance");
        }
        start_sequence = chunk.next_sequence;
    }
    Ok(events)
}

fn validate_protocol(version: u32) -> Result<()> {
    if version != STRUCTURED_EXECUTION_PROTOCOL_VERSION {
        bail!("unsupported structured execution protocol version {version}");
    }
    Ok(())
}

fn node_to_proto(node: &StructuredNode) -> proto::StructuredExecutionNode {
    proto::StructuredExecutionNode {
        experimental_feature: node.experimental_feature.clone(),
        experimental_feature_enabled: node.experimental_feature_enabled,
        debug_capability: node.debug_capability.as_ref().map(|capability| {
            let (supported, reason) = match capability {
                StructuredDebugCapability::Supported => (true, None),
                StructuredDebugCapability::Unsupported(reason) => {
                    (false, Some(bounded_text(reason, MAX_DIAGNOSTIC_BYTES)))
                }
            };
            proto::StructuredExecutionDebugCapability { supported, reason }
        }),
        node_id: node.id.0.clone(),
        parent_id: node.parent_id.as_ref().map(|id| id.0.clone()),
        label: bounded_text(&node.label, MAX_LABEL_BYTES),
        kind: node_kind_to_proto(node.kind),
        path: node.path.as_ref().map(ProjectPath::to_proto),
    }
}

fn node_from_proto(node: proto::StructuredExecutionNode) -> Result<StructuredNode> {
    validate_identifier("node ID", &node.node_id)?;
    if let Some(feature) = node.experimental_feature.as_deref() {
        task::validate_experimental_feature_requirement(feature)?;
    }
    let path = node
        .path
        .map(|path| ProjectPath::from_proto(path).context("invalid structured node path"))
        .transpose()?;
    Ok(StructuredNode {
        experimental_feature: node.experimental_feature,
        experimental_feature_enabled: node.experimental_feature_enabled,
        debug_capability: node
            .debug_capability
            .map(|capability| {
                if capability.supported {
                    if capability.reason.is_some() {
                        bail!("supported Debug capability has a disabled reason");
                    }
                    Ok(StructuredDebugCapability::Supported)
                } else {
                    let reason = capability
                        .reason
                        .context("unsupported Debug capability has no reason")?;
                    if reason.trim().is_empty() {
                        bail!("unsupported Debug capability has an empty reason");
                    }
                    Ok(StructuredDebugCapability::Unsupported(bounded_text(
                        &reason,
                        MAX_DIAGNOSTIC_BYTES,
                    )))
                }
            })
            .transpose()?,
        id: StructuredNodeId(node.node_id),
        parent_id: node.parent_id.map(StructuredNodeId),
        label: bounded_text(&node.label, MAX_LABEL_BYTES),
        kind: node_kind_from_proto(node.kind)?,
        path,
    })
}

fn run_to_proto(
    run: &StructuredRun,
    visible_nodes: &[StructuredNode],
    visible_worktrees: Option<&HashSet<WorktreeId>>,
    include_path: Option<&dyn Fn(&ProjectPath) -> bool>,
) -> proto::StructuredExecutionRun {
    let visible_ids = visible_nodes
        .iter()
        .map(|node| &node.id)
        .collect::<HashSet<_>>();
    let visible_events = run
        .events
        .iter()
        .filter(|event| {
            visible_ids.contains(&event.node_id)
                && event.location.as_ref().is_none_or(|path| {
                    path_is_visible(path, visible_worktrees)
                        && include_path.is_none_or(|include_path| include_path(path))
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    proto::StructuredExecutionRun {
        run_id: run.id.0.clone(),
        discovery_generation: run.discovery_generation.0,
        phase: run_phase_to_proto(run.phase),
        scope_node_ids: run
            .scope_node_ids
            .iter()
            .filter(|id| visible_ids.contains(id))
            .map(|id| id.0.clone())
            .collect(),
        summary: Some(summarize_events(&visible_events).into()),
        truncated: run.truncated || visible_events.len() != run.events.len(),
        diagnostic: run
            .diagnostic
            .as_deref()
            .map(|value| bounded_text(value, MAX_DIAGNOSTIC_BYTES)),
    }
}

fn run_has_visible_scope(run: &StructuredRun, visible_nodes: &[StructuredNode]) -> bool {
    run.scope_node_ids
        .iter()
        .any(|node_id| visible_nodes.iter().any(|node| &node.id == node_id))
}

fn run_from_proto(run: proto::StructuredExecutionRun) -> Result<StructuredRun> {
    validate_identifier("run ID", &run.run_id)?;
    for node_id in &run.scope_node_ids {
        validate_identifier("scope node ID", node_id)?;
    }
    let mut result = StructuredRun::new(
        StructuredRunId(run.run_id),
        DiscoveryGeneration(run.discovery_generation),
        run.scope_node_ids
            .into_iter()
            .map(StructuredNodeId)
            .collect(),
    );
    result.phase = run_phase_from_proto(run.phase)?;
    result.summary = run.summary.map(Into::into).unwrap_or_default();
    result.truncated = run.truncated;
    result.diagnostic = run
        .diagnostic
        .map(|value| bounded_text(&value, MAX_DIAGNOSTIC_BYTES));
    Ok(result)
}

fn event_to_proto(event: &StructuredExecutionEvent) -> proto::StructuredExecutionEvent {
    proto::StructuredExecutionEvent {
        sequence: event.sequence,
        node_id: event.node_id.0.clone(),
        state: node_state_to_proto(event.state),
        duration_millis: event.duration_millis,
        message: event
            .message
            .as_deref()
            .map(|value| bounded_text(value, MAX_DIAGNOSTIC_BYTES)),
        location: event.location.as_ref().map(ProjectPath::to_proto),
    }
}

fn event_from_proto(event: proto::StructuredExecutionEvent) -> Result<StructuredExecutionEvent> {
    validate_identifier("node ID", &event.node_id)?;
    let location = event
        .location
        .map(|path| ProjectPath::from_proto(path).context("invalid structured event path"))
        .transpose()?;
    Ok(StructuredExecutionEvent {
        sequence: event.sequence,
        node_id: StructuredNodeId(event.node_id),
        state: node_state_from_proto(event.state)?,
        duration_millis: event.duration_millis,
        message: event
            .message
            .map(|value| bounded_text(&value, MAX_DIAGNOSTIC_BYTES)),
        location,
    })
}

impl From<StructuredExecutionSummary> for proto::StructuredExecutionSummary {
    fn from(summary: StructuredExecutionSummary) -> Self {
        Self {
            total: summary.total,
            queued: summary.queued,
            running: summary.running,
            passed: summary.passed,
            failed: summary.failed,
            skipped: summary.skipped,
            cancelled: summary.cancelled,
        }
    }
}

impl From<proto::StructuredExecutionSummary> for StructuredExecutionSummary {
    fn from(summary: proto::StructuredExecutionSummary) -> Self {
        Self {
            total: summary.total,
            queued: summary.queued,
            running: summary.running,
            passed: summary.passed,
            failed: summary.failed,
            skipped: summary.skipped,
            cancelled: summary.cancelled,
        }
    }
}

fn node_kind_to_proto(kind: StructuredNodeKind) -> i32 {
    match kind {
        StructuredNodeKind::Provider => 1,
        StructuredNodeKind::Suite => 2,
        StructuredNodeKind::Group => 3,
        StructuredNodeKind::Case => 4,
    }
}

fn node_kind_from_proto(kind: i32) -> Result<StructuredNodeKind> {
    match kind {
        1 => Ok(StructuredNodeKind::Provider),
        2 => Ok(StructuredNodeKind::Suite),
        3 => Ok(StructuredNodeKind::Group),
        4 => Ok(StructuredNodeKind::Case),
        _ => bail!("invalid structured node kind {kind}"),
    }
}

pub(crate) fn provider_status_to_proto(status: StructuredProviderStatus) -> i32 {
    match status {
        StructuredProviderStatus::Loading => 1,
        StructuredProviderStatus::Current => 2,
        StructuredProviderStatus::Empty => 3,
        StructuredProviderStatus::Partial => 4,
        StructuredProviderStatus::Stale => 5,
        StructuredProviderStatus::Error => 6,
        StructuredProviderStatus::Restricted => 7,
        StructuredProviderStatus::Disconnected => 8,
        StructuredProviderStatus::Mismatch => 9,
    }
}

fn provider_status_from_proto(status: i32) -> Result<StructuredProviderStatus> {
    match status {
        1 => Ok(StructuredProviderStatus::Loading),
        2 => Ok(StructuredProviderStatus::Current),
        3 => Ok(StructuredProviderStatus::Empty),
        4 => Ok(StructuredProviderStatus::Partial),
        5 => Ok(StructuredProviderStatus::Stale),
        6 => Ok(StructuredProviderStatus::Error),
        7 => Ok(StructuredProviderStatus::Restricted),
        8 => Ok(StructuredProviderStatus::Disconnected),
        9 => Ok(StructuredProviderStatus::Mismatch),
        _ => bail!("invalid structured provider status {status}"),
    }
}

fn run_phase_to_proto(phase: StructuredRunPhase) -> i32 {
    match phase {
        StructuredRunPhase::Queued => 1,
        StructuredRunPhase::Running => 2,
        StructuredRunPhase::Completed => 3,
        StructuredRunPhase::Cancelled => 4,
        StructuredRunPhase::SpawnError => 5,
    }
}

fn run_phase_from_proto(phase: i32) -> Result<StructuredRunPhase> {
    match phase {
        1 => Ok(StructuredRunPhase::Queued),
        2 => Ok(StructuredRunPhase::Running),
        3 => Ok(StructuredRunPhase::Completed),
        4 => Ok(StructuredRunPhase::Cancelled),
        5 => Ok(StructuredRunPhase::SpawnError),
        _ => bail!("invalid structured run phase {phase}"),
    }
}

fn node_state_to_proto(state: StructuredNodeState) -> i32 {
    match state {
        StructuredNodeState::Queued => 1,
        StructuredNodeState::Running => 2,
        StructuredNodeState::Passed => 3,
        StructuredNodeState::Failed => 4,
        StructuredNodeState::Skipped => 5,
        StructuredNodeState::Cancelled => 6,
    }
}

fn node_state_from_proto(state: i32) -> Result<StructuredNodeState> {
    match state {
        1 => Ok(StructuredNodeState::Queued),
        2 => Ok(StructuredNodeState::Running),
        3 => Ok(StructuredNodeState::Passed),
        4 => Ok(StructuredNodeState::Failed),
        5 => Ok(StructuredNodeState::Skipped),
        6 => Ok(StructuredNodeState::Cancelled),
        _ => bail!("invalid structured node state {state}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn structured_same_generation_renews_only_known_experimental_availability() {
        let mut snapshot = fake_provider(2);
        let marked = snapshot
            .nodes
            .iter_mut()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("fixture case");
        marked.experimental_feature = Some("rust-exact-doctest-actions-experimental".into());
        marked.experimental_feature_enabled = Some(true);
        let provider_id = snapshot.provider_id.clone();
        let mut state = StructuredExecutionState::new(1);
        state
            .apply_discovery(1, snapshot.clone(), None)
            .expect("initial provider");
        state
            .begin_run(
                1,
                &provider_id,
                StructuredRun::new(
                    StructuredRunId("run".into()),
                    DiscoveryGeneration(2),
                    vec![StructuredNodeId("case-a".into())],
                ),
            )
            .expect("actual state run");
        let baseline = state.provider(&provider_id).expect("provider").clone();
        for enabled in [Some(false), None, Some(true)] {
            let mut changed = state.provider(&provider_id).expect("provider").clone();
            changed
                .nodes
                .iter_mut()
                .find(|node| node.experimental_feature.is_some())
                .expect("marked case")
                .experimental_feature_enabled = enabled;
            assert_eq!(
                state
                    .apply_discovery(1, changed, None)
                    .expect("availability-only renewal"),
                StructuredApplyOutcome::Applied
            );
            let renewed = state.provider(&provider_id).expect("provider");
            assert_eq!(renewed.current_run, baseline.current_run);
            assert_eq!(renewed.last_complete_run, baseline.last_complete_run);
            assert_eq!(renewed.completed_runs, baseline.completed_runs);
        }
        let renewed = state.provider(&provider_id).expect("provider").clone();
        assert_eq!(renewed, baseline);
        let mut identity = renewed.clone();
        identity.nodes.first_mut().expect("root").label.push('!');
        assert!(state.apply_discovery(1, identity, None).is_err());
        let mut order = renewed.clone();
        order.nodes.reverse();
        assert!(state.apply_discovery(1, order, None).is_err());
        let mut ordinary = renewed.clone();
        ordinary
            .nodes
            .first_mut()
            .expect("root")
            .experimental_feature_enabled = Some(true);
        assert!(state.apply_discovery(1, ordinary, None).is_err());
        let mut provider = renewed.clone();
        provider.diagnostic = Some("changed".into());
        assert!(state.apply_discovery(1, provider, None).is_err());
        let mut run = renewed.clone();
        run.current_run = None;
        run.nodes
            .iter_mut()
            .find(|node| node.experimental_feature.is_some())
            .expect("marked case")
            .experimental_feature_enabled = Some(false);
        assert!(state.apply_discovery(1, run, None).is_err());
        let mut marker = renewed.clone();
        marker
            .nodes
            .iter_mut()
            .find(|node| node.experimental_feature.is_some())
            .expect("marked case")
            .experimental_feature = Some("rust-coverage-experimental".into());
        assert!(state.apply_discovery(1, marker, None).is_err());
        assert_eq!(state.provider(&provider_id), Some(&renewed));
    }

    #[gpui::test]
    async fn structured_export_renews_host_hint_and_relay_preserves_upstream(
        cx: &mut gpui::TestAppContext,
    ) {
        const FEATURE: &str = "rust-exact-doctest-actions-experimental";
        cx.update(|cx| crate::task_store::set_experimental_feature_for_test(FEATURE, false, cx));
        let project = crate::Project::test(crate::FakeFs::new(cx.executor()), [], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let mut snapshot = fake_provider(2);
        for node in &mut snapshot.nodes {
            node.path = None;
        }
        let marked = snapshot
            .nodes
            .iter_mut()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("fixture case");
        marked.experimental_feature = Some(FEATURE.into());
        marked.experimental_feature_enabled = Some(true);
        let provider_id = snapshot.provider_id.clone();
        let local = cx.new(|_| StructuredExecutionStore::local(worktree_store.clone(), 1));
        local.update(cx, |store, _| {
            store
                .state_mut()
                .apply_discovery(1, snapshot, None)
                .expect("local fixture DATA");
        });
        let request = proto::GetStructuredExecutionSnapshot {
            provider_id: provider_id.0.clone(),
            discovery_generation: 2,
            page_size: MAX_STRUCTURED_PAGE_SIZE as u32,
            ..Default::default()
        };
        let host_off = local.read_with(cx, |store, cx| {
            store
                .snapshot_page_for_peer(&request, &HashSet::new(), cx)
                .expect("host Off page")
        });
        assert!(
            host_off
                .nodes
                .iter()
                .any(|node| node.experimental_feature.as_deref() == Some(FEATURE)
                    && node.experimental_feature_enabled == Some(false))
        );
        assert!(
            host_off
                .nodes
                .iter()
                .filter(|node| node.experimental_feature.is_none())
                .all(|node| node.experimental_feature_enabled.is_none())
        );
        cx.update(|cx| crate::task_store::set_experimental_feature_for_test(FEATURE, true, cx));
        let host_on = local.read_with(cx, |store, cx| {
            store
                .snapshot_page_for_peer(&request, &HashSet::new(), cx)
                .expect("host On page")
        });
        assert!(
            host_on
                .nodes
                .iter()
                .any(|node| node.experimental_feature.as_deref() == Some(FEATURE)
                    && node.experimental_feature_enabled == Some(true))
        );
        assert_eq!(host_on.discovery_generation, host_off.discovery_generation);
        let client = Arc::new(SnapshotClient::default());
        client
            .replies
            .lock()
            .push_back(host_off.into_envelope(0, None, None));
        let relay = cx.new(|_| {
            StructuredExecutionStore::remote(1, AnyProtoClient::new(client.clone()), worktree_store)
        });
        relay
            .update(cx, |store, cx| store.refresh_provider(provider_id, cx))
            .await
            .expect("actual typed upstream page receiver");
        let forwarded = relay.read_with(cx, |store, cx| {
            store
                .snapshot_page_for_peer(&request, &HashSet::new(), cx)
                .expect("relay page")
        });
        assert!(
            forwarded
                .nodes
                .iter()
                .any(|node| node.experimental_feature.as_deref() == Some(FEATURE)
                    && node.experimental_feature_enabled == Some(false)),
            "relay-local On must not turn host Off DATA into an enabled hint"
        );
        assert_eq!(client.requests.lock().len(), 1);
    }

    #[test]
    fn structured_nodes_preserve_host_experimental_enabled_data() {
        for enabled in [None, Some(false), Some(true)] {
            let node = StructuredNode {
                experimental_feature: Some("rust-exact-doctest-actions-experimental".into()),
                experimental_feature_enabled: enabled,
                debug_capability: None,
                id: StructuredNodeId("case".into()),
                parent_id: None,
                label: "doctest".into(),
                kind: StructuredNodeKind::Case,
                path: None,
            };
            let wire = node_to_proto(&node);
            assert_eq!(wire.experimental_feature_enabled, enabled);
            assert_eq!(node_from_proto(wire).expect("valid node DATA"), node);
        }
        let ordinary = fake_provider(1)
            .nodes
            .into_iter()
            .next()
            .expect("ordinary fixture node");
        assert_eq!(ordinary.experimental_feature, None);
        assert_eq!(ordinary.experimental_feature_enabled, None);
        assert_eq!(
            node_from_proto(node_to_proto(&ordinary)).expect("ordinary node DATA"),
            ordinary
        );
    }

    #[test]
    fn doctor_test_node_scope_inherits_module_and_rejects_missing_or_cyclic_ancestry() {
        let path = ProjectPath {
            worktree_id: WorktreeId::from_proto(7),
            path: util::rel_path::RelPath::empty_arc(),
        };
        let module = StructuredNodeId("module".into());
        let suite = StructuredNodeId("suite".into());
        let case = StructuredNodeId("case".into());
        let mut snapshot = StructuredProviderSnapshot::discovery(
            StructuredProviderId("jvm-junit".into()),
            DiscoveryGeneration(1),
            StructuredProviderStatus::Stale,
            vec![
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: module.clone(),
                    parent_id: None,
                    label: "module".into(),
                    kind: StructuredNodeKind::Group,
                    path: Some(path.clone()),
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: suite.clone(),
                    parent_id: Some(module),
                    label: "suite".into(),
                    kind: StructuredNodeKind::Suite,
                    path: None,
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: case.clone(),
                    parent_id: Some(suite.clone()),
                    label: "method".into(),
                    kind: StructuredNodeKind::Case,
                    path: None,
                },
            ],
        );
        assert_eq!(snapshot.project_path_for_node(&case), Some(&path));
        assert!(
            snapshot
                .project_path_for_node(&StructuredNodeId("absent".into()))
                .is_none()
        );
        let other_path = ProjectPath {
            worktree_id: WorktreeId::from_proto(9),
            path: util::rel_path::RelPath::empty_arc(),
        };
        let other = StructuredNodeId("other-root-case".into());
        snapshot.nodes.push(StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: other.clone(),
            parent_id: None,
            label: "different project root".into(),
            kind: StructuredNodeKind::Case,
            path: Some(other_path.clone()),
        });
        assert_eq!(snapshot.project_path_for_node(&other), Some(&other_path));
        assert_ne!(
            snapshot.project_path_for_node(&other),
            snapshot.project_path_for_node(&case)
        );
        snapshot
            .nodes
            .iter_mut()
            .find(|node| node.id == suite)
            .expect("Fixture suite")
            .parent_id = Some(case.clone());
        assert!(snapshot.project_path_for_node(&case).is_none());
        snapshot
            .nodes
            .iter_mut()
            .find(|node| node.id == suite)
            .expect("Fixture suite")
            .parent_id = Some(StructuredNodeId("absent-parent".into()));
        assert!(snapshot.project_path_for_node(&case).is_none());
    }

    use std::sync::Arc;

    use futures::{FutureExt as _, future::BoxFuture};
    use gpui::AppContext as _;
    use parking_lot::Mutex;
    use proto::EnvelopedMessage as _;
    use util::rel_path::RelPath;

    use super::*;

    #[derive(Default)]
    struct SnapshotClient {
        handlers: Mutex<rpc::ProtoMessageHandlerSet>,
        replies: Mutex<VecDeque<proto::Envelope>>,
        requests: Mutex<Vec<proto::Envelope>>,
        deferred: Mutex<Option<futures::channel::oneshot::Receiver<proto::Envelope>>>,
    }

    impl rpc::ProtoClient for SnapshotClient {
        fn request(
            &self,
            envelope: proto::Envelope,
            _: &'static str,
        ) -> BoxFuture<'static, Result<proto::Envelope>> {
            self.requests.lock().push(envelope);
            if let Some(deferred) = self.deferred.lock().take() {
                return async move { deferred.await.context("deferred response was dropped") }
                    .boxed();
            }
            futures::future::ready(
                self.replies
                    .lock()
                    .pop_front()
                    .context("unexpected request"),
            )
            .boxed()
        }

        fn send(&self, _: proto::Envelope, _: &'static str) -> Result<()> {
            bail!("unexpected send")
        }

        fn send_response(&self, _: proto::Envelope, _: &'static str) -> Result<()> {
            bail!("unexpected response")
        }

        fn message_handler_set(&self) -> &Mutex<rpc::ProtoMessageHandlerSet> {
            &self.handlers
        }

        fn is_via_collab(&self) -> bool {
            false
        }

        fn has_wsl_interop(&self) -> bool {
            false
        }
    }

    fn remote_page(
        start: usize,
        count: usize,
        next: u32,
    ) -> proto::StructuredExecutionSnapshotPage {
        proto::StructuredExecutionSnapshotPage {
            protocol_version: STRUCTURED_EXECUTION_PROTOCOL_VERSION,
            project_generation: 1,
            provider_id: "web-tests".into(),
            discovery_generation: 1,
            status: provider_status_to_proto(StructuredProviderStatus::Current),
            nodes: (start..start + count)
                .map(|index| {
                    node_to_proto(&StructuredNode {
                        experimental_feature: None,
                        experimental_feature_enabled: None,
                        debug_capability: None,
                        id: StructuredNodeId(format!("case-{index}")),
                        parent_id: None,
                        label: format!("Case {index}"),
                        kind: StructuredNodeKind::Case,
                        path: None,
                    })
                })
                .collect(),
            next_page_start: next,
            ..Default::default()
        }
    }

    async fn receive_pages(
        pages: Vec<proto::StructuredExecutionSnapshotPage>,
    ) -> Result<StructuredProviderSnapshot> {
        receive_responses(
            pages
                .into_iter()
                .map(|page| page.into_envelope(0, None, None))
                .collect(),
        )
        .await
    }

    async fn receive_responses(
        replies: Vec<proto::Envelope>,
    ) -> Result<StructuredProviderSnapshot> {
        let client = Arc::new(SnapshotClient::default());
        client.replies.lock().extend(replies);
        fetch_remote_provider(
            1,
            AnyProtoClient::new(client),
            StructuredProviderId("web-tests".into()),
            Vec::new(),
        )
        .await
        .map(|(_, snapshot)| snapshot)
    }

    #[gpui::test]
    async fn structured_execution_debug_capability_survives_scoped_paging() {
        let provider_id = StructuredProviderId("web-tests".into());
        let generation = DiscoveryGeneration(3);
        let mut nodes = Vec::new();
        for (index, capability) in [
            Some(StructuredDebugCapability::Supported),
            Some(StructuredDebugCapability::Unsupported(
                "Unsupported harness".into(),
            )),
            None,
            Some(StructuredDebugCapability::Unsupported(
                "Hidden capability".into(),
            )),
        ]
        .into_iter()
        .enumerate()
        {
            nodes.push(StructuredNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: capability,
                id: StructuredNodeId(format!("case-{index}")),
                parent_id: None,
                label: format!("Case {index}"),
                kind: StructuredNodeKind::Case,
                path: Some(ProjectPath {
                    worktree_id: WorktreeId::from_usize(if index == 3 { 2 } else { 1 }),
                    path: Arc::from(RelPath::from_unix_str("tests.rs").expect("fixture path")),
                }),
            });
        }
        let mut state = StructuredExecutionState::new(1);
        state
            .apply_discovery(
                1,
                StructuredProviderSnapshot::discovery(
                    provider_id.clone(),
                    generation,
                    StructuredProviderStatus::Current,
                    nodes.clone(),
                ),
                None,
            )
            .expect("host discovery");
        let visible = HashSet::from([WorktreeId::from_usize(1)]);
        let first = snapshot_page_to_proto(&state, &provider_id, generation, 0, 2, Some(&visible))
            .expect("first page");
        assert_eq!(first.next_page_start, 2);
        let last = snapshot_page_to_proto(&state, &provider_id, generation, 2, 2, Some(&visible))
            .expect("last page");
        assert_eq!(last.next_page_start, 0);
        use proto::Message as _;
        let pages = vec![first, last]
            .into_iter()
            .map(|page| {
                proto::StructuredExecutionSnapshotPage::decode(page.encode_to_vec().as_slice())
                    .expect("encoded page")
            })
            .collect();
        let received = receive_pages(pages).await.expect("remote pages");
        assert_eq!(
            received.nodes,
            nodes.into_iter().take(3).collect::<Vec<_>>()
        );
        assert!(received.partial);

        let node = received.nodes.first().expect("visible node");
        let mut wire = node_to_proto(node);
        wire.debug_capability = None;
        assert!(
            node_from_proto(wire.clone())
                .expect("legacy node")
                .debug_capability
                .is_none()
        );
        for capability in [
            proto::StructuredExecutionDebugCapability {
                supported: true,
                reason: Some("contradictory".into()),
            },
            proto::StructuredExecutionDebugCapability {
                supported: false,
                reason: None,
            },
            proto::StructuredExecutionDebugCapability {
                supported: false,
                reason: Some("  ".into()),
            },
        ] {
            wire.debug_capability = Some(capability);
            assert!(node_from_proto(wire.clone()).is_err());
        }
        wire.debug_capability = Some(proto::StructuredExecutionDebugCapability {
            supported: false,
            reason: Some("é".repeat(MAX_DIAGNOSTIC_BYTES)),
        });
        let decoded = node_from_proto(wire).expect("bounded reason");
        let Some(StructuredDebugCapability::Unsupported(reason)) = decoded.debug_capability else {
            panic!("unsupported capability")
        };
        assert!(reason.len() <= MAX_DIAGNOSTIC_BYTES);
        assert!(reason.ends_with('é'));
    }

    #[gpui::test(iterations = 10)]
    async fn structured_execution_remote_queued_replies_cannot_cross_epochs(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let project = crate::Project::test(crate::FakeFs::new(cx.executor()), [], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let client = Arc::new(SnapshotClient::default());
        let (send, receive) = futures::channel::oneshot::channel();
        *client.deferred.lock() = Some(receive);
        let store = cx.new(|_| {
            StructuredExecutionStore::remote(1, AnyProtoClient::new(client.clone()), worktree_store)
        });
        let provider_id = StructuredProviderId("web-tests".into());
        store.update(cx, |store, _| {
            store.state_mut().replace_remote_snapshot(
                1,
                StructuredProviderSnapshot::discovery(
                    provider_id.clone(),
                    DiscoveryGeneration(1),
                    StructuredProviderStatus::Empty,
                    Vec::new(),
                ),
            );
        });

        let old = store.update(cx, |store, cx| {
            store.refresh_provider(provider_id.clone(), cx)
        });
        cx.run_until_parked();
        assert_eq!(
            client.requests.lock().len(),
            1,
            "old request reached the client"
        );
        let mut stale_page = remote_page(0, 0, 0);
        stale_page.discovery_generation = 2;
        send.send(stale_page.into_envelope(0, None, None))
            .expect("old reply was queued before reconnect");
        store.update(cx, |store, cx| {
            store.disconnected_from_host(cx);
            store.reconnected_to_host(cx);
        });
        old.await
            .expect_err("old connection reply must be rejected");
        store.read_with(cx, |store, _| {
            let cached = store
                .state()
                .provider(&provider_id)
                .expect("cached provider");
            assert_eq!(cached.discovery_generation, DiscoveryGeneration(1));
            assert_eq!(cached.status, StructuredProviderStatus::Stale);
        });

        let mut fresh_page = remote_page(0, 0, 0);
        fresh_page.discovery_generation = 3;
        client
            .replies
            .lock()
            .push_back(fresh_page.into_envelope(0, None, None));
        let refreshed = store
            .update(cx, |store, cx| {
                store.refresh_provider(provider_id.clone(), cx)
            })
            .await
            .expect("fresh connection accepts a new provider generation");
        assert_eq!(refreshed.discovery_generation, DiscoveryGeneration(3));
        assert_eq!(client.requests.lock().len(), 2);

        let (send, receive) = futures::channel::oneshot::channel();
        *client.deferred.lock() = Some(receive);
        let superseded = store.update(cx, |store, cx| {
            store.refresh_provider(provider_id.clone(), cx)
        });
        cx.run_until_parked();
        assert_eq!(client.requests.lock().len(), 3);
        let mut superseded_page = remote_page(0, 0, 0);
        superseded_page.discovery_generation = 4;
        send.send(superseded_page.into_envelope(0, None, None))
            .expect("superseded reply was queued");
        let mut latest_page = remote_page(0, 0, 0);
        latest_page.discovery_generation = 5;
        client
            .replies
            .lock()
            .push_back(latest_page.into_envelope(0, None, None));
        let latest = store.update(cx, |store, cx| {
            store.refresh_provider(provider_id.clone(), cx)
        });
        superseded
            .await
            .expect_err("superseded queued reply must be rejected");
        let latest = latest.await.expect("latest refresh succeeds");
        assert_eq!(latest.discovery_generation, DiscoveryGeneration(5));
        store.read_with(cx, |store, _| {
            assert_eq!(
                store
                    .state()
                    .provider(&provider_id)
                    .map(|snapshot| snapshot.discovery_generation),
                Some(DiscoveryGeneration(5))
            );
        });
        assert_eq!(client.requests.lock().len(), 4);
    }

    #[gpui::test]
    async fn structured_execution_remote_pages_enforce_limits_before_continuing() {
        let oversized = receive_pages(vec![remote_page(0, MAX_STRUCTURED_PAGE_SIZE + 1, 0)])
            .await
            .expect_err("oversized wire page must be rejected");
        assert!(oversized.to_string().contains("page limit"));
        for page in [remote_page(0, 0, 1), remote_page(0, 1, 2)] {
            let error = receive_pages(vec![page])
                .await
                .expect_err("invalid continuation");
            assert!(error.to_string().contains("invalid continuation"));
        }
        let pages = (0..MAX_STRUCTURED_NODES)
            .step_by(MAX_STRUCTURED_PAGE_SIZE)
            .map(|start| {
                let count = MAX_STRUCTURED_PAGE_SIZE.min(MAX_STRUCTURED_NODES - start);
                let next = if start + count == MAX_STRUCTURED_NODES {
                    0
                } else {
                    (start + count) as u32
                };
                remote_page(start, count, next)
            })
            .collect::<Vec<_>>();
        let snapshot = receive_pages(pages.clone())
            .await
            .expect("maximum legal remote snapshot");
        assert_eq!(snapshot.nodes.len(), MAX_STRUCTURED_NODES);
        let mut continued = pages.clone();
        continued.last_mut().expect("last page").next_page_start = MAX_STRUCTURED_NODES as u32;
        let error = receive_pages(continued)
            .await
            .expect_err("must stop at the total bound");
        assert!(error.to_string().contains("invalid continuation"));
        let mut oversized_total = pages;
        let last = oversized_total.last_mut().expect("last page");
        last.nodes.push(node_to_proto(&StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: StructuredNodeId("overflow".into()),
            parent_id: None,
            label: "Overflow".into(),
            kind: StructuredNodeKind::Case,
            path: None,
        }));
        let error = receive_pages(oversized_total)
            .await
            .expect_err("aggregate limit must be enforced");
        assert!(error.to_string().contains("node limit"));
    }

    fn remote_event_chunk(
        start: usize,
        count: usize,
        complete: bool,
    ) -> proto::StructuredExecutionEventChunk {
        proto::StructuredExecutionEventChunk {
            protocol_version: STRUCTURED_EXECUTION_PROTOCOL_VERSION,
            project_generation: 1,
            provider_id: "web-tests".into(),
            run_id: "run".into(),
            discovery_generation: 1,
            events: (start..start + count)
                .map(|sequence| {
                    event_to_proto(&StructuredExecutionEvent {
                        sequence: sequence as u64,
                        node_id: StructuredNodeId("case".into()),
                        state: StructuredNodeState::Passed,
                        duration_millis: None,
                        message: None,
                        location: None,
                    })
                })
                .collect(),
            next_sequence: (start + count) as u64,
            complete,
            truncated: false,
        }
    }

    async fn receive_event_chunks(
        chunks: Vec<proto::StructuredExecutionEventChunk>,
    ) -> Result<Vec<StructuredExecutionEvent>> {
        let client = Arc::new(SnapshotClient::default());
        client.replies.lock().extend(
            chunks
                .into_iter()
                .map(|chunk| chunk.into_envelope(0, None, None)),
        );
        fetch_remote_events(
            1,
            1,
            &AnyProtoClient::new(client),
            &StructuredProviderId("web-tests".into()),
            &StructuredRun::new(
                StructuredRunId("run".into()),
                DiscoveryGeneration(1),
                Vec::new(),
            ),
            &[],
        )
        .await
    }

    #[gpui::test]
    async fn structured_execution_remote_pages_pin_discovery_generation() {
        let client = Arc::new(SnapshotClient::default());
        let mut first = remote_page(0, 1, 1);
        first.discovery_generation = 7;
        let mut second = remote_page(1, 1, 0);
        second.discovery_generation = 7;
        client.replies.lock().extend([
            first.clone().into_envelope(0, None, None),
            second.clone().into_envelope(0, None, None),
        ]);
        fetch_remote_provider(
            1,
            AnyProtoClient::new(client.clone()),
            StructuredProviderId("web-tests".into()),
            Vec::new(),
        )
        .await
        .expect("matching pages");
        let requests = client
            .requests
            .lock()
            .iter()
            .filter_map(|envelope| match envelope.payload.as_ref() {
                Some(proto::envelope::Payload::GetStructuredExecutionSnapshot(request)) => {
                    Some((request.page_start, request.discovery_generation))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(requests, vec![(0, 0), (1, 7)]);
        second.discovery_generation = 8;
        let error = receive_pages(vec![first, second])
            .await
            .expect_err("changed page generation must fail");
        assert!(error.to_string().contains("page generation mismatch"));
    }

    #[gpui::test]
    async fn structured_execution_remote_snapshot_validates_run_generations() {
        for (current, generation, valid) in [
            (true, 2, true),
            (true, 1, false),
            (false, 1, true),
            (false, 3, false),
        ] {
            let mut page = remote_page(0, 1, 0);
            page.project_generation = 7;
            page.discovery_generation = 2;
            let mut run = StructuredRun::new(
                StructuredRunId("run".into()),
                DiscoveryGeneration(generation),
                Vec::new(),
            );
            run.phase = if current {
                StructuredRunPhase::Running
            } else {
                StructuredRunPhase::Completed
            };
            if current {
                page.current_run = Some(run_to_proto(&run, &[], None, None));
            } else {
                page.last_complete_run = Some(run_to_proto(&run, &[], None, None));
            }
            let mut chunk = remote_event_chunk(0, 1, true);
            chunk.project_generation = 7;
            chunk.discovery_generation = generation;
            for event in &mut chunk.events {
                event.node_id = "case-0".into();
            }
            let result = receive_responses(vec![
                page.into_envelope(0, None, None),
                chunk.into_envelope(0, None, None),
            ])
            .await;
            if valid {
                let snapshot = result.expect("matching current and older retained runs are valid");
                let run = if current {
                    snapshot.current_run
                } else {
                    snapshot.last_complete_run
                }
                .expect("requested run retained");
                assert_eq!(run.discovery_generation, DiscoveryGeneration(generation));
                assert_eq!(run.events.len(), 1);
            } else {
                assert!(
                    result
                        .expect_err("invalid run generation must fail")
                        .to_string()
                        .contains("run generation mismatch")
                );
            }
        }
    }

    #[gpui::test]
    async fn structured_execution_remote_events_reject_generation_mismatch() {
        for (project_generation, discovery_generation) in [(2, 1), (1, 2)] {
            for preceding_valid_chunk in [false, true] {
                let mut chunks = Vec::new();
                if preceding_valid_chunk {
                    chunks.push(remote_event_chunk(0, 1, false));
                }
                let mut chunk = remote_event_chunk(usize::from(preceding_valid_chunk), 1, true);
                chunk.project_generation = project_generation;
                chunk.discovery_generation = discovery_generation;
                chunks.push(chunk);
                let error = receive_event_chunks(chunks)
                    .await
                    .expect_err("events from another generation must not be accepted");
                assert!(error.to_string().contains("generation mismatch"));
            }
        }
    }

    #[gpui::test]
    async fn structured_execution_remote_events_enforce_limits_before_conversion() {
        let mut empty = remote_event_chunk(0, 0, false);
        empty.next_sequence = 1;
        let error = receive_event_chunks(vec![empty])
            .await
            .expect_err("empty chunks cannot request another page");
        assert!(error.to_string().contains("invalid continuation"));
        let error = receive_event_chunks(vec![remote_event_chunk(
            0,
            MAX_STRUCTURED_EVENT_CHUNK_SIZE + 1,
            true,
        )])
        .await
        .expect_err("oversized event chunk must be rejected");
        assert!(error.to_string().contains("chunk limit"));
        let chunks = (0..MAX_STRUCTURED_EVENTS)
            .step_by(MAX_STRUCTURED_EVENT_CHUNK_SIZE)
            .map(|start| {
                let count = MAX_STRUCTURED_EVENT_CHUNK_SIZE.min(MAX_STRUCTURED_EVENTS - start);
                remote_event_chunk(start, count, start + count == MAX_STRUCTURED_EVENTS)
            })
            .collect::<Vec<_>>();
        let events = receive_event_chunks(chunks.clone())
            .await
            .expect("maximum legal remote events");
        assert_eq!(events.len(), MAX_STRUCTURED_EVENTS);
        let mut continued = chunks.clone();
        continued.last_mut().expect("last chunk").complete = false;
        let error = receive_event_chunks(continued)
            .await
            .expect_err("must stop at the aggregate event bound");
        assert!(error.to_string().contains("invalid continuation"));
        let mut oversized = chunks;
        let extra = remote_event_chunk(MAX_STRUCTURED_EVENTS, 1, true).events;
        oversized
            .last_mut()
            .expect("last chunk")
            .events
            .extend(extra);
        let error = receive_event_chunks(oversized)
            .await
            .expect_err("aggregate event limit must be enforced");
        assert!(error.to_string().contains("event limit"));
    }

    fn project_path(worktree: u64, path: &'static str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_proto(worktree),
            path: Arc::from(RelPath::from_unix_str(path).expect("fixture path should be relative")),
        }
    }

    fn fake_provider(generation: u64) -> StructuredProviderSnapshot {
        let provider = StructuredNodeId("js-provider".to_string());
        let suite = StructuredNodeId("web-suite".to_string());
        StructuredProviderSnapshot::discovery(
            StructuredProviderId("web-tests".to_string()),
            DiscoveryGeneration(generation),
            StructuredProviderStatus::Current,
            vec![
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: provider.clone(),
                    parent_id: None,
                    label: "Web tests".to_string(),
                    kind: StructuredNodeKind::Provider,
                    path: None,
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: suite.clone(),
                    parent_id: Some(provider),
                    label: "Browser suite".to_string(),
                    kind: StructuredNodeKind::Suite,
                    path: Some(project_path(1, "web/tests.js")),
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: StructuredNodeId("case-a".to_string()),
                    parent_id: Some(suite.clone()),
                    label: "renders a page".to_string(),
                    kind: StructuredNodeKind::Case,
                    path: Some(project_path(1, "web/tests.js")),
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: StructuredNodeId("private-case".to_string()),
                    parent_id: Some(suite),
                    label: "private case".to_string(),
                    kind: StructuredNodeKind::Case,
                    path: Some(project_path(2, "private/tests.js")),
                },
            ],
        )
    }

    #[gpui::test]
    async fn doctor_restart_save_blocks_queued_runs_until_terminal(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
        });
        let project = crate::Project::test(crate::FakeFs::new(cx.executor()), [], cx).await;
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let provider = fake_provider(4);
        let provider_id = provider.provider_id.clone();
        let run_id = StructuredRunId("doctor-restart-queued".into());
        assert!(!project.read_with(cx, |project, cx| {
            project.has_active_structured_execution(cx)
        }));
        store.update(cx, |store, _| {
            let generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(generation, provider, None)
                .expect("Discovery");
            store
                .state_mut()
                .begin_run(
                    generation,
                    &provider_id,
                    StructuredRun::new(
                        run_id.clone(),
                        DiscoveryGeneration(4),
                        vec![StructuredNodeId("case-a".into())],
                    ),
                )
                .expect("Queued test without terminal");
        });
        assert!(project.read_with(cx, |project, cx| {
            project.has_active_structured_execution(cx)
        }));
        for phase in [StructuredRunPhase::Running, StructuredRunPhase::Cancelled] {
            store.update(cx, |store, _| {
                let generation = store.state().project_generation();
                store
                    .state_mut()
                    .set_run_phase(generation, &provider_id, &run_id, phase, None)
                    .expect("Observed phase");
            });
            assert_eq!(
                project.read_with(cx, |project, cx| project
                    .has_active_structured_execution(cx)),
                !phase.is_terminal()
            );
        }
    }

    #[gpui::test]
    async fn structured_execution_task_subscription_does_not_reenter_store(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let project = crate::Project::test(crate::FakeFs::new(cx.executor()), [], cx).await;
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let provider = fake_provider(4);
        let provider_id = provider.provider_id.clone();
        let run_id = StructuredRunId("task-subscription".to_string());
        let handle = StructuredTaskHandle::new(task::TaskId("task-subscription".to_string()));
        store.update(cx, |store, cx| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(project_generation, provider, None)
                .expect("publish discovery");
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &provider_id,
                    StructuredRun::new(
                        run_id.clone(),
                        DiscoveryGeneration(4),
                        vec![StructuredNodeId("case-a".to_string())],
                    ),
                )
                .expect("begin run");
            store
                .observe_task_handle(provider_id.clone(), run_id, &handle, cx)
                .expect("subscribe without a nested store update");
        });
        cx.run_until_parked();
        assert_eq!(
            store.read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&provider_id)
                    .and_then(|provider| provider.current_run.as_ref())
                    .map(|run| run.phase)
            }),
            Some(StructuredRunPhase::Queued)
        );
        cx.update(|cx| {
            assert!(handle.mark_running(None, cx));
        });
        cx.run_until_parked();
        assert_eq!(
            store.read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&provider_id)
                    .and_then(|provider| provider.current_run.as_ref())
                    .map(|run| run.phase)
            }),
            Some(StructuredRunPhase::Running)
        );
        cx.update(|cx| {
            assert!(handle.mark_cancelled(None, true, cx));
        });
        cx.run_until_parked();
        assert_eq!(
            store.read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&provider_id)
                    .and_then(|provider| provider.current_run.as_ref())
                    .map(|run| run.phase)
            }),
            Some(StructuredRunPhase::Cancelled)
        );
    }

    #[test]
    fn structured_execution_state_rejects_duplicates_gaps_and_stale_generations() {
        let mut state = StructuredExecutionState::new(11);
        let provider = fake_provider(4);
        let provider_id = provider.provider_id.clone();
        assert_eq!(
            state
                .apply_discovery(11, provider.clone(), None)
                .expect("discovery should apply"),
            StructuredApplyOutcome::Applied
        );
        assert_eq!(
            state
                .apply_discovery(11, provider, None)
                .expect("identical generation should be idempotent"),
            StructuredApplyOutcome::Duplicate
        );
        let run_id = StructuredRunId("run-1".to_string());
        state
            .begin_run(
                11,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    DiscoveryGeneration(4),
                    vec![StructuredNodeId("case-a".to_string())],
                ),
            )
            .expect("run should begin");
        let event = StructuredExecutionEvent {
            sequence: 0,
            node_id: StructuredNodeId("case-a".to_string()),
            state: StructuredNodeState::Passed,
            duration_millis: Some(8),
            message: Some("safe failure summary".to_string()),
            location: Some(project_path(1, "web/tests.js")),
        };
        assert_eq!(
            state
                .apply_event(11, &provider_id, &run_id, event.clone(), None)
                .expect("event should apply"),
            StructuredApplyOutcome::Applied
        );
        assert_eq!(
            state
                .apply_event(11, &provider_id, &run_id, event, None)
                .expect("duplicate should be idempotent"),
            StructuredApplyOutcome::Duplicate
        );
        let gap = StructuredExecutionEvent {
            sequence: 2,
            node_id: StructuredNodeId("case-a".to_string()),
            state: StructuredNodeState::Failed,
            duration_millis: None,
            message: None,
            location: None,
        };
        assert!(
            state
                .apply_event(11, &provider_id, &run_id, gap, None)
                .expect_err("sequence gaps must fail")
                .to_string()
                .contains("gap")
        );
        assert!(
            state
                .apply_discovery(10, fake_provider(5), None)
                .expect_err("cross-project generation must fail")
                .to_string()
                .contains("cross-project")
        );
        assert!(
            state
                .apply_discovery(11, fake_provider(3), None)
                .expect_err("stale discovery must fail")
                .to_string()
                .contains("stale")
        );
        let newer_run_id = StructuredRunId("run-2".to_string());
        state
            .begin_run(
                11,
                &provider_id,
                StructuredRun::new(
                    newer_run_id,
                    DiscoveryGeneration(4),
                    vec![StructuredNodeId("case-a".to_string())],
                ),
            )
            .expect("newer run should supersede the active run");
        let late_event = StructuredExecutionEvent {
            sequence: 1,
            node_id: StructuredNodeId("case-a".to_string()),
            state: StructuredNodeState::Passed,
            duration_millis: None,
            message: None,
            location: None,
        };
        assert!(
            state
                .apply_event(11, &provider_id, &run_id, late_event, None)
                .expect_err("events from superseded runs must fail")
                .to_string()
                .contains("stale")
        );
    }

    #[test]
    fn structured_execution_contract_remains_ecosystem_neutral() {
        let source = include_str!("structured_execution.rs")
            .split_once("#[cfg(test)]")
            .map(|(production_source, _)| production_source)
            .expect("structured execution tests should follow production code");
        for forbidden in [
            concat!("cargo_", "metadata"),
            concat!("Cargo", "Workspace"),
            concat!("terminal_", "bytes"),
            concat!("project_", "env"),
        ] {
            assert!(
                !source.contains(forbidden),
                "generic structured execution source contains forbidden term {forbidden}"
            );
        }
    }

    #[test]
    fn structured_task_bridge_maps_lifecycle_and_rejects_late_events() {
        let mut state = StructuredExecutionState::new(21);
        let provider = fake_provider(8);
        let provider_id = provider.provider_id.clone();
        state
            .apply_discovery(21, provider, None)
            .expect("provider discovery should apply");
        let run_id = StructuredRunId("task-run".to_string());
        state
            .begin_run(
                21,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    DiscoveryGeneration(8),
                    vec![StructuredNodeId("case-a".to_string())],
                ),
            )
            .expect("run should begin");
        let running = StructuredTaskLifecycleEvent {
            task_id: task::TaskId("task-a".to_string()),
            state: StructuredTaskState::Running {
                terminal_id: Some(task::StructuredTerminalId(9)),
            },
        };
        apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &running)
            .expect("running lifecycle should apply");
        let completed = StructuredTaskLifecycleEvent {
            task_id: running.task_id.clone(),
            state: StructuredTaskState::Completed {
                terminal_id: Some(task::StructuredTerminalId(9)),
                exit_code: Some(7),
                success: false,
            },
        };
        apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &completed)
            .expect("completion should apply");
        let snapshot = state
            .provider(&provider_id)
            .expect("provider should remain");
        assert_eq!(
            snapshot
                .last_complete_run
                .as_ref()
                .and_then(|run| run.diagnostic.as_deref()),
            Some("Task exited with status 7")
        );
        let completed_run = snapshot
            .last_complete_run
            .as_ref()
            .expect("completed task should be retained");
        assert_eq!(completed_run.summary.total, 1);
        assert_eq!(completed_run.summary.running, 0);
        assert_eq!(completed_run.summary.failed, 1);
        assert_eq!(completed_run.events.len(), 2);
        assert_eq!(
            completed_run.events.last().map(|event| event.state),
            Some(StructuredNodeState::Failed)
        );

        let late_cancel = StructuredTaskLifecycleEvent {
            task_id: running.task_id,
            state: StructuredTaskState::Cancelled {
                terminal_id: Some(task::StructuredTerminalId(9)),
                termination_confirmed: true,
            },
        };
        assert!(
            apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &late_cancel)
                .expect_err("terminal runs must reject late lifecycle events")
                .to_string()
                .contains("terminal")
        );
    }

    #[test]
    fn structured_task_bridge_preserves_unknown_children_for_aggregate_runs() {
        let provider_id = StructuredProviderId("aggregate-tests".into());
        let provider = StructuredProviderSnapshot::discovery(
            provider_id.clone(),
            DiscoveryGeneration(8),
            StructuredProviderStatus::Current,
            [
                ("provider", None, "Tests", StructuredNodeKind::Provider),
                (
                    "package",
                    Some("provider"),
                    "Package",
                    StructuredNodeKind::Suite,
                ),
                (
                    "target",
                    Some("package"),
                    "Unit target",
                    StructuredNodeKind::Group,
                ),
                (
                    "module",
                    Some("target"),
                    "Module",
                    StructuredNodeKind::Group,
                ),
                (
                    "case-a",
                    Some("module"),
                    "First case",
                    StructuredNodeKind::Case,
                ),
                (
                    "case-b",
                    Some("module"),
                    "Second case",
                    StructuredNodeKind::Case,
                ),
            ]
            .into_iter()
            .map(|(identifier, parent, label, kind)| StructuredNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                id: StructuredNodeId(identifier.into()),
                parent_id: parent.map(|identifier| StructuredNodeId(identifier.into())),
                label: label.into(),
                kind,
                path: None,
            })
            .collect(),
        );
        for scope_node_ids in [
            vec![StructuredNodeId("package".into())],
            vec![StructuredNodeId("target".into())],
            vec![StructuredNodeId("module".into())],
            vec![
                StructuredNodeId("case-a".into()),
                StructuredNodeId("case-b".into()),
            ],
        ] {
            for success in [true, false] {
                let mut state = StructuredExecutionState::new(21);
                state
                    .apply_discovery(21, provider.clone(), None)
                    .expect("aggregate discovery should apply");
                let run_id = StructuredRunId("aggregate-run".into());
                state
                    .begin_run(
                        21,
                        &provider_id,
                        StructuredRun::new(
                            run_id.clone(),
                            DiscoveryGeneration(8),
                            scope_node_ids.clone(),
                        ),
                    )
                    .expect("aggregate scope should be admitted");
                for (task_state, phase) in [
                    (StructuredTaskState::Queued, StructuredRunPhase::Queued),
                    (
                        StructuredTaskState::Running {
                            terminal_id: Some(task::StructuredTerminalId(17)),
                        },
                        StructuredRunPhase::Running,
                    ),
                    (
                        StructuredTaskState::Completed {
                            terminal_id: Some(task::StructuredTerminalId(17)),
                            exit_code: Some(if success { 0 } else { 7 }),
                            success,
                        },
                        StructuredRunPhase::Completed,
                    ),
                ] {
                    apply_task_lifecycle(
                        &mut state,
                        21,
                        &provider_id,
                        &run_id,
                        &StructuredTaskLifecycleEvent {
                            task_id: task::TaskId("aggregate-task".into()),
                            state: task_state,
                        },
                    )
                    .expect("aggregate lifecycle should apply");
                    let run = state
                        .provider(&provider_id)
                        .and_then(|provider| provider.current_run.as_ref())
                        .expect("aggregate run should remain visible");
                    assert_eq!(run.phase, phase);
                    assert_eq!(run.scope_node_ids, scope_node_ids);
                    assert_eq!(run.discovery_generation, DiscoveryGeneration(8));
                    assert!(run.events.is_empty());
                    assert!(run.latest_states.is_empty());
                    assert_eq!(run.summary, StructuredExecutionSummary::default());
                }
                let completed = state
                    .provider(&provider_id)
                    .and_then(|provider| provider.last_complete_run.as_ref())
                    .expect("aggregate completion should be retained");
                assert_eq!(completed.id, run_id);
                assert_eq!(completed.phase, StructuredRunPhase::Completed);
                assert_eq!(
                    completed.diagnostic.as_deref(),
                    if success {
                        None
                    } else {
                        Some("Task exited with status 7")
                    }
                );
                assert!(completed.events.is_empty());
                assert!(completed.latest_states.is_empty());
                assert_eq!(completed.summary, StructuredExecutionSummary::default());
            }
        }
    }

    #[test]
    fn structured_task_bridge_accepts_only_cancellation_confirmation() {
        let mut state = StructuredExecutionState::new(21);
        let provider = fake_provider(8);
        let provider_id = provider.provider_id.clone();
        state
            .apply_discovery(21, provider, None)
            .expect("provider discovery should apply");
        let run_id = StructuredRunId("cancelled-task-run".to_string());
        state
            .begin_run(
                21,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    DiscoveryGeneration(8),
                    vec![StructuredNodeId("case-a".to_string())],
                ),
            )
            .expect("run should begin");
        let cancelled = StructuredTaskLifecycleEvent {
            task_id: task::TaskId("task-a".to_string()),
            state: StructuredTaskState::Cancelled {
                terminal_id: Some(task::StructuredTerminalId(9)),
                termination_confirmed: false,
            },
        };
        apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &cancelled)
            .expect("cancellation request should apply");
        let confirmed = StructuredTaskLifecycleEvent {
            task_id: cancelled.task_id.clone(),
            state: StructuredTaskState::Cancelled {
                terminal_id: Some(task::StructuredTerminalId(9)),
                termination_confirmed: true,
            },
        };
        assert_eq!(
            apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &confirmed)
                .expect("host termination confirmation should apply"),
            StructuredApplyOutcome::Applied
        );
        assert_eq!(
            apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &confirmed)
                .expect("duplicate confirmation should be harmless"),
            StructuredApplyOutcome::Duplicate
        );
        let snapshot = state
            .provider(&provider_id)
            .expect("provider should remain");
        let current = snapshot.current_run.as_ref().expect("current run");
        assert_eq!(current.phase, StructuredRunPhase::Cancelled);
        assert!(current.diagnostic.is_none());
        assert!(current.events.iter().all(|event| event.message.is_none()));
        let retained = snapshot.completed_runs.back().expect("retained run");
        assert!(retained.diagnostic.is_none());
        assert!(retained.events.iter().all(|event| event.message.is_none()));
        let late_completion = StructuredTaskLifecycleEvent {
            task_id: cancelled.task_id,
            state: StructuredTaskState::Completed {
                terminal_id: Some(task::StructuredTerminalId(9)),
                exit_code: Some(0),
                success: true,
            },
        };
        assert!(
            apply_task_lifecycle(&mut state, 21, &provider_id, &run_id, &late_completion).is_err()
        );
    }

    #[test]
    fn structured_execution_filters_paths_and_enforces_retention_and_bounds() {
        let limits = StructuredExecutionLimits {
            nodes: 3,
            events_per_run: 2,
            message_bytes_per_run: 8,
            completed_runs: 1,
        };
        let mut state = StructuredExecutionState::with_limits(1, limits);
        let visible = HashSet::from_iter([WorktreeId::from_proto(1)]);
        let provider = fake_provider(1);
        let provider_id = provider.provider_id.clone();
        assert_eq!(
            state
                .apply_discovery(1, provider, Some(&visible))
                .expect("filtered discovery should apply"),
            StructuredApplyOutcome::Truncated
        );
        let snapshot = state.provider(&provider_id).expect("provider should exist");
        assert_eq!(snapshot.nodes.len(), 3);
        assert!(snapshot.partial);
        assert!(snapshot.nodes.iter().all(|node| {
            node.path
                .as_ref()
                .is_none_or(|path| path.worktree_id == WorktreeId::from_proto(1))
        }));

        for index in 0..2 {
            let run_id = StructuredRunId(format!("run-{index}"));
            state
                .begin_run(
                    1,
                    &provider_id,
                    StructuredRun::new(
                        run_id.clone(),
                        DiscoveryGeneration(1),
                        vec![StructuredNodeId("case-a".to_string())],
                    ),
                )
                .expect("run should begin");
            state
                .apply_event(
                    1,
                    &provider_id,
                    &run_id,
                    StructuredExecutionEvent {
                        sequence: 0,
                        node_id: StructuredNodeId("case-a".to_string()),
                        state: StructuredNodeState::Passed,
                        duration_millis: None,
                        message: Some("secret-value-is-too-long".to_string()),
                        location: None,
                    },
                    Some(&visible),
                )
                .expect("event should be bounded");
            state
                .set_run_phase(
                    1,
                    &provider_id,
                    &run_id,
                    StructuredRunPhase::Completed,
                    None,
                )
                .expect("run should complete");
        }
        let snapshot = state
            .provider(&provider_id)
            .expect("provider should remain");
        assert_eq!(snapshot.completed_runs.len(), 1);
        let retained = snapshot
            .last_complete_run
            .as_ref()
            .expect("last complete run should be separate");
        assert!(retained.truncated);
        assert_eq!(retained.events[0].message.as_deref(), Some("secret-v"));
    }

    #[test]
    fn structured_jvm_results_hide_revoked_modules_and_generated_sources() {
        let provider_id = StructuredProviderId("jvm-junit".to_string());
        let root_id = StructuredNodeId("jvm-root".to_string());
        let group_id = StructuredNodeId("jvm-module".to_string());
        let case_id = StructuredNodeId("jvm-case".to_string());
        let manifest = project_path(1, "module/pom.xml");
        let generated = project_path(1, "module/target/generated-sources/Secret.java");
        let snapshot = StructuredProviderSnapshot::discovery(
            provider_id.clone(),
            DiscoveryGeneration(1),
            StructuredProviderStatus::Current,
            vec![
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: root_id.clone(),
                    parent_id: None,
                    label: "Java tests".to_string(),
                    kind: StructuredNodeKind::Provider,
                    path: None,
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: group_id.clone(),
                    parent_id: Some(root_id),
                    label: "private module".to_string(),
                    kind: StructuredNodeKind::Group,
                    path: Some(manifest.clone()),
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: case_id.clone(),
                    parent_id: Some(group_id),
                    label: "private test".to_string(),
                    kind: StructuredNodeKind::Case,
                    path: None,
                },
            ],
        );
        let mut state = StructuredExecutionState::new(1);
        state
            .apply_discovery(1, snapshot, None)
            .expect("JVM discovery applies");
        let run_id = StructuredRunId("jvm-run".to_string());
        state
            .begin_run(
                1,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    DiscoveryGeneration(1),
                    vec![case_id.clone()],
                ),
            )
            .expect("run begins");
        state
            .apply_event(
                1,
                &provider_id,
                &run_id,
                StructuredExecutionEvent {
                    sequence: 0,
                    node_id: case_id,
                    state: StructuredNodeState::Failed,
                    duration_millis: None,
                    message: Some("private failure".to_string()),
                    location: Some(generated),
                },
                None,
            )
            .expect("result applies");
        state
            .apply_event(
                1,
                &provider_id,
                &run_id,
                StructuredExecutionEvent {
                    sequence: 1,
                    node_id: StructuredNodeId("jvm-case".to_string()),
                    state: StructuredNodeState::Failed,
                    duration_millis: None,
                    message: Some("hidden worktree failure".to_string()),
                    location: Some(project_path(2, "private/Hidden.java")),
                },
                None,
            )
            .expect("cross-worktree result applies");
        state
            .set_run_phase(
                1,
                &provider_id,
                &run_id,
                StructuredRunPhase::Completed,
                None,
            )
            .expect("run completes");

        let visible_worktrees = HashSet::from([WorktreeId::from_proto(1)]);
        let worktree_filtered = snapshot_page_to_proto_filtered(
            &state,
            &provider_id,
            DiscoveryGeneration(1),
            0,
            MAX_STRUCTURED_PAGE_SIZE,
            Some(&visible_worktrees),
            None,
        )
        .expect("hidden worktree result is filtered");
        assert_eq!(
            worktree_filtered
                .last_complete_run
                .as_ref()
                .and_then(|run| run.summary.as_ref())
                .map(|summary| summary.total),
            Some(1)
        );
        assert!(!format!("{worktree_filtered:?}").contains("hidden worktree failure"));
        let revoked = snapshot_page_to_proto_filtered(
            &state,
            &provider_id,
            DiscoveryGeneration(1),
            0,
            MAX_STRUCTURED_PAGE_SIZE,
            Some(&visible_worktrees),
            Some(&|_| false),
        )
        .expect("revoked snapshot is filtered");
        assert_eq!(revoked.nodes.len(), 1);
        assert!(revoked.partial);
        assert_eq!(
            revoked.status,
            provider_status_to_proto(StructuredProviderStatus::Partial)
        );
        assert!(revoked.current_run.is_none());
        assert!(revoked.last_complete_run.is_none());
        assert!(!format!("{revoked:?}").contains("private"));
        assert!(
            event_chunk_to_proto(
                &state,
                &provider_id,
                &run_id,
                0,
                MAX_STRUCTURED_EVENT_CHUNK_SIZE,
                Some(&visible_worktrees),
                Some(&|_| false),
            )
            .expect_err("a revoked module cannot reveal its run")
            .to_string()
            .contains("unknown structured run")
        );

        let manifest_only = |path: &ProjectPath| path == &manifest;
        let allowed_module = snapshot_page_to_proto_filtered(
            &state,
            &provider_id,
            DiscoveryGeneration(1),
            0,
            MAX_STRUCTURED_PAGE_SIZE,
            Some(&visible_worktrees),
            Some(&manifest_only),
        )
        .expect("module remains visible without generated source");
        assert_eq!(allowed_module.nodes.len(), 3);
        assert_eq!(
            allowed_module
                .last_complete_run
                .as_ref()
                .and_then(|run| run.summary.as_ref())
                .map(|summary| summary.total),
            Some(0)
        );
        let filtered_events = event_chunk_to_proto(
            &state,
            &provider_id,
            &run_id,
            0,
            MAX_STRUCTURED_EVENT_CHUNK_SIZE,
            Some(&visible_worktrees),
            Some(&manifest_only),
        )
        .expect("generated source event is filtered");
        assert!(filtered_events.events.is_empty());
        assert!(!format!("{filtered_events:?}").contains("private failure"));
    }

    #[test]
    fn rust_result_remains_available_for_a_permitted_source() {
        let provider_id = StructuredProviderId("rust-tests".to_string());
        let case_id = StructuredNodeId("rust-case".to_string());
        let source = project_path(1, "src/lib.rs");
        let mut state = StructuredExecutionState::new(1);
        state
            .apply_discovery(
                1,
                StructuredProviderSnapshot::discovery(
                    provider_id.clone(),
                    DiscoveryGeneration(1),
                    StructuredProviderStatus::Current,
                    vec![
                        StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: StructuredNodeId("rust-root".to_string()),
                            parent_id: None,
                            label: "Rust tests".to_string(),
                            kind: StructuredNodeKind::Provider,
                            path: None,
                        },
                        StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: case_id.clone(),
                            parent_id: Some(StructuredNodeId("rust-root".to_string())),
                            label: "it_works".to_string(),
                            kind: StructuredNodeKind::Case,
                            path: Some(source.clone()),
                        },
                    ],
                ),
                None,
            )
            .expect("Rust discovery applies");
        let run_id = StructuredRunId("rust-run".to_string());
        state
            .begin_run(
                1,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    DiscoveryGeneration(1),
                    vec![case_id.clone()],
                ),
            )
            .expect("Rust run begins");
        state
            .apply_event(
                1,
                &provider_id,
                &run_id,
                StructuredExecutionEvent {
                    sequence: 0,
                    node_id: case_id,
                    state: StructuredNodeState::Passed,
                    duration_millis: None,
                    message: None,
                    location: Some(source.clone()),
                },
                None,
            )
            .expect("Rust result applies");
        state
            .set_run_phase(
                1,
                &provider_id,
                &run_id,
                StructuredRunPhase::Completed,
                None,
            )
            .expect("Rust run completes");

        let visible_worktrees = HashSet::from([WorktreeId::from_proto(1)]);
        let include_source = |path: &ProjectPath| path == &source;
        let page = snapshot_page_to_proto_filtered(
            &state,
            &provider_id,
            DiscoveryGeneration(1),
            0,
            MAX_STRUCTURED_PAGE_SIZE,
            Some(&visible_worktrees),
            Some(&include_source),
        )
        .expect("permitted Rust result is shared");
        assert_eq!(page.nodes.len(), 2);
        assert_eq!(
            page.last_complete_run
                .as_ref()
                .and_then(|run| run.summary.as_ref())
                .map(|summary| summary.passed),
            Some(1)
        );
        let events = event_chunk_to_proto(
            &state,
            &provider_id,
            &run_id,
            0,
            MAX_STRUCTURED_EVENT_CHUNK_SIZE,
            Some(&visible_worktrees),
            Some(&include_source),
        )
        .expect("permitted Rust event is shared");
        assert_eq!(events.events.len(), 1);
    }

    #[test]
    fn rust_workspace_bounds_ten_thousand_structured_test_nodes_deterministically() {
        let provider_id = StructuredProviderId("scale-tests".to_string());
        let provider_node_id = StructuredNodeId("scale-provider".to_string());
        let nodes = std::iter::once(StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: provider_node_id.clone(),
            parent_id: None,
            label: "Scale tests".to_string(),
            kind: StructuredNodeKind::Provider,
            path: None,
        })
        .chain((0..10_000).map(|index| StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: StructuredNodeId(format!("scale-case-{index:05}")),
            parent_id: Some(provider_node_id.clone()),
            label: format!("case {index:05}"),
            kind: StructuredNodeKind::Case,
            path: Some(project_path(1, "tests/scale.rs")),
        }))
        .collect::<Vec<_>>();
        let snapshot = StructuredProviderSnapshot::discovery(
            provider_id.clone(),
            DiscoveryGeneration(1),
            StructuredProviderStatus::Current,
            nodes,
        );

        let mut first = StructuredExecutionState::new(1);
        let mut second = StructuredExecutionState::new(1);
        assert_eq!(
            first
                .apply_discovery(1, snapshot.clone(), None)
                .expect("large discovery should be bounded"),
            StructuredApplyOutcome::Truncated
        );
        assert_eq!(
            second
                .apply_discovery(1, snapshot, None)
                .expect("repeated large discovery should be bounded"),
            StructuredApplyOutcome::Truncated
        );
        let first_provider = first
            .provider(&provider_id)
            .expect("provider should be retained");
        let second_provider = second
            .provider(&provider_id)
            .expect("provider should be retained");
        assert_eq!(first_provider.nodes.len(), MAX_STRUCTURED_NODES);
        assert_eq!(first_provider.nodes, second_provider.nodes);
        assert_eq!(first_provider.status, StructuredProviderStatus::Partial);
        assert!(first_provider.partial);
        assert!(
            first_provider
                .diagnostic
                .as_deref()
                .is_some_and(|message| message.contains("10000"))
        );
    }

    #[test]
    fn structured_execution_proto_rejects_malformed_enums_and_bounds_pages() {
        assert!(
            node_from_proto(proto::StructuredExecutionNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                node_id: "node".to_string(),
                parent_id: None,
                label: "label".to_string(),
                kind: 99,
                path: None,
            })
            .expect_err("unknown node kind must fail")
            .to_string()
            .contains("invalid")
        );
        let mut state = StructuredExecutionState::new(4);
        let provider = fake_provider(2);
        let provider_id = provider.provider_id.clone();
        state
            .apply_discovery(4, provider, None)
            .expect("fixture should apply");
        let page = snapshot_page_to_proto(
            &state,
            &provider_id,
            DiscoveryGeneration(2),
            0,
            usize::MAX,
            None,
        )
        .expect("page should serialize");
        assert!(page.nodes.len() <= MAX_STRUCTURED_PAGE_SIZE);
        let encoded = format!("{page:?}");
        assert!(!encoded.contains("project_env"));
        assert!(!encoded.contains("/Users/"));
        assert!(!encoded.contains("terminal_bytes"));
    }

    #[test]
    fn structured_execution_proto_includes_run_metadata_only_on_the_first_page() {
        let mut state = StructuredExecutionState::new(4);
        let provider = fake_provider(2);
        let provider_id = provider.provider_id.clone();
        let run = StructuredRun::new(
            StructuredRunId("run".to_string()),
            DiscoveryGeneration(2),
            vec![StructuredNodeId("case-a".to_string())],
        );
        state
            .apply_discovery(4, provider, None)
            .expect("fixture should apply");
        state
            .begin_run(4, &provider_id, run)
            .expect("fixture run should begin");
        state
            .set_run_phase(
                4,
                &provider_id,
                &StructuredRunId("run".to_string()),
                StructuredRunPhase::Completed,
                None,
            )
            .expect("fixture run should complete");

        let first_page =
            snapshot_page_to_proto(&state, &provider_id, DiscoveryGeneration(2), 0, 1, None)
                .expect("first page should serialize");
        let continuation_page =
            snapshot_page_to_proto(&state, &provider_id, DiscoveryGeneration(2), 1, 1, None)
                .expect("continuation page should serialize");

        assert!(first_page.current_run.is_some());
        assert!(first_page.last_complete_run.is_some());
        assert!(continuation_page.current_run.is_none());
        assert!(continuation_page.last_complete_run.is_none());
    }
}
