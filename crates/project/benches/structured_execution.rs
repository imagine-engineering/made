use std::{
    hint::black_box,
    mem::size_of,
    time::{Duration, Instant},
};

use criterion::{Criterion, criterion_group, criterion_main};
use project::structured_execution::{
    DiscoveryGeneration, MAX_STRUCTURED_NODES, MAX_STRUCTURED_PAGE_SIZE, StructuredDebugCapability,
    StructuredExecutionEvent, StructuredExecutionState, StructuredNode, StructuredNodeId,
    StructuredNodeKind, StructuredNodeState, StructuredProviderId, StructuredProviderSnapshot,
    StructuredProviderStatus, StructuredRun, StructuredRunId, snapshot_page_to_proto,
};

const DISCOVERY_BUDGET: Duration = Duration::from_secs(2);
const EVENT_REDUCTION_BUDGET: Duration = Duration::from_secs(2);
const PAGINATION_BUDGET: Duration = Duration::from_millis(100);
const RETAINED_MODEL_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const MAXIMUM_DEBUG_REASON_BYTES: usize = 4096;

fn provider_id() -> StructuredProviderId {
    StructuredProviderId("synthetic-provider".to_string())
}

fn nodes() -> Vec<StructuredNode> {
    let root_id = StructuredNodeId("provider".to_string());
    let mut nodes = Vec::with_capacity(MAX_STRUCTURED_NODES);
    nodes.push(StructuredNode {
        experimental_feature: None,
        experimental_feature_enabled: None,
        debug_capability: None,
        id: root_id.clone(),
        parent_id: None,
        label: "Synthetic provider".to_string(),
        kind: StructuredNodeKind::Provider,
        path: None,
    });
    for index in 1..MAX_STRUCTURED_NODES {
        nodes.push(StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: StructuredNodeId(format!("case-{index:05}")),
            parent_id: Some(root_id.clone()),
            label: format!("Synthetic case {index:05}"),
            kind: StructuredNodeKind::Case,
            path: None,
        });
    }
    nodes
}

fn capability_nodes() -> Vec<StructuredNode> {
    let mut nodes = nodes();
    let prefix = "Debug unavailable: ";
    let reason = format!(
        "{prefix}{}",
        "x".repeat(MAXIMUM_DEBUG_REASON_BYTES - prefix.len())
    );
    for (index, node) in nodes.iter_mut().enumerate().skip(1) {
        node.debug_capability = match index % 3 {
            0 => Some(StructuredDebugCapability::Unsupported(reason.clone())),
            1 => Some(StructuredDebugCapability::Supported),
            _ => None,
        };
    }
    nodes
}

fn verify_capability_pages(state: &StructuredExecutionState, nodes: &[StructuredNode]) {
    let mut expected = nodes.iter();
    let mut page_start = 0;
    loop {
        let page = snapshot_page_to_proto(
            state,
            &provider_id(),
            DiscoveryGeneration(1),
            page_start,
            MAX_STRUCTURED_PAGE_SIZE,
            None,
        )
        .expect("capability page should serialize");
        for node in page.nodes {
            let expected = expected.next().expect("page contains only fixture nodes");
            assert_eq!(node.node_id, expected.id.0);
            match (&expected.debug_capability, &node.debug_capability) {
                (None, None) => {}
                (Some(StructuredDebugCapability::Supported), Some(fact)) => {
                    assert!(fact.supported);
                    assert!(fact.reason.is_none());
                }
                (Some(StructuredDebugCapability::Unsupported(reason)), Some(fact)) => {
                    assert!(!fact.supported);
                    assert_eq!(fact.reason.as_ref(), Some(reason));
                    assert_eq!(reason.len(), MAXIMUM_DEBUG_REASON_BYTES);
                }
                _ => panic!("capability changed during paging"),
            }
        }
        if page.next_page_start == 0 {
            break;
        }
        page_start = page.next_page_start as usize;
    }
    assert!(expected.next().is_none(), "all fixture nodes were paged");
}

fn discovery(nodes: Vec<StructuredNode>) -> StructuredExecutionState {
    let mut state = StructuredExecutionState::new(1);
    state
        .apply_discovery(
            1,
            StructuredProviderSnapshot::discovery(
                provider_id(),
                DiscoveryGeneration(1),
                StructuredProviderStatus::Current,
                nodes,
            ),
            None,
        )
        .expect("synthetic discovery should apply");
    state
}

fn reduce_events(nodes: &[StructuredNode]) -> StructuredExecutionState {
    let mut state = discovery(nodes.to_vec());
    let provider_id = provider_id();
    let run_id = StructuredRunId("synthetic-run".to_string());
    state
        .begin_run(
            1,
            &provider_id,
            StructuredRun::new(
                run_id.clone(),
                DiscoveryGeneration(1),
                nodes.iter().skip(1).map(|node| node.id.clone()).collect(),
            ),
        )
        .expect("synthetic run should begin");
    for (sequence, node) in nodes.iter().skip(1).enumerate() {
        state
            .apply_event(
                1,
                &provider_id,
                &run_id,
                StructuredExecutionEvent {
                    sequence: sequence as u64,
                    node_id: node.id.clone(),
                    state: StructuredNodeState::Passed,
                    duration_millis: Some(1),
                    message: None,
                    location: None,
                },
                None,
            )
            .expect("synthetic event should reduce");
    }
    state
}

fn paginate(state: &StructuredExecutionState) -> usize {
    let mut page_start = 0;
    let mut node_count = 0;
    loop {
        let page = snapshot_page_to_proto(
            state,
            &provider_id(),
            DiscoveryGeneration(1),
            page_start,
            MAX_STRUCTURED_PAGE_SIZE,
            None,
        )
        .expect("synthetic page should serialize");
        node_count += black_box(page.nodes).len();
        if page.next_page_start == 0 {
            break;
        }
        page_start = page.next_page_start as usize;
    }
    node_count
}

fn retained_model_bytes(state: &StructuredExecutionState) -> usize {
    let provider = state
        .provider(&provider_id())
        .expect("synthetic provider should exist");
    let mut bytes = size_of::<StructuredExecutionState>()
        + size_of::<StructuredProviderSnapshot>()
        + provider.nodes.capacity() * size_of::<StructuredNode>();
    for node in &provider.nodes {
        bytes += node.id.0.capacity()
            + node.label.capacity()
            + node.parent_id.as_ref().map_or(0, |id| id.0.capacity())
            + match &node.debug_capability {
                Some(StructuredDebugCapability::Unsupported(reason)) => reason.capacity(),
                _ => 0,
            };
    }
    let indexed_id_bytes = provider
        .nodes
        .iter()
        .map(|node| node.id.0.capacity())
        .sum::<usize>();
    bytes += indexed_id_bytes + provider.nodes.len() * (size_of::<StructuredNodeId>() + 32);
    if let Some(run) = &provider.current_run {
        bytes += size_of::<StructuredRun>()
            + run.scope_node_ids.capacity() * size_of::<StructuredNodeId>()
            + run.events.capacity() * size_of::<StructuredExecutionEvent>();
        bytes += run
            .scope_node_ids
            .iter()
            .map(|id| id.0.capacity())
            .sum::<usize>();
        bytes += run
            .events
            .iter()
            .map(|event| event.node_id.0.capacity())
            .sum::<usize>();
        bytes += run.events.len() * (size_of::<(StructuredNodeId, StructuredNodeState)>() + 32)
            + run
                .events
                .iter()
                .map(|event| event.node_id.0.capacity())
                .sum::<usize>();
    }
    bytes
}

fn benchmark_workload(criterion: &mut Criterion, nodes: Vec<StructuredNode>, group_name: &str) {
    let reason_bytes = nodes
        .iter()
        .filter_map(|node| match &node.debug_capability {
            Some(StructuredDebugCapability::Unsupported(reason)) => Some(reason.len()),
            _ => None,
        })
        .sum::<usize>();

    let started = Instant::now();
    let discovered = discovery(nodes.clone());
    let discovery_elapsed = started.elapsed();
    assert_eq!(paginate(&discovered), MAX_STRUCTURED_NODES);
    verify_capability_pages(&discovered, &nodes);
    assert!(
        discovery_elapsed <= DISCOVERY_BUDGET,
        "10,000-node discovery took {discovery_elapsed:?}, exceeding {DISCOVERY_BUDGET:?}"
    );

    let started = Instant::now();
    let reduced = reduce_events(&nodes);
    let event_elapsed = started.elapsed();
    let run = reduced
        .provider(&provider_id())
        .and_then(|provider| provider.current_run.as_ref())
        .expect("synthetic run should exist");
    assert_eq!(run.summary.total as usize, MAX_STRUCTURED_NODES - 1);
    assert_eq!(run.summary.passed as usize, MAX_STRUCTURED_NODES - 1);
    assert!(
        event_elapsed <= EVENT_REDUCTION_BUDGET,
        "10,000-node event reduction took {event_elapsed:?}, exceeding {EVENT_REDUCTION_BUDGET:?}"
    );

    let started = Instant::now();
    assert_eq!(paginate(&reduced), MAX_STRUCTURED_NODES);
    let pagination_elapsed = started.elapsed();
    assert!(
        pagination_elapsed <= PAGINATION_BUDGET,
        "10,000-node pagination took {pagination_elapsed:?}, exceeding {PAGINATION_BUDGET:?}"
    );

    let retained_bytes = retained_model_bytes(&reduced);
    assert!(
        retained_bytes >= reason_bytes,
        "capability reason heap storage is omitted: retained {retained_bytes} bytes for {reason_bytes} reason bytes"
    );
    assert!(
        retained_bytes <= RETAINED_MODEL_BUDGET_BYTES,
        "10,000-node state retained {retained_bytes} bytes, exceeding {RETAINED_MODEL_BUDGET_BYTES}"
    );
    eprintln!(
        "structured-execution-budget nodes={MAX_STRUCTURED_NODES} discovery_ms={} event_reduction_ms={} pagination_ms={} retained_model_bytes={retained_bytes} reason_bytes={reason_bytes} workload={group_name}",
        discovery_elapsed.as_millis(),
        event_elapsed.as_millis(),
        pagination_elapsed.as_millis(),
    );

    let mut group = criterion.benchmark_group(group_name);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("discovery", |bencher| {
        bencher.iter(|| black_box(discovery(black_box(nodes.clone()))))
    });
    group.bench_function("event_reduction", |bencher| {
        bencher.iter(|| black_box(reduce_events(black_box(&nodes))))
    });
    group.bench_function("pagination", |bencher| {
        bencher.iter(|| black_box(paginate(black_box(&reduced))))
    });
    group.finish();
}

fn structured_execution_benchmark(criterion: &mut Criterion) {
    benchmark_workload(criterion, nodes(), "structured_execution_10000_nodes");
    benchmark_workload(
        criterion,
        capability_nodes(),
        "structured_execution_10000_nodes_capabilities",
    );
}

criterion_group!(benches, structured_execution_benchmark);
criterion_main!(benches);
