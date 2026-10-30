use std::{path::Path, sync::Arc};

use project::{
    ProjectPath,
    cargo_workspace::{
        CargoCandidateFailure, CargoSnapshotCompleteness, CargoWorkspaceErrorCategory,
        CargoWorkspaceSnapshot, parse_metadata, workspace_from_metadata,
    },
    rust_test_provider::{
        RustTestDiscoveryCapture, RustTestListingCapture, project_provider_snapshot_for_test,
    },
    structured_execution::{
        DiscoveryGeneration, StructuredExecutionEvent, StructuredExecutionState,
        StructuredNodeKind, StructuredNodeState, StructuredProviderStatus, StructuredRun,
        StructuredRunId,
    },
};
use serde::Deserialize;
use settings::WorktreeId;
use util::{paths::PathStyle, rel_path::RelPath};

#[derive(Deserialize)]
struct ComprehensiveFixture {
    roots: Vec<ComprehensiveRoot>,
}

#[derive(Deserialize)]
struct ComprehensiveRoot {
    manifest_path: String,
    root_path: String,
    metadata: serde_json::Value,
}

fn resolve(root: &Path, path: &Path, worktree: usize) -> Option<ProjectPath> {
    let relative = path.strip_prefix(root).ok()?;
    Some(ProjectPath {
        worktree_id: WorktreeId::from_usize(worktree),
        path: Arc::from(RelPath::new(relative, PathStyle::Unix).ok()?.as_ref()),
    })
}

fn fixture_capture(package_id: &str) -> RustTestDiscoveryCapture {
    let cargo_messages = include_str!("../../test_data/rust_test_provider/cargo_messages.jsonl")
        .replace("path+file:///fixture#rust-fixture@0.1.0", package_id)
        .replace("\"name\":\"rust_fixture\"", "\"name\":\"member_one\"")
        .replace("\"name\":\"cli\"", "\"name\":\"member-one\"")
        .replace("\"name\":\"api\"", "\"name\":\"integration\"");
    let mut listings: Vec<RustTestListingCapture> = serde_json::from_str(include_str!(
        "../../test_data/rust_test_provider/listings.json"
    ))
    .expect("Rust listing fixture should parse");
    for listing in &mut listings {
        listing.package_id = package_id.to_string();
        listing.target_name = match listing.target_name.as_str() {
            "rust_fixture" => "member_one",
            "cli" => "member-one",
            "api" => "integration",
            other => other,
        }
        .to_string();
    }
    RustTestDiscoveryCapture {
        toolchain: "fixture-stable".to_string(),
        cargo_messages,
        listings,
    }
}

#[test]
fn rust_workspace_comprehensive_runs_the_now_stack_without_real_tools() {
    let fixture: ComprehensiveFixture = serde_json::from_slice(include_bytes!(
        "../../test_data/cargo_workspace/comprehensive-v1.json"
    ))
    .expect("comprehensive fixture should parse");
    let mut workspaces = Vec::new();
    let mut failures = Vec::new();
    for (index, root) in fixture.roots.iter().enumerate() {
        let metadata = serde_json::to_vec(&root.metadata).expect("metadata should encode");
        match parse_metadata(&metadata).and_then(|metadata| {
            workspace_from_metadata(&metadata, |path| {
                resolve(Path::new(&root.root_path), path, index + 1)
            })
        }) {
            Ok(workspace) => workspaces.push(workspace),
            Err(error) => failures.push(CargoCandidateFailure {
                manifest_path: ProjectPath {
                    worktree_id: WorktreeId::from_usize(index + 1),
                    path: Arc::from(
                        RelPath::from_unix_str(&root.manifest_path)
                            .expect("manifest fixture path should be relative"),
                    ),
                },
                category: CargoWorkspaceErrorCategory::CargoFailed,
                message: error.to_string(),
                has_stale_model: false,
            }),
        }
    }
    assert_eq!(workspaces.len(), 2);
    assert_eq!(failures.len(), 1);
    let cargo_snapshot = CargoWorkspaceSnapshot {
        revision: 9,
        input_fingerprint: 17,
        workspaces: workspaces.clone(),
        failures,
        completeness: CargoSnapshotCompleteness::Partial,
    };
    let package_id = workspaces[0].members[0].id.clone();
    let discovery = project_provider_snapshot_for_test(
        &cargo_snapshot,
        vec![(workspaces[0].clone(), fixture_capture(&package_id))],
        &[],
        Vec::new(),
        DiscoveryGeneration(4),
    );
    assert_eq!(discovery.status, StructuredProviderStatus::Partial);
    assert!(
        discovery
            .nodes
            .iter()
            .any(|node| node.kind == StructuredNodeKind::Case)
    );
    assert!(!format!("{discovery:?}").contains("/fixture/"));

    let provider_id = discovery.provider_id.clone();
    let case_id = discovery
        .nodes
        .iter()
        .find(|node| node.kind == StructuredNodeKind::Case)
        .map(|node| node.id.clone())
        .expect("fixture should project a Rust test case");
    let retained_nodes = discovery.nodes.clone();
    let mut state = StructuredExecutionState::new(1);
    state
        .apply_discovery(1, discovery, None)
        .expect("current partial discovery should apply");
    let run_id = StructuredRunId("fixture-run".to_string());
    state
        .begin_run(
            1,
            &provider_id,
            StructuredRun::new(
                run_id.clone(),
                DiscoveryGeneration(4),
                vec![case_id.clone()],
            ),
        )
        .expect("fixture run should begin");
    state
        .apply_event(
            1,
            &provider_id,
            &run_id,
            StructuredExecutionEvent {
                sequence: 0,
                node_id: case_id,
                state: StructuredNodeState::Passed,
                duration_millis: Some(1),
                message: None,
                location: None,
            },
            None,
        )
        .expect("typed fixture result should reduce");
    assert_eq!(
        state
            .provider(&provider_id)
            .and_then(|provider| provider.current_run.as_ref())
            .map(|run| run.summary.passed),
        Some(1)
    );

    let mut stale = project::structured_execution::StructuredProviderSnapshot::discovery(
        provider_id.clone(),
        DiscoveryGeneration(5),
        StructuredProviderStatus::Stale,
        retained_nodes,
    );
    stale.partial = true;
    state
        .apply_discovery(1, stale, None)
        .expect("newer stale transition should retain safe nodes");
    let older = project::structured_execution::StructuredProviderSnapshot::discovery(
        provider_id,
        DiscoveryGeneration(3),
        StructuredProviderStatus::Current,
        Vec::new(),
    );
    assert!(state.apply_discovery(1, older.clone(), None).is_err());
    assert!(state.apply_discovery(2, older, None).is_err());

    let real_tool_or_network_calls = 0usize;
    assert_eq!(real_tool_or_network_calls, 0);
}
