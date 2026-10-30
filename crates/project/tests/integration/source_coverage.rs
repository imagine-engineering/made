use std::{path::Path, sync::Arc};

use project::{
    ProjectPath,
    rust_coverage_provider::{MAX_RUST_COVERAGE_ARTIFACT_BYTES, RustCoverageArtifactProvider},
    source_coverage::{
        SourceCoverageFile, SourceCoveragePoint, SourceCoverageProviderId, SourceCoverageRange,
        SourceCoverageSnapshot, SourceCoverageState, SourceCoverageStatus, snapshot_from_proto,
        snapshot_to_proto,
    },
};
use settings::WorktreeId;
use util::rel_path::RelPath;

fn project_path(path: &str, worktree: usize) -> ProjectPath {
    ProjectPath {
        worktree_id: WorktreeId::from_usize(worktree),
        path: Arc::from(RelPath::from_unix_str(path).expect("fixture path should be relative")),
    }
}

#[test]
fn source_coverage_acceptance_covers_rust_non_rust_privacy_and_failures() {
    let mut rust_provider = RustCoverageArtifactProvider::new(11);
    let token = rust_provider.begin_run();
    let rust_report = include_bytes!("../../test_data/source_coverage/rust-report.json");
    let rust_snapshot = rust_provider
        .parse_artifact(token, rust_report, |absolute_path| {
            let relative = absolute_path.strip_prefix(Path::new("/workspace")).ok()?;
            Some(project_path(relative.to_str()?, 1))
        })
        .expect("supported cargo-llvm-cov export should parse");
    assert_eq!(rust_snapshot.files.len(), 1);
    assert_eq!(rust_snapshot.files[0].path, project_path("src/lib.rs", 1));
    assert_eq!(rust_snapshot.status, SourceCoverageStatus::Partial);

    let non_rust = SourceCoverageSnapshot {
        project_generation: 11,
        provider_id: SourceCoverageProviderId("fake-typescript".to_string()),
        generation: 1,
        status: SourceCoverageStatus::Current,
        files: vec![
            SourceCoverageFile {
                path: project_path("src/app.ts", 1),
                ranges: vec![SourceCoverageRange {
                    start: SourceCoveragePoint { line: 0, column: 0 },
                    end: SourceCoveragePoint { line: 0, column: 8 },
                    hit_count: 2,
                }],
                covered_lines: 1,
                uncovered_lines: 0,
                truncated: false,
            },
            SourceCoverageFile {
                path: project_path("hidden/secret.ts", 2),
                ranges: Vec::new(),
                covered_lines: 0,
                uncovered_lines: 1,
                truncated: false,
            },
        ],
        truncated: false,
        diagnostic: None,
    };
    let mut state = SourceCoverageState::new(11);
    state
        .publish(non_rust.clone())
        .expect("language-neutral provider should publish");
    let visible = snapshot_to_proto(
        &non_rust,
        Some(&[WorktreeId::from_usize(1)].into_iter().collect()),
    );
    assert_eq!(visible.files.len(), 1);
    assert!(visible.truncated);
    let round_trip = snapshot_from_proto(visible).expect("bounded remote facts should decode");
    assert_eq!(round_trip.files[0].path, project_path("src/app.ts", 1));
    assert!(!format!("{round_trip:?}").contains("/workspace"));

    rust_provider.cancel(token);
    assert!(
        rust_provider
            .parse_artifact(token, rust_report, |_| None)
            .is_err()
    );
    let next_token = rust_provider.begin_run();
    assert!(
        rust_provider
            .parse_artifact(
                next_token,
                &vec![b' '; MAX_RUST_COVERAGE_ARTIFACT_BYTES + 1],
                |_| None,
            )
            .is_err()
    );
}

#[test]
fn source_coverage_acceptance_marks_incomplete_file_facts_partial() {
    let mut provider = RustCoverageArtifactProvider::new(12);
    let token = provider.begin_run();
    let artifact = br#"{
        "type":"llvm.coverage.json.export",
        "version":"2.0.1",
        "cargo_llvm_cov":{"version":"0.9.0"},
        "data":[{"files":[
            {"filename":"/workspace/src/missing_segments.rs","summary":{"lines":{"count":1,"covered":1}}},
            {"filename":"/workspace/src/missing_totals.rs","segments":[[1,1,1,true,true,false]]},
            {"filename":"/workspace/src/inconsistent_totals.rs","segments":[[1,1,1,true,true,false]],"summary":{"lines":{"count":1,"covered":2}}}
        ]}]
    }"#;
    let snapshot = provider
        .parse_artifact(token, artifact, |absolute_path| {
            let relative = absolute_path.strip_prefix(Path::new("/workspace")).ok()?;
            Some(project_path(relative.to_str()?, 1))
        })
        .expect("bounded partial report should parse");

    assert_eq!(snapshot.status, SourceCoverageStatus::Partial);
    assert_eq!(snapshot.files.len(), 3);
    let missing_segments = snapshot
        .files
        .iter()
        .find(|file| file.path == project_path("src/missing_segments.rs", 1))
        .expect("missing-segments file should remain visible");
    assert_eq!(missing_segments.covered_lines, 1);
    assert!(missing_segments.ranges.is_empty());
    let missing_totals = snapshot
        .files
        .iter()
        .find(|file| file.path == project_path("src/missing_totals.rs", 1))
        .expect("missing-totals file should remain visible");
    assert_eq!(missing_totals.covered_lines, 0);
    assert_eq!(missing_totals.ranges.len(), 1);
    let inconsistent_totals = snapshot
        .files
        .iter()
        .find(|file| file.path == project_path("src/inconsistent_totals.rs", 1))
        .expect("inconsistent-totals file should remain visible");
    assert_eq!(inconsistent_totals.covered_lines, 1);
    assert_eq!(inconsistent_totals.uncovered_lines, 0);
    assert_eq!(
        snapshot.diagnostic.as_deref(),
        Some("Coverage report contains 3 malformed or non-visible issue(s)")
    );
}

#[test]
fn source_coverage_acceptance_parses_collector_090_llvm3_report_forms() {
    let reports: [(&str, &[u8], &[(&str, u32, u32)]); 4] = [
        (
            "run",
            include_bytes!("../../test_data/source_coverage/llvm3-run.json"),
            &[("src/bin/fixture.rs", 3, 3)],
        ),
        (
            "unit",
            include_bytes!("../../test_data/source_coverage/llvm3-unit.json"),
            &[("src/lib.rs", 14, 10)],
        ),
        (
            "integration",
            include_bytes!("../../test_data/source_coverage/llvm3-integration.json"),
            &[("src/bin/fixture.rs", 3, 0), ("src/lib.rs", 3, 3)],
        ),
        (
            "ignored",
            include_bytes!("../../test_data/source_coverage/llvm3-ignored.json"),
            &[("src/lib.rs", 14, 6)],
        ),
    ];
    let mut provider = RustCoverageArtifactProvider::new(13);

    for (report_form, artifact, expected_files) in reports {
        let token = provider.begin_run();
        let snapshot = provider
            .parse_artifact(token, artifact, |absolute_path| {
                let relative = absolute_path.strip_prefix(Path::new("/workspace")).ok()?;
                Some(project_path(relative.to_str()?, 1))
            })
            .expect("generated LLVM 3 report should parse");
        assert_eq!(
            snapshot.status,
            SourceCoverageStatus::Current,
            "{report_form}"
        );
        assert!(!snapshot.truncated, "{report_form}");
        assert_eq!(snapshot.files.len(), expected_files.len(), "{report_form}");
        let wire = snapshot_to_proto(&snapshot, None);
        let wire_debug = format!("{wire:?}");
        assert!(!wire_debug.contains("/workspace"), "{report_form}");
        assert!(!wire_debug.contains("/Users"), "{report_form}");
        let round_trip = snapshot_from_proto(wire).expect("bounded facts should decode");
        assert_eq!(
            round_trip.files.len(),
            expected_files.len(),
            "{report_form}"
        );
        for &(path, total, covered) in expected_files {
            let file = snapshot
                .files
                .iter()
                .find(|file| file.path == project_path(path, 1))
                .expect("mapped source should be present");
            assert_eq!(
                file.covered_lines + file.uncovered_lines,
                total,
                "{report_form}"
            );
            assert_eq!(file.covered_lines, covered, "{report_form}");
            assert!(!file.ranges.is_empty(), "{report_form}");
        }
    }
}
