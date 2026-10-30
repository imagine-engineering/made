use std::{
    hint::black_box,
    mem::size_of,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{Criterion, criterion_group, criterion_main};
use project::{
    ProjectPath,
    cargo_workspace::{CargoWorkspaceModel, parse_metadata, workspace_from_metadata},
};
use serde_json::{Value, json};
use settings::WorktreeId;
use util::{paths::PathStyle, rel_path::RelPath};

const PACKAGE_COUNT: usize = 1_000;
const CONVERSION_BUDGET: Duration = Duration::from_secs(2);
const RETAINED_MODEL_BUDGET_BYTES: usize = 32 * 1024 * 1024;

fn large_metadata_bytes() -> Vec<u8> {
    let comprehensive: Value = serde_json::from_slice(include_bytes!(
        "../test_data/cargo_workspace/comprehensive-v1.json"
    ))
    .expect("comprehensive fixture should be valid JSON");
    let template = comprehensive["roots"][1]["metadata"]["packages"][0].clone();
    let mut packages = Vec::with_capacity(PACKAGE_COUNT);
    let mut members = Vec::with_capacity(PACKAGE_COUNT);
    let mut nodes = Vec::with_capacity(PACKAGE_COUNT);
    for index in 0..PACKAGE_COUNT {
        let name = format!("package-{index:04}");
        let id = format!("path+file:///fixture/large/{name}#1.0.0");
        let manifest_path = format!("/fixture/large/{name}/Cargo.toml");
        let source_path = format!("/fixture/large/{name}/src/main.rs");
        let mut package = template.clone();
        package["name"] = json!(name);
        package["id"] = json!(id);
        package["manifest_path"] = json!(manifest_path);
        package["targets"][0]["name"] = package["name"].clone();
        package["targets"][0]["src_path"] = json!(source_path);
        packages.push(package);
        members.push(json!(id));
        nodes.push(json!({"id": id, "dependencies": [], "deps": [], "features": []}));
    }
    serde_json::to_vec(&json!({
        "packages": packages,
        "workspace_members": members,
        "workspace_default_members": ["path+file:///fixture/large/package-0000#1.0.0"],
        "resolve": {"root": null, "nodes": nodes},
        "target_directory": "/fixture/large/target",
        "version": 1,
        "workspace_root": "/fixture/large"
    }))
    .expect("large metadata fixture should serialize")
}

fn convert(bytes: &[u8]) -> CargoWorkspaceModel {
    let metadata = parse_metadata(bytes).expect("large metadata should parse");
    workspace_from_metadata(&metadata, |path| {
        let relative = path.strip_prefix(Path::new("/fixture/large")).ok()?;
        Some(ProjectPath {
            worktree_id: WorktreeId::from_proto(1),
            path: Arc::from(RelPath::new(relative, PathStyle::Unix).ok()?.as_ref()),
        })
    })
    .expect("large metadata should convert")
}

fn retained_model_bytes(model: &CargoWorkspaceModel) -> usize {
    let mut bytes = size_of::<CargoWorkspaceModel>()
        + model.display_name.capacity()
        + model.members.capacity() * size_of::<project::cargo_workspace::CargoPackageModel>();
    for member in &model.members {
        bytes += member.id.capacity()
            + member.name.capacity()
            + member.version.capacity()
            + member.manifest_path.path.as_unix_str().len()
            + member.targets.capacity() * size_of::<project::cargo_workspace::CargoTargetModel>()
            + member.features.capacity() * size_of::<project::cargo_workspace::CargoFeatureModel>()
            + member.dependencies.capacity()
                * size_of::<project::cargo_workspace::CargoDependencyModel>();
        for target in &member.targets {
            bytes += target.name.capacity()
                + target.edition.capacity()
                + target
                    .source_path
                    .as_ref()
                    .map_or(0, |path| path.path.as_unix_str().len())
                + target
                    .source_display_path
                    .as_ref()
                    .map_or(0, String::capacity)
                + target
                    .crate_types
                    .iter()
                    .map(String::capacity)
                    .sum::<usize>()
                + target
                    .required_features
                    .iter()
                    .map(String::capacity)
                    .sum::<usize>();
        }
        for feature in &member.features {
            bytes += feature.name.capacity()
                + feature.expands.iter().map(String::capacity).sum::<usize>();
        }
        for dependency in &member.dependencies {
            bytes += dependency.declaration_name.capacity()
                + dependency.rename.as_ref().map_or(0, String::capacity)
                + dependency.version_requirement.capacity()
                + dependency
                    .requested_features
                    .iter()
                    .map(String::capacity)
                    .sum::<usize>()
                + dependency.target.as_ref().map_or(0, String::capacity)
                + dependency
                    .resolved_name
                    .as_ref()
                    .map_or(0, String::capacity)
                + dependency
                    .resolved_version
                    .as_ref()
                    .map_or(0, String::capacity);
        }
    }
    bytes
}

fn cargo_workspace_benchmark(criterion: &mut Criterion) {
    let bytes = large_metadata_bytes();
    let started = Instant::now();
    let model = convert(&bytes);
    let elapsed = started.elapsed();
    let retained_bytes = retained_model_bytes(&model);
    assert_eq!(model.members.len(), PACKAGE_COUNT);
    assert!(
        elapsed <= CONVERSION_BUDGET,
        "1,000-package conversion took {elapsed:?}, exceeding {CONVERSION_BUDGET:?}"
    );
    assert!(
        retained_bytes <= RETAINED_MODEL_BUDGET_BYTES,
        "1,000-package model retained {retained_bytes} bytes, exceeding {RETAINED_MODEL_BUDGET_BYTES}"
    );
    eprintln!(
        "cargo-workspace-budget packages={PACKAGE_COUNT} elapsed_ms={} retained_model_bytes={retained_bytes}",
        elapsed.as_millis()
    );

    let mut group = criterion.benchmark_group("cargo_workspace_1000_packages");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("parse_and_convert", |bencher| {
        bencher.iter(|| black_box(convert(black_box(&bytes))))
    });
    group.finish();
}

criterion_group!(benches, cargo_workspace_benchmark);
criterion_main!(benches);
