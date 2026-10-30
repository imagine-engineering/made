use std::{path::Path, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use fs::FakeFs;
use gpui::{AppContext as _, TestAppContext};
use project::{
    Project, ProjectPath,
    cargo_workspace::{
        CargoConfigurationCompleteness, CargoConfigurationDiagnosticCategory,
        CargoDependencyDeclarationOrigin, CargoDependencyKind, CargoDependencyLockStatus,
        CargoDependencySourceKind, CargoFeatureEnabled, CargoHostCompilerModel,
        CargoHostCompilerStatus, CargoProfileModel, CargoProfileOrigin, CargoTargetKind,
        CargoToolchainFormat, CargoWorkspaceConfiguration, enrich_dependency_provenance,
        parse_cargo_profiles, parse_metadata, parse_rust_toolchain, parse_rustc_verbose_version,
        workspace_from_metadata,
    },
    cargo_workspace_store::{
        CargoConfigurationProbe, CargoConfigurationProbeRequest, CargoMetadataRequest,
        CargoMetadataRunner, CargoWorkspaceStore,
    },
    trusted_worktrees::{self, DbTrustedPaths, TrustedWorktrees},
};
use serde::Deserialize;
use serde_json::json;
use settings::WorktreeId;
use util::{paths::PathStyle, rel_path::RelPath};

fn resolve(root: &Path, path: &Path) -> Option<ProjectPath> {
    let relative = path.strip_prefix(root).ok()?;
    Some(ProjectPath {
        worktree_id: WorktreeId::from_usize(1),
        path: Arc::from(RelPath::new(relative, PathStyle::Unix).ok()?.as_ref()),
    })
}

fn fixture_path(path: &str) -> ProjectPath {
    ProjectPath {
        worktree_id: WorktreeId::from_usize(1),
        path: Arc::from(RelPath::from_unix_str(path).expect("fixture path should be valid")),
    }
}

#[derive(Deserialize)]
struct ComprehensiveFixture {
    schema_version: u32,
    roots: Vec<ComprehensiveRoot>,
    profile_manifests: Vec<ComprehensiveTextFixture>,
    toolchain_declarations: Vec<ComprehensiveTextFixture>,
}

#[derive(Deserialize)]
struct ComprehensiveRoot {
    manifest_path: String,
    root_path: String,
    metadata: serde_json::Value,
}

#[derive(Deserialize)]
struct ComprehensiveTextFixture {
    path: String,
    contents: String,
}

#[derive(Deserialize)]
struct DependencyProvenanceFixture {
    schema_version: u32,
    metadata_file: String,
    manifests: Vec<ComprehensiveTextFixture>,
    lock_contents: String,
}

#[test]
fn cargo_dependency_provenance_acceptance_is_bounded_and_private() {
    let fixture: DependencyProvenanceFixture = serde_json::from_slice(include_bytes!(
        "../../test_data/cargo_workspace/dependency-provenance-v1.json"
    ))
    .expect("dependency provenance fixture should be valid JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.metadata_file, "workspace-v1.json");
    let metadata = parse_metadata(include_bytes!(
        "../../test_data/cargo_workspace/workspace-v1.json"
    ))
    .expect("metadata fixture should parse");
    let mut workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("metadata fixture should convert");
    let manifests = fixture
        .manifests
        .into_iter()
        .map(|manifest| (fixture_path(&manifest.path), manifest.contents))
        .collect();
    enrich_dependency_provenance(&mut workspace, &manifests, Some(&fixture.lock_contents));

    let primary = workspace
        .members
        .iter()
        .find(|package| package.name == "member-one")
        .expect("primary fixture package should exist");
    let inherited = primary
        .dependencies
        .iter()
        .find(|dependency| dependency.rename.as_deref() == Some("renamed_member"))
        .expect("renamed dependency should exist");
    assert_eq!(
        inherited.declaration_origin,
        CargoDependencyDeclarationOrigin::WorkspaceInherited
    );
    assert_eq!(inherited.resolved_instances.len(), 1);
    assert_eq!(
        inherited.resolved_instances[0].lock_status,
        CargoDependencyLockStatus::Locked
    );
    let git = primary
        .dependencies
        .iter()
        .find(|dependency| dependency.source_kind == CargoDependencySourceKind::Git)
        .expect("Git dependency should exist");
    assert!(git.local_manifest.is_none());
    assert!(format!("{workspace:?}").find("/Users/").is_none());

    enrich_dependency_provenance(&mut workspace, &manifests, None);
    assert_eq!(
        workspace
            .members
            .iter()
            .find(|package| package.name == "member-one")
            .and_then(|package| package
                .dependencies
                .iter()
                .find(|dependency| { dependency.rename.as_deref() == Some("renamed_member") }))
            .and_then(|dependency| dependency.resolved_instances.first())
            .map(|resolved| resolved.lock_status),
        Some(CargoDependencyLockStatus::MissingLockfile)
    );
}

#[test]
fn cargo_workspace_comprehensive_fixture() {
    let fixture: ComprehensiveFixture = serde_json::from_slice(include_bytes!(
        "../../test_data/cargo_workspace/comprehensive-v1.json"
    ))
    .expect("comprehensive fixture should be valid JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.roots.len(), 3);

    let mut converted = Vec::new();
    let mut failures = Vec::new();
    for root in &fixture.roots {
        let encoded = serde_json::to_vec(&root.metadata).expect("metadata should serialize");
        match parse_metadata(&encoded).and_then(|metadata| {
            workspace_from_metadata(&metadata, |path| resolve(Path::new(&root.root_path), path))
        }) {
            Ok(workspace) => converted.push((root, workspace)),
            Err(error) => failures.push((root.manifest_path.as_str(), error.to_string())),
        }
    }

    assert_eq!(converted.len(), 2);
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].0, "broken/Cargo.toml");
    assert!(failures[0].1.contains("format version"));

    let virtual_workspace = &converted[0].1;
    assert!(virtual_workspace.is_virtual);
    let primary = virtual_workspace
        .members
        .iter()
        .find(|member| member.name == "duplicate-name")
        .expect("virtual member should exist");
    let target_kinds = primary
        .targets
        .iter()
        .map(|target| &target.kind)
        .collect::<Vec<_>>();
    assert!(target_kinds.contains(&&CargoTargetKind::Library));
    assert!(target_kinds.contains(&&CargoTargetKind::Binary));
    assert!(target_kinds.contains(&&CargoTargetKind::Example));
    assert!(target_kinds.contains(&&CargoTargetKind::Test));
    assert!(target_kinds.contains(&&CargoTargetKind::Bench));
    assert!(target_kinds.contains(&&CargoTargetKind::BuildScript));
    assert!(
        target_kinds
            .iter()
            .any(|kind| matches!(kind, CargoTargetKind::Other(_)))
    );
    assert!(primary.dependencies.iter().any(|dependency| {
        dependency.kind == CargoDependencyKind::Normal
            && dependency.rename.as_deref() == Some("renamed_member")
            && dependency.optional
            && dependency.target.as_deref() == Some("cfg(unix)")
            && dependency.source_kind == CargoDependencySourceKind::Path
    }));
    assert!(primary.dependencies.iter().any(|dependency| {
        dependency.kind == CargoDependencyKind::Development
            && dependency.source_kind == CargoDependencySourceKind::Registry
    }));
    assert!(primary.dependencies.iter().any(|dependency| {
        dependency.kind == CargoDependencyKind::Build
            && dependency.source_kind == CargoDependencySourceKind::Registry
    }));
    assert!(
        primary
            .dependencies
            .iter()
            .any(|dependency| { dependency.source_kind == CargoDependencySourceKind::Git })
    );

    let standalone = &converted[1].1;
    assert!(!standalone.is_virtual);
    assert_eq!(standalone.members[0].name, primary.name);

    let parsed_profiles = fixture
        .profile_manifests
        .iter()
        .map(|entry| (entry.path.as_str(), parse_cargo_profiles(&entry.contents)))
        .collect::<Vec<_>>();
    assert!(parsed_profiles[0].1.is_ok());
    assert!(parsed_profiles[1].1.is_err());

    for declaration in &fixture.toolchain_declarations {
        let toolchain =
            parse_rust_toolchain(fixture_path(&declaration.path), &declaration.contents)
                .expect("toolchain declaration should parse");
        assert!(toolchain.channel.is_some());
    }
}

#[test]
fn cargo_workspace_fixture_covers_model_kinds_and_stable_order() {
    let metadata = parse_metadata(include_bytes!(
        "../../test_data/cargo_workspace/workspace-v1.json"
    ))
    .expect("workspace fixture should parse");
    let model = workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
        .expect("workspace fixture should convert");

    assert!(model.is_virtual);
    assert_eq!(model.members.len(), 2);
    let member = model
        .members
        .iter()
        .find(|member| member.name == "member-one")
        .expect("member-one should be present");
    assert!(!member.id.contains("/workspace"));
    assert!(member.id.contains("member-one/Cargo.toml"));
    assert!(member.is_default_member);
    assert_eq!(member.targets.len(), 7);
    assert!(
        member
            .targets
            .iter()
            .any(|target| target.kind == CargoTargetKind::Library)
    );
    assert!(
        member
            .targets
            .iter()
            .any(|target| target.kind == CargoTargetKind::Binary)
    );
    assert!(
        member
            .targets
            .iter()
            .any(|target| target.kind == CargoTargetKind::Example)
    );
    assert!(
        member
            .targets
            .iter()
            .any(|target| target.kind == CargoTargetKind::Test)
    );
    assert!(
        member
            .targets
            .iter()
            .any(|target| target.kind == CargoTargetKind::Bench)
    );
    assert!(
        member
            .targets
            .iter()
            .any(|target| target.kind == CargoTargetKind::BuildScript)
    );
    assert!(
        member
            .targets
            .iter()
            .any(|target| matches!(target.kind, CargoTargetKind::Other(_)))
    );

    assert_eq!(
        member
            .features
            .iter()
            .find(|feature| feature.name == "extra")
            .map(|feature| feature.enabled),
        Some(CargoFeatureEnabled::Enabled)
    );
    assert_eq!(
        member
            .features
            .iter()
            .find(|feature| feature.name == "implicit")
            .map(|feature| feature.defined),
        Some(false)
    );
    assert!(member.dependencies.iter().any(|dependency| {
        dependency.kind == CargoDependencyKind::Normal
            && dependency.rename.as_deref() == Some("renamed_member")
            && dependency.source_kind == CargoDependencySourceKind::Path
            && dependency.resolved_version.as_deref() == Some("0.2.0")
            && dependency.resolved_workspace_member.is_some()
    }));
    assert!(
        member
            .dependencies
            .iter()
            .any(|dependency| dependency.kind == CargoDependencyKind::Development)
    );
    assert!(
        member
            .dependencies
            .iter()
            .any(|dependency| dependency.kind == CargoDependencyKind::Build)
    );
    assert!(
        member
            .dependencies
            .iter()
            .any(|dependency| dependency.source_kind == CargoDependencySourceKind::Git)
    );
    assert!(
        member
            .dependencies
            .iter()
            .any(|dependency| dependency.source_kind == CargoDependencySourceKind::Registry)
    );

    let second = workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
        .expect("a second conversion should succeed");
    assert_eq!(model, second);
}

#[test]
fn standalone_fixture_is_a_non_virtual_single_member_workspace() {
    let metadata = parse_metadata(include_bytes!(
        "../../test_data/cargo_workspace/standalone-v1.json"
    ))
    .expect("standalone fixture should parse");
    let model = workspace_from_metadata(&metadata, |path| resolve(Path::new("/standalone"), path))
        .expect("standalone fixture should convert");
    assert!(!model.is_virtual);
    assert_eq!(model.members.len(), 1);
    assert_eq!(model.members[0].name, "standalone");
}

#[test]
fn metadata_parser_rejects_unsupported_and_malformed_documents() {
    assert!(parse_metadata(br#"{"version":2}"#).is_err());
    assert!(parse_metadata(br#"not-json"#).is_err());
}

#[test]
fn cargo_workspace_configuration_parsers_are_bounded_and_fallible() {
    let profiles = parse_cargo_profiles(include_str!(
        "../../test_data/cargo_workspace/profiles-custom.toml"
    ))
    .expect("custom profile fixture should parse");
    assert_eq!(
        profiles,
        vec![
            CargoProfileModel {
                name: "dev".to_string(),
                origin: CargoProfileOrigin::Declared,
            },
            CargoProfileModel {
                name: "release".to_string(),
                origin: CargoProfileOrigin::Implicit,
            },
            CargoProfileModel {
                name: "ci".to_string(),
                origin: CargoProfileOrigin::Declared,
            },
            CargoProfileModel {
                name: "ship".to_string(),
                origin: CargoProfileOrigin::Declared,
            },
        ]
    );
    assert!(
        parse_cargo_profiles(include_str!(
            "../../test_data/cargo_workspace/profiles-malformed.toml"
        ))
        .is_err()
    );

    let declared = parse_rust_toolchain(
        fixture_path("rust-toolchain.toml"),
        include_str!("../../test_data/cargo_workspace/rust-toolchain.toml"),
    )
    .expect("TOML toolchain fixture should parse");
    assert_eq!(declared.format, CargoToolchainFormat::Toml);
    assert_eq!(declared.channel.as_deref(), Some("stable"));
    assert_eq!(declared.components, ["rustfmt", "clippy"]);
    assert_eq!(declared.targets, ["wasm32-unknown-unknown"]);
    let legacy = parse_rust_toolchain(
        fixture_path("rust-toolchain"),
        include_str!("../../test_data/cargo_workspace/rust-toolchain-legacy"),
    )
    .expect("legacy toolchain fixture should parse");
    assert_eq!(legacy.format, CargoToolchainFormat::Legacy);
    assert_eq!(legacy.channel.as_deref(), Some("1.90.0"));

    let compiler = parse_rustc_verbose_version(include_bytes!(
        "../../test_data/cargo_workspace/rustc-vv.txt"
    ))
    .expect("rustc fixture should parse");
    assert_eq!(compiler.status, CargoHostCompilerStatus::Available);
    assert_eq!(compiler.release.as_deref(), Some("1.90.0"));
    assert_eq!(
        compiler.host_target.as_deref(),
        Some("x86_64-unknown-linux-gnu")
    );
    assert!(parse_rustc_verbose_version(b"rustc 1.90.0\n").is_err());
}

#[test]
fn cargo_workspace_configuration_retains_only_safe_stale_facts() {
    let mut previous = CargoWorkspaceConfiguration::unresolved();
    previous.host_compiler = CargoHostCompilerModel {
        status: CargoHostCompilerStatus::Available,
        release: Some("1.90.0".to_string()),
        host_target: Some("x86_64-unknown-linux-gnu".to_string()),
        stale: false,
    };
    previous.declared_toolchain = Some(
        parse_rust_toolchain(
            fixture_path("rust-toolchain.toml"),
            include_str!("../../test_data/cargo_workspace/rust-toolchain.toml"),
        )
        .expect("toolchain fixture should parse"),
    );
    let mut current = CargoWorkspaceConfiguration::unresolved();
    current.host_compiler.status = CargoHostCompilerStatus::Failed;
    current.add_diagnostic(
        None,
        CargoConfigurationDiagnosticCategory::CompilerProbe,
        "probe failed",
    );
    current.retain_stale_safe_facts(&previous);
    assert!(current.host_compiler.stale);
    assert_eq!(
        current.host_compiler.release,
        previous.host_compiler.release
    );
    assert_eq!(current.declared_toolchain, previous.declared_toolchain);
}

struct FixtureRunner {
    output: Vec<u8>,
    requests: parking_lot::Mutex<Vec<CargoMetadataRequest>>,
}

struct FixtureConfigurationProbe {
    output: Result<Vec<u8>, String>,
    requests: parking_lot::Mutex<Vec<CargoConfigurationProbeRequest>>,
}

#[async_trait]
impl CargoConfigurationProbe for FixtureConfigurationProbe {
    async fn run(&self, request: CargoConfigurationProbeRequest) -> Result<Vec<u8>> {
        self.requests.lock().push(request);
        self.output.clone().map_err(anyhow::Error::msg)
    }
}

#[async_trait]
impl CargoMetadataRunner for FixtureRunner {
    async fn run(&self, request: CargoMetadataRequest) -> Result<Vec<u8>> {
        self.requests.lock().push(request);
        Ok(self.output.clone())
    }
}

#[gpui::test]
async fn cargo_workspace_store_uses_an_injected_runner(cx: &mut TestAppContext) {
    crate::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        "/root",
        json!({
            "Cargo.toml": include_str!("../../test_data/cargo_workspace/profiles-custom.toml"),
            "rust-toolchain.toml": include_str!("../../test_data/cargo_workspace/rust-toolchain.toml"),
            ".cargo": { "config.toml": "[build]\ntarget = \"wasm32-unknown-unknown\"" }
        }),
    )
    .await;
    let project = Project::test(fs, [Path::new("/root")], cx).await;
    let output = String::from_utf8_lossy(include_bytes!(
        "../../test_data/cargo_workspace/workspace-v1.json"
    ))
    .replace("/workspace", "/root")
    .into_bytes();
    let runner = Arc::new(FixtureRunner {
        output,
        requests: parking_lot::Mutex::new(Vec::new()),
    });
    let configuration_probe = Arc::new(FixtureConfigurationProbe {
        output: Ok(include_bytes!("../../test_data/cargo_workspace/rustc-vv.txt").to_vec()),
        requests: parking_lot::Mutex::new(Vec::new()),
    });
    let (worktree_store, environment) = cx.update(|cx| {
        let project = project.read(cx);
        (project.worktree_store(), project.environment().clone())
    });
    let store = cx.new({
        let runner = runner.clone();
        let configuration_probe = configuration_probe.clone();
        move |_| {
            CargoWorkspaceStore::local_with_runners(
                worktree_store,
                environment,
                runner,
                configuration_probe,
            )
        }
    });
    let snapshot = store
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("fixture runner should produce a snapshot");
    assert_eq!(snapshot.workspaces.len(), 1);
    assert!(snapshot.failures.is_empty());
    let configuration = &snapshot.workspaces[0].configuration;
    assert_eq!(
        configuration.completeness,
        CargoConfigurationCompleteness::Complete
    );
    assert_eq!(configuration.profiles.len(), 4);
    assert_eq!(
        configuration
            .declared_toolchain
            .as_ref()
            .and_then(|toolchain| toolchain.channel.as_deref()),
        Some("stable")
    );
    assert_eq!(
        configuration.host_compiler.host_target.as_deref(),
        Some("x86_64-unknown-linux-gnu")
    );
    let requests = runner.requests.lock();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].manifest_path, Path::new("/root/Cargo.toml"));
    assert_eq!(requests[0].working_directory, Path::new("/root"));
    let configuration_requests = configuration_probe.requests.lock();
    assert_eq!(configuration_requests.len(), 1);
    assert_eq!(
        configuration_requests[0].working_directory,
        Path::new("/root")
    );
}

#[gpui::test]
async fn cargo_workspace_restricted_store_runs_no_injected_commands(cx: &mut TestAppContext) {
    crate::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/restricted", json!({ "Cargo.toml": "[workspace]" }))
        .await;
    let project = Project::test(fs, [Path::new("/restricted")], cx).await;
    let runner = Arc::new(FixtureRunner {
        output: include_bytes!("../../test_data/cargo_workspace/workspace-v1.json").to_vec(),
        requests: parking_lot::Mutex::new(Vec::new()),
    });
    let configuration_probe = Arc::new(FixtureConfigurationProbe {
        output: Ok(include_bytes!("../../test_data/cargo_workspace/rustc-vv.txt").to_vec()),
        requests: parking_lot::Mutex::new(Vec::new()),
    });
    let (worktree_store, environment) = cx.update(|cx| {
        let project = project.read(cx);
        (project.worktree_store(), project.environment().clone())
    });
    cx.update(|cx| {
        if cx.try_global::<TrustedWorktrees>().is_some() {
            cx.remove_global::<TrustedWorktrees>();
        }
        trusted_worktrees::init(DbTrustedPaths::default(), cx);
        trusted_worktrees::track_worktree_trust(worktree_store.clone(), None, None, None, cx);
    });
    let store = cx.new({
        let runner = runner.clone();
        let configuration_probe = configuration_probe.clone();
        move |_| {
            CargoWorkspaceStore::local_with_runners(
                worktree_store,
                environment,
                runner,
                configuration_probe,
            )
        }
    });
    let snapshot = store
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("restricted refresh should return a scoped failure");
    assert!(snapshot.workspaces.is_empty());
    assert_eq!(snapshot.failures.len(), 1);
    assert!(runner.requests.lock().is_empty());
    assert!(configuration_probe.requests.lock().is_empty());
}

#[gpui::test]
async fn doctor_cargo_cache_reads_without_execution_and_invalidates_cancelled_refresh(
    cx: &mut TestAppContext,
) {
    crate::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/root", json!({"Cargo.toml": "[workspace]"}))
        .await;
    let project = Project::test(fs.clone(), [Path::new("/root")], cx).await;
    let runner = Arc::new(FixtureRunner {
        output: String::from_utf8_lossy(include_bytes!(
            "../../test_data/cargo_workspace/workspace-v1.json"
        ))
        .replace("/workspace", "/root")
        .into_bytes(),
        requests: parking_lot::Mutex::new(Vec::new()),
    });
    let (worktree_store, environment) = project.read_with(cx, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    let store = cx.new({
        let runner = runner.clone();
        move |_| CargoWorkspaceStore::local_with_runner(worktree_store, environment, runner)
    });
    assert!(store.read_with(cx, |store, cx| store.cached_snapshot(cx).is_none()));
    assert!(runner.requests.lock().is_empty());
    let snapshot = store
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Reviewed fixture refresh completes");
    for _ in 0..3 {
        store.read_with(cx, |store, cx| {
            let (cached, current) = store
                .cached_snapshot(cx)
                .expect("Completed owner snapshot retained");
            assert_eq!(cached, &snapshot);
            assert!(current);
        });
    }
    assert_eq!(runner.requests.lock().len(), 1);
    let cancelled = store.update(cx, |store, cx| store.refresh(cx));
    drop(cancelled);
    cx.run_until_parked();
    store.read_with(cx, |store, cx| {
        let (cached, current) = store
            .cached_snapshot(cx)
            .expect("Cancellation retains historical evidence");
        assert_eq!(cached, &snapshot);
        assert!(!current);
    });
    assert_eq!(runner.requests.lock().len(), 1);
}

struct DoctorFailingRunner {
    successful: FixtureRunner,
    fail: std::sync::atomic::AtomicBool,
}

#[async_trait]
impl CargoMetadataRunner for DoctorFailingRunner {
    async fn run(&self, request: CargoMetadataRequest) -> Result<Vec<u8>> {
        if self.fail.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!("Reviewed fixture refresh failed")
        }
        self.successful.run(request).await
    }
}

#[gpui::test]
async fn doctor_cargo_cache_changed_inputs_and_failure_preserve_stale_evidence(
    cx: &mut TestAppContext,
) {
    crate::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/root", json!({"Cargo.toml": "[workspace]"}))
        .await;
    let project = Project::test(fs.clone(), [Path::new("/root")], cx).await;
    let runner = Arc::new(DoctorFailingRunner {
        successful: FixtureRunner {
            output: String::from_utf8_lossy(include_bytes!(
                "../../test_data/cargo_workspace/workspace-v1.json"
            ))
            .replace("/workspace", "/root")
            .into_bytes(),
            requests: parking_lot::Mutex::new(Vec::new()),
        },
        fail: std::sync::atomic::AtomicBool::new(false),
    });
    let (worktree_store, environment) = project.read_with(cx, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    let store = cx.new({
        let runner = runner.clone();
        move |_| CargoWorkspaceStore::local_with_runner(worktree_store, environment, runner)
    });
    let snapshot = store
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Fixture refresh");
    fs.insert_tree("/root", json!({"Cargo.lock": "version = 4"}))
        .await;
    cx.run_until_parked();
    store.read_with(cx, |store, cx| {
        let (cached, current) = store.cached_snapshot(cx).expect("Historical evidence");
        assert_eq!(cached, &snapshot);
        assert!(!current, "Changed lockfile invalidates metadata");
    });
    assert_eq!(runner.successful.requests.lock().len(), 1);
    runner
        .fail
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let failure = store
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Scoped failure snapshot");
    assert!(!failure.failures.is_empty());
    store.read_with(cx, |store, cx| {
        let (cached, current) = store
            .cached_snapshot(cx)
            .expect("Failure retains earlier evidence");
        assert_eq!(cached, &snapshot);
        assert!(!current);
    });
}

#[gpui::test]
async fn doctor_cargo_cache_preserves_first_failed_observation(cx: &mut TestAppContext) {
    crate::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree("/root", json!({"Cargo.toml": "[workspace]"}))
        .await;
    let project = Project::test(fs, [Path::new("/root")], cx).await;
    let runner = Arc::new(DoctorFailingRunner {
        successful: FixtureRunner {
            output: Vec::new(),
            requests: parking_lot::Mutex::new(Vec::new()),
        },
        fail: std::sync::atomic::AtomicBool::new(true),
    });
    let (worktree_store, environment) = project.read_with(cx, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    let store = cx.new({
        let runner = runner.clone();
        move |_| CargoWorkspaceStore::local_with_runner(worktree_store, environment, runner)
    });
    let failure = store
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("Scoped failure result");
    assert!(!failure.failures.is_empty());
    store.read_with(cx, |store, cx| {
        let (cached, current) = store
            .cached_snapshot(cx)
            .expect("First failed result remains inspectable");
        assert_eq!(cached, &failure);
        assert!(current, "Current failure is not a ready check");
    });
    assert!(runner.successful.requests.lock().is_empty());
}

fn doctor_cargo_graph_fixture() -> serde_json::Value {
    let mut metadata: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../test_data/cargo_workspace/workspace-v1.json"
    ))
    .expect("Cargo metadata fixture");
    let template = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .and_then(|packages| packages.first())
        .expect("Template")
        .clone();
    let public_source = "registry+https://github.com/rust-lang/crates.io-index";
    let private_source = "registry+https://PRIVATE_GRAPH_TOKEN@private.example/index";
    let definitions = [
        ("shared", "1.0.0", public_source, "public-old"),
        ("shared", "2.0.0", public_source, "public-new"),
        ("shared", "1.0.0", private_source, "private-shared"),
        ("transitive", "0.4.0", public_source, "transitive"),
    ];
    for (name, version, source, identifier) in definitions {
        let mut package = template.clone();
        let object = package.as_object_mut().expect("Package");
        object.insert("name".into(), json!(name));
        object.insert("version".into(), json!(version));
        object.insert("id".into(), json!(identifier));
        object.insert("source".into(), json!(source));
        object.insert(
            "manifest_path".into(),
            json!(format!("/cargo/PRIVATE_GRAPH_PATH/{identifier}/Cargo.toml")),
        );
        object.insert("dependencies".into(), json!([]));
        object.insert("targets".into(), json!([]));
        object.insert("features".into(), json!({"default": []}));
        metadata
            .get_mut("packages")
            .and_then(serde_json::Value::as_array_mut)
            .expect("Packages")
            .push(package);
        metadata
            .pointer_mut("/resolve/nodes")
            .and_then(serde_json::Value::as_array_mut)
            .expect("Nodes")
            .push(json!({"id":identifier,"dependencies":[],"deps":[],"features":["default"]}));
    }
    let nodes = metadata
        .pointer_mut("/resolve/nodes")
        .and_then(serde_json::Value::as_array_mut)
        .expect("Nodes");
    let root = nodes.first_mut().expect("Root");
    for (alias, identifier) in [
        ("shared_old", "public-old"),
        ("shared_new", "public-new"),
        ("private_alias", "private-shared"),
    ] {
        root.get_mut("deps")
            .and_then(serde_json::Value::as_array_mut)
            .expect("Edges")
            .push(json!({"name":alias,"pkg":identifier,"dep_kinds":[{"kind":null,"target":null}]}));
        root.get_mut("dependencies")
            .and_then(serde_json::Value::as_array_mut)
            .expect("Dependencies")
            .push(json!(identifier));
    }
    let old = nodes
        .iter_mut()
        .find(|node| node.get("id") == Some(&json!("public-old")))
        .expect("Transitive parent");
    old["deps"] = json!([{"name":"transitive_alias","pkg":"transitive","dep_kinds":[{"kind":"build","target":"cfg(unix)"}]}]);
    old["dependencies"] = json!(["transitive"]);
    metadata
}

#[test]
fn doctor_cargo_graph_retains_transitive_multiple_versions_sources_features_and_exact_lock() {
    use project::cargo_workspace::{
        CargoSnapshotCompleteness, CargoWorkspaceSnapshot, validate_cargo_snapshot_bounds,
    };
    let metadata =
        parse_metadata(&serde_json::to_vec(&doctor_cargo_graph_fixture()).expect("Metadata"))
            .expect("Parsed metadata");
    let mut workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Owner graph");
    let lock = "version = 4\n[[package]]\nname='shared'\nversion='1.0.0'\nsource='registry+https://github.com/rust-lang/crates.io-index'\n[[package]]\nname='shared'\nversion='2.0.0'\nsource='registry+https://github.com/rust-lang/crates.io-index'\n";
    enrich_dependency_provenance(&mut workspace, &Default::default(), Some(lock));
    let graph = workspace
        .dependency_graph
        .as_ref()
        .expect("Full retained graph");
    assert_eq!(graph.completeness, CargoSnapshotCompleteness::Complete);
    assert_eq!(graph.nodes.len(), 6);
    assert_eq!(graph.roots.len(), 2);
    let shared = graph
        .nodes
        .iter()
        .filter(|node| node.name == "shared")
        .collect::<Vec<_>>();
    assert_eq!(shared.len(), 3);
    let public = shared
        .iter()
        .filter(|node| node.crates_io)
        .collect::<Vec<_>>();
    assert_eq!(public.len(), 2);
    assert!(
        public
            .iter()
            .all(|node| node.lock_status == CargoDependencyLockStatus::Locked)
    );
    let private = shared
        .iter()
        .find(|node| !node.crates_io)
        .expect("Private instance");
    assert_eq!(private.lock_status, CargoDependencyLockStatus::NotLocked);
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.alias == "transitive_alias"
                && edge.kind == CargoDependencyKind::Build
                && edge.target.as_deref() == Some("cfg(unix)"))
    );
    assert!(
        graph
            .nodes
            .iter()
            .filter(|node| node.name == "shared")
            .all(|node| node.enabled_features == ["default"])
    );
    assert!(graph.lock_identity.is_some());
    let debug = format!("{graph:?}");
    assert!(!debug.contains("PRIVATE_GRAPH_TOKEN"));
    assert!(!debug.contains("PRIVATE_GRAPH_PATH"));
    validate_cargo_snapshot_bounds(&CargoWorkspaceSnapshot {
        revision: 1,
        input_fingerprint: 2,
        workspaces: vec![workspace],
        failures: Vec::new(),
        completeness: CargoSnapshotCompleteness::Complete,
    })
    .expect("Bounded snapshot");
}

#[test]
fn doctor_cargo_graph_missing_resolution_private_workspace_and_stale_lock_are_not_full_proof() {
    use project::cargo_workspace::CargoSnapshotCompleteness;
    let mut value = doctor_cargo_graph_fixture();
    value["resolve"] = serde_json::Value::Null;
    let metadata =
        parse_metadata(&serde_json::to_vec(&value).expect("Metadata")).expect("Metadata");
    let workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Declarations remain available");
    assert_eq!(
        workspace
            .dependency_graph
            .expect("Unavailable resolution projection")
            .completeness,
        CargoSnapshotCompleteness::Partial
    );
    let metadata =
        parse_metadata(&serde_json::to_vec(&doctor_cargo_graph_fixture()).expect("Metadata"))
            .expect("Metadata");
    let mut workspace = workspace_from_metadata(&metadata, |path| {
        if path.starts_with("/workspace/member-two") {
            None
        } else {
            resolve(Path::new("/workspace"), path)
        }
    })
    .expect("Visible members");
    let graph = workspace.dependency_graph.as_ref().expect("Partial graph");
    assert_eq!(graph.completeness, CargoSnapshotCompleteness::Partial);
    assert!(graph.nodes.iter().all(|node| node.name != "member-two"));
    enrich_dependency_provenance(&mut workspace, &Default::default(), Some("malformed=["));
    assert!(
        workspace
            .dependency_graph
            .as_ref()
            .expect("Graph")
            .nodes
            .iter()
            .all(|node| node.lock_status == CargoDependencyLockStatus::Unknown)
    );
}

#[test]
fn doctor_cargo_graph_invalid_and_oversized_metadata_are_rejected_before_retention() {
    use project::cargo_workspace::{
        CargoSnapshotCompleteness, CargoWorkspaceSnapshot, MAX_CARGO_METADATA_BYTES,
        validate_cargo_snapshot_bounds,
    };
    assert!(parse_metadata(&vec![b' '; MAX_CARGO_METADATA_BYTES + 1]).is_err());
    let metadata =
        parse_metadata(&serde_json::to_vec(&doctor_cargo_graph_fixture()).expect("Metadata"))
            .expect("Metadata");
    let mut workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Graph");
    let graph = workspace.dependency_graph.as_mut().expect("Graph");
    graph.edges.first_mut().expect("Edge").child = u32::MAX;
    assert!(graph.validate().is_err());
    let mut workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Graph");
    workspace.members.first_mut().expect("Member").features = (0..65537)
        .map(|index| project::cargo_workspace::CargoFeatureModel {
            name: format!("feature-{index}"),
            defined: true,
            enabled: CargoFeatureEnabled::Unknown,
            expands: Vec::new(),
        })
        .collect();
    let snapshot = CargoWorkspaceSnapshot {
        revision: 1,
        input_fingerprint: 2,
        workspaces: vec![workspace],
        failures: Vec::new(),
        completeness: CargoSnapshotCompleteness::Complete,
    };
    assert!(validate_cargo_snapshot_bounds(&snapshot).is_err());
}

#[test]
fn doctor_cargo_graph_cycles_and_node_truncation_preserve_explicit_coverage() {
    use project::cargo_workspace::{CargoSnapshotCompleteness, MAX_CARGO_GRAPH_NODES};
    let mut value = doctor_cargo_graph_fixture();
    let nodes = value
        .pointer_mut("/resolve/nodes")
        .and_then(serde_json::Value::as_array_mut)
        .expect("Nodes");
    let transitive = nodes
        .iter_mut()
        .find(|node| node.get("id") == Some(&json!("transitive")))
        .expect("Transitive node");
    transitive["dependencies"] = json!(["public-old"]);
    transitive["deps"] =
        json!([{"name":"cycle","pkg":"public-old","dep_kinds":[{"kind":"dev","target":null}]}]);
    let metadata =
        parse_metadata(&serde_json::to_vec(&value).expect("Metadata")).expect("Metadata");
    let workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Cyclic observed graph");
    assert!(
        workspace
            .dependency_graph
            .expect("Graph")
            .edges
            .iter()
            .any(|edge| edge.alias == "cycle" && edge.kind == CargoDependencyKind::Development)
    );
    let template = value
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .and_then(|packages| packages.get(2))
        .expect("External template")
        .clone();
    for index in 0..MAX_CARGO_GRAPH_NODES {
        let identifier = format!("extra-{index}");
        let mut package = template.clone();
        package["id"] = json!(identifier);
        package["name"] = json!(identifier);
        value
            .get_mut("packages")
            .and_then(serde_json::Value::as_array_mut)
            .expect("Packages")
            .push(package);
        value
            .pointer_mut("/resolve/nodes")
            .and_then(serde_json::Value::as_array_mut)
            .expect("Nodes")
            .push(json!({"id":identifier,"dependencies":[],"deps":[],"features":[]}));
    }
    let metadata =
        parse_metadata(&serde_json::to_vec(&value).expect("Bounded metadata")).expect("Metadata");
    let workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Bounded owner graph");
    let graph = workspace.dependency_graph.expect("Truncated graph");
    assert_eq!(graph.nodes.len(), MAX_CARGO_GRAPH_NODES);
    assert_eq!(graph.completeness, CargoSnapshotCompleteness::Partial);
}

#[test]
fn doctor_cargo_graph_public_sparse_lock_identity_is_canonical_and_private_remains_distinct() {
    let mut value = doctor_cargo_graph_fixture();
    let packages = value
        .get_mut("packages")
        .and_then(serde_json::Value::as_array_mut)
        .expect("Packages");
    let public = packages
        .iter_mut()
        .find(|package| package.get("id") == Some(&json!("public-old")))
        .expect("Public crate");
    public["source"] = json!("sparse+https://index.crates.io/");
    let metadata =
        parse_metadata(&serde_json::to_vec(&value).expect("Metadata")).expect("Metadata");
    let mut workspace =
        workspace_from_metadata(&metadata, |path| resolve(Path::new("/workspace"), path))
            .expect("Graph");
    enrich_dependency_provenance(
        &mut workspace,
        &Default::default(),
        Some(
            "version=4\n[[package]]\nname='shared'\nversion='1.0.0'\nsource='registry+https://github.com/rust-lang/crates.io-index'\n",
        ),
    );
    let graph = workspace.dependency_graph.as_ref().expect("Graph");
    assert_eq!(
        graph
            .nodes
            .iter()
            .find(|node| node.name == "shared" && node.version == "1.0.0" && node.crates_io)
            .expect("Canonical public node")
            .lock_status,
        CargoDependencyLockStatus::Locked
    );
    assert_eq!(
        graph
            .nodes
            .iter()
            .find(|node| node.name == "shared" && !node.crates_io)
            .expect("Private node")
            .lock_status,
        CargoDependencyLockStatus::NotLocked
    );
}
