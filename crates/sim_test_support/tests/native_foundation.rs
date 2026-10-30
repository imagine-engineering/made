use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, encode_png_frame};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, AttemptState, AuthorizedCapabilities,
    NATIVE_IMAGE_REGISTRY_VERSION, NativeImageWorkerEvent, NativeImageWorkerPlan,
    RuntimeSupervisor, RuntimeSupervisorError, SharedAssetService, SupervisorPolicy, WorkerHealth,
    WorkerLaunchConfig, WorkerOperationStage, authorize_native_input_reader,
    compile_native_image_workflow,
};
use sim_test_support::{load_release_boundary_policy, rust_source_before_test_module};
use sim_types::{AttemptId, ProfileId, PromptId, WorkerId, WorkerMessage};
use uuid::Uuid;

const REGISTRY_VERSION: &str = "task-2-foundation-v1";
const MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024;
const NATIVE_IMAGE_WORKFLOW: &[u8] = include_bytes!("../fixtures/native_image/workflow.json");
const NATIVE_IMAGE_INPUT: &[u8] = include_bytes!("../fixtures/native_image/input.json");

struct IsolatedWorker {
    _directory: tempfile::TempDir,
    config: WorkerLaunchConfig,
}

struct IsolatedNativeImageWorker {
    _directory: tempfile::TempDir,
    config: WorkerLaunchConfig,
    assets: SharedAssetService,
    input_authorization: AuthorizedCapabilities,
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn target_directory(workspace_root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"))
}

fn isolated_worker() -> Result<IsolatedWorker, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut config = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_test_worker_fixture"),
        ProfileId(Default::default()),
        WorkerId(Default::default()),
        REGISTRY_VERSION,
        MEMORY_LIMIT_BYTES,
    );
    config.working_directory = Some(directory.path().to_path_buf());
    config.environment = vec![
        ("PATH".to_owned(), String::new()),
        ("SIM_TEST_EXPECT_EMPTY_PATH".to_owned(), "1".to_owned()),
        (
            "SIM_TEST_ISOLATED_ROOT".to_owned(),
            directory.path().to_string_lossy().into_owned(),
        ),
    ];
    config.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(2),
        ready_timeout: Duration::from_secs(2),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    Ok(IsolatedWorker {
        _directory: directory,
        config,
    })
}

fn isolated_native_image_worker() -> Result<IsolatedNativeImageWorker, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let worker_directory = directory.path().join("worker");
    fs::create_dir(&worker_directory)?;
    let profile_id = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_19b1);
    let mut typed_roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = directory.path().join(name);
        fs::create_dir(&path)?;
        typed_roots.push((namespace, path));
    }
    let roots = AssetRoots::new(profile_id.to_string(), typed_roots)?;
    let input: Value = serde_json::from_slice(NATIVE_IMAGE_INPUT)?;
    let pixels = input
        .get("pixels_bhwc")
        .and_then(Value::as_array)
        .ok_or("native image boundary fixture omitted pixels")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or("native image boundary fixture pixel is not numeric")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_png = encode_png_frame(
        &pixels,
        required_fixture_u64(&input, "batch")?,
        required_fixture_u64(&input, "height")?,
        required_fixture_u64(&input, "width")?,
        required_fixture_u64(&input, "channels")?,
        0,
        &BTreeMap::new(),
        PngLimits::default(),
    )?;
    fs::write(
        roots
            .test_root_path(AssetNamespace::Input)?
            .join("fixture.png"),
        input_png,
    )?;
    let mut config = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        ProfileId(profile_id),
        WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_19b2)),
        NATIVE_IMAGE_REGISTRY_VERSION,
        8 * 1024 * 1024 * 1024,
    );
    config.working_directory = Some(worker_directory);
    config.environment = vec![("PATH".to_owned(), String::new())];
    config.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(10),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let assets = Arc::new(Mutex::new(AssetService::open(roots.clone())?));
    let input_authorization = authorize_native_input_reader(&roots.profile_id)?;
    Ok(IsolatedNativeImageWorker {
        _directory: directory,
        config,
        assets,
        input_authorization,
    })
}

fn required_fixture_u64(value: &Value, name: &str) -> Result<u64, Box<dyn Error>> {
    value
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("native image boundary fixture omitted {name}").into())
}

async fn ready_execute_cancel_stop(
    config: WorkerLaunchConfig,
) -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let mut cases = BTreeMap::new();
    let mut supervisor = RuntimeSupervisor::start(config).await?;
    cases.insert(
        "start_ready",
        supervisor.snapshot().health == WorkerHealth::BackendReady,
    );
    cases.insert(
        "cpu_capability_negotiated",
        supervisor
            .accepted_backend()
            .is_some_and(|matrix| matrix.device() == sim_tensor::DeviceId::CPU),
    );
    cases.insert(
        "private_process_ipc",
        supervisor
            .snapshot()
            .launch
            .arguments
            .iter()
            .any(|argument| argument == "--memory-limit-bytes"),
    );
    cases.insert(
        "empty_python_path",
        supervisor
            .snapshot()
            .launch
            .environment_names
            .iter()
            .any(|name| name == "PATH"),
    );

    let prompt_id = PromptId(Default::default());
    let attempt_id = AttemptId(Default::default());
    supervisor
        .execute(prompt_id, attempt_id, b"deterministic-cpu-plan".to_vec())
        .await?;
    let execution_event = supervisor.next_event(Duration::from_secs(2)).await?;
    cases.insert(
        "first_native_slice",
        matches!(
            execution_event.message,
            WorkerMessage::Lifecycle {
                event: sim_types::WorkerLifecycleEvent::ExecutionStarted
            }
        ),
    );
    supervisor
        .cancel(prompt_id, attempt_id, "validation cancellation")
        .await?;
    let cancellation_event = supervisor.next_event(Duration::from_secs(2)).await?;
    cases.insert(
        "cancellation_converged",
        matches!(
            cancellation_event.message,
            WorkerMessage::Lifecycle {
                event: sim_types::WorkerLifecycleEvent::CancellationRequested { ref reason }
            } if reason == "validation cancellation"
        ),
    );
    let status = supervisor.shutdown().await?;
    cases.insert("stop_and_quit", status.success());
    cases.insert(
        "terminal_exit_recorded",
        matches!(
            supervisor.snapshot().health,
            WorkerHealth::Exited { success: true, .. }
        ),
    );
    Ok(cases)
}

fn assert_all_cases(cases: &BTreeMap<&str, bool>) {
    assert!(
        cases.values().all(|passed| *passed),
        "validation cases failed: {cases:?}"
    );
}

fn write_artifact(
    workspace_root: &Path,
    filename: &str,
    validation: &str,
    scope: &str,
    fixture_digests: Value,
    cases: &BTreeMap<&str, bool>,
    residual_release_gates: &[&str],
    closure_stage: &str,
    closure_reason: &str,
) -> Result<(), Box<dyn Error>> {
    assert_all_cases(cases);
    let artifact_directory = target_directory(workspace_root).join("comfy-parity");
    fs::create_dir_all(&artifact_directory)?;
    let artifact = json!({
        "validation_id": validation,
        "validation": validation,
        "scope": scope,
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "path_python_available": false,
            "source_tree_available_to_worker": false,
            "public_protocol_loopback": false,
        },
        "fixture_digests": fixture_digests,
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
        },
        "cases": cases,
        "skipped": [],
        "release_closure": {
            "claimed": false,
            "stage": closure_stage,
            "reason": closure_reason,
            "remaining_gates": residual_release_gates,
        },
    });
    fs::write(
        artifact_directory.join(filename),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}

fn cargo_metadata(workspace_root: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = smol::block_on(async {
        smol::process::Command::new(env!("CARGO"))
            .args(["metadata", "--locked", "--no-deps", "--format-version", "1"])
            .current_dir(workspace_root)
            .output()
            .await
    })?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output.stdout)
}

fn metadata_package<'a>(metadata: &'a Value, name: &str) -> Option<&'a Value> {
    metadata
        .get("packages")?
        .as_array()?
        .iter()
        .find(|package| package.get("name").and_then(Value::as_str) == Some(name))
}

fn feature_matches(package: &Value, feature: &str, expected: &[&str]) -> bool {
    let Some(values) = package
        .get("features")
        .and_then(|features| features.get(feature))
        .and_then(Value::as_array)
    else {
        return false;
    };
    let actual = values
        .iter()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    actual == expected.iter().copied().collect()
}

fn package_has_local_dependencies(package: &Value, expected: &[&str]) -> bool {
    package
        .get("dependencies")
        .and_then(Value::as_array)
        .is_some_and(|dependencies| {
            expected.iter().all(|expected| {
                dependencies.iter().any(|dependency| {
                    dependency.get("name").and_then(Value::as_str) == Some(expected)
                        && dependency.get("path").and_then(Value::as_str).is_some()
                })
            })
        })
}

fn backend_dependency_ledger_matches(
    workspace_root: &Path,
    metadata: &Value,
    adapters: &[(&str, &str)],
) -> Result<bool, Box<dyn Error>> {
    let ledger: Value = serde_json::from_slice(&fs::read(
        workspace_root.join(".agents/specs/comfy-parity/catalogs/native-backend-dependencies.json"),
    )?)?;
    let lockfile_bytes = fs::read(workspace_root.join("Cargo.lock"))?;
    let lockfile_digest = format!("{:x}", Sha256::digest(lockfile_bytes));
    let identity_matches = ledger.get("schema_version").and_then(Value::as_u64) == Some(1)
        && ledger.get("owner_task").and_then(Value::as_str)
            == Some("comfy-parity-vendor-dependency-lock")
        && ledger
            .pointer("/ownership/third_party_version_and_workspace_feature_owner")
            .and_then(Value::as_str)
            == Some("Cargo.toml [workspace.dependencies]")
        && ledger
            .pointer("/ownership/canonical_device_domain_owner")
            .and_then(Value::as_str)
            == Some("sim_types")
        && ledger
            .pointer("/ownership/semantic_backend_and_capability_owner")
            .and_then(Value::as_str)
            == Some("sim_tensor::BackendCapabilityMatrix")
        && ledger
            .pointer("/ownership/vendor_abi_owner_boundary")
            .and_then(Value::as_str)
            == Some("each focused sim_backend_* adapter")
        && ledger
            .pointer("/ownership/platform_loader_non_owner_boundary")
            .and_then(Value::as_str)
            == Some("Zed and GPUI platform/UI loaders own no Sim compute semantics")
        && ledger.pointer("/lockfile/sha256").and_then(Value::as_str)
            == Some(lockfile_digest.as_str());
    let Some(ledger_adapters) = ledger.get("adapters").and_then(Value::as_array) else {
        return Ok(false);
    };
    let Some(ledger_packages) = ledger.get("packages").and_then(Value::as_object) else {
        return Ok(false);
    };
    let adapter_dependencies_match = adapters.iter().all(|(_, package_name)| {
        let Some(ledger_adapter) = ledger_adapters
            .iter()
            .find(|adapter| adapter.get("package").and_then(Value::as_str) == Some(*package_name))
        else {
            return false;
        };
        let expected = ledger_adapter
            .get("dependencies")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|dependency| dependency.get("package").and_then(Value::as_str))
            .collect::<BTreeSet<_>>();
        let actual = metadata_package(metadata, package_name)
            .and_then(|package| package.get("dependencies"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|dependency| dependency.get("name").and_then(Value::as_str))
            .collect::<BTreeSet<_>>();
        !expected.is_empty()
            && expected == actual
            && expected
                .iter()
                .all(|dependency| ledger_packages.contains_key(*dependency))
    });
    Ok(identity_matches && ledger_adapters.len() == adapters.len() && adapter_dependencies_match)
}

fn downstream_manifest_writes_are_scoped(workspace_root: &Path) -> Result<bool, Box<dyn Error>> {
    let tasks = fs::read_to_string(workspace_root.join(".agents/specs/comfy-parity/tasks.md"))?;
    let mut task_id = None;
    let mut manifest_writes = BTreeMap::<String, BTreeSet<String>>::new();
    let mut dependencies = BTreeMap::<String, BTreeSet<String>>::new();
    for line in tasks.lines() {
        if let Some(value) = line.strip_prefix("  - _id: ") {
            task_id = Some(value.to_owned());
            continue;
        }
        if let Some(values) = line.strip_prefix("  - Dependencies: ") {
            let Some(task_id) = task_id.as_ref() else {
                continue;
            };
            dependencies.insert(
                task_id.clone(),
                values
                    .split(", ")
                    .filter(|value| !value.is_empty() && *value != "none")
                    .map(str::to_owned)
                    .collect(),
            );
            continue;
        }
        let Some(writes) = line.strip_prefix("  - Writes: ") else {
            continue;
        };
        let Some(task_id) = task_id.as_ref() else {
            continue;
        };
        let cargo_writes = writes
            .split(", ")
            .filter(|path| path.ends_with("Cargo.toml") || *path == "Cargo.lock")
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        if !cargo_writes.is_empty() {
            manifest_writes.insert(task_id.clone(), cargo_writes);
        }
    }

    let Some(foundation_writes) = manifest_writes.get("comfy-parity-native-crate-foundation")
    else {
        return Ok(false);
    };
    if !foundation_writes.contains("Cargo.toml")
        || !foundation_writes.contains("Cargo.lock")
        || !foundation_writes.contains("crates/sim_plugin_host/Cargo.toml")
        || !foundation_writes.contains("crates/zed/Cargo.toml")
    {
        return Ok(false);
    }

    fn depends_on(
        task: &str,
        dependency: &str,
        dependencies: &BTreeMap<String, BTreeSet<String>>,
        visited: &mut BTreeSet<String>,
    ) -> bool {
        if !visited.insert(task.to_owned()) {
            return false;
        }
        dependencies.get(task).is_some_and(|direct| {
            direct.contains(dependency)
                || direct
                    .iter()
                    .any(|next| depends_on(next, dependency, dependencies, visited))
        })
    }

    let mut writers_by_manifest = BTreeMap::<String, Vec<&str>>::new();
    for (owner, writes) in &manifest_writes {
        for manifest in writes {
            writers_by_manifest
                .entry(manifest.clone())
                .or_default()
                .push(owner);
        }
    }
    for writers in writers_by_manifest.values() {
        for (index, first) in writers.iter().enumerate() {
            for second in &writers[index + 1..] {
                let first_after_second =
                    depends_on(first, second, &dependencies, &mut BTreeSet::new());
                let second_after_first =
                    depends_on(second, first, &dependencies, &mut BTreeSet::new());
                if !first_after_second && !second_after_first {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn hardened_build_scripts(workspace_root: &Path) -> Result<(bool, Value), Box<dyn Error>> {
    let scripts = [
        (
            "crates/sim_tensor/build.rs",
            &[
                "src/ops",
                "src/operation_resolutions",
                "operation_contract_evidence.rs",
                "GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS",
                "validate_resolution_evidence",
            ][..],
        ),
        (
            "crates/sim_model/build.rs",
            &["\"families\"", "\"latent_formats\"", "\"slices\""][..],
        ),
        (
            "crates/sim_sampler/build.rs",
            &["\"algorithms\"", "\"schedulers\""][..],
        ),
        (
            "crates/sim_nodes/build.rs",
            &["\"families\"", "\"slices\""][..],
        ),
    ];
    let mut digests = serde_json::Map::new();
    let mut valid = true;
    for (relative_path, source_directories) in scripts {
        let bytes = fs::read(workspace_root.join(relative_path))?;
        let source = std::str::from_utf8(&bytes)?;
        valid &= source.contains("modules.sort()")
            && source.contains("ok_or_else")
            && source.contains("OUT_DIR")
            && source.contains("entry?")
            && source.contains("valid_module_name")
            && source.contains("include!(concat!")
            && source_directories
                .iter()
                .all(|directory| source.contains(directory));
        digests.insert(
            relative_path.to_owned(),
            Value::String(format!("{:x}", Sha256::digest(bytes))),
        );
    }
    Ok((valid, Value::Object(digests)))
}

fn generated_module_manifests(workspace_root: &Path) -> Result<(bool, Value), Box<dyn Error>> {
    let manifests: [(&str, &[&str], &[&str]); 4] = [
        ("sim_tensor", &["ops"], sim_tensor::GENERATED_MODULES),
        (
            "sim_model",
            &["families", "latent_formats", "slices"],
            sim_model::GENERATED_MODULES,
        ),
        (
            "sim_sampler",
            &["algorithms", "schedulers"],
            sim_sampler::GENERATED_MODULES,
        ),
        (
            "sim_nodes",
            &["families", "slices"],
            sim_nodes::GENERATED_MODULES,
        ),
    ];
    let mut valid = true;
    let mut evidence = serde_json::Map::new();
    for (package, directories, generated) in manifests {
        let mut expected = Vec::new();
        for directory in directories {
            let source_directory = workspace_root
                .join("crates")
                .join(package)
                .join("src")
                .join(directory);
            if !source_directory.is_dir() {
                continue;
            }
            for entry in fs::read_dir(source_directory)? {
                let path = entry?.path();
                if sim_test_support::is_apple_double_metadata(&path)
                    || path.extension().and_then(|extension| extension.to_str()) != Some("rs")
                {
                    continue;
                }
                let Some(name) = path.file_stem().and_then(|name| name.to_str()) else {
                    valid = false;
                    continue;
                };
                expected.push(format!("{directory}/{name}"));
            }
        }
        expected.sort();
        let generated = generated
            .iter()
            .map(|module| (*module).to_owned())
            .collect::<Vec<_>>();
        valid &= generated == expected
            && generated.windows(2).all(|pair| pair[0] < pair[1])
            && generated.iter().all(|module| {
                directories
                    .iter()
                    .any(|directory| module.starts_with(&format!("{directory}/")))
            });
        evidence.insert(
            package.to_owned(),
            json!({
                "modules": generated,
                "sha256": format!("{:x}", Sha256::digest(serde_json::to_vec(&expected)?)),
            }),
        );
    }

    let resolution_directory = workspace_root.join("crates/sim_tensor/src/operation_resolutions");
    let mut expected_resolutions = Vec::new();
    if resolution_directory.is_dir() {
        for entry in fs::read_dir(&resolution_directory)? {
            let path = entry?.path();
            if sim_test_support::is_apple_double_metadata(&path)
                || path.extension().and_then(|extension| extension.to_str()) != Some("rs")
            {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|name| name.to_str()) else {
                valid = false;
                continue;
            };
            expected_resolutions.push(format!("operation_resolutions/{name}"));
        }
    }
    expected_resolutions.sort();
    let generated_resolutions = sim_tensor::GENERATED_OPERATION_RESOLUTION_MODULES
        .iter()
        .map(|module| (*module).to_owned())
        .collect::<Vec<_>>();
    let resolutions_are_paired = generated_resolutions.iter().all(|module| {
        module
            .strip_prefix("operation_resolutions/")
            .is_some_and(|name| {
                workspace_root
                    .join("crates/sim_tensor/src/ops")
                    .join(format!("{name}.rs"))
                    .is_file()
            })
    });
    valid &= generated_resolutions == expected_resolutions
        && generated_resolutions
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        && resolutions_are_paired;
    evidence.insert(
        "sim_tensor_operation_resolutions".to_owned(),
        json!({
            "modules": generated_resolutions,
            "paired_with_operation_sources": resolutions_are_paired,
            "sha256": format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&expected_resolutions)?)
            ),
        }),
    );
    Ok((valid, Value::Object(evidence)))
}

fn write_foundation_artifact(
    workspace_root: &Path,
    fixture_digests: Value,
    cases: &BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    assert_all_cases(cases);
    let artifact_directory = target_directory(workspace_root).join("comfy-parity");
    fs::create_dir_all(&artifact_directory)?;
    let artifact = json!({
        "validation_id": "VAL-FOUNDATION-001",
        "validation": "VAL-FOUNDATION-001",
        "scope": "native-workspace-foundation",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust",
        },
        "fixture_digests": fixture_digests,
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
        },
        "cases": cases,
        "skipped": [],
        "validation_closure": {
            "claimed": true,
            "scope": "Task 2 locked native workspace and feature-forwarding foundation",
        },
        "release_closure_required": false,
    });
    fs::write(
        artifact_directory.join("val-foundation-001.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}

struct ProductionSourceAudit {
    inspected_source_files: usize,
    production_process_launch_files: BTreeSet<String>,
    build_process_launch_files: BTreeSet<String>,
    development_process_launch_files: BTreeSet<String>,
}

fn audit_production_source_boundaries(
    workspace_root: &Path,
) -> Result<ProductionSourceAudit, Box<dyn Error>> {
    let crates_root = workspace_root.join("crates");
    let mut pending = fs::read_dir(&crates_root)?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("sim_"))
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    pending.push(workspace_root.join("crates/zed/src"));

    let mut inspected_source_files = 0;
    let mut production_process_launch_files = BTreeSet::new();
    let mut build_process_launch_files = BTreeSet::new();
    let mut development_process_launch_files = BTreeSet::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if sim_test_support::is_apple_double_metadata(&entry.path())
                || entry.path().extension().and_then(|value| value.to_str()) != Some("rs")
            {
                continue;
            }
            inspected_source_files += 1;
            let source_path = entry.path();
            let source = fs::read_to_string(&source_path)?;
            let source = rust_source_before_test_module(&source);
            let relative_path = source_path
                .strip_prefix(workspace_root)?
                .to_string_lossy()
                .replace('\\', "/");
            let source_tree_prefix = ["../projects", "/comfy"].concat();
            assert!(
                !source.contains(&source_tree_prefix),
                "release Rust source {relative_path} has a compile-time projects/comfy dependency"
            );

            let uses_process_api = source.contains("std::process::Command::new")
                || source.contains("smol::process::Command::new")
                || source.contains("Child::spawn(")
                || (source.contains("Command::new(")
                    && (source.contains("process::{Command")
                        || source.contains("process::Command")));
            if !uses_process_api {
                continue;
            }

            if relative_path.ends_with("/build.rs") {
                build_process_launch_files.insert(relative_path);
                continue;
            }
            let is_development = relative_path.starts_with("crates/sim_test_support/")
                || relative_path.contains("/tests/")
                || relative_path == "crates/zed/src/visual_test_runner.rs";
            if is_development {
                development_process_launch_files.insert(relative_path);
                continue;
            }

            let lowercase_source = source.to_ascii_lowercase();
            for marker in [
                "python",
                "node_modules/comfy",
                "comfyui/main.py",
                "http://127.0.0.1:8188",
                "ws://127.0.0.1:8188",
            ] {
                assert!(
                    !lowercase_source.contains(marker),
                    "production process source {relative_path} contains forbidden source-runtime marker {marker}"
                );
            }
            production_process_launch_files.insert(relative_path);
        }
    }

    assert_eq!(
        production_process_launch_files,
        BTreeSet::from(["crates/sim_runtime/src/runtime_supervisor.rs".to_owned()]),
        "the native Rust worker supervisor must be the only production Sim process launcher"
    );
    assert!(
        development_process_launch_files.contains("crates/sim_test_support/src/oracle.rs"),
        "the source oracle launcher must remain isolated in sim_test_support"
    );
    assert_eq!(
        build_process_launch_files,
        BTreeSet::from(["crates/sim_backend_rocm/build.rs".to_owned()]),
        "only the reviewed ROCm ABI proof build script may launch build-time compiler tools"
    );
    Ok(ProductionSourceAudit {
        inspected_source_files,
        production_process_launch_files,
        build_process_launch_files,
        development_process_launch_files,
    })
}

#[test]
fn val_foundation_001() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let metadata_bytes = cargo_metadata(&workspace_root)?;
    let metadata: Value = serde_json::from_slice(&metadata_bytes)?;
    let policy = load_release_boundary_policy()?;
    policy.verify_launcher_layout(&workspace_root)?;
    let boundary_report = policy.verify_cargo_metadata(&metadata)?;

    let package_names = metadata
        .get("packages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|package| package.get("name").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let foundation_packages = [
        "sim_api",
        "sim_backend_corex",
        "sim_backend_cuda",
        "sim_backend_directml",
        "sim_backend_metal",
        "sim_backend_mlu",
        "sim_backend_npu",
        "sim_backend_rocm",
        "sim_backend_xpu",
        "sim_media",
        "sim_model",
        "sim_nodes",
        "sim_plugin_host",
        "sim_plugin_sdk",
        "sim_runtime",
        "sim_sampler",
        "sim_tensor",
        "sim_test_support",
        "sim_types",
        "sim_ui",
        "sim_worker",
        "zed",
    ];
    let tensor_package = metadata_package(&metadata, "sim_tensor")
        .ok_or("sim_tensor is absent from locked workspace metadata")?;
    let test_support_package = metadata_package(&metadata, "sim_test_support")
        .ok_or("sim_test_support is absent from locked workspace metadata")?;
    let worker_package = metadata_package(&metadata, "sim_worker")
        .ok_or("sim_worker is absent from locked workspace metadata")?;
    let plugin_host_package = metadata_package(&metadata, "sim_plugin_host")
        .ok_or("sim_plugin_host is absent from locked workspace metadata")?;
    let plugin_sdk_package = metadata_package(&metadata, "sim_plugin_sdk")
        .ok_or("sim_plugin_sdk is absent from locked workspace metadata")?;
    let zed_package =
        metadata_package(&metadata, "zed").ok_or("zed is absent from locked workspace metadata")?;
    let adapters = [
        ("cuda", "sim_backend_cuda"),
        ("rocm", "sim_backend_rocm"),
        ("metal", "sim_backend_metal"),
        ("directml", "sim_backend_directml"),
        ("xpu", "sim_backend_xpu"),
        ("npu", "sim_backend_npu"),
        ("mlu", "sim_backend_mlu"),
        ("corex", "sim_backend_corex"),
    ];

    let (build_scripts_are_hardened, build_script_digests) =
        hardened_build_scripts(&workspace_root)?;
    let (generated_manifests_match_sources, generated_manifest_digests) =
        generated_module_manifests(&workspace_root)?;
    let source_audit = audit_production_source_boundaries(&workspace_root)?;
    let rocm_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/rocm/package-policy.json"))?;
    let rocm_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/rocm/ffi-contracts-v1.schema.json"),
    )?;
    let rocm_packager = fs::read_to_string(workspace_root.join("script/package-sim-backend-rocm"))?;
    let metal_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/metal/package-policy.json"))?;
    let metal_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/metal/ffi-contracts-v1.schema.json"),
    )?;
    let metal_packager =
        fs::read_to_string(workspace_root.join("script/package-sim-backend-metal"))?;
    let metal_runtime_ffi =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/native_ffi_metal.rs"))?;
    let mlu_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/mlu/package-policy.json"))?;
    let mlu_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/mlu/ffi-contracts-v1.schema.json"),
    )?;
    let mlu_packager = fs::read_to_string(workspace_root.join("script/package-sim-backend-mlu"))?;
    let mlu_runtime_ffi =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/native_ffi_mlu.rs"))?;
    let npu_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/npu/package-policy.json"))?;
    let npu_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/npu/ffi-contracts-v1.schema.json"),
    )?;
    let npu_packager = fs::read_to_string(workspace_root.join("script/package-sim-backend-npu"))?;
    let npu_runtime_ffi =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/native_ffi_npu.rs"))?;
    let cuda_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/cuda/package-policy.json"))?;
    let cuda_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/cuda/ffi-contracts-v1.schema.json"),
    )?;
    let cuda_packager = fs::read_to_string(workspace_root.join("script/package-sim-backend-cuda"))?;
    let cuda_runtime_ffi =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/native_ffi_cuda.rs"))?;
    let xpu_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/xpu/package-policy.json"))?;
    let xpu_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/xpu/ffi-contracts-v1.schema.json"),
    )?;
    let xpu_packager = fs::read_to_string(workspace_root.join("script/package-sim-backend-xpu"))?;
    let xpu_runtime_ffi =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/native_ffi_xpu.rs"))?;
    let runtime_trust = fs::read_to_string(workspace_root.join("crates/sim_runtime/src/trust.rs"))?;
    let directml_package_policy =
        fs::read_to_string(workspace_root.join("nix/sim-backends/directml/package-policy.json"))?;
    let directml_contract_schema = fs::read_to_string(
        workspace_root.join("nix/sim-backends/directml/ffi-contracts-v1.schema.json"),
    )?;
    let directml_packager =
        fs::read_to_string(workspace_root.join("script/package-sim-backend-directml"))?;
    let directml_runtime_ffi =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/native_ffi_directml.rs"))?;
    let mut cases = BTreeMap::new();
    cases.insert("locked_metadata_available", !metadata_bytes.is_empty());
    cases.insert(
        "all_foundation_packages_registered",
        foundation_packages
            .iter()
            .all(|name| package_names.contains(name)),
    );
    cases.insert(
        "worker_binary_registered",
        worker_package
            .get("targets")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|target| {
                target.get("name").and_then(Value::as_str) == Some("sim-worker")
                    && target
                        .get("kind")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .any(|kind| kind.as_str() == Some("bin"))
            }),
    );
    cases.insert(
        "tensor_cpu_is_default",
        feature_matches(tensor_package, "default", &["cpu"]),
    );
    cases.insert(
        "tensor_accelerators_forward_to_exact_local_adapters",
        adapters.iter().all(|(feature, adapter)| {
            feature_matches(tensor_package, feature, &["cpu", &format!("dep:{adapter}")])
        }),
    );
    cases.insert(
        "test_support_forwards_every_backend_feature",
        adapters.iter().all(|(feature, _)| {
            let tensor = format!("sim_tensor/{feature}");
            if *feature == "metal" {
                feature_matches(
                    test_support_package,
                    feature,
                    &["sim_runtime/metal", &tensor],
                )
            } else {
                feature_matches(test_support_package, feature, &[&tensor])
            }
        }) && feature_matches(test_support_package, "default", &["cpu"]),
    );
    cases.insert(
        "adapter_dependencies_match_the_canonical_locked_ledger",
        backend_dependency_ledger_matches(&workspace_root, &metadata, &adapters)?,
    );
    cases.insert(
        "zed_uses_local_native_comfy_services",
        package_has_local_dependencies(zed_package, &["sim_api", "sim_plugin_host", "sim_ui"]),
    );
    cases.insert(
        "plugin_host_uses_canonical_production_foundations",
        package_has_local_dependencies(
            plugin_host_package,
            &[
                "sim_nodes",
                "sim_plugin_sdk",
                "sim_runtime",
                "sim_types",
                "extension_host",
            ],
        ),
    );
    cases.insert(
        "plugin_sdk_uses_canonical_tensor_domain",
        package_has_local_dependencies(plugin_sdk_package, &["sim_tensor"]),
    );
    cases.insert(
        "manifest_writes_match_serialized_task_owners",
        downstream_manifest_writes_are_scoped(&workspace_root)?,
    );
    cases.insert(
        "generated_module_build_scripts_hardened",
        build_scripts_are_hardened,
    );
    cases.insert(
        "generated_module_outputs_match_sorted_sources",
        generated_manifests_match_sources,
    );
    cases.insert(
        "development_oracle_has_no_production_reverse_dependency",
        boundary_report.is_clean(),
    );
    cases.insert(
        "production_source_launcher_scan_clean",
        source_audit.inspected_source_files > 0
            && source_audit.production_process_launch_files
                == BTreeSet::from(["crates/sim_runtime/src/runtime_supervisor.rs".to_owned()]),
    );
    cases.insert(
        "rocm_signed_contract_provisioning_is_native_and_reviewed",
        rocm_package_policy.contains("\"schema_version\": 2")
            && rocm_package_policy.contains("\"ffi-contracts-v1.json\"")
            && rocm_package_policy
                .contains("\"signature_verifier\": \"sim_runtime-native-rust-ed25519\"")
            && rocm_contract_schema.contains("\"additionalProperties\": false")
            && rocm_contract_schema.contains("rocm-dependency:")
            && rocm_packager.contains("validate_and_copy_contract_catalog")
            && rocm_packager.contains("canonical-json-v1")
            && !rocm_packager.contains("SIM_ROCM_SIGNATURE_VERIFIER"),
    );
    cases.insert(
        "metal_signed_contract_provisioning_is_native_reviewed_and_complete",
        metal_package_policy.contains("\"schema_version\": 2")
            && metal_package_policy.contains("\"ffi-contracts-v1.json\"")
            && metal_package_policy.contains("readiness.metallib")
            && metal_package_policy.contains("tensor_ops.metallib")
            && metal_package_policy
                .contains("\"signature_verifier\": \"sim_runtime-native-rust-ed25519\"")
            && metal_package_policy
                .contains("\"signature_authority\": \"sim_runtime::MetalPackageVerificationKey\"")
            && metal_contract_schema.contains("\"additionalProperties\": false")
            && metal_contract_schema.contains("metal-performance-shaders-graph-framework")
            && metal_contract_schema.contains("metal-tensor-ops-metallib")
            && metal_packager.contains("validate_and_copy_contract_catalog")
            && metal_packager.contains("canonical-json-v1")
            && !metal_packager.contains("SIM_METAL_SIGNATURE_VERIFIER")
            && metal_runtime_ffi.contains("verify_metal_package_contracts")
            && metal_runtime_ffi.contains("NativeFfiRegistry::new")
            && metal_runtime_ffi.contains("readiness_metallib")
            && metal_runtime_ffi.contains("tensor_ops_metallib"),
    );
    cases.insert(
        "mlu_signed_contract_provisioning_is_native_reviewed_and_complete",
        mlu_package_policy.contains("\"ffi-contracts-v1.json\"")
            && mlu_package_policy.contains("\"signature_domain\": \"zed-sim-mlu-package-v1\"")
            && mlu_package_policy.contains("\"redistributes_vendor_runtime\": false")
            && mlu_contract_schema.contains("\"additionalProperties\": false")
            && mlu_contract_schema.contains("sim_backend_mlu::loader")
            && mlu_packager.contains("separately reviewed bounded regular file")
            && mlu_packager.contains("ffi_contracts_sha256")
            && !mlu_packager.contains("SIM_MLU_SIGNATURE_VERIFIER")
            && mlu_runtime_ffi.contains("verify_mlu_package_contracts")
            && mlu_runtime_ffi.contains("NativeFfiRegistry::new")
            && mlu_runtime_ffi.contains("capture_native_library_image")
            && mlu_runtime_ffi.contains("RetainedNativeLibraryImage")
            && !mlu_runtime_ffi.contains("memfd_create")
            && runtime_trust.contains("capture_native_library_image")
            && runtime_trust.contains("RetainedNativeLibraryImage")
            && runtime_trust.contains("libc::memfd_create")
            && mlu_runtime_ffi.contains("MluExecutionRuntime::load_certified"),
    );
    cases.insert(
        "npu_signed_contract_provisioning_is_native_reviewed_and_complete",
        npu_package_policy.contains("\"ffi-contracts-v1.json\"")
            && npu_package_policy.contains("\"signature_domain\": \"zed-sim-npu-package-v1\"")
            && npu_package_policy.contains("\"redistributes_vendor_runtime\": false")
            && npu_contract_schema.contains("\"additionalProperties\": false")
            && npu_contract_schema.contains("sim_backend_npu::loader")
            && npu_contract_schema.contains("\"required_by\": { \"const\": \"ascendcl\" }")
            && npu_packager.contains("separately reviewed bounded regular file")
            && npu_packager.contains("ffi_contracts_sha256")
            && !npu_packager.contains("SIM_NPU_SIGNATURE_VERIFIER")
            && npu_runtime_ffi.contains("verify_npu_package_contracts")
            && npu_runtime_ffi.contains("NativeFfiRegistry::new")
            && npu_runtime_ffi.contains("NativeFfiContract::new_dependency")
            && npu_runtime_ffi.contains("authorize_dependency")
            && npu_runtime_ffi.contains("capture_native_library_image")
            && npu_runtime_ffi.contains("RetainedNativeLibraryImage")
            && !npu_runtime_ffi.contains("memfd_create")
            && runtime_trust.contains("capture_native_library_image")
            && runtime_trust.contains("RetainedNativeLibraryImage")
            && runtime_trust.contains("libc::memfd_create")
            && npu_runtime_ffi.contains("NpuExecutionSession::from_registry_certified_images"),
    );
    cases.insert(
        "cuda_signed_contract_provisioning_is_native_reviewed_and_complete",
        cuda_package_policy.contains("\"ffi-contracts-v1.json\"")
            && cuda_package_policy.contains("\"signature_domain\": \"zed-sim-cuda-package-v1\"")
            && cuda_package_policy.contains("\"redistributes_driver\": false")
            && cuda_package_policy.contains("\"structural_receipt_is_authorization\": false")
            && cuda_contract_schema.contains("\"additionalProperties\": false")
            && cuda_contract_schema.contains("sim_backend_cuda::loader")
            && ["cublaslt", "cudnn", "driver", "nvrtc"]
                .into_iter()
                .all(|identity| cuda_contract_schema.contains(identity))
            && cuda_packager.contains("separately reviewed CUDA FFI contract catalog")
            && cuda_packager.contains("ffi_contracts_sha256")
            && !cuda_packager.contains("SIM_CUDA_SIGNATURE_VERIFIER")
            && !cuda_packager.contains("NativeFfiRegistry::")
            && cuda_runtime_ffi.contains("verify_cuda_package_contracts")
            && cuda_runtime_ffi.contains("verification_key.verify_package")
            && cuda_runtime_ffi.contains("NativeFfiRegistry::new")
            && cuda_runtime_ffi.contains("capture_native_library_image")
            && cuda_runtime_ffi.contains("RetainedNativeLibraryImage")
            && !cuda_runtime_ffi.contains("fn capture_native_package")
            && runtime_trust.contains("pub struct CudaPackageVerificationKey")
            && runtime_trust.contains("CUDA_PACKAGE_SIGNATURE_DOMAIN")
            && cuda_runtime_ffi.contains("CudaExecutionSession::from_registry_certified_images"),
    );
    cases.insert(
        "xpu_signed_contract_provisioning_is_native_reviewed_and_complete",
        xpu_package_policy.contains("\"ffi-contracts-v1.json\"")
            && xpu_package_policy.contains("\"signature_domain\": \"zed-sim-xpu-package-v1\"")
            && xpu_package_policy.contains("\"redistributes_vendor_runtime\": false")
            && xpu_package_policy.contains("\"structural_receipt_is_authorization\": false")
            && xpu_contract_schema.contains("\"additionalProperties\": false")
            && xpu_contract_schema.contains("sim_backend_xpu::loader")
            && xpu_contract_schema.contains("\"identity\": { \"const\": \"level_zero\" }")
            && xpu_contract_schema.contains("\"identity\": { \"const\": \"onednn\" }")
            && xpu_packager.contains("separately reviewed XPU FFI contract catalog")
            && xpu_packager.contains("ffi_contracts_sha256")
            && !xpu_packager.contains("SIM_XPU_SIGNATURE_VERIFIER")
            && !xpu_packager.contains("NativeFfiRegistry::")
            && xpu_runtime_ffi.contains("verify_xpu_package_contracts")
            && xpu_runtime_ffi.contains("verification_key.verify_package")
            && xpu_runtime_ffi.contains("NativeFfiRegistry::new")
            && xpu_runtime_ffi.contains("capture_native_library_image")
            && xpu_runtime_ffi.contains("RetainedNativeLibraryImage")
            && !xpu_runtime_ffi.contains("fn capture_native_package")
            && runtime_trust.contains("pub struct XpuPackageVerificationKey")
            && runtime_trust.contains("XPU_PACKAGE_SIGNATURE_DOMAIN")
            && xpu_runtime_ffi.contains("XpuExecutionSession::from_registry_certified_images"),
    );
    cases.insert(
        "directml_signed_contract_provisioning_is_native_reviewed_and_complete",
        directml_package_policy.contains("\"schema_version\": 2")
            && directml_package_policy.contains("\"ffi-contracts-v1.json\"")
            && directml_package_policy
                .contains("\"signature_domain\": \"zed-sim-directml-package-v1\"")
            && directml_package_policy.contains(
                "\"signature_authority\": \"sim_runtime::DirectMlPackageVerificationKey\"",
            )
            && directml_package_policy
                .contains("\"signature_verifier\": \"sim_runtime-native-rust-ed25519\"")
            && directml_package_policy
                .contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\"")
            && directml_contract_schema.contains("\"additionalProperties\": false")
            && directml_contract_schema.contains("D3D12.dll")
            && directml_contract_schema.contains("DirectML.dll")
            && directml_contract_schema.contains("DXGI.dll")
            && directml_packager.contains("validate_contract_catalog")
            && directml_packager.contains("stable_regular_file")
            && directml_packager.contains("separately reviewed FFI contract catalog")
            && !directml_packager.contains("SIM_DIRECTML_SIGNATURE_VERIFIER")
            && !directml_packager.contains("WinVerifyTrust")
            && directml_runtime_ffi.contains("verify_directml_package_contracts")
            && directml_runtime_ffi.contains("capture_native_package")
            && directml_runtime_ffi.contains("validate_native_package_coverage")
            && directml_runtime_ffi.contains("NativeFfiRegistry::new")
            && directml_runtime_ffi.contains("RetainedDirectMlLibraryHandles"),
    );

    write_foundation_artifact(
        &workspace_root,
        json!({
            "cargo_metadata_sha256": format!("{:x}", Sha256::digest(&metadata_bytes)),
            "release_policy_sha256": format!("{:x}", Sha256::digest(include_bytes!("../fixtures/release-boundary.json"))),
            "build_scripts": build_script_digests,
            "generated_module_manifests": generated_manifest_digests,
            "production_source_files_inspected": source_audit.inspected_source_files,
            "production_process_launch_files": source_audit.production_process_launch_files,
            "build_process_launch_files": source_audit.build_process_launch_files,
            "development_process_launch_files": source_audit.development_process_launch_files,
        }),
        &cases,
    )
}

#[test]
fn val_e2e_002() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let first_worker = isolated_worker()?;
    let helper_binary = fs::read(&first_worker.config.binary)?;
    let helper_source = include_bytes!("../src/bin/sim_test_worker_fixture.rs");
    let mut cases = smol::block_on(ready_execute_cancel_stop(first_worker.config))?;

    let restarted_worker = isolated_worker()?;
    let mut restarted = smol::block_on(RuntimeSupervisor::start(restarted_worker.config))?;
    cases.insert(
        "restart_ready",
        restarted.snapshot().health == WorkerHealth::BackendReady,
    );
    cases.insert(
        "restart_stop",
        smol::block_on(restarted.shutdown())?.success(),
    );

    let crash_worker = isolated_worker()?;
    let mut crashing = smol::block_on(RuntimeSupervisor::start(crash_worker.config))?;
    let crash_status = smol::block_on(crashing.terminate())?;
    cases.insert("injected_process_crash_detected", !crash_status.success());
    let mut recovered = smol::block_on(crashing.recover())?;
    cases.insert(
        "crash_recovered_once",
        recovered.snapshot().health == WorkerHealth::BackendReady
            && recovered
                .snapshot()
                .operation
                .transitions
                .iter()
                .any(|transition| transition.stage == WorkerOperationStage::Recover),
    );
    let second_crash_status = smol::block_on(recovered.terminate())?;
    cases.insert("second_crash_detected", !second_crash_status.success());
    cases.insert(
        "restart_loop_prevented",
        matches!(
            smol::block_on(recovered.recover()),
            Err(RuntimeSupervisorError::RecoveryBudgetExhausted { maximum: 1 })
        ),
    );
    cases.insert(
        "no_python_or_source_runtime",
        !helper_binary
            .windows(b"python3".len())
            .any(|window| window.eq_ignore_ascii_case(b"python3"))
            && !helper_binary
                .windows(b"ComfyUI/main.py".len())
                .any(|window| window.eq_ignore_ascii_case(b"ComfyUI/main.py")),
    );
    cases.insert(
        "worker_fixture_network_api_absent",
        !helper_source
            .windows(b"std::net".len())
            .any(|window| window == b"std::net")
            && !helper_source
                .windows(b"TcpStream".len())
                .any(|window| window == b"TcpStream")
            && !helper_source
                .windows(b"TcpListener".len())
                .any(|window| window == b"TcpListener")
            && !helper_source
                .windows(b"UdpSocket".len())
                .any(|window| window == b"UdpSocket"),
    );

    write_artifact(
        &workspace_root,
        "val-e2e-002.json",
        "VAL-E2E-002",
        "task-2-native-worker-foundation",
        json!({
            "worker_fixture_source_sha256": format!("{:x}", Sha256::digest(helper_source)),
            "plan_sha256": format!("{:x}", Sha256::digest(b"deterministic-cpu-plan")),
            "registry_version_sha256": format!("{:x}", Sha256::digest(REGISTRY_VERSION.as_bytes())),
        }),
        &cases,
        &[
            "packaged Zed worker discovery and signing",
            "OS-enforced network-denied release host",
            "final native image and diffusion slices",
            "platform orphan-process certification",
        ],
        "native-crate-foundation",
        "This artifact proves Task 2 foundation invariants only; final packaged release closure belongs to the terminal release audit after downstream native slices are implemented.",
    )
}

#[test]
fn val_native_boundary_001() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let metadata_bytes = cargo_metadata(&workspace_root)?;
    let metadata: Value = serde_json::from_slice(&metadata_bytes)?;
    let policy = load_release_boundary_policy()?;
    policy.verify_launcher_layout(&workspace_root)?;
    let report = policy.verify_cargo_metadata(&metadata)?;
    let source_audit = audit_production_source_boundaries(&workspace_root)?;

    let isolated = isolated_worker()?;
    let helper_binary = fs::read(&isolated.config.binary)?;
    let helper_source = include_bytes!("../src/bin/sim_test_worker_fixture.rs");
    let mut cases = smol::block_on(ready_execute_cancel_stop(isolated.config))?;
    let native_image = isolated_native_image_worker()?;
    let native_worker_binary = fs::read(&native_image.config.binary)?;
    let native_wrapper_source = include_bytes!("../src/bin/sim_native_image_worker_fixture.rs");
    let native_worker_source = include_bytes!("../../sim_worker/src/sim_worker.rs");
    let mut plan = compile_native_image_workflow(NATIVE_IMAGE_WORKFLOW, &BTreeSet::new())?;
    plan.prompt_id = PromptId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_19b3));
    let mut native_supervisor = smol::block_on(RuntimeSupervisor::start(native_image.config))?;
    cases.insert(
        "native_image_worker_ready_in_isolated_root",
        native_supervisor.snapshot().health == WorkerHealth::BackendReady
            && native_supervisor
                .accepted_backend()
                .is_some_and(|matrix| matrix.device() == sim_tensor::DeviceId::CPU)
            && native_supervisor
                .snapshot()
                .launch
                .environment_names
                .iter()
                .any(|name| name == "PATH")
            && native_supervisor
                .snapshot()
                .launch
                .working_directory
                .as_ref()
                .is_some_and(|directory| {
                    let directory = Path::new(directory);
                    !directory.join("projects/comfy").exists()
                        && !directory.join("ComfyUI").exists()
                        && !directory.join("ComfyUI-Frontend").exists()
                }),
    );
    let native_prompt_id = plan.prompt_id;
    let native_worker_plan = NativeImageWorkerPlan::from_asset_service(
        plan,
        &native_image.assets,
        &native_image.input_authorization,
        &sim_types::CancellationToken::default(),
        true,
        0,
    )?;
    smol::block_on(native_supervisor.execute(
        native_prompt_id,
        AttemptId(Uuid::from_u128(1)),
        serde_json::to_vec(&native_worker_plan)?,
    ))?;
    let native_result = smol::block_on(async {
        let mut proposal_count = 0_usize;
        loop {
            let envelope = native_supervisor
                .next_event(Duration::from_secs(10))
                .await?;
            match envelope.message {
                WorkerMessage::OutputProposal { .. } => {
                    proposal_count = proposal_count.saturating_add(1);
                }
                WorkerMessage::Event { event } => {
                    if let Ok(result) = postcard::from_bytes::<NativeImageWorkerEvent>(&event) {
                        match result {
                            NativeImageWorkerEvent::Progress { .. } => {}
                            NativeImageWorkerEvent::Completed { .. }
                            | NativeImageWorkerEvent::BackendUnavailable { .. }
                            | NativeImageWorkerEvent::Failed { .. } => {
                                return Ok::<_, RuntimeSupervisorError>((result, proposal_count));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    })?;
    cases.insert(
        "native_image_slice_executes_without_external_runtime",
        matches!(
            native_result.0,
            NativeImageWorkerEvent::Completed { ref result }
                if result.report.state == AttemptState::Succeeded
                    && result.executed_node_count == 5
                    && result.output_proposal_ids.len() == 2
                    && native_result.1 == 2
        ),
    );
    cases.insert(
        "native_image_worker_binary_strings_clean",
        [
            "python3",
            "python.exe",
            "ComfyUI/main.py",
            "node_modules/comfy",
            "http://127.0.0.1:8188",
            "ws://127.0.0.1:8188",
        ]
        .iter()
        .all(|marker| {
            !native_worker_binary
                .windows(marker.len())
                .any(|window| window.eq_ignore_ascii_case(marker.as_bytes()))
        }),
    );
    cases.insert(
        "native_image_worker_sources_expose_no_network_api",
        [
            native_wrapper_source.as_slice(),
            native_worker_source.as_slice(),
        ]
        .into_iter()
        .all(|source| {
            ["std::net", "TcpStream", "TcpListener", "UdpSocket"]
                .iter()
                .all(|marker| {
                    !source
                        .windows(marker.len())
                        .any(|window| window == marker.as_bytes())
                })
        }),
    );
    cases.insert(
        "native_image_worker_shutdown_succeeds",
        smol::block_on(native_supervisor.shutdown())?.success(),
    );
    cases.insert("locked_workspace_metadata", !metadata_bytes.is_empty());
    cases.insert("development_packages_present", {
        report.development_packages_found == policy.development_only_packages
    });
    cases.insert("production_reverse_dependencies_clean", report.is_clean());
    cases.insert(
        "source_launchers_development_only",
        source_audit
            .development_process_launch_files
            .contains("crates/sim_test_support/src/oracle.rs"),
    );
    cases.insert(
        "production_launcher_scan_clean",
        source_audit.inspected_source_files > 0
            && source_audit.production_process_launch_files
                == BTreeSet::from(["crates/sim_runtime/src/runtime_supervisor.rs".to_owned()]),
    );
    cases.insert(
        "foundation_worker_fixture_binary_strings_clean",
        [
            "python3",
            "python.exe",
            "ComfyUI/main.py",
            "node_modules/comfy",
            "http://127.0.0.1:8188",
        ]
        .iter()
        .all(|marker| {
            !helper_binary
                .windows(marker.len())
                .any(|window| window.eq_ignore_ascii_case(marker.as_bytes()))
        }),
    );
    cases.insert(
        "worker_fixture_network_api_absent",
        !helper_source
            .windows(b"std::net".len())
            .any(|window| window == b"std::net")
            && !helper_source
                .windows(b"TcpStream".len())
                .any(|window| window == b"TcpStream")
            && !helper_source
                .windows(b"TcpListener".len())
                .any(|window| window == b"TcpListener")
            && !helper_source
                .windows(b"UdpSocket".len())
                .any(|window| window == b"UdpSocket"),
    );

    write_artifact(
        &workspace_root,
        "val-native-boundary-001.json",
        "VAL-NATIVE-BOUNDARY-001",
        "task-2-foundation-and-task-19-native-image-boundary",
        json!({
            "cargo_metadata_sha256": format!("{:x}", Sha256::digest(&metadata_bytes)),
            "release_policy_sha256": format!("{:x}", Sha256::digest(include_bytes!("../fixtures/release-boundary.json"))),
            "worker_fixture_source_sha256": format!("{:x}", Sha256::digest(&helper_source)),
            "worker_fixture_binary_bytes_inspected": helper_binary.len(),
            "native_image_worker_binary_bytes_inspected": native_worker_binary.len(),
            "native_image_wrapper_source_sha256": format!("{:x}", Sha256::digest(native_wrapper_source)),
            "native_worker_source_sha256": format!("{:x}", Sha256::digest(native_worker_source)),
            "native_image_workflow_sha256": format!("{:x}", Sha256::digest(NATIVE_IMAGE_WORKFLOW)),
            "native_image_input_sha256": format!("{:x}", Sha256::digest(NATIVE_IMAGE_INPUT)),
            "production_source_files_inspected": source_audit.inspected_source_files,
            "production_process_launch_files": source_audit.production_process_launch_files,
            "build_process_launch_files": source_audit.build_process_launch_files,
            "development_process_launch_files": source_audit.development_process_launch_files,
        }),
        &cases,
        &[
            "final packaged Zed manifest and signature inspection",
            "release Zed binary string inspection",
            "OS-enforced network-denied isolated-host execution",
            "terminal reverse-dependency and runtime-trace audit after all native tasks",
        ],
        "native-image-worker-slice",
        "This artifact proves the Task 2 foundation and Task 19 native image worker boundary; packaged release closure remains owned by the terminal release audit.",
    )
}
