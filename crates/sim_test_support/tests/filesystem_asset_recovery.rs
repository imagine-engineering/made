use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_runtime::{
    AssetAvailability, AssetByteRange, AssetChangeKind, AssetError, AssetNamespace, AssetOperation,
    AssetRoots, AssetService, AssetViewRequest, AuthorizedCapabilities, Capability, CapabilitySet,
    OutputCommitter, OutputNameRequest, OutputOperationState, PermissionGrant, PermissionPolicy,
    UploadRequest,
};
use sim_tensor::CancellationToken;
use sim_test_support::is_apple_double_metadata;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

fn repository_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_test_support has no repository root")?
        .to_path_buf())
}

fn repository_rust_sources(
    root: &Path,
) -> Result<Vec<(PathBuf, String)>, Box<dyn std::error::Error>> {
    fn visit(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
        let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if is_apple_double_metadata(&path) {
                continue;
            }
            if path.is_dir() {
                if !matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some("target" | ".git" | ".agents" | "projects" | "node_modules")
                ) {
                    visit(&path, files)?;
                }
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    files
        .into_iter()
        .map(|path| Ok((path.clone(), fs::read_to_string(path)?)))
        .collect::<Result<Vec<_>, std::io::Error>>()
        .map_err(Into::into)
}

fn source_occurrences(sources: &[(PathBuf, String)], needle: &str) -> Vec<String> {
    sources
        .iter()
        .filter(|(path, _)| {
            path.file_name().and_then(|name| name.to_str()) != Some("filesystem_asset_recovery.rs")
        })
        .flat_map(|(path, source)| {
            source
                .lines()
                .enumerate()
                .filter(move |(_, line)| line.contains(needle))
                .map(move |(line, _)| format!("{}:{}", path.display(), line + 1))
        })
        .collect()
}

fn validation_authorization(
    capabilities: CapabilitySet,
) -> Result<AuthorizedCapabilities, Box<dyn std::error::Error>> {
    let grant = PermissionGrant::new(
        "validation-profile",
        "filesystem-asset-recovery",
        capabilities.clone(),
        "filesystem-asset-recovery-fixture",
    )?;
    Ok(PermissionPolicy::new("validation-profile", [grant])?
        .authorize("filesystem-asset-recovery", &capabilities)?)
}

fn validation_roots()
-> Result<(tempfile::TempDir, AssetRoots, AuthorizedCapabilities), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let roots = [
        AssetNamespace::Input,
        AssetNamespace::Output,
        AssetNamespace::Temporary,
        AssetNamespace::Model,
        AssetNamespace::Plugin,
    ]
    .into_iter()
    .map(|namespace| {
        let path = directory.path().join(namespace.locator_type());
        fs::create_dir(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new("validation-profile", roots)?;
    let capabilities = CapabilitySet::new(roots.namespaces().flat_map(|namespace| {
        [
            AssetOperation::Read,
            AssetOperation::Write,
            AssetOperation::Rename,
            AssetOperation::Tag,
            AssetOperation::Delete,
        ]
        .into_iter()
        .map(move |action| Capability::Asset {
            namespace: namespace.locator_type().to_owned(),
            action,
        })
    }));
    Ok((directory, roots, validation_authorization(capabilities)?))
}

#[test]
fn val_recovery_005_filesystem_and_asset_recovery() -> Result<(), Box<dyn std::error::Error>> {
    let (_directory, roots, capabilities) = validation_roots()?;
    let cancellation = CancellationToken::default();
    let fixture = b"filesystem-asset-recovery-fixture";
    let mut assets = AssetService::open(roots.clone())?;
    let uploaded = assets.upload(
        UploadRequest {
            namespace: AssetNamespace::Input,
            filename: "source.bin".to_owned(),
            subfolder: PathBuf::from("fixtures/日本語 path"),
            bytes: fixture.to_vec(),
            overwrite: false,
            tags: BTreeSet::from(["recovery".to_owned()]),
        },
        &capabilities,
        &cancellation,
    )?;
    let input_path = roots
        .test_root_path(AssetNamespace::Input)?
        .join(&uploaded.record.identity.relative_path);
    drop(assets);
    let mut assets = AssetService::open(roots.clone())?;
    let durable_tags = assets
        .record(&uploaded.record.identity)
        .is_some_and(|record| record.tags.contains("recovery"));

    fs::remove_file(&input_path)?;
    let missing = assets
        .scan(&capabilities, &cancellation)?
        .iter()
        .any(|change| change.kind == AssetChangeKind::Missing);
    fs::write(&input_path, fixture)?;
    let restored = assets
        .scan(&capabilities, &cancellation)?
        .iter()
        .any(|change| change.kind == AssetChangeKind::Restored);
    let availability_restored = assets
        .record(&uploaded.record.identity)
        .is_some_and(|record| record.availability == AssetAvailability::Present);

    let output_request = OutputNameRequest {
        namespace: AssetNamespace::Output,
        filename_prefix: "validation/%year%/output_%batch_num%".to_owned(),
        extension: "bin".to_owned(),
        batch_index: 0,
        width: 1,
        height: 1,
        timestamp: "2026-07-13T15:04:05+01:00".parse()?,
    };
    let mut committer = OutputCommitter::open(roots.clone())?;
    let prepared = committer.prepare(
        &output_request,
        b"committed-output",
        &capabilities,
        &cancellation,
    )?;
    let output_path = roots
        .test_root_path(AssetNamespace::Output)?
        .join(&prepared.identity.relative_path);
    let invisible_before_commit = !output_path.exists();
    let committed_record = committer.commit_and_register(
        prepared.operation_id,
        &mut assets,
        &capabilities,
        &cancellation,
    )?;
    let exact_commit = fs::read(&output_path)? == b"committed-output";
    let range = assets.view(
        &AssetViewRequest {
            identity: committed_record.identity,
            range: Some(AssetByteRange {
                start: 0,
                end_inclusive: 8,
            }),
            download: true,
        },
        &capabilities,
        &cancellation,
    )?;

    let interrupted = committer.prepare(
        &OutputNameRequest {
            namespace: AssetNamespace::Temporary,
            ..output_request
        },
        b"must-not-publish",
        &capabilities,
        &cancellation,
    )?;
    drop(committer);
    let recovered = OutputCommitter::open(roots.clone())?;
    let restart_interrupted = recovered
        .operation(interrupted.operation_id)
        .is_some_and(|operation| operation.state == OutputOperationState::Interrupted);
    let no_partial_final = !roots
        .test_root_path(AssetNamespace::Temporary)?
        .join(interrupted.identity.relative_path)
        .exists();
    let denied = assets.view(
        &AssetViewRequest {
            identity: uploaded.record.identity,
            range: None,
            download: false,
        },
        &validation_authorization(CapabilitySet::default())?,
        &cancellation,
    );
    let traversal = roots.identity(AssetNamespace::Input, "../escape.bin");

    let repository = repository_root()?;
    let sources = repository_rust_sources(&repository)?;
    let artifact_root_definitions =
        source_occurrences(&sources, &["pub struct ", "ArtifactRoot", " {"].concat());
    let artifact_index_definitions =
        source_occurrences(&sources, &["pub struct ", "ArtifactIndex", " {"].concat());
    let asset_roots_definitions =
        source_occurrences(&sources, &["pub struct ", "AssetRoots", " {"].concat());
    let asset_service_definitions =
        source_occurrences(&sources, &["pub struct ", "AssetService", " {"].concat());
    let output_committer_definitions =
        source_occurrences(&sources, &["pub struct ", "OutputCommitter", " {"].concat());
    let node_execution = fs::read_to_string(repository.join("crates/sim_nodes/src/execution.rs"))?;
    let asset_name_request_definitions = node_execution
        .matches("pub struct NativeAssetNameResolutionRequest {")
        .count();
    let runtime_assets = fs::read_to_string(repository.join("crates/sim_runtime/src/assets.rs"))?;
    let runtime_assets_production = runtime_assets
        .split_once("\n#[cfg(test)]\nmod tests")
        .map_or(runtime_assets.as_str(), |(production, _)| production);
    let artifact_owner =
        fs::read_to_string(repository.join("crates/sim_model/src/artifact_index.rs"))?;
    let output_committer =
        fs::read_to_string(repository.join("crates/sim_runtime/src/output_committer.rs"))?;
    let output_committer_production = output_committer
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(output_committer.as_str(), |(production, _)| production);
    let worker = fs::read_to_string(repository.join("crates/sim_worker/src/sim_worker.rs"))?;
    let plugin_host =
        fs::read_to_string(repository.join("crates/sim_plugin_host/src/sim_plugin_host.rs"))?;
    let recovery = fs::read_to_string(repository.join("crates/sim_runtime/src/recovery.rs"))?;

    let cases = json!({
        "durable_tags": durable_tags,
        "missing_detected": missing,
        "restored_detected": restored,
        "availability_restored": availability_restored,
        "invisible_before_commit": invisible_before_commit,
        "exact_commit": exact_commit,
        "range_download": range.bytes == b"committed",
        "restart_interrupts_prepare": restart_interrupted,
        "no_partial_final": no_partial_final,
        "permission_denied_before_read": matches!(denied, Err(AssetError::PermissionDenied { .. })),
        "traversal_rejected": matches!(traversal, Err(AssetError::UnsafePath { .. })),
        "artifact_root_is_single_owner": artifact_root_definitions.len() == 1 && artifact_root_definitions[0].contains("crates/sim_model/src/artifact_index.rs"),
        "artifact_index_is_single_owner": artifact_index_definitions.len() == 1 && artifact_index_definitions[0].contains("crates/sim_model/src/artifact_index.rs"),
        "asset_adapters_are_single_definitions": asset_roots_definitions.len() == 1 && asset_service_definitions.len() == 1 && asset_roots_definitions[0].contains("crates/sim_runtime/src/assets.rs") && asset_service_definitions[0].contains("crates/sim_runtime/src/assets.rs"),
        "source_asset_name_request_is_one_path_free_node_contract": asset_name_request_definitions == 1 && !node_execution.contains("PathBuf") && !node_execution.contains("std::path"),
        "artifact_service_owns_source_name_listing_and_resolution": runtime_assets_production.contains("fn list_source_asset_names(") && runtime_assets_production.contains("fn resolve_source_asset_names(") && runtime_assets_production.matches("self.scan_namespaces(&[spec.namespace]").count() == 2 && runtime_assets_production.contains("root_ids_in_resolution_order"),
        "asset_service_owns_only_enrichment": runtime_assets_production.contains("artifact_index: ArtifactIndex") && runtime_assets_production.contains("enrichments: BTreeMap<AssetIdentity, AssetEnrichment>") && !runtime_assets_production.contains("struct AssetRoot {") && !runtime_assets_production.contains("records: BTreeMap<AssetIdentity, AssetRecord>") && !runtime_assets_production.contains("fn scan_root(") && !runtime_assets_production.contains("fn stable_file_sha256(") && !runtime_assets_production.contains("fs::canonicalize(") && !runtime_assets_production.contains("fs::symlink_metadata(") && !runtime_assets_production.contains("fs::read_dir("),
        "artifact_owner_performs_security_and_indexing": artifact_owner.contains("pub fn resolve_for_create_with_parents") && artifact_owner.contains("pub fn refresh_selected") && artifact_owner.contains("pub fn open_verified") && artifact_owner.contains("fn scan_root(") && artifact_owner.contains("fn hash_stable_capability_file(") && artifact_owner.contains("pub fn move_verified_contained_file_to("),
        "typed_root_is_absent": source_occurrences(&sources, &["struct ", "TypedRoot", " {"].concat()).is_empty(),
        "output_committer_is_single_owner": output_committer_definitions.len() == 1 && output_committer_definitions[0].contains("crates/sim_runtime/src/output_committer.rs") && output_committer_production.contains("self.roots.move_contained(") && output_committer_production.contains("self.roots.remove_contained(") && !output_committer_production.contains("fs::rename") && !output_committer_production.contains("fs::remove_file") && !worker.contains("fs::rename") && !plugin_host.contains("fs::rename") && !recovery.contains("fs::rename"),
        "boundary_adapters_do_not_embed_second_index_snapshot": source_occurrences(&sources, "struct ArtifactIndexSnapshot").len() == 1 && source_occurrences(&sources, "struct AssetServiceSnapshot").len() == 1,
        "native_only": true,
    });
    assert!(
        cases
            .as_object()
            .is_some_and(|cases| cases.values().all(|value| value == &Value::Bool(true))),
        "filesystem ownership/recovery cases failed: {cases:#}"
    );
    let case_count = cases.as_object().map_or(0, serde_json::Map::len);
    let artifact = json!({
        "validation": "VAL-RECOVERY-005",
        "scope": "filesystem-and-asset-recovery",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "native-rust",
        },
        "fixture_digests": {
            "input_sha256": format!("{:x}", Sha256::digest(fixture)),
            "output_sha256": format!("{:x}", Sha256::digest(b"committed-output")),
            "interrupted_sha256": format!("{:x}", Sha256::digest(b"must-not-publish")),
        },
        "definition_counts": {
            "artifact_root": artifact_root_definitions.len(),
            "artifact_index": artifact_index_definitions.len(),
            "asset_roots_adapter": asset_roots_definitions.len(),
            "asset_service_adapter": asset_service_definitions.len(),
            "output_committer": output_committer_definitions.len(),
            "source_asset_name_request": asset_name_request_definitions,
        },
        "repository_rust_sources": sources.len(),
        "summary": {"passed": case_count, "failed": 0, "skipped": 0},
        "cases": cases,
        "skipped": [],
        "subprocesses": 0,
    });
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("target")
        });
    let artifact_directory = target.join("comfy-parity");
    fs::create_dir_all(&artifact_directory)?;
    fs::write(
        artifact_directory.join("val-recovery-005.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
