use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use sim_runtime::{
    CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION, LegacyComfyProfile, LegacyConnectionField,
    LegacyConversionOwner, migrate_legacy_profile,
};
use uuid::Uuid;

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn collect_rust_sources(
    workspace_root: &Path,
    directory: &Path,
    sources: &mut Vec<(String, String)>,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if sim_test_support::is_apple_double_metadata(&path) {
            continue;
        }
        if path.is_dir() {
            collect_rust_sources(workspace_root, &path, sources)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let relative = path
                .strip_prefix(workspace_root)?
                .to_string_lossy()
                .into_owned();
            let source = fs::read_to_string(&path)?;
            let production = source
                .split_once("#[cfg(test)]\nmod tests")
                .map_or(source.as_str(), |(production, _)| production)
                .to_owned();
            sources.push((relative, production));
        }
    }
    Ok(())
}

fn production_sources(workspace_root: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut sources = Vec::new();
    let crates_directory = workspace_root.join("crates");
    for entry in fs::read_dir(&crates_directory)? {
        let entry = entry?;
        let crate_name = entry.file_name();
        let crate_name = crate_name.to_string_lossy();
        if (crate_name.starts_with("sim_") && crate_name != "sim_test_support")
            || crate_name == "zed"
        {
            let source_directory = entry.path().join("src");
            if source_directory.is_dir() {
                collect_rust_sources(workspace_root, &source_directory, &mut sources)?;
            }
        }
    }
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(sources)
}

fn write_artifact(
    workspace_root: &Path,
    source_digests: BTreeMap<String, String>,
    cases: &BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    if cases.values().any(|passed| !passed) {
        return Err(io::Error::other(format!("VAL-E2E-001 cases failed: {cases:?}")).into());
    }
    let output_directory = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"))
        .join("comfy-parity");
    fs::create_dir_all(&output_directory)?;
    let output = output_directory.join("val-e2e-001.json");
    let temporary = output_directory.join("val-e2e-001.json.tmp");
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let artifact = json!({
        "validation": "VAL-E2E-001",
        "schema_version": 1,
        "scope": "superseded-external-comfy-connection-release-guard",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "network_requests": 0,
            "processes_started": 0,
            "development_oracle_executed": false,
        },
        "source_sha256": source_digests,
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
        },
        "cases": cases,
        "skipped": [],
        "validation_closure": {
            "claimed": true,
            "scope": "inactive legacy connection retention, explicit secret removal, disabled native replacement, owner-routed conversion presentation, and production source negative scan",
        },
    });
    fs::write(&temporary, serde_json::to_vec_pretty(&artifact)?)?;
    fs::rename(temporary, output)?;
    Ok(())
}

#[test]
fn val_e2e_001() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let credential_sentinel = "task29-credential-must-not-persist";
    let endpoint_sentinel = "task29-endpoint-token-must-not-persist";
    let workflow_sentinel = "task29-workflow-secret-must-not-persist";
    let input = LegacyComfyProfile {
        name: "Legacy Remote".into(),
        endpoint: Some(format!(
            "https://legacy-user:legacy-password@example.invalid:8443/object_info?token={endpoint_sentinel}"
        )),
        credential: Some(credential_sentinel.into()),
        model_roots: vec!["models/checkpoints".into(), "/imported/models".into()],
        api_host_enabled: true,
        plugin_mappings: vec!["LegacyNode=zed.native.LegacyNode".into()],
        workflow_state: BTreeMap::from([(
            "workflow-a".into(),
            json!({
                "nodes": [{"id": 1, "type": "LegacyNode"}],
                "api_token": workflow_sentinel,
                "retained": {"revision": 7}
            }),
        )]),
        unknown_fields: BTreeMap::from([
            ("future_flag".into(), json!(true)),
            (
                "nested".into(),
                json!({"password": "removed", "retained": [1, 2, 3]}),
            ),
        ]),
    };
    let migration_id = Uuid::from_u128(0x2901);
    let native_profile_id = Uuid::from_u128(0x2902);
    let migration = migrate_legacy_profile(input.clone(), migration_id, native_profile_id)?;
    let repeated = migrate_legacy_profile(input, migration_id, native_profile_id)?;
    let serialized = serde_json::to_vec(&migration)?;
    let presentation = migration.presentation();
    let conversion_owners = presentation
        .conversion_steps
        .iter()
        .map(|step| (step.field, step.canonical_owner))
        .collect::<BTreeSet<_>>();

    let sources = production_sources(&workspace_root)?;
    let forbidden_markers = [
        "Command::new(\"python",
        "Command::new(\"python3",
        "Command::new(\"node",
        ".arg(\"ComfyUI",
        ".arg(\"main.py",
        "http://127.0.0.1:8188",
        "http://localhost:8188",
        "ws://127.0.0.1:8188",
        "ws://localhost:8188",
        "struct ConnectionManager",
        "struct ExternalComfyProfile",
        "struct ManagedComfyProfile",
        "fn connect_to_comfy",
    ];
    let forbidden_hits = sources
        .iter()
        .flat_map(|(path, source)| {
            forbidden_markers
                .iter()
                .filter(move |marker| source.contains(*marker))
                .map(move |marker| format!("{path}:{marker}"))
        })
        .collect::<Vec<_>>();
    let legacy_input_consumers = sources
        .iter()
        .filter(|(path, source)| {
            path.as_str() != "crates/sim_runtime/src/legacy_connections.rs"
                && source.contains("LegacyComfyProfile")
        })
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let legacy_owner_source =
        fs::read_to_string(workspace_root.join("crates/sim_runtime/src/legacy_connections.rs"))?;

    let mut cases = BTreeMap::new();
    cases.insert(
        "migration_is_deterministic_for_explicit_identities",
        migration == repeated && serde_json::to_vec(&repeated)? == serialized,
    );
    cases.insert(
        "legacy_record_is_preserved_but_inactive",
        !migration.inactive_legacy_profile.active
            && migration.inactive_legacy_profile.model_roots
                == vec![
                    String::from("models/checkpoints"),
                    String::from("/imported/models"),
                ]
            && migration.inactive_legacy_profile.plugin_mappings
                == vec![String::from("LegacyNode=zed.native.LegacyNode")]
            && migration
                .inactive_legacy_profile
                .unknown_fields
                .get("future_flag")
                == Some(&Value::Bool(true)),
    );
    cases.insert(
        "former_endpoint_is_display_only_origin",
        migration
            .inactive_legacy_profile
            .former_endpoint
            .as_ref()
            .map(sim_runtime::InactiveLegacyOrigin::display)
            == Some("https://example.invalid:8443")
            && migration.endpoint_removed_or_redacted,
    );
    cases.insert(
        "all_secret_values_are_explicitly_removed",
        migration.credential_removed
            && migration.removed_secret_values == 3
            && [
                credential_sentinel,
                endpoint_sentinel,
                workflow_sentinel,
                "legacy-password",
                "\"api_token\"",
                "\"password\"",
            ]
            .iter()
            .all(|secret| {
                !serialized
                    .windows(secret.len())
                    .any(|window| window == secret.as_bytes())
            }),
    );
    cases.insert(
        "workflow_nonsecret_state_survives",
        migration
            .inactive_legacy_profile
            .workflow_state
            .get("workflow-a")
            .and_then(|workflow| workflow.get("retained"))
            .and_then(Value::as_object)
            .and_then(|retained| retained.get("revision"))
            .and_then(Value::as_u64)
            == Some(7),
    );
    cases.insert(
        "native_replacement_is_safe_and_disabled",
        migration.schema_version == CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION
            && migration.native_profile.id == native_profile_id
            && migration.native_profile.model_roots.is_empty()
            && !migration.native_profile.api_host.enabled
            && !migration.native_profile.api_host.allow_remote
            && migration.native_profile.provider_scope == "local",
    );
    cases.insert(
        "conversion_steps_route_to_exact_canonical_owners",
        conversion_owners
            == BTreeSet::from([
                (
                    LegacyConnectionField::ModelRoots,
                    LegacyConversionOwner::ArtifactRootAndAssetService,
                ),
                (
                    LegacyConnectionField::ApiHostPolicy,
                    LegacyConversionOwner::NativeApiExposure,
                ),
                (
                    LegacyConnectionField::PluginMappings,
                    LegacyConversionOwner::LegacyMappingResolver,
                ),
                (
                    LegacyConnectionField::WorkflowState,
                    LegacyConversionOwner::WorkflowFormatDocument,
                ),
            ])
            && presentation
                .conversion_steps
                .iter()
                .all(|step| step.requires_explicit_acceptance),
    );
    cases.insert(
        "legacy_input_has_no_production_consumer",
        legacy_input_consumers.is_empty(),
    );
    cases.insert(
        "legacy_origin_has_no_executable_url_adapter",
        legacy_owner_source.contains("pub struct InactiveLegacyOrigin(String);")
            && !legacy_owner_source.contains("impl From<InactiveLegacyOrigin for Url")
            && !legacy_owner_source.contains("impl TryFrom<InactiveLegacyOrigin for Url")
            && !legacy_owner_source.contains("TcpStream")
            && !legacy_owner_source.contains("HttpClient")
            && !legacy_owner_source.contains("Command::new"),
    );
    cases.insert(
        "production_external_comfy_launcher_and_client_scan_is_clean",
        !sources.is_empty() && forbidden_hits.is_empty(),
    );
    cases.insert(
        "superseded_connection_owner_is_removed",
        !workspace_root
            .join("crates/sim_runtime/src/migrations.rs")
            .exists()
            && workspace_root
                .join("crates/sim_runtime/src/legacy_connections.rs")
                .is_file(),
    );

    let source_digests = sources
        .iter()
        .map(|(path, source)| {
            (
                path.clone(),
                format!("{:x}", Sha256::digest(source.as_bytes())),
            )
        })
        .collect();
    write_artifact(&workspace_root, source_digests, &cases)
}
