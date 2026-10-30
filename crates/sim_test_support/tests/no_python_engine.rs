use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use sim_runtime::{
    CURRENT_LEGACY_INSTALLATION_MIGRATION_VERSION, LegacyInstallationConversionOwner,
    LegacyInstallationField, LegacyInstallationImport, LegacyLifecycleAction,
    LegacyLifecycleEvidence, PluginPolicy, decode_legacy_installation_migration,
    migrate_legacy_installation,
};
use sim_test_support::rust_source_before_test_module;
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
            let production = rust_source_before_test_module(&source).to_owned();
            sources.push((relative, production));
        }
    }
    Ok(())
}

fn production_sources(workspace_root: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut sources = Vec::new();
    let crates_directory = workspace_root.join("crates");
    for entry in fs::read_dir(crates_directory)? {
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
        return Err(
            io::Error::other(format!("VAL-LEGACY-ENGINE-001 cases failed: {cases:?}")).into(),
        );
    }
    let output_directory = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"))
        .join("comfy-parity");
    fs::create_dir_all(&output_directory)?;
    let output = output_directory.join("val-legacy-engine-001.json");
    let temporary = output_directory.join("val-legacy-engine-001.json.tmp");
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let artifact = json!({
        "validation": "VAL-LEGACY-ENGINE-001",
        "schema_version": 1,
        "scope": "read-only-legacy-python-engine-evidence-and-canonical-native-owner-routing",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "network_requests": 0,
            "processes_started": 0,
            "filesystem_probes": 0,
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
            "scope": "bounded inactive installation evidence, secret removal delegation, refused Python/Git/Comfy lifecycle actions, disabled native settings projection, exact conversion owners, and production definition/call-site negative scan",
        },
    });
    fs::write(&temporary, serde_json::to_vec_pretty(&artifact)?)?;
    fs::rename(temporary, output)?;
    Ok(())
}

#[test]
fn val_legacy_engine_001() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let credential_sentinel = "task30-credential-must-not-persist";
    let settings_sentinel = "task30-settings-secret-must-not-persist";
    let lifecycle_sentinel = "task30-lifecycle-secret-must-not-persist";
    let input = LegacyInstallationImport {
        name: "Legacy Desktop Installation".into(),
        installation_location: Some("/legacy/ComfyUI".into()),
        model_roots: vec!["/legacy/ComfyUI/models".into()],
        workflow_stores: vec!["/legacy/ComfyUI/user/default/workflows".into()],
        output_stores: vec!["/legacy/ComfyUI/output".into()],
        extension_references: vec!["custom_nodes/legacy-node".into()],
        settings: BTreeMap::from([
            ("preview_method".into(), Value::String("auto".into())),
            (
                "nested".into(),
                json!({"api_token": settings_sentinel, "retained": 7}),
            ),
        ]),
        credentials: BTreeMap::from([(
            "manager_account".into(),
            Value::String(credential_sentinel.into()),
        )]),
        lifecycle: LegacyLifecycleEvidence {
            python_environment_managed: true,
            git_custom_nodes_managed: true,
            comfy_server_managed: true,
            was_running: true,
            automatic_updates_enabled: true,
            unknown_fields: BTreeMap::from([(
                "nested".into(),
                json!({"password": lifecycle_sentinel, "retained": true}),
            )]),
        },
        requested_lifecycle_actions: vec![
            LegacyLifecycleAction::Launch,
            LegacyLifecycleAction::Connect,
            LegacyLifecycleAction::Update,
            LegacyLifecycleAction::Delete,
            LegacyLifecycleAction::Reconfigure,
        ],
        unknown_fields: BTreeMap::from([("source_revision".into(), Value::from(30))]),
    };
    let migration_id = Uuid::from_u128(0x3000_0001);
    let native_profile_id = Uuid::from_u128(0x3000_0002);
    let migration = migrate_legacy_installation(input.clone(), migration_id, native_profile_id)?;
    let repeated = migrate_legacy_installation(input, migration_id, native_profile_id)?;
    let serialized = serde_json::to_vec(&migration)?;
    let decoded = decode_legacy_installation_migration(&serialized)?;
    let presentation = migration.presentation();
    let conversion_owners = presentation
        .conversion_steps
        .iter()
        .map(|step| (step.field, step.canonical_owner))
        .collect::<BTreeSet<_>>();

    let sources = production_sources(&workspace_root)?;
    let owner_path = "crates/sim_runtime/src/legacy_installations.rs";
    let owner_source = sources
        .iter()
        .find_map(|(path, source)| (path == owner_path).then_some(source))
        .ok_or("legacy installation owner source was not scanned")?;
    let migration_definition_paths = sources
        .iter()
        .filter(|(_, source)| source.contains("pub struct LegacyInstallationMigrationResult"))
        .map(|(path, _)| path.as_str())
        .collect::<Vec<_>>();
    let input_consumer_paths = sources
        .iter()
        .filter(|(path, source)| path != owner_path && source.contains("LegacyInstallationImport"))
        .map(|(path, _)| path.as_str())
        .collect::<Vec<_>>();
    let process_owner_paths = sources
        .iter()
        .filter(|(path, source)| {
            path.starts_with("crates/sim_")
                && (source.contains("std::process::Command::new")
                    || source.contains("smol::process::Command::new")
                    || source.contains("Command::new(") && source.contains("process::Command"))
        })
        .map(|(path, _)| path.as_str())
        .collect::<BTreeSet<_>>();
    let forbidden_lifecycle_markers = [
        "Command::new(\"python\")",
        "Command::new(\"python3\")",
        "Command::new(\"pip\")",
        ".arg(\"main.py",
        ".arg(\"ComfyUI",
    ];
    let forbidden_lifecycle_hits = sources
        .iter()
        .flat_map(|(path, source)| {
            forbidden_lifecycle_markers
                .iter()
                .filter(move |marker| source.contains(*marker))
                .map(move |marker| format!("{path}:{marker}"))
        })
        .collect::<Vec<_>>();
    let forbidden_custom_node_git_hits = sources
        .iter()
        .filter(|(path, source)| {
            path.starts_with("crates/sim_") && source.contains("Command::new(\"git\")")
        })
        .map(|(path, _)| path.as_str())
        .collect::<Vec<_>>();
    let forbidden_owner_apis = [
        "std::path",
        "PathBuf",
        "Path::",
        "std::fs",
        "fs::",
        "Command::new",
        "std::process",
        "TcpStream",
        "TcpListener",
        "HttpClient",
        "reqwest",
        "ArtifactRoot::",
        "AssetService::",
        "WorkflowFormatDocument::",
        "OutputCommitter::",
        "SettingsStore::",
        "RuntimeSupervisor::",
    ];
    let owner_api_hits = forbidden_owner_apis
        .iter()
        .filter(|marker| owner_source.contains(**marker))
        .copied()
        .collect::<Vec<_>>();

    let inactive = &migration.inactive_legacy_installation;
    let mut authority_tampered = serde_json::to_value(&migration)?;
    *authority_tampered
        .pointer_mut("/native_profile/provider_scope")
        .ok_or("missing native provider scope")? = Value::String("remote".into());
    let authority_tamper_rejected =
        decode_legacy_installation_migration(&serde_json::to_vec(&authority_tampered)?).is_err();
    let mut location_tampered = serde_json::to_value(&migration)?;
    *location_tampered
        .pointer_mut("/inactive_legacy_installation/installation_location_hint")
        .ok_or("missing installation location hint")? =
        Value::String("https://example.invalid/ComfyUI".into());
    let executable_location_tamper_rejected =
        decode_legacy_installation_migration(&serde_json::to_vec(&location_tampered)?).is_err();
    let mut cases = BTreeMap::new();
    cases.insert(
        "migration_is_deterministic_and_checked_on_decode",
        migration == repeated
            && migration == decoded
            && serde_json::to_vec(&repeated)? == serialized,
    );
    cases.insert(
        "legacy_installation_is_preserved_read_only",
        !inactive.active
            && inactive.read_only
            && inactive.installation_location_hint.as_deref() == Some("/legacy/ComfyUI")
            && inactive.model_roots == ["/legacy/ComfyUI/models"]
            && inactive.workflow_stores == ["/legacy/ComfyUI/user/default/workflows"]
            && inactive.output_stores == ["/legacy/ComfyUI/output"]
            && inactive.extension_references == ["custom_nodes/legacy-node"],
    );
    cases.insert(
        "all_requested_legacy_lifecycle_actions_are_refused",
        migration.refused_lifecycle_actions
            == [
                LegacyLifecycleAction::Launch,
                LegacyLifecycleAction::Connect,
                LegacyLifecycleAction::Update,
                LegacyLifecycleAction::Delete,
                LegacyLifecycleAction::Reconfigure,
            ]
            && presentation.refused_lifecycle_actions == migration.refused_lifecycle_actions,
    );
    cases.insert(
        "credentials_and_recursive_secret_fields_are_removed",
        migration.credentials_removed
            && migration.removed_secret_values == 3
            && [
                credential_sentinel,
                settings_sentinel,
                lifecycle_sentinel,
                "\"api_token\"",
                "\"password\"",
            ]
            .iter()
            .all(|secret| {
                !serialized
                    .windows(secret.len())
                    .any(|window| window == secret.as_bytes())
            })
            && inactive
                .settings
                .get("nested")
                .and_then(|value| value.get("retained"))
                .and_then(Value::as_u64)
                == Some(7),
    );
    cases.insert(
        "native_replacement_delegates_to_safe_settings_projection",
        migration.schema_version == CURRENT_LEGACY_INSTALLATION_MIGRATION_VERSION
            && migration.native_profile.id == native_profile_id
            && migration.native_profile.model_roots.is_empty()
            && !migration.native_profile.api_host.enabled
            && !migration.native_profile.api_host.allow_remote
            && migration.native_profile.plugin_policy == PluginPolicy::Disabled
            && owner_source.contains("NativeRuntimeProfile::disabled_migration_replacement"),
    );
    cases.insert(
        "decoded_evidence_cannot_override_native_authority_or_location",
        authority_tamper_rejected
            && executable_location_tamper_rejected
            && owner_source.contains("self.native_profile == expected_native_profile"),
    );
    cases.insert(
        "retained_fields_route_to_exact_canonical_owners",
        conversion_owners
            == BTreeSet::from([
                (
                    LegacyInstallationField::ModelRoots,
                    LegacyInstallationConversionOwner::ArtifactRootAndAssetService,
                ),
                (
                    LegacyInstallationField::WorkflowStores,
                    LegacyInstallationConversionOwner::WorkflowFormatDocument,
                ),
                (
                    LegacyInstallationField::OutputStores,
                    LegacyInstallationConversionOwner::OutputCommitterAndAssetService,
                ),
                (
                    LegacyInstallationField::Settings,
                    LegacyInstallationConversionOwner::SettingsStore,
                ),
                (
                    LegacyInstallationField::ExtensionReferences,
                    LegacyInstallationConversionOwner::LegacyMappingResolver,
                ),
            ])
            && presentation
                .conversion_steps
                .iter()
                .all(|step| step.requires_explicit_acceptance),
    );
    cases.insert(
        "legacy_evidence_reuses_the_canonical_secret_sanitizer",
        owner_source.contains("sanitize_legacy_fields(")
            && !owner_source.contains("fn sanitize_legacy_fields")
            && !owner_source.contains("fn is_secret_key"),
    );
    cases.insert(
        "legacy_installation_has_one_owner_and_no_production_consumer",
        migration_definition_paths == [owner_path] && input_consumer_paths.is_empty(),
    );
    cases.insert(
        "legacy_adapter_owns_no_foundational_service_or_effect",
        owner_api_hits.is_empty(),
    );
    cases.insert(
        "runtime_supervisor_remains_the_only_comfy_process_owner",
        process_owner_paths == BTreeSet::from(["crates/sim_runtime/src/runtime_supervisor.rs"]),
    );
    cases.insert(
        "production_python_git_and_comfy_lifecycle_scan_is_clean",
        !sources.is_empty()
            && forbidden_lifecycle_hits.is_empty()
            && forbidden_custom_node_git_hits.is_empty(),
    );

    let source_digests = [
        "crates/sim_runtime/src/legacy_connections.rs",
        owner_path,
        "crates/sim_runtime/src/runtime_supervisor.rs",
        "crates/sim_runtime/src/settings.rs",
    ]
    .into_iter()
    .map(|path| {
        let source = sources
            .iter()
            .find_map(|(candidate, source)| (candidate == path).then_some(source))
            .ok_or_else(|| format!("missing scanned source {path}"))?;
        Ok((
            path.to_owned(),
            format!("{:x}", Sha256::digest(source.as_bytes())),
        ))
    })
    .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
    write_artifact(&workspace_root, source_digests, &cases)
}
