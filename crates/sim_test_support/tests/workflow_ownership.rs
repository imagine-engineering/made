use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{MetadataDocument, MetadataLimits};
use sim_runtime::{
    ContentRevision, EmbeddedPrimary, GraphDocument, WorkflowAuthority, WorkflowFormatDocument,
    WorkflowSaveCoordinator, WorkflowStorageProvider, import_embedded_metadata,
};
use sim_types::{NonFiniteJsonKind, normalize_json_non_finite};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use uuid::Uuid;

fn repository_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_test_support has no repository root")?
        .to_path_buf())
}

fn rust_sources(root: &Path) -> Result<Vec<(PathBuf, String)>, Box<dyn std::error::Error>> {
    fn visit(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
        let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if sim_test_support::is_apple_double_metadata(&path) {
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
    let mut matches = Vec::new();
    for (path, source) in sources {
        if path.file_name().and_then(|name| name.to_str()) == Some("workflow_ownership.rs") {
            continue;
        }
        for (line_index, line) in source.lines().enumerate() {
            if line.contains(needle) {
                matches.push(format!("{}:{}", path.display(), line_index + 1));
            }
        }
    }
    matches
}

fn file_sha256(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn glb_with_json(json: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut json = json.to_vec();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total = 20usize.checked_add(json.len()).ok_or("GLB size overflow")?;
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(b"glTF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(total)?.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(json.len())?.to_le_bytes());
    bytes.extend_from_slice(b"JSON");
    bytes.extend_from_slice(&json);
    Ok(bytes)
}

#[test]
fn val_workflow_ownership_001() -> Result<(), Box<dyn std::error::Error>> {
    let root = repository_root()?;
    let sources = rust_sources(&root)?;
    let workflow_definitions = source_occurrences(
        &sources,
        &["pub struct ", "WorkflowFormatDocument"].concat(),
    );
    let metadata_definitions =
        source_occurrences(&sources, &["pub struct ", "MetadataDocument"].concat());
    let normalizer_definitions = source_occurrences(
        &sources,
        &["pub fn ", "normalize_json_non_finite", "("].concat(),
    );
    let save_definitions = source_occurrences(
        &sources,
        &["pub struct ", "WorkflowSaveCoordinator"].concat(),
    );
    let superseded_workflow_definitions =
        source_occurrences(&sources, &["pub struct ", "WorkflowDocument"].concat());
    let superseded_publication_definitions =
        source_occurrences(&sources, &["enum ", "PublicationState"].concat());
    let copied_normalizer_definitions =
        source_occurrences(&sources, &["fn ", "normalize_non_finite", "("].concat());

    let workflow_source = br#"{"version":0.4,"last_node_id":0,"last_link_id":0,"nodes":[],"links":[],"groups":[],"config":{},"extra":{"future":NaN}}"#;
    let (normalized, tokens) = normalize_json_non_finite(workflow_source);
    let normalized_value: Value = serde_json::from_slice(&normalized)?;
    let normalizer_semantics = normalized_value["extra"]["future"].is_null()
        && tokens.len() == 1
        && tokens[0].kind == NonFiniteJsonKind::Nan
        && workflow_source.get(
            tokens[0].byte_offset
                ..tokens[0]
                    .byte_offset
                    .checked_add(tokens[0].source_length)
                    .ok_or("non-finite token range overflow")?,
        ) == Some(b"NaN".as_slice());

    let workflow = WorkflowFormatDocument::parse(workflow_source)?;
    let graph = GraphDocument::from_workflow(&workflow)?;
    let workflow_adapter_semantics = workflow.original_bytes() == workflow_source
        && workflow.non_finite_tokens() == tokens
        && graph.root.nodes.is_empty()
        && graph.root.links.is_empty();

    let svg = br#"<svg><metadata><![CDATA[{"workflow":{"version":0.4,"nodes":[],"links":[],"groups":[],"config":{},"extra":{"future":NaN}}}]]></metadata></svg>"#;
    let svg_document =
        MetadataDocument::parse(svg, Some("fixture.svg"), None, MetadataLimits::default())?;
    let svg_import = import_embedded_metadata(&svg_document)?;
    let glb = glb_with_json(
        br#"{"asset":{"version":"2.0","extras":{"workflow":{"version":0.4,"nodes":[],"links":[],"groups":[],"config":{},"extra":{"future":Infinity}}}}}"#,
    )?;
    let glb_document =
        MetadataDocument::parse(&glb, Some("fixture.glb"), None, MetadataLimits::default())?;
    let glb_import = import_embedded_metadata(&glb_document)?;
    let embedded_adapter_semantics = !svg_import.executes_on_import
        && !glb_import.executes_on_import
        && matches!(svg_import.primary, EmbeddedPrimary::Workflow(_))
        && matches!(glb_import.primary, EmbeddedPrimary::Workflow(_))
        && svg_document.original_bytes() == svg
        && glb_document.original_bytes() == glb;

    let base = b"base".to_vec();
    let mut conflict = WorkflowSaveCoordinator::new(
        "workflow.json",
        WorkflowStorageProvider::LocalFile,
        base.clone(),
    )?;
    conflict.edit(b"local".to_vec())?;
    conflict.observe_external_change(b"external".to_vec())?;
    let conflict_semantics = conflict.authority() == WorkflowAuthority::Conflict
        && conflict.comparison().base == base
        && conflict.comparison().local == b"local"
        && conflict.comparison().external == Some(b"external".as_slice());
    conflict.keep_local()?;

    let mut committed =
        WorkflowSaveCoordinator::new("workflow.json", WorkflowStorageProvider::LocalFile, base)?;
    committed.edit(b"committed".to_vec())?;
    let operation_id = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1401);
    let observed_revision = committed.base().revision.clone();
    let prepared = committed.prepare_save(
        operation_id,
        observed_revision.clone(),
        "renamed.json",
        false,
    )?;
    let replayed = committed.prepare_save(
        operation_id,
        observed_revision.clone(),
        "renamed.json",
        false,
    )?;
    let committed_revision = ContentRevision::from_bytes(&prepared.bytes);
    committed.commit_save(operation_id, observed_revision, committed_revision)?;
    committed.switch_provider_after_committed_save(WorkflowStorageProvider::Provider {
        identifier: "provider.fixture".to_owned(),
    })?;
    let commit_semantics = prepared == replayed
        && committed.authority() == WorkflowAuthority::InSync
        && committed.document_identity() == "renamed.json"
        && matches!(
            committed.provider(),
            WorkflowStorageProvider::Provider { identifier } if identifier == "provider.fixture"
        );

    let mut detached = WorkflowSaveCoordinator::new(
        "workflow.json",
        WorkflowStorageProvider::LocalFile,
        b"draft".to_vec(),
    )?;
    detached.detach_local_file_to_draft("draft-identity")?;
    let detached = WorkflowSaveCoordinator::decode(&detached.encode()?)?;
    let detach_semantics = detached.authority() == WorkflowAuthority::LocalDirty
        && detached.provider() == &WorkflowStorageProvider::Draft
        && detached.document_identity() == "draft-identity"
        && detached.external().is_none()
        && detached.prepared().is_none();

    let workflow_formats_source =
        fs::read_to_string(root.join("crates/sim_runtime/src/workflow_formats.rs"))?;
    let coordinator_start = workflow_formats_source
        .find(&["pub struct ", "WorkflowSaveCoordinator", " {"].concat())
        .ok_or("WorkflowSaveCoordinator definition is missing")?;
    let coordinator_body = workflow_formats_source
        .get(coordinator_start..)
        .and_then(|source| source.split_once("\n}\n\n#[derive").map(|(body, _)| body))
        .ok_or("WorkflowSaveCoordinator definition is incomplete")?;
    let coordinator_fields_are_private = coordinator_body
        .lines()
        .skip(1)
        .filter(|line| !line.trim_start().starts_with("#"))
        .all(|line| !line.trim_start().starts_with("pub "));
    let direct_ui_mutations_are_absent = [
        "save_coordinator.authority =",
        "save_coordinator.provider =",
        "save_coordinator.document_identity =",
        "save_coordinator.base =",
        "save_coordinator.external =",
        "save_coordinator.prepared =",
    ]
    .into_iter()
    .all(|needle| source_occurrences(&sources, needle).is_empty());

    let normalizer_calls = source_occurrences(&sources, "normalize_json_non_finite");
    let cases = BTreeMap::from([
        (
            "embedded_metadata_is_single_owner",
            metadata_definitions.len() == 1
                && metadata_definitions[0].contains("crates/sim_media/src/metadata.rs")
                && source_occurrences(&sources, "fn parse_svg(bytes:").len() == 1
                && source_occurrences(&sources, "fn parse_glb(bytes:").len() == 1
                && source_occurrences(&sources, "fn write_glb(bytes:").len() == 1,
        ),
        (
            "embedded_metadata_preserves_no_execute_semantics",
            embedded_adapter_semantics,
        ),
        (
            "non_finite_compatibility_is_single_owner",
            normalizer_definitions.len() == 1
                && normalizer_definitions[0].contains("crates/sim_types/src/json_compat.rs")
                && copied_normalizer_definitions.is_empty()
                && normalizer_calls.iter().any(|location| {
                    location.contains("crates/sim_runtime/src/workflow_formats.rs")
                })
                && normalizer_calls
                    .iter()
                    .any(|location| location.contains("crates/sim_media/src/metadata.rs")),
        ),
        (
            "non_finite_compatibility_preserves_semantics",
            normalizer_semantics,
        ),
        (
            "save_conflict_reducer_preserves_semantics",
            conflict_semantics,
        ),
        (
            "save_commit_replay_and_provider_transition",
            commit_semantics,
        ),
        ("save_detach_transition_survives_restart", detach_semantics),
        (
            "save_state_is_private_and_ui_uses_transitions",
            coordinator_fields_are_private && direct_ui_mutations_are_absent,
        ),
        (
            "superseded_workflow_parser_is_absent",
            superseded_workflow_definitions.is_empty(),
        ),
        (
            "superseded_publication_state_is_absent",
            superseded_publication_definitions.is_empty(),
        ),
        (
            "workflow_document_adapter_preserves_semantics",
            workflow_adapter_semantics,
        ),
        (
            "workflow_document_is_single_owner",
            workflow_definitions.len() == 1
                && workflow_definitions[0].contains("crates/sim_runtime/src/workflow_formats.rs"),
        ),
        (
            "workflow_save_is_single_owner",
            save_definitions.len() == 1
                && save_definitions[0].contains("crates/sim_runtime/src/workflow_formats.rs"),
        ),
    ]);
    assert!(
        cases.values().all(|passed| *passed),
        "workflow ownership cases failed: {cases:#?}"
    );

    let fixture_paths = [
        ".agents/specs/comfy-parity/ownership-policy.json",
        "crates/sim_types/src/json_compat.rs",
        "crates/sim_types/src/workflow.rs",
        "crates/sim_runtime/src/workflow_formats.rs",
        "crates/sim_runtime/src/graph.rs",
        "crates/sim_media/src/metadata.rs",
        "crates/sim_media/src/png.rs",
        "crates/sim_ui/src/graph.rs",
        "crates/sim_ui/src/workflow_item.rs",
    ];
    let fixture_digests = fixture_paths
        .into_iter()
        .map(|relative| Ok((relative.to_owned(), file_sha256(&root.join(relative))?)))
        .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
    let artifact = json!({
        "validation": "VAL-WORKFLOW-OWNERSHIP-001",
        "scope": "workflow-metadata-save-authoritative-ownership",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "native-rust",
            "network_requests": 0,
            "external_processes": 0,
        },
        "fixture_digests": fixture_digests,
        "definition_counts": {
            "workflow_document": workflow_definitions.len(),
            "metadata_document": metadata_definitions.len(),
            "non_finite_normalizer": normalizer_definitions.len(),
            "workflow_save_coordinator": save_definitions.len(),
            "superseded_workflow_document": superseded_workflow_definitions.len(),
            "superseded_publication_state": superseded_publication_definitions.len(),
            "copied_non_finite_normalizer": copied_normalizer_definitions.len(),
        },
        "cases": cases,
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
        },
    });
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    let artifact_path = root.join("target/comfy-parity/val-workflow-ownership-001.json");
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(artifact_path, bytes)?;
    Ok(())
}
