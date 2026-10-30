use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde_json::json;
use sha2::{Digest, Sha256};
use sim_api::{
    HttpCapabilities, HttpLimits, NativeApiHost, NativeApiServer, NativeApiServerConfig,
    NativeHeadlessCancellation, NativeHttpServices, NativeServiceRequest, NativeServiceResponse,
    WebSocketLimits,
    security::{
        ApiSecurityConfig, ApiSecurityError, IdempotencySnapshot, IdempotencySnapshotStore,
    },
};
use sim_plugin_host::check_plugin_cancellation;
use sim_plugin_sdk::InvocationError;
use sim_tensor::TensorError;
use sim_types::{
    AttemptId, CancellationError, CancellationToken, MAX_ENCODED_PREVIEW_BYTES,
    MAX_WORKER_FRAME_BYTES, ProfileId, RequestId, WORKER_PROTOCOL_VERSION, WorkerEnvelope,
    WorkerId, WorkerMessage, WorkerProtocolError, decode_worker_frame, encode_worker_frame,
};
use uuid::Uuid;

const CANONICAL_DEFINITION: &str = "crates/sim_types/src/cancellation.rs";
const OWNERSHIP_CATALOG: &str = ".agents/specs/comfy-parity/catalogs/authoritative-ownership.csv";
const DEVELOPMENT_ORACLE: &str = "projects/comfy/comfy-cli/comfy_cli/cancellation.py";
const FIXTURE_SOURCES: &[&str] = &[
    OWNERSHIP_CATALOG,
    CANONICAL_DEFINITION,
    "crates/sim_types/src/worker_protocol.rs",
    "crates/sim_tensor/src/operation.rs",
    "crates/sim_tensor/src/sim_tensor.rs",
    "crates/sim_runtime/src/native_execution_controller.rs",
    "crates/sim_worker/src/sim_worker.rs",
    "crates/sim_plugin_host/src/capabilities.rs",
    "crates/sim_api/src/headless.rs",
    "crates/sim_api/src/transport.rs",
    "crates/sim_test_support/tests/cancellation_ownership.rs",
];

#[derive(Default)]
struct MemorySnapshotStore {
    snapshot: Mutex<Option<IdempotencySnapshot>>,
}

impl IdempotencySnapshotStore for MemorySnapshotStore {
    fn load(&self) -> Result<Option<IdempotencySnapshot>, ApiSecurityError> {
        self.snapshot
            .lock()
            .map(|snapshot| snapshot.clone())
            .map_err(|_| ApiSecurityError::Persistence("validation snapshot lock failed".into()))
    }

    fn save(&self, snapshot: &IdempotencySnapshot) -> Result<(), ApiSecurityError> {
        self.snapshot
            .lock()
            .map(|mut stored| *stored = Some(snapshot.clone()))
            .map_err(|_| ApiSecurityError::Persistence("validation snapshot lock failed".into()))
    }
}

struct ProbeServices;

impl NativeHttpServices for ProbeServices {
    fn dispatch(
        &self,
        request: NativeServiceRequest,
    ) -> Result<NativeServiceResponse, sim_api::NativeServiceError> {
        Ok(NativeServiceResponse::json(
            200,
            json!({"feature_id": request.route.canonical_feature_id, "native": true}),
        ))
    }
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

fn envelope(message: WorkerMessage) -> WorkerEnvelope {
    WorkerEnvelope {
        version: WORKER_PROTOCOL_VERSION,
        profile_id: ProfileId(Uuid::nil()),
        worker_id: WorkerId(Uuid::nil()),
        request_id: RequestId(Uuid::nil()),
        prompt_id: None,
        attempt_id: Some(AttemptId(Uuid::nil())),
        sequence: 1,
        registry_version: "val-cancel-001".to_owned(),
        message,
        extensions: BTreeMap::new(),
    }
}

fn parse_csv_record(line: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut characters = line.chars().peekable();
    let mut quoted = false;
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                field.push('"');
                characters.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unterminated CSV field").into());
    }
    fields.push(field);
    Ok(fields)
}

fn cancellation_catalog_row(
    workspace_root: &Path,
) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let catalog = fs::read_to_string(workspace_root.join(OWNERSHIP_CATALOG))?;
    let mut lines = catalog.lines();
    let header = parse_csv_record(lines.next().ok_or("ownership catalog is empty")?)?;
    for line in lines {
        let values = parse_csv_record(line)?;
        if values.len() != header.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ownership catalog row width does not match its header",
            )
            .into());
        }
        let row: BTreeMap<_, _> = header.iter().cloned().zip(values).collect();
        if row
            .get("concern")
            .is_some_and(|value| value == "cancellation")
        {
            return Ok(row);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "ownership catalog has no cancellation row",
    )
    .into())
}

fn collect_rust_sources(directory: &Path, sources: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if sim_test_support::is_apple_double_metadata(&path) {
            continue;
        }
        if path.is_dir() {
            if matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some(".git" | "target" | "node_modules" | ".venv")
            ) {
                continue;
            }
            collect_rust_sources(&path, sources)?;
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            sources.push(path);
        }
    }
    Ok(())
}

fn raw_string_open(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut cursor = start;
    if matches!(bytes.get(cursor), Some(b'b' | b'c')) {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hashes_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    (bytes.get(cursor) == Some(&b'"')).then_some((cursor + 1, cursor - hashes_start))
}

fn rust_identifier_tokens(source: &str) -> Vec<(&str, usize)> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut line = 1;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\n' {
            line += 1;
            cursor += 1;
            continue;
        }
        if bytes[cursor..].starts_with(b"//") {
            cursor += 2;
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            continue;
        }
        if bytes[cursor..].starts_with(b"/*") {
            cursor += 2;
            let mut depth = 1_u64;
            while cursor < bytes.len() && depth != 0 {
                if bytes[cursor..].starts_with(b"/*") {
                    depth += 1;
                    cursor += 2;
                } else if bytes[cursor..].starts_with(b"*/") {
                    depth -= 1;
                    cursor += 2;
                } else {
                    if bytes[cursor] == b'\n' {
                        line += 1;
                    }
                    cursor += 1;
                }
            }
            continue;
        }
        if let Some((content_start, hashes)) = raw_string_open(bytes, cursor) {
            cursor = content_start;
            while cursor < bytes.len() {
                if bytes[cursor] == b'\n' {
                    line += 1;
                    cursor += 1;
                    continue;
                }
                if bytes[cursor] == b'"'
                    && bytes
                        .get(cursor + 1..cursor + 1 + hashes)
                        .is_some_and(|suffix| suffix.iter().all(|byte| *byte == b'#'))
                {
                    cursor += 1 + hashes;
                    break;
                }
                cursor += 1;
            }
            continue;
        }
        let string_prefix =
            matches!(bytes.get(cursor), Some(b'b' | b'c')) && bytes.get(cursor + 1) == Some(&b'"');
        if bytes[cursor] == b'"' || string_prefix {
            cursor += usize::from(string_prefix) + 1;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' => cursor = (cursor + 2).min(bytes.len()),
                    b'"' => {
                        cursor += 1;
                        break;
                    }
                    b'\n' => {
                        line += 1;
                        cursor += 1;
                    }
                    _ => cursor += 1,
                }
            }
            continue;
        }
        if bytes[cursor] == b'\'' {
            let mut end = cursor + 1;
            if bytes.get(end) == Some(&b'\\') {
                end = (end + 2).min(bytes.len());
            } else if let Some(character) = source.get(end..).and_then(|value| value.chars().next())
            {
                end += character.len_utf8();
            }
            if bytes.get(end) == Some(&b'\'') {
                cursor = end + 1;
                continue;
            }
        }
        let raw_identifier = bytes[cursor..].starts_with(b"r#")
            && bytes
                .get(cursor + 2)
                .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_');
        let identifier_start = cursor + if raw_identifier { 2 } else { 0 };
        if bytes
            .get(identifier_start)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        {
            let mut end = identifier_start + 1;
            while bytes
                .get(end)
                .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            {
                end += 1;
            }
            if let Some(identifier) = source.get(identifier_start..end) {
                tokens.push((identifier, line));
            }
            cursor = end;
            continue;
        }
        cursor += 1;
    }
    tokens
}

fn cancellation_definition_lines(source: &str) -> Vec<usize> {
    rust_identifier_tokens(source)
        .windows(2)
        .filter_map(|tokens| {
            (tokens[0].0 == "struct" && tokens[1].0 == "CancellationToken").then_some(tokens[0].1)
        })
        .collect()
}

fn cancellation_definitions(workspace_root: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut sources = Vec::new();
    collect_rust_sources(workspace_root, &mut sources)?;
    sources.sort();
    let mut definitions = Vec::new();
    for path in sources {
        let source = fs::read_to_string(&path)?;
        for line in cancellation_definition_lines(&source) {
            definitions.push(format!(
                "{}:{line}",
                path.strip_prefix(workspace_root)?.display(),
            ));
        }
    }
    Ok(definitions)
}

#[test]
fn cancellation_definition_scanner_ignores_non_executable_text() {
    let source = concat!(
        "// pub struct ",
        "CancellationToken\n",
        "const TEXT: &str = \"pub struct ",
        "CancellationToken\";\n",
        "const RAW: &str = r#\"pub struct ",
        "CancellationToken\"#;\n",
        "/* pub struct ",
        "CancellationToken */\n",
        "pub\nstruct ",
        "CancellationToken { cancelled: bool }\n",
    );
    assert_eq!(cancellation_definition_lines(source), vec![6]);
}

fn fixture_digests(
    workspace_root: &Path,
) -> Result<BTreeMap<&'static str, String>, Box<dyn Error>> {
    let mut digests = BTreeMap::new();
    for relative in FIXTURE_SOURCES {
        let bytes = fs::read(workspace_root.join(relative))?;
        digests.insert(*relative, format!("{:x}", Sha256::digest(bytes)));
    }
    Ok(digests)
}

#[test]
fn val_cancel_001_canonical_cancellation_ownership() -> Result<(), Box<dyn Error>> {
    let workspace_root = workspace_root()?;
    let artifact_directory = target_directory(&workspace_root).join("comfy-parity");
    let artifact_path = artifact_directory.join("val-cancel-001.json");
    match fs::remove_file(&artifact_path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let mut cases = BTreeMap::new();

    let token = CancellationToken::default();
    let clone = token.clone();
    cases.insert("canonical_clone_starts_active", clone.check().is_ok());
    cases.insert("canonical_first_cancel_wins", token.cancel());
    cases.insert("canonical_second_cancel_is_monotonic", !clone.cancel());
    cases.insert("canonical_clone_observes_cancel", clone.is_cancelled());
    cases.insert(
        "tensor_error_adapter",
        matches!(TensorError::from(CancellationError), TensorError::Cancelled),
    );

    let plugin_token = CancellationToken::default();
    plugin_token.cancel();
    cases.insert(
        "plugin_error_adapter",
        matches!(
            check_plugin_cancellation(&plugin_token),
            Err(InvocationError::Cancelled)
        ),
    );

    let headless = NativeHeadlessCancellation::default();
    cases.insert("headless_first_reason_wins", headless.cancel("first")?);
    cases.insert(
        "headless_rejects_second_reason",
        !headless.cancel("second")?,
    );
    cases.insert(
        "headless_reason_is_stable",
        headless.reason()?.as_deref() == Some("first"),
    );

    let host = Arc::new(NativeApiHost::new(
        "val-cancel-001",
        Arc::new(ProbeServices),
        HttpLimits::default(),
        HttpCapabilities::default(),
        WebSocketLimits::default(),
        ApiSecurityConfig::loopback(),
        Arc::new(sim_runtime::PermissionPolicy::native_runtime_services(
            "val-cancel-001",
        )?),
        Arc::new(MemorySnapshotStore::default()),
    )?);
    let server = NativeApiServer::start(
        host,
        NativeApiServerConfig::new(SocketAddr::from(([127, 0, 0, 1], 0))),
    )?;
    server.shutdown()?;
    cases.insert("native_transport_shutdown_converges", true);

    let worker_cancel = envelope(WorkerMessage::Cancel {
        reason: "operator".to_owned(),
    });
    let worker_token = CancellationToken::default();
    cases.insert(
        "production_worker_cancel_adapter",
        sim_worker::apply_worker_control_cancellation(&worker_cancel.message, &worker_token)
            == Some(true)
            && worker_token.is_cancelled(),
    );
    cases.insert(
        "worker_frame_bound_literal_is_16_mib",
        MAX_WORKER_FRAME_BYTES == 16 * 1024 * 1024,
    );
    cases.insert(
        "worker_event_bound_literal_is_4_mib",
        MAX_ENCODED_PREVIEW_BYTES == 4 * 1024 * 1024,
    );
    let encoded_cancel = encode_worker_frame(&worker_cancel)?;
    cases.insert(
        "worker_cancel_projection_round_trips",
        decode_worker_frame(&encoded_cancel)? == worker_cancel,
    );
    cases.insert(
        "worker_event_exact_bound_is_accepted",
        encode_worker_frame(&envelope(WorkerMessage::Event {
            event: vec![0; MAX_ENCODED_PREVIEW_BYTES],
        }))
        .is_ok(),
    );
    cases.insert(
        "worker_event_over_bound_is_rejected",
        matches!(
            encode_worker_frame(&envelope(WorkerMessage::Event {
                event: vec![0; MAX_ENCODED_PREVIEW_BYTES + 1],
            })),
            Err(WorkerProtocolError::OversizedEvent)
        ),
    );
    cases.insert(
        "worker_frame_encode_over_bound_is_rejected",
        matches!(
            encode_worker_frame(&envelope(WorkerMessage::Execute {
                plan: vec![0; MAX_WORKER_FRAME_BYTES],
            })),
            Err(WorkerProtocolError::Oversized)
        ),
    );
    let mut exact_frame = vec![0; MAX_WORKER_FRAME_BYTES + 4];
    exact_frame[..4].copy_from_slice(
        &u32::try_from(MAX_WORKER_FRAME_BYTES)
            .map_err(|_| "worker frame bound exceeds u32")?
            .to_le_bytes(),
    );
    let exact_decode = decode_worker_frame(&exact_frame);
    cases.insert(
        "worker_frame_exact_decode_bound_is_admitted",
        !matches!(
            exact_decode,
            Err(WorkerProtocolError::Oversized | WorkerProtocolError::LengthMismatch)
        ),
    );
    let mut oversized_frame = vec![0; MAX_WORKER_FRAME_BYTES + 5];
    oversized_frame[..4].copy_from_slice(
        &u32::try_from(MAX_WORKER_FRAME_BYTES + 1)
            .map_err(|_| "worker oversized frame bound exceeds u32")?
            .to_le_bytes(),
    );
    cases.insert(
        "worker_frame_decode_over_bound_is_rejected",
        matches!(
            decode_worker_frame(&oversized_frame),
            Err(WorkerProtocolError::Oversized)
        ),
    );
    let oversized_prefix = u32::try_from(MAX_WORKER_FRAME_BYTES + 1)
        .map_err(|_| "worker oversized prefix exceeds u32")?
        .to_le_bytes();
    cases.insert(
        "production_worker_reader_rejects_over_bound_before_allocation",
        matches!(
            sim_worker::read_frame(oversized_prefix.as_slice()),
            Err(sim_worker::FrameError::TooLarge)
        ),
    );

    let definitions = cancellation_definitions(&workspace_root)?;
    let canonical_prefix = format!("{CANONICAL_DEFINITION}:");
    cases.insert(
        "single_repository_rust_definition",
        definitions.len() == 1
            && definitions
                .first()
                .is_some_and(|definition| definition.starts_with(&canonical_prefix)),
    );

    let catalog_row = cancellation_catalog_row(&workspace_root)?;
    let definition_hits = catalog_row
        .get("definition_hits")
        .ok_or("cancellation catalog row has no definition inventory")?;
    let production_call_sites = catalog_row
        .get("production_call_sites")
        .ok_or("cancellation catalog row has no production call-site inventory")?;
    let current_status = catalog_row
        .get("current_status")
        .ok_or("cancellation catalog row has no current status")?;
    cases.insert(
        "catalog_confirms_authoritative_owner",
        catalog_row
            .get("canonical_owner")
            .is_some_and(|owner| owner == "sim_types::CancellationToken")
            && matches!(
                current_status.as_str(),
                "authoritative_owner_confirmed"
                    | "consolidation_required[known_integration_gap]"
                    | "consolidation_required[competing_definitions_present,known_integration_gap]"
            ),
    );
    cases.insert(
        "catalog_assigns_known_integration_gap",
        current_status == "authoritative_owner_confirmed"
            || (catalog_row.get("consolidation_tasks").is_some_and(|tasks| {
                tasks.contains("comfy-parity-execution-output-owner-consolidation")
            }) && catalog_row.get("decision_reason").is_some_and(|reason| {
                reason.contains("Task 23") && reason.contains("worker failure cancelled flag")
            })),
    );
    cases.insert(
        "catalog_records_canonical_definition",
        definition_hits.contains(&format!("canonical@{CANONICAL_DEFINITION}")),
    );
    cases.insert(
        "catalog_excludes_development_oracle_from_production",
        definition_hits.contains(&format!("development_reference@{DEVELOPMENT_ORACLE}"))
            && !production_call_sites.contains(DEVELOPMENT_ORACLE),
    );
    for required_consumer in [
        "crates/sim_api/src/headless.rs",
        "crates/sim_api/src/transport.rs",
        "crates/sim_plugin_host/src/capabilities.rs",
        "crates/sim_runtime/src/native_execution_controller.rs",
        "crates/sim_tensor/src/operation.rs",
        "crates/sim_worker/src/sim_worker.rs",
    ] {
        cases.insert(
            required_consumer,
            production_call_sites.contains(required_consumer),
        );
    }

    if cases.values().any(|passed| !passed) {
        return Err(io::Error::other(format!(
            "VAL-CANCEL-001 cases failed: {cases:?}; definitions: {definitions:?}"
        ))
        .into());
    }

    fs::create_dir_all(&artifact_directory)?;
    let artifact = json!({
        "validation_id": "VAL-CANCEL-001",
        "validation": "VAL-CANCEL-001",
        "scope": "canonical-comfy-cancellation-state-and-checked-adapters",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "development_oracle_executed": false,
        },
        "fixture_digests": fixture_digests(&workspace_root)?,
        "inventory": {
            "canonical_definition": CANONICAL_DEFINITION,
            "definition_hits": definitions,
            "production_call_sites": production_call_sites,
            "development_oracle": DEVELOPMENT_ORACLE,
            "development_oracle_source_tree_required": false,
        },
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
        },
        "cases": cases,
        "skipped": [],
        "validation_closure": {
            "claimed": true,
            "scope": "Task 3 canonical cancellation state, adapters, transport shutdown, and worker bounds",
        },
        "release_closure_required": false,
    });
    let temporary_artifact = artifact_directory.join("val-cancel-001.json.tmp");
    match fs::remove_file(&temporary_artifact) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    fs::write(&temporary_artifact, serde_json::to_vec_pretty(&artifact)?)?;
    fs::rename(temporary_artifact, artifact_path)?;
    Ok(())
}
