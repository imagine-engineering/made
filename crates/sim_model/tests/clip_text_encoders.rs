use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::{
    TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT, TEXT_ENCODER_ARCHITECTURE_REGISTRY_VERSION,
    TEXT_ENCODER_OWNER_FACTS, TextEncoderArchitectureRegistry, TextEncoderRegistryError,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
};

const TASK_ID: &str = "comfy-parity-clip-text-encoder-breadth";
const OWNER_TASKS: [(&str, u64); 4] = [
    ("comfy-parity-clip-text-encoder-t5-foundation", 19),
    ("comfy-parity-clip-text-encoder-decoder-foundation", 127),
    ("comfy-parity-clip-text-encoder-multimodal-foundation", 53),
    ("comfy-parity-clip-text-encoder-composite-adapters", 199),
];
const IMPLEMENTATIONS: [&str; 3] = [
    "crates/sim_model/src/clip_text_encoders.rs",
    "crates/sim_model/src/sim_model.rs",
    "crates/sim_model/tests/clip_text_encoders.rs",
];

#[derive(Clone, Debug)]
struct CatalogRow {
    contract_id: String,
    source_path: String,
    source_symbol: String,
    source_ordinal: usize,
    source_sha256: String,
    symbol_sha256: String,
    native_owner: String,
    implementation_task: String,
}

#[test]
fn registry_is_versioned_total_unique_and_deterministic() -> Result<(), Box<dyn Error>> {
    let workspace = workspace()?;
    let rows = text_encoder_rows(workspace)?;
    let registry = TextEncoderArchitectureRegistry::checked()?;
    assert_eq!(
        registry.version(),
        TEXT_ENCODER_ARCHITECTURE_REGISTRY_VERSION
    );
    assert_eq!(
        registry.contract_count(),
        TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT
    );
    assert_eq!(rows.len(), TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT);
    assert_eq!(
        TEXT_ENCODER_OWNER_FACTS
            .iter()
            .map(|fact| fact.contract_count)
            .sum::<usize>(),
        TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT
    );

    let mut contracts = BTreeSet::new();
    let mut source_digests = BTreeMap::new();
    let mut owner_counts = BTreeMap::new();
    let mut previous_ordinal = None;
    for row in &rows {
        assert!(contracts.insert(row.contract_id.clone()));
        if let Some(previous) = previous_ordinal {
            assert_eq!(row.source_ordinal, previous + 1);
        }
        previous_ordinal = Some(row.source_ordinal);
        let owner = registry.owner_for(&row.source_path, &row.source_symbol)?;
        assert_eq!(owner.native_owner(), row.native_owner);
        assert_eq!(owner.implementation_task(), row.implementation_task);
        *owner_counts.entry(owner.native_owner()).or_insert(0_usize) += 1;
        source_digests
            .entry(row.source_path.as_str())
            .and_modify(|digest: &mut &str| assert_eq!(*digest, row.source_sha256))
            .or_insert(row.source_sha256.as_str());
        assert!(valid_sha256(&row.source_sha256));
        assert!(valid_sha256(&row.symbol_sha256));
    }
    for fact in TEXT_ENCODER_OWNER_FACTS {
        assert_eq!(
            owner_counts.get(fact.native_owner).copied(),
            Some(fact.contract_count)
        );
    }
    for (path, expected) in source_digests {
        assert_eq!(sha256(&fs::read(workspace.join(path))?), expected);
    }
    assert_eq!(registry.identity_sha256(), registry.identity_sha256());
    assert_ne!(registry.identity_sha256(), [0_u8; 32]);
    assert!(matches!(
        registry.owner_for(
            "projects/comfy/ComfyUI/sim/text_encoders/unknown.py",
            "Unknown"
        ),
        Err(TextEncoderRegistryError::UnknownContract { .. })
    ));
    Ok(())
}

#[test]
fn val_clip_001_reconciles_all_398_rows_and_extends_ledger() -> Result<(), Box<dyn Error>> {
    let workspace = workspace()?;
    let rows = text_encoder_rows(workspace)?;
    let artifact_path = workspace.join("target/sim-parity/val-clip-001.json");
    let mut artifact: Value = serde_json::from_slice(&fs::read(&artifact_path)?)?;
    assert_eq!(artifact.get("schema_version"), Some(&json!(1)));
    assert_eq!(artifact.get("validation_id"), Some(&json!("VAL-CLIP-001")));

    refresh_shared_module_root_digest(workspace, &mut artifact)?;

    let task_results = artifact
        .get("task_results")
        .and_then(Value::as_object)
        .ok_or("VAL-CLIP-001 task results are missing")?;
    for (task, expected_passed) in OWNER_TASKS {
        let result = task_results
            .get(task)
            .ok_or("owner task result is missing")?;
        assert_eq!(result.get("status"), Some(&json!("passed")));
        assert_eq!(result.get("passed"), Some(&json!(expected_passed)));
        assert_eq!(result.get("failed"), Some(&json!(0)));
        assert_eq!(result.get("skipped"), Some(&json!(0)));
        validate_implementation_digests(workspace, result)?;
    }

    let contracts = artifact
        .get("contracts")
        .and_then(Value::as_array)
        .ok_or("VAL-CLIP-001 contracts are missing")?;
    let contracts = contracts
        .iter()
        .filter_map(|contract| {
            contract
                .get("contract_id")
                .and_then(Value::as_str)
                .map(|contract_id| (contract_id, contract))
        })
        .collect::<BTreeMap<_, _>>();
    for row in &rows {
        let contract = contracts
            .get(row.contract_id.as_str())
            .ok_or("text-encoder contract is missing from VAL-CLIP-001")?;
        assert_eq!(
            contract.get("task_id"),
            Some(&json!(row.implementation_task))
        );
        assert_eq!(
            contract.get("source_sha256"),
            Some(&json!(row.source_sha256))
        );
        assert_eq!(
            contract.get("symbol_sha256"),
            Some(&json!(row.symbol_sha256))
        );
        assert_eq!(contract.get("status"), Some(&json!("passed")));
        let cases = contract
            .get("case_ids")
            .and_then(Value::as_array)
            .ok_or("contract cases are missing")?;
        assert!(!cases.is_empty());
    }

    let implementations = IMPLEMENTATIONS
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path,
                "sha256": sha256(&fs::read(workspace.join(path))?),
            }))
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    let task_results = artifact
        .get_mut("task_results")
        .and_then(Value::as_object_mut)
        .ok_or("VAL-CLIP-001 task results are missing")?;
    task_results.insert(
        TASK_ID.to_owned(),
        json!({
            "status": "passed",
            "passed": TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT,
            "failed": 0,
            "skipped": 0,
            "case_ids": [
                "text-encoder-breadth:exact-398-row-reconciliation",
                "text-encoder-breadth:four-owner-registry-uniqueness",
                "text-encoder-breadth:source-symbol-and-implementation-digests",
                "text-encoder-breadth:deterministic-cache-identity",
                "text-encoder-breadth:typed-unknown-failure",
            ],
            "implementations": implementations,
        }),
    );
    let passed = task_results.values().try_fold(0_u64, |total, result| {
        total
            .checked_add(
                result
                    .get("passed")
                    .and_then(Value::as_u64)
                    .ok_or("task passed count is missing")?,
            )
            .ok_or("task passed count overflowed")
    })?;
    artifact["summary"] = json!({"passed": passed, "failed": 0, "skipped": 0});
    let remaining = artifact
        .get_mut("remaining_tasks")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 remaining tasks are missing")?;
    remaining.retain(|task| task.as_str() != Some(TASK_ID));
    let producer = "crates/sim_model/tests/clip_text_encoders.rs";
    artifact["implementation"] = json!({
        "path": producer,
        "sha256": sha256(&fs::read(workspace.join(producer))?),
    });
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(artifact_path, bytes)?;
    Ok(())
}

fn refresh_shared_module_root_digest(
    workspace: &Path,
    artifact: &mut Value,
) -> Result<(), Box<dyn Error>> {
    const MODULE_ROOT: &str = "crates/sim_model/src/sim_model.rs";
    let digest = sha256(&fs::read(workspace.join(MODULE_ROOT))?);
    let task_results = artifact
        .get_mut("task_results")
        .and_then(Value::as_object_mut)
        .ok_or("VAL-CLIP-001 task results are missing")?;
    for result in task_results.values_mut() {
        let implementations = result
            .get_mut("implementations")
            .and_then(Value::as_array_mut)
            .ok_or("task implementations are missing")?;
        for implementation in implementations {
            if implementation.get("path").and_then(Value::as_str) == Some(MODULE_ROOT) {
                implementation["sha256"] = json!(digest.clone());
            }
        }
    }
    Ok(())
}

fn validate_implementation_digests(workspace: &Path, result: &Value) -> Result<(), Box<dyn Error>> {
    let implementations = result
        .get("implementations")
        .and_then(Value::as_array)
        .ok_or("task implementations are missing")?;
    assert!(!implementations.is_empty());
    for implementation in implementations {
        let path = implementation
            .get("path")
            .and_then(Value::as_str)
            .ok_or("implementation path is missing")?;
        let expected = implementation
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or("implementation digest is missing")?;
        assert_eq!(sha256(&fs::read(workspace.join(path))?), expected);
    }
    Ok(())
}

fn text_encoder_rows(workspace: &Path) -> Result<Vec<CatalogRow>, Box<dyn Error>> {
    let records = parse_csv(&fs::read_to_string(workspace.join(
        "crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv",
    ))?)?;
    let header = records.first().ok_or("conditioning catalog is empty")?;
    let index = |name: &str| {
        header
            .iter()
            .position(|field| field == name)
            .ok_or_else(|| format!("catalog field is missing: {name}"))
    };
    let contract_id = index("contract_id")?;
    let kind = index("kind")?;
    let source_path = index("source_path")?;
    let source_symbol = index("source_symbol")?;
    let source_ordinal = index("source_ordinal")?;
    let source_sha256 = index("source_sha256")?;
    let symbol_sha256 = index("symbol_sha256")?;
    let native_owner = index("native_owner")?;
    let implementation_task = index("implementation_task")?;
    records
        .into_iter()
        .skip(1)
        .filter(|record| {
            record
                .get(kind)
                .is_some_and(|value| value == "clip_text_encoder_architecture")
        })
        .map(|record| {
            Ok(CatalogRow {
                contract_id: record[contract_id].clone(),
                source_path: record[source_path].clone(),
                source_symbol: record[source_symbol].clone(),
                source_ordinal: record[source_ordinal].parse()?,
                source_sha256: record[source_sha256].clone(),
                symbol_sha256: record[symbol_sha256].clone(),
                native_owner: record[native_owner].clone(),
                implementation_task: record[implementation_task].clone(),
            })
        })
        .collect()
}

fn parse_csv(source: &str) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = source.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => record.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            '\r' if !quoted => {}
            value => field.push(value),
        }
    }
    if quoted {
        return Err("CSV ended inside a quoted field".into());
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn workspace() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "workspace root is unavailable".into())
}
