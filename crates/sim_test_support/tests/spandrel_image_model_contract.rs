use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

const CONTRACT_BYTES: &[u8] = include_bytes!(
    "../../../.agents/specs/comfy-parity/catalogs/spandrel-image-model-contract.json"
);
const FIXTURE_BYTES: &[u8] =
    include_bytes!("../fixtures/models/spandrel-image-model-contract/contract-summary.json");

fn object<'a>(value: &'a Value, key: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing object {key}"))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {key}"))
}

#[test]
fn spandrel_contract_pins_snapshots_registry_and_license_rejections() -> Result<(), String> {
    let contract: Value =
        serde_json::from_slice(CONTRACT_BYTES).map_err(|error| error.to_string())?;
    let snapshots = object(&contract, "source_snapshots")?;
    let spandrel = snapshots
        .get("spandrel")
        .and_then(Value::as_object)
        .ok_or_else(|| "missing spandrel snapshot".to_string())?;
    let extra = snapshots
        .get("spandrel_extra_arches")
        .and_then(Value::as_object)
        .ok_or_else(|| "missing extra snapshot".to_string())?;
    assert_eq!(
        spandrel.get("version").and_then(Value::as_str),
        Some("0.4.2")
    );
    assert_eq!(spandrel.get("tag").and_then(Value::as_str), Some("v0.4.2"));
    assert_eq!(
        spandrel.get("commit").and_then(Value::as_str),
        Some("724cca389f28c38e1050689d4862a452fd644484")
    );
    assert_eq!(
        spandrel.get("baseline_tree_sha256").and_then(Value::as_str),
        Some("e1870c42b314fddb290f4d5322a03743076d98d0c6d288fc73691e3013994bbb")
    );
    assert_eq!(
        spandrel.get("included_file_count").and_then(Value::as_u64),
        Some(180)
    );
    assert_eq!(extra.get("version").and_then(Value::as_str), Some("0.2.0"));
    assert_eq!(extra.get("tag").and_then(Value::as_str), Some("v0.4.0"));
    assert_eq!(
        extra.get("commit").and_then(Value::as_str),
        Some("a1db3f5debbeeacbe02fb4114c69feee56ba5e21")
    );
    assert_eq!(
        extra.get("baseline_tree_sha256").and_then(Value::as_str),
        Some("7c0915d2e0df7db2131117087744fa5e73954dcad72aa785386d6bf8c1efb3aa")
    );
    assert_eq!(
        extra.get("included_file_count").and_then(Value::as_u64),
        Some(52)
    );

    let rows = array(&contract, "architectures")?;
    assert_eq!(rows.len(), 52);
    let mut identifiers = BTreeSet::new();
    for (ordinal, row) in rows.iter().enumerate() {
        let row = row
            .as_object()
            .ok_or_else(|| format!("architecture row {ordinal} is not an object"))?;
        assert_eq!(
            row.get("ordinal").and_then(Value::as_u64),
            Some(ordinal as u64)
        );
        let identifier = row
            .get("architecture_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("architecture row {ordinal} has no identifier"))?;
        assert!(identifiers.insert(identifier));
        assert_eq!(
            row.get("support_disposition").and_then(Value::as_str),
            Some("rejected")
        );
        assert_eq!(
            row.get("license_artifacts")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );
        assert!(
            row.get("model_use_disposition")
                .and_then(Value::as_str)
                .is_some_and(|value| value.contains("no-model-weights-approved"))
        );
        assert!(
            row.get("detection_predicate")
                .and_then(Value::as_str)
                .is_some()
        );
        assert!(row.get("equation_sha256").and_then(Value::as_str).is_some());
        assert!(
            row.get("dependency_sha256")
                .and_then(Value::as_str)
                .is_some()
        );
    }
    assert_eq!(identifiers.len(), 52);
    Ok(())
}

#[test]
fn spandrel_contract_preserves_optional_outcomes_and_json_only_boundary() -> Result<(), String> {
    let contract: Value =
        serde_json::from_slice(CONTRACT_BYTES).map_err(|error| error.to_string())?;
    let outcomes = array(&contract, "optional_extra_outcomes")?;
    let names = outcomes
        .iter()
        .map(|outcome| outcome.get("outcome").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            Some("absent-or-import-failure"),
            Some("successful-add"),
            Some("add-failure")
        ]
    );
    let boundary = object(&contract, "source_boundary")?;
    assert!(
        boundary
            .get("production_runtime")
            .and_then(Value::as_str)
            .is_some_and(|value| value.contains("native Rust only") && value.contains("no Python"))
    );
    assert_eq!(
        contract
            .pointer("/task_projection/implementation_leaves")
            .and_then(Value::as_array)
            .map(Vec::len),
        Some(0)
    );

    let fixture: Value =
        serde_json::from_slice(FIXTURE_BYTES).map_err(|error| error.to_string())?;
    let actual_digest = format!("{:x}", Sha256::digest(CONTRACT_BYTES));
    assert_eq!(
        fixture.get("catalog_sha256").and_then(Value::as_str),
        Some(actual_digest.as_str())
    );
    assert!(!String::from_utf8_lossy(FIXTURE_BYTES).contains(".safetensors"));
    assert!(!String::from_utf8_lossy(FIXTURE_BYTES).contains(".pth"));
    Ok(())
}
