use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CanonicalReference, ContractInventoryKind, DType, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, MemoryFormatReference, NamespaceReference,
    OPERATION_CONTRACTS, TypedReferenceContract, generated_value_or_constant_contract_06 as leaf,
};
use std::{collections::BTreeSet, fs, io, path::Path};

const OWNER: &str =
    "comfy-parity-tensor-ops-value-or-constant-contract-comfy-tensor-op-e43e68ab67d6";
type ContractFacade = fn() -> Option<TypedReferenceContract>;

fn reference_cases() -> [(
    ContractFacade,
    &'static str,
    &'static str,
    CanonicalReference,
); 4] {
    [
        (
            leaf::torch_package_path_contract,
            leaf::TORCH_PACKAGE_PATH_OPERATION_ID,
            "torch.__path__",
            CanonicalReference::Namespace(NamespaceReference::TorchPackagePath),
        ),
        (
            leaf::torch_bfloat16_contract,
            leaf::TORCH_BFLOAT16_OPERATION_ID,
            "torch.bfloat16",
            CanonicalReference::DType(DType::Bf16),
        ),
        (
            leaf::torch_preserve_format_contract,
            leaf::TORCH_PRESERVE_FORMAT_OPERATION_ID,
            "torch.preserve_format",
            CanonicalReference::MemoryFormat(MemoryFormatReference::PreserveFormat),
        ),
        (
            leaf::torch_uint16_contract,
            leaf::TORCH_UINT16_OPERATION_ID,
            "torch.uint16",
            CanonicalReference::DType(DType::U16),
        ),
    ]
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, io::Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other(format!("reference evidence is missing {field}")))
}

#[test]
fn value_and_constant_facades_use_the_canonical_reference_owner()
-> Result<(), Box<dyn std::error::Error>> {
    for (facade, operation_id, target, semantic) in reference_cases() {
        let contract = facade().ok_or_else(|| format!("typed contract is missing for {target}"))?;
        assert_eq!(contract.operation_id(), operation_id);
        assert_eq!(contract.canonical_target(), target);
        assert_eq!(
            contract.inventory_kind(),
            ContractInventoryKind::NamespaceValueReference
        );
        assert_eq!(contract.semantic(), semantic);
        assert_eq!(
            leaf::assigned_value_or_constant_contract(operation_id),
            Some(contract)
        );
    }
    Ok(())
}

#[test]
fn assigned_catalog_rows_are_exact_and_never_enter_kernel_resolution()
-> Result<(), Box<dyn std::error::Error>> {
    let expected_ids = reference_cases()
        .map(|(_, operation_id, _, _)| operation_id)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let assigned_ids = leaf::ASSIGNED_VALUE_OR_CONSTANT_REFERENCES
        .iter()
        .map(|(operation_id, _)| *operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(assigned_ids, expected_ids);
    assert_eq!(assigned_ids.len(), 4);
    assert!(leaf::assigned_value_or_constant_contract("COMFY-TENSOR-OP-UNASSIGNED").is_none());
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"value_or_constant_contract_06"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "value_or_constant_contract_06")
    );
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .all(|contract| !expected_ids.contains(contract.operation_id))
    );

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = fs::read_to_string(
        workspace_root
            .join(".agents/specs/comfy-parity/catalogs/native-tensor-operation-contracts.csv"),
    )?;
    let owned_rows = catalog
        .lines()
        .filter(|row| row.contains(OWNER))
        .collect::<Vec<_>>();
    assert_eq!(owned_rows.len(), expected_ids.len());
    for operation_id in &expected_ids {
        let matching_rows = owned_rows
            .iter()
            .filter(|row| row.starts_with(&format!("{operation_id},")))
            .collect::<Vec<_>>();
        assert_eq!(
            matching_rows.len(),
            1,
            "catalog ownership for {operation_id}"
        );
        let row = matching_rows[0];
        assert!(row.contains(",namespace_value_reference,"));
        assert!(row.contains(",value_or_constant_contract_06,false,not executable,"));
        assert!(row.contains(&format!("\"\"resolution_owner_task_id\"\":\"\"{OWNER}\"\"")));
    }
    for reference in OPERATION_CONTRACTS
        .iter()
        .filter_map(|record| record.typed_reference())
    {
        if !expected_ids.contains(reference.operation_id()) {
            assert!(leaf::assigned_value_or_constant_contract(reference.operation_id()).is_none());
        }
    }

    let source = include_str!("../../src/ops/value_or_constant_contract_06.rs");
    for executable_surface in [
        "ResolvedOperationContract",
        "TensorBackend",
        "CpuBackend",
        "ExecutionContext",
        "CancellationToken",
        "pub fn execute",
        "pub fn forward",
        "pub fn backward",
    ] {
        assert!(
            !source.contains(executable_surface),
            "reference-only leaf contains executable surface {executable_surface}"
        );
    }
    Ok(())
}

#[test]
fn typed_reference_evidence_is_exact_distinct_and_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_directory = workspace_root
        .join("crates/sim_test_support/fixtures/tensor_operations/value_or_constant_contract_06");
    let fixture_entries = fs::read_dir(&fixture_directory)?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(fixture_entries.len(), 4);
    assert!(fixture_entries.iter().all(|entry| {
        entry
            .file_type()
            .is_ok_and(|file_type| file_type.is_file() && !file_type.is_symlink())
            && entry.path().extension().and_then(|value| value.to_str()) == Some("json")
    }));
    let expected_fixture_fields = [
        "schema_version",
        "resolution_module",
        "operation_id",
        "baseline_overload_id",
        "baseline_fixture_sha256",
        "owner_task_id",
        "inventory_kind",
        "canonical_target",
        "reference_semantic",
        "canonical_rust_semantic",
        "executable",
        "source_observations",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut fixture_digests = BTreeSet::new();
    for (file_name, operation_id, target, category, semantic, expected_digest) in [
        (
            "torch_package_path.json",
            leaf::TORCH_PACKAGE_PATH_OPERATION_ID,
            "torch.__path__",
            "namespace",
            "NamespaceReference::TorchPackagePath",
            "3d4870d04350ef4e49af136c15c9a427c6c61475b52c4151f1693abbe5de9196",
        ),
        (
            "torch_bfloat16.json",
            leaf::TORCH_BFLOAT16_OPERATION_ID,
            "torch.bfloat16",
            "dtype",
            "DType::Bf16",
            "7b9a0c250223ba56394c83dba92711f50b68a61dc8fcb97d57e098fd3d2a5c40",
        ),
        (
            "torch_preserve_format.json",
            leaf::TORCH_PRESERVE_FORMAT_OPERATION_ID,
            "torch.preserve_format",
            "layout-or-memory-format",
            "MemoryFormatReference::PreserveFormat",
            "2d6d61c097abb099e6fce5ccfc93240c343c3c7f93ec8f55aaa5954ca54decb0",
        ),
        (
            "torch_uint16.json",
            leaf::TORCH_UINT16_OPERATION_ID,
            "torch.uint16",
            "dtype",
            "DType::U16",
            "e421cf814f4fb6caa7928f3c47cca7a44162e8f9e157a4eb08cf098f7f9bf17c",
        ),
    ] {
        let fixture_path = fixture_directory.join(file_name);
        let bytes = fs::read(fixture_path)?;
        let fixture_digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(fixture_digest, expected_digest);
        assert!(fixture_digests.insert(fixture_digest));

        let fixture: Value = serde_json::from_slice(&bytes)?;
        let fixture_fields = fixture
            .as_object()
            .ok_or("reference evidence must be an object")?
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(fixture_fields, expected_fixture_fields);
        assert_eq!(
            fixture.get("schema_version").and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            required_string(&fixture, "resolution_module")?,
            "value_or_constant_contract_06"
        );
        assert_eq!(required_string(&fixture, "operation_id")?, operation_id);
        assert_eq!(
            required_string(&fixture, "baseline_overload_id")?,
            format!("{operation_id}:reference")
        );
        assert_eq!(required_string(&fixture, "owner_task_id")?, OWNER);
        assert_eq!(
            required_string(&fixture, "inventory_kind")?,
            "namespace_value_reference"
        );
        assert_eq!(required_string(&fixture, "canonical_target")?, target);
        assert_eq!(
            required_string(&fixture, "canonical_rust_semantic")?,
            semantic
        );
        assert_eq!(
            fixture.get("executable").and_then(Value::as_bool),
            Some(false)
        );
        let observations = fixture
            .get("source_observations")
            .and_then(Value::as_array)
            .ok_or("reference evidence is missing source observations")?;
        assert!(!observations.is_empty());
        for observation in observations {
            let fields = observation
                .as_object()
                .ok_or("source observation must be an object")?;
            assert_eq!(
                fields.keys().map(String::as_str).collect::<BTreeSet<_>>(),
                BTreeSet::from(["path", "line", "use"])
            );
            let source_path = required_string(observation, "path")?;
            let source_path = Path::new(source_path);
            assert!(!source_path.as_os_str().is_empty());
            assert!(!source_path.is_absolute());
            assert!(!source_path.components().any(|component| matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )));
            assert!(
                observation
                    .get("line")
                    .and_then(Value::as_u64)
                    .is_some_and(|line| line > 0)
            );
            assert!(!required_string(observation, "use")?.is_empty());
        }

        let reference_semantic = fixture
            .get("reference_semantic")
            .and_then(Value::as_object)
            .ok_or("reference semantic must be an object")?;
        assert_eq!(
            reference_semantic
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["category", "value"])
        );
        let reference_semantic = Value::Object(reference_semantic.clone());
        assert_eq!(required_string(&reference_semantic, "category")?, category);
        assert_eq!(required_string(&reference_semantic, "value")?, target);

        let operation_suffix = operation_id
            .strip_prefix("COMFY-TENSOR-OP-")
            .ok_or("operation ID has an invalid prefix")?
            .to_ascii_lowercase();
        let baseline_path = workspace_root.join(format!(
            "crates/sim_test_support/fixtures/tensor_signatures/contracts/sim-tensor-op-{operation_suffix}.json"
        ));
        let baseline_bytes = fs::read(baseline_path)?;
        let baseline_digest = format!("{:x}", Sha256::digest(&baseline_bytes));
        assert_eq!(
            required_string(&fixture, "baseline_fixture_sha256")?,
            baseline_digest
        );
        assert_ne!(baseline_digest, expected_digest);
        let baseline: Value = serde_json::from_slice(&baseline_bytes)?;
        assert_eq!(required_string(&baseline, "operation_id")?, operation_id);
        assert_eq!(required_string(&baseline, "canonical_target")?, target);
        assert_eq!(
            baseline.get("reference_semantic"),
            fixture.get("reference_semantic")
        );
    }
    assert_eq!(fixture_digests.len(), 4);
    Ok(())
}
