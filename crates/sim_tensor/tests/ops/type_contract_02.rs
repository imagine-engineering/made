use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CanonicalReference, ContractInventoryKind, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OPERATION_CONTRACTS, TypeMarkerReference,
    generated_type_contract_02::{
        ASSIGNED_TYPE_REFERENCES, SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID,
        SIM_DISABLE_WEIGHT_INIT_RMS_NORM_REFERENCE, SIM_MANUAL_CAST_OPERATION_ID,
        SIM_MANUAL_CAST_REFERENCE, TORCH_RMS_NORM_OPERATION_ID, TORCH_RMS_NORM_REFERENCE,
        assigned_type_contract, sim_disable_weight_init_rms_norm_contract,
        sim_manual_cast_contract, torch_rms_norm_contract,
    },
};
use std::{fs, io, path::Path};

const OWNER_TASK_ID: &str = "comfy-parity-tensor-ops-type-contract-comfy-tensor-op-ba6ae52d4258";

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, io::Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other(format!("type-reference evidence is missing {field}")))
}

#[test]
fn type_facades_use_the_canonical_reference_owner() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        ASSIGNED_TYPE_REFERENCES,
        &[
            (
                SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID,
                SIM_DISABLE_WEIGHT_INIT_RMS_NORM_REFERENCE,
            ),
            (SIM_MANUAL_CAST_OPERATION_ID, SIM_MANUAL_CAST_REFERENCE,),
            (TORCH_RMS_NORM_OPERATION_ID, TORCH_RMS_NORM_REFERENCE),
        ]
    );
    let disable_weight_init_rms_norm = sim_disable_weight_init_rms_norm_contract()
        .ok_or("sim.ops.disable_weight_init.RMSNorm typed contract is missing")?;
    assert_eq!(
        disable_weight_init_rms_norm.operation_id(),
        SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID
    );
    assert_eq!(
        disable_weight_init_rms_norm.canonical_target(),
        "sim.ops.disable_weight_init.RMSNorm"
    );
    assert_eq!(
        disable_weight_init_rms_norm.inventory_kind(),
        ContractInventoryKind::TypeReference
    );
    assert_eq!(
        SIM_DISABLE_WEIGHT_INIT_RMS_NORM_REFERENCE,
        TypeMarkerReference::ComfyDisableWeightInitRmsNorm
    );
    assert_eq!(
        disable_weight_init_rms_norm.semantic(),
        CanonicalReference::TypeMarker(SIM_DISABLE_WEIGHT_INIT_RMS_NORM_REFERENCE)
    );

    let manual_cast =
        sim_manual_cast_contract().ok_or("sim.ops.manual_cast typed contract is missing")?;
    assert_eq!(manual_cast.operation_id(), SIM_MANUAL_CAST_OPERATION_ID);
    assert_eq!(manual_cast.canonical_target(), "sim.ops.manual_cast");
    assert_eq!(
        manual_cast.inventory_kind(),
        ContractInventoryKind::TypeReference
    );
    assert_eq!(
        SIM_MANUAL_CAST_REFERENCE,
        TypeMarkerReference::ComfyManualCast
    );
    assert_eq!(
        manual_cast.semantic(),
        CanonicalReference::TypeMarker(SIM_MANUAL_CAST_REFERENCE)
    );

    let torch_rms_norm =
        torch_rms_norm_contract().ok_or("torch.nn.RMSNorm typed contract is missing")?;
    assert_eq!(torch_rms_norm.operation_id(), TORCH_RMS_NORM_OPERATION_ID);
    assert_eq!(torch_rms_norm.canonical_target(), "torch.nn.RMSNorm");
    assert_eq!(
        torch_rms_norm.inventory_kind(),
        ContractInventoryKind::TypeReference
    );
    assert_eq!(TORCH_RMS_NORM_REFERENCE, TypeMarkerReference::RmsNorm);
    assert_eq!(
        torch_rms_norm.semantic(),
        CanonicalReference::TypeMarker(TORCH_RMS_NORM_REFERENCE)
    );
    assert!(assigned_type_contract("COMFY-TENSOR-OP-UNASSIGNED").is_none());
    Ok(())
}

#[test]
fn type_references_never_enter_the_kernel_resolution_registry() {
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"type_contract_02"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "type_contract_02")
    );
    let assigned = [
        SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID,
        SIM_MANUAL_CAST_OPERATION_ID,
        TORCH_RMS_NORM_OPERATION_ID,
    ];
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .all(|contract| !assigned.contains(&contract.operation_id))
    );
    assert_eq!(assigned.len(), ASSIGNED_TYPE_REFERENCES.len());
    assert!(
        assigned
            .iter()
            .zip(ASSIGNED_TYPE_REFERENCES)
            .all(|(operation_id, (assigned_id, _))| operation_id == assigned_id)
    );

    let source = include_str!("../../src/ops/type_contract_02.rs");
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
}

#[test]
fn task_claims_exactly_its_three_catalog_references() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog = fs::read_to_string(
        workspace_root
            .join("crates/sim_test_support/parity/catalogs/native-tensor-operation-contracts.csv"),
    )?;
    let owned_rows = catalog
        .lines()
        .filter(|row| row.contains(OWNER_TASK_ID))
        .collect::<Vec<_>>();
    assert_eq!(owned_rows.len(), ASSIGNED_TYPE_REFERENCES.len());
    for (operation_id, marker) in ASSIGNED_TYPE_REFERENCES {
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
        assert!(row.contains(",type_reference,"));
        assert!(row.contains(",type_contract_02,false,not executable,"));
        assert!(row.contains(&format!(
            "\"\"resolution_owner_task_id\"\":\"\"{OWNER_TASK_ID}\"\""
        )));
        let contract = assigned_type_contract(operation_id)
            .ok_or_else(|| format!("assigned catalog reference {operation_id} is unresolved"))?;
        assert_eq!(contract.semantic(), CanonicalReference::TypeMarker(*marker));
    }

    for reference in OPERATION_CONTRACTS
        .iter()
        .filter_map(|record| record.typed_reference())
    {
        if !ASSIGNED_TYPE_REFERENCES
            .iter()
            .any(|(operation_id, _)| *operation_id == reference.operation_id())
        {
            assert!(assigned_type_contract(reference.operation_id()).is_none());
        }
    }
    Ok(())
}

#[test]
fn typed_reference_evidence_is_exact_and_hash_sealed() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (
        file_name,
        operation_id,
        baseline_overload_id,
        baseline_digest,
        target,
        semantic,
        expected_digest,
    ) in [
        (
            "sim_disable_weight_init_rms_norm.json",
            SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID,
            "COMFY-TENSOR-OP-CA96BCF6B334:reference",
            "5c71f30bea62a3576c3ffc90337770a7cbd21c847cac2606f59524e5e75fc32e",
            "sim.ops.disable_weight_init.RMSNorm",
            "TypeMarkerReference::ComfyDisableWeightInitRmsNorm",
            "b9789980c7660b1930223cc702b324766372c8b61add99f5af2addd8a7de8aaa",
        ),
        (
            "sim_manual_cast.json",
            SIM_MANUAL_CAST_OPERATION_ID,
            "COMFY-TENSOR-OP-BA6AE52D4258:reference",
            "7596402499977020acedbc9800f1ac21a727a86f30708e4692579426fc9125fc",
            "sim.ops.manual_cast",
            "TypeMarkerReference::ComfyManualCast",
            "b730cd5cb25ea938f5605b57d9e712b753e6963f0debc2b758d8be38e49d5176",
        ),
        (
            "torch_nn_rms_norm.json",
            TORCH_RMS_NORM_OPERATION_ID,
            "COMFY-TENSOR-OP-FD4EE56E61FC:reference",
            "aeaa3a109be9ab51ac3468abfbd8ce9e52fc30829c4b8a1b56262e33800af8f9",
            "torch.nn.RMSNorm",
            "TypeMarkerReference::RmsNorm",
            "ac728577df5b602e07977f5771c4f6f67b56f8a21795bfd48195805e1f5c9dfd",
        ),
    ] {
        let fixture_path = workspace_root
            .join("crates/sim_test_support/fixtures/tensor_operations/type_contract_02")
            .join(file_name);
        let bytes = fs::read(fixture_path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(digest, expected_digest);
        let fixture: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            required_string(&fixture, "resolution_module")?,
            "type_contract_02"
        );
        assert_eq!(required_string(&fixture, "operation_id")?, operation_id);
        assert_eq!(
            required_string(&fixture, "baseline_overload_id")?,
            baseline_overload_id
        );
        assert_eq!(
            required_string(&fixture, "baseline_fixture_sha256")?,
            baseline_digest
        );
        assert_eq!(required_string(&fixture, "owner_task_id")?, OWNER_TASK_ID);
        assert_eq!(required_string(&fixture, "canonical_target")?, target);
        let reference_semantic = fixture
            .get("reference_semantic")
            .ok_or("type-reference evidence is missing reference_semantic")?;
        assert_eq!(
            required_string(reference_semantic, "category")?,
            "type-marker"
        );
        assert_eq!(required_string(reference_semantic, "value")?, target);
        assert_eq!(
            required_string(&fixture, "canonical_rust_semantic")?,
            semantic
        );
        assert_eq!(
            fixture.get("executable").and_then(Value::as_bool),
            Some(false)
        );
        assert_ne!(baseline_digest, expected_digest);
        assert!(
            fixture
                .get("source_observations")
                .and_then(Value::as_array)
                .is_some_and(|observations| !observations.is_empty())
        );
    }
    Ok(())
}
