use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CanonicalReference, ContractInventoryKind, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    NamespaceReference, VersionValueReference,
    generated_namespace_contract_01::{
        SIM_OPS_OPERATION_ID, SIM_OPS_REFERENCE, TORCH_CUDA_VERSION_OPERATION_ID,
        TORCH_CUDA_VERSION_REFERENCE, TORCH_NEURAL_NETWORK_OPERATION_ID,
        TORCH_NEURAL_NETWORK_REFERENCE, sim_ops_contract, torch_cuda_version_contract,
        torch_neural_network_contract,
    },
};
use std::{fs, io, path::Path};

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, io::Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other(format!("reference evidence is missing {field}")))
}

#[test]
fn namespace_and_version_facades_use_the_canonical_reference_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let sim_ops = sim_ops_contract().ok_or("sim.ops typed contract is missing")?;
    assert_eq!(sim_ops.operation_id(), SIM_OPS_OPERATION_ID);
    assert_eq!(sim_ops.canonical_target(), "sim.ops");
    assert_eq!(
        sim_ops.inventory_kind(),
        ContractInventoryKind::NamespaceValueReference
    );
    assert_eq!(SIM_OPS_REFERENCE, NamespaceReference::ComfyOps);
    assert_eq!(
        sim_ops.semantic(),
        CanonicalReference::Namespace(SIM_OPS_REFERENCE)
    );

    let torch_neural_network =
        torch_neural_network_contract().ok_or("torch.nn typed contract is missing")?;
    assert_eq!(
        torch_neural_network.operation_id(),
        TORCH_NEURAL_NETWORK_OPERATION_ID
    );
    assert_eq!(torch_neural_network.canonical_target(), "torch.nn");
    assert_eq!(
        torch_neural_network.inventory_kind(),
        ContractInventoryKind::NamespaceValueReference
    );
    assert_eq!(
        TORCH_NEURAL_NETWORK_REFERENCE,
        NamespaceReference::TorchNeuralNetwork
    );
    assert_eq!(
        torch_neural_network.semantic(),
        CanonicalReference::Namespace(TORCH_NEURAL_NETWORK_REFERENCE)
    );

    let torch_cuda_version =
        torch_cuda_version_contract().ok_or("torch.version.cuda typed contract is missing")?;
    assert_eq!(
        torch_cuda_version.operation_id(),
        TORCH_CUDA_VERSION_OPERATION_ID
    );
    assert_eq!(torch_cuda_version.canonical_target(), "torch.version.cuda");
    assert_eq!(
        torch_cuda_version.inventory_kind(),
        ContractInventoryKind::NamespaceValueReference
    );
    assert_eq!(TORCH_CUDA_VERSION_REFERENCE, VersionValueReference::Cuda);
    assert_eq!(
        torch_cuda_version.semantic(),
        CanonicalReference::VersionValue(TORCH_CUDA_VERSION_REFERENCE)
    );
    Ok(())
}

#[test]
fn non_executable_references_never_enter_the_kernel_resolution_registry() {
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "namespace_contract_01")
    );
    let assigned = [
        SIM_OPS_OPERATION_ID,
        TORCH_NEURAL_NETWORK_OPERATION_ID,
        TORCH_CUDA_VERSION_OPERATION_ID,
    ];
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .all(|contract| !assigned.contains(&contract.operation_id))
    );
}

#[test]
fn typed_reference_evidence_is_exact_and_hash_sealed() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let owner = "comfy-parity-tensor-ops-namespace-contract-comfy-tensor-op-764a8e60b071";
    for (
        file_name,
        operation_id,
        baseline_overload_id,
        baseline_digest,
        target,
        semantic_category,
        semantic,
        expected_digest,
    ) in [
        (
            "sim_ops.json",
            SIM_OPS_OPERATION_ID,
            "COMFY-TENSOR-OP-7D7398921719:reference",
            "e679b10b84eb0f356e3c728f1614334eb78a9fae25860755751f774b1fe89a23",
            "sim.ops",
            "namespace",
            "NamespaceReference::ComfyOps",
            "41b886ee2c5b746312f85b63717294166df3b1303425f7c7e423bd74a4a9202b",
        ),
        (
            "torch_nn.json",
            TORCH_NEURAL_NETWORK_OPERATION_ID,
            "COMFY-TENSOR-OP-764A8E60B071:reference",
            "425efadd438871c2a344f36d48f02038a6d86a683d1a8e285537eb7e190ffd1b",
            "torch.nn",
            "namespace",
            "NamespaceReference::TorchNeuralNetwork",
            "eb849cb8795b52fa6ba8d9f7100f08802ba0aece640939d8c3ed0f4e5ff92d1f",
        ),
        (
            "torch_version_cuda.json",
            TORCH_CUDA_VERSION_OPERATION_ID,
            "COMFY-TENSOR-OP-7A62A3A11490:reference",
            "1fafffdcb8efaa56f20e17f0ff9581d3791e3173df27d9ec1cc528812880b8e8",
            "torch.version.cuda",
            "version-value",
            "VersionValueReference::Cuda",
            "df789a6ae302fbd8c29f2abfbbfdd79b9563dbb02b5fd2b47fea6e0691ffe0a2",
        ),
    ] {
        let fixture_path = workspace_root
            .join("crates/sim_test_support/fixtures/tensor_operations/namespace_contract_01")
            .join(file_name);
        let bytes = fs::read(fixture_path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(digest, expected_digest);
        let fixture: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            required_string(&fixture, "resolution_module")?,
            "namespace_contract_01"
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
        assert_eq!(required_string(&fixture, "owner_task_id")?, owner);
        assert_eq!(required_string(&fixture, "canonical_target")?, target);
        let reference_semantic = fixture
            .get("reference_semantic")
            .ok_or("reference evidence is missing reference_semantic")?;
        assert_eq!(
            required_string(reference_semantic, "category")?,
            semantic_category
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
