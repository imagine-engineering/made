use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    GENERATED_MODULES, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OPERATION_CONTRACTS,
    validate_generated_operation_release_closure, validate_operation_contracts,
    validate_operation_resolution_evidence,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const EXPECTED_CALLABLE_COUNT: usize = 511;
const EXPECTED_EXTERNAL_DISPOSITION_COUNT: usize = 7;
const EXPECTED_REFERENCE_COUNT: usize = 82;
const EXPECTED_RESOLUTION_MODULE_COUNT: usize = 51;
const EXPECTED_EVIDENCE_FILE_COUNT: usize = 511;

#[test]
fn val_tensor_001_exact_native_operation_breadth_closure() -> Result<(), Box<dyn std::error::Error>>
{
    assert_eq!(OPERATION_CONTRACTS.len(), 600);
    validate_operation_contracts(OPERATION_CONTRACTS)?;
    validate_generated_operation_release_closure(OPERATION_CONTRACTS)?;

    let mut references = OPERATION_CONTRACTS
        .iter()
        .filter_map(|contract| contract.typed_reference())
        .map(|reference| (reference.operation_id(), reference))
        .collect::<BTreeMap<_, _>>();
    let reference_count = references.len();
    let resolution_count = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .map(|slice| slice.len())
        .sum::<usize>();
    assert_eq!(reference_count, EXPECTED_REFERENCE_COUNT);
    assert_eq!(resolution_count, EXPECTED_CALLABLE_COUNT);
    assert_eq!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES.len(),
        EXPECTED_RESOLUTION_MODULE_COUNT,
    );
    assert_eq!(
        OPERATION_CONTRACTS.len() - reference_count - resolution_count,
        EXPECTED_EXTERNAL_DISPOSITION_COUNT,
    );

    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let evidence_root = repository_root
        .join("crates/sim_test_support/fixtures/tensor_operations")
        .canonicalize()?;
    let baseline_root =
        repository_root.join("crates/sim_test_support/fixtures/tensor_signatures/contracts");
    let reference_modules = [
        "namespace_contract_01",
        "type_contract_01",
        "type_contract_02",
        "value_or_constant_contract_01",
        "value_or_constant_contract_02",
        "value_or_constant_contract_03",
        "value_or_constant_contract_04",
        "value_or_constant_contract_05",
        "value_or_constant_contract_06",
    ];
    for module in reference_modules {
        for entry in fs::read_dir(evidence_root.join(module))? {
            let path = entry?.path();
            let evidence: Value = serde_json::from_slice(&fs::read(&path)?)?;
            let operation_id = evidence["operation_id"]
                .as_str()
                .ok_or("reference evidence must have an operation ID")?;
            let reference = references.remove(operation_id).ok_or_else(|| {
                format!("unexpected or repeated reference operation {operation_id}")
            })?;
            let baseline_bytes = fs::read(
                baseline_root.join(format!("{}.json", operation_id.to_ascii_lowercase())),
            )?;
            let baseline: Value = serde_json::from_slice(&baseline_bytes)?;
            let digest = format!("{:x}", Sha256::digest(&baseline_bytes));
            assert_eq!(evidence["resolution_module"], module);
            assert_eq!(evidence["canonical_target"], reference.canonical_target());
            assert_eq!(evidence["executable"], false);
            assert_eq!(
                evidence["baseline_fixture_sha256"],
                digest,
                "{}",
                path.display()
            );
            assert_eq!(baseline["operation_id"], operation_id);
            assert_eq!(
                evidence["reference_semantic"],
                baseline["reference_semantic"]
            );
        }
    }
    assert!(
        references.is_empty(),
        "missing typed-reference evidence for {references:?}"
    );

    let mut operation_ids = BTreeSet::new();
    let mut overload_ids = BTreeSet::new();
    let mut evidence_paths = BTreeSet::new();

    for slice in GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES {
        assert!(!slice.is_empty());
        for resolution in slice.iter() {
            assert_eq!(resolution.resolution_module, slice.module_name);
            assert!(operation_ids.insert(resolution.operation_id));
            assert!(overload_ids.insert(resolution.overload_id));
            evidence_paths.insert(resolution.evidence_fixture);
            validate_operation_resolution_evidence(&repository_root, resolution)?;
            assert_ne!(
                resolution.baseline_fixture_sha256,
                resolution.evidence_fixture_sha256,
            );

            let expected_prefix = format!(
                "crates/sim_test_support/fixtures/tensor_operations/{}/",
                slice.module_name,
            );
            assert!(resolution.evidence_fixture.starts_with(&expected_prefix));
            let evidence_path = repository_root.join(resolution.evidence_fixture);
            let canonical_evidence_path = evidence_path.canonicalize()?;
            assert!(canonical_evidence_path.starts_with(&evidence_root));
            let digest = format!("{:x}", Sha256::digest(std::fs::read(evidence_path)?));
            assert_eq!(digest, resolution.evidence_fixture_sha256);
        }
    }

    assert_eq!(operation_ids.len(), EXPECTED_CALLABLE_COUNT);
    assert_eq!(overload_ids.len(), EXPECTED_CALLABLE_COUNT);
    assert_eq!(evidence_paths.len(), EXPECTED_EVIDENCE_FILE_COUNT);

    let expected_backends = [
        "amd_rocm_sim_model_0014",
        "apple_metal_mps_sim_model_0015",
        "cambricon_mlu_sim_model_0017",
        "cpu_sim_model_0016",
        "directml_sim_model_0018",
        "huawei_ascend_npu_sim_model_0019",
        "intel_xpu_sim_model_0021",
        "nvidia_cuda_sim_model_0022",
    ];
    for backend in expected_backends {
        let wrapper = format!("ops/backend_{backend}");
        assert!(GENERATED_MODULES.contains(&wrapper.as_str()));
        assert!(
            repository_root
                .join(format!("crates/sim_tensor/src/ops/backend_{backend}.rs"))
                .is_file(),
        );
        assert!(
            repository_root
                .join(format!("crates/sim_tensor/src/backends/{backend}.rs"))
                .is_file(),
        );
    }
    Ok(())
}
