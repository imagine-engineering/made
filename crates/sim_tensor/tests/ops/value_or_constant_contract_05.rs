use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    ContractInventoryKind, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OPERATION_CONTRACTS, TypedReferenceContract,
    generated_value_or_constant_contract_05 as leaf,
};
use std::{collections::BTreeSet, fs, io, path::Path};

const OWNER: &str =
    "comfy-parity-tensor-ops-value-or-constant-contract-comfy-tensor-op-b37b2e52befe";
type ContractFacade = fn() -> Option<TypedReferenceContract>;

fn facades() -> [ContractFacade; 12] {
    [
        leaf::cudnn_benchmark_contract,
        leaf::torch_complex64_contract,
        leaf::torch_float_contract,
        leaf::torch_float8_e4m3fn_contract,
        leaf::torch_float8_e5m2fnuz_contract,
        leaf::torch_int16_contract,
        leaf::torch_nn_hardswish_contract,
        leaf::torch_nn_selu_contract,
        leaf::xpu_total_memory_contract,
        leaf::interpolation_nearest_contract,
        leaf::functional_interpolation_bicubic_contract,
        leaf::xformers_module_version_contract,
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
    assert_eq!(leaf::ASSIGNED_VALUE_OR_CONSTANT_REFERENCES.len(), 12);
    for (facade, (operation_id, semantic)) in facades()
        .into_iter()
        .zip(leaf::ASSIGNED_VALUE_OR_CONSTANT_REFERENCES)
    {
        let contract =
            facade().ok_or_else(|| format!("typed contract is missing for {operation_id}"))?;
        assert_eq!(contract.operation_id(), *operation_id);
        assert_eq!(contract.semantic(), *semantic);
        assert_eq!(
            contract.inventory_kind(),
            ContractInventoryKind::NamespaceValueReference
        );
        assert_eq!(
            leaf::assigned_value_or_constant_contract(operation_id),
            Some(contract)
        );
    }
    assert!(leaf::assigned_value_or_constant_contract("COMFY-TENSOR-OP-UNASSIGNED").is_none());
    Ok(())
}

#[test]
fn assigned_catalog_rows_are_exact_and_never_enter_kernel_resolution() {
    let assigned_ids = leaf::ASSIGNED_VALUE_OR_CONSTANT_REFERENCES
        .iter()
        .map(|(operation_id, _)| *operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(assigned_ids.len(), 12);
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"value_or_constant_contract_05"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "value_or_constant_contract_05")
    );
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .all(|contract| !assigned_ids.contains(contract.operation_id))
    );
}

#[test]
fn no_unassigned_catalog_row_is_claimed() {
    let assigned_operation_ids = leaf::ASSIGNED_VALUE_OR_CONSTANT_REFERENCES
        .iter()
        .map(|(operation_id, _)| *operation_id)
        .collect::<BTreeSet<_>>();
    let mut claimed_operation_ids = BTreeSet::new();
    for contract in OPERATION_CONTRACTS {
        let Some(reference) = contract.typed_reference() else {
            continue;
        };
        if assigned_operation_ids.contains(reference.operation_id()) {
            assert!(leaf::assigned_value_or_constant_contract(reference.operation_id()).is_some());
            assert!(claimed_operation_ids.insert(reference.operation_id()));
        } else {
            assert!(leaf::assigned_value_or_constant_contract(reference.operation_id()).is_none());
        }
    }
    assert_eq!(claimed_operation_ids, assigned_operation_ids);
}

#[test]
fn typed_reference_evidence_is_exact_and_hash_sealed() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_directory = workspace_root
        .join("crates/sim_test_support/fixtures/tensor_operations/value_or_constant_contract_05");
    let fixture_entries = fs::read_dir(&fixture_directory)?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(fixture_entries.len(), 12);
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
    for (file_name, expected_semantic, expected_digest) in [
        (
            "cudnn_benchmark.json",
            "BooleanCapabilityReference::CudnnBenchmark",
            "74922f7a1790cc3a72ec843b5eb299cd0a1bce5c9b2b904c2974d35b57a72b1b",
        ),
        (
            "torch_complex64.json",
            "DType::Complex64",
            "a5509bbc0dcb00e0b6012ded3504b3f1baaa9d051c352c8c8724000282b4fa0f",
        ),
        (
            "torch_float.json",
            "DType::F32",
            "14cf499b4d0c047da2b0f1a6deb18d66de5ad62dede60665ca3737915ba547c4",
        ),
        (
            "torch_float8_e4m3fn.json",
            "DType::Float8E4m3Fn",
            "7b38c5004b2f2407aca8e95e5bf6ddb6a0f37f3ac914137d90cab4417428871c",
        ),
        (
            "torch_float8_e5m2fnuz.json",
            "DType::Float8E5m2Fnuz",
            "4197168e5d437a39ca01ee221d365c4a19223ef06fd34eb59d4627ab93b0817a",
        ),
        (
            "torch_int16.json",
            "DType::I16",
            "70408a51c8f67cf51d21ad82a5d522c6f8f7e2e9e8659278a23919fd6de943e0",
        ),
        (
            "torch_nn_hardswish.json",
            "FunctionReference::Hardswish",
            "fb88bfaf07908313cf7f137e6329f2cace2a71cfa1200140a29fdbfa1262fb6e",
        ),
        (
            "torch_nn_selu.json",
            "FunctionReference::Selu",
            "1eb1a772a52bde6dac53b7f4bfa515ef86d930a54b1a7a8f1577e370cdaab4fe",
        ),
        (
            "xpu_total_memory.json",
            "DevicePropertyReference::XpuTotalMemory",
            "c361fc22bb3aaf59c964c29a76df2c774339aebade40e1eb45ec5597284da3d5",
        ),
        (
            "interpolation_nearest.json",
            "EnumVariantReference::InterpolationNearest",
            "f7b71f81b8342d794ee11edc1d71ac06e14a6803eb962e90002f195b992c15fc",
        ),
        (
            "functional_interpolation_bicubic.json",
            "EnumVariantReference::FunctionalInterpolationBicubic",
            "ef358e6427472ea68489f1b7de6178cb9a96638955b3713496fdf3745282ebd7",
        ),
        (
            "xformers_module_version.json",
            "VersionValueReference::XformersModule",
            "34d0aa21ee10ab6c36cd056e40b3d9317c4c9acec6893a912ae89ac7d1dcfd33",
        ),
    ] {
        let fixture_path = fixture_directory.join(file_name);
        let bytes = fs::read(fixture_path)?;
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected_digest);
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
            "value_or_constant_contract_05"
        );
        assert_eq!(required_string(&fixture, "owner_task_id")?, OWNER);
        assert_eq!(
            required_string(&fixture, "inventory_kind")?,
            "namespace_value_reference"
        );
        assert_eq!(
            required_string(&fixture, "canonical_rust_semantic")?,
            expected_semantic
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
            let path = required_string(observation, "path")?;
            assert!(!path.is_empty());
            assert!(!Path::new(path).is_absolute());
            assert!(!Path::new(path).components().any(|component| matches!(
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

        let operation_id = required_string(&fixture, "operation_id")?;
        let contract = leaf::assigned_value_or_constant_contract(operation_id)
            .ok_or_else(|| format!("fixture claims an unassigned operation {operation_id}"))?;
        assert_eq!(
            required_string(&fixture, "canonical_target")?,
            contract.canonical_target()
        );
        assert_eq!(
            required_string(&fixture, "baseline_overload_id")?,
            format!("{operation_id}:reference")
        );

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
        assert_eq!(
            required_string(&baseline, "canonical_target")?,
            contract.canonical_target()
        );
        assert_eq!(
            baseline.get("reference_semantic"),
            fixture.get("reference_semantic")
        );
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
    }
    Ok(())
}
