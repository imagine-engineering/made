use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    FunctionReference, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, NumericConstantReference, OPERATION_CONTRACTS,
    TensorPropertyReference, TypedReferenceContract,
    generated_value_or_constant_contract_01 as leaf,
};
use std::{collections::BTreeSet, fs, io, path::Path};

const OWNER: &str =
    "comfy-parity-tensor-ops-value-or-constant-contract-comfy-tensor-op-137ded7f8918";

type ContractFacade = fn() -> Option<TypedReferenceContract>;

fn reference_cases() -> [(
    ContractFacade,
    &'static str,
    &'static str,
    CanonicalReference,
); 12] {
    [
        (
            leaf::cuda_matmul_allow_fp16_accumulation_contract,
            leaf::CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_OPERATION_ID,
            "torch.backends.cuda.matmul.allow_fp16_accumulation",
            CanonicalReference::BooleanCapability(
                BooleanCapabilityReference::CudaMatmulAllowFp16Accumulation,
            ),
        ),
        (
            leaf::inverse_fft_real_contract,
            leaf::INVERSE_FFT_REAL_OPERATION_ID,
            "torch.fft.ifftn().real",
            CanonicalReference::TensorProperty(TensorPropertyReference::InverseFftReal),
        ),
        (
            leaf::torch_int_contract,
            leaf::TORCH_INT_OPERATION_ID,
            "torch.int",
            CanonicalReference::DType(DType::I32),
        ),
        (
            leaf::torch_int64_contract,
            leaf::TORCH_INT64_OPERATION_ID,
            "torch.int64",
            CanonicalReference::DType(DType::I64),
        ),
        (
            leaf::torch_log10_contract,
            leaf::TORCH_LOG10_OPERATION_ID,
            "torch.log10",
            CanonicalReference::Function(FunctionReference::Log10),
        ),
        (
            leaf::torch_long_contract,
            leaf::TORCH_LONG_OPERATION_ID,
            "torch.long",
            CanonicalReference::DType(DType::I64),
        ),
        (
            leaf::torch_nn_mish_contract,
            leaf::TORCH_NN_MISH_OPERATION_ID,
            "torch.nn.Mish",
            CanonicalReference::Function(FunctionReference::Mish),
        ),
        (
            leaf::torch_pi_contract,
            leaf::TORCH_PI_OPERATION_ID,
            "torch.pi",
            CanonicalReference::NumericConstant(NumericConstantReference::Pi),
        ),
        (
            leaf::torch_uint32_contract,
            leaf::TORCH_UINT32_OPERATION_ID,
            "torch.uint32",
            CanonicalReference::DType(DType::U32),
        ),
        (
            leaf::unique_shape_contract,
            leaf::UNIQUE_SHAPE_OPERATION_ID,
            "torch.unique().shape",
            CanonicalReference::TensorProperty(TensorPropertyReference::UniqueShape),
        ),
        (
            leaf::vandermonde_transpose_contract,
            leaf::VANDERMONDE_TRANSPOSE_OPERATION_ID,
            "torch.vander().T",
            CanonicalReference::TensorProperty(TensorPropertyReference::VandermondeTranspose),
        ),
        (
            leaf::xpu_stream_contract,
            leaf::XPU_STREAM_OPERATION_ID,
            "torch.xpu.stream",
            CanonicalReference::Function(FunctionReference::XpuStream),
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
    for (facade, operation_id, target, expected_semantic) in reference_cases() {
        let contract = facade().ok_or_else(|| format!("typed contract is missing for {target}"))?;
        assert_eq!(contract.operation_id(), operation_id);
        assert_eq!(contract.canonical_target(), target);
        assert_eq!(
            contract.inventory_kind(),
            ContractInventoryKind::NamespaceValueReference
        );
        assert_eq!(contract.semantic(), expected_semantic);
    }
    Ok(())
}

#[test]
fn assigned_catalog_rows_are_exact_and_never_enter_kernel_resolution() {
    let expected_ids = reference_cases()
        .map(|(_, operation_id, _, _)| operation_id)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let assigned_ids = leaf::ASSIGNED_VALUE_OR_CONSTANT_REFERENCES
        .iter()
        .map(|(operation_id, _)| *operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(assigned_ids, expected_ids);
    assert_eq!(assigned_ids.len(), 12);
    assert!(leaf::assigned_value_or_constant_contract("COMFY-TENSOR-OP-UNASSIGNED").is_none());
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"value_or_constant_contract_01"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "value_or_constant_contract_01")
    );
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .all(|contract| !expected_ids.contains(contract.operation_id))
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
        .join("crates/sim_test_support/fixtures/tensor_operations/value_or_constant_contract_01");
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
    for (
        file_name,
        operation_id,
        baseline_digest,
        target,
        semantic_category,
        semantic,
        expected_digest,
    ) in [
        (
            "cuda_matmul_allow_fp16_accumulation.json",
            leaf::CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_OPERATION_ID,
            "153f93db224ad0dbf72ab7b65fe1fff0b91310f46276b789c0dea4edecd4ed9b",
            "torch.backends.cuda.matmul.allow_fp16_accumulation",
            "boolean-capability",
            "BooleanCapabilityReference::CudaMatmulAllowFp16Accumulation",
            "48b331ddc53d8bd46bb43b9aade89d57ec9f67ac16d646a1b66475643f2720a3",
        ),
        (
            "inverse_fft_real.json",
            leaf::INVERSE_FFT_REAL_OPERATION_ID,
            "11879526b6692551f7c9a7224953bd12d62aee2dd11ecf84945e1d35c4ff827b",
            "torch.fft.ifftn().real",
            "tensor-property",
            "TensorPropertyReference::InverseFftReal",
            "d1b5c1a554f75c33579c920a2d4dfbaade9e0817b8616c3139ea436998105a98",
        ),
        (
            "torch_int.json",
            leaf::TORCH_INT_OPERATION_ID,
            "049ad62ffa941fd8aea327421c3a85beae2591c1b7c71c1a6cc92af5fa59e5f4",
            "torch.int",
            "dtype",
            "DType::I32",
            "99875abd1935a739a6ce56c16be996be48bddf415a824aec36985664f2667a87",
        ),
        (
            "torch_int64.json",
            leaf::TORCH_INT64_OPERATION_ID,
            "372115f3cb476554809cb64c1c04949b92050b5a9b6ab8a9023ada67c2575cf5",
            "torch.int64",
            "dtype",
            "DType::I64",
            "151f65006f882f3371993d8cc57787d28335552d43bf4ec93333fabe4366a924",
        ),
        (
            "torch_log10.json",
            leaf::TORCH_LOG10_OPERATION_ID,
            "bd33bea2135b43fdf8908f8d7f64c8160e09fb2fa8fd32df1caaf229c1cf3c11",
            "torch.log10",
            "function-reference",
            "FunctionReference::Log10",
            "554acd7ce554975a948f7d97b313f6c7f13111649ee01dd239076eeb4df25684",
        ),
        (
            "torch_long.json",
            leaf::TORCH_LONG_OPERATION_ID,
            "dc38ba68a6730373a1a1435cf85ca78cad783ecfcec9038309335c582addfca0",
            "torch.long",
            "dtype",
            "DType::I64",
            "c7188eea7630e144a3fa0d2f816c765e231b0d6b432e2e16c77a391605bac1df",
        ),
        (
            "torch_nn_mish.json",
            leaf::TORCH_NN_MISH_OPERATION_ID,
            "6dd7c22ef191362cd3e3e694ad6c8c5c31bda4946951702808b42b18f11438ee",
            "torch.nn.Mish",
            "function-reference",
            "FunctionReference::Mish",
            "ee7b47c43a718838008ee4c81b8a7c7d39e230b5db91a24314fa365ad3cc3918",
        ),
        (
            "torch_pi.json",
            leaf::TORCH_PI_OPERATION_ID,
            "45434b25f66d140aeff674d2a645fd3a1eec84b13c5d10439a4eb6dad80e1b34",
            "torch.pi",
            "numeric-constant",
            "NumericConstantReference::Pi",
            "8038920a1ef25f7b534e4e65b2ee8a847b02f72e10b3360063b7601146fea8bc",
        ),
        (
            "torch_uint32.json",
            leaf::TORCH_UINT32_OPERATION_ID,
            "91fbbb507f5e14b51e450059e09dd3e3f78e09b081f0d6594fc72493d55f47ac",
            "torch.uint32",
            "dtype",
            "DType::U32",
            "d81892480a00a90a62a20e7c62aab66c97f7c36f770654a9432eccee078d932b",
        ),
        (
            "unique_shape.json",
            leaf::UNIQUE_SHAPE_OPERATION_ID,
            "7b48bb453b9b6014899df177db2bc90eada84707a50dc76889c7e3eb71801bd4",
            "torch.unique().shape",
            "tensor-property",
            "TensorPropertyReference::UniqueShape",
            "1ae1fa257158f01bfab44163775b6bebb5badb3ee8d0c90e23e8aca6669a03f5",
        ),
        (
            "vandermonde_transpose.json",
            leaf::VANDERMONDE_TRANSPOSE_OPERATION_ID,
            "725bf55c2a8a7c04a6081b37cc422df57edb6e75283ee0c28a7e8296b85f492b",
            "torch.vander().T",
            "tensor-property",
            "TensorPropertyReference::VandermondeTranspose",
            "af56b1f3e54312d1f2078d72d4e2366d784420ce83ed63012266ba65a93e37d1",
        ),
        (
            "xpu_stream.json",
            leaf::XPU_STREAM_OPERATION_ID,
            "22a692fc2d72668e2a09b0f341c148c9631d317c4411404364db2c2a9553aa48",
            "torch.xpu.stream",
            "function-reference",
            "FunctionReference::XpuStream",
            "0db86d4df9cbf923f2a5d4f9e7f005c4467f9ab39cfa2504f6bf4931d4c36ef6",
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
            "value_or_constant_contract_01"
        );
        assert_eq!(required_string(&fixture, "operation_id")?, operation_id);
        assert_eq!(
            required_string(&fixture, "baseline_overload_id")?,
            format!("{operation_id}:reference")
        );
        assert_eq!(
            required_string(&fixture, "baseline_fixture_sha256")?,
            baseline_digest
        );
        assert_eq!(required_string(&fixture, "owner_task_id")?, OWNER);
        assert_eq!(
            required_string(&fixture, "inventory_kind")?,
            "namespace_value_reference"
        );
        assert_eq!(required_string(&fixture, "canonical_target")?, target);
        let reference_semantic = fixture
            .get("reference_semantic")
            .ok_or("reference evidence is missing reference_semantic")?;
        assert_eq!(
            reference_semantic
                .as_object()
                .ok_or("reference semantic must be an object")?
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["category", "value"])
        );
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
    }
    Ok(())
}
