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
            "487fdd43d983370d0259636a04c95587e6b3884896c6acf107624edfe21fc28b",
            "torch.backends.cuda.matmul.allow_fp16_accumulation",
            "boolean-capability",
            "BooleanCapabilityReference::CudaMatmulAllowFp16Accumulation",
            "308f8cab5e461708f4a8f67d7a882f44b1be9c185dfee97f5b21285da9cf12b0",
        ),
        (
            "inverse_fft_real.json",
            leaf::INVERSE_FFT_REAL_OPERATION_ID,
            "1b5ce6a8449725085e8d1ddb9ed000c5dfe4febb5e5fea6fac3e61b3b57f4f16",
            "torch.fft.ifftn().real",
            "tensor-property",
            "TensorPropertyReference::InverseFftReal",
            "5a02c0a040a2c799a2d35b19d2aee6e37418500672f26aaedc45772b027d2e77",
        ),
        (
            "torch_int.json",
            leaf::TORCH_INT_OPERATION_ID,
            "401d18c7a5a53a5581c98b8aebe7cc953402a842df3216834a9513fcd1a0a634",
            "torch.int",
            "dtype",
            "DType::I32",
            "c5ae706c027220f3d481328baf57e980b818cc5a533479c07720cfb9af7b42f0",
        ),
        (
            "torch_int64.json",
            leaf::TORCH_INT64_OPERATION_ID,
            "fdbf2f6441abcedc9c7731b7ac3a68b82897636835065c4b5dd14a03fc7aefe8",
            "torch.int64",
            "dtype",
            "DType::I64",
            "60d715d29c01046f9d17ff0a56471060e4d2e39ca16b864bc2a52fe6f6b19d0f",
        ),
        (
            "torch_log10.json",
            leaf::TORCH_LOG10_OPERATION_ID,
            "44c22ed0ad7f116b8799ba9404ebdd2ebbf6d9143a4322fe16772c48d481cbdb",
            "torch.log10",
            "function-reference",
            "FunctionReference::Log10",
            "1af1c7fdda92538d7b3883e8b6e4c182b9c949cc4492765aaae500f68f1ce05a",
        ),
        (
            "torch_long.json",
            leaf::TORCH_LONG_OPERATION_ID,
            "d0840ec303592b94113a2fb64040f039953ea09a4e6d30211c759dabc45ed30e",
            "torch.long",
            "dtype",
            "DType::I64",
            "436b3519bea6e3a02f0970973ab75be5c8d63f5a9f260bd795985ab4514e5982",
        ),
        (
            "torch_nn_mish.json",
            leaf::TORCH_NN_MISH_OPERATION_ID,
            "ba74debc97bd2a540e11288095dc518c09cd0b985bdfc72802291d080df91588",
            "torch.nn.Mish",
            "function-reference",
            "FunctionReference::Mish",
            "846fd58d483c240da3bae55f80e97b64ee6f7912fed1b8149e52b2d1e8194830",
        ),
        (
            "torch_pi.json",
            leaf::TORCH_PI_OPERATION_ID,
            "1f660550f8044a558aa9b4fc608d599282168b2a61a167263a1576c3aba7ba42",
            "torch.pi",
            "numeric-constant",
            "NumericConstantReference::Pi",
            "5a2ce8cae24dd87a42e0dc7debaef4680fb8cf725328ff6dcaf47942bebb935f",
        ),
        (
            "torch_uint32.json",
            leaf::TORCH_UINT32_OPERATION_ID,
            "9192b2b4fb23748dcf022566a6e3a83715282db2e4d60699768aeba7e3081ff8",
            "torch.uint32",
            "dtype",
            "DType::U32",
            "d0eec8b14e372456ba06d6ee98b93de478aec2bd3c6aee2daa84863c7c52bc14",
        ),
        (
            "unique_shape.json",
            leaf::UNIQUE_SHAPE_OPERATION_ID,
            "34475726bde13c3c1d4b185f01be5c817df0f2cf192634495ce56850b1bf874b",
            "torch.unique().shape",
            "tensor-property",
            "TensorPropertyReference::UniqueShape",
            "7e0e2c1f3db767bb15ac064164a25a8e8b2fd6af23c0caf60e3dc4a30659f21f",
        ),
        (
            "vandermonde_transpose.json",
            leaf::VANDERMONDE_TRANSPOSE_OPERATION_ID,
            "85434e561ba34d62be66f99fbf5ea12d04fc4b853cff1943f059bbbabed56f3c",
            "torch.vander().T",
            "tensor-property",
            "TensorPropertyReference::VandermondeTranspose",
            "56a0f040fac014bbfbbf367dce17499816834814adb58463e19ae9df84feaf58",
        ),
        (
            "xpu_stream.json",
            leaf::XPU_STREAM_OPERATION_ID,
            "636d0b02eefade96231831593c5a00a98ab562fa9feaa01ee09c4d9fcdab998e",
            "torch.xpu.stream",
            "function-reference",
            "FunctionReference::XpuStream",
            "0918f862da45624de3bbcefdc1601491eb201b9ee4cfe4d4d1a634cdb552eafa",
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
