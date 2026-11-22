use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    EnumVariantReference, FunctionReference, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, MemoryFormatReference,
    NumericConstantReference, OPERATION_CONTRACTS, TypeMarkerReference, TypedReferenceContract,
    generated_value_or_constant_contract_04 as leaf,
};
use std::{collections::BTreeSet, fs, io, path::Path};

const OWNER: &str =
    "comfy-parity-tensor-ops-value-or-constant-contract-comfy-tensor-op-8a525d4e1849";
type ContractFacade = fn() -> Option<TypedReferenceContract>;

fn reference_cases() -> [(
    ContractFacade,
    &'static str,
    &'static str,
    CanonicalReference,
); 12] {
    [
        (
            leaf::cuda_matmul_allow_tf32_contract,
            leaf::CUDA_MATMUL_ALLOW_TF32_OPERATION_ID,
            "torch.backends.cuda.matmul.allow_tf32",
            CanonicalReference::BooleanCapability(BooleanCapabilityReference::CudaMatmulAllowTf32),
        ),
        (
            leaf::cudnn_allow_tf32_contract,
            leaf::CUDNN_ALLOW_TF32_OPERATION_ID,
            "torch.backends.cudnn.allow_tf32",
            CanonicalReference::BooleanCapability(BooleanCapabilityReference::CudnnAllowTf32),
        ),
        (
            leaf::cudnn_enabled_contract,
            leaf::CUDNN_ENABLED_OPERATION_ID,
            "torch.backends.cudnn.enabled",
            CanonicalReference::BooleanCapability(BooleanCapabilityReference::CudnnEnabled),
        ),
        (
            leaf::channels_last_contract,
            leaf::CHANNELS_LAST_OPERATION_ID,
            "torch.channels_last",
            CanonicalReference::MemoryFormat(MemoryFormatReference::Layout(Layout::ChannelsLast)),
        ),
        (
            leaf::cuda_out_of_memory_error_contract,
            leaf::CUDA_OUT_OF_MEMORY_ERROR_OPERATION_ID,
            "torch.cuda.OutOfMemoryError",
            CanonicalReference::TypeMarker(TypeMarkerReference::CudaOutOfMemoryError),
        ),
        (
            leaf::float_info_bits_contract,
            leaf::FLOAT_INFO_BITS_OPERATION_ID,
            "torch.finfo().bits",
            CanonicalReference::NumericConstant(NumericConstantReference::FloatInfoBits),
        ),
        (
            leaf::float_info_maximum_contract,
            leaf::FLOAT_INFO_MAXIMUM_OPERATION_ID,
            "torch.finfo().max",
            CanonicalReference::NumericConstant(NumericConstantReference::FloatInfoMaximum),
        ),
        (
            leaf::torch_float32_contract,
            leaf::TORCH_FLOAT32_OPERATION_ID,
            "torch.float32",
            CanonicalReference::DType(DType::F32),
        ),
        (
            leaf::torch_nn_hardtanh_contract,
            leaf::TORCH_NN_HARDTANH_OPERATION_ID,
            "torch.nn.Hardtanh",
            CanonicalReference::Function(FunctionReference::Hardtanh),
        ),
        (
            leaf::sdp_cudnn_attention_contract,
            leaf::SDP_CUDNN_ATTENTION_OPERATION_ID,
            "torch.nn.attention.SDPBackend.CUDNN_ATTENTION",
            CanonicalReference::EnumVariant(EnumVariantReference::SdpCudnnAttention),
        ),
        (
            leaf::sdp_efficient_attention_contract,
            leaf::SDP_EFFICIENT_ATTENTION_OPERATION_ID,
            "torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION",
            CanonicalReference::EnumVariant(EnumVariantReference::SdpEfficientAttention),
        ),
        (
            leaf::sdp_math_contract,
            leaf::SDP_MATH_OPERATION_ID,
            "torch.nn.attention.SDPBackend.MATH",
            CanonicalReference::EnumVariant(EnumVariantReference::SdpMath),
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
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"value_or_constant_contract_04"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "value_or_constant_contract_04")
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
        .join("crates/sim_test_support/fixtures/tensor_operations/value_or_constant_contract_04");
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
    for (file_name, operation_id, baseline_digest, target, category, semantic, expected_digest) in [
        (
            "cuda_matmul_allow_tf32.json",
            leaf::CUDA_MATMUL_ALLOW_TF32_OPERATION_ID,
            "88917ab9c5a38c2346eba351f66a51e5cb2985532b384be8af82f7da917d3fee",
            "torch.backends.cuda.matmul.allow_tf32",
            "boolean-capability",
            "BooleanCapabilityReference::CudaMatmulAllowTf32",
            "15f5b7f4885aef02bcd2164b1d7f53b59d7be402bba3ce62ee0f97def8e825a5",
        ),
        (
            "cudnn_allow_tf32.json",
            leaf::CUDNN_ALLOW_TF32_OPERATION_ID,
            "1b1a3c06762fad8ca2b89c0efbb3c33ea8dc03db9f770c9f3a97dd3eff73945d",
            "torch.backends.cudnn.allow_tf32",
            "boolean-capability",
            "BooleanCapabilityReference::CudnnAllowTf32",
            "fd9572b4d80b769b3b36b0c118f919098ec9761f2b3d4d4ffceef6911a04c945",
        ),
        (
            "cudnn_enabled.json",
            leaf::CUDNN_ENABLED_OPERATION_ID,
            "3fd48e18f8fe138b31f39256d14a9bab2d1a9bc5f5cd5c96d5b7678d681616fe",
            "torch.backends.cudnn.enabled",
            "boolean-capability",
            "BooleanCapabilityReference::CudnnEnabled",
            "2849e43d55a25f40898464f0b5b74f923048ec1394704e97fcb57ea1b654f896",
        ),
        (
            "channels_last.json",
            leaf::CHANNELS_LAST_OPERATION_ID,
            "83828d854a52d390f6b2c3a2dc823771965fbda7318745bbf50ffd2c100cd8ab",
            "torch.channels_last",
            "layout-or-memory-format",
            "MemoryFormatReference::Layout(Layout::ChannelsLast)",
            "5827c2fadb6eb758b379b967aef3c4493a9ada2fb45581a263941ae1b991a06e",
        ),
        (
            "cuda_out_of_memory_error.json",
            leaf::CUDA_OUT_OF_MEMORY_ERROR_OPERATION_ID,
            "81f0c343326501786a610f00169056d6504dfa3d74841e851bba9c7928959215",
            "torch.cuda.OutOfMemoryError",
            "type-marker",
            "TypeMarkerReference::CudaOutOfMemoryError",
            "30b6368870a74e4557f8dde589c47f435b768b9fa2dac96a873f2e6904d315e6",
        ),
        (
            "float_info_bits.json",
            leaf::FLOAT_INFO_BITS_OPERATION_ID,
            "1839be4a32506d5f74fd0c4a2c5d12a33a624e4ff5985e521aa2e4cc4f9a2200",
            "torch.finfo().bits",
            "numeric-constant",
            "NumericConstantReference::FloatInfoBits",
            "e491c3b16020a8869c844ec19f0d4dff23038a07863e147bb439f5e9a28bdfb9",
        ),
        (
            "float_info_maximum.json",
            leaf::FLOAT_INFO_MAXIMUM_OPERATION_ID,
            "5a2fd40ce8a7414d99a0854d2193c2ac028c029bfad921be4f5bf671bd4864af",
            "torch.finfo().max",
            "numeric-constant",
            "NumericConstantReference::FloatInfoMaximum",
            "0889d1536b143f8f21e92b87144a94d774d810fa97ba7fae0e39c473dea04f2a",
        ),
        (
            "torch_float32.json",
            leaf::TORCH_FLOAT32_OPERATION_ID,
            "64b19d186e0f5645f9b08eef42373c310a7163f96cad881151d2817473937add",
            "torch.float32",
            "dtype",
            "DType::F32",
            "87b6750487672b712cca410c50e6a99dfb5efaf7af7bfb799e99085c86b7e77e",
        ),
        (
            "torch_nn_hardtanh.json",
            leaf::TORCH_NN_HARDTANH_OPERATION_ID,
            "9c8c112126af1c761924d744ef8ef7962cf1b069f538abb726b2d11811b59cae",
            "torch.nn.Hardtanh",
            "function-reference",
            "FunctionReference::Hardtanh",
            "9448cc5f7566de51db853f46dac416884bfcabb9be5a0b6638a9779f7efa4ee7",
        ),
        (
            "sdp_cudnn_attention.json",
            leaf::SDP_CUDNN_ATTENTION_OPERATION_ID,
            "0a5899e5a5ff6ab21165c83dfd4d9780840f902014a4e649bc4ff04b5b6f201f",
            "torch.nn.attention.SDPBackend.CUDNN_ATTENTION",
            "enum-variant",
            "EnumVariantReference::SdpCudnnAttention",
            "83fc19926b53764c5df9043571efbb08e1151be8ea9a948b0582d558b46c2480",
        ),
        (
            "sdp_efficient_attention.json",
            leaf::SDP_EFFICIENT_ATTENTION_OPERATION_ID,
            "edd0f49a81c2e093dafea7cfeb8bc873b04bb3e9d8257029b7c788a2f2d276ab",
            "torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION",
            "enum-variant",
            "EnumVariantReference::SdpEfficientAttention",
            "d7d64c7b72ac5edb04d349ceddc20d81503af1f326db946d72164040644a69df",
        ),
        (
            "sdp_math.json",
            leaf::SDP_MATH_OPERATION_ID,
            "03fcd689ec3a616e205c2246aa2ccb8234dabaf7f4706dbdaf4d2930b4aebc2c",
            "torch.nn.attention.SDPBackend.MATH",
            "enum-variant",
            "EnumVariantReference::SdpMath",
            "5e9ca693a211dddb5f7d5390246b193b08b6afe3982059d2295151254c36404f",
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
            "value_or_constant_contract_04"
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
        assert_eq!(required_string(reference_semantic, "category")?, category);
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
