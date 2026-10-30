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
            "7bb31566c57fb1121849b41137274df5fb64423a4d334299088f5db705fcdd79",
            "torch.backends.cuda.matmul.allow_tf32",
            "boolean-capability",
            "BooleanCapabilityReference::CudaMatmulAllowTf32",
            "7198df69aa99fbaa66476478e0983acad907a9d9791f208c418f5bc7158d042c",
        ),
        (
            "cudnn_allow_tf32.json",
            leaf::CUDNN_ALLOW_TF32_OPERATION_ID,
            "d9e7f81011aaa537f8fbcd3734ff52ecd99e36cb87bdaffcf1cc132ca1cfc1fc",
            "torch.backends.cudnn.allow_tf32",
            "boolean-capability",
            "BooleanCapabilityReference::CudnnAllowTf32",
            "84d8f39d87a1d4ad23f52893732e5518fbfca271bb3f6191328cf1ebdefde57d",
        ),
        (
            "cudnn_enabled.json",
            leaf::CUDNN_ENABLED_OPERATION_ID,
            "de79f08a6ae3d7d6e87d07794ccb8748dd1a86f4b6d6d414a12054c214cb84b8",
            "torch.backends.cudnn.enabled",
            "boolean-capability",
            "BooleanCapabilityReference::CudnnEnabled",
            "4f3a5def075cc06fc52cd4b3773e437b3015d0040ff67b08c63de4847bc87744",
        ),
        (
            "channels_last.json",
            leaf::CHANNELS_LAST_OPERATION_ID,
            "96f1413d7731b572e5f9d8ab41023245d2458cbcfc3ab95fa5217478457e9fc8",
            "torch.channels_last",
            "layout-or-memory-format",
            "MemoryFormatReference::Layout(Layout::ChannelsLast)",
            "809171a7826a59f921f579a20ee71875da54e925133016728c35887f70960046",
        ),
        (
            "cuda_out_of_memory_error.json",
            leaf::CUDA_OUT_OF_MEMORY_ERROR_OPERATION_ID,
            "34d54f031433087ebbb2b3b65f7a72ded0d207f4a6801e8c77012f4a814fa8a2",
            "torch.cuda.OutOfMemoryError",
            "type-marker",
            "TypeMarkerReference::CudaOutOfMemoryError",
            "dc7ad4df9abe2bef98d16c5a50b98bbe0086daa5647568a34bc5049e02d897d4",
        ),
        (
            "float_info_bits.json",
            leaf::FLOAT_INFO_BITS_OPERATION_ID,
            "e612cb63e1467365867ebc87df5e16f59ebcc74026126c3af155fcf23b30e08e",
            "torch.finfo().bits",
            "numeric-constant",
            "NumericConstantReference::FloatInfoBits",
            "56d798325ab30d6cf2ebef3a09bb05322f559966f15e13347ba80d891fa45e2e",
        ),
        (
            "float_info_maximum.json",
            leaf::FLOAT_INFO_MAXIMUM_OPERATION_ID,
            "1fcac4a84499e05833599d20fb1129944ceff2bbe7d514d9d9ea9dda21179473",
            "torch.finfo().max",
            "numeric-constant",
            "NumericConstantReference::FloatInfoMaximum",
            "a60bd4310f7a4043741412bef45b50cb5ce1feef811c50a680dab0fbaa9bd6fe",
        ),
        (
            "torch_float32.json",
            leaf::TORCH_FLOAT32_OPERATION_ID,
            "4c260fdf809b6e79c3922279f02e3b3c11f7c3e3852af2c42668fd1fc1f3e75f",
            "torch.float32",
            "dtype",
            "DType::F32",
            "936628b3c2f11b31517e1a5cdebf5d7c88efc1979404b8ce9ebae13e2e715092",
        ),
        (
            "torch_nn_hardtanh.json",
            leaf::TORCH_NN_HARDTANH_OPERATION_ID,
            "9b9322477af42bc0af4261a6ec72fe0265da2719d8f7b96eae180d7facd5a189",
            "torch.nn.Hardtanh",
            "function-reference",
            "FunctionReference::Hardtanh",
            "9666b60d527209680c07c61bd5372a40134f226ab9785edf247ae406cbb8db68",
        ),
        (
            "sdp_cudnn_attention.json",
            leaf::SDP_CUDNN_ATTENTION_OPERATION_ID,
            "046556492790c7a831f4fba8ad326615ac08910e89d90e779cf3d2de8bb9dad3",
            "torch.nn.attention.SDPBackend.CUDNN_ATTENTION",
            "enum-variant",
            "EnumVariantReference::SdpCudnnAttention",
            "6ad0d31885f5588b732f1242a314ef37d3251a948eb9ae8c636f0df08dc919d0",
        ),
        (
            "sdp_efficient_attention.json",
            leaf::SDP_EFFICIENT_ATTENTION_OPERATION_ID,
            "2fe67cf923057759d8a1a8ad6d8a7fcee9a1ad7dc1a6e99d66539ea0a1c1cc1f",
            "torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION",
            "enum-variant",
            "EnumVariantReference::SdpEfficientAttention",
            "fcfafe216373c369ede379afa0181d413c22ccb2e36e090cc8fe1cf53ddedb96",
        ),
        (
            "sdp_math.json",
            leaf::SDP_MATH_OPERATION_ID,
            "cd8da4992786d6dd51059c0288c9542370fa432a346e80cc8f2557930428928a",
            "torch.nn.attention.SDPBackend.MATH",
            "enum-variant",
            "EnumVariantReference::SdpMath",
            "5cdaa3476f6f0aafd2d1b64f0028bb0c1f06d24ea7ab2cfc229caebc304f0642",
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
