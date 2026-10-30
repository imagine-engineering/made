use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CanonicalReference, ContractInventoryKind, DType, DevicePropertyReference,
    EnumVariantReference, FunctionReference, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, NumericConstantReference, OPERATION_CONTRACTS,
    TensorPropertyReference, TypeMarkerReference, TypedReferenceContract, VersionValueReference,
    generated_value_or_constant_contract_03 as leaf,
};
use std::{collections::BTreeSet, fs, io, path::Path};

const OWNER: &str =
    "comfy-parity-tensor-ops-value-or-constant-contract-comfy-tensor-op-6542124fe760";

type ContractFacade = fn() -> Option<TypedReferenceContract>;

fn reference_cases() -> [(
    ContractFacade,
    &'static str,
    &'static str,
    CanonicalReference,
); 12] {
    [
        (
            leaf::accelerator_error_contract,
            leaf::ACCELERATOR_ERROR_OPERATION_ID,
            "torch.AcceleratorError",
            CanonicalReference::TypeMarker(TypeMarkerReference::AcceleratorError),
        ),
        (
            leaf::autograd_once_differentiable_contract,
            leaf::AUTOGRAD_ONCE_DIFFERENTIABLE_OPERATION_ID,
            "torch.autograd.function.once_differentiable",
            CanonicalReference::Function(FunctionReference::AutogradOnceDifferentiable),
        ),
        (
            leaf::cuda_gcn_architecture_name_contract,
            leaf::CUDA_GCN_ARCHITECTURE_NAME_OPERATION_ID,
            "torch.cuda.get_device_properties().gcnArchName",
            CanonicalReference::DeviceProperty(DevicePropertyReference::CudaGcnArchitectureName),
        ),
        (
            leaf::float_info_minimum_contract,
            leaf::FLOAT_INFO_MINIMUM_OPERATION_ID,
            "torch.finfo().min",
            CanonicalReference::NumericConstant(NumericConstantReference::FloatInfoMinimum),
        ),
        (
            leaf::torch_float16_contract,
            leaf::TORCH_FLOAT16_OPERATION_ID,
            "torch.float16",
            CanonicalReference::DType(DType::F16),
        ),
        (
            leaf::torch_float8_e4m3fnuz_contract,
            leaf::TORCH_FLOAT8_E4M3FNUZ_OPERATION_ID,
            "torch.float8_e4m3fnuz",
            CanonicalReference::DType(DType::Float8E4m3Fnuz),
        ),
        (
            leaf::torch_float8_e5m2_contract,
            leaf::TORCH_FLOAT8_E5M2_OPERATION_ID,
            "torch.float8_e5m2",
            CanonicalReference::DType(DType::Float8E5m2),
        ),
        (
            leaf::torch_infinity_contract,
            leaf::TORCH_INFINITY_OPERATION_ID,
            "torch.inf",
            CanonicalReference::NumericConstant(NumericConstantReference::Infinity),
        ),
        (
            leaf::median_values_contract,
            leaf::MEDIAN_VALUES_OPERATION_ID,
            "torch.median().values",
            CanonicalReference::TensorProperty(TensorPropertyReference::MedianValues),
        ),
        (
            leaf::sdp_flash_attention_contract,
            leaf::SDP_FLASH_ATTENTION_OPERATION_ID,
            "torch.nn.attention.SDPBackend.FLASH_ATTENTION",
            CanonicalReference::EnumVariant(EnumVariantReference::SdpFlashAttention),
        ),
        (
            leaf::torch_uint64_contract,
            leaf::TORCH_UINT64_OPERATION_ID,
            "torch.uint64",
            CanonicalReference::DType(DType::U64),
        ),
        (
            leaf::torch_version_contract,
            leaf::TORCH_VERSION_OPERATION_ID,
            "torch.version.__version__",
            CanonicalReference::VersionValue(VersionValueReference::Torch),
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
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"value_or_constant_contract_03"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "value_or_constant_contract_03")
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
fn typed_reference_evidence_is_exact_distinct_and_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_directory = workspace_root
        .join("crates/sim_test_support/fixtures/tensor_operations/value_or_constant_contract_03");
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
    let fixture_cases = [
        (
            "accelerator_error.json",
            leaf::ACCELERATOR_ERROR_OPERATION_ID,
            "d8dfb2decc09a579fc0cc863c7e63c089dddabc341138808b41654866409b6ae",
            "torch.AcceleratorError",
            "type-marker",
            "TypeMarkerReference::AcceleratorError",
            "2b7fc5f2fc17709d102ed45dd1d80b990ddfaab697feee9fe52c775fffd1104f",
        ),
        (
            "autograd_once_differentiable.json",
            leaf::AUTOGRAD_ONCE_DIFFERENTIABLE_OPERATION_ID,
            "90c080b2c77b4a8bf4df6dea7d9b224dbf6aa797a33b2270d8d6a26d7a61d58b",
            "torch.autograd.function.once_differentiable",
            "function-reference",
            "FunctionReference::AutogradOnceDifferentiable",
            "2fc2b8cebdb8d52a5159fe4d4ea55df03bc0bdccb6a966e721128c6f81dfe496",
        ),
        (
            "cuda_gcn_architecture_name.json",
            leaf::CUDA_GCN_ARCHITECTURE_NAME_OPERATION_ID,
            "d3ede1c3d1ec7bb04f8c51ec45bcd6b06eb49d5c86e48808a79fe5234e95109c",
            "torch.cuda.get_device_properties().gcnArchName",
            "device-property",
            "DevicePropertyReference::CudaGcnArchitectureName",
            "5de7f31118fbc13737f381856eec1325dc69056c302903de646c10b036792604",
        ),
        (
            "float_info_minimum.json",
            leaf::FLOAT_INFO_MINIMUM_OPERATION_ID,
            "2c7d8f19f02bcf7f1398393c4b48abcc8b6fcc221095af91b410014875b8e40c",
            "torch.finfo().min",
            "numeric-constant",
            "NumericConstantReference::FloatInfoMinimum",
            "c17f9541a755ee4bb54e20e625b0ec26c2ce5900546a582167c268a8c8d36fb5",
        ),
        (
            "torch_float16.json",
            leaf::TORCH_FLOAT16_OPERATION_ID,
            "08746432ce89562750a9ad487a4fdfc779b26fac9746c09bf1226568056c3aeb",
            "torch.float16",
            "dtype",
            "DType::F16",
            "ee356db3c5719cb5654fa142311cef2cf60b99045cea102a4b0993afe4a53cc2",
        ),
        (
            "torch_float8_e4m3fnuz.json",
            leaf::TORCH_FLOAT8_E4M3FNUZ_OPERATION_ID,
            "6a1aee7dc4cac65f50c0fc865e497b78788bced343cbaeab98274298b04bb99a",
            "torch.float8_e4m3fnuz",
            "dtype",
            "DType::Float8E4m3Fnuz",
            "640040f5eed56ccd13d4df3f5611ed7fbf652321cdb67cf19cc58f0f4a80450a",
        ),
        (
            "torch_float8_e5m2.json",
            leaf::TORCH_FLOAT8_E5M2_OPERATION_ID,
            "37dfdcbfe99ad6f8ffc872894a20b7f9330e621e01e8c2f4c6220544e107bfc3",
            "torch.float8_e5m2",
            "dtype",
            "DType::Float8E5m2",
            "17e6a4cc26ba3254fc0aa57065df939a72c0cd024cfa834287543eb5c5d2d663",
        ),
        (
            "torch_infinity.json",
            leaf::TORCH_INFINITY_OPERATION_ID,
            "b8e4a8f05918b8a3a6eaacf60c5a7400d525d266aacb40fddb380bba362f802b",
            "torch.inf",
            "numeric-constant",
            "NumericConstantReference::Infinity",
            "91604279a32d610977e263b82f9401280d20525a1a3e92910124d7fd2f2c04b5",
        ),
        (
            "median_values.json",
            leaf::MEDIAN_VALUES_OPERATION_ID,
            "89df1ff77173889c9750ab1b14d7a4f13e6e5ea831eb54faabab061a74571147",
            "torch.median().values",
            "tensor-property",
            "TensorPropertyReference::MedianValues",
            "c9e2869715c7b4d58d807379f5eac8caafdf9a103e6b8a74fd08a8fbd6b2eb56",
        ),
        (
            "sdp_flash_attention.json",
            leaf::SDP_FLASH_ATTENTION_OPERATION_ID,
            "52ddda17c93e73961b6d5274c0a33c0ec94f309cf01fbd8dd5aaba5746e2aca3",
            "torch.nn.attention.SDPBackend.FLASH_ATTENTION",
            "enum-variant",
            "EnumVariantReference::SdpFlashAttention",
            "d6577b7886293f1f927346fe6b7bdfe34bed87638118ea82f99f897ff8e4f262",
        ),
        (
            "torch_uint64.json",
            leaf::TORCH_UINT64_OPERATION_ID,
            "2df2a7e3f237e8021ad9dcfe524e221758141ebe463e5fadb7100229c30a0beb",
            "torch.uint64",
            "dtype",
            "DType::U64",
            "b21dacb954628a419972f76f85d792870e0ce37edb31a3563f911f0c6500700b",
        ),
        (
            "torch_version.json",
            leaf::TORCH_VERSION_OPERATION_ID,
            "3b2ec96cfd4cbca07b78407aac962cf3a4eb60fac51d66b215b6dfc10237d280",
            "torch.version.__version__",
            "version-value",
            "VersionValueReference::Torch",
            "8546385eed1d3b37a52e83634892078740d116602917166980a1ab66135ac542",
        ),
    ];
    let mut fixture_digests = BTreeSet::new();
    for (
        file_name,
        operation_id,
        baseline_digest,
        target,
        semantic_category,
        semantic,
        expected_digest,
    ) in fixture_cases
    {
        let fixture_path = fixture_directory.join(file_name);
        let bytes = fs::read(fixture_path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(digest, expected_digest);
        assert!(fixture_digests.insert(digest));
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
            "value_or_constant_contract_03"
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
    assert_eq!(fixture_digests.len(), reference_cases().len());
    Ok(())
}
