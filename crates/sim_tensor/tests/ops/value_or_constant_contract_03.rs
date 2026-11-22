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
            "a28f99143dbc63e7464123f419ebc2dfe11bb7cd0168dff546f2806b41afd634",
            "torch.AcceleratorError",
            "type-marker",
            "TypeMarkerReference::AcceleratorError",
            "40dd28df23ef40099707e8e6cf930f0a943b4541a5fb45b566f854ff7ecda893",
        ),
        (
            "autograd_once_differentiable.json",
            leaf::AUTOGRAD_ONCE_DIFFERENTIABLE_OPERATION_ID,
            "0d8979c97c15de6427b0a4d35aeacdbb3b8dae41f18534e5486986bc9f5465e0",
            "torch.autograd.function.once_differentiable",
            "function-reference",
            "FunctionReference::AutogradOnceDifferentiable",
            "544dbb66c4fd747928eafcabd8d6bae7e0c8c5fa405f2995f98bc7242711c029",
        ),
        (
            "cuda_gcn_architecture_name.json",
            leaf::CUDA_GCN_ARCHITECTURE_NAME_OPERATION_ID,
            "cc4d303b62f179b72b84b6f71e447afcc00d076b689fcadc3b4fb262a9b99d7c",
            "torch.cuda.get_device_properties().gcnArchName",
            "device-property",
            "DevicePropertyReference::CudaGcnArchitectureName",
            "78641f656253878bc9f92c073ead8d618c060f5ab43a2ac74e28fbb72be773bb",
        ),
        (
            "float_info_minimum.json",
            leaf::FLOAT_INFO_MINIMUM_OPERATION_ID,
            "9359614174becefd77152e981902ffca8f5eb9ca52d0139e883f1ba67edd016d",
            "torch.finfo().min",
            "numeric-constant",
            "NumericConstantReference::FloatInfoMinimum",
            "b2f83103d1bebbd198ade57c47558027778da1092fa0eb1bb631167ff81773ad",
        ),
        (
            "torch_float16.json",
            leaf::TORCH_FLOAT16_OPERATION_ID,
            "81ec605a9304fdd621b54a2d517c16b590c32b855d0dfd317a01f4615051b14a",
            "torch.float16",
            "dtype",
            "DType::F16",
            "0b073ece91e7b7d99fc54055e32a8d9f5412998415e0e72fa01ff4b882c12e26",
        ),
        (
            "torch_float8_e4m3fnuz.json",
            leaf::TORCH_FLOAT8_E4M3FNUZ_OPERATION_ID,
            "866455e7266687cedf91e854e06a0fa4ebba74bcf83f042a1ec631d1d76ddc5b",
            "torch.float8_e4m3fnuz",
            "dtype",
            "DType::Float8E4m3Fnuz",
            "6bc008208025975c28a283183a5e62c9e31bcfa5b0284ac00fc6a332eb37c40c",
        ),
        (
            "torch_float8_e5m2.json",
            leaf::TORCH_FLOAT8_E5M2_OPERATION_ID,
            "7804c704b4d4ac40c8846c065d48ee6ebe550ba3a6d46ff49921813edaca6711",
            "torch.float8_e5m2",
            "dtype",
            "DType::Float8E5m2",
            "aba408ea84fdd3c2e872d4cef7c63e81d5b3a6ad0b21a31dd4d528ddb818b84a",
        ),
        (
            "torch_infinity.json",
            leaf::TORCH_INFINITY_OPERATION_ID,
            "cbe5b8cb3ff67d6e30dee97dc84cb756269f7eef72999bccf6b48f13227afd91",
            "torch.inf",
            "numeric-constant",
            "NumericConstantReference::Infinity",
            "da97201dc076ceaf99227d5e34ec85eb1afa2972b80725a233e70a87da8c2241",
        ),
        (
            "median_values.json",
            leaf::MEDIAN_VALUES_OPERATION_ID,
            "96e2d3b7a2be977efa92092d27d00cb045429e952936c82a3ba4a22361e35234",
            "torch.median().values",
            "tensor-property",
            "TensorPropertyReference::MedianValues",
            "bcd36718b1fd8ef9e30f758fa3830c168888a93455a62a2a4ad68c339a14faa7",
        ),
        (
            "sdp_flash_attention.json",
            leaf::SDP_FLASH_ATTENTION_OPERATION_ID,
            "47fd44f807aff094cd5b7273199a42dc31f9bf91a52065d31b4a26b764902e78",
            "torch.nn.attention.SDPBackend.FLASH_ATTENTION",
            "enum-variant",
            "EnumVariantReference::SdpFlashAttention",
            "137ca744c4a706101593b05d7c057ca6f151e3e3db4ae17145f105fed64485c1",
        ),
        (
            "torch_uint64.json",
            leaf::TORCH_UINT64_OPERATION_ID,
            "f2e52d7fc78d2d5b0e6ea5620f165ce637ebb01898da79551ed7757bb72cf464",
            "torch.uint64",
            "dtype",
            "DType::U64",
            "11658942b95f0fd01c6459789ed3d9e6e7ed335b738bc08544fc54bdc5d1b4f3",
        ),
        (
            "torch_version.json",
            leaf::TORCH_VERSION_OPERATION_ID,
            "2e551fa060cc8b04c722b85c06d7022573231647dc644c953c49b7a6f8ab9da2",
            "torch.version.__version__",
            "version-value",
            "VersionValueReference::Torch",
            "6851758989d4da1437ec2c30a566fc5a29c76a93895fba7cb7c176644e462d86",
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
