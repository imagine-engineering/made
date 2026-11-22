use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    FunctionReference, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, NumericConstantReference, OPERATION_CONTRACTS,
    TypedReferenceContract, VersionValueReference, generated_value_or_constant_contract_02 as leaf,
};
use std::{
    collections::BTreeSet,
    fs, io,
    path::{Component, Path},
};

const OWNER: &str =
    "comfy-parity-tensor-ops-value-or-constant-contract-comfy-tensor-op-4373837eb7ff";

type ContractFacade = fn() -> Option<TypedReferenceContract>;

fn reference_cases() -> [(
    ContractFacade,
    &'static str,
    &'static str,
    CanonicalReference,
); 12] {
    [
        (
            leaf::torch_bool_contract,
            leaf::TORCH_BOOL_OPERATION_ID,
            "torch.bool",
            CanonicalReference::DType(DType::Bool),
        ),
        (
            leaf::torch_finfo_eps_contract,
            leaf::TORCH_FINFO_EPS_OPERATION_ID,
            "torch.finfo().eps",
            CanonicalReference::NumericConstant(NumericConstantReference::FloatInfoEpsilon),
        ),
        (
            leaf::torch_float64_contract,
            leaf::TORCH_FLOAT64_OPERATION_ID,
            "torch.float64",
            CanonicalReference::DType(DType::F64),
        ),
        (
            leaf::torch_float8_e8m0fnu_contract,
            leaf::TORCH_FLOAT8_E8M0FNU_OPERATION_ID,
            "torch.float8_e8m0fnu",
            CanonicalReference::DType(DType::Float8E8m0Fnu),
        ),
        (
            leaf::torch_int32_contract,
            leaf::TORCH_INT32_OPERATION_ID,
            "torch.int32",
            CanonicalReference::DType(DType::I32),
        ),
        (
            leaf::torch_int8_contract,
            leaf::TORCH_INT8_OPERATION_ID,
            "torch.int8",
            CanonicalReference::DType(DType::I8),
        ),
        (
            leaf::torch_nn_softsign_contract,
            leaf::TORCH_NN_SOFTSIGN_OPERATION_ID,
            "torch.nn.Softsign",
            CanonicalReference::Function(FunctionReference::Softsign),
        ),
        (
            leaf::torch_uint8_contract,
            leaf::TORCH_UINT8_OPERATION_ID,
            "torch.uint8",
            CanonicalReference::DType(DType::U8),
        ),
        (
            leaf::torch_version_hip_contract,
            leaf::TORCH_VERSION_HIP_OPERATION_ID,
            "torch.version.hip",
            CanonicalReference::VersionValue(VersionValueReference::Hip),
        ),
        (
            leaf::xpu_has_fp16_contract,
            leaf::XPU_HAS_FP16_OPERATION_ID,
            "torch.xpu.get_device_properties().has_fp16",
            CanonicalReference::BooleanCapability(BooleanCapabilityReference::XpuHasFp16),
        ),
        (
            leaf::xformers_version_contract,
            leaf::XFORMERS_VERSION_OPERATION_ID,
            "xformers.__version__",
            CanonicalReference::VersionValue(VersionValueReference::Xformers),
        ),
        (
            leaf::xformers_has_cpp_library_contract,
            leaf::XFORMERS_HAS_CPP_LIBRARY_OPERATION_ID,
            "xformers._has_cpp_library",
            CanonicalReference::BooleanCapability(
                BooleanCapabilityReference::XformersHasCppLibrary,
            ),
        ),
    ]
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, io::Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other(format!("reference evidence is missing {field}")))
}

fn source_marker(target: &str) -> &str {
    match target {
        "torch.finfo().eps" => "torch.finfo(",
        "torch.xpu.get_device_properties().has_fp16" => "torch.xpu.get_device_properties(",
        target => target,
    }
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
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"value_or_constant_contract_02"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "value_or_constant_contract_02")
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
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let source_root = workspace_root
        .join("projects/comfy/ComfyUI")
        .canonicalize()?;
    let fixture_directory = workspace_root
        .join("crates/sim_test_support/fixtures/tensor_operations/value_or_constant_contract_02");
    let cases = [
        (
            "torch_bool.json",
            leaf::TORCH_BOOL_OPERATION_ID,
            "a423c71a60aeb588ce2eb08878faeb3d54b2c1994957fc771593911c04be21cc",
            "torch.bool",
            "dtype",
            "DType::Bool",
            "5f8ec4a54b5fff9216604cbc82aefea30f8dbaa5143991a81bc2c81b00cce4e3",
        ),
        (
            "torch_finfo_eps.json",
            leaf::TORCH_FINFO_EPS_OPERATION_ID,
            "832d8accb378bb7f27b7581db904deb64b51f51386e31f39d355fcf080389d59",
            "torch.finfo().eps",
            "numeric-constant",
            "NumericConstantReference::FloatInfoEpsilon",
            "d3b665bd2936df6efe075d5429c03730fe432e5d4fafa59f1810e7ff04c7b547",
        ),
        (
            "torch_float64.json",
            leaf::TORCH_FLOAT64_OPERATION_ID,
            "92afcfb2c73e24e7138c0d2218b95cc6a97c8f9a8dab4b2602e4627e8eea3303",
            "torch.float64",
            "dtype",
            "DType::F64",
            "e1059e349ae750a283363f47d7cf3e6f2c4b2e9cfe9a99fb044c48a5a3aff671",
        ),
        (
            "torch_float8_e8m0fnu.json",
            leaf::TORCH_FLOAT8_E8M0FNU_OPERATION_ID,
            "b0119d9a4f432ee68e306a8ce7d047ee29608fb48e85780a1e67fba638181a05",
            "torch.float8_e8m0fnu",
            "dtype",
            "DType::Float8E8m0Fnu",
            "95a53e021f2c778494305a09a1daa97dd7ffcc0d7acde787efc09ef5ac0dced2",
        ),
        (
            "torch_int32.json",
            leaf::TORCH_INT32_OPERATION_ID,
            "b91698b95bc0ec866ed073bd0c65336c9c4bd448101016a36e69ec141ef50e66",
            "torch.int32",
            "dtype",
            "DType::I32",
            "156bd87e5f1b06eb1d25409f785613aeeef246ad3804c36e1818376914485a2d",
        ),
        (
            "torch_int8.json",
            leaf::TORCH_INT8_OPERATION_ID,
            "4fcf80453176d0a77516b4804c10ad86b80389c451677caa72340b050889aee0",
            "torch.int8",
            "dtype",
            "DType::I8",
            "59b593458dcd999194e205cddcd300cef092e146acc4c06b9712e55703ea28e9",
        ),
        (
            "torch_nn_softsign.json",
            leaf::TORCH_NN_SOFTSIGN_OPERATION_ID,
            "a35d832d645afbe13cd74a556b9ebaf861572cf2d1305cbd7a70271cbdc5b8e4",
            "torch.nn.Softsign",
            "function-reference",
            "FunctionReference::Softsign",
            "b85048cdd9cf102a01a6e4eabd4bb80723eb0b27ad56b86f800ec76c0c6ce6e2",
        ),
        (
            "torch_uint8.json",
            leaf::TORCH_UINT8_OPERATION_ID,
            "98ae7549707c222b94f2dedd569bf9840c14dbd35b19aacf74a9ae71d505670d",
            "torch.uint8",
            "dtype",
            "DType::U8",
            "f6d15ad16143b21c0aa996b1b4ca38d29d0188b4398c39754fce162ffa46e222",
        ),
        (
            "torch_version_hip.json",
            leaf::TORCH_VERSION_HIP_OPERATION_ID,
            "f3209024a86e9beba83466b96d4bd7a298b15591243ec60c8e4ba99311c60e81",
            "torch.version.hip",
            "version-value",
            "VersionValueReference::Hip",
            "7206023ce1eb04858cb498e67b61a01d4bc540fc727696bc3670f42034c4db89",
        ),
        (
            "xpu_has_fp16.json",
            leaf::XPU_HAS_FP16_OPERATION_ID,
            "f342c35723486821f33f8817b345cf6bb939970cb509b14987100d075aa6d260",
            "torch.xpu.get_device_properties().has_fp16",
            "boolean-capability",
            "BooleanCapabilityReference::XpuHasFp16",
            "2623fe4b7d886c0a027616a2b98343b86334a90afa3fd1a8b49926c44928643c",
        ),
        (
            "xformers_version.json",
            leaf::XFORMERS_VERSION_OPERATION_ID,
            "cddf4349310695ae4f48e0e3b7d738975b58d2eb154a23b12dd22edbab5504dc",
            "xformers.__version__",
            "version-value",
            "VersionValueReference::Xformers",
            "75e3ed3d6ab32dbcde6aa08939aa6dd9c45da827baa96358804357159a3ecf86",
        ),
        (
            "xformers_has_cpp_library.json",
            leaf::XFORMERS_HAS_CPP_LIBRARY_OPERATION_ID,
            "de7b184aacff48f776c7037d53b2836a83102d8b956c1ac009eb01d309de4d52",
            "xformers._has_cpp_library",
            "boolean-capability",
            "BooleanCapabilityReference::XformersHasCppLibrary",
            "b5d1368a478f4c2fddbface9209f9703399018f12c9e37661503f6652d68f38f",
        ),
    ];
    let expected_fixture_names = cases
        .iter()
        .map(|(file_name, ..)| *file_name)
        .collect::<BTreeSet<_>>();
    let fixture_entries = fs::read_dir(&fixture_directory)?.collect::<Result<Vec<_>, _>>()?;
    let actual_fixture_names = fixture_entries
        .iter()
        .map(|entry| entry.file_name())
        .map(|file_name| {
            file_name
                .into_string()
                .map_err(|_| io::Error::other("fixture filename must be UTF-8"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(
        actual_fixture_names,
        expected_fixture_names
            .iter()
            .map(|name| (*name).to_owned())
            .collect()
    );
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
    let mut fixture_digests = BTreeSet::new();
    for (
        file_name,
        operation_id,
        baseline_digest,
        target,
        semantic_category,
        semantic,
        expected_digest,
    ) in cases
    {
        let fixture_path = fixture_directory.join(file_name);
        let bytes = fs::read(fixture_path)?;
        let fixture_digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(fixture_digest, expected_digest);
        assert!(fixture_digests.insert(fixture_digest));
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
            "value_or_constant_contract_02"
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
            let observation_fields = observation
                .as_object()
                .ok_or("source observation must be an object")?;
            assert_eq!(
                observation_fields
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(["path", "line", "use"])
            );
            let source_path = required_string(observation, "path")?;
            assert!(!source_path.is_empty());
            assert!(!source_path.contains('\\'));
            let relative_source_path = Path::new(source_path);
            assert!(!relative_source_path.is_absolute());
            assert!(
                relative_source_path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
            );
            assert!(relative_source_path.starts_with("projects/comfy/ComfyUI"));
            let mut checked_path = workspace_root.clone();
            for component in relative_source_path.components() {
                let Component::Normal(component) = component else {
                    return Err(io::Error::other("source path component is not normal").into());
                };
                checked_path.push(component);
                assert!(
                    !fs::symlink_metadata(&checked_path)?
                        .file_type()
                        .is_symlink()
                );
            }
            let canonical_source_path = checked_path.canonicalize()?;
            assert!(canonical_source_path.starts_with(&source_root));
            assert!(canonical_source_path.is_file());
            let source_line_number = observation
                .get("line")
                .and_then(Value::as_u64)
                .filter(|line| *line > 0)
                .ok_or("source observation line must be positive")?;
            let source_line_index = usize::try_from(source_line_number - 1)?;
            let source_contents = fs::read_to_string(&canonical_source_path)?;
            let source_line = source_contents
                .lines()
                .nth(source_line_index)
                .ok_or("source observation line is outside the source file")?;
            assert!(
                source_line.contains(source_marker(target)),
                "source observation for {target} does not contain its canonical marker: {source_line}"
            );
            assert!(!required_string(observation, "use")?.is_empty());
        }
    }
    assert_eq!(fixture_digests.len(), 12);
    Ok(())
}
