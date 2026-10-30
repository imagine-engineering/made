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
            "6680c6999a879e2cbcd79a6fc2af1ab27b45b0b4bacc64c363d6a0ec55dd2865",
            "torch.bool",
            "dtype",
            "DType::Bool",
            "aef98ed15c30d32f439db0e088f8d8624c02fcde2096177ec46bc32965c979e3",
        ),
        (
            "torch_finfo_eps.json",
            leaf::TORCH_FINFO_EPS_OPERATION_ID,
            "a3e3bb9c9d8ed69d68c58e03c509923b5c2054ef4ddc57f24521b91d8cc128cb",
            "torch.finfo().eps",
            "numeric-constant",
            "NumericConstantReference::FloatInfoEpsilon",
            "80ba03742565bd98dd579f8adda05bb043e7fad63e5b41010c9805dab9670bac",
        ),
        (
            "torch_float64.json",
            leaf::TORCH_FLOAT64_OPERATION_ID,
            "9ffe17aeb508dd3d3151216881e45b14e04174062fa3a9586161d169997f5ce6",
            "torch.float64",
            "dtype",
            "DType::F64",
            "f5580c94ea90ffe0e67b7c39303472c753d5554ccbd8d6fcdbe11ea027254bd6",
        ),
        (
            "torch_float8_e8m0fnu.json",
            leaf::TORCH_FLOAT8_E8M0FNU_OPERATION_ID,
            "8df627e290bc8324609da8ebafa37593097451d9ede40f0c2168fb920c247aba",
            "torch.float8_e8m0fnu",
            "dtype",
            "DType::Float8E8m0Fnu",
            "dcb5a1d641051eecace4e348f0a259146010c85aacdfab2c2430d40192ec962a",
        ),
        (
            "torch_int32.json",
            leaf::TORCH_INT32_OPERATION_ID,
            "4ba7bb84eb0501598228922b49deecde0a2d171ed81c2faaf9a1a21c891670ce",
            "torch.int32",
            "dtype",
            "DType::I32",
            "21ee996192e5a89b0e11ecccc038931eb827e9ca2f55608d8948e8782c1b7d73",
        ),
        (
            "torch_int8.json",
            leaf::TORCH_INT8_OPERATION_ID,
            "af138d78273bbc43e5409ad38110f7a12e0a948bf6a17e596649bdbb3133912c",
            "torch.int8",
            "dtype",
            "DType::I8",
            "e8ff6106e203f7e395b3fb4c8398bf666bc557cb1525b3df21da7ab7f978a8e2",
        ),
        (
            "torch_nn_softsign.json",
            leaf::TORCH_NN_SOFTSIGN_OPERATION_ID,
            "5ba4c9ee0bef80587862174cc7ebf563c183fe6334891616de16b90c9d396c61",
            "torch.nn.Softsign",
            "function-reference",
            "FunctionReference::Softsign",
            "46f498176b358f0b1a8c98d7bf569d0cea40272c9b43bb2ade74b9134e80be2e",
        ),
        (
            "torch_uint8.json",
            leaf::TORCH_UINT8_OPERATION_ID,
            "65497cdf050559ef4768604f795b939d4125e995d823c94af43471dff5aa8a1c",
            "torch.uint8",
            "dtype",
            "DType::U8",
            "12a5594cd53c29e9e39d7d3690b179721997dbacb2daf7f3a5bfc4a1d4bcd970",
        ),
        (
            "torch_version_hip.json",
            leaf::TORCH_VERSION_HIP_OPERATION_ID,
            "0c3d98e059ab1013026b700a29d85ed2edaf2bc0217e1d1022744968e6a0259f",
            "torch.version.hip",
            "version-value",
            "VersionValueReference::Hip",
            "e74b80c3f3440b41f77d397aedf44e4a3112ed68cb8b901f51d844a5c1643364",
        ),
        (
            "xpu_has_fp16.json",
            leaf::XPU_HAS_FP16_OPERATION_ID,
            "a02fd517445031f6b53ee5e0266eca986e8edc3c8d6b387cf79fc1d93e6c7fd4",
            "torch.xpu.get_device_properties().has_fp16",
            "boolean-capability",
            "BooleanCapabilityReference::XpuHasFp16",
            "2b39ad5b3e4f92c24f7679e6b3329222303f0575b000f93c8a035fc58444ec82",
        ),
        (
            "xformers_version.json",
            leaf::XFORMERS_VERSION_OPERATION_ID,
            "72ff4db8eb86a19e7e30b62756a9336a1aa03f95aa289eb13478287d395f69a2",
            "xformers.__version__",
            "version-value",
            "VersionValueReference::Xformers",
            "db554af8df21edab3f3cdf46f1cfb2d97d363d352b101cca221bc360f081077a",
        ),
        (
            "xformers_has_cpp_library.json",
            leaf::XFORMERS_HAS_CPP_LIBRARY_OPERATION_ID,
            "9554f48a40c3c8fe7c3a1d9100c77b037bb9a714055df0be2e7c1c9dff1623f8",
            "xformers._has_cpp_library",
            "boolean-capability",
            "BooleanCapabilityReference::XformersHasCppLibrary",
            "c475b97c0994f516e65fe54070d1988476ced8a419a3d095005c883e5dc7773c",
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
