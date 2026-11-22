use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CanonicalReference, ContractInventoryKind, GENERATED_OPERATION_RESOLUTION_MODULES,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OPERATION_CONTRACTS, TypeMarkerReference,
    generated_type_contract_01::{
        ASSIGNED_TYPE_REFERENCES, SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID,
        SIM_CAST_WEIGHT_BIAS_OP_REFERENCE, SIM_DISABLE_WEIGHT_INIT_OPERATION_ID,
        SIM_DISABLE_WEIGHT_INIT_REFERENCE, TORCH_AUTOGRAD_FUNCTION_OPERATION_ID,
        TORCH_AUTOGRAD_FUNCTION_REFERENCE, TORCH_CONV_TRANSPOSE_1D_OPERATION_ID,
        TORCH_CONV_TRANSPOSE_1D_REFERENCE, TORCH_CONV_TRANSPOSE_2D_OPERATION_ID,
        TORCH_CONV_TRANSPOSE_2D_REFERENCE, TORCH_DATASET_OPERATION_ID, TORCH_DATASET_REFERENCE,
        TORCH_DTYPE_OPERATION_ID, TORCH_DTYPE_REFERENCE, TORCH_EMPTY_DEVICE_OPERATION_ID,
        TORCH_EMPTY_DEVICE_REFERENCE, TORCH_JIT_FINAL_OPERATION_ID, TORCH_JIT_FINAL_REFERENCE,
        TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID, TORCH_LEARNING_RATE_SCHEDULER_REFERENCE,
        TORCH_LONG_TENSOR_OPERATION_ID, TORCH_LONG_TENSOR_REFERENCE, TORCH_OPTIMIZER_OPERATION_ID,
        TORCH_OPTIMIZER_REFERENCE, assigned_type_contract, sim_cast_weight_bias_op_contract,
        sim_disable_weight_init_contract, torch_autograd_function_contract,
        torch_conv_transpose_1d_contract, torch_conv_transpose_2d_contract, torch_dataset_contract,
        torch_dtype_contract, torch_empty_device_contract, torch_jit_final_contract,
        torch_learning_rate_scheduler_contract, torch_long_tensor_contract,
        torch_optimizer_contract,
    },
};
use std::{collections::HashSet, fs, io, path::Path};

const OWNER_TASK_ID: &str = "comfy-parity-tensor-ops-type-contract-comfy-tensor-op-0aa720652f2f";

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, io::Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| io::Error::other(format!("reference evidence is missing {field}")))
}

#[test]
fn type_facades_use_the_canonical_reference_owner() -> Result<(), Box<dyn std::error::Error>> {
    let facade_contracts = [
        sim_cast_weight_bias_op_contract(),
        sim_disable_weight_init_contract(),
        torch_long_tensor_contract(),
        torch_autograd_function_contract(),
        torch_dtype_contract(),
        torch_empty_device_contract(),
        torch_jit_final_contract(),
        torch_conv_transpose_1d_contract(),
        torch_conv_transpose_2d_contract(),
        torch_optimizer_contract(),
        torch_learning_rate_scheduler_contract(),
        torch_dataset_contract(),
    ];
    let expected = [
        (
            SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID,
            "comfy.ops.CastWeightBiasOp",
            SIM_CAST_WEIGHT_BIAS_OP_REFERENCE,
        ),
        (
            SIM_DISABLE_WEIGHT_INIT_OPERATION_ID,
            "comfy.ops.disable_weight_init",
            SIM_DISABLE_WEIGHT_INIT_REFERENCE,
        ),
        (
            TORCH_LONG_TENSOR_OPERATION_ID,
            "torch.LongTensor",
            TORCH_LONG_TENSOR_REFERENCE,
        ),
        (
            TORCH_AUTOGRAD_FUNCTION_OPERATION_ID,
            "torch.autograd.Function",
            TORCH_AUTOGRAD_FUNCTION_REFERENCE,
        ),
        (
            TORCH_DTYPE_OPERATION_ID,
            "torch.dtype",
            TORCH_DTYPE_REFERENCE,
        ),
        (
            TORCH_EMPTY_DEVICE_OPERATION_ID,
            "torch.empty().device",
            TORCH_EMPTY_DEVICE_REFERENCE,
        ),
        (
            TORCH_JIT_FINAL_OPERATION_ID,
            "torch.jit.Final",
            TORCH_JIT_FINAL_REFERENCE,
        ),
        (
            TORCH_CONV_TRANSPOSE_1D_OPERATION_ID,
            "torch.nn.ConvTranspose1d",
            TORCH_CONV_TRANSPOSE_1D_REFERENCE,
        ),
        (
            TORCH_CONV_TRANSPOSE_2D_OPERATION_ID,
            "torch.nn.ConvTranspose2d",
            TORCH_CONV_TRANSPOSE_2D_REFERENCE,
        ),
        (
            TORCH_OPTIMIZER_OPERATION_ID,
            "torch.optim.Optimizer",
            TORCH_OPTIMIZER_REFERENCE,
        ),
        (
            TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID,
            "torch.optim.lr_scheduler._LRScheduler",
            TORCH_LEARNING_RATE_SCHEDULER_REFERENCE,
        ),
        (
            TORCH_DATASET_OPERATION_ID,
            "torch.utils.data.Dataset",
            TORCH_DATASET_REFERENCE,
        ),
    ];
    assert_eq!(
        ASSIGNED_TYPE_REFERENCES,
        expected.map(|(id, _, marker)| (id, marker))
    );
    let mut unique_operation_ids = HashSet::new();
    for ((operation_id, target, marker), facade_contract) in
        expected.into_iter().zip(facade_contracts)
    {
        assert!(unique_operation_ids.insert(operation_id));
        let contract = assigned_type_contract(operation_id)
            .ok_or_else(|| format!("typed contract is missing for {operation_id}"))?;
        assert_eq!(facade_contract, Some(contract));
        assert_eq!(contract.operation_id(), operation_id);
        assert_eq!(contract.canonical_target(), target);
        assert_eq!(
            contract.inventory_kind(),
            ContractInventoryKind::TypeReference
        );
        assert_eq!(contract.semantic(), CanonicalReference::TypeMarker(marker));
    }
    assert_eq!(unique_operation_ids.len(), 12);
    assert!(assigned_type_contract("COMFY-TENSOR-OP-UNASSIGNED").is_none());
    Ok(())
}

#[test]
fn reference_only_leaf_never_enters_the_kernel_resolution_registry() {
    assert!(!GENERATED_OPERATION_RESOLUTION_MODULES.contains(&"type_contract_01"));
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .all(|slice| slice.module_name != "type_contract_01")
    );
    assert!(
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| slice.iter())
            .all(|contract| ASSIGNED_TYPE_REFERENCES
                .iter()
                .all(|(operation_id, _)| *operation_id != contract.operation_id))
    );
}

#[test]
fn no_unassigned_catalog_row_is_claimed() {
    let assigned_operation_ids = ASSIGNED_TYPE_REFERENCES
        .iter()
        .map(|(operation_id, _)| *operation_id)
        .collect::<HashSet<_>>();
    let mut claimed_operation_ids = HashSet::new();
    for contract in OPERATION_CONTRACTS {
        let Some(reference) = contract.typed_reference() else {
            continue;
        };
        if assigned_operation_ids.contains(reference.operation_id()) {
            assert!(assigned_type_contract(reference.operation_id()).is_some());
            assert!(claimed_operation_ids.insert(reference.operation_id()));
        } else {
            assert!(assigned_type_contract(reference.operation_id()).is_none());
        }
    }
    assert_eq!(claimed_operation_ids, assigned_operation_ids);
}

#[test]
fn typed_reference_evidence_is_exact_and_hash_sealed() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_directory =
        workspace_root.join("crates/sim_test_support/fixtures/tensor_operations/type_contract_01");
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
    .collect::<HashSet<_>>();
    for (file_name, operation_id, baseline_digest, target, rust_semantic, expected_digest) in [
        (
            "comfy_cast_weight_bias_op.json",
            SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID,
            "1ed1ff7223aa5417664957e8dd26b9744ab316b6238c1e177316c0f1e7bfbf5e",
            "comfy.ops.CastWeightBiasOp",
            "TypeMarkerReference::ComfyCastWeightBiasOp",
            "fcd03f3dfb6de6d311191849eaa31df6de2b54e722923fabfbe229c83e43baff",
        ),
        (
            "comfy_disable_weight_init.json",
            SIM_DISABLE_WEIGHT_INIT_OPERATION_ID,
            "c1c354002c45171b6d8dc487315f7245f8df60dc82051f19015b7aafda4e3b83",
            "comfy.ops.disable_weight_init",
            "TypeMarkerReference::ComfyDisableWeightInit",
            "16c279111fcd4b6c8c634f07f4120b49f15c21a3c48973d57ce4e372f632df6d",
        ),
        (
            "torch_long_tensor.json",
            TORCH_LONG_TENSOR_OPERATION_ID,
            "f0bd15f2322a0b385ceb9ad0cbb3996c162d58f5939dcf73633bdc011de351c4",
            "torch.LongTensor",
            "TypeMarkerReference::LongTensor",
            "848677ba2e706600beb8086eef3bb1956964055ae0ec9b15c17c54e1c7df1b80",
        ),
        (
            "torch_autograd_function.json",
            TORCH_AUTOGRAD_FUNCTION_OPERATION_ID,
            "5d7ec4359eb23edab404280ebc69ff8d9c7a07546cb1ea096c1dce005bd7bee7",
            "torch.autograd.Function",
            "TypeMarkerReference::AutogradFunction",
            "32aa1b117093dfed45587323a966eb809bd4b0aae310cfcb8f0bfa8d75eeaab4",
        ),
        (
            "torch_dtype.json",
            TORCH_DTYPE_OPERATION_ID,
            "6ba06e70389cfd9597edf1af60b31abba4802774eac18a61b0c7bd9a733f92a3",
            "torch.dtype",
            "TypeMarkerReference::DType",
            "6121349f03ddf54d6dfc9ebdf4901e57e66923001e846d1f50c0db86f74cdd1f",
        ),
        (
            "torch_empty_device.json",
            TORCH_EMPTY_DEVICE_OPERATION_ID,
            "0eb245042d4aa9c7983c10ad90fcba13cad4b2f3a53c81b7dba1e0d0782ad03c",
            "torch.empty().device",
            "TypeMarkerReference::EmptyTensorDevice",
            "7abae8ae88aadf3d06547820f136b6044a301f84781af10661058864aec99516",
        ),
        (
            "torch_jit_final.json",
            TORCH_JIT_FINAL_OPERATION_ID,
            "8132c604ebae4cf92a239d1609037d89c3f6bf80891f21aeb7e10244dc41ddbc",
            "torch.jit.Final",
            "TypeMarkerReference::JitFinal",
            "9ccf33ea8d5f68e784855ef2c33d1fc0b46cd4aa40e9ddf4087ef5de128c4a80",
        ),
        (
            "torch_conv_transpose_1d.json",
            TORCH_CONV_TRANSPOSE_1D_OPERATION_ID,
            "df5eea93ea5a8ff7da4a6a32250cdb862dd7f2e4ab9b86554ff379d59b2419bc",
            "torch.nn.ConvTranspose1d",
            "TypeMarkerReference::ConvTranspose1d",
            "1deb801a6608a93013eec26e648fc32c9f3eda77de6c6a076e858a84dca70729",
        ),
        (
            "torch_conv_transpose_2d.json",
            TORCH_CONV_TRANSPOSE_2D_OPERATION_ID,
            "1da77a297aad69bd3cbe9c8d75452ef615f0a81e853a05eb162c0b4421f4c649",
            "torch.nn.ConvTranspose2d",
            "TypeMarkerReference::ConvTranspose2d",
            "6e24a09e1a7f45f3fa0142178da5c7607af0a9aa4c41c3096c6a47adb5c0bc61",
        ),
        (
            "torch_optimizer.json",
            TORCH_OPTIMIZER_OPERATION_ID,
            "3d53faa00a3e20dd669fd5b1a321beee07d9c60d89227d4367b795889e701fc5",
            "torch.optim.Optimizer",
            "TypeMarkerReference::Optimizer",
            "c10532b5a4e4da1a3f887dbcaa3dd139a8a0f46b0d146c2bd729f47e127b9cba",
        ),
        (
            "torch_learning_rate_scheduler.json",
            TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID,
            "bccf3fb66257fc735f7a3bc495df1aa04f8af6daee7f5c916f0d869cc6b89669",
            "torch.optim.lr_scheduler._LRScheduler",
            "TypeMarkerReference::LearningRateScheduler",
            "f7ae712029e7461aea75d379d1fb2b552b5312247874794bc5f2e20d70a961b0",
        ),
        (
            "torch_dataset.json",
            TORCH_DATASET_OPERATION_ID,
            "3f130ac4f3e888e856e742c9d8fbee5e727074fb77cfc88db3a290f569f48c56",
            "torch.utils.data.Dataset",
            "TypeMarkerReference::Dataset",
            "d2863296b10ca26c77ee51a6e01257eb4a4d50b3531076a147b6262c7771c442",
        ),
    ] {
        let fixture_path = fixture_directory.join(file_name);
        let bytes = fs::read(fixture_path)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(digest, expected_digest);
        let fixture: Value = serde_json::from_slice(&bytes)?;
        let fixture_fields = fixture
            .as_object()
            .ok_or("reference evidence must be an object")?
            .keys()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        assert_eq!(fixture_fields, expected_fixture_fields);
        assert_eq!(
            fixture.get("schema_version").and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            required_string(&fixture, "resolution_module")?,
            "type_contract_01"
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
        assert_eq!(required_string(&fixture, "owner_task_id")?, OWNER_TASK_ID);
        assert_eq!(
            required_string(&fixture, "inventory_kind")?,
            "type_reference"
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
                .collect::<HashSet<_>>(),
            HashSet::from(["category", "value"])
        );
        assert_eq!(
            required_string(reference_semantic, "category")?,
            "type-marker"
        );
        assert_eq!(required_string(reference_semantic, "value")?, target);
        assert_eq!(
            required_string(&fixture, "canonical_rust_semantic")?,
            rust_semantic
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
                fields.keys().map(String::as_str).collect::<HashSet<_>>(),
                HashSet::from(["path", "line", "use"])
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

#[test]
fn every_assigned_reference_is_a_type_marker() {
    assert_eq!(ASSIGNED_TYPE_REFERENCES.len(), 12);
    assert!(ASSIGNED_TYPE_REFERENCES.iter().all(|(_, marker)| matches!(
        marker,
        TypeMarkerReference::ComfyCastWeightBiasOp
            | TypeMarkerReference::ComfyDisableWeightInit
            | TypeMarkerReference::LongTensor
            | TypeMarkerReference::AutogradFunction
            | TypeMarkerReference::DType
            | TypeMarkerReference::EmptyTensorDevice
            | TypeMarkerReference::JitFinal
            | TypeMarkerReference::ConvTranspose1d
            | TypeMarkerReference::ConvTranspose2d
            | TypeMarkerReference::Optimizer
            | TypeMarkerReference::LearningRateScheduler
            | TypeMarkerReference::Dataset
    )));
}
