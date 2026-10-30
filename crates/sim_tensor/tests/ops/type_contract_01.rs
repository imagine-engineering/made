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
            "sim.ops.CastWeightBiasOp",
            SIM_CAST_WEIGHT_BIAS_OP_REFERENCE,
        ),
        (
            SIM_DISABLE_WEIGHT_INIT_OPERATION_ID,
            "sim.ops.disable_weight_init",
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
            "sim_cast_weight_bias_op.json",
            SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID,
            "b0fa628e84eb7e3284f8b03e318ceeb8532cba6262530fd9ce9862ddf8fcf42d",
            "sim.ops.CastWeightBiasOp",
            "TypeMarkerReference::ComfyCastWeightBiasOp",
            "9623bc32f0612498148a49185e2bbb5e4c8078f95245e0a98e7573076b04a1ee",
        ),
        (
            "sim_disable_weight_init.json",
            SIM_DISABLE_WEIGHT_INIT_OPERATION_ID,
            "1fd119e7af47dfb53441e02bf95f67ebb95017868a6c89f7ca416b6d4b12ff41",
            "sim.ops.disable_weight_init",
            "TypeMarkerReference::ComfyDisableWeightInit",
            "6135c4c01d10515d92d0bc83a9e164f7746b9e7b51d096308e32585ab1d46d59",
        ),
        (
            "torch_long_tensor.json",
            TORCH_LONG_TENSOR_OPERATION_ID,
            "cd4b5f3155f14d454a32f1cf90fcf7ed2dd5d737b2480d600cefb775783fd464",
            "torch.LongTensor",
            "TypeMarkerReference::LongTensor",
            "e156e78017a2959f1d2353a00b85c61c52afaaa089a4e9d2e08f0018c6a59a2d",
        ),
        (
            "torch_autograd_function.json",
            TORCH_AUTOGRAD_FUNCTION_OPERATION_ID,
            "01458c436f1e5cf5e4f870c0411639ce376cd93aed3819c122e9530e54ea6e8d",
            "torch.autograd.Function",
            "TypeMarkerReference::AutogradFunction",
            "19e99a452dd3a1b10248f03241356dc9f3dd0fb294b962c067f27856679f1cd8",
        ),
        (
            "torch_dtype.json",
            TORCH_DTYPE_OPERATION_ID,
            "3e124d5be195b452d47cae4d2db163a7e42a195ad07e7462103e2552d00877ab",
            "torch.dtype",
            "TypeMarkerReference::DType",
            "a454fcd648da82c077e6750df531f543414d209da6373c3de14ebc86a64ff5fa",
        ),
        (
            "torch_empty_device.json",
            TORCH_EMPTY_DEVICE_OPERATION_ID,
            "f95d135fd55de28cb0bf4831a92c24503d20a72e299acea3edf3a78c32c5831c",
            "torch.empty().device",
            "TypeMarkerReference::EmptyTensorDevice",
            "794c4ffc3187eaba7da3cf51de21ab520b5d7a3765c38a3569c0d69d3910f315",
        ),
        (
            "torch_jit_final.json",
            TORCH_JIT_FINAL_OPERATION_ID,
            "724606e6857bf1f5f2293b4ca822ef86a299a56d5302c53912660fcf08dba34f",
            "torch.jit.Final",
            "TypeMarkerReference::JitFinal",
            "f4a477a008ecd27a57f504a2a5c0e3f1a37a4a2a7cf74410116331d4fe64fa05",
        ),
        (
            "torch_conv_transpose_1d.json",
            TORCH_CONV_TRANSPOSE_1D_OPERATION_ID,
            "7b244f1b6c1637970a7f7e72729c968b544d16487211067937b7da4c7dc13079",
            "torch.nn.ConvTranspose1d",
            "TypeMarkerReference::ConvTranspose1d",
            "f2d7a59fabd3acd1796163a2e4678c7cc8a454d11e041f82ac6844174164751f",
        ),
        (
            "torch_conv_transpose_2d.json",
            TORCH_CONV_TRANSPOSE_2D_OPERATION_ID,
            "491dbe0ffcad2de52863adff9370ff4e0917d7dbf2c184b1441aeaa8dc6b1c9b",
            "torch.nn.ConvTranspose2d",
            "TypeMarkerReference::ConvTranspose2d",
            "af1be22c5e43578259c2c8f9e6a28458a0d16c4261180c4c57943658e14e36f0",
        ),
        (
            "torch_optimizer.json",
            TORCH_OPTIMIZER_OPERATION_ID,
            "ea7000c2112a4dc57681d06c0c03626f4bbac41215f57ec2e50bf19fbcc8a25c",
            "torch.optim.Optimizer",
            "TypeMarkerReference::Optimizer",
            "35aa3b28db91cc900e3caeba9ca28923b6e4ba90fa1ae3823721e0c7ca6e1913",
        ),
        (
            "torch_learning_rate_scheduler.json",
            TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID,
            "2d8b3789d89b551aaf29b4efc784b64e669933d4f2f9beadc563bd455a15e0bf",
            "torch.optim.lr_scheduler._LRScheduler",
            "TypeMarkerReference::LearningRateScheduler",
            "564568eedd0c639d8da35506c46b7e82f1187499b1b7df779c2b9fddbc590343",
        ),
        (
            "torch_dataset.json",
            TORCH_DATASET_OPERATION_ID,
            "8c9fe87cb5ababa27781ff15d5da4c6256e0dbcc010749bc134657d6abfd29ad",
            "torch.utils.data.Dataset",
            "TypeMarkerReference::Dataset",
            "ba57a86642a7d1955595c051471ddcb6dfdc9f745eb033e382d4b2d950b0412b",
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
