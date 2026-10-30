use serde_json::Value;
use sha2::{Digest as _, Sha256};
use sim_model::{
    NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT, NATIVE_UPSCALE_ARCHITECTURE_COUNT,
    NATIVE_UPSCALE_CONTRACT_ID, NATIVE_UPSCALE_CONTRACT_SCHEMA_VERSION,
    NATIVE_UPSCALE_CONTRACT_SHA256, NativeUpscaleContractError, NativeUpscaleDetection,
    NativeUpscaleStateDictionaryLayout, compiled_native_upscale_contract,
    validate_native_upscale_contract_candidate,
};
use sim_types::CancellationToken;
use std::collections::BTreeSet;

const CONTRACT: &str = include_str!(
    "../../../crates/sim_test_support/parity/catalogs/spandrel-image-model-contract.json"
);

fn mutate_contract(
    change: impl FnOnce(&mut Value) -> Result<(), String>,
) -> Result<String, String> {
    let mut value: Value = serde_json::from_str(CONTRACT).map_err(|error| error.to_string())?;
    change(&mut value)?;
    serde_json::to_string_pretty(&value).map_err(|error| error.to_string())
}

fn architectures_mut(value: &mut Value) -> Result<&mut Vec<Value>, String> {
    value
        .get_mut("architectures")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "architectures are missing".to_owned())
}

fn object_field_mut<'a>(value: &'a mut Value, pointer: &str) -> Result<&'a mut Value, String> {
    value
        .pointer_mut(pointer)
        .ok_or_else(|| format!("missing field {pointer}"))
}

fn detected_architecture(
    contract: &sim_model::NativeUpscaleRuntimeContract,
    state_keys: BTreeSet<String>,
) -> Result<&str, String> {
    let cancellation = CancellationToken::default();
    let NativeUpscaleDetection::Unavailable { architecture } = contract
        .detect_state_keys(&state_keys, &cancellation)
        .map_err(|error| error.to_string())?;
    Ok(&architecture.architecture_id)
}

#[test]
fn embedded_upscale_contract_is_exact_ordered_and_zero_admission() -> Result<(), String> {
    assert_eq!(NATIVE_UPSCALE_CONTRACT_SCHEMA_VERSION, 1);
    assert_eq!(
        NATIVE_UPSCALE_CONTRACT_ID,
        "zed-sim-spandrel-image-model-contract-v1"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(CONTRACT.as_bytes())),
        NATIVE_UPSCALE_CONTRACT_SHA256
    );
    validate_native_upscale_contract_candidate(CONTRACT).map_err(|error| error.to_string())?;

    let contract = compiled_native_upscale_contract().map_err(|error| error.to_string())?;
    assert_eq!(
        contract.architectures().len(),
        NATIVE_UPSCALE_ARCHITECTURE_COUNT
    );
    assert_eq!(
        contract.admitted_architecture_count(),
        NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT
    );
    assert_eq!(NATIVE_UPSCALE_ARCHITECTURE_COUNT, 52);
    assert_eq!(NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT, 0);
    assert_eq!(
        contract
            .architectures()
            .first()
            .map(|architecture| architecture.architecture_id.as_str()),
        Some("Compact")
    );
    assert_eq!(
        contract
            .architectures()
            .get(41)
            .map(|architecture| architecture.architecture_id.as_str()),
        Some("AuraSR")
    );
    assert_eq!(
        contract
            .architectures()
            .get(42)
            .map(|architecture| architecture.architecture_id.as_str()),
        Some("SRFormer")
    );
    assert_eq!(
        contract
            .architectures()
            .last()
            .map(|architecture| architecture.architecture_id.as_str()),
        Some("MIRNet2")
    );

    for (ordinal, architecture) in contract.architectures().iter().enumerate() {
        assert_eq!(architecture.ordinal, ordinal);
        assert_eq!(architecture.support_disposition, "rejected");
        assert!(architecture.license_artifacts.is_empty());
        assert_eq!(
            architecture.model_use_disposition,
            "no-model-weights-approved; evaluate model rights independently"
        );
        assert_eq!(
            contract
                .architecture(&architecture.architecture_id)
                .map(|candidate| candidate.ordinal),
            Some(ordinal)
        );
    }
    Ok(())
}

#[test]
fn every_compiled_detector_preserves_source_order_and_fails_unavailable() -> Result<(), String> {
    let contract = compiled_native_upscale_contract().map_err(|error| error.to_string())?;
    let cancellation = CancellationToken::default();
    for architecture in contract.architectures() {
        let state_keys = architecture
            .detection_state_keys
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let NativeUpscaleDetection::Unavailable {
            architecture: detected,
        } = contract
            .detect_state_keys(&state_keys, &cancellation)
            .map_err(|error| {
                format!(
                    "detector {} did not match its source keys: {error}",
                    architecture.architecture_id
                )
            })?;
        assert_eq!(
            detected.architecture_id, architecture.architecture_id,
            "source-order detector selected the wrong architecture for {}",
            architecture.architecture_id
        );
        assert_eq!(detected.support_disposition, "rejected");
    }

    let unknown = BTreeSet::from(["not.a.spandrel.state.key".to_owned()]);
    assert_eq!(
        contract.detect_state_keys(&unknown, &cancellation),
        Err(NativeUpscaleContractError::NoArchitectureMatch)
    );
    cancellation.cancel();
    assert_eq!(
        contract.detect_state_keys(&unknown, &cancellation),
        Err(NativeUpscaleContractError::Cancelled)
    );
    Ok(())
}

#[test]
fn canonical_state_keys_preserve_wrappers_prefixes_and_source_priority() -> Result<(), String> {
    let contract = compiled_native_upscale_contract().map_err(|error| error.to_string())?;
    let compact = contract
        .architecture("Compact")
        .ok_or_else(|| "Compact contract is missing".to_owned())?;
    let prefixed = compact
        .detection_state_keys
        .iter()
        .map(|key| format!("module.netG.{key}"))
        .collect::<BTreeSet<_>>();
    for layout in [
        NativeUpscaleStateDictionaryLayout::Flat,
        NativeUpscaleStateDictionaryLayout::ModelStateDict,
        NativeUpscaleStateDictionaryLayout::StateDict,
        NativeUpscaleStateDictionaryLayout::ParamsEma,
        NativeUpscaleStateDictionaryLayout::ParamsDashEma,
        NativeUpscaleStateDictionaryLayout::Params,
        NativeUpscaleStateDictionaryLayout::Model,
        NativeUpscaleStateDictionaryLayout::Net,
        NativeUpscaleStateDictionaryLayout::SingleMapping,
    ] {
        let cancellation = CancellationToken::default();
        let NativeUpscaleDetection::Unavailable { architecture } = contract
            .detect_wrapped_state_keys(layout, &prefixed, &cancellation)
            .map_err(|error| error.to_string())?;
        assert_eq!(architecture.architecture_id, "Compact");
    }

    let swin2sr_and_swinir = ["Swin2SR", "SwinIR"]
        .into_iter()
        .flat_map(|architecture_id| {
            contract
                .architecture(architecture_id)
                .into_iter()
                .flat_map(|architecture| architecture.detection_state_keys.iter().cloned())
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        detected_architecture(&contract, swin2sr_and_swinir)?,
        "Swin2SR"
    );

    let mmrealsr_and_esrgan = ["MMRealSR", "ESRGAN"]
        .into_iter()
        .flat_map(|architecture_id| {
            contract
                .architecture(architecture_id)
                .into_iter()
                .flat_map(|architecture| architecture.detection_state_keys.iter().cloned())
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        detected_architecture(&contract, mmrealsr_and_esrgan)?,
        "MMRealSR"
    );
    Ok(())
}

#[test]
fn candidate_validation_rejects_drift_reordering_ambiguity_and_admission() -> Result<(), String> {
    let unsupported_schema = mutate_contract(|value| {
        *object_field_mut(value, "/schema_version")? = Value::from(2);
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&unsupported_schema),
        Err(NativeUpscaleContractError::UnsupportedSchema(2))
    );

    let missing = mutate_contract(|value| {
        architectures_mut(value)?.pop();
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&missing),
        Err(NativeUpscaleContractError::InvalidArchitectureCount(51))
    );

    let reordered = mutate_contract(|value| {
        architectures_mut(value)?.swap(0, 1);
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&reordered),
        Err(NativeUpscaleContractError::InvalidOrdinal {
            expected: 0,
            actual: 1,
        })
    );

    let duplicate = mutate_contract(|value| {
        let rows = architectures_mut(value)?;
        let identifier = rows
            .first()
            .and_then(|row| row.get("architecture_id"))
            .cloned()
            .ok_or_else(|| "first architecture has no identifier".to_owned())?;
        let second = rows
            .get_mut(1)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "second architecture is missing".to_owned())?;
        second.insert("architecture_id".to_owned(), identifier);
        Ok(())
    })?;
    assert!(matches!(
        validate_native_upscale_contract_candidate(&duplicate),
        Err(NativeUpscaleContractError::DuplicateArchitecture(_))
    ));

    let ambiguous = mutate_contract(|value| {
        let rows = architectures_mut(value)?;
        let equation_family = rows
            .first()
            .and_then(|row| row.get("equation_family_id"))
            .cloned()
            .ok_or_else(|| "first architecture has no equation family".to_owned())?;
        let second = rows
            .get_mut(1)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "second architecture is missing".to_owned())?;
        second.insert("equation_family_id".to_owned(), equation_family);
        Ok(())
    })?;
    assert!(matches!(
        validate_native_upscale_contract_candidate(&ambiguous),
        Err(NativeUpscaleContractError::AmbiguousDetection(_))
    ));

    let malformed_detector = mutate_contract(|value| {
        *object_field_mut(value, "/architectures/0/detection_predicate")? =
            Value::from("KeyCondition.unsupported('body.0.weight')");
        Ok(())
    })?;
    assert!(matches!(
        validate_native_upscale_contract_candidate(&malformed_detector),
        Err(NativeUpscaleContractError::InvalidDetection(_))
    ));

    let forged_projection = mutate_contract(|value| {
        object_field_mut(value, "/architectures/0/detection_state_keys")?
            .as_array_mut()
            .ok_or_else(|| "detection keys are missing".to_owned())?
            .push(Value::from("forged.weight"));
        Ok(())
    })?;
    assert!(matches!(
        validate_native_upscale_contract_candidate(&forged_projection),
        Err(NativeUpscaleContractError::InvalidDetectionKeyProjection(_))
    ));

    let admitted = mutate_contract(|value| {
        *object_field_mut(value, "/architectures/0/support_disposition")? = Value::from("admitted");
        Ok(())
    })?;
    assert!(matches!(
        validate_native_upscale_contract_candidate(&admitted),
        Err(NativeUpscaleContractError::UnlicensedArchitecture(_))
    ));

    let unsupported_descriptor = mutate_contract(|value| {
        *object_field_mut(value, "/architectures/0/descriptor_disposition")? =
            Value::from("resize-to-fit");
        Ok(())
    })?;
    assert!(matches!(
        validate_native_upscale_contract_candidate(&unsupported_descriptor),
        Err(NativeUpscaleContractError::InvalidArchitecture(_))
    ));

    let source_drift = mutate_contract(|value| {
        *object_field_mut(value, "/architectures/0/source_sha256")? =
            Value::from("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&source_drift),
        Err(NativeUpscaleContractError::CatalogDigestMismatch)
    );
    Ok(())
}

#[test]
fn candidate_validation_rejects_snapshot_summary_outcome_and_task_drift() -> Result<(), String> {
    let snapshot_drift = mutate_contract(|value| {
        *object_field_mut(value, "/source_snapshots/spandrel/included_file_count")? =
            Value::from(179);
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&snapshot_drift),
        Err(NativeUpscaleContractError::InvalidSourceSnapshot(
            "spandrel".to_owned()
        ))
    );

    let summary_drift = mutate_contract(|value| {
        *object_field_mut(value, "/summary/admitted_count")? = Value::from(1);
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&summary_drift),
        Err(NativeUpscaleContractError::InvalidSummary)
    );

    let outcome_drift = mutate_contract(|value| {
        *object_field_mut(value, "/optional_extra_outcomes/1/registry")? =
            Value::from("EXTRA before MAIN");
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&outcome_drift),
        Err(NativeUpscaleContractError::InvalidOptionalExtraOutcomes)
    );

    let task_drift = mutate_contract(|value| {
        object_field_mut(value, "/task_projection/implementation_leaves")?
            .as_array_mut()
            .ok_or_else(|| "implementation leaves are missing".to_owned())?
            .push(Value::from("forged-equation-owner"));
        Ok(())
    })?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&task_drift),
        Err(NativeUpscaleContractError::InvalidTaskProjection)
    );

    let formatting_only = serde_json::to_string(
        &serde_json::from_str::<Value>(CONTRACT).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    assert_eq!(
        validate_native_upscale_contract_candidate(&formatting_only),
        Err(NativeUpscaleContractError::CatalogDigestMismatch)
    );
    Ok(())
}

#[test]
fn production_upscale_contract_has_no_oracle_execution_or_filesystem_boundary() -> Result<(), String>
{
    let source = include_str!("../src/upscale_contract.rs");
    let manifest = include_str!("../Cargo.toml");
    let source_tree = ["../../../pro", "jects/sim"].concat();
    for forbidden in [
        "std::process",
        "std::fs",
        "Command::new",
        "pyo3",
        "cpython",
        "resize-to-fit",
        "fallback architecture",
    ] {
        assert!(
            !source.contains(forbidden),
            "production upscale contract contains forbidden boundary {forbidden}"
        );
    }
    for macro_name in ["include_bytes!", "include_str!"] {
        let forbidden = format!("{macro_name}(\"{source_tree}");
        assert!(
            !source.contains(&forbidden),
            "production upscale contract contains forbidden boundary {forbidden}"
        );
    }
    for forbidden_dependency in ["pyo3", "cpython", "spandrel"] {
        assert!(
            !manifest.lines().any(|line| {
                line.split_once('=')
                    .is_some_and(|(name, _)| name.trim() == forbidden_dependency)
            }),
            "sim_model declares forbidden dependency {forbidden_dependency}"
        );
    }
    assert!(source.contains("spandrel-image-model-contract.json"));
    Ok(())
}
