use std::{fs, path::Path};

#[test]
fn migration_inventory_is_bounded_and_cannot_mint_authority()
-> Result<(), Box<dyn std::error::Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let operation_modules = [
        ("external_tensor_kernel_01", 0),
        ("external_tensor_kernel_02", 0),
        ("external_tensor_kernel_03", 0),
        ("accelerated_attention_kernel_01", 0),
        ("activation_normalization_functional_01", 0),
        ("sim_operator_indirection_01", 0),
        ("sim_operator_indirection_02", 0),
    ];

    for (module, expected_legacy_contexts) in operation_modules {
        let source = fs::read_to_string(crate_root.join(format!("src/ops/{module}.rs")))?;
        assert!(
            source.contains("with_context_exact_native"),
            "{module} has no canonical ExecutionContext entry point"
        );
        assert_eq!(
            source.matches("ScratchReservation::none()").count(),
            expected_legacy_contexts,
            "{module} changed its bounded compatibility-context inventory"
        );
        assert!(
            !source.contains("authorize_workspace("),
            "{module} must not mint workspace authority"
        );
        assert!(
            !source.contains("scratch.bytes() == 0") && !source.contains("scratch.bytes()==0"),
            "{module} must select compatibility explicitly, not from authorization size"
        );
        assert!(
            !source.contains("LegacyCompatibility")
                && !source.contains("tracked_workspace")
                && !source.contains("legacy_context")
                && !source.contains("legacy_execution_context"),
            "{module} retained a superseded compatibility staging path"
        );
    }

    let image_ops = fs::read_to_string(crate_root.join("src/image_ops.rs"))?;
    let image_ops_production = image_ops
        .split("#[cfg(test)]")
        .next()
        .ok_or("image_ops production source is missing")?;
    assert!(image_ops_production.contains("pub fn from_logical_chw("));
    assert!(image_ops_production.contains("context: &ExecutionContext<'_>"));
    assert!(!image_ops_production.contains("ScratchReservation::none()"));
    assert!(!image_ops_production.contains("authorize_workspace("));

    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("repository root is missing")?;
    let model_attention =
        fs::read_to_string(repository_root.join("crates/sim_model/src/attention.rs"))?;
    let model_attention_production = model_attention
        .split("#[cfg(test)]")
        .next()
        .ok_or("model attention production source is missing")?;
    assert!(model_attention_production.contains("scaled_dot_product_attention_with_context"));
    assert!(!model_attention_production.contains("ScratchReservation::none()"));
    assert!(!model_attention_production.contains("authorize_workspace("));

    let part_four =
        fs::read_to_string(crate_root.join("src/ops/elementwise_or_runtime_operation_04.rs"))?;
    assert!(!part_four.contains("tensor_to_f32_exact_native"));
    let part_nine =
        fs::read_to_string(crate_root.join("src/ops/elementwise_or_runtime_operation_09.rs"))?;
    assert!(
        part_nine
            .matches("cast_to_with_context_exact_native(")
            .count()
            >= 2
    );
    let part_thirteen =
        fs::read_to_string(crate_root.join("src/ops/elementwise_or_runtime_operation_13.rs"))?;
    for canonical_softmax in [
        "softmax_with_context_exact_native(",
        "softmax_vjp_with_context_exact_native(",
        "softmax_jvp_with_context_exact_native(",
    ] {
        assert!(part_thirteen.contains(canonical_softmax));
    }
    let part_fifteen =
        fs::read_to_string(crate_root.join("src/ops/elementwise_or_runtime_operation_15.rs"))?;
    assert!(part_fifteen.contains("float_tensor_with_context_exact_native"));
    assert!(part_fifteen.contains("long_with_context_exact_native"));
    let part_twenty =
        fs::read_to_string(crate_root.join("src/ops/elementwise_or_runtime_operation_20.rs"))?;
    assert!(part_twenty.contains("int_method_with_context_exact_native"));
    Ok(())
}
