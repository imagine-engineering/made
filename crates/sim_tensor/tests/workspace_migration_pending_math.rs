use std::{fs, path::Path};

#[test]
fn pending_math_inventory_has_one_caller_context_path_per_operation()
-> Result<(), Box<dyn std::error::Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let modules = [
        ("linear_algebra_01", 0),
        ("linear_algebra_02", 0),
        ("neural_network_functional_01", 0),
    ];
    for (module, expected_legacy_contexts) in modules {
        let source = fs::read_to_string(crate_root.join(format!("src/ops/{module}.rs")))?;
        assert!(
            source.contains("CpuWorkspaceVec"),
            "{module} does not use the canonical workspace container"
        );
        assert!(
            !source.contains("authorize_workspace("),
            "{module} must not mint workspace authority"
        );
        assert_eq!(
            source.matches("ScratchReservation::none()").count(),
            expected_legacy_contexts,
            "{module} changed its bounded compatibility-context inventory"
        );
        let canonical_names = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix("pub fn "))
            .filter_map(|line| line.split_once('(').map(|(name, _)| name))
            .filter_map(|name| name.strip_suffix("_with_context_exact_native"))
            .collect::<std::collections::BTreeSet<_>>();
        let legacy_names = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix("pub fn "))
            .filter_map(|line| line.split_once('(').map(|(name, _)| name))
            .filter_map(|name| name.strip_suffix("_exact_native"))
            .filter(|name| !name.ends_with("_with_context"))
            .collect::<std::collections::BTreeSet<_>>();
        let missing = legacy_names
            .difference(&canonical_names)
            .copied()
            .collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "{module} lacks canonical paths for {missing:?}"
        );
    }

    let tensor_source = fs::read_to_string(crate_root.join("src/sim_tensor.rs"))?;
    assert!(tensor_source.contains("pub fn linear_element_bytes("));
    let linear_source = fs::read_to_string(crate_root.join("src/ops/linear_algebra_01.rs"))?;
    let reader = linear_source
        .split("pub(crate) fn tensor_f64_with_context")
        .nth(1)
        .and_then(|tail| tail.split("fn upload_f64").next())
        .ok_or("canonical tensor reader is unavailable")?;
    assert!(reader.contains("linear_element_bytes"));
    assert!(!reader.contains("collect::<Result<Vec"));
    Ok(())
}
