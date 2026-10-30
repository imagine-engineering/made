use std::{fs, path::Path};

#[test]
fn migration_inventory_is_bounded_and_cannot_mint_authority()
-> Result<(), Box<dyn std::error::Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut modules = (12..=23)
        .map(|part| format!("elementwise_or_runtime_operation_{part:02}"))
        .collect::<Vec<_>>();
    modules.push("indexing_masking_01".to_owned());
    modules.push("indexing_masking_02".to_owned());

    for module in modules {
        let source = fs::read_to_string(crate_root.join(format!("src/ops/{module}.rs")))?;
        assert!(
            source.contains("with_context_exact_native"),
            "{module} has no canonical ExecutionContext entry point"
        );
        assert_eq!(
            source.matches("ScratchReservation::none()").count(),
            0,
            "{module} must not synthesize an unbound execution context"
        );
        assert!(
            !source.contains("authorize_workspace("),
            "{module} must not mint workspace authority"
        );
        assert!(!source.contains("Legacy(Vec"));
        assert!(!source.contains("LegacyCompatibility"));
        assert!(!source.contains("legacy_context("));
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
        assert!(
            canonical_names.is_disjoint(&legacy_names),
            "{module} retains paired cancellation-only compatibility entrypoints"
        );
    }
    Ok(())
}
