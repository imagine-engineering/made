#[path = "support/compute_report.rs"]
mod compute_report;

include!(concat!(env!("OUT_DIR"), "/generated_scheduler_tests.rs"));

#[test]
fn generated_scheduler_manifest_is_sorted_and_unique() {
    let modules = sim_sampler::GENERATED_MODULES
        .iter()
        .filter(|module| module.starts_with("schedulers/"))
        .copied()
        .collect::<Vec<_>>();
    assert!(modules.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn val_scheduler_001_catalog_arrays() -> Result<(), Box<dyn std::error::Error>> {
    compute_report::write_catalog_report(
        "scheduler",
        "schedulers",
        "catalog scheduler sigma arrays through generated native integration fixtures",
        9,
        sim_sampler::GENERATED_SCHEDULER_DEFINITIONS
            .iter()
            .map(|definition| compute_report::CatalogDefinition {
                feature_id: definition.feature_id,
                identity: definition.identity,
                implementation_module: definition.implementation_module,
                source_ordinal: definition.source_ordinal,
            }),
    )
}
