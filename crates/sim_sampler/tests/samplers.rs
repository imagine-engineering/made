#[path = "support/compute_report.rs"]
mod compute_report;

include!(concat!(env!("OUT_DIR"), "/generated_sampler_tests.rs"));

#[test]
fn generated_sampler_manifest_is_sorted_and_unique() {
    let modules = sim_sampler::GENERATED_MODULES
        .iter()
        .filter(|module| module.starts_with("algorithms/"))
        .copied()
        .collect::<Vec<_>>();
    assert!(modules.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn val_sampler_001_catalog_trajectories() -> Result<(), Box<dyn std::error::Error>> {
    compute_report::write_catalog_report(
        "sampler",
        "algorithms",
        "catalog sampler trajectories through generated native integration fixtures",
        44,
        sim_sampler::GENERATED_SAMPLER_DEFINITIONS
            .iter()
            .map(|definition| compute_report::CatalogDefinition {
                feature_id: definition.feature_id,
                identity: definition.identity,
                implementation_module: definition.implementation_module,
                source_ordinal: definition.source_ordinal,
            }),
    )
}
