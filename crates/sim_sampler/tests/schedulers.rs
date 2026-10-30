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
