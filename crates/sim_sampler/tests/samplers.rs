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
