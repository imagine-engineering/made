#[cfg(feature = "xpu")]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/backends/intel_xpu_sim_model_0021.rs"
));
