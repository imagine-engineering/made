use sim_model::{VaeOperation, VaeTileAxisFormula};
use std::{error::Error, path::Path};

#[path = "support/vae_tiling_ownership.rs"]
mod vae_tiling_ownership;

#[test]
fn val_vae_001_one_dimensional_and_causal_three_dimensional_formulas_match_source()
-> Result<(), Box<dyn Error>> {
    let audio = VaeTileAxisFormula::checked_linear(2_048)?;
    assert_eq!(audio.output_extent(VaeOperation::Encode, 4_096)?, 2);
    assert_eq!(audio.output_extent(VaeOperation::Decode, 2)?, 4_096);
    assert_eq!(audio.output_position(VaeOperation::Encode, 3_072)?, 2);

    let mochi_time = VaeTileAxisFormula::checked_causal(6)?;
    assert_eq!(mochi_time.output_extent(VaeOperation::Encode, 1)?, 1);
    assert_eq!(mochi_time.output_extent(VaeOperation::Encode, 7)?, 2);
    assert_eq!(mochi_time.output_extent(VaeOperation::Decode, 2)?, 7);
    assert_eq!(mochi_time.output_position(VaeOperation::Encode, 9)?, 2);
    assert_eq!(mochi_time.output_position(VaeOperation::Decode, 2)?, 12);
    Ok(())
}

#[test]
fn val_vae_001_tiling_extends_the_canonical_vae_and_tensor_owners() -> Result<(), Box<dyn Error>> {
    vae_tiling_ownership::check_tiling_ownership(Path::new(env!("CARGO_MANIFEST_DIR")))
}
