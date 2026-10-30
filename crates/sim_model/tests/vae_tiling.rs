use sim_model::{VaeOperation, VaeTileAxisFormula};
use std::{error::Error, fs, path::Path};

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
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let vae = fs::read_to_string(root.join("src/vae.rs"))?;
    let tiling = fs::read_to_string(root.join("src/vae_tiling.rs"))?;
    let production_tiling = tiling.split("#[cfg(test)]").next().unwrap_or(&tiling);
    let model_root = fs::read_to_string(root.join("src/sim_model.rs"))?;

    assert_eq!(vae.matches("pub struct VaeTilePlan").count(), 1);
    assert!(!production_tiling.contains("struct VaeTilePlan"));
    assert!(model_root.contains("mod vae_tiling;"));
    assert!(!model_root.contains("pub mod vae_tiling;"));
    assert!(production_tiling.contains("backend.reserve_workspace(context"));
    assert!(production_tiling.contains("context.check()?"));
    for forbidden in [
        "CpuBackend",
        "CpuWorkspaceAuthority",
        "authorize_workspace",
        "CancellationToken::default",
        "python",
        "Command::new",
    ] {
        assert!(
            !production_tiling.contains(forbidden),
            "unexpected competing owner: {forbidden}"
        );
    }
    assert!(!production_tiling.to_ascii_lowercase().contains("retry"));
    Ok(())
}
