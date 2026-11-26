use std::{error::Error, fs, path::Path};

pub(super) fn check_tiling_ownership(root: &Path) -> Result<(), Box<dyn Error>> {
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
