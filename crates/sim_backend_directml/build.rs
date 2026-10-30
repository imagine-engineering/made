use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_DIRECTML_TARGET={target}");
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=LICENSES");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    for required in [
        "\"abi_floor\": \"1.13.1\"",
        "\"target_version\": \"0x6200\"",
        "\"minimum_windows_build\": 19041",
        "aarch64-pc-windows-msvc",
        "x86_64-pc-windows-msvc",
        "D3D12CreateDevice",
        "CreateDXGIFactory2",
        "DMLCreateDevice",
        "DMLCreateDevice1",
        "sim_backend_directml::loader",
        "a38cef0d59f314fbcc0cd6551c5a762db7cfdaf8a977f85df32a0b1e279d3ba7",
    ] {
        if !manifest.contains(required) {
            return Err(format!("reviewed DirectML ABI manifest omits {required}").into());
        }
    }
    if manifest.contains("rustc-link-lib=DirectML") {
        return Err("DirectML must be loaded through the checked runtime loader".into());
    }
    Ok(())
}
