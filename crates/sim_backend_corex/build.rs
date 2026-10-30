use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_COREX_TARGET={target}");
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=LICENSES");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    for required in [
        "CoreX-IXRT-0.8-ABI-profile",
        "blocked-missing-vendor-headers",
        "x86_64-unknown-linux-gnu",
        "SIM_COREX_ROOT",
        "IXRT_HOME",
        "signed_package_roots",
        "libixrt.so",
        "libixblas.so",
        "0528f3ae5da5dd2255f21966b82bedcb2de65582",
        "sim_backend_corex::loader",
        "sim_runtime::NativeFfiRegistry",
    ] {
        if !manifest.contains(required) {
            return Err(format!("CoreX ABI provenance manifest omits {required}").into());
        }
    }
    Ok(())
}
