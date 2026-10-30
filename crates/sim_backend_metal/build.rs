use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_METAL_TARGET={target}");
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=abi/reviewed-bindings-v1.txt");
    println!("cargo:rerun-if-changed=abi/execution-v1.json");
    println!("cargo:rerun-if-changed=abi/reviewed-execution-bindings-v1.txt");
    println!("cargo:rerun-if-changed=kernels/readiness.metal");
    println!("cargo:rerun-if-changed=kernels/tensor_ops.metal");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    let reviewed = fs::read_to_string("abi/reviewed-bindings-v1.txt")?;
    for required in [
        "macos-13-metal-3",
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "MTLCreateSystemDefaultDevice",
        "MPSSupportsMTLDevice",
        "zed_sim_metal_readiness_v1",
        "sim_backend_metal::loader",
    ] {
        if !manifest.contains(required) || !reviewed.contains(required) {
            return Err(format!("reviewed Metal ABI evidence omits {required}").into());
        }
    }
    if target.ends_with("apple-darwin") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
    Ok(())
}
