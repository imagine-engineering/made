use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_XPU_TARGET={target}");
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=abi/reviewed-execution-bindings-v1.txt");
    println!("cargo:rerun-if-changed=abi/verify-execution-bindings.c");
    println!("cargo:rerun-if-changed=LICENSES");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    for required in [
        "level-zero-loader-1.11.0-api-1.6.3+onednn-3.5.0",
        "x86_64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "libze_loader.so.1",
        "ze_loader.dll",
        "libdnnl.so.3",
        "dnnl.dll",
        "zeDriverGetApiVersion",
        "dnnl_version",
        "zeDeviceGetProperties",
        "zeDeviceGetMemoryProperties",
        "dnnl_binary_primitive_desc_create",
        "dnnl_primitive_execute",
        "SIM_XPU_ROOT",
        "ONEAPI_ROOT",
        "sim_backend_xpu::loader",
        "sim_runtime::NativeFfiRegistry",
    ] {
        if !manifest.contains(required) {
            return Err(format!("reviewed XPU ABI manifest omits {required}").into());
        }
    }
    Ok(())
}
