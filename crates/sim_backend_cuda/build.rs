use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_CUDA_TARGET={target}");
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=LICENSES");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    for required in [
        "cuda-12.2.0-cublaslt-12.2.5.6-cudnn-9.0.0.312",
        "aarch64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
        "libcuda.so.1",
        "nvcuda.dll",
        "libnvrtc.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
        "cuInit",
        "cuDriverGetVersion",
        "nvrtcVersion",
        "cublasLtGetVersion",
        "cudnnGetVersion",
        "SIM_CUDA_ROOT",
        "CUDA_PATH",
        "sim_backend_cuda::loader",
        "sim_runtime::NativeFfiRegistry",
        "sim_types::NativeBackendBindingStatus",
        "e752b21d073b4fdaf19957cd8a63fd3babe46bc26a05d79b8d928258a65a92de",
        "1082d51d3b564bace8ef6fc6ee335b668b2bfa517f57c06efd428263e5c21855",
        "6cb707f3e93193c9894c3a9037aa0319d3b5a58f28fe1e3a1c491c1150d3b49a",
        "6f784db48abd2094e0145cc18e6be42661f6c83a257adf15c9442d365fdf5ffd",
        "a2b4436404c3a9a4231d667c811f24e9ddac256ab3f30c7a486120550abd78d5",
        "16da88110bccc18283eeb7a2834a059b3656744082469a50b300e5db98c43739",
    ] {
        if !manifest.contains(required) {
            return Err(format!("reviewed CUDA ABI manifest omits {required}").into());
        }
    }
    Ok(())
}
