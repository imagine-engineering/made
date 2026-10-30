use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_MLU_TARGET={target}");
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=abi/reviewed-bindings-v1.txt");
    println!("cargo:rerun-if-changed=abi/verify-bindings.c");
    println!("cargo:rerun-if-changed=LICENSES");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    for required in [
        "neuware-1.20-cnnl-1.20.4-cnrt-6.6.0",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "libcnrt.so",
        "libcnnl.so",
        "cnrtGetLibVersion",
        "cnnlGetLibVersion",
        "cnrtErrorNoDevice",
        "cnrtErrorNoMem",
        "CNNL_STATUS_ALLOC_FAILED",
        "SIM_MLU_ROOT",
        "NEUWARE_HOME",
        "sim_backend_mlu::loader",
        "sim_runtime::NativeFfiRegistry",
    ] {
        if !manifest.contains(required) {
            return Err(format!("reviewed MLU ABI manifest omits {required}").into());
        }
    }
    let reviewed_bindings = fs::read_to_string("abi/reviewed-bindings-v1.txt")?;
    for required in [
        "CNRT package SHA-256: 70e23ae6197f68f9c0440db985f7a20dc174837b4190ba6c70d70117e3d8fb51",
        "CNRT header SHA-256: 0c8ca727c2db85c60f69cf8cd96157ee1d547b175ddeaf45c0e94108f794a8f2",
        "CNNL package SHA-256: 8874376ecdb81fb555d523197c922baace34d5a6abe567b68e1e74254f2b11e8",
        "CNNL header SHA-256: 69eaa4fa560fa02e1e636a0d19917244a2badd94a960721e57fa36059f6fa031",
        "cnrtRet_t cnrtSuccess = 0",
        "cnrtRet_t cnrtErrorNoDevice = 100004",
        "cnrtRet_t cnrtErrorNoMem = 100100",
        "cnnlStatus_t CNNL_STATUS_SUCCESS = 0",
        "cnnlStatus_t CNNL_STATUS_ALLOC_FAILED = 2",
    ] {
        if !reviewed_bindings.lines().any(|line| line == required) {
            return Err(format!("reviewed MLU execution bindings omit {required}").into());
        }
    }
    Ok(())
}
