use std::{env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=abi/symbols-v1.json");
    println!("cargo:rerun-if-changed=abi/reviewed-bindings-v1.txt");
    println!("cargo:rerun-if-changed=abi/verify-execution-bindings.sh");
    println!("cargo:rerun-if-changed=LICENSES");
    let target = env::var("TARGET")?;
    println!("cargo:rustc-env=SIM_NPU_TARGET={target}");

    let manifest = fs::read_to_string("abi/symbols-v1.json")?;
    for required in [
        "CANN-8.0.RC3-AscendCL",
        "aarch64-unknown-linux-gnu",
        "x86_64-unknown-linux-gnu",
        "SIM_ASCEND_ROOT",
        "ASCEND_HOME_PATH",
        "libascendcl.so",
        "libruntime.so",
        "aclrtGetVersion",
        "aclrtGetMemInfo",
        "aclrtCreateEvent",
        "aclrtSetCurrentContext",
        "aclopExecuteV2",
        "0b4481f131bfa8b311ee6e1f7a926eb3fdcfffc0e0165fb64ed4fd8e4036cb81",
        "91d8bd8a346bda371c8175066ac5155fb27ccfe4ba63091730ec29dcd96dd091",
        "sim_backend_npu::loader",
        "zed-sim-npu-package-v1",
    ] {
        if !manifest.contains(required) {
            return Err(format!("reviewed NPU ABI manifest omits {required}").into());
        }
    }
    let reviewed = fs::read_to_string("abi/reviewed-bindings-v1.txt")?;
    for required in [
        "source_sha256=91d8bd8a346bda371c8175066ac5155fb27ccfe4ba63091730ec29dcd96dd091",
        "aclDataType=ACL_DT_UNDEFINED:-1,ACL_FLOAT:0,ACL_FLOAT16:1,ACL_INT8:2,ACL_INT32:3,ACL_UINT8:4,ACL_INT16:6,ACL_UINT16:7,ACL_UINT32:8,ACL_INT64:9,ACL_UINT64:10,ACL_DOUBLE:11,ACL_BOOL:12,ACL_STRING:13,ACL_COMPLEX64:16,ACL_COMPLEX128:17,ACL_BF16:27,ACL_INT4:29,ACL_UINT1:30,ACL_COMPLEX32:33",
        "aclFormat=ACL_FORMAT_UNDEFINED:-1,ACL_FORMAT_NCHW:0,ACL_FORMAT_NHWC:1,ACL_FORMAT_ND:2,ACL_FORMAT_NC1HWC0:3,ACL_FORMAT_FRACTAL_Z:4,ACL_FORMAT_NC1HWC0_C04:12,ACL_FORMAT_HWCN:16,ACL_FORMAT_NDHWC:27,ACL_FORMAT_FRACTAL_NZ:29,ACL_FORMAT_NCDHW:30,ACL_FORMAT_NDC1HWC0:32,ACL_FRACTAL_Z_3D:33,ACL_FORMAT_NC:35,ACL_FORMAT_NCL:47",
        "aclrtMemAttr=ACL_DDR_MEM:0,ACL_HBM_MEM:1,ACL_DDR_MEM_HUGE:2,ACL_DDR_MEM_NORMAL:3,ACL_HBM_MEM_HUGE:4,ACL_HBM_MEM_NORMAL:5,ACL_DDR_MEM_P2P_HUGE:6,ACL_DDR_MEM_P2P_NORMAL:7,ACL_HBM_MEM_P2P_HUGE:8,ACL_HBM_MEM_P2P_NORMAL:9",
        "aclError aclopExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[], int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[], aclopAttr *attr, aclrtStream stream)",
    ] {
        if !reviewed.lines().any(|line| line == required) {
            return Err(format!("reviewed NPU execution bindings omit {required}").into());
        }
    }
    Ok(())
}
