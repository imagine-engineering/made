use serde::Deserialize;
use std::ffi::{c_char, c_int, c_uint, c_void};
use thiserror::Error;

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "cuda-12.2.0-cublaslt-12.2.5.6-cudnn-9.0.0.312";
pub const CUDA_DRIVER_VERSION_MINIMUM: c_int = 12_020;
pub const CUBLASLT_VERSION_MINIMUM: usize = 120_205;
pub const CUDNN_VERSION_MINIMUM: usize = 90_000;
pub const UNSAFE_OWNER: &str = "sim_backend_cuda::loader";
pub const CERTIFICATE_OWNER: &str = "sim_runtime::NativeFfiRegistry";

const TARGETS: [&str; 3] = [
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
];
const DISCOVERY_ORDER: [&str; 4] = [
    "SIM_CUDA_ROOT",
    "CUDA_PATH",
    "signed_package_roots",
    "installed_driver_library",
];
const LIBRARY_IDS: [&str; 4] = ["cublaslt", "cudnn", "driver", "nvrtc"];
const LAYOUTS: [(&str, usize, usize); 15] = [
    ("CUcontext", 8, 8),
    ("CUdevice", 4, 4),
    ("CUdeviceptr", 8, 8),
    ("CUevent", 8, 8),
    ("CUfunction", 8, 8),
    ("CUmodule", 8, 8),
    ("CUresult", 4, 4),
    ("CUstream", 8, 8),
    ("CUuuid", 16, 1),
    ("cublasLtHandle_t", 8, 8),
    ("cublasStatus_t", 4, 4),
    ("cudnnHandle_t", 8, 8),
    ("cudnnStatus_t", 4, 4),
    ("nvrtcProgram", 8, 8),
    ("nvrtcResult", 4, 4),
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbiManifest {
    pub schema_version: u32,
    pub backend: String,
    pub abi_floor: String,
    pub targets: Vec<String>,
    pub discovery_order: Vec<String>,
    pub libraries: Vec<LibraryContract>,
    pub layouts: Vec<LayoutContract>,
    pub versions: VersionContract,
    pub unsafe_owner: String,
    pub certificate_owner: String,
    pub binding_status_owner: String,
    pub package_policy: PackagePolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LibraryContract {
    pub id: String,
    pub linux_filename: String,
    pub windows_filename: String,
    pub headers: Vec<HeaderContract>,
    pub reviewed_source: String,
    pub reviewed_source_sha256: String,
    pub symbols: Vec<SymbolContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HeaderContract {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SymbolContract {
    pub name: String,
    pub signature: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayoutContract {
    pub name: String,
    pub size: usize,
    pub align: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VersionContract {
    pub cuda_driver_minimum: c_int,
    pub nvrtc_major: c_int,
    pub nvrtc_minimum_minor: c_int,
    pub cublaslt_minimum: usize,
    pub cudnn_minimum: usize,
    pub cudnn_abi_major: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PackagePolicy {
    pub redistribute_driver: bool,
    pub approved_redistributables: Vec<String>,
    pub license_approval_required: bool,
    pub signature_algorithm: String,
    pub signature_domain: String,
    pub runtime_compilation_for_core_kernels: bool,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AbiManifestError {
    #[error("CUDA ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("CUDA ABI manifest violates the reviewed contract: {0}")]
    Contract(String),
}

impl AbiManifest {
    pub fn embedded() -> Result<Self, AbiManifestError> {
        let manifest = serde_json::from_str::<Self>(ABI_MANIFEST_JSON)
            .map_err(|error| AbiManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), AbiManifestError> {
        if self.schema_version != 1
            || self.backend != "cuda"
            || self.abi_floor != ABI_FLOOR
            || self.unsafe_owner != UNSAFE_OWNER
            || self.certificate_owner != CERTIFICATE_OWNER
            || self.binding_status_owner != "sim_types::NativeBackendBindingStatus"
        {
            return Err(contract(
                "identity, ABI floor, or authoritative owner differs",
            ));
        }
        require_exact_strings(&self.targets, &TARGETS, "targets")?;
        require_exact_strings(&self.discovery_order, &DISCOVERY_ORDER, "discovery order")?;
        if self.libraries.len() != LIBRARY_IDS.len() {
            return Err(contract("library coverage is incomplete"));
        }
        for (library, expected_id) in self.libraries.iter().zip(LIBRARY_IDS) {
            if library.id != expected_id
                || library.reviewed_source.is_empty()
                || !is_sha256(&library.reviewed_source_sha256)
                || library.symbols.is_empty()
            {
                return Err(contract(format!("library contract {expected_id} differs")));
            }
            let expected_headers = expected_headers(expected_id);
            if library.headers.len() != expected_headers.len()
                || library
                    .headers
                    .iter()
                    .zip(expected_headers)
                    .any(|(actual, expected)| {
                        actual.path != expected.0 || actual.sha256 != expected.1
                    })
            {
                return Err(contract(format!(
                    "reviewed header set for {expected_id} differs"
                )));
            }
            let expected_names = expected_symbols(expected_id);
            if library.symbols.len() != expected_names.len()
                || library
                    .symbols
                    .iter()
                    .zip(expected_names)
                    .any(|(actual, expected)| {
                        actual.name != *expected || actual.signature.is_empty()
                    })
            {
                return Err(contract(format!("symbol set for {expected_id} differs")));
            }
        }
        let driver = &self.libraries[2];
        if driver.linux_filename != "libcuda.so.1" || driver.windows_filename != "nvcuda.dll" {
            return Err(contract("installed driver filenames differ"));
        }
        if self.layouts.len() != LAYOUTS.len()
            || self.layouts.iter().zip(LAYOUTS).any(|(actual, expected)| {
                actual.name != expected.0 || actual.size != expected.1 || actual.align != expected.2
            })
        {
            return Err(contract("64-bit C ABI layouts differ"));
        }
        if self.versions.cuda_driver_minimum != CUDA_DRIVER_VERSION_MINIMUM
            || self.versions.nvrtc_major != 12
            || self.versions.nvrtc_minimum_minor != 2
            || self.versions.cublaslt_minimum != CUBLASLT_VERSION_MINIMUM
            || self.versions.cudnn_minimum != CUDNN_VERSION_MINIMUM
            || self.versions.cudnn_abi_major != 9
        {
            return Err(contract("runtime version gates differ"));
        }
        if self.package_policy.redistribute_driver
            || !self.package_policy.approved_redistributables.is_empty()
            || !self.package_policy.license_approval_required
            || self.package_policy.signature_algorithm != "ed25519"
            || self.package_policy.signature_domain != "zed-sim-cuda-package-v1"
            || self.package_policy.runtime_compilation_for_core_kernels
        {
            return Err(contract(
                "package, license, signature, or compilation policy differs",
            ));
        }
        Ok(())
    }
}

fn expected_symbols(library: &str) -> &'static [&'static str] {
    match library {
        "cublaslt" => &[
            "cublasLtCreate",
            "cublasLtDestroy",
            "cublasLtGetCudartVersion",
            "cublasLtGetVersion",
        ],
        "cudnn" => &[
            "cudnnCreate",
            "cudnnDestroy",
            "cudnnGetCudartVersion",
            "cudnnGetVersion",
            "cudnnSetStream",
        ],
        "driver" => &[
            "cuCtxCreate_v2",
            "cuCtxDestroy_v2",
            "cuCtxSetCurrent",
            "cuDeviceGet",
            "cuDeviceGetCount",
            "cuDriverGetVersion",
            "cuEventCreate",
            "cuEventDestroy_v2",
            "cuEventRecord",
            "cuEventSynchronize",
            "cuInit",
            "cuLaunchKernel",
            "cuMemAlloc_v2",
            "cuMemFree_v2",
            "cuMemGetInfo_v2",
            "cuMemcpyDtoHAsync_v2",
            "cuMemcpyHtoDAsync_v2",
            "cuModuleGetFunction",
            "cuModuleLoadData",
            "cuModuleUnload",
            "cuStreamCreate",
            "cuStreamDestroy_v2",
            "cuStreamSynchronize",
        ],
        "nvrtc" => &[
            "nvrtcCompileProgram",
            "nvrtcCreateProgram",
            "nvrtcDestroyProgram",
            "nvrtcGetProgramLog",
            "nvrtcGetProgramLogSize",
            "nvrtcGetPTX",
            "nvrtcGetPTXSize",
            "nvrtcVersion",
        ],
        _ => &[],
    }
}

fn expected_headers(library: &str) -> &'static [(&'static str, &'static str)] {
    match library {
        "cublaslt" => &[(
            "include/cublasLt.h",
            "e752b21d073b4fdaf19957cd8a63fd3babe46bc26a05d79b8d928258a65a92de",
        )],
        "cudnn" => &[
            (
                "include/cudnn_v9.h",
                "1082d51d3b564bace8ef6fc6ee335b668b2bfa517f57c06efd428263e5c21855",
            ),
            (
                "include/cudnn_graph_v9.h",
                "6cb707f3e93193c9894c3a9037aa0319d3b5a58f28fe1e3a1c491c1150d3b49a",
            ),
            (
                "include/cudnn_version_v9.h",
                "6f784db48abd2094e0145cc18e6be42661f6c83a257adf15c9442d365fdf5ffd",
            ),
        ],
        "driver" => &[(
            "include/cuda.h",
            "a2b4436404c3a9a4231d667c811f24e9ddac256ab3f30c7a486120550abd78d5",
        )],
        "nvrtc" => &[(
            "include/nvrtc.h",
            "16da88110bccc18283eeb7a2834a059b3656744082469a50b300e5db98c43739",
        )],
        _ => &[],
    }
}

fn require_exact_strings(
    actual: &[String],
    expected: &[&str],
    label: &str,
) -> Result<(), AbiManifestError> {
    if actual
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
    {
        Ok(())
    } else {
        Err(contract(format!("{label} differs from the reviewed order")))
    }
}

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn contract(message: impl Into<String>) -> AbiManifestError {
    AbiManifestError::Contract(message.into())
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CuResult(pub c_int);

impl CuResult {
    pub const SUCCESS: Self = Self(0);
}

pub const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
pub const CUDA_ERROR_DEVICE_UNAVAILABLE: i32 = 46;
pub const CUDA_ERROR_INVALID_CONTEXT: i32 = 201;
pub const CUDA_ERROR_CONTEXT_IS_DESTROYED: i32 = 709;
pub const CUDA_ERROR_LAUNCH_FAILED: i32 = 719;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvrtcResult(pub c_int);

impl NvrtcResult {
    pub const SUCCESS: Self = Self(0);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CublasStatus(pub c_int);

impl CublasStatus {
    pub const SUCCESS: Self = Self(0);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudnnStatus(pub c_int);

impl CudnnStatus {
    pub const SUCCESS: Self = Self(0);
}

pub type CuDevice = c_int;
pub type CuDevicePtr = u64;
pub type CuContext = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuEvent = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type NvrtcProgram = *mut c_void;
pub type CublasLtHandle = *mut c_void;
pub type CudnnHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CuUuid {
    pub bytes: [u8; 16],
}

pub type CuInit = unsafe extern "system" fn(c_uint) -> CuResult;
pub type CuDriverGetVersion = unsafe extern "system" fn(*mut c_int) -> CuResult;
pub type CuDeviceGetCount = unsafe extern "system" fn(*mut c_int) -> CuResult;
pub type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, c_int) -> CuResult;
pub type CuCtxCreate = unsafe extern "system" fn(*mut CuContext, c_uint, CuDevice) -> CuResult;
pub type CuCtxDestroy = unsafe extern "system" fn(CuContext) -> CuResult;
pub type CuCtxSetCurrent = unsafe extern "system" fn(CuContext) -> CuResult;
pub type CuMemGetInfo = unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult;
pub type CuMemAlloc = unsafe extern "system" fn(*mut CuDevicePtr, usize) -> CuResult;
pub type CuMemFree = unsafe extern "system" fn(CuDevicePtr) -> CuResult;
pub type CuMemcpyHtoDAsync =
    unsafe extern "system" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult;
pub type CuMemcpyDtoHAsync =
    unsafe extern "system" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult;
pub type CuStreamCreate = unsafe extern "system" fn(*mut CuStream, c_uint) -> CuResult;
pub type CuStreamDestroy = unsafe extern "system" fn(CuStream) -> CuResult;
pub type CuStreamSynchronize = unsafe extern "system" fn(CuStream) -> CuResult;
pub type CuEventCreate = unsafe extern "system" fn(*mut CuEvent, c_uint) -> CuResult;
pub type CuEventDestroy = unsafe extern "system" fn(CuEvent) -> CuResult;
pub type CuEventRecord = unsafe extern "system" fn(CuEvent, CuStream) -> CuResult;
pub type CuEventSynchronize = unsafe extern "system" fn(CuEvent) -> CuResult;
pub type CuModuleLoadData = unsafe extern "system" fn(*mut CuModule, *const c_void) -> CuResult;
pub type CuModuleUnload = unsafe extern "system" fn(CuModule) -> CuResult;
pub type CuModuleGetFunction =
    unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
pub type CuLaunchKernel = unsafe extern "system" fn(
    CuFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;

pub type NvrtcVersion = unsafe extern "system" fn(*mut c_int, *mut c_int) -> NvrtcResult;
pub type NvrtcCreateProgram = unsafe extern "system" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
pub type NvrtcCompileProgram =
    unsafe extern "system" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
pub type NvrtcDestroyProgram = unsafe extern "system" fn(*mut NvrtcProgram) -> NvrtcResult;
pub type NvrtcGetProgramLogSize =
    unsafe extern "system" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
pub type NvrtcGetProgramLog = unsafe extern "system" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
pub type NvrtcGetPtxSize = unsafe extern "system" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
pub type NvrtcGetPtx = unsafe extern "system" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;

pub type CublasLtCreate = unsafe extern "system" fn(*mut CublasLtHandle) -> CublasStatus;
pub type CublasLtDestroy = unsafe extern "system" fn(CublasLtHandle) -> CublasStatus;
pub type CublasLtGetVersion = unsafe extern "system" fn() -> usize;
pub type CublasLtGetCudartVersion = unsafe extern "system" fn() -> usize;

pub type CudnnCreate = unsafe extern "system" fn(*mut CudnnHandle) -> CudnnStatus;
pub type CudnnDestroy = unsafe extern "system" fn(CudnnHandle) -> CudnnStatus;
pub type CudnnGetVersion = unsafe extern "system" fn() -> usize;
pub type CudnnGetCudartVersion = unsafe extern "system" fn() -> usize;
pub type CudnnSetStream = unsafe extern "system" fn(CudnnHandle, CuStream) -> CudnnStatus;

const _: () = {
    assert!(std::mem::size_of::<CuDevice>() == 4);
    assert!(std::mem::align_of::<CuDevice>() == 4);
    assert!(std::mem::size_of::<CuDevicePtr>() == 8);
    assert!(std::mem::align_of::<CuDevicePtr>() == 8);
    assert!(std::mem::size_of::<CuUuid>() == 16);
    assert!(std::mem::align_of::<CuUuid>() == 1);
};
