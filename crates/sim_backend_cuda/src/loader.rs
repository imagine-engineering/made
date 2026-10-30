#![cfg_attr(
    not(any(
        all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(target_os = "windows", target_arch = "x86_64")
    )),
    allow(dead_code)
)]

use crate::abi::{
    ABI_FLOOR, AbiManifest, CERTIFICATE_OWNER, CUBLASLT_VERSION_MINIMUM,
    CUDA_DRIVER_VERSION_MINIMUM, CUDNN_VERSION_MINIMUM, CuCtxCreate, CuCtxDestroy, CuCtxSetCurrent,
    CuDevice, CuDeviceGet, CuDeviceGetCount, CuDevicePtr, CuDriverGetVersion, CuEventCreate,
    CuEventDestroy, CuEventRecord, CuEventSynchronize, CuInit, CuLaunchKernel, CuMemAlloc,
    CuMemFree, CuMemGetInfo, CuMemcpyDtoHAsync, CuMemcpyHtoDAsync, CuModuleGetFunction,
    CuModuleLoadData, CuModuleUnload, CuResult, CuStreamCreate, CuStreamDestroy,
    CuStreamSynchronize, CublasLtCreate, CublasLtDestroy, CublasLtGetCudartVersion,
    CublasLtGetVersion, CublasStatus, CudnnCreate, CudnnDestroy, CudnnGetCudartVersion,
    CudnnGetVersion, CudnnSetStream, CudnnStatus, LibraryContract, NvrtcCompileProgram,
    NvrtcCreateProgram, NvrtcDestroyProgram, NvrtcGetProgramLog, NvrtcGetProgramLogSize,
    NvrtcGetPtx, NvrtcGetPtxSize, NvrtcVersion,
};
use std::{
    any::Any,
    collections::BTreeMap,
    env,
    ffi::{CString, OsString, c_void},
    marker::PhantomData,
    path::{Component, Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};
use thiserror::Error;

const MAXIMUM_LIBRARY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CORE_PTX: &[u8] = include_bytes!("../kernels/core-v1.ptx");
pub const CORE_PTX_SHA256: &str =
    "b3f61b727de366a7cda5874ff0cad2f7e6dee1ad547bfabd5511e1bed0fafe33";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscoverySource {
    SimCudaRoot,
    CudaPath,
    SignedPackage,
    InstalledDriver,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiscoveryEnvironment {
    pub sim_cuda_root: Option<OsString>,
    pub cuda_path: Option<OsString>,
}

impl DiscoveryEnvironment {
    pub fn from_process() -> Self {
        Self {
            sim_cuda_root: env::var_os("SIM_CUDA_ROOT"),
            cuda_path: env::var_os("CUDA_PATH"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct SignedPackageRoot<'certificate> {
    path: PathBuf,
    _certificate: PhantomData<&'certificate ()>,
}

impl<'certificate> SignedPackageRoot<'certificate> {
    /// Projects a package root already admitted by the runtime trust layer.
    ///
    /// # Safety
    ///
    /// The caller must retain the signer-bound package certificate and immutable package tree for
    /// this value's lifetime. This constructor never verifies a signature or authorizes a library.
    pub unsafe fn from_runtime_verified_path<Certificate: ?Sized>(
        _certificate: &'certificate Certificate,
        path: PathBuf,
    ) -> Result<Self, CudaLoadError> {
        validate_root(&path, DiscoverySource::SignedPackage)?;
        Ok(Self {
            path,
            _certificate: PhantomData,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaLibraryCandidates {
    pub source: DiscoverySource,
    pub root: Option<PathBuf>,
    pub libraries: BTreeMap<String, PathBuf>,
}

pub fn discovery_candidates(
    target: &str,
    environment: &DiscoveryEnvironment,
    signed_package_roots: &[SignedPackageRoot<'_>],
) -> Result<Vec<CudaLibraryCandidates>, CudaLoadError> {
    ensure_target(target)?;
    let manifest =
        AbiManifest::embedded().map_err(|error| CudaLoadError::Manifest(error.to_string()))?;
    let mut roots = Vec::new();
    push_environment_root(
        &mut roots,
        environment.sim_cuda_root.as_ref(),
        "SIM_CUDA_ROOT",
        DiscoverySource::SimCudaRoot,
    )?;
    push_environment_root(
        &mut roots,
        environment.cuda_path.as_ref(),
        "CUDA_PATH",
        DiscoverySource::CudaPath,
    )?;
    for root in signed_package_roots {
        push_unique_root(
            &mut roots,
            DiscoverySource::SignedPackage,
            root.path.clone(),
        );
    }

    let mut plans = roots
        .into_iter()
        .map(|(source, root)| CudaLibraryCandidates {
            libraries: manifest
                .libraries
                .iter()
                .map(|library| {
                    let filename = filename_for_target(library, target);
                    (library.id.clone(), library_path(&root, target, filename))
                })
                .collect(),
            source,
            root: Some(root),
        })
        .collect::<Vec<_>>();
    let driver = manifest
        .libraries
        .iter()
        .find(|library| library.id == "driver")
        .ok_or_else(|| CudaLoadError::Manifest("driver contract is missing".to_owned()))?;
    plans.push(CudaLibraryCandidates {
        source: DiscoverySource::InstalledDriver,
        root: None,
        libraries: BTreeMap::from([(
            "driver".to_owned(),
            PathBuf::from(filename_for_target(driver, target)),
        )]),
    });
    Ok(plans)
}

fn library_path(root: &Path, target: &str, filename: &str) -> PathBuf {
    if target == "x86_64-pc-windows-msvc" {
        root.join("bin").join(filename)
    } else {
        root.join("lib64").join(filename)
    }
}

fn filename_for_target<'a>(library: &'a LibraryContract, target: &str) -> &'a str {
    if target == "x86_64-pc-windows-msvc" {
        &library.windows_filename
    } else {
        &library.linux_filename
    }
}

fn push_environment_root(
    roots: &mut Vec<(DiscoverySource, PathBuf)>,
    value: Option<&OsString>,
    variable: &'static str,
    source: DiscoverySource,
) -> Result<(), CudaLoadError> {
    let Some(value) = value else {
        return Ok(());
    };
    let value = value
        .to_str()
        .ok_or(CudaLoadError::InvalidEnvironment { variable })?;
    if value.is_empty() {
        return Ok(());
    }
    let root = PathBuf::from(value);
    validate_root(&root, source)?;
    push_unique_root(roots, source, root);
    Ok(())
}

fn push_unique_root(
    roots: &mut Vec<(DiscoverySource, PathBuf)>,
    source: DiscoverySource,
    root: PathBuf,
) {
    if !roots.iter().any(|(_, existing)| existing == &root) {
        roots.push((source, root));
    }
}

fn validate_root(path: &Path, source: DiscoverySource) -> Result<(), CudaLoadError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(CudaLoadError::InvalidRoot {
            discovery_source: source,
            path: path.to_owned(),
        });
    }
    Ok(())
}

pub fn validate_discovered_library(path: &Path, library: &str) -> Result<(), CudaLoadError> {
    let metadata = path
        .symlink_metadata()
        .map_err(|_| CudaLoadError::InvalidLibrary {
            library: library.to_owned(),
            path: path.to_owned(),
        })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAXIMUM_LIBRARY_BYTES
    {
        return Err(CudaLoadError::InvalidLibrary {
            library: library.to_owned(),
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn ensure_target(target: &str) -> Result<(), CudaLoadError> {
    if matches!(
        target,
        "aarch64-unknown-linux-gnu" | "x86_64-pc-windows-msvc" | "x86_64-unknown-linux-gnu"
    ) {
        Ok(())
    } else {
        Err(CudaLoadError::UnsupportedTarget {
            target: target.to_owned(),
        })
    }
}

pub struct RegistryCertifiedCudaImages {
    images: BTreeMap<&'static str, NonNull<c_void>>,
    core_ptx: Box<[u8]>,
    _certification: Arc<dyn Any + Send + Sync>,
}

impl RegistryCertifiedCudaImages {
    /// Projects exact retained images and kernel bytes already certified by
    /// `sim_runtime::NativeFfiRegistry` into the CUDA resource owner.
    ///
    /// # Safety
    ///
    /// Every handle must refer to the exact immutable image named by its registry certificate,
    /// and `certification` must retain those handles and all certificates for this value's
    /// lifetime. Package receipts, discovery, installed drivers, and feature compilation are not
    /// sufficient authorization.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_registry_certified_images(
        certification: Arc<dyn Any + Send + Sync>,
        driver: *mut c_void,
        nvrtc: *mut c_void,
        cublaslt: *mut c_void,
        cudnn: *mut c_void,
        core_ptx: &[u8],
        core_ptx_sha256: &str,
    ) -> Result<Self, CudaLoadError> {
        if core_ptx_sha256 != CORE_PTX_SHA256 || core_ptx != CORE_PTX {
            return Err(CudaLoadError::CertifiedKernelMismatch);
        }
        let mut images = BTreeMap::new();
        for (library, handle) in [
            ("driver", driver),
            ("nvrtc", nvrtc),
            ("cublaslt", cublaslt),
            ("cudnn", cudnn),
        ] {
            let handle = NonNull::new(handle).ok_or_else(|| CudaLoadError::UncertifiedHandle {
                library: library.to_owned(),
            })?;
            images.insert(library, handle);
        }
        Ok(Self {
            images,
            core_ptx: core_ptx.into(),
            _certification: certification,
        })
    }

    fn image(&self, library: &'static str) -> Result<NonNull<c_void>, CudaLoadError> {
        self.images
            .get(library)
            .copied()
            .ok_or_else(|| CudaLoadError::MissingCertifiedLibrary {
                library: library.to_owned(),
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeVersions {
    pub cuda_driver: i32,
    pub nvrtc_major: i32,
    pub nvrtc_minor: i32,
    pub cublaslt: usize,
    pub cublaslt_cudart: usize,
    pub cudnn: usize,
    pub cudnn_cudart: usize,
}

impl RuntimeVersions {
    pub fn validate(self) -> Result<Self, CudaLoadError> {
        if self.cuda_driver < CUDA_DRIVER_VERSION_MINIMUM {
            return Err(CudaLoadError::VersionTooOld {
                library: "driver",
                required: CUDA_DRIVER_VERSION_MINIMUM.to_string(),
                actual: self.cuda_driver.to_string(),
            });
        }
        if self.nvrtc_major != 12 || self.nvrtc_minor < 2 {
            return Err(CudaLoadError::VersionTooOld {
                library: "nvrtc",
                required: "12.2".to_owned(),
                actual: format!("{}.{}", self.nvrtc_major, self.nvrtc_minor),
            });
        }
        if self.cublaslt < CUBLASLT_VERSION_MINIMUM {
            return Err(CudaLoadError::VersionTooOld {
                library: "cublaslt",
                required: CUBLASLT_VERSION_MINIMUM.to_string(),
                actual: self.cublaslt.to_string(),
            });
        }
        if self.cudnn < CUDNN_VERSION_MINIMUM || self.cudnn / 10_000 != 9 {
            return Err(CudaLoadError::VersionTooOld {
                library: "cudnn",
                required: "9.x (at least 90000)".to_owned(),
                actual: self.cudnn.to_string(),
            });
        }
        for (library, cudart) in [
            ("cublaslt", self.cublaslt_cudart),
            ("cudnn", self.cudnn_cudart),
        ] {
            if cudart < CUDA_DRIVER_VERSION_MINIMUM as usize {
                return Err(CudaLoadError::VersionTooOld {
                    library,
                    required: CUDA_DRIVER_VERSION_MINIMUM.to_string(),
                    actual: cudart.to_string(),
                });
            }
        }
        Ok(self)
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CudaLoadError {
    #[error("NVIDIA CUDA target is unsupported: {target}")]
    UnsupportedTarget { target: String },
    #[error("{variable} is not valid Unicode")]
    InvalidEnvironment { variable: &'static str },
    #[error("invalid {discovery_source:?} CUDA discovery root {path}")]
    InvalidRoot {
        discovery_source: DiscoverySource,
        path: PathBuf,
    },
    #[error(
        "CUDA library {library} at {path} is missing, linked, empty, nonregular, or over 2 GiB"
    )]
    InvalidLibrary { library: String, path: PathBuf },
    #[error("CUDA ABI manifest is invalid: {0}")]
    Manifest(String),
    #[error("certified CUDA library {library} is missing")]
    MissingCertifiedLibrary { library: String },
    #[error("certified CUDA library {library} has a null retained module handle")]
    UncertifiedHandle { library: String },
    #[error("certified CUDA core PTX bytes or digest differ from the reviewed package payload")]
    CertifiedKernelMismatch,
    #[error("required CUDA symbol {symbol} is missing from {library}")]
    MissingSymbol {
        library: &'static str,
        symbol: &'static str,
    },
    #[error("CUDA library {library} version {actual} is below or incompatible with {required}")]
    VersionTooOld {
        library: &'static str,
        required: String,
        actual: String,
    },
    #[error("CUDA operation {operation} failed with status {status}")]
    CallFailed {
        operation: &'static str,
        status: i32,
    },
    #[error("CUDA operation {operation} returned a null resource")]
    NullResource { operation: &'static str },
    #[error("invalid CUDA {operation} argument: {reason}")]
    InvalidArgument {
        operation: &'static str,
        reason: &'static str,
    },
}

struct CudaSymbols {
    cu_init: CuInit,
    cu_driver_get_version: CuDriverGetVersion,
    cu_device_get_count: CuDeviceGetCount,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_destroy: CuCtxDestroy,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_mem_get_info: CuMemGetInfo,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod_async: CuMemcpyHtoDAsync,
    cu_memcpy_dtoh_async: CuMemcpyDtoHAsync,
    cu_stream_create: CuStreamCreate,
    cu_stream_destroy: CuStreamDestroy,
    cu_stream_synchronize: CuStreamSynchronize,
    _cu_event_create: CuEventCreate,
    _cu_event_destroy: CuEventDestroy,
    _cu_event_record: CuEventRecord,
    _cu_event_synchronize: CuEventSynchronize,
    cu_module_load_data: CuModuleLoadData,
    cu_module_unload: CuModuleUnload,
    cu_module_get_function: CuModuleGetFunction,
    cu_launch_kernel: CuLaunchKernel,
    _nvrtc_create_program: NvrtcCreateProgram,
    _nvrtc_compile_program: NvrtcCompileProgram,
    _nvrtc_destroy_program: NvrtcDestroyProgram,
    _nvrtc_get_program_log_size: NvrtcGetProgramLogSize,
    _nvrtc_get_program_log: NvrtcGetProgramLog,
    _nvrtc_get_ptx_size: NvrtcGetPtxSize,
    _nvrtc_get_ptx: NvrtcGetPtx,
    nvrtc_version: NvrtcVersion,
    cublaslt_create: CublasLtCreate,
    cublaslt_destroy: CublasLtDestroy,
    cublaslt_get_version: CublasLtGetVersion,
    cublaslt_get_cudart_version: CublasLtGetCudartVersion,
    cudnn_create: CudnnCreate,
    cudnn_destroy: CudnnDestroy,
    cudnn_get_version: CudnnGetVersion,
    cudnn_get_cudart_version: CudnnGetCudartVersion,
    cudnn_set_stream: CudnnSetStream,
}

impl CudaSymbols {
    unsafe fn load(images: &RegistryCertifiedCudaImages) -> Result<Self, CudaLoadError> {
        macro_rules! symbol {
            ($library:literal, $name:literal, $type:ty) => {{
                let address = required_address(images, $library, $name)?;
                unsafe { std::mem::transmute::<*mut c_void, $type>(address.as_ptr()) }
            }};
        }
        Ok(Self {
            cu_init: symbol!("driver", "cuInit", CuInit),
            cu_driver_get_version: symbol!("driver", "cuDriverGetVersion", CuDriverGetVersion),
            cu_device_get_count: symbol!("driver", "cuDeviceGetCount", CuDeviceGetCount),
            cu_device_get: symbol!("driver", "cuDeviceGet", CuDeviceGet),
            cu_ctx_create: symbol!("driver", "cuCtxCreate_v2", CuCtxCreate),
            cu_ctx_destroy: symbol!("driver", "cuCtxDestroy_v2", CuCtxDestroy),
            cu_ctx_set_current: symbol!("driver", "cuCtxSetCurrent", CuCtxSetCurrent),
            cu_mem_get_info: symbol!("driver", "cuMemGetInfo_v2", CuMemGetInfo),
            cu_mem_alloc: symbol!("driver", "cuMemAlloc_v2", CuMemAlloc),
            cu_mem_free: symbol!("driver", "cuMemFree_v2", CuMemFree),
            cu_memcpy_htod_async: symbol!("driver", "cuMemcpyHtoDAsync_v2", CuMemcpyHtoDAsync),
            cu_memcpy_dtoh_async: symbol!("driver", "cuMemcpyDtoHAsync_v2", CuMemcpyDtoHAsync),
            cu_stream_create: symbol!("driver", "cuStreamCreate", CuStreamCreate),
            cu_stream_destroy: symbol!("driver", "cuStreamDestroy_v2", CuStreamDestroy),
            cu_stream_synchronize: symbol!("driver", "cuStreamSynchronize", CuStreamSynchronize),
            _cu_event_create: symbol!("driver", "cuEventCreate", CuEventCreate),
            _cu_event_destroy: symbol!("driver", "cuEventDestroy_v2", CuEventDestroy),
            _cu_event_record: symbol!("driver", "cuEventRecord", CuEventRecord),
            _cu_event_synchronize: symbol!("driver", "cuEventSynchronize", CuEventSynchronize),
            cu_module_load_data: symbol!("driver", "cuModuleLoadData", CuModuleLoadData),
            cu_module_unload: symbol!("driver", "cuModuleUnload", CuModuleUnload),
            cu_module_get_function: symbol!("driver", "cuModuleGetFunction", CuModuleGetFunction),
            cu_launch_kernel: symbol!("driver", "cuLaunchKernel", CuLaunchKernel),
            _nvrtc_create_program: symbol!("nvrtc", "nvrtcCreateProgram", NvrtcCreateProgram),
            _nvrtc_compile_program: symbol!("nvrtc", "nvrtcCompileProgram", NvrtcCompileProgram),
            _nvrtc_destroy_program: symbol!("nvrtc", "nvrtcDestroyProgram", NvrtcDestroyProgram),
            _nvrtc_get_program_log_size: symbol!(
                "nvrtc",
                "nvrtcGetProgramLogSize",
                NvrtcGetProgramLogSize
            ),
            _nvrtc_get_program_log: symbol!("nvrtc", "nvrtcGetProgramLog", NvrtcGetProgramLog),
            _nvrtc_get_ptx_size: symbol!("nvrtc", "nvrtcGetPTXSize", NvrtcGetPtxSize),
            _nvrtc_get_ptx: symbol!("nvrtc", "nvrtcGetPTX", NvrtcGetPtx),
            nvrtc_version: symbol!("nvrtc", "nvrtcVersion", NvrtcVersion),
            cublaslt_create: symbol!("cublaslt", "cublasLtCreate", CublasLtCreate),
            cublaslt_destroy: symbol!("cublaslt", "cublasLtDestroy", CublasLtDestroy),
            cublaslt_get_version: symbol!("cublaslt", "cublasLtGetVersion", CublasLtGetVersion),
            cublaslt_get_cudart_version: symbol!(
                "cublaslt",
                "cublasLtGetCudartVersion",
                CublasLtGetCudartVersion
            ),
            cudnn_create: symbol!("cudnn", "cudnnCreate", CudnnCreate),
            cudnn_destroy: symbol!("cudnn", "cudnnDestroy", CudnnDestroy),
            cudnn_get_version: symbol!("cudnn", "cudnnGetVersion", CudnnGetVersion),
            cudnn_get_cudart_version: symbol!(
                "cudnn",
                "cudnnGetCudartVersion",
                CudnnGetCudartVersion
            ),
            cudnn_set_stream: symbol!("cudnn", "cudnnSetStream", CudnnSetStream),
        })
    }
}

fn required_address(
    images: &RegistryCertifiedCudaImages,
    library: &'static str,
    symbol: &'static str,
) -> Result<NonNull<c_void>, CudaLoadError> {
    let image = images.image(library)?;
    let name =
        CString::new(symbol).map_err(|_| CudaLoadError::MissingSymbol { library, symbol })?;
    let address = unsafe { platform_symbol(image, &name) };
    NonNull::new(address).ok_or(CudaLoadError::MissingSymbol { library, symbol })
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
unsafe fn platform_symbol(handle: NonNull<c_void>, name: &CString) -> *mut c_void {
    unsafe { libc::dlsym(handle.as_ptr(), name.as_ptr()) }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
unsafe fn platform_symbol(handle: NonNull<c_void>, name: &CString) -> *mut c_void {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    unsafe { GetProcAddress(handle.as_ptr(), name.as_ptr().cast()) }
}

#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ),
    all(target_os = "windows", target_arch = "x86_64")
)))]
unsafe fn platform_symbol(_handle: NonNull<c_void>, _name: &CString) -> *mut c_void {
    std::ptr::null_mut()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaAbiProbe {
    pub(crate) versions: RuntimeVersions,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeCudaDeviceFacts {
    pub(crate) device_ordinal: usize,
    pub(crate) name: String,
    pub(crate) total_memory_bytes: u64,
    pub(crate) maximum_allocation_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeCudaElementType {
    F16,
    F32,
}

impl NativeCudaElementType {
    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

struct NativeCudaAllocation {
    pointer: CuDevicePtr,
    bytes: usize,
    dimensions: Vec<i64>,
    element_type: NativeCudaElementType,
}

pub(crate) struct OwnedCudaCore {
    _images: RegistryCertifiedCudaImages,
    symbols: CudaSymbols,
    probe: CudaAbiProbe,
    device_facts: NativeCudaDeviceFacts,
    context: Option<NonNull<c_void>>,
    stream: Option<NonNull<c_void>>,
    module: Option<NonNull<c_void>>,
    add_f32: Option<NonNull<c_void>>,
    cublaslt: Option<NonNull<c_void>>,
    cudnn: Option<NonNull<c_void>>,
    allocations: BTreeMap<u64, NativeCudaAllocation>,
}

// No pointer or vendor resource escapes this crate-private type. CudaExecutionSession serializes
// every call and teardown through one mutex before it is allowed to cross threads.
unsafe impl Send for OwnedCudaCore {}

impl OwnedCudaCore {
    pub(crate) fn load_certified(
        images: RegistryCertifiedCudaImages,
        device_ordinal: usize,
    ) -> Result<Self, CudaLoadError> {
        ensure_target(env!("SIM_CUDA_TARGET"))?;
        let symbols = unsafe { CudaSymbols::load(&images)? };
        check_cu("cuInit", unsafe { (symbols.cu_init)(0) })?;

        let mut cuda_driver = 0;
        check_cu("cuDriverGetVersion", unsafe {
            (symbols.cu_driver_get_version)(&mut cuda_driver)
        })?;
        let mut nvrtc_major = 0;
        let mut nvrtc_minor = 0;
        check_nvrtc("nvrtcVersion", unsafe {
            (symbols.nvrtc_version)(&mut nvrtc_major, &mut nvrtc_minor)
        })?;
        let versions = RuntimeVersions {
            cuda_driver,
            nvrtc_major,
            nvrtc_minor,
            cublaslt: unsafe { (symbols.cublaslt_get_version)() },
            cublaslt_cudart: unsafe { (symbols.cublaslt_get_cudart_version)() },
            cudnn: unsafe { (symbols.cudnn_get_version)() },
            cudnn_cudart: unsafe { (symbols.cudnn_get_cudart_version)() },
        }
        .validate()?;

        let mut device_count = 0;
        check_cu("cuDeviceGetCount", unsafe {
            (symbols.cu_device_get_count)(&mut device_count)
        })?;
        let device_count =
            usize::try_from(device_count).map_err(|_| CudaLoadError::InvalidArgument {
                operation: "cuDeviceGetCount",
                reason: "driver returned a negative device count",
            })?;
        if device_ordinal >= device_count {
            return Err(CudaLoadError::InvalidArgument {
                operation: "cuDeviceGet",
                reason: "device ordinal is outside the certified inventory",
            });
        }
        let ordinal =
            i32::try_from(device_ordinal).map_err(|_| CudaLoadError::InvalidArgument {
                operation: "cuDeviceGet",
                reason: "device ordinal exceeds i32",
            })?;
        let mut device: CuDevice = 0;
        check_cu("cuDeviceGet", unsafe {
            (symbols.cu_device_get)(&mut device, ordinal)
        })?;
        let mut context = std::ptr::null_mut();
        check_cu("cuCtxCreate_v2", unsafe {
            (symbols.cu_ctx_create)(&mut context, 0, device)
        })?;
        let context = NonNull::new(context).ok_or(CudaLoadError::NullResource {
            operation: "cuCtxCreate_v2",
        })?;

        let mut core = Self {
            _images: images,
            symbols,
            probe: CudaAbiProbe { versions },
            device_facts: NativeCudaDeviceFacts {
                device_ordinal,
                name: format!("NVIDIA CUDA device {device_ordinal}"),
                total_memory_bytes: 1,
                maximum_allocation_bytes: 1,
            },
            context: Some(context),
            stream: None,
            module: None,
            add_f32: None,
            cublaslt: None,
            cudnn: None,
            allocations: BTreeMap::new(),
        };
        core.initialize_resources()?;
        Ok(core)
    }

    fn initialize_resources(&mut self) -> Result<(), CudaLoadError> {
        self.make_current()?;
        let mut free = 0;
        let mut total = 0;
        check_cu("cuMemGetInfo_v2", unsafe {
            (self.symbols.cu_mem_get_info)(&mut free, &mut total)
        })?;
        if free == 0 || total == 0 || free > total {
            return Err(CudaLoadError::InvalidArgument {
                operation: "cuMemGetInfo_v2",
                reason: "device memory facts are empty or inconsistent",
            });
        }
        self.device_facts.total_memory_bytes =
            u64::try_from(total).map_err(|_| CudaLoadError::InvalidArgument {
                operation: "cuMemGetInfo_v2",
                reason: "total memory exceeds u64",
            })?;
        self.device_facts.maximum_allocation_bytes =
            u64::try_from(free).map_err(|_| CudaLoadError::InvalidArgument {
                operation: "cuMemGetInfo_v2",
                reason: "free memory exceeds u64",
            })?;

        let mut stream = std::ptr::null_mut();
        check_cu("cuStreamCreate", unsafe {
            (self.symbols.cu_stream_create)(&mut stream, 0)
        })?;
        let stream = NonNull::new(stream).ok_or(CudaLoadError::NullResource {
            operation: "cuStreamCreate",
        })?;
        self.stream = Some(stream);

        let mut cublaslt = std::ptr::null_mut();
        check_cublas("cublasLtCreate", unsafe {
            (self.symbols.cublaslt_create)(&mut cublaslt)
        })?;
        self.cublaslt = Some(NonNull::new(cublaslt).ok_or(CudaLoadError::NullResource {
            operation: "cublasLtCreate",
        })?);

        let mut cudnn = std::ptr::null_mut();
        check_cudnn("cudnnCreate", unsafe {
            (self.symbols.cudnn_create)(&mut cudnn)
        })?;
        let cudnn = NonNull::new(cudnn).ok_or(CudaLoadError::NullResource {
            operation: "cudnnCreate",
        })?;
        check_cudnn("cudnnSetStream", unsafe {
            (self.symbols.cudnn_set_stream)(cudnn.as_ptr(), stream.as_ptr())
        })?;
        self.cudnn = Some(cudnn);

        let ptx = CString::new(self._images.core_ptx.as_ref()).map_err(|_| {
            CudaLoadError::InvalidArgument {
                operation: "cuModuleLoadData",
                reason: "certified PTX contains an interior NUL byte",
            }
        })?;
        let mut module = std::ptr::null_mut();
        check_cu("cuModuleLoadData", unsafe {
            (self.symbols.cu_module_load_data)(&mut module, ptx.as_ptr().cast())
        })?;
        let module = NonNull::new(module).ok_or(CudaLoadError::NullResource {
            operation: "cuModuleLoadData",
        })?;
        self.module = Some(module);

        let mut function = std::ptr::null_mut();
        check_cu("cuModuleGetFunction", unsafe {
            (self.symbols.cu_module_get_function)(
                &mut function,
                module.as_ptr(),
                c"zed_cuda_add_f32".as_ptr(),
            )
        })?;
        self.add_f32 = Some(NonNull::new(function).ok_or(CudaLoadError::NullResource {
            operation: "cuModuleGetFunction",
        })?);
        Ok(())
    }

    pub(crate) const fn probe(&self) -> &CudaAbiProbe {
        &self.probe
    }

    pub(crate) const fn device_facts(&self) -> &NativeCudaDeviceFacts {
        &self.device_facts
    }

    fn make_current(&self) -> Result<(), CudaLoadError> {
        let context = self.context.ok_or(CudaLoadError::NullResource {
            operation: "cuda_context",
        })?;
        check_cu("cuCtxSetCurrent", unsafe {
            (self.symbols.cu_ctx_set_current)(context.as_ptr())
        })
    }

    pub(crate) fn allocate(
        &mut self,
        id: u64,
        dimensions: &[i64],
        element_type: NativeCudaElementType,
    ) -> Result<usize, CudaLoadError> {
        let bytes = dimensions
            .iter()
            .try_fold(element_type.byte_width(), |bytes, dimension| {
                let dimension =
                    usize::try_from(*dimension).map_err(|_| CudaLoadError::InvalidArgument {
                        operation: "cuMemAlloc_v2",
                        reason: "tensor dimensions are invalid",
                    })?;
                if dimension == 0 {
                    return Err(CudaLoadError::InvalidArgument {
                        operation: "cuMemAlloc_v2",
                        reason: "tensor dimensions must be nonzero",
                    });
                }
                bytes
                    .checked_mul(dimension)
                    .ok_or(CudaLoadError::InvalidArgument {
                        operation: "cuMemAlloc_v2",
                        reason: "tensor byte count overflows usize",
                    })
            })?;
        if self.allocations.contains_key(&id) {
            return Err(CudaLoadError::InvalidArgument {
                operation: "cuMemAlloc_v2",
                reason: "allocation identifier is duplicated",
            });
        }
        self.make_current()?;
        let mut pointer = 0;
        check_cu("cuMemAlloc_v2", unsafe {
            (self.symbols.cu_mem_alloc)(&mut pointer, bytes)
        })?;
        if pointer == 0 {
            return Err(CudaLoadError::NullResource {
                operation: "cuMemAlloc_v2",
            });
        }
        self.allocations.insert(
            id,
            NativeCudaAllocation {
                pointer,
                bytes,
                dimensions: dimensions.to_vec(),
                element_type,
            },
        );
        Ok(bytes)
    }

    pub(crate) fn release_allocation(&mut self, id: u64) -> Result<(), CudaLoadError> {
        let allocation = self
            .allocations
            .remove(&id)
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuMemFree_v2",
                reason: "allocation identifier is unknown",
            })?;
        self.make_current()?;
        check_cu("cuMemFree_v2", unsafe {
            (self.symbols.cu_mem_free)(allocation.pointer)
        })
    }

    pub(crate) fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        source: &[u8],
    ) -> Result<(), CudaLoadError> {
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuMemcpyHtoDAsync_v2",
                reason: "allocation identifier is unknown",
            })?;
        let end = offset
            .checked_add(source.len())
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuMemcpyHtoDAsync_v2",
                reason: "copy range overflows usize",
            })?;
        if end > allocation.bytes {
            return Err(CudaLoadError::InvalidArgument {
                operation: "cuMemcpyHtoDAsync_v2",
                reason: "copy range exceeds allocation",
            });
        }
        let pointer = allocation.pointer.checked_add(offset as u64).ok_or(
            CudaLoadError::InvalidArgument {
                operation: "cuMemcpyHtoDAsync_v2",
                reason: "device pointer offset overflows",
            },
        )?;
        let stream = self.stream.ok_or(CudaLoadError::NullResource {
            operation: "cuda_stream",
        })?;
        self.make_current()?;
        check_cu("cuMemcpyHtoDAsync_v2", unsafe {
            (self.symbols.cu_memcpy_htod_async)(
                pointer,
                source.as_ptr().cast(),
                source.len(),
                stream.as_ptr(),
            )
        })
    }

    pub(crate) fn copy_to_host(
        &mut self,
        id: u64,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), CudaLoadError> {
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuMemcpyDtoHAsync_v2",
                reason: "allocation identifier is unknown",
            })?;
        let end = offset
            .checked_add(destination.len())
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuMemcpyDtoHAsync_v2",
                reason: "copy range overflows usize",
            })?;
        if end > allocation.bytes {
            return Err(CudaLoadError::InvalidArgument {
                operation: "cuMemcpyDtoHAsync_v2",
                reason: "copy range exceeds allocation",
            });
        }
        let pointer = allocation.pointer.checked_add(offset as u64).ok_or(
            CudaLoadError::InvalidArgument {
                operation: "cuMemcpyDtoHAsync_v2",
                reason: "device pointer offset overflows",
            },
        )?;
        let stream = self.stream.ok_or(CudaLoadError::NullResource {
            operation: "cuda_stream",
        })?;
        self.make_current()?;
        check_cu("cuMemcpyDtoHAsync_v2", unsafe {
            (self.symbols.cu_memcpy_dtoh_async)(
                destination.as_mut_ptr().cast(),
                pointer,
                destination.len(),
                stream.as_ptr(),
            )
        })?;
        self.synchronize()
    }

    pub(crate) fn add(&mut self, left: u64, right: u64, output: u64) -> Result<(), CudaLoadError> {
        let left = self
            .allocations
            .get(&left)
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuLaunchKernel",
                reason: "left allocation identifier is unknown",
            })?;
        let right = self
            .allocations
            .get(&right)
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuLaunchKernel",
                reason: "right allocation identifier is unknown",
            })?;
        let output = self
            .allocations
            .get(&output)
            .ok_or(CudaLoadError::InvalidArgument {
                operation: "cuLaunchKernel",
                reason: "output allocation identifier is unknown",
            })?;
        if left.dimensions != right.dimensions
            || left.dimensions != output.dimensions
            || left.element_type != NativeCudaElementType::F32
            || right.element_type != NativeCudaElementType::F32
            || output.element_type != NativeCudaElementType::F32
        {
            return Err(CudaLoadError::InvalidArgument {
                operation: "cuLaunchKernel",
                reason: "reviewed core kernel requires matching F32 allocations",
            });
        }
        let element_count = left.bytes / std::mem::size_of::<f32>();
        if element_count == 0 {
            return Ok(());
        }
        let function = self.add_f32.ok_or(CudaLoadError::NullResource {
            operation: "zed_cuda_add_f32",
        })?;
        let stream = self.stream.ok_or(CudaLoadError::NullResource {
            operation: "cuda_stream",
        })?;
        let block_size = 256_u32;
        let grid_size = element_count
            .div_ceil(block_size as usize)
            .try_into()
            .map_err(|_| CudaLoadError::InvalidArgument {
                operation: "cuLaunchKernel",
                reason: "grid size exceeds u32",
            })?;
        let mut left_pointer = left.pointer;
        let mut right_pointer = right.pointer;
        let mut output_pointer = output.pointer;
        let mut count =
            u64::try_from(element_count).map_err(|_| CudaLoadError::InvalidArgument {
                operation: "cuLaunchKernel",
                reason: "element count exceeds u64",
            })?;
        let mut arguments = [
            (&mut left_pointer as *mut CuDevicePtr).cast::<c_void>(),
            (&mut right_pointer as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output_pointer as *mut CuDevicePtr).cast::<c_void>(),
            (&mut count as *mut u64).cast::<c_void>(),
        ];
        self.make_current()?;
        check_cu("cuLaunchKernel", unsafe {
            (self.symbols.cu_launch_kernel)(
                function.as_ptr(),
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                stream.as_ptr(),
                arguments.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }

    pub(crate) fn synchronize(&mut self) -> Result<(), CudaLoadError> {
        let stream = self.stream.ok_or(CudaLoadError::NullResource {
            operation: "cuda_stream",
        })?;
        self.make_current()?;
        check_cu("cuStreamSynchronize", unsafe {
            (self.symbols.cu_stream_synchronize)(stream.as_ptr())
        })
    }
}

impl Drop for OwnedCudaCore {
    fn drop(&mut self) {
        if let Err(error) = self.make_current() {
            eprintln!("sim_backend_cuda: failed to make context current during teardown: {error}");
        }
        for allocation in std::mem::take(&mut self.allocations).into_values() {
            let status = unsafe { (self.symbols.cu_mem_free)(allocation.pointer) };
            if status != CuResult::SUCCESS {
                eprintln!(
                    "sim_backend_cuda: failed to free allocation during teardown: status {}",
                    status.0
                );
            }
        }
        if let Some(cudnn) = self.cudnn.take() {
            let status = unsafe { (self.symbols.cudnn_destroy)(cudnn.as_ptr()) };
            if status != CudnnStatus::SUCCESS {
                eprintln!(
                    "sim_backend_cuda: failed to destroy cuDNN: status {}",
                    status.0
                );
            }
        }
        if let Some(cublaslt) = self.cublaslt.take() {
            let status = unsafe { (self.symbols.cublaslt_destroy)(cublaslt.as_ptr()) };
            if status != CublasStatus::SUCCESS {
                eprintln!(
                    "sim_backend_cuda: failed to destroy cuBLASLt: status {}",
                    status.0
                );
            }
        }
        if let Some(module) = self.module.take() {
            let status = unsafe { (self.symbols.cu_module_unload)(module.as_ptr()) };
            if status != CuResult::SUCCESS {
                eprintln!(
                    "sim_backend_cuda: failed to unload module: status {}",
                    status.0
                );
            }
        }
        if let Some(stream) = self.stream.take() {
            let status = unsafe { (self.symbols.cu_stream_destroy)(stream.as_ptr()) };
            if status != CuResult::SUCCESS {
                eprintln!(
                    "sim_backend_cuda: failed to destroy stream: status {}",
                    status.0
                );
            }
        }
        if let Some(context) = self.context.take() {
            let status = unsafe { (self.symbols.cu_ctx_destroy)(context.as_ptr()) };
            if status != CuResult::SUCCESS {
                eprintln!(
                    "sim_backend_cuda: failed to destroy context: status {}",
                    status.0
                );
            }
        }
    }
}

fn check_cu(operation: &'static str, status: CuResult) -> Result<(), CudaLoadError> {
    if status == CuResult::SUCCESS {
        Ok(())
    } else {
        Err(CudaLoadError::CallFailed {
            operation,
            status: status.0,
        })
    }
}

fn check_nvrtc(
    operation: &'static str,
    status: crate::abi::NvrtcResult,
) -> Result<(), CudaLoadError> {
    if status == crate::abi::NvrtcResult::SUCCESS {
        Ok(())
    } else {
        Err(CudaLoadError::CallFailed {
            operation,
            status: status.0,
        })
    }
}

fn check_cublas(operation: &'static str, status: CublasStatus) -> Result<(), CudaLoadError> {
    if status == CublasStatus::SUCCESS {
        Ok(())
    } else {
        Err(CudaLoadError::CallFailed {
            operation,
            status: status.0,
        })
    }
}

fn check_cudnn(operation: &'static str, status: CudnnStatus) -> Result<(), CudaLoadError> {
    if status == CudnnStatus::SUCCESS {
        Ok(())
    } else {
        Err(CudaLoadError::CallFailed {
            operation,
            status: status.0,
        })
    }
}

pub fn unavailable_reason() -> String {
    format!(
        "NVIDIA CUDA unavailable on {}: {} requires registry-certified retained driver, NVRTC, cuBLASLt, and cuDNN images; discovery, package receipts, version probes, and feature compilation are never authorization (certificate owner: {CERTIFICATE_OWNER})",
        env!("SIM_CUDA_TARGET"),
        ABI_FLOOR,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn certified_images() -> Result<RegistryCertifiedCudaImages, CudaLoadError> {
        let handle = NonNull::<c_void>::dangling().as_ptr();
        unsafe {
            RegistryCertifiedCudaImages::from_registry_certified_images(
                Arc::new(()),
                handle,
                handle,
                handle,
                handle,
                CORE_PTX,
                CORE_PTX_SHA256,
            )
        }
    }

    #[test]
    fn discovery_order_and_target_gate_are_exact() -> Result<(), Box<dyn std::error::Error>> {
        let environment = DiscoveryEnvironment {
            sim_cuda_root: Some(OsString::from("/opt/sim-cuda")),
            cuda_path: Some(OsString::from("/opt/cuda")),
        };
        let certificate = ();
        let package = unsafe {
            SignedPackageRoot::from_runtime_verified_path(
                &certificate,
                PathBuf::from("/opt/signed-cuda"),
            )?
        };
        let candidates =
            discovery_candidates("x86_64-unknown-linux-gnu", &environment, &[package])?;
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.source)
                .collect::<Vec<_>>(),
            vec![
                DiscoverySource::SimCudaRoot,
                DiscoverySource::CudaPath,
                DiscoverySource::SignedPackage,
                DiscoverySource::InstalledDriver,
            ]
        );
        assert_eq!(
            candidates[0].libraries.get("driver"),
            Some(&PathBuf::from("/opt/sim-cuda/lib64/libcuda.so.1"))
        );
        assert!(matches!(
            discovery_candidates("aarch64-apple-darwin", &environment, &[]),
            Err(CudaLoadError::UnsupportedTarget { .. })
        ));
        Ok(())
    }

    #[test]
    fn ordinary_paths_are_observations_not_certificates() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = std::env::temp_dir().join(format!("sim-cuda-loader-{}", std::process::id()));
        match fs::remove_dir_all(&root) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        fs::create_dir_all(root.join("lib64"))?;
        let library = root.join("lib64/libcuda.so.1");
        fs::write(&library, b"not a loadable library")?;
        validate_discovered_library(&library, "driver")?;
        let status = unavailable_reason();
        assert!(status.contains("never authorization"));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn certificate_projection_rejects_null_images_and_changed_kernel_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let valid = certified_images()?;
        assert_eq!(valid.images.len(), 4);
        let handle = NonNull::<c_void>::dangling().as_ptr();
        assert!(matches!(
            unsafe {
                RegistryCertifiedCudaImages::from_registry_certified_images(
                    Arc::new(()),
                    std::ptr::null_mut(),
                    handle,
                    handle,
                    handle,
                    CORE_PTX,
                    CORE_PTX_SHA256,
                )
            },
            Err(CudaLoadError::UncertifiedHandle { library }) if library == "driver"
        ));
        assert!(matches!(
            unsafe {
                RegistryCertifiedCudaImages::from_registry_certified_images(
                    Arc::new(()),
                    handle,
                    handle,
                    handle,
                    handle,
                    b"changed PTX",
                    CORE_PTX_SHA256,
                )
            },
            Err(CudaLoadError::CertifiedKernelMismatch)
        ));
        Ok(())
    }

    #[test]
    fn runtime_version_contract_fails_closed() {
        let valid = RuntimeVersions {
            cuda_driver: 12_020,
            nvrtc_major: 12,
            nvrtc_minor: 2,
            cublaslt: 120_205,
            cublaslt_cudart: 12_020,
            cudnn: 90_000,
            cudnn_cudart: 12_020,
        };
        assert_eq!(valid.validate(), Ok(valid));
        assert!(matches!(
            RuntimeVersions {
                cuda_driver: 12_010,
                ..valid
            }
            .validate(),
            Err(CudaLoadError::VersionTooOld {
                library: "driver",
                ..
            })
        ));
        assert!(matches!(
            RuntimeVersions {
                cudnn: 100_000,
                ..valid
            }
            .validate(),
            Err(CudaLoadError::VersionTooOld {
                library: "cudnn",
                ..
            })
        ));
    }

    #[test]
    fn package_and_binding_owners_are_not_vendor_adapter_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.certificate_owner, CERTIFICATE_OWNER);
        assert!(!manifest.package_policy.redistribute_driver);
        assert!(manifest.package_policy.approved_redistributables.is_empty());
        assert!(!manifest.package_policy.runtime_compilation_for_core_kernels);
        Ok(())
    }

    #[test]
    fn core_ptx_requires_exact_certificate_bound_bytes_and_digest()
    -> Result<(), Box<dyn std::error::Error>> {
        let handle = NonNull::<c_void>::dangling().as_ptr();
        certified_images()?;
        assert!(matches!(
            unsafe {
                RegistryCertifiedCudaImages::from_registry_certified_images(
                    Arc::new(()),
                    handle,
                    handle,
                    handle,
                    handle,
                    b".version 7.8\n",
                    CORE_PTX_SHA256,
                )
            },
            Err(CudaLoadError::CertifiedKernelMismatch)
        ));
        assert!(matches!(
            unsafe {
                RegistryCertifiedCudaImages::from_registry_certified_images(
                    Arc::new(()),
                    handle,
                    handle,
                    handle,
                    handle,
                    CORE_PTX,
                    &"0".repeat(64),
                )
            },
            Err(CudaLoadError::CertifiedKernelMismatch)
        ));
        Ok(())
    }
}
