use std::{
    any::Any,
    collections::BTreeMap,
    env,
    ffi::{CStr, OsString, c_void},
    marker::PhantomData,
    path::{Component, Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
use std::ffi::CString;

use thiserror::Error;

use crate::abi::{
    ABI_FLOOR, ACL_SUCCESS, AclCreateDataBuffer, AclCreateTensorDesc, AclDataBuffer, AclDataType,
    AclDestroyDataBuffer, AclDestroyTensorDesc, AclFinalize, AclFormat, AclGetRecentErrMsg,
    AclInit, AclTensorDesc, AclopExecuteV2, AclrtCreateContext, AclrtCreateEvent,
    AclrtCreateStream, AclrtDestroyContext, AclrtDestroyEvent, AclrtDestroyStream, AclrtFree,
    AclrtGetDeviceCount, AclrtGetMemInfo, AclrtGetSocName, AclrtGetVersion, AclrtMalloc,
    AclrtMemcpy, AclrtMemcpyAsync, AclrtRecordEvent, AclrtResetDevice, AclrtSetCurrentContext,
    AclrtSetDevice, AclrtSynchronizeEvent, AclrtSynchronizeStream, CannVersion,
};

const ASCENDCL_RELATIVE_PATH: &str = "lib64/libascendcl.so";
const RUNTIME_RELATIVE_PATH: &str = "lib64/libruntime.so";
const MAXIMUM_LIBRARY_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscoverySource {
    SimAscendRoot,
    AscendHomePath,
    SignedPackage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryRoot {
    pub source: DiscoverySource,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiscoveryEnvironment {
    pub sim_ascend_root: Option<OsString>,
    pub ascend_home_path: Option<OsString>,
}

impl DiscoveryEnvironment {
    pub fn from_process() -> Self {
        Self {
            sim_ascend_root: env::var_os("SIM_ASCEND_ROOT"),
            ascend_home_path: env::var_os("ASCEND_HOME_PATH"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct SignedPackageRoot<'certificate> {
    path: PathBuf,
    certificate_lifetime: PhantomData<&'certificate ()>,
}

impl<'certificate> SignedPackageRoot<'certificate> {
    /// Creates the path projection of a package already admitted by the runtime trust owner.
    ///
    /// # Safety
    ///
    /// The caller must retain the successful signer-bound package certificate and immutable
    /// package image issued by the runtime trust layer for `path`. This constructor does not
    /// verify a signature and cannot turn an ordinary filesystem path into trusted input.
    pub unsafe fn from_runtime_verified_path<Certificate: ?Sized>(
        _certificate: &'certificate Certificate,
        path: PathBuf,
    ) -> Result<Self, NpuLoadError> {
        validate_root(&path, DiscoverySource::SignedPackage)?;
        Ok(Self {
            path,
            certificate_lifetime: PhantomData,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NpuLibraryCandidates {
    pub root: DiscoveryRoot,
    pub ascendcl: PathBuf,
    pub runtime: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredNpuLibraries {
    pub root: DiscoveryRoot,
    pub ascendcl: PathBuf,
    pub runtime: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LibraryFailure {
    pub source: DiscoverySource,
    pub library: &'static str,
    pub path: PathBuf,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum NpuLoadError {
    #[error(
        "Huawei Ascend NPU is unavailable on unsupported target {target}; expected x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu"
    )]
    UnsupportedTarget { target: String },
    #[error("{variable} is not valid Unicode")]
    InvalidEnvironment { variable: &'static str },
    #[error(
        "invalid {discovery_source:?} Ascend discovery root {path}: roots must be absolute and traversal-free"
    )]
    InvalidRoot {
        discovery_source: DiscoverySource,
        path: PathBuf,
    },
    #[error(
        "Ascend libraries are unavailable; checked {checked_roots} roots in SIM_ASCEND_ROOT, ASCEND_HOME_PATH, signed-package order: {failures:?}"
    )]
    MissingLibraries {
        checked_roots: usize,
        failures: Vec<LibraryFailure>,
    },
    #[error(
        "required Ascend library {library} at {path} is missing, a symlink, not regular, empty, or larger than 2 GiB"
    )]
    InvalidLibrary {
        library: &'static str,
        path: PathBuf,
    },
    #[error("required {library} symbol {symbol} is missing for {abi_floor}")]
    MissingSymbol {
        library: &'static str,
        symbol: &'static str,
        abi_floor: &'static str,
    },
    #[error("AscendCL version probe failed with aclError {code}")]
    VersionProbeFailed { code: i32 },
    #[error("AscendCL operation {operation} failed with aclError {code}")]
    AclCallFailed { operation: &'static str, code: i32 },
    #[error("invalid AscendCL {operation} argument: {reason}")]
    InvalidArgument {
        operation: &'static str,
        reason: &'static str,
    },
    #[error("AscendCL returned invalid API version {major}.{minor}.{patch}")]
    InvalidRuntimeVersion { major: i32, minor: i32, patch: i32 },
    #[error("CANN package version {found:?} is below the required 8.0.RC3 floor")]
    VersionTooOld { found: CannVersion },
    #[error(
        "NPU runtime handles must be non-null retained images certified by sim_runtime::NativeFfiRegistry"
    )]
    UncertifiedHandles,
}

pub fn supported_target() -> bool {
    cfg!(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))
}

pub fn unavailable_reason() -> String {
    if supported_target() {
        format!(
            "Huawei Ascend NPU requires {ABI_FLOOR}, libascendcl.so and libruntime.so from SIM_ASCEND_ROOT, ASCEND_HOME_PATH, or a signed package root, plus sim_runtime::NativeFfiRegistry-certified retained handles; feature compilation alone never certifies availability"
        )
    } else {
        format!(
            "Huawei Ascend NPU unsupported target {}; expected x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu and sim_runtime::NativeFfiRegistry certification",
            env!("SIM_NPU_TARGET")
        )
    }
}

pub fn validate_package_version(version: CannVersion) -> Result<(), NpuLoadError> {
    if version < CannVersion::FLOOR {
        return Err(NpuLoadError::VersionTooOld { found: version });
    }
    Ok(())
}

pub fn discover_library_candidates(
    environment: &DiscoveryEnvironment,
    signed_package_roots: &[SignedPackageRoot<'_>],
) -> Result<Vec<NpuLibraryCandidates>, NpuLoadError> {
    discover_library_candidates_for_target(
        env!("SIM_NPU_TARGET"),
        environment,
        signed_package_roots,
    )
}

pub fn discover_library_candidates_for_target(
    target: &str,
    environment: &DiscoveryEnvironment,
    signed_package_roots: &[SignedPackageRoot<'_>],
) -> Result<Vec<NpuLibraryCandidates>, NpuLoadError> {
    ensure_target(target)?;
    let mut roots = Vec::new();
    push_environment_root(
        &mut roots,
        environment.sim_ascend_root.as_ref(),
        "SIM_ASCEND_ROOT",
        DiscoverySource::SimAscendRoot,
    )?;
    push_environment_root(
        &mut roots,
        environment.ascend_home_path.as_ref(),
        "ASCEND_HOME_PATH",
        DiscoverySource::AscendHomePath,
    )?;
    for package in signed_package_roots {
        push_unique_root(
            &mut roots,
            DiscoveryRoot {
                source: DiscoverySource::SignedPackage,
                path: package.path.clone(),
            },
        );
    }

    roots
        .into_iter()
        .map(|root| {
            validate_root(&root.path, root.source)?;
            Ok(NpuLibraryCandidates {
                ascendcl: root.path.join(ASCENDCL_RELATIVE_PATH),
                runtime: root.path.join(RUNTIME_RELATIVE_PATH),
                root,
            })
        })
        .collect()
}

pub fn discover_installed_libraries(
    environment: &DiscoveryEnvironment,
    signed_package_roots: &[SignedPackageRoot<'_>],
) -> Result<DiscoveredNpuLibraries, NpuLoadError> {
    discover_installed_libraries_for_target(
        env!("SIM_NPU_TARGET"),
        environment,
        signed_package_roots,
    )
}

pub fn discover_installed_libraries_for_target(
    target: &str,
    environment: &DiscoveryEnvironment,
    signed_package_roots: &[SignedPackageRoot<'_>],
) -> Result<DiscoveredNpuLibraries, NpuLoadError> {
    let candidates =
        discover_library_candidates_for_target(target, environment, signed_package_roots)?;
    let checked_roots = candidates.len();
    let mut failures = Vec::new();
    for candidate in candidates {
        if validate_library(&candidate.ascendcl, "libascendcl.so").is_err() {
            failures.push(LibraryFailure {
                source: candidate.root.source,
                library: "libascendcl.so",
                path: candidate.ascendcl,
            });
            continue;
        }
        if validate_library(&candidate.runtime, "libruntime.so").is_err() {
            failures.push(LibraryFailure {
                source: candidate.root.source,
                library: "libruntime.so",
                path: candidate.runtime,
            });
            continue;
        }
        return Ok(DiscoveredNpuLibraries {
            root: candidate.root,
            ascendcl: candidate.ascendcl,
            runtime: candidate.runtime,
        });
    }
    Err(NpuLoadError::MissingLibraries {
        checked_roots,
        failures,
    })
}

fn ensure_target(target: &str) -> Result<(), NpuLoadError> {
    if !matches!(
        target,
        "x86_64-unknown-linux-gnu" | "aarch64-unknown-linux-gnu"
    ) {
        return Err(NpuLoadError::UnsupportedTarget {
            target: target.to_owned(),
        });
    }
    Ok(())
}

fn push_environment_root(
    roots: &mut Vec<DiscoveryRoot>,
    value: Option<&OsString>,
    variable: &'static str,
    source: DiscoverySource,
) -> Result<(), NpuLoadError> {
    let Some(value) = value else {
        return Ok(());
    };
    let path = value
        .to_str()
        .ok_or(NpuLoadError::InvalidEnvironment { variable })?;
    if path.is_empty() {
        return Ok(());
    }
    let root = DiscoveryRoot {
        source,
        path: PathBuf::from(path),
    };
    validate_root(&root.path, source)?;
    push_unique_root(roots, root);
    Ok(())
}

fn push_unique_root(roots: &mut Vec<DiscoveryRoot>, root: DiscoveryRoot) {
    if !roots.iter().any(|existing| existing.path == root.path) {
        roots.push(root);
    }
}

fn validate_root(path: &Path, source: DiscoverySource) -> Result<(), NpuLoadError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(NpuLoadError::InvalidRoot {
            discovery_source: source,
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn validate_library(path: &Path, library: &'static str) -> Result<(), NpuLoadError> {
    let metadata = path
        .symlink_metadata()
        .map_err(|_| NpuLoadError::InvalidLibrary {
            library,
            path: path.to_owned(),
        })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAXIMUM_LIBRARY_BYTES
    {
        return Err(NpuLoadError::InvalidLibrary {
            library,
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn check_acl(operation: &'static str, code: i32) -> Result<(), NpuLoadError> {
    if code != ACL_SUCCESS {
        return Err(NpuLoadError::AclCallFailed { operation, code });
    }
    Ok(())
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
unsafe fn required_symbol(
    handle: NonNull<c_void>,
    symbol: &'static str,
) -> Result<*mut c_void, NpuLoadError> {
    let symbol_name = CString::new(symbol).map_err(|_| NpuLoadError::MissingSymbol {
        library: "libascendcl.so",
        symbol,
        abi_floor: ABI_FLOOR,
    })?;
    let pointer = unsafe { libc::dlsym(handle.as_ptr(), symbol_name.as_ptr()) };
    if pointer.is_null() {
        return Err(NpuLoadError::MissingSymbol {
            library: "libascendcl.so",
            symbol,
            abi_floor: ABI_FLOOR,
        });
    }
    Ok(pointer)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeNpuProbe {
    pub(crate) api_version: (u32, u32, u32),
    pub(crate) device_count: u32,
    pub(crate) device_id: u32,
    pub(crate) device_name: String,
    pub(crate) free_memory_bytes: usize,
    pub(crate) total_memory_bytes: usize,
}

pub struct RegistryCertifiedNpuImages {
    ascendcl: NonNull<c_void>,
    runtime: NonNull<c_void>,
    _certification: Arc<dyn Any + Send + Sync>,
}

impl RegistryCertifiedNpuImages {
    /// Projects exact retained images certified by `sim_runtime::NativeFfiRegistry`.
    ///
    /// # Safety
    ///
    /// Both handles must refer to the immutable certified `libascendcl.so` and `libruntime.so`
    /// images and remain live through `certification`. Discovery results, installed SDK paths,
    /// package receipts, and compiled features do not satisfy this contract.
    pub unsafe fn from_registry_certified_handles(
        certification: Arc<dyn Any + Send + Sync>,
        ascendcl: *mut c_void,
        runtime: *mut c_void,
    ) -> Result<Self, NpuLoadError> {
        Ok(Self {
            ascendcl: NonNull::new(ascendcl).ok_or(NpuLoadError::UncertifiedHandles)?,
            runtime: NonNull::new(runtime).ok_or(NpuLoadError::UncertifiedHandles)?,
            _certification: certification,
        })
    }
}

#[cfg_attr(
    not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )),
    allow(dead_code)
)]
struct NativeNpuSymbols {
    acl_create_data_buffer: AclCreateDataBuffer,
    acl_create_tensor_desc: AclCreateTensorDesc,
    acl_destroy_data_buffer: AclDestroyDataBuffer,
    acl_destroy_tensor_desc: AclDestroyTensorDesc,
    acl_finalize: AclFinalize,
    acl_get_recent_err_msg: AclGetRecentErrMsg,
    acl_init: AclInit,
    aclop_execute_v2: AclopExecuteV2,
    aclrt_create_context: AclrtCreateContext,
    aclrt_create_event: AclrtCreateEvent,
    aclrt_create_stream: AclrtCreateStream,
    aclrt_destroy_context: AclrtDestroyContext,
    aclrt_destroy_event: AclrtDestroyEvent,
    aclrt_destroy_stream: AclrtDestroyStream,
    aclrt_free: AclrtFree,
    aclrt_get_device_count: AclrtGetDeviceCount,
    aclrt_get_mem_info: AclrtGetMemInfo,
    aclrt_get_soc_name: AclrtGetSocName,
    aclrt_get_version: AclrtGetVersion,
    aclrt_malloc: AclrtMalloc,
    aclrt_memcpy: AclrtMemcpy,
    aclrt_memcpy_async: AclrtMemcpyAsync,
    aclrt_record_event: AclrtRecordEvent,
    aclrt_reset_device: AclrtResetDevice,
    aclrt_set_current_context: AclrtSetCurrentContext,
    aclrt_set_device: AclrtSetDevice,
    aclrt_synchronize_event: AclrtSynchronizeEvent,
    aclrt_synchronize_stream: AclrtSynchronizeStream,
}

impl NativeNpuSymbols {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    unsafe fn load(images: &RegistryCertifiedNpuImages) -> Result<Self, NpuLoadError> {
        macro_rules! resolve {
            ($name:literal, $type:ty) => {{
                let pointer = unsafe { required_symbol(images.ascendcl, $name)? };
                unsafe { std::mem::transmute::<*mut c_void, $type>(pointer) }
            }};
        }
        Ok(Self {
            acl_create_data_buffer: resolve!("aclCreateDataBuffer", AclCreateDataBuffer),
            acl_create_tensor_desc: resolve!("aclCreateTensorDesc", AclCreateTensorDesc),
            acl_destroy_data_buffer: resolve!("aclDestroyDataBuffer", AclDestroyDataBuffer),
            acl_destroy_tensor_desc: resolve!("aclDestroyTensorDesc", AclDestroyTensorDesc),
            acl_finalize: resolve!("aclFinalize", AclFinalize),
            acl_get_recent_err_msg: resolve!("aclGetRecentErrMsg", AclGetRecentErrMsg),
            acl_init: resolve!("aclInit", AclInit),
            aclop_execute_v2: resolve!("aclopExecuteV2", AclopExecuteV2),
            aclrt_create_context: resolve!("aclrtCreateContext", AclrtCreateContext),
            aclrt_create_event: resolve!("aclrtCreateEvent", AclrtCreateEvent),
            aclrt_create_stream: resolve!("aclrtCreateStream", AclrtCreateStream),
            aclrt_destroy_context: resolve!("aclrtDestroyContext", AclrtDestroyContext),
            aclrt_destroy_event: resolve!("aclrtDestroyEvent", AclrtDestroyEvent),
            aclrt_destroy_stream: resolve!("aclrtDestroyStream", AclrtDestroyStream),
            aclrt_free: resolve!("aclrtFree", AclrtFree),
            aclrt_get_device_count: resolve!("aclrtGetDeviceCount", AclrtGetDeviceCount),
            aclrt_get_mem_info: resolve!("aclrtGetMemInfo", AclrtGetMemInfo),
            aclrt_get_soc_name: resolve!("aclrtGetSocName", AclrtGetSocName),
            aclrt_get_version: resolve!("aclrtGetVersion", AclrtGetVersion),
            aclrt_malloc: resolve!("aclrtMalloc", AclrtMalloc),
            aclrt_memcpy: resolve!("aclrtMemcpy", AclrtMemcpy),
            aclrt_memcpy_async: resolve!("aclrtMemcpyAsync", AclrtMemcpyAsync),
            aclrt_record_event: resolve!("aclrtRecordEvent", AclrtRecordEvent),
            aclrt_reset_device: resolve!("aclrtResetDevice", AclrtResetDevice),
            aclrt_set_current_context: resolve!("aclrtSetCurrentContext", AclrtSetCurrentContext),
            aclrt_set_device: resolve!("aclrtSetDevice", AclrtSetDevice),
            aclrt_synchronize_event: resolve!("aclrtSynchronizeEvent", AclrtSynchronizeEvent),
            aclrt_synchronize_stream: resolve!("aclrtSynchronizeStream", AclrtSynchronizeStream),
        })
    }
}

struct NativeAllocation {
    pointer: NonNull<c_void>,
    bytes: usize,
}

struct NativeStream {
    pointer: NonNull<c_void>,
}

struct NativeEvent {
    pointer: NonNull<c_void>,
    stream_id: u64,
}

pub(crate) struct OwnedNpuCore {
    images: RegistryCertifiedNpuImages,
    symbols: NativeNpuSymbols,
    active: bool,
    context: Option<NonNull<c_void>>,
    probe: NativeNpuProbe,
    allocations: BTreeMap<u64, NativeAllocation>,
    streams: BTreeMap<u64, NativeStream>,
    events: BTreeMap<u64, NativeEvent>,
}

// AscendCL context selection is process-thread state. Every call is serialized by the execution
// session mutex and re-establishes the selected device and context before touching a resource.
unsafe impl Send for OwnedNpuCore {}

#[cfg_attr(
    not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )),
    allow(dead_code)
)]
impl OwnedNpuCore {
    /// Creates one owned execution core from exact handles retained by NativeFfiRegistry.
    ///
    /// # Safety
    ///
    /// The caller must be the canonical runtime certificate owner and must not initialize a
    /// second AscendCL lifecycle for these images.
    pub(crate) unsafe fn load_certified(
        images: RegistryCertifiedNpuImages,
        device_id: u32,
    ) -> Result<Self, NpuLoadError> {
        #[cfg(not(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )))]
        {
            drop(images);
            let _device_id = device_id;
            return Err(NpuLoadError::UnsupportedTarget {
                target: env!("SIM_NPU_TARGET").to_owned(),
            });
        }

        #[cfg(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        {
            let symbols = unsafe { NativeNpuSymbols::load(&images)? };
            check_acl("aclInit", unsafe { (symbols.acl_init)(std::ptr::null()) })?;
            let mut core = Self {
                images,
                symbols,
                active: true,
                context: None,
                probe: NativeNpuProbe {
                    api_version: (0, 0, 0),
                    device_count: 0,
                    device_id,
                    device_name: String::new(),
                    free_memory_bytes: 0,
                    total_memory_bytes: 0,
                },
                allocations: BTreeMap::new(),
                streams: BTreeMap::new(),
                events: BTreeMap::new(),
            };
            core.initialize_device()?;
            Ok(core)
        }
    }

    fn initialize_device(&mut self) -> Result<(), NpuLoadError> {
        self.probe.api_version = self.api_version()?;
        validate_package_version(CannVersion {
            major: self.probe.api_version.0,
            minor: self.probe.api_version.1,
            release_candidate: self.probe.api_version.2,
        })?;
        let mut device_count = 0;
        check_acl("aclrtGetDeviceCount", unsafe {
            (self.symbols.aclrt_get_device_count)(&mut device_count)
        })?;
        if device_count == 0 || self.probe.device_id >= device_count {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtGetDeviceCount",
                reason: "selected device is outside the nonzero certified device count",
            });
        }
        self.probe.device_count = device_count;
        let device_id = self.device_id_i32()?;
        check_acl("aclrtSetDevice", unsafe {
            (self.symbols.aclrt_set_device)(device_id)
        })?;
        let mut context = std::ptr::null_mut();
        check_acl("aclrtCreateContext", unsafe {
            (self.symbols.aclrt_create_context)(&mut context, device_id)
        })?;
        self.context = Some(NonNull::new(context).ok_or(NpuLoadError::AclCallFailed {
            operation: "aclrtCreateContext",
            code: ACL_SUCCESS,
        })?);
        self.select_context()?;

        let name = unsafe { (self.symbols.aclrt_get_soc_name)() };
        if name.is_null() {
            return Err(NpuLoadError::AclCallFailed {
                operation: "aclrtGetSocName",
                code: ACL_SUCCESS,
            });
        }
        let name = unsafe { CStr::from_ptr(name) }.to_str().map_err(|_| {
            NpuLoadError::InvalidArgument {
                operation: "aclrtGetSocName",
                reason: "device name is not valid UTF-8",
            }
        })?;
        if name.is_empty() || name.len() > 256 {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtGetSocName",
                reason: "device name must contain 1..=256 bytes",
            });
        }
        self.probe.device_name = name.to_owned();

        let mut free_memory = 0;
        let mut total_memory = 0;
        check_acl("aclrtGetMemInfo", unsafe {
            (self.symbols.aclrt_get_mem_info)(
                crate::abi::AclrtMemAttr::Hbm,
                &mut free_memory,
                &mut total_memory,
            )
        })?;
        if total_memory == 0 || free_memory > total_memory {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtGetMemInfo",
                reason: "memory totals must be nonzero and free must not exceed total",
            });
        }
        self.probe.free_memory_bytes = free_memory;
        self.probe.total_memory_bytes = total_memory;
        Ok(())
    }

    fn api_version(&self) -> Result<(u32, u32, u32), NpuLoadError> {
        let mut major = 0;
        let mut minor = 0;
        let mut patch = 0;
        check_acl("aclrtGetVersion", unsafe {
            (self.symbols.aclrt_get_version)(&mut major, &mut minor, &mut patch)
        })?;
        if major <= 0 || minor < 0 || patch < 0 {
            return Err(NpuLoadError::InvalidRuntimeVersion {
                major,
                minor,
                patch,
            });
        }
        Ok((major as u32, minor as u32, patch as u32))
    }

    pub(crate) fn probe(&self) -> &NativeNpuProbe {
        &self.probe
    }

    fn device_id_i32(&self) -> Result<i32, NpuLoadError> {
        i32::try_from(self.probe.device_id).map_err(|_| NpuLoadError::InvalidArgument {
            operation: "device",
            reason: "device ID exceeds the reviewed AscendCL ABI",
        })
    }

    fn select_context(&self) -> Result<(), NpuLoadError> {
        let context = self.context.ok_or(NpuLoadError::InvalidArgument {
            operation: "aclrtSetCurrentContext",
            reason: "certified context is closed",
        })?;
        check_acl("aclrtSetDevice", unsafe {
            (self.symbols.aclrt_set_device)(self.device_id_i32()?)
        })?;
        check_acl("aclrtSetCurrentContext", unsafe {
            (self.symbols.aclrt_set_current_context)(context.as_ptr())
        })
    }

    pub(crate) fn allocate(&mut self, id: u64, bytes: usize) -> Result<(), NpuLoadError> {
        self.select_context()?;
        if id == 0 || bytes == 0 || self.allocations.contains_key(&id) {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtMalloc",
                reason: "resource ID and size must be nonzero and unique",
            });
        }
        let mut pointer = std::ptr::null_mut();
        check_acl("aclrtMalloc", unsafe {
            (self.symbols.aclrt_malloc)(
                &mut pointer,
                bytes,
                crate::abi::AclrtMemMallocPolicy::HugeFirst,
            )
        })?;
        let pointer = NonNull::new(pointer).ok_or(NpuLoadError::AclCallFailed {
            operation: "aclrtMalloc",
            code: ACL_SUCCESS,
        })?;
        self.allocations
            .insert(id, NativeAllocation { pointer, bytes });
        Ok(())
    }

    pub(crate) fn release_allocation(&mut self, id: u64) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(NpuLoadError::InvalidArgument {
                operation: "aclrtFree",
                reason: "allocation is closed",
            })?;
        check_acl("aclrtFree", unsafe {
            (self.symbols.aclrt_free)(allocation.pointer.as_ptr())
        })?;
        self.allocations.remove(&id);
        Ok(())
    }

    pub(crate) fn create_stream(&mut self, id: u64) -> Result<(), NpuLoadError> {
        self.select_context()?;
        if id == 0 || self.streams.contains_key(&id) {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtCreateStream",
                reason: "stream resource ID must be nonzero and unique",
            });
        }
        let mut stream = std::ptr::null_mut();
        check_acl("aclrtCreateStream", unsafe {
            (self.symbols.aclrt_create_stream)(&mut stream)
        })?;
        let pointer = NonNull::new(stream).ok_or(NpuLoadError::AclCallFailed {
            operation: "aclrtCreateStream",
            code: ACL_SUCCESS,
        })?;
        self.streams.insert(id, NativeStream { pointer });
        Ok(())
    }

    pub(crate) fn synchronize_stream(&mut self, id: u64) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let stream = self.streams.get(&id).ok_or(NpuLoadError::InvalidArgument {
            operation: "aclrtSynchronizeStream",
            reason: "stream is closed",
        })?;
        check_acl("aclrtSynchronizeStream", unsafe {
            (self.symbols.aclrt_synchronize_stream)(stream.pointer.as_ptr())
        })
    }

    pub(crate) fn release_stream(&mut self, id: u64) -> Result<(), NpuLoadError> {
        self.synchronize_stream(id)?;
        let stream = self.streams.get(&id).ok_or(NpuLoadError::InvalidArgument {
            operation: "aclrtDestroyStream",
            reason: "stream is closed",
        })?;
        check_acl("aclrtDestroyStream", unsafe {
            (self.symbols.aclrt_destroy_stream)(stream.pointer.as_ptr())
        })?;
        self.streams.remove(&id);
        Ok(())
    }

    pub(crate) fn copy_from_host(
        &mut self,
        id: u64,
        offset: usize,
        source: &[u8],
    ) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let allocation = self.allocation(id, offset, source.len(), "aclrtMemcpy")?;
        let destination = pointer_at_offset(allocation.pointer, offset)?;
        check_acl("aclrtMemcpy", unsafe {
            (self.symbols.aclrt_memcpy)(
                destination,
                allocation.bytes - offset,
                source.as_ptr().cast(),
                source.len(),
                crate::abi::AclrtMemcpyKind::HostToDevice,
            )
        })
    }

    pub(crate) fn copy_to_host(
        &mut self,
        id: u64,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let allocation = self.allocation(id, offset, destination.len(), "aclrtMemcpy")?;
        let source = pointer_at_offset(allocation.pointer, offset)?;
        check_acl("aclrtMemcpy", unsafe {
            (self.symbols.aclrt_memcpy)(
                destination.as_mut_ptr().cast(),
                destination.len(),
                source.cast_const(),
                destination.len(),
                crate::abi::AclrtMemcpyKind::DeviceToHost,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_device_to_device(
        &mut self,
        destination_id: u64,
        destination_offset: usize,
        source_id: u64,
        source_offset: usize,
        bytes: usize,
    ) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let destination =
            self.allocation(destination_id, destination_offset, bytes, "aclrtMemcpy")?;
        let source = self.allocation(source_id, source_offset, bytes, "aclrtMemcpy")?;
        let destination_pointer = pointer_at_offset(destination.pointer, destination_offset)?;
        let source_pointer = pointer_at_offset(source.pointer, source_offset)?;
        check_acl("aclrtMemcpy", unsafe {
            (self.symbols.aclrt_memcpy)(
                destination_pointer,
                destination.bytes - destination_offset,
                source_pointer.cast_const(),
                bytes,
                crate::abi::AclrtMemcpyKind::DeviceToDevice,
            )
        })
    }

    pub(crate) fn create_event(&mut self, id: u64, stream_id: u64) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let stream = self
            .streams
            .get(&stream_id)
            .ok_or(NpuLoadError::InvalidArgument {
                operation: "aclrtRecordEvent",
                reason: "stream is closed",
            })?;
        if id == 0 || self.events.contains_key(&id) {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtCreateEvent",
                reason: "event resource ID must be nonzero and unique",
            });
        }
        let mut event = std::ptr::null_mut();
        check_acl("aclrtCreateEvent", unsafe {
            (self.symbols.aclrt_create_event)(&mut event)
        })?;
        let pointer = NonNull::new(event).ok_or(NpuLoadError::AclCallFailed {
            operation: "aclrtCreateEvent",
            code: ACL_SUCCESS,
        })?;
        if let Err(error) = check_acl("aclrtRecordEvent", unsafe {
            (self.symbols.aclrt_record_event)(pointer.as_ptr(), stream.pointer.as_ptr())
        }) {
            if let Err(cleanup_error) = check_acl("aclrtDestroyEvent", unsafe {
                (self.symbols.aclrt_destroy_event)(pointer.as_ptr())
            }) {
                eprintln!("failed to destroy unrecorded AscendCL event: {cleanup_error}");
            }
            return Err(error);
        }
        self.events.insert(id, NativeEvent { pointer, stream_id });
        Ok(())
    }

    pub(crate) fn synchronize_event(&mut self, id: u64) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let event = self.events.get(&id).ok_or(NpuLoadError::InvalidArgument {
            operation: "aclrtSynchronizeEvent",
            reason: "event is closed",
        })?;
        if !self.streams.contains_key(&event.stream_id) {
            return Err(NpuLoadError::InvalidArgument {
                operation: "aclrtSynchronizeEvent",
                reason: "event stream is closed",
            });
        }
        check_acl("aclrtSynchronizeEvent", unsafe {
            (self.symbols.aclrt_synchronize_event)(event.pointer.as_ptr())
        })
    }

    pub(crate) fn release_event(&mut self, id: u64) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let event = self.events.get(&id).ok_or(NpuLoadError::InvalidArgument {
            operation: "aclrtDestroyEvent",
            reason: "event is closed",
        })?;
        check_acl("aclrtDestroyEvent", unsafe {
            (self.symbols.aclrt_destroy_event)(event.pointer.as_ptr())
        })?;
        self.events.remove(&id);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add(
        &mut self,
        stream_id: u64,
        left_id: u64,
        right_id: u64,
        output_id: u64,
        dimensions: &[i64],
        element_type: AclDataType,
        required_bytes: usize,
    ) -> Result<(), NpuLoadError> {
        self.select_context()?;
        let stream = self
            .streams
            .get(&stream_id)
            .ok_or(NpuLoadError::InvalidArgument {
                operation: "aclopExecuteV2",
                reason: "stream is closed",
            })?;
        let left = self.allocation(left_id, 0, required_bytes, "aclopExecuteV2")?;
        let right = self.allocation(right_id, 0, required_bytes, "aclopExecuteV2")?;
        let output = self.allocation(output_id, 0, required_bytes, "aclopExecuteV2")?;
        let rank = i32::try_from(dimensions.len()).map_err(|_| NpuLoadError::InvalidArgument {
            operation: "aclCreateTensorDesc",
            reason: "tensor rank exceeds the reviewed integer ABI",
        })?;
        let left_descriptor =
            NativeTensorDescriptor::new(&self.symbols, element_type, rank, dimensions)?;
        let right_descriptor =
            NativeTensorDescriptor::new(&self.symbols, element_type, rank, dimensions)?;
        let output_descriptor =
            NativeTensorDescriptor::new(&self.symbols, element_type, rank, dimensions)?;
        let left_buffer = NativeDataBuffer::new(&self.symbols, left.pointer, required_bytes)?;
        let right_buffer = NativeDataBuffer::new(&self.symbols, right.pointer, required_bytes)?;
        let output_buffer = NativeDataBuffer::new(&self.symbols, output.pointer, required_bytes)?;
        let mut input_descriptors = [left_descriptor.pointer(), right_descriptor.pointer()];
        let mut input_buffers = [left_buffer.pointer(), right_buffer.pointer()];
        let mut output_descriptors = [output_descriptor.pointer()];
        let mut output_buffers = [output_buffer.pointer()];
        check_acl("aclopExecuteV2", unsafe {
            (self.symbols.aclop_execute_v2)(
                c"Add".as_ptr(),
                2,
                input_descriptors.as_mut_ptr(),
                input_buffers.as_mut_ptr(),
                1,
                output_descriptors.as_mut_ptr(),
                output_buffers.as_mut_ptr(),
                std::ptr::null_mut(),
                stream.pointer.as_ptr(),
            )
        })
    }

    fn allocation(
        &self,
        id: u64,
        offset: usize,
        length: usize,
        operation: &'static str,
    ) -> Result<&NativeAllocation, NpuLoadError> {
        let allocation = self
            .allocations
            .get(&id)
            .ok_or(NpuLoadError::InvalidArgument {
                operation,
                reason: "allocation is closed",
            })?;
        if length == 0
            || offset
                .checked_add(length)
                .is_none_or(|end| end > allocation.bytes)
        {
            return Err(NpuLoadError::InvalidArgument {
                operation,
                reason: "resource range is empty, overflows, or exceeds the allocation",
            });
        }
        Ok(allocation)
    }
}

impl Drop for OwnedNpuCore {
    fn drop(&mut self) {
        let event_ids = self.events.keys().copied().collect::<Vec<_>>();
        for event_id in event_ids {
            if let Err(error) = self.release_event(event_id) {
                eprintln!("failed to release owned AscendCL event: {error}");
            }
        }
        let stream_ids = self.streams.keys().copied().collect::<Vec<_>>();
        for stream_id in stream_ids {
            if let Err(error) = self.release_stream(stream_id) {
                eprintln!("failed to release owned AscendCL stream: {error}");
            }
        }
        let allocation_ids = self.allocations.keys().copied().collect::<Vec<_>>();
        for allocation_id in allocation_ids {
            if let Err(error) = self.release_allocation(allocation_id) {
                eprintln!("failed to release owned AscendCL allocation: {error}");
            }
        }
        if let Some(context) = self.context {
            if let Err(error) = check_acl("aclrtDestroyContext", unsafe {
                (self.symbols.aclrt_destroy_context)(context.as_ptr())
            }) {
                eprintln!("failed to destroy owned AscendCL context: {error}");
            } else {
                self.context = None;
            }
        }
        if self.probe.device_count != 0 {
            match self.device_id_i32() {
                Ok(device_id) => {
                    if let Err(error) = check_acl("aclrtResetDevice", unsafe {
                        (self.symbols.aclrt_reset_device)(device_id)
                    }) {
                        eprintln!("failed to reset owned AscendCL device: {error}");
                    }
                }
                Err(error) => eprintln!("failed to map owned AscendCL device: {error}"),
            }
        }
        if self.active {
            if let Err(error) = check_acl("aclFinalize", unsafe { (self.symbols.acl_finalize)() }) {
                eprintln!("failed to finalize owned AscendCL session: {error}");
            } else {
                self.active = false;
            }
        }
        let _retained_images = (self.images.ascendcl, self.images.runtime);
        let _recent_error_owner = self.symbols.acl_get_recent_err_msg as usize;
        let _async_copy_owner = self.symbols.aclrt_memcpy_async as usize;
    }
}

struct NativeTensorDescriptor<'symbols> {
    symbols: &'symbols NativeNpuSymbols,
    pointer: NonNull<c_void>,
}

impl<'symbols> NativeTensorDescriptor<'symbols> {
    fn new(
        symbols: &'symbols NativeNpuSymbols,
        element_type: AclDataType,
        rank: i32,
        dimensions: &[i64],
    ) -> Result<Self, NpuLoadError> {
        let pointer = unsafe {
            (symbols.acl_create_tensor_desc)(element_type, rank, dimensions.as_ptr(), AclFormat::Nd)
        };
        Ok(Self {
            symbols,
            pointer: NonNull::new(pointer).ok_or(NpuLoadError::AclCallFailed {
                operation: "aclCreateTensorDesc",
                code: ACL_SUCCESS,
            })?,
        })
    }

    fn pointer(&self) -> AclTensorDesc {
        self.pointer.as_ptr()
    }
}

impl Drop for NativeTensorDescriptor<'_> {
    fn drop(&mut self) {
        unsafe { (self.symbols.acl_destroy_tensor_desc)(self.pointer.as_ptr()) };
    }
}

struct NativeDataBuffer<'symbols> {
    symbols: &'symbols NativeNpuSymbols,
    pointer: NonNull<c_void>,
}

impl<'symbols> NativeDataBuffer<'symbols> {
    fn new(
        symbols: &'symbols NativeNpuSymbols,
        allocation: NonNull<c_void>,
        bytes: usize,
    ) -> Result<Self, NpuLoadError> {
        let pointer = unsafe { (symbols.acl_create_data_buffer)(allocation.as_ptr(), bytes) };
        Ok(Self {
            symbols,
            pointer: NonNull::new(pointer).ok_or(NpuLoadError::AclCallFailed {
                operation: "aclCreateDataBuffer",
                code: ACL_SUCCESS,
            })?,
        })
    }

    fn pointer(&self) -> AclDataBuffer {
        self.pointer.as_ptr()
    }
}

impl Drop for NativeDataBuffer<'_> {
    fn drop(&mut self) {
        if let Err(error) = check_acl("aclDestroyDataBuffer", unsafe {
            (self.symbols.acl_destroy_data_buffer)(self.pointer.as_ptr())
        }) {
            eprintln!("failed to destroy AscendCL data buffer: {error}");
        }
    }
}

fn pointer_at_offset(pointer: NonNull<c_void>, offset: usize) -> Result<*mut c_void, NpuLoadError> {
    let address = pointer.as_ptr().cast::<u8>().wrapping_add(offset);
    if address.is_null() {
        return Err(NpuLoadError::InvalidArgument {
            operation: "resource pointer",
            reason: "offset produced a null pointer",
        });
    }
    Ok(address.cast())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn unique_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must follow the Unix epoch")
            .as_nanos();
        env::temp_dir().join(format!("sim-npu-{label}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn discovery_order_is_exact_and_deduplicated() {
        let first = PathBuf::from("/opt/zed/ascend");
        let second = PathBuf::from("/opt/vendor/ascend");
        let package = unsafe {
            SignedPackageRoot::from_runtime_verified_path(&(), second.clone())
                .expect("absolute package path")
        };
        let candidates = discover_library_candidates_for_target(
            "x86_64-unknown-linux-gnu",
            &DiscoveryEnvironment {
                sim_ascend_root: Some(first.clone().into_os_string()),
                ascend_home_path: Some(second.into_os_string()),
            },
            &[package],
        )
        .expect("supported deterministic discovery");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].root.source, DiscoverySource::SimAscendRoot);
        assert_eq!(candidates[0].ascendcl, first.join(ASCENDCL_RELATIVE_PATH));
        assert_eq!(candidates[1].root.source, DiscoverySource::AscendHomePath);
    }

    #[test]
    fn traversal_root_is_rejected_before_filesystem_access() {
        let error = discover_library_candidates_for_target(
            "aarch64-unknown-linux-gnu",
            &DiscoveryEnvironment {
                sim_ascend_root: Some(OsString::from("/opt/ascend/../host")),
                ascend_home_path: None,
            },
            &[],
        )
        .expect_err("traversal must fail");
        assert!(matches!(error, NpuLoadError::InvalidRoot { .. }));
    }

    #[test]
    fn installed_discovery_rejects_missing_or_symlinked_library()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = unique_root("tamper");
        fs::create_dir_all(root.join("lib64"))?;
        fs::write(root.join(ASCENDCL_RELATIVE_PATH), b"fixture")?;
        let environment = DiscoveryEnvironment {
            sim_ascend_root: Some(root.clone().into_os_string()),
            ascend_home_path: None,
        };
        assert!(matches!(
            discover_installed_libraries_for_target("x86_64-unknown-linux-gnu", &environment, &[]),
            Err(NpuLoadError::MissingLibraries {
                checked_roots: 1,
                ..
            })
        ));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                root.join(ASCENDCL_RELATIVE_PATH),
                root.join(RUNTIME_RELATIVE_PATH),
            )?;
            assert!(matches!(
                discover_installed_libraries_for_target(
                    "x86_64-unknown-linux-gnu",
                    &environment,
                    &[]
                ),
                Err(NpuLoadError::MissingLibraries {
                    checked_roots: 1,
                    ..
                })
            ));
            fs::remove_file(root.join(RUNTIME_RELATIVE_PATH))?;
        }
        fs::write(root.join(RUNTIME_RELATIVE_PATH), b"fixture")?;
        let discovered =
            discover_installed_libraries_for_target("x86_64-unknown-linux-gnu", &environment, &[])?;
        assert_eq!(discovered.root.path, root);
        fs::remove_dir_all(&discovered.root.path)?;
        Ok(())
    }

    #[test]
    fn version_floor_and_certified_image_projection_fail_closed() {
        let unsupported = discover_library_candidates_for_target(
            "aarch64-apple-darwin",
            &DiscoveryEnvironment::default(),
            &[],
        )
        .expect_err("non-Linux target must fail before discovery");
        assert!(matches!(
            unsupported,
            NpuLoadError::UnsupportedTarget { .. }
        ));
        assert!(matches!(
            validate_package_version(CannVersion {
                major: 7,
                minor: 0,
                release_candidate: 9
            }),
            Err(NpuLoadError::VersionTooOld { .. })
        ));
        assert!(validate_package_version(CannVersion::FLOOR).is_ok());
        let result = unsafe {
            RegistryCertifiedNpuImages::from_registry_certified_handles(
                Arc::new(()),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("null handles must fail before session construction"),
        };
        assert_eq!(error, NpuLoadError::UncertifiedHandles);
    }
}
