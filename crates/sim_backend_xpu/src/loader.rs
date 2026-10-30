#![cfg_attr(
    not(all(
        target_arch = "x86_64",
        any(target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use crate::abi::{
    ABI_FLOOR, DnnlBinaryPrimitiveDescCreate, DnnlEngineCreate, DnnlEngineDestroy,
    DnnlEngineGetCount, DnnlExecArg, DnnlMemoryCreate, DnnlMemoryDescCreateWithStrides,
    DnnlMemoryDescDestroy, DnnlMemoryDestroy, DnnlMemoryMapData, DnnlMemoryUnmapData,
    DnnlPrimitiveCreate, DnnlPrimitiveDescDestroy, DnnlPrimitiveDestroy, DnnlPrimitiveExecute,
    DnnlStatus, DnnlStreamCreate, DnnlStreamDestroy, DnnlStreamWait, DnnlVersionFn,
    LEVEL_ZERO_MINIMUM_API_VERSION, ONEDNN_MINIMUM_MAJOR, ONEDNN_MINIMUM_MINOR, ZeApiVersion,
    ZeCommandQueueCreate, ZeCommandQueueDesc, ZeCommandQueueDestroy, ZeCommandQueueGroupProperties,
    ZeCommandQueueSynchronize, ZeContextCreate, ZeContextDesc, ZeContextDestroy, ZeDeviceGet,
    ZeDeviceGetCommandQueueGroupProperties, ZeDeviceGetMemoryProperties, ZeDeviceGetProperties,
    ZeDeviceMemoryProperties, ZeDeviceProperties, ZeDriverGet, ZeDriverGetApiVersion, ZeInit,
    ZeResult,
};
use std::{
    any::Any,
    collections::BTreeMap,
    ffi::{CStr, CString, c_void},
    path::PathBuf,
    ptr::NonNull,
    sync::Arc,
};
use thiserror::Error;

const MAXIMUM_DRIVERS: u32 = 64;
const MAXIMUM_DEVICES_PER_DRIVER: u32 = 256;
const MAXIMUM_QUEUE_GROUPS: u32 = 128;
const MAXIMUM_MEMORY_PROPERTIES: u32 = 64;
const LEVEL_ZERO_LIBRARY_ID: &str = "level_zero";
const ONEDNN_LIBRARY_ID: &str = "onednn";
const DNNL_GPU: i32 = 2;
const DNNL_STREAM_DEFAULT_FLAGS: u32 = 1;
const ZE_COMPUTE_QUEUE_FLAG: u32 = 1;
const ZE_DEVICE_TYPE_GPU: i32 = 1;
const DNNL_BINARY_ADD: i32 = 131_056;
const DNNL_F16: i32 = 1;
const DNNL_F32: i32 = 3;
const DNNL_ARG_SRC_0: i32 = 1;
const DNNL_ARG_SRC_1: i32 = 2;
const DNNL_ARG_DST: i32 = 17;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscoverySource {
    SimXpuRoot,
    OneApiRoot,
    SignedPackageRoot,
    SystemLevelZeroLoader,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryCandidate {
    source: DiscoverySource,
    level_zero: LibraryLocation,
    onednn: Option<LibraryLocation>,
}

impl DiscoveryCandidate {
    pub const fn source(&self) -> DiscoverySource {
        self.source
    }

    pub const fn level_zero(&self) -> &LibraryLocation {
        &self.level_zero
    }

    pub const fn onednn(&self) -> Option<&LibraryLocation> {
        self.onednn.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LibraryLocation {
    AbsolutePath(PathBuf),
    SystemLoaderName(&'static str),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryPlan {
    target: String,
    candidates: Vec<DiscoveryCandidate>,
}

impl DiscoveryPlan {
    pub fn from_sources(
        target: impl Into<String>,
        sim_xpu_root: Option<PathBuf>,
        oneapi_root: Option<PathBuf>,
        signed_package_roots: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, XpuLoadError> {
        let target = target.into();
        ensure_target(&target)?;
        let mut roots = Vec::new();
        push_root(&mut roots, DiscoverySource::SimXpuRoot, sim_xpu_root)?;
        push_root(&mut roots, DiscoverySource::OneApiRoot, oneapi_root)?;
        for root in signed_package_roots {
            push_root(&mut roots, DiscoverySource::SignedPackageRoot, Some(root))?;
        }
        let mut candidates = roots
            .into_iter()
            .map(|(source, root)| root_candidate(&target, source, root))
            .collect::<Vec<_>>();
        candidates.push(DiscoveryCandidate {
            source: DiscoverySource::SystemLevelZeroLoader,
            level_zero: LibraryLocation::SystemLoaderName(if target.contains("windows") {
                "ze_loader.dll"
            } else {
                "libze_loader.so.1"
            }),
            onednn: None,
        });
        Ok(Self { target, candidates })
    }

    pub fn from_environment(
        target: impl Into<String>,
        signed_package_roots: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, XpuLoadError> {
        Self::from_sources(
            target,
            std::env::var_os("SIM_XPU_ROOT").map(PathBuf::from),
            std::env::var_os("ONEAPI_ROOT").map(PathBuf::from),
            signed_package_roots,
        )
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn candidates(&self) -> &[DiscoveryCandidate] {
        &self.candidates
    }
}

fn push_root(
    roots: &mut Vec<(DiscoverySource, PathBuf)>,
    source: DiscoverySource,
    root: Option<PathBuf>,
) -> Result<(), XpuLoadError> {
    let Some(root) = root else {
        return Ok(());
    };
    if !root.is_absolute() {
        return Err(XpuLoadError::InvalidDiscoveryRoot {
            discovery_source: source,
            root: root.display().to_string(),
        });
    }
    if roots.iter().any(|(_, candidate)| candidate == &root) {
        return Ok(());
    }
    roots.push((source, root));
    Ok(())
}

fn root_candidate(target: &str, source: DiscoverySource, root: PathBuf) -> DiscoveryCandidate {
    let (level_zero, onednn) = if target.contains("windows") {
        (root.join("bin/ze_loader.dll"), root.join("bin/dnnl.dll"))
    } else {
        (
            root.join("lib/libze_loader.so.1"),
            root.join("lib/libdnnl.so.3"),
        )
    };
    DiscoveryCandidate {
        source,
        level_zero: LibraryLocation::AbsolutePath(level_zero),
        onednn: Some(LibraryLocation::AbsolutePath(onednn)),
    }
}

fn ensure_target(target: &str) -> Result<(), XpuLoadError> {
    if matches!(
        target,
        "x86_64-pc-windows-msvc" | "x86_64-unknown-linux-gnu"
    ) {
        Ok(())
    } else {
        Err(XpuLoadError::UnsupportedTarget(target.to_owned()))
    }
}

pub struct RegistryCertifiedXpuImages {
    level_zero: NonNull<c_void>,
    onednn: NonNull<c_void>,
    _certification: Arc<dyn Any + Send + Sync>,
}

impl RegistryCertifiedXpuImages {
    /// Projects the exact retained images certified by `sim_runtime::NativeFfiRegistry`.
    ///
    /// # Safety
    ///
    /// Both handles must refer to the immutable certified Level Zero and oneDNN images and must
    /// remain live through `certification`. Discovery results, SDK paths, package receipts, and
    /// compiled features do not satisfy this contract.
    pub unsafe fn from_registry_certified_images(
        certification: Arc<dyn Any + Send + Sync>,
        level_zero: *mut c_void,
        onednn: *mut c_void,
    ) -> Result<Self, XpuLoadError> {
        Ok(Self {
            level_zero: NonNull::new(level_zero).ok_or_else(|| {
                XpuLoadError::UncertifiedHandle {
                    library: LEVEL_ZERO_LIBRARY_ID.to_owned(),
                }
            })?,
            onednn: NonNull::new(onednn).ok_or_else(|| XpuLoadError::UncertifiedHandle {
                library: ONEDNN_LIBRARY_ID.to_owned(),
            })?,
            _certification: certification,
        })
    }
}

#[derive(Clone, Copy)]
struct LevelZeroSymbols {
    command_queue_create: ZeCommandQueueCreate,
    command_queue_destroy: ZeCommandQueueDestroy,
    command_queue_synchronize: ZeCommandQueueSynchronize,
    context_create: ZeContextCreate,
    context_destroy: ZeContextDestroy,
    device_get: ZeDeviceGet,
    device_get_queue_groups: ZeDeviceGetCommandQueueGroupProperties,
    device_get_memory_properties: ZeDeviceGetMemoryProperties,
    device_get_properties: ZeDeviceGetProperties,
    driver_get: ZeDriverGet,
    driver_get_api_version: ZeDriverGetApiVersion,
    initialize: ZeInit,
}

#[derive(Clone, Copy)]
struct OneDnnSymbols {
    binary_primitive_desc_create: DnnlBinaryPrimitiveDescCreate,
    engine_create: DnnlEngineCreate,
    engine_destroy: DnnlEngineDestroy,
    engine_get_count: DnnlEngineGetCount,
    memory_create: DnnlMemoryCreate,
    memory_destroy: DnnlMemoryDestroy,
    memory_desc_create_with_strides: DnnlMemoryDescCreateWithStrides,
    memory_desc_destroy: DnnlMemoryDescDestroy,
    memory_map_data: DnnlMemoryMapData,
    memory_unmap_data: DnnlMemoryUnmapData,
    primitive_create: DnnlPrimitiveCreate,
    primitive_desc_destroy: DnnlPrimitiveDescDestroy,
    primitive_destroy: DnnlPrimitiveDestroy,
    primitive_execute: DnnlPrimitiveExecute,
    stream_create: DnnlStreamCreate,
    stream_destroy: DnnlStreamDestroy,
    stream_wait: DnnlStreamWait,
    version: DnnlVersionFn,
}

impl LevelZeroSymbols {
    unsafe fn load(images: &RegistryCertifiedXpuImages) -> Result<Self, XpuLoadError> {
        let image = images.level_zero;
        Ok(Self {
            command_queue_create: unsafe { symbol(image, "zeCommandQueueCreate")? },
            command_queue_destroy: unsafe { symbol(image, "zeCommandQueueDestroy")? },
            command_queue_synchronize: unsafe { symbol(image, "zeCommandQueueSynchronize")? },
            context_create: unsafe { symbol(image, "zeContextCreate")? },
            context_destroy: unsafe { symbol(image, "zeContextDestroy")? },
            device_get: unsafe { symbol(image, "zeDeviceGet")? },
            device_get_queue_groups: unsafe {
                symbol(image, "zeDeviceGetCommandQueueGroupProperties")?
            },
            device_get_memory_properties: unsafe { symbol(image, "zeDeviceGetMemoryProperties")? },
            device_get_properties: unsafe { symbol(image, "zeDeviceGetProperties")? },
            driver_get: unsafe { symbol(image, "zeDriverGet")? },
            driver_get_api_version: unsafe { symbol(image, "zeDriverGetApiVersion")? },
            initialize: unsafe { symbol(image, "zeInit")? },
        })
    }
}

impl OneDnnSymbols {
    unsafe fn load(images: &RegistryCertifiedXpuImages) -> Result<Self, XpuLoadError> {
        let image = images.onednn;
        Ok(Self {
            binary_primitive_desc_create: unsafe {
                symbol(image, "dnnl_binary_primitive_desc_create")?
            },
            engine_create: unsafe { symbol(image, "dnnl_engine_create")? },
            engine_destroy: unsafe { symbol(image, "dnnl_engine_destroy")? },
            engine_get_count: unsafe { symbol(image, "dnnl_engine_get_count")? },
            memory_create: unsafe { symbol(image, "dnnl_memory_create")? },
            memory_destroy: unsafe { symbol(image, "dnnl_memory_destroy")? },
            memory_desc_create_with_strides: unsafe {
                symbol(image, "dnnl_memory_desc_create_with_strides")?
            },
            memory_desc_destroy: unsafe { symbol(image, "dnnl_memory_desc_destroy")? },
            memory_map_data: unsafe { symbol(image, "dnnl_memory_map_data")? },
            memory_unmap_data: unsafe { symbol(image, "dnnl_memory_unmap_data")? },
            primitive_create: unsafe { symbol(image, "dnnl_primitive_create")? },
            primitive_desc_destroy: unsafe { symbol(image, "dnnl_primitive_desc_destroy")? },
            primitive_destroy: unsafe { symbol(image, "dnnl_primitive_destroy")? },
            primitive_execute: unsafe { symbol(image, "dnnl_primitive_execute")? },
            stream_create: unsafe { symbol(image, "dnnl_stream_create")? },
            stream_destroy: unsafe { symbol(image, "dnnl_stream_destroy")? },
            stream_wait: unsafe { symbol(image, "dnnl_stream_wait")? },
            version: unsafe { symbol(image, "dnnl_version")? },
        })
    }
}

unsafe fn symbol<T: Copy>(image: NonNull<c_void>, name: &str) -> Result<T, XpuLoadError> {
    if std::mem::size_of::<T>() != std::mem::size_of::<*mut c_void>() {
        return Err(XpuLoadError::InvalidFunctionPointerLayout);
    }
    let name_string = CString::new(name).map_err(|_| XpuLoadError::InvalidSymbolName)?;
    let address = unsafe { platform_symbol(image, &name_string)? };
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&address.as_ptr()) })
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
unsafe fn platform_symbol(
    handle: NonNull<c_void>,
    name: &CStr,
) -> Result<NonNull<c_void>, XpuLoadError> {
    unsafe {
        libc::dlerror();
    }
    let address = unsafe { libc::dlsym(handle.as_ptr(), name.as_ptr()) };
    NonNull::new(address).ok_or_else(|| {
        let message = unsafe {
            let error = libc::dlerror();
            if error.is_null() {
                "symbol not found".to_owned()
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        };
        XpuLoadError::SymbolResolution {
            symbol: name.to_string_lossy().into_owned(),
            message,
        }
    })
}

#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
unsafe fn platform_symbol(
    handle: NonNull<c_void>,
    name: &CStr,
) -> Result<NonNull<c_void>, XpuLoadError> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    let address = unsafe { GetProcAddress(handle.as_ptr(), name.as_ptr().cast()) };
    NonNull::new(address).ok_or_else(|| XpuLoadError::SymbolResolution {
        symbol: name.to_string_lossy().into_owned(),
        message: "GetProcAddress returned null".to_owned(),
    })
}

#[cfg(not(all(
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "windows")
)))]
unsafe fn platform_symbol(
    _handle: NonNull<c_void>,
    name: &CStr,
) -> Result<NonNull<c_void>, XpuLoadError> {
    Err(XpuLoadError::UnsupportedHostSymbolResolution {
        symbol: name.to_string_lossy().into_owned(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XpuAbiProbe {
    pub driver_api_versions: Vec<ZeApiVersion>,
    pub device_counts: Vec<u32>,
    pub onednn_version: (i32, i32, i32),
    pub onednn_gpu_engine_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeXpuDeviceFacts {
    pub(crate) device_ordinal: usize,
    pub(crate) name: String,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) total_memory_bytes: u64,
    pub(crate) maximum_allocation_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeXpuElementType {
    F16,
    F32,
}

impl NativeXpuElementType {
    const fn dnnl(self) -> i32 {
        match self {
            Self::F16 => DNNL_F16,
            Self::F32 => DNNL_F32,
        }
    }

    const fn byte_width(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

struct NativeXpuMemory {
    descriptor: NonNull<c_void>,
    memory: NonNull<c_void>,
    dimensions: Vec<i64>,
    element_type: NativeXpuElementType,
    bytes: usize,
}

pub(crate) struct OwnedXpuCore {
    _images: RegistryCertifiedXpuImages,
    level_zero: LevelZeroSymbols,
    onednn: OneDnnSymbols,
    probe: XpuAbiProbe,
    device_facts: NativeXpuDeviceFacts,
    context: Option<NonNull<c_void>>,
    queue: Option<NonNull<c_void>>,
    engine: Option<NonNull<c_void>>,
    stream: Option<NonNull<c_void>>,
    allocations: BTreeMap<u64, NativeXpuMemory>,
}

// No pointer or vendor resource escapes this crate-private type. XpuExecutionSession serializes
// every call and teardown through one mutex before it is allowed to cross threads.
unsafe impl Send for OwnedXpuCore {}

impl OwnedXpuCore {
    /// Creates the sole owned XPU vendor session from retained registry-certified images.
    ///
    pub(crate) fn load_certified(
        images: RegistryCertifiedXpuImages,
        device_ordinal: usize,
    ) -> Result<Self, XpuLoadError> {
        let level_zero = unsafe { LevelZeroSymbols::load(&images)? };
        let onednn = unsafe { OneDnnSymbols::load(&images)? };
        let probe = unsafe { probe_symbols(level_zero, onednn)? };
        let (driver, device) = flattened_device(level_zero, device_ordinal)?;
        let device_facts = query_device_facts(level_zero, device, device_ordinal)?;
        if device_ordinal >= probe.onednn_gpu_engine_count {
            return Err(XpuLoadError::OneDnnEngineIndex {
                index: device_ordinal,
                count: probe.onednn_gpu_engine_count,
            });
        }
        let mut core = Self {
            _images: images,
            level_zero,
            onednn,
            probe,
            device_facts,
            context: None,
            queue: None,
            engine: None,
            stream: None,
            allocations: BTreeMap::new(),
        };
        core.create_native_resources(driver, device, device_ordinal)?;
        Ok(core)
    }

    fn create_native_resources(
        &mut self,
        driver: NonNull<c_void>,
        device: NonNull<c_void>,
        device_ordinal: usize,
    ) -> Result<(), XpuLoadError> {
        let descriptor = ZeContextDesc::default();
        let mut context = std::ptr::null_mut();
        check_ze("zeContextCreate", unsafe {
            (self.level_zero.context_create)(driver.as_ptr(), &descriptor, &mut context)
        })?;
        self.context = Some(NonNull::new(context).ok_or(XpuLoadError::NullResource("ze_context"))?);

        let queue_group = get_queue_groups(self.level_zero, device)?
            .into_iter()
            .enumerate()
            .find(|(_, properties)| {
                properties.flags & ZE_COMPUTE_QUEUE_FLAG != 0 && properties.queue_count > 0
            })
            .ok_or(XpuLoadError::NoComputeQueueGroup)?;
        let ordinal = u32::try_from(queue_group.0).map_err(|_| XpuLoadError::CountOverflow)?;
        let queue_descriptor = ZeCommandQueueDesc::asynchronous(ordinal, 0);
        let mut queue = std::ptr::null_mut();
        check_ze("zeCommandQueueCreate", unsafe {
            (self.level_zero.command_queue_create)(
                self.context
                    .ok_or(XpuLoadError::NullResource("ze_context"))?
                    .as_ptr(),
                device.as_ptr(),
                &queue_descriptor,
                &mut queue,
            )
        })?;
        self.queue =
            Some(NonNull::new(queue).ok_or(XpuLoadError::NullResource("ze_command_queue"))?);

        let mut engine = std::ptr::null_mut();
        check_dnnl("dnnl_engine_create", unsafe {
            (self.onednn.engine_create)(&mut engine, DNNL_GPU, device_ordinal)
        })?;
        self.engine = Some(NonNull::new(engine).ok_or(XpuLoadError::NullResource("dnnl_engine"))?);
        let mut stream = std::ptr::null_mut();
        check_dnnl("dnnl_stream_create", unsafe {
            (self.onednn.stream_create)(
                &mut stream,
                self.engine
                    .ok_or(XpuLoadError::NullResource("dnnl_engine"))?
                    .as_ptr(),
                DNNL_STREAM_DEFAULT_FLAGS,
            )
        })?;
        self.stream = Some(NonNull::new(stream).ok_or(XpuLoadError::NullResource("dnnl_stream"))?);
        Ok(())
    }

    pub(crate) fn probe(&self) -> &XpuAbiProbe {
        &self.probe
    }

    pub(crate) fn device_facts(&self) -> &NativeXpuDeviceFacts {
        &self.device_facts
    }

    pub(crate) fn allocate(
        &mut self,
        resource_id: u64,
        dimensions: &[i64],
        element_type: NativeXpuElementType,
    ) -> Result<usize, XpuLoadError> {
        if resource_id == 0 || self.allocations.contains_key(&resource_id) {
            return Err(XpuLoadError::InvalidArgument {
                operation: "dnnl_memory_create",
                reason: "resource identifier must be nonzero and unique",
            });
        }
        let (strides, bytes) = contiguous_layout(dimensions, element_type.byte_width())?;
        let rank = i32::try_from(dimensions.len()).map_err(|_| XpuLoadError::CountOverflow)?;
        let mut descriptor = std::ptr::null_mut();
        check_dnnl("dnnl_memory_desc_create_with_strides", unsafe {
            (self.onednn.memory_desc_create_with_strides)(
                &mut descriptor,
                rank,
                dimensions.as_ptr(),
                element_type.dnnl(),
                strides.as_ptr(),
            )
        })?;
        let descriptor =
            NonNull::new(descriptor).ok_or(XpuLoadError::NullResource("dnnl_memory_desc"))?;
        let mut memory = std::ptr::null_mut();
        if let Err(error) = check_dnnl("dnnl_memory_create", unsafe {
            (self.onednn.memory_create)(
                &mut memory,
                descriptor.as_ptr(),
                self.engine
                    .ok_or(XpuLoadError::NullResource("dnnl_engine"))?
                    .as_ptr(),
                usize::MAX as *mut c_void,
            )
        }) {
            report_drop_error(
                "dnnl_memory_desc_destroy",
                unsafe { (self.onednn.memory_desc_destroy)(descriptor.as_ptr()) }.0,
            );
            return Err(error);
        }
        let Some(memory) = NonNull::new(memory) else {
            report_drop_error(
                "dnnl_memory_desc_destroy",
                unsafe { (self.onednn.memory_desc_destroy)(descriptor.as_ptr()) }.0,
            );
            return Err(XpuLoadError::NullResource("dnnl_memory"));
        };
        self.allocations.insert(
            resource_id,
            NativeXpuMemory {
                descriptor,
                memory,
                dimensions: dimensions.to_vec(),
                element_type,
                bytes,
            },
        );
        Ok(bytes)
    }

    pub(crate) fn release_allocation(&mut self, resource_id: u64) -> Result<(), XpuLoadError> {
        self.synchronize()?;
        let allocation = self
            .allocations
            .get(&resource_id)
            .ok_or(XpuLoadError::ClosedResource)?;
        check_dnnl("dnnl_memory_destroy", unsafe {
            (self.onednn.memory_destroy)(allocation.memory.as_ptr())
        })?;
        check_dnnl("dnnl_memory_desc_destroy", unsafe {
            (self.onednn.memory_desc_destroy)(allocation.descriptor.as_ptr())
        })?;
        self.allocations.remove(&resource_id);
        Ok(())
    }

    pub(crate) fn copy_from_host(
        &mut self,
        resource_id: u64,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), XpuLoadError> {
        self.synchronize()?;
        let allocation = self.allocation(resource_id, offset, bytes.len())?;
        let mut mapped = std::ptr::null_mut();
        check_dnnl("dnnl_memory_map_data", unsafe {
            (self.onednn.memory_map_data)(allocation.memory.as_ptr(), &mut mapped)
        })?;
        let mapped = NonNull::new(mapped).ok_or(XpuLoadError::NullResource("mapped_memory"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        check_dnnl("dnnl_memory_unmap_data", unsafe {
            (self.onednn.memory_unmap_data)(allocation.memory.as_ptr(), mapped.as_ptr())
        })
    }

    pub(crate) fn copy_to_host(
        &mut self,
        resource_id: u64,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), XpuLoadError> {
        self.synchronize()?;
        let allocation = self.allocation(resource_id, offset, bytes.len())?;
        let mut mapped = std::ptr::null_mut();
        check_dnnl("dnnl_memory_map_data", unsafe {
            (self.onednn.memory_map_data)(allocation.memory.as_ptr(), &mut mapped)
        })?;
        let mapped = NonNull::new(mapped).ok_or(XpuLoadError::NullResource("mapped_memory"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr().cast::<u8>().add(offset),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        check_dnnl("dnnl_memory_unmap_data", unsafe {
            (self.onednn.memory_unmap_data)(allocation.memory.as_ptr(), mapped.as_ptr())
        })
    }

    pub(crate) fn add(
        &mut self,
        left_id: u64,
        right_id: u64,
        output_id: u64,
    ) -> Result<(), XpuLoadError> {
        let left = self.allocation(left_id, 0, 1)?;
        let right = self.allocation(right_id, 0, 1)?;
        let output = self.allocation(output_id, 0, 1)?;
        if left.dimensions != right.dimensions
            || left.dimensions != output.dimensions
            || left.element_type != right.element_type
            || left.element_type != output.element_type
            || left.bytes != right.bytes
            || left.bytes != output.bytes
        {
            return Err(XpuLoadError::IncompatibleAddResources);
        }
        let mut primitive_descriptor = std::ptr::null_mut();
        check_dnnl("dnnl_binary_primitive_desc_create", unsafe {
            (self.onednn.binary_primitive_desc_create)(
                &mut primitive_descriptor,
                self.engine
                    .ok_or(XpuLoadError::NullResource("dnnl_engine"))?
                    .as_ptr(),
                DNNL_BINARY_ADD,
                left.descriptor.as_ptr(),
                right.descriptor.as_ptr(),
                output.descriptor.as_ptr(),
                std::ptr::null(),
            )
        })?;
        let primitive_descriptor = NonNull::new(primitive_descriptor)
            .ok_or(XpuLoadError::NullResource("dnnl_primitive_desc"))?;
        let mut primitive = std::ptr::null_mut();
        let create = check_dnnl("dnnl_primitive_create", unsafe {
            (self.onednn.primitive_create)(&mut primitive, primitive_descriptor.as_ptr())
        });
        let descriptor_destroy =
            unsafe { (self.onednn.primitive_desc_destroy)(primitive_descriptor.as_ptr()) };
        check_dnnl("dnnl_primitive_desc_destroy", descriptor_destroy)?;
        create?;
        let primitive =
            NonNull::new(primitive).ok_or(XpuLoadError::NullResource("dnnl_primitive"))?;
        let arguments = [
            DnnlExecArg {
                argument: DNNL_ARG_SRC_0,
                memory: left.memory.as_ptr(),
            },
            DnnlExecArg {
                argument: DNNL_ARG_SRC_1,
                memory: right.memory.as_ptr(),
            },
            DnnlExecArg {
                argument: DNNL_ARG_DST,
                memory: output.memory.as_ptr(),
            },
        ];
        let execute = check_dnnl("dnnl_primitive_execute", unsafe {
            (self.onednn.primitive_execute)(
                primitive.as_ptr(),
                self.stream
                    .ok_or(XpuLoadError::NullResource("dnnl_stream"))?
                    .as_ptr(),
                3,
                arguments.as_ptr(),
            )
        });
        let destroy = check_dnnl("dnnl_primitive_destroy", unsafe {
            (self.onednn.primitive_destroy)(primitive.as_ptr())
        });
        execute?;
        destroy
    }

    pub(crate) fn synchronize(&mut self) -> Result<(), XpuLoadError> {
        check_dnnl("dnnl_stream_wait", unsafe {
            (self.onednn.stream_wait)(
                self.stream
                    .ok_or(XpuLoadError::NullResource("dnnl_stream"))?
                    .as_ptr(),
            )
        })?;
        check_ze("zeCommandQueueSynchronize", unsafe {
            (self.level_zero.command_queue_synchronize)(
                self.queue
                    .ok_or(XpuLoadError::NullResource("ze_command_queue"))?
                    .as_ptr(),
                u64::MAX,
            )
        })
    }

    fn allocation(
        &self,
        resource_id: u64,
        offset: usize,
        length: usize,
    ) -> Result<&NativeXpuMemory, XpuLoadError> {
        let allocation = self
            .allocations
            .get(&resource_id)
            .ok_or(XpuLoadError::ClosedResource)?;
        if length == 0
            || offset
                .checked_add(length)
                .is_none_or(|end| end > allocation.bytes)
        {
            return Err(XpuLoadError::ResourceBounds {
                offset,
                length,
                available: allocation.bytes,
            });
        }
        Ok(allocation)
    }
}

impl Drop for OwnedXpuCore {
    fn drop(&mut self) {
        if self.stream.is_some() && self.queue.is_some() {
            if let Err(error) = self.synchronize() {
                eprintln!("sim_backend_xpu: synchronization failed during teardown: {error}");
            }
        }
        let allocation_ids = self.allocations.keys().copied().collect::<Vec<_>>();
        for resource_id in allocation_ids {
            if let Some(allocation) = self.allocations.remove(&resource_id) {
                report_drop_error(
                    "dnnl_memory_destroy",
                    unsafe { (self.onednn.memory_destroy)(allocation.memory.as_ptr()) }.0,
                );
                report_drop_error(
                    "dnnl_memory_desc_destroy",
                    unsafe { (self.onednn.memory_desc_destroy)(allocation.descriptor.as_ptr()) }.0,
                );
            }
        }
        if let Some(stream) = self.stream.take() {
            report_drop_error(
                "dnnl_stream_destroy",
                unsafe { (self.onednn.stream_destroy)(stream.as_ptr()) }.0,
            );
        }
        if let Some(engine) = self.engine.take() {
            report_drop_error(
                "dnnl_engine_destroy",
                unsafe { (self.onednn.engine_destroy)(engine.as_ptr()) }.0,
            );
        }
        if let Some(queue) = self.queue.take() {
            report_drop_error(
                "zeCommandQueueDestroy",
                unsafe { (self.level_zero.command_queue_destroy)(queue.as_ptr()) }.0,
            );
        }
        if let Some(context) = self.context.take() {
            report_drop_error(
                "zeContextDestroy",
                unsafe { (self.level_zero.context_destroy)(context.as_ptr()) }.0,
            );
        }
    }
}

fn contiguous_layout(
    dimensions: &[i64],
    byte_width: usize,
) -> Result<(Vec<i64>, usize), XpuLoadError> {
    if dimensions.is_empty() || dimensions.len() > 12 || dimensions.iter().any(|value| *value <= 0)
    {
        return Err(XpuLoadError::InvalidArgument {
            operation: "dnnl_memory_desc_create_with_strides",
            reason: "dimensions must have rank 1..=12 and be positive",
        });
    }
    let mut strides = vec![0; dimensions.len()];
    let mut elements = 1usize;
    for index in (0..dimensions.len()).rev() {
        strides[index] = i64::try_from(elements).map_err(|_| XpuLoadError::CountOverflow)?;
        let dimension =
            usize::try_from(dimensions[index]).map_err(|_| XpuLoadError::CountOverflow)?;
        elements = elements
            .checked_mul(dimension)
            .ok_or(XpuLoadError::CountOverflow)?;
    }
    let bytes = elements
        .checked_mul(byte_width)
        .ok_or(XpuLoadError::CountOverflow)?;
    Ok((strides, bytes))
}

fn flattened_device(
    symbols: LevelZeroSymbols,
    requested_ordinal: usize,
) -> Result<(NonNull<c_void>, NonNull<c_void>), XpuLoadError> {
    let drivers = get_drivers(symbols)?;
    let mut ordinal = 0usize;
    for driver in drivers {
        for device in get_devices(symbols, driver)? {
            if ordinal == requested_ordinal {
                return Ok((driver, device));
            }
            ordinal = ordinal.checked_add(1).ok_or(XpuLoadError::CountOverflow)?;
        }
    }
    Err(XpuLoadError::DeviceOrdinal {
        requested: requested_ordinal,
        count: ordinal,
    })
}

fn query_device_facts(
    symbols: LevelZeroSymbols,
    device: NonNull<c_void>,
    device_ordinal: usize,
) -> Result<NativeXpuDeviceFacts, XpuLoadError> {
    let mut properties = ZeDeviceProperties::default();
    check_ze("zeDeviceGetProperties", unsafe {
        (symbols.device_get_properties)(device.as_ptr(), &mut properties)
    })?;
    if properties.device_type != ZE_DEVICE_TYPE_GPU {
        return Err(XpuLoadError::NonGpuDevice {
            ordinal: device_ordinal,
            device_type: properties.device_type,
        });
    }
    if properties.maximum_memory_allocation_size == 0 {
        return Err(XpuLoadError::InvalidDeviceProperties(
            "maximum allocation size is zero",
        ));
    }
    let name_end = properties
        .name
        .iter()
        .position(|character| *character == 0)
        .ok_or(XpuLoadError::InvalidDeviceProperties(
            "device name is not null terminated",
        ))?;
    let name_bytes = properties.name[..name_end]
        .iter()
        .map(|character| *character as u8)
        .collect::<Vec<_>>();
    let name = std::str::from_utf8(&name_bytes)
        .map_err(|_| XpuLoadError::InvalidDeviceProperties("device name is not UTF-8"))?
        .trim();
    if name.is_empty() {
        return Err(XpuLoadError::InvalidDeviceProperties(
            "device name is empty",
        ));
    }
    let memory_properties = get_memory_properties(symbols, device)?;
    let total_memory_bytes = memory_properties.iter().try_fold(0u64, |total, memory| {
        total
            .checked_add(memory.total_size)
            .ok_or(XpuLoadError::CountOverflow)
    })?;
    if total_memory_bytes == 0 {
        return Err(XpuLoadError::InvalidDeviceProperties(
            "total device memory is zero",
        ));
    }
    Ok(NativeXpuDeviceFacts {
        device_ordinal,
        name: name.to_owned(),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        total_memory_bytes,
        maximum_allocation_bytes: properties.maximum_memory_allocation_size,
    })
}

fn get_memory_properties(
    symbols: LevelZeroSymbols,
    device: NonNull<c_void>,
) -> Result<Vec<ZeDeviceMemoryProperties>, XpuLoadError> {
    let mut count = 0;
    check_ze("zeDeviceGetMemoryProperties", unsafe {
        (symbols.device_get_memory_properties)(device.as_ptr(), &mut count, std::ptr::null_mut())
    })?;
    if count > MAXIMUM_MEMORY_PROPERTIES {
        return Err(XpuLoadError::UnboundedVendorCount {
            resource: "Level Zero memory properties",
            count,
            maximum: MAXIMUM_MEMORY_PROPERTIES,
        });
    }
    let length = usize::try_from(count).map_err(|_| XpuLoadError::CountOverflow)?;
    let mut properties = vec![ZeDeviceMemoryProperties::default(); length];
    if count > 0 {
        check_ze("zeDeviceGetMemoryProperties", unsafe {
            (symbols.device_get_memory_properties)(
                device.as_ptr(),
                &mut count,
                properties.as_mut_ptr(),
            )
        })?;
    }
    let final_length = usize::try_from(count).map_err(|_| XpuLoadError::CountOverflow)?;
    if final_length > properties.len() {
        return Err(XpuLoadError::VendorCountIncreased {
            resource: "Level Zero memory properties",
        });
    }
    properties.truncate(final_length);
    Ok(properties)
}

unsafe fn probe_symbols(
    level_zero: LevelZeroSymbols,
    onednn: OneDnnSymbols,
) -> Result<XpuAbiProbe, XpuLoadError> {
    check_ze("zeInit", unsafe { (level_zero.initialize)(0) })?;
    let drivers = get_drivers(level_zero)?;
    if drivers.is_empty() {
        return Err(XpuLoadError::NoLevelZeroDriver);
    }
    let mut driver_api_versions = Vec::with_capacity(drivers.len());
    let mut device_counts = Vec::with_capacity(drivers.len());
    for driver in drivers {
        let mut version = ZeApiVersion(0);
        check_ze("zeDriverGetApiVersion", unsafe {
            (level_zero.driver_get_api_version)(driver.as_ptr(), &mut version)
        })?;
        if version < LEVEL_ZERO_MINIMUM_API_VERSION {
            return Err(XpuLoadError::LevelZeroVersion {
                major: version.major(),
                minor: version.minor(),
            });
        }
        driver_api_versions.push(version);
        let devices = get_devices(level_zero, driver)?;
        device_counts.push(u32::try_from(devices.len()).map_err(|_| XpuLoadError::CountOverflow)?);
    }
    if device_counts.iter().all(|count| *count == 0) {
        return Err(XpuLoadError::NoLevelZeroDevice);
    }

    let version = NonNull::new(unsafe { (onednn.version)() }.cast_mut())
        .ok_or(XpuLoadError::NullResource("dnnl_version"))?;
    let version = unsafe { version.as_ref() };
    validate_onednn_version(version)?;
    let engine_count = unsafe { (onednn.engine_get_count)(DNNL_GPU) };
    if engine_count == 0 {
        return Err(XpuLoadError::NoOneDnnGpuEngine);
    }
    Ok(XpuAbiProbe {
        driver_api_versions,
        device_counts,
        onednn_version: (version.major, version.minor, version.patch),
        onednn_gpu_engine_count: engine_count,
    })
}

fn validate_onednn_version(version: &crate::abi::DnnlVersion) -> Result<(), XpuLoadError> {
    if version.major != ONEDNN_MINIMUM_MAJOR || version.minor < ONEDNN_MINIMUM_MINOR {
        Err(XpuLoadError::OneDnnVersion {
            major: version.major,
            minor: version.minor,
            patch: version.patch,
        })
    } else {
        Ok(())
    }
}

fn get_drivers(symbols: LevelZeroSymbols) -> Result<Vec<NonNull<c_void>>, XpuLoadError> {
    let mut count = 0;
    check_ze("zeDriverGet", unsafe {
        (symbols.driver_get)(&mut count, std::ptr::null_mut())
    })?;
    bounded_handles(
        "Level Zero drivers",
        count,
        MAXIMUM_DRIVERS,
        |count, output| unsafe { (symbols.driver_get)(count, output) },
    )
}

fn get_devices(
    symbols: LevelZeroSymbols,
    driver: NonNull<c_void>,
) -> Result<Vec<NonNull<c_void>>, XpuLoadError> {
    let mut count = 0;
    check_ze("zeDeviceGet", unsafe {
        (symbols.device_get)(driver.as_ptr(), &mut count, std::ptr::null_mut())
    })?;
    bounded_handles(
        "Level Zero devices",
        count,
        MAXIMUM_DEVICES_PER_DRIVER,
        |count, output| unsafe { (symbols.device_get)(driver.as_ptr(), count, output) },
    )
}

fn bounded_handles(
    resource: &'static str,
    mut count: u32,
    maximum: u32,
    fill: impl FnOnce(&mut u32, *mut *mut c_void) -> ZeResult,
) -> Result<Vec<NonNull<c_void>>, XpuLoadError> {
    if count > maximum {
        return Err(XpuLoadError::UnboundedVendorCount {
            resource,
            count,
            maximum,
        });
    }
    let capacity = usize::try_from(count).map_err(|_| XpuLoadError::CountOverflow)?;
    let mut handles = vec![std::ptr::null_mut(); capacity];
    if count > 0 {
        check_ze(resource, fill(&mut count, handles.as_mut_ptr()))?;
    }
    if count > maximum {
        return Err(XpuLoadError::UnboundedVendorCount {
            resource,
            count,
            maximum,
        });
    }
    let final_length = usize::try_from(count).map_err(|_| XpuLoadError::CountOverflow)?;
    if final_length > handles.len() {
        return Err(XpuLoadError::VendorCountIncreased { resource });
    }
    handles.truncate(final_length);
    handles
        .into_iter()
        .map(|handle| NonNull::new(handle).ok_or(XpuLoadError::NullResource(resource)))
        .collect()
}

fn get_queue_groups(
    symbols: LevelZeroSymbols,
    device: NonNull<c_void>,
) -> Result<Vec<ZeCommandQueueGroupProperties>, XpuLoadError> {
    let mut count = 0;
    check_ze("zeDeviceGetCommandQueueGroupProperties", unsafe {
        (symbols.device_get_queue_groups)(device.as_ptr(), &mut count, std::ptr::null_mut())
    })?;
    if count > MAXIMUM_QUEUE_GROUPS {
        return Err(XpuLoadError::UnboundedVendorCount {
            resource: "Level Zero queue groups",
            count,
            maximum: MAXIMUM_QUEUE_GROUPS,
        });
    }
    let length = usize::try_from(count).map_err(|_| XpuLoadError::CountOverflow)?;
    let mut properties = vec![ZeCommandQueueGroupProperties::default(); length];
    if count > 0 {
        check_ze("zeDeviceGetCommandQueueGroupProperties", unsafe {
            (symbols.device_get_queue_groups)(device.as_ptr(), &mut count, properties.as_mut_ptr())
        })?;
    }
    let final_length = usize::try_from(count).map_err(|_| XpuLoadError::CountOverflow)?;
    if final_length > properties.len() {
        return Err(XpuLoadError::VendorCountIncreased {
            resource: "Level Zero queue groups",
        });
    }
    properties.truncate(final_length);
    Ok(properties)
}

fn check_ze(symbol: &'static str, status: ZeResult) -> Result<(), XpuLoadError> {
    if status == ZeResult::SUCCESS {
        Ok(())
    } else {
        Err(XpuLoadError::VendorCall {
            library: LEVEL_ZERO_LIBRARY_ID,
            symbol,
            status: status.0,
        })
    }
}

fn check_dnnl(symbol: &'static str, status: DnnlStatus) -> Result<(), XpuLoadError> {
    if status == DnnlStatus::SUCCESS {
        Ok(())
    } else {
        Err(XpuLoadError::VendorCall {
            library: ONEDNN_LIBRARY_ID,
            symbol,
            status: status.0,
        })
    }
}

fn report_drop_error(symbol: &str, status: i32) {
    if status != 0 {
        eprintln!("sim_backend_xpu: {symbol} failed during resource release: status {status}");
    }
}

pub fn unavailable_reason() -> String {
    let target = env!("SIM_XPU_TARGET");
    if !matches!(
        target,
        "x86_64-pc-windows-msvc" | "x86_64-unknown-linux-gnu"
    ) {
        return format!(
            "XPU ABI is unavailable on target {target}; supported targets are x86_64-unknown-linux-gnu and x86_64-pc-windows-msvc"
        );
    }
    format!(
        "XPU ABI {ABI_FLOOR} is unbound: discovery and package receipts are not authorization; sim_runtime::NativeFfiRegistry must certify retained Level Zero and oneDNN images before native tensor integration"
    )
}

#[derive(Debug, Error)]
pub enum XpuLoadError {
    #[error("XPU ABI target is unsupported: {0}")]
    UnsupportedTarget(String),
    #[error("XPU discovery root from {discovery_source:?} must be absolute: {root}")]
    InvalidDiscoveryRoot {
        discovery_source: DiscoverySource,
        root: String,
    },
    #[error("XPU retained module handle is null for {library}")]
    UncertifiedHandle { library: String },
    #[error("XPU device ordinal {requested} is outside the flattened device count {count}")]
    DeviceOrdinal { requested: usize, count: usize },
    #[error("XPU symbol name contains an interior null byte")]
    InvalidSymbolName,
    #[error("XPU function pointer does not match the platform pointer layout")]
    InvalidFunctionPointerLayout,
    #[error("XPU symbol resolution failed for {symbol}: {message}")]
    SymbolResolution { symbol: String, message: String },
    #[error("XPU symbol resolution for {symbol} is unavailable on this host")]
    UnsupportedHostSymbolResolution { symbol: String },
    #[error("XPU vendor call {library}::{symbol} failed with status {status}")]
    VendorCall {
        library: &'static str,
        symbol: &'static str,
        status: i32,
    },
    #[error("Level Zero driver API {major}.{minor} is below reviewed API 1.6")]
    LevelZeroVersion { major: u16, minor: u16 },
    #[error("oneDNN runtime {major}.{minor}.{patch} is outside reviewed 3.5.x ABI")]
    OneDnnVersion { major: i32, minor: i32, patch: i32 },
    #[error("no Level Zero driver is available")]
    NoLevelZeroDriver,
    #[error("no Level Zero device is available")]
    NoLevelZeroDevice,
    #[error("no oneDNN GPU engine is available")]
    NoOneDnnGpuEngine,
    #[error("the selected XPU device is not a GPU: ordinal {ordinal}, type {device_type}")]
    NonGpuDevice { ordinal: usize, device_type: i32 },
    #[error("XPU device properties are invalid: {0}")]
    InvalidDeviceProperties(&'static str),
    #[error("the selected XPU device has no compute command queue group")]
    NoComputeQueueGroup,
    #[error("oneDNN GPU engine index {index} is outside engine count {count}")]
    OneDnnEngineIndex { index: usize, count: usize },
    #[error("XPU vendor returned null resource {0}")]
    NullResource(&'static str),
    #[error("XPU {operation} argument is invalid: {reason}")]
    InvalidArgument {
        operation: &'static str,
        reason: &'static str,
    },
    #[error("XPU resource has already been released")]
    ClosedResource,
    #[error("XPU resource range {offset}..+{length} exceeds {available} bytes")]
    ResourceBounds {
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("XPU Add requires matching dimensions, element types, and byte lengths")]
    IncompatibleAddResources,
    #[error("XPU vendor {resource} count {count} exceeds bound {maximum}")]
    UnboundedVendorCount {
        resource: &'static str,
        count: u32,
        maximum: u32,
    },
    #[error("XPU vendor increased {resource} count between bounded queries")]
    VendorCountIncreased { resource: &'static str },
    #[error("XPU vendor count cannot be represented on this host")]
    CountOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn discovery_is_ordered_but_never_authoritative() -> Result<(), XpuLoadError> {
        let plan = DiscoveryPlan::from_sources(
            "x86_64-unknown-linux-gnu",
            Some(PathBuf::from("/sim")),
            Some(PathBuf::from("/oneapi")),
            [PathBuf::from("/signed")],
        )?;
        assert_eq!(plan.candidates.len(), 4);
        assert_eq!(plan.candidates[0].source, DiscoverySource::SimXpuRoot);
        assert_eq!(plan.candidates[1].source, DiscoverySource::OneApiRoot);
        assert_eq!(
            plan.candidates[2].source,
            DiscoverySource::SignedPackageRoot
        );
        assert_eq!(
            plan.candidates[3].source,
            DiscoverySource::SystemLevelZeroLoader
        );
        assert!(plan.candidates[3].onednn.is_none());
        Ok(())
    }

    #[test]
    fn discovery_rejects_relative_and_unsupported_inputs() {
        assert!(matches!(
            DiscoveryPlan::from_sources(
                "x86_64-unknown-linux-gnu",
                Some(PathBuf::from("relative")),
                None,
                [],
            ),
            Err(XpuLoadError::InvalidDiscoveryRoot { .. })
        ));
        assert!(matches!(
            DiscoveryPlan::from_sources("aarch64-apple-darwin", None, None, []),
            Err(XpuLoadError::UnsupportedTarget(_))
        ));
    }

    #[test]
    fn certified_images_are_owned_and_reject_null_handles() -> Result<(), XpuLoadError> {
        let certification = Arc::new(());
        let retained = certification.clone();
        let pointer = NonNull::<u8>::dangling().as_ptr().cast::<c_void>();
        let images = unsafe {
            RegistryCertifiedXpuImages::from_registry_certified_images(retained, pointer, pointer)?
        };
        assert_eq!(Arc::strong_count(&certification), 2);
        drop(images);
        assert_eq!(Arc::strong_count(&certification), 1);

        let error = unsafe {
            RegistryCertifiedXpuImages::from_registry_certified_images(
                Arc::new(()),
                std::ptr::null_mut(),
                pointer,
            )
        }
        .err()
        .ok_or(XpuLoadError::InvalidDeviceProperties(
            "null certified image unexpectedly succeeded",
        ))?;
        assert!(matches!(error, XpuLoadError::UncertifiedHandle { .. }));
        Ok(())
    }

    #[test]
    fn retained_images_have_one_opaque_owned_resource_graph() {
        let source = include_str!("loader.rs");
        assert!(!source.contains(&["PhantomData<&'", "certificate"].concat()));
        assert!(!source.contains(&["pub struct LevelZero", "Context"].concat()));
        assert!(!source.contains(&["pub struct OneDnn", "Stream"].concat()));
        assert!(source.contains("pub(crate) struct OwnedXpuCore"));
        assert!(source.contains("_images: RegistryCertifiedXpuImages"));
        let bound_constructor = ["NativeBackendBindingStatus::", "bound"].concat();
        assert!(!source.contains(&bound_constructor));
        let path_loader = ["Load", "Library"].concat();
        assert!(!source.contains(&path_loader));
    }

    #[test]
    fn contiguous_layout_is_checked_and_row_major() -> Result<(), XpuLoadError> {
        assert_eq!(contiguous_layout(&[2, 3, 4], 4)?, (vec![12, 4, 1], 96));
        assert!(matches!(
            contiguous_layout(&[], 4),
            Err(XpuLoadError::InvalidArgument { .. })
        ));
        assert!(matches!(
            contiguous_layout(&[i64::MAX, i64::MAX], 4),
            Err(XpuLoadError::CountOverflow)
        ));
        Ok(())
    }

    #[test]
    fn unavailable_state_names_the_canonical_registry() {
        let reason = unavailable_reason();
        assert!(reason.contains("NativeFfiRegistry") || reason.contains("supported targets"));
    }

    #[test]
    fn onednn_floor_accepts_newer_minor_versions_within_major_three() {
        let version = crate::abi::DnnlVersion {
            major: 3,
            minor: 6,
            patch: 0,
            hash: std::ptr::null(),
            cpu_runtime: 0,
            gpu_runtime: 0,
        };
        assert!(validate_onednn_version(&version).is_ok());

        let incompatible = crate::abi::DnnlVersion {
            major: 4,
            ..version
        };
        assert!(matches!(
            validate_onednn_version(&incompatible),
            Err(XpuLoadError::OneDnnVersion { .. })
        ));
    }

    #[test]
    fn ordinary_paths_remain_discovery_observations() {
        let candidate = root_candidate(
            "x86_64-pc-windows-msvc",
            DiscoverySource::SignedPackageRoot,
            PathBuf::from("C:/signed"),
        );
        assert!(matches!(
            candidate.level_zero(),
            LibraryLocation::AbsolutePath(path) if path.ends_with(Path::new("bin/ze_loader.dll"))
        ));
    }
}
