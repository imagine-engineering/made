use crate::abi::{
    ABI_FLOOR, AbiManifest, CnnlCreate, CnnlCreateOpTensorDescriptor, CnnlCreateTensorDescriptor,
    CnnlDataType, CnnlDestroy, CnnlDestroyOpTensorDescriptor, CnnlDestroyTensorDescriptor,
    CnnlGetLibVersion, CnnlNanPropagation, CnnlOpTensor, CnnlOpTensorDescription,
    CnnlSetOpTensorDescriptor, CnnlSetQueue, CnnlSetTensorDescriptor, CnnlStatus, CnnlTensorLayout,
    CnrtFree, CnrtGetDeviceCount, CnrtGetLibVersion, CnrtMalloc, CnrtMemTransferDirection,
    CnrtMemcpy, CnrtQueueCreate, CnrtQueueDestroy, CnrtQueueSync, CnrtSetDevice, CnrtStatus,
    LibraryContract, UNSAFE_OWNER,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    marker::PhantomData,
    path::{Path, PathBuf},
    ptr::NonNull,
};
use thiserror::Error;

#[cfg(test)]
const LIBRARY_LOAD_ORDER: [&str; 2] = ["cnrt", "cnnl"];
const REVIEWED_SYMBOL_COUNT: usize = 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryPlan {
    roots: Vec<DiscoveryRoot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DiscoveryRoot {
    source: &'static str,
    root: PathBuf,
}

impl DiscoveryPlan {
    pub fn from_sources(
        sim_mlu_root: Option<PathBuf>,
        neuware_home: Option<PathBuf>,
        signed_package_roots: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, MluLoadError> {
        let mut roots = Vec::new();
        if let Some(root) = sim_mlu_root {
            push_root(&mut roots, "SIM_MLU_ROOT", root)?;
        }
        if let Some(root) = neuware_home {
            push_root(&mut roots, "NEUWARE_HOME", root)?;
        }
        for root in signed_package_roots {
            push_root(&mut roots, "signed_package_roots", root)?;
        }
        Ok(Self { roots })
    }

    pub fn from_environment(
        signed_package_roots: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, MluLoadError> {
        Self::from_sources(
            std::env::var_os("SIM_MLU_ROOT").map(PathBuf::from),
            std::env::var_os("NEUWARE_HOME").map(PathBuf::from),
            signed_package_roots,
        )
    }

    pub fn candidates(&self) -> Vec<(&'static str, PathBuf, PathBuf)> {
        self.roots
            .iter()
            .map(|root| {
                (
                    root.source,
                    root.root.join("lib64/libcnrt.so"),
                    root.root.join("lib64/libcnnl.so"),
                )
            })
            .collect()
    }
}

fn push_root(
    roots: &mut Vec<DiscoveryRoot>,
    source: &'static str,
    root: PathBuf,
) -> Result<(), MluLoadError> {
    if !root.is_absolute() {
        return Err(MluLoadError::InvalidDiscoveryRoot {
            source_name: source,
            root: root.display().to_string(),
        });
    }
    if roots.iter().any(|candidate| candidate.root == root) {
        return Ok(());
    }
    roots.push(DiscoveryRoot { source, root });
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
pub struct RegistryCertifiedImage {
    pub library_id: String,
    pub digest_sha256: String,
    pub abi_version: String,
    pub required_symbols: BTreeSet<String>,
    pub unsafe_owner: String,
    pub retained_image_path: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CertifiedMluImages<'certificate> {
    images: BTreeMap<String, RegistryCertifiedImage>,
    certificate_lifetime: PhantomData<&'certificate ()>,
}

impl<'certificate> CertifiedMluImages<'certificate> {
    /// # Safety
    ///
    /// Every row must be a direct projection of a live
    /// `sim_runtime::NativeFfiRegistry` certificate, and each retained path must name the
    /// immutable image whose digest the certificate covers. Discovery results, package metadata,
    /// or feature compilation do not satisfy this contract.
    pub(crate) unsafe fn from_registry_certificates<W: ?Sized>(
        _certificate_session: &'certificate W,
        images: impl IntoIterator<Item = RegistryCertifiedImage>,
    ) -> Result<Self, MluLoadError> {
        let manifest =
            AbiManifest::embedded().map_err(|error| MluLoadError::Manifest(error.to_string()))?;
        let mut checked = BTreeMap::new();
        for image in images {
            let library = manifest
                .libraries
                .iter()
                .find(|library| library.id == image.library_id)
                .ok_or_else(|| MluLoadError::UnexpectedCertifiedLibrary {
                    library: image.library_id.clone(),
                })?;
            validate_certificate_projection(&image, library)?;
            let library_id = image.library_id.clone();
            if checked.insert(library_id.clone(), image).is_some() {
                return Err(MluLoadError::DuplicateCertifiedLibrary {
                    library: library_id,
                });
            }
        }
        for library in &manifest.libraries {
            if !checked.contains_key(&library.id) {
                return Err(MluLoadError::MissingCertifiedLibrary {
                    library: library.id.clone(),
                });
            }
        }
        Ok(Self {
            images: checked,
            certificate_lifetime: PhantomData,
        })
    }

    #[cfg(any(
        test,
        all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    ))]
    fn image(&self, library: &str) -> Result<&RegistryCertifiedImage, MluLoadError> {
        self.images
            .get(library)
            .ok_or_else(|| MluLoadError::MissingCertifiedLibrary {
                library: library.to_owned(),
            })
    }
}

fn validate_certificate_projection(
    image: &RegistryCertifiedImage,
    library: &LibraryContract,
) -> Result<(), MluLoadError> {
    if image.abi_version != ABI_FLOOR || image.unsafe_owner != UNSAFE_OWNER {
        return Err(MluLoadError::CertificateMismatch {
            library: image.library_id.clone(),
        });
    }
    if !is_sha256(&image.digest_sha256) {
        return Err(MluLoadError::InvalidCertificateDigest {
            library: image.library_id.clone(),
        });
    }
    if !is_sealed_fd_path(&image.retained_image_path) {
        return Err(MluLoadError::UnsealedImagePath {
            library: image.library_id.clone(),
            path: image.retained_image_path.display().to_string(),
        });
    }
    let expected_symbols = library
        .symbols
        .iter()
        .map(|symbol| symbol.name.clone())
        .collect::<BTreeSet<_>>();
    if image.required_symbols != expected_symbols {
        if let Some(symbol) = expected_symbols.difference(&image.required_symbols).next() {
            return Err(MluLoadError::CertificateMissingSymbol {
                library: image.library_id.clone(),
                symbol: symbol.clone(),
            });
        }
        return Err(MluLoadError::CertificateMismatch {
            library: image.library_id.clone(),
        });
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_sealed_fd_path(path: &Path) -> bool {
    let Some(value) = path.to_str() else {
        return false;
    };
    let Some(descriptor) = value.strip_prefix("/proc/self/fd/") else {
        return false;
    };
    !descriptor.is_empty() && descriptor.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(target_os = "linux")]
fn verify_immutable_sealed_fd(path: &Path) -> bool {
    let Some(value) = path.to_str() else {
        return false;
    };
    let Some(descriptor) = value
        .strip_prefix("/proc/self/fd/")
        .and_then(|value| value.parse::<i32>().ok())
    else {
        return false;
    };
    let required = libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
    let actual = unsafe { libc::fcntl(descriptor, libc::F_GET_SEALS) };
    actual >= 0 && actual & required == required
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct LibraryVersion {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MluAbiProbe {
    pub target: String,
    pub abi_floor: String,
    pub cnrt_version: LibraryVersion,
    pub cnnl_version: LibraryVersion,
    pub symbol_count: usize,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum MluLoadError {
    #[error("Cambricon MLU target is unsupported: {target}")]
    UnsupportedTarget { target: String },
    #[error("{source_name} must contain an absolute MLU SDK root, got {root}")]
    InvalidDiscoveryRoot {
        source_name: &'static str,
        root: String,
    },
    #[error("MLU ABI manifest is invalid: {0}")]
    Manifest(String),
    #[error("certified MLU library {library} is missing")]
    MissingCertifiedLibrary { library: String },
    #[error("certified MLU library {library} is duplicated")]
    DuplicateCertifiedLibrary { library: String },
    #[error("certified library {library} is not part of the reviewed MLU ABI")]
    UnexpectedCertifiedLibrary { library: String },
    #[error("certificate identity, ABI, or unsafe owner differs for {library}")]
    CertificateMismatch { library: String },
    #[error("certificate digest is not lowercase SHA-256 for {library}")]
    InvalidCertificateDigest { library: String },
    #[error("certificate for {library} omits required symbol {symbol}")]
    CertificateMissingSymbol { library: String, symbol: String },
    #[error(
        "certified library {library} uses ordinary path {path}; a retained sealed /proc/self/fd image is required"
    )]
    UnsealedImagePath { library: String, path: String },
    #[error("failed to open certified MLU image {library}: {reason}")]
    LibraryOpen { library: String, reason: String },
    #[error("required MLU symbol {symbol} is missing from {library}")]
    MissingSymbol { library: String, symbol: String },
    #[error("MLU version call failed in {library} with status {status}")]
    VersionCall { library: String, status: i32 },
    #[error("MLU library {library} version {actual} is below or incompatible with {required}")]
    Version {
        library: String,
        required: String,
        actual: String,
    },
    #[error("MLU operation {operation} failed with status {status}")]
    CallFailed {
        operation: &'static str,
        status: i32,
    },
    #[error("MLU operation {operation} returned a null resource")]
    NullResource { operation: &'static str },
    #[error("invalid MLU {operation} argument: {reason}")]
    InvalidArgument {
        operation: &'static str,
        reason: &'static str,
    },
}

#[cfg(test)]
trait ProbeSystem {
    type Handle;

    fn open(
        &mut self,
        library: &LibraryContract,
        image: &RegistryCertifiedImage,
    ) -> Result<Self::Handle, MluLoadError>;
    fn require_symbol(
        &mut self,
        handle: &Self::Handle,
        library: &LibraryContract,
        symbol: &str,
    ) -> Result<(), MluLoadError>;
    fn version(
        &mut self,
        handle: &Self::Handle,
        library: &LibraryContract,
    ) -> Result<LibraryVersion, MluLoadError>;
}

#[cfg(test)]
fn probe_with_system<S: ProbeSystem>(
    system: &mut S,
    target: &str,
    images: &CertifiedMluImages<'_>,
) -> Result<MluAbiProbe, MluLoadError> {
    let manifest =
        AbiManifest::embedded().map_err(|error| MluLoadError::Manifest(error.to_string()))?;
    if !manifest.targets.iter().any(|candidate| candidate == target) {
        return Err(MluLoadError::UnsupportedTarget {
            target: target.to_owned(),
        });
    }

    let mut versions = BTreeMap::new();
    let mut symbol_count = 0;
    for library_id in LIBRARY_LOAD_ORDER {
        let library = manifest
            .libraries
            .iter()
            .find(|candidate| candidate.id == library_id)
            .ok_or_else(|| MluLoadError::Manifest(format!("missing {library_id}")))?;
        let image = images.image(library_id)?;
        let handle = system.open(library, image)?;
        for symbol in &library.symbols {
            system.require_symbol(&handle, library, &symbol.name)?;
            symbol_count += 1;
        }
        let version = system.version(&handle, library)?;
        validate_version(library_id, version)?;
        versions.insert(library_id, version);
    }
    let cnrt_version = versions
        .get("cnrt")
        .copied()
        .ok_or_else(|| MluLoadError::Manifest("missing cnrt version".to_owned()))?;
    let cnnl_version = versions
        .get("cnnl")
        .copied()
        .ok_or_else(|| MluLoadError::Manifest("missing cnnl version".to_owned()))?;
    Ok(MluAbiProbe {
        target: target.to_owned(),
        abi_floor: ABI_FLOOR.to_owned(),
        cnrt_version,
        cnnl_version,
        symbol_count,
    })
}

fn validate_version(library: &str, actual: LibraryVersion) -> Result<(), MluLoadError> {
    let (required, compatible) = match library {
        "cnrt" => (
            LibraryVersion {
                major: 6,
                minor: 6,
                patch: 0,
            },
            actual.major == 6
                && actual
                    >= LibraryVersion {
                        major: 6,
                        minor: 6,
                        patch: 0,
                    },
        ),
        "cnnl" => (
            LibraryVersion {
                major: 1,
                minor: 20,
                patch: 4,
            },
            actual.major == 1
                && actual
                    >= LibraryVersion {
                        major: 1,
                        minor: 20,
                        patch: 4,
                    },
        ),
        _ => return Err(MluLoadError::Manifest(format!("unknown library {library}"))),
    };
    if compatible {
        Ok(())
    } else {
        Err(MluLoadError::Version {
            library: library.to_owned(),
            required: version_string(required),
            actual: version_string(actual),
        })
    }
}

fn version_string(version: LibraryVersion) -> String {
    format!("{}.{}.{}", version.major, version.minor, version.patch)
}

struct MluSymbols {
    cnrt_get_lib_version: CnrtGetLibVersion,
    cnrt_get_device_count: CnrtGetDeviceCount,
    cnrt_set_device: CnrtSetDevice,
    cnrt_malloc: CnrtMalloc,
    cnrt_free: CnrtFree,
    cnrt_memcpy: CnrtMemcpy,
    cnrt_queue_create: CnrtQueueCreate,
    cnrt_queue_destroy: CnrtQueueDestroy,
    cnrt_queue_sync: CnrtQueueSync,
    cnnl_get_lib_version: CnnlGetLibVersion,
    cnnl_create: CnnlCreate,
    cnnl_create_op_tensor_descriptor: CnnlCreateOpTensorDescriptor,
    cnnl_destroy: CnnlDestroy,
    cnnl_destroy_op_tensor_descriptor: CnnlDestroyOpTensorDescriptor,
    cnnl_set_queue: CnnlSetQueue,
    cnnl_create_tensor_descriptor: CnnlCreateTensorDescriptor,
    cnnl_destroy_tensor_descriptor: CnnlDestroyTensorDescriptor,
    cnnl_set_tensor_descriptor: CnnlSetTensorDescriptor,
    cnnl_set_op_tensor_descriptor: CnnlSetOpTensorDescriptor,
    cnnl_op_tensor: CnnlOpTensor,
}

struct MluCallSurface<'authority> {
    symbols: MluSymbols,
    authority_lifetime: PhantomData<&'authority ()>,
}

pub struct MluRuntime<'certificate> {
    calls: MluCallSurface<'certificate>,
    _retained_images: platform::RetainedHandles,
}

struct SerializedAllocation {
    pointer: NonNull<c_void>,
    bytes: usize,
    device: u32,
}

struct SerializedQueue {
    pointer: NonNull<c_void>,
    device: u32,
}

pub(crate) struct SerializedMluCore {
    runtime: MluRuntime<'static>,
    device_count: u32,
    allocations: BTreeMap<u64, SerializedAllocation>,
    queues: BTreeMap<u64, SerializedQueue>,
}

// SAFETY: no pointer or borrowed vendor resource can escape this type. Its crate-private API
// requires exclusive access for every call, and `MluExecutionRuntime` places it behind one mutex.
unsafe impl Send for SerializedMluCore {}

impl<'certificate> MluRuntime<'certificate> {
    pub fn load(
        images: &'certificate CertifiedMluImages<'certificate>,
    ) -> Result<Self, MluLoadError> {
        let (retained_images, symbols) = platform::load_runtime(images)?;
        let runtime = Self {
            calls: MluCallSurface {
                symbols,
                authority_lifetime: PhantomData,
            },
            _retained_images: retained_images,
        };
        runtime.probe()?;
        Ok(runtime)
    }

    pub fn probe(&self) -> Result<MluAbiProbe, MluLoadError> {
        self.calls.probe(env!("SIM_MLU_TARGET"))
    }

    pub fn device_count(&self) -> Result<u32, MluLoadError> {
        self.calls.device_count()
    }

    pub fn set_device(&mut self, device_id: i32) -> Result<(), MluLoadError> {
        self.calls.set_device(device_id)
    }

    fn allocate(&self, bytes: usize) -> Result<MluAllocation<'_, 'certificate>, MluLoadError> {
        self.calls.allocate(bytes)
    }

    fn create_queue(&self) -> Result<MluQueue<'_, 'certificate>, MluLoadError> {
        self.calls.create_queue()
    }

    fn create_cnnl_context(&self) -> Result<MluCnnlContext<'_, 'certificate>, MluLoadError> {
        self.calls.create_cnnl_context()
    }

    pub(crate) fn into_serialized_core(self) -> Result<SerializedMluCore, MluLoadError> {
        let device_count = self.device_count()?;
        let MluRuntime {
            calls,
            _retained_images,
        } = self;
        let runtime = MluRuntime {
            calls: MluCallSurface {
                symbols: calls.symbols,
                authority_lifetime: PhantomData,
            },
            _retained_images,
        };
        Ok(SerializedMluCore {
            runtime,
            device_count,
            allocations: BTreeMap::new(),
            queues: BTreeMap::new(),
        })
    }
}

impl SerializedMluCore {
    pub(crate) fn probe(&self) -> Result<MluAbiProbe, MluLoadError> {
        self.runtime.probe()
    }

    pub(crate) const fn device_count(&self) -> u32 {
        self.device_count
    }

    fn select_device(&mut self, device: u32) -> Result<(), MluLoadError> {
        if device >= self.device_count {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtSetDevice",
                reason: "device ID is outside the certified device count",
            });
        }
        let device = i32::try_from(device).map_err(|_| MluLoadError::InvalidArgument {
            operation: "cnrtSetDevice",
            reason: "device ID exceeds the reviewed signed ABI",
        })?;
        self.runtime.set_device(device)
    }

    pub(crate) fn allocate(
        &mut self,
        resource_id: u64,
        device: u32,
        bytes: usize,
    ) -> Result<(), MluLoadError> {
        self.select_device(device)?;
        if self.allocations.contains_key(&resource_id) {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtMalloc",
                reason: "allocation resource ID is duplicated",
            });
        }
        let mut allocation = self.runtime.allocate(bytes)?;
        let pointer = allocation
            .pointer
            .take()
            .ok_or(MluLoadError::NullResource {
                operation: "cnrtMalloc",
            })?;
        self.allocations.insert(
            resource_id,
            SerializedAllocation {
                pointer,
                bytes,
                device,
            },
        );
        Ok(())
    }

    pub(crate) fn release_allocation(&mut self, resource_id: u64) -> Result<(), MluLoadError> {
        let allocation =
            self.allocations
                .get(&resource_id)
                .ok_or(MluLoadError::InvalidArgument {
                    operation: "cnrtFree",
                    reason: "allocation resource is closed",
                })?;
        let device = allocation.device;
        let pointer = allocation.pointer;
        self.select_device(device)?;
        check_cnrt("cnrtFree", unsafe {
            (self.runtime.calls.symbols.cnrt_free)(pointer.as_ptr())
        })?;
        self.allocations.remove(&resource_id);
        Ok(())
    }

    pub(crate) fn create_queue(
        &mut self,
        resource_id: u64,
        device: u32,
    ) -> Result<(), MluLoadError> {
        self.select_device(device)?;
        if self.queues.contains_key(&resource_id) {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtQueueCreate",
                reason: "queue resource ID is duplicated",
            });
        }
        let mut queue = self.runtime.create_queue()?;
        let pointer = queue.queue.take().ok_or(MluLoadError::NullResource {
            operation: "cnrtQueueCreate",
        })?;
        self.queues
            .insert(resource_id, SerializedQueue { pointer, device });
        Ok(())
    }

    pub(crate) fn synchronize_queue(&mut self, resource_id: u64) -> Result<(), MluLoadError> {
        let queue = self
            .queues
            .get(&resource_id)
            .ok_or(MluLoadError::InvalidArgument {
                operation: "cnrtQueueSync",
                reason: "queue resource is closed",
            })?;
        let device = queue.device;
        let pointer = queue.pointer;
        self.select_device(device)?;
        check_cnrt("cnrtQueueSync", unsafe {
            (self.runtime.calls.symbols.cnrt_queue_sync)(pointer.as_ptr())
        })
    }

    pub(crate) fn release_queue(&mut self, resource_id: u64) -> Result<(), MluLoadError> {
        let queue = self
            .queues
            .get(&resource_id)
            .ok_or(MluLoadError::InvalidArgument {
                operation: "cnrtQueueDestroy",
                reason: "queue resource is closed",
            })?;
        let device = queue.device;
        let pointer = queue.pointer;
        self.select_device(device)?;
        check_cnrt("cnrtQueueSync", unsafe {
            (self.runtime.calls.symbols.cnrt_queue_sync)(pointer.as_ptr())
        })?;
        check_cnrt("cnrtQueueDestroy", unsafe {
            (self.runtime.calls.symbols.cnrt_queue_destroy)(pointer.as_ptr())
        })?;
        self.queues.remove(&resource_id);
        Ok(())
    }

    pub(crate) fn copy_from_host(
        &mut self,
        resource_id: u64,
        offset: usize,
        source: &[u8],
    ) -> Result<(), MluLoadError> {
        let allocation = self.allocation(resource_id, offset, source.len(), "cnrtMemcpy")?;
        let device = allocation.device;
        let pointer = allocation.pointer;
        self.select_device(device)?;
        let destination = pointer_at(pointer, offset)?;
        check_cnrt("cnrtMemcpy", unsafe {
            (self.runtime.calls.symbols.cnrt_memcpy)(
                destination,
                source.as_ptr().cast_mut().cast(),
                source.len(),
                CnrtMemTransferDirection::HostToDevice,
            )
        })
    }

    pub(crate) fn copy_to_host(
        &mut self,
        resource_id: u64,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), MluLoadError> {
        let allocation = self.allocation(resource_id, offset, destination.len(), "cnrtMemcpy")?;
        let device = allocation.device;
        let pointer = allocation.pointer;
        self.select_device(device)?;
        let source = pointer_at(pointer, offset)?;
        check_cnrt("cnrtMemcpy", unsafe {
            (self.runtime.calls.symbols.cnrt_memcpy)(
                destination.as_mut_ptr().cast(),
                source,
                destination.len(),
                CnrtMemTransferDirection::DeviceToHost,
            )
        })
    }

    pub(crate) fn copy_device_to_device(
        &mut self,
        destination_id: u64,
        destination_offset: usize,
        source_id: u64,
        source_offset: usize,
        bytes: usize,
    ) -> Result<(), MluLoadError> {
        let destination =
            self.allocation(destination_id, destination_offset, bytes, "cnrtMemcpy")?;
        let source = self.allocation(source_id, source_offset, bytes, "cnrtMemcpy")?;
        if destination.device != source.device {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtMemcpy",
                reason: "reviewed device-to-device copy requires one selected device",
            });
        }
        let device = destination.device;
        let destination = pointer_at(destination.pointer, destination_offset)?;
        let source = pointer_at(source.pointer, source_offset)?;
        self.select_device(device)?;
        check_cnrt("cnrtMemcpy", unsafe {
            (self.runtime.calls.symbols.cnrt_memcpy)(
                destination,
                source,
                bytes,
                CnrtMemTransferDirection::DeviceToDevice,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add(
        &mut self,
        queue_id: u64,
        left_id: u64,
        right_id: u64,
        output_id: u64,
        dimensions: &[i32],
        data_type: CnnlDataType,
        byte_width: usize,
    ) -> Result<(), MluLoadError> {
        let required = dimensions
            .iter()
            .try_fold(byte_width, |bytes, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| bytes.checked_mul(dimension))
            })
            .ok_or(MluLoadError::InvalidArgument {
                operation: "cnnlOpTensor",
                reason: "tensor element count overflows or has a nonpositive dimension",
            })?;
        let queue = self
            .queues
            .get(&queue_id)
            .ok_or(MluLoadError::InvalidArgument {
                operation: "cnnlSetQueue",
                reason: "queue resource is closed",
            })?;
        let left = self.allocation(left_id, 0, required, "cnnlOpTensor")?;
        let right = self.allocation(right_id, 0, required, "cnnlOpTensor")?;
        let output = self.allocation(output_id, 0, required, "cnnlOpTensor")?;
        if left.device != queue.device
            || right.device != queue.device
            || output.device != queue.device
        {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnnlOpTensor",
                reason: "queue and allocations must belong to one selected device",
            });
        }
        let device = queue.device;
        let queue_pointer = queue.pointer;
        let left_pointer = left.pointer;
        let right_pointer = right.pointer;
        let output_pointer = output.pointer;
        self.select_device(device)?;
        let queue = MluQueue {
            calls: &self.runtime.calls,
            queue: Some(queue_pointer),
            release_on_drop: false,
        };
        let left = MluAllocation {
            calls: &self.runtime.calls,
            pointer: Some(left_pointer),
            bytes: required,
            release_on_drop: false,
        };
        let right = MluAllocation {
            calls: &self.runtime.calls,
            pointer: Some(right_pointer),
            bytes: required,
            release_on_drop: false,
        };
        let mut output = MluAllocation {
            calls: &self.runtime.calls,
            pointer: Some(output_pointer),
            bytes: required,
            release_on_drop: false,
        };
        let mut context = self.runtime.create_cnnl_context()?;
        {
            let binding = context.bind_queue(&queue)?;
            let left_descriptor =
                binding.create_tensor_descriptor(dimensions, data_type, byte_width)?;
            let right_descriptor =
                binding.create_tensor_descriptor(dimensions, data_type, byte_width)?;
            let output_descriptor =
                binding.create_tensor_descriptor(dimensions, data_type, byte_width)?;
            let operation = binding.create_add_descriptor(data_type)?;
            binding.add(
                &operation,
                &left_descriptor,
                &left,
                &right_descriptor,
                &right,
                &output_descriptor,
                &mut output,
            )?;
        }
        queue.synchronize()
    }

    fn allocation(
        &self,
        resource_id: u64,
        offset: usize,
        length: usize,
        operation: &'static str,
    ) -> Result<&SerializedAllocation, MluLoadError> {
        let allocation =
            self.allocations
                .get(&resource_id)
                .ok_or(MluLoadError::InvalidArgument {
                    operation,
                    reason: "allocation resource is closed",
                })?;
        if length == 0
            || offset
                .checked_add(length)
                .is_none_or(|end| end > allocation.bytes)
        {
            return Err(MluLoadError::InvalidArgument {
                operation,
                reason: "copy or tensor range is empty, overflows, or exceeds the allocation",
            });
        }
        Ok(allocation)
    }
}

impl Drop for SerializedMluCore {
    fn drop(&mut self) {
        let queue_ids = self.queues.keys().copied().collect::<Vec<_>>();
        for queue_id in queue_ids {
            if let Err(error) = self.release_queue(queue_id) {
                eprintln!("failed to release serialized MLU queue: {error}");
            }
        }
        let allocation_ids = self.allocations.keys().copied().collect::<Vec<_>>();
        for allocation_id in allocation_ids {
            if let Err(error) = self.release_allocation(allocation_id) {
                eprintln!("failed to release serialized MLU allocation: {error}");
            }
        }
    }
}

fn pointer_at(pointer: NonNull<c_void>, offset: usize) -> Result<*mut c_void, MluLoadError> {
    let pointer = pointer.as_ptr().cast::<u8>();
    Ok(pointer.wrapping_add(offset).cast())
}

impl<'authority> MluCallSurface<'authority> {
    fn probe(&self, target: &str) -> Result<MluAbiProbe, MluLoadError> {
        if !matches!(
            target,
            "aarch64-unknown-linux-gnu" | "x86_64-unknown-linux-gnu"
        ) {
            return Err(MluLoadError::UnsupportedTarget {
                target: target.to_owned(),
            });
        }
        let cnrt_version = self.cnrt_version()?;
        let cnnl_version = self.cnnl_version();
        validate_version("cnrt", cnrt_version)?;
        validate_version("cnnl", cnnl_version)?;
        Ok(MluAbiProbe {
            target: target.to_owned(),
            abi_floor: ABI_FLOOR.to_owned(),
            cnrt_version,
            cnnl_version,
            symbol_count: REVIEWED_SYMBOL_COUNT,
        })
    }

    fn cnrt_version(&self) -> Result<LibraryVersion, MluLoadError> {
        let mut major = 0;
        let mut minor = 0;
        let mut patch = 0;
        check_cnrt("cnrtGetLibVersion", unsafe {
            (self.symbols.cnrt_get_lib_version)(&mut major, &mut minor, &mut patch)
        })?;
        Ok(LibraryVersion {
            major,
            minor,
            patch,
        })
    }

    fn cnnl_version(&self) -> LibraryVersion {
        let mut major = 0;
        let mut minor = 0;
        let mut patch = 0;
        unsafe { (self.symbols.cnnl_get_lib_version)(&mut major, &mut minor, &mut patch) };
        LibraryVersion {
            major,
            minor,
            patch,
        }
    }

    fn device_count(&self) -> Result<u32, MluLoadError> {
        let mut count = 0;
        check_cnrt("cnrtGetDeviceCount", unsafe {
            (self.symbols.cnrt_get_device_count)(&mut count)
        })?;
        Ok(count)
    }

    fn set_device(&self, device_id: i32) -> Result<(), MluLoadError> {
        if device_id < 0 {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtSetDevice",
                reason: "device ID must be nonnegative",
            });
        }
        check_cnrt("cnrtSetDevice", unsafe {
            (self.symbols.cnrt_set_device)(device_id)
        })
    }

    fn allocate(&self, bytes: usize) -> Result<MluAllocation<'_, 'authority>, MluLoadError> {
        if bytes == 0 {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtMalloc",
                reason: "allocation size must be nonzero",
            });
        }
        let mut pointer = std::ptr::null_mut();
        check_cnrt("cnrtMalloc", unsafe {
            (self.symbols.cnrt_malloc)(&mut pointer, bytes)
        })?;
        Ok(MluAllocation {
            calls: self,
            pointer: Some(NonNull::new(pointer).ok_or(MluLoadError::NullResource {
                operation: "cnrtMalloc",
            })?),
            bytes,
            release_on_drop: true,
        })
    }

    fn create_queue(&self) -> Result<MluQueue<'_, 'authority>, MluLoadError> {
        let mut queue = std::ptr::null_mut();
        check_cnrt("cnrtQueueCreate", unsafe {
            (self.symbols.cnrt_queue_create)(&mut queue)
        })?;
        Ok(MluQueue {
            calls: self,
            queue: Some(NonNull::new(queue).ok_or(MluLoadError::NullResource {
                operation: "cnrtQueueCreate",
            })?),
            release_on_drop: true,
        })
    }

    fn create_cnnl_context(&self) -> Result<MluCnnlContext<'_, 'authority>, MluLoadError> {
        let mut handle = std::ptr::null_mut();
        check_cnnl("cnnlCreate", unsafe {
            (self.symbols.cnnl_create)(&mut handle)
        })?;
        Ok(MluCnnlContext {
            calls: self,
            handle: Some(NonNull::new(handle).ok_or(MluLoadError::NullResource {
                operation: "cnnlCreate",
            })?),
        })
    }

    #[cfg(test)]
    fn copy(
        &self,
        destination: &mut MluAllocation<'_, 'authority>,
        source: &MluAllocation<'_, 'authority>,
        bytes: usize,
        direction: CnrtMemTransferDirection,
    ) -> Result<(), MluLoadError> {
        if !std::ptr::eq(destination.calls, self) || !std::ptr::eq(source.calls, self) {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnrtMemcpy",
                reason: "both allocations must belong to this certified runtime",
            });
        }
        validate_copy_bounds(destination.bytes, source.bytes, bytes)?;
        check_cnrt("cnrtMemcpy", unsafe {
            (self.symbols.cnrt_memcpy)(destination.pointer(), source.pointer(), bytes, direction)
        })
    }
}

pub struct MluAllocation<'runtime, 'authority> {
    calls: &'runtime MluCallSurface<'authority>,
    pointer: Option<NonNull<c_void>>,
    bytes: usize,
    release_on_drop: bool,
}

impl<'runtime, 'authority> MluAllocation<'runtime, 'authority> {
    fn pointer(&self) -> *mut c_void {
        self.pointer.map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }

    fn release_inner(&mut self) -> Result<(), MluLoadError> {
        if !self.release_on_drop {
            self.pointer = None;
            return Ok(());
        }
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        check_cnrt("cnrtFree", unsafe {
            (self.calls.symbols.cnrt_free)(pointer.as_ptr())
        })?;
        self.pointer = None;
        Ok(())
    }
}

impl Drop for MluAllocation<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.release_inner() {
            eprintln!("failed to release MLU allocation: {error}");
        }
    }
}

pub struct MluQueue<'runtime, 'authority> {
    calls: &'runtime MluCallSurface<'authority>,
    queue: Option<NonNull<c_void>>,
    release_on_drop: bool,
}

impl MluQueue<'_, '_> {
    pub fn synchronize(&self) -> Result<(), MluLoadError> {
        let queue = self.queue.ok_or(MluLoadError::InvalidArgument {
            operation: "cnrtQueueSync",
            reason: "queue is closed",
        })?;
        check_cnrt("cnrtQueueSync", unsafe {
            (self.calls.symbols.cnrt_queue_sync)(queue.as_ptr())
        })
    }

    fn close_inner(&mut self) -> Result<(), MluLoadError> {
        if !self.release_on_drop {
            self.queue = None;
            return Ok(());
        }
        let Some(queue) = self.queue else {
            return Ok(());
        };
        check_cnrt("cnrtQueueDestroy", unsafe {
            (self.calls.symbols.cnrt_queue_destroy)(queue.as_ptr())
        })?;
        self.queue = None;
        Ok(())
    }
}

impl Drop for MluQueue<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.close_inner() {
            eprintln!("failed to destroy MLU queue: {error}");
        }
    }
}

pub struct MluCnnlContext<'runtime, 'authority> {
    calls: &'runtime MluCallSurface<'authority>,
    handle: Option<NonNull<c_void>>,
}

impl<'runtime, 'authority> MluCnnlContext<'runtime, 'authority> {
    pub fn bind_queue<'binding>(
        &'binding mut self,
        queue: &'binding MluQueue<'runtime, 'authority>,
    ) -> Result<MluCnnlQueueBinding<'binding, 'runtime, 'authority>, MluLoadError> {
        if !std::ptr::eq(self.calls, queue.calls) {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnnlSetQueue",
                reason: "CNNL context and queue must belong to one certified runtime",
            });
        }
        let handle = self.handle.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlSetQueue",
            reason: "CNNL context is closed",
        })?;
        let queue_pointer = queue.queue.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlSetQueue",
            reason: "queue is closed",
        })?;
        check_cnnl("cnnlSetQueue", unsafe {
            (self.calls.symbols.cnnl_set_queue)(handle.as_ptr(), queue_pointer.as_ptr())
        })?;
        Ok(MluCnnlQueueBinding {
            context: self,
            _queue: queue,
        })
    }

    pub fn create_tensor_descriptor(
        &self,
    ) -> Result<MluTensorDescriptor<'_, 'runtime, 'authority>, MluLoadError> {
        let mut descriptor = std::ptr::null_mut();
        check_cnnl("cnnlCreateTensorDescriptor", unsafe {
            (self.calls.symbols.cnnl_create_tensor_descriptor)(&mut descriptor)
        })?;
        Ok(MluTensorDescriptor {
            context: self,
            descriptor: Some(NonNull::new(descriptor).ok_or(MluLoadError::NullResource {
                operation: "cnnlCreateTensorDescriptor",
            })?),
            element_count: 0,
            required_bytes: 0,
        })
    }

    fn close_inner(&mut self) -> Result<(), MluLoadError> {
        let Some(handle) = self.handle else {
            return Ok(());
        };
        check_cnnl("cnnlDestroy", unsafe {
            (self.calls.symbols.cnnl_destroy)(handle.as_ptr())
        })?;
        self.handle = None;
        Ok(())
    }
}

impl Drop for MluCnnlContext<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.close_inner() {
            eprintln!("failed to destroy CNNL context: {error}");
        }
    }
}

pub struct MluCnnlQueueBinding<'binding, 'runtime, 'authority> {
    context: &'binding mut MluCnnlContext<'runtime, 'authority>,
    _queue: &'binding MluQueue<'runtime, 'authority>,
}

impl MluCnnlQueueBinding<'_, '_, '_> {
    #[cfg(test)]
    fn create_f32_tensor_descriptor(
        &self,
        dimensions: &[i32],
    ) -> Result<MluTensorDescriptor<'_, '_, '_>, MluLoadError> {
        let mut descriptor = self.context.create_tensor_descriptor()?;
        descriptor.set_array(dimensions, CnnlDataType::Float, 4)?;
        Ok(descriptor)
    }

    pub(crate) fn create_tensor_descriptor(
        &self,
        dimensions: &[i32],
        data_type: CnnlDataType,
        byte_width: usize,
    ) -> Result<MluTensorDescriptor<'_, '_, '_>, MluLoadError> {
        let mut descriptor = self.context.create_tensor_descriptor()?;
        descriptor.set_array(dimensions, data_type, byte_width)?;
        Ok(descriptor)
    }

    #[cfg(test)]
    fn create_f32_add_descriptor(&self) -> Result<MluOpTensorDescriptor<'_, '_, '_>, MluLoadError> {
        let mut descriptor = std::ptr::null_mut();
        check_cnnl("cnnlCreateOpTensorDescriptor", unsafe {
            (self.context.calls.symbols.cnnl_create_op_tensor_descriptor)(&mut descriptor)
        })?;
        let mut descriptor = MluOpTensorDescriptor {
            context: self.context,
            descriptor: Some(NonNull::new(descriptor).ok_or(MluLoadError::NullResource {
                operation: "cnnlCreateOpTensorDescriptor",
            })?),
        };
        descriptor.configure_add(CnnlDataType::Float)?;
        Ok(descriptor)
    }

    pub(crate) fn create_add_descriptor(
        &self,
        data_type: CnnlDataType,
    ) -> Result<MluOpTensorDescriptor<'_, '_, '_>, MluLoadError> {
        let mut descriptor = std::ptr::null_mut();
        check_cnnl("cnnlCreateOpTensorDescriptor", unsafe {
            (self.context.calls.symbols.cnnl_create_op_tensor_descriptor)(&mut descriptor)
        })?;
        let mut descriptor = MluOpTensorDescriptor {
            context: self.context,
            descriptor: Some(NonNull::new(descriptor).ok_or(MluLoadError::NullResource {
                operation: "cnnlCreateOpTensorDescriptor",
            })?),
        };
        descriptor.configure_add(data_type)?;
        Ok(descriptor)
    }

    #[cfg(test)]
    fn add_f32(
        &self,
        operation: &MluOpTensorDescriptor<'_, '_, '_>,
        left_descriptor: &MluTensorDescriptor<'_, '_, '_>,
        left: &MluAllocation<'_, '_>,
        right_descriptor: &MluTensorDescriptor<'_, '_, '_>,
        right: &MluAllocation<'_, '_>,
        output_descriptor: &MluTensorDescriptor<'_, '_, '_>,
        output: &mut MluAllocation<'_, '_>,
    ) -> Result<(), MluLoadError> {
        self.add(
            operation,
            left_descriptor,
            left,
            right_descriptor,
            right,
            output_descriptor,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add(
        &self,
        operation: &MluOpTensorDescriptor<'_, '_, '_>,
        left_descriptor: &MluTensorDescriptor<'_, '_, '_>,
        left: &MluAllocation<'_, '_>,
        right_descriptor: &MluTensorDescriptor<'_, '_, '_>,
        right: &MluAllocation<'_, '_>,
        output_descriptor: &MluTensorDescriptor<'_, '_, '_>,
        output: &mut MluAllocation<'_, '_>,
    ) -> Result<(), MluLoadError> {
        let calls = self.context.calls;
        if !std::ptr::eq(operation.context.calls, calls)
            || !std::ptr::eq(left_descriptor.context.calls, calls)
            || !std::ptr::eq(right_descriptor.context.calls, calls)
            || !std::ptr::eq(output_descriptor.context.calls, calls)
            || !std::ptr::eq(left.calls, calls)
            || !std::ptr::eq(right.calls, calls)
            || !std::ptr::eq(output.calls, calls)
        {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnnlOpTensor",
                reason: "descriptors and allocations must belong to one certified runtime",
            });
        }
        if left_descriptor.required_bytes > left.bytes
            || right_descriptor.required_bytes > right.bytes
            || output_descriptor.required_bytes > output.bytes
            || left_descriptor.element_count != right_descriptor.element_count
            || left_descriptor.element_count != output_descriptor.element_count
        {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnnlOpTensor",
                reason: "descriptor shapes must match and fit their allocations",
            });
        }
        let handle = self.context.handle.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlOpTensor",
            reason: "CNNL context is closed",
        })?;
        let operation = operation.descriptor.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlOpTensor",
            reason: "operation descriptor is closed",
        })?;
        let left_descriptor = left_descriptor
            .descriptor
            .ok_or(MluLoadError::InvalidArgument {
                operation: "cnnlOpTensor",
                reason: "left descriptor is closed",
            })?;
        let right_descriptor =
            right_descriptor
                .descriptor
                .ok_or(MluLoadError::InvalidArgument {
                    operation: "cnnlOpTensor",
                    reason: "right descriptor is closed",
                })?;
        let output_descriptor =
            output_descriptor
                .descriptor
                .ok_or(MluLoadError::InvalidArgument {
                    operation: "cnnlOpTensor",
                    reason: "output descriptor is closed",
                })?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        check_cnnl("cnnlOpTensor", unsafe {
            (calls.symbols.cnnl_op_tensor)(
                handle.as_ptr(),
                operation.as_ptr(),
                std::ptr::from_ref(&alpha).cast(),
                left_descriptor.as_ptr(),
                left.pointer().cast_const(),
                std::ptr::from_ref(&alpha).cast(),
                right_descriptor.as_ptr(),
                right.pointer().cast_const(),
                std::ptr::null_mut(),
                0,
                std::ptr::from_ref(&beta).cast(),
                output_descriptor.as_ptr(),
                output.pointer(),
            )
        })
    }
}

pub struct MluTensorDescriptor<'context, 'runtime, 'authority> {
    context: &'context MluCnnlContext<'runtime, 'authority>,
    descriptor: Option<NonNull<c_void>>,
    element_count: usize,
    required_bytes: usize,
}

impl MluTensorDescriptor<'_, '_, '_> {
    fn set_array(
        &mut self,
        dimensions: &[i32],
        data_type: CnnlDataType,
        byte_width: usize,
    ) -> Result<(), MluLoadError> {
        if dimensions.is_empty()
            || dimensions.len() > 8
            || dimensions.iter().any(|value| *value <= 0)
        {
            return Err(MluLoadError::InvalidArgument {
                operation: "cnnlSetTensorDescriptor",
                reason: "rank must be 1 through 8 and every dimension must be positive",
            });
        }
        let element_count = dimensions.iter().try_fold(1_usize, |count, dimension| {
            count.checked_mul(*dimension as usize)
        });
        let element_count = element_count.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlSetTensorDescriptor",
            reason: "tensor element count overflows",
        })?;
        let required_bytes =
            element_count
                .checked_mul(byte_width)
                .ok_or(MluLoadError::InvalidArgument {
                    operation: "cnnlSetTensorDescriptor",
                    reason: "tensor byte count overflows",
                })?;
        let descriptor = self.descriptor.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlSetTensorDescriptor",
            reason: "tensor descriptor is closed",
        })?;
        check_cnnl("cnnlSetTensorDescriptor", unsafe {
            (self.context.calls.symbols.cnnl_set_tensor_descriptor)(
                descriptor.as_ptr(),
                CnnlTensorLayout::Array,
                data_type,
                dimensions.len() as i32,
                dimensions.as_ptr(),
            )
        })?;
        self.element_count = element_count;
        self.required_bytes = required_bytes;
        Ok(())
    }

    fn close_inner(&mut self) -> Result<(), MluLoadError> {
        let Some(descriptor) = self.descriptor else {
            return Ok(());
        };
        check_cnnl("cnnlDestroyTensorDescriptor", unsafe {
            (self.context.calls.symbols.cnnl_destroy_tensor_descriptor)(descriptor.as_ptr())
        })?;
        self.descriptor = None;
        Ok(())
    }
}

pub struct MluOpTensorDescriptor<'context, 'runtime, 'authority> {
    context: &'context MluCnnlContext<'runtime, 'authority>,
    descriptor: Option<NonNull<c_void>>,
}

impl MluOpTensorDescriptor<'_, '_, '_> {
    fn configure_add(&mut self, data_type: CnnlDataType) -> Result<(), MluLoadError> {
        let descriptor = self.descriptor.ok_or(MluLoadError::InvalidArgument {
            operation: "cnnlSetOpTensorDescriptor",
            reason: "operation descriptor is closed",
        })?;
        check_cnnl("cnnlSetOpTensorDescriptor", unsafe {
            (self.context.calls.symbols.cnnl_set_op_tensor_descriptor)(
                descriptor.as_ptr(),
                CnnlOpTensorDescription::Add,
                data_type,
                CnnlNanPropagation::Propagate,
            )
        })
    }

    fn close_inner(&mut self) -> Result<(), MluLoadError> {
        let Some(descriptor) = self.descriptor else {
            return Ok(());
        };
        check_cnnl("cnnlDestroyOpTensorDescriptor", unsafe {
            (self.context.calls.symbols.cnnl_destroy_op_tensor_descriptor)(descriptor.as_ptr())
        })?;
        self.descriptor = None;
        Ok(())
    }
}

impl Drop for MluOpTensorDescriptor<'_, '_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.close_inner() {
            eprintln!("failed to destroy CNNL operation descriptor: {error}");
        }
    }
}

impl Drop for MluTensorDescriptor<'_, '_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.close_inner() {
            eprintln!("failed to destroy CNNL tensor descriptor: {error}");
        }
    }
}

#[cfg(test)]
fn validate_copy_bounds(
    destination_bytes: usize,
    source_bytes: usize,
    bytes: usize,
) -> Result<(), MluLoadError> {
    if bytes == 0 || bytes > destination_bytes || bytes > source_bytes {
        return Err(MluLoadError::InvalidArgument {
            operation: "cnrtMemcpy",
            reason: "copy size must be nonzero and fit both allocations",
        });
    }
    Ok(())
}

fn check_cnrt(operation: &'static str, status: CnrtStatus) -> Result<(), MluLoadError> {
    if status != CnrtStatus::SUCCESS {
        return Err(MluLoadError::CallFailed {
            operation,
            status: status.0,
        });
    }
    Ok(())
}

fn check_cnnl(operation: &'static str, status: CnnlStatus) -> Result<(), MluLoadError> {
    if status != CnnlStatus::SUCCESS {
        return Err(MluLoadError::CallFailed {
            operation,
            status: status.0,
        });
    }
    Ok(())
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod platform {
    use super::*;
    use std::{
        ffi::{CStr, CString, c_void},
        ptr::NonNull,
    };

    struct Handle(NonNull<c_void>);

    pub(super) struct RetainedHandles {
        _cnrt: Handle,
        _cnnl: Handle,
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            // Vendor runtimes may retain process resources. We still balance each probe handle;
            // a destructor cannot return a typed error, so a failed dlclose remains visible.
            let status = unsafe { libc::dlclose(self.0.as_ptr()) };
            if status != 0 {
                eprintln!(
                    "failed to close certified MLU probe image: {}",
                    last_dl_error()
                );
            }
        }
    }

    struct System;

    impl System {
        fn open(
            &mut self,
            library: &LibraryContract,
            image: &RegistryCertifiedImage,
        ) -> Result<Handle, MluLoadError> {
            if !verify_immutable_sealed_fd(&image.retained_image_path) {
                return Err(MluLoadError::UnsealedImagePath {
                    library: library.id.clone(),
                    path: image.retained_image_path.display().to_string(),
                });
            }
            let path = CString::new(image.retained_image_path.as_os_str().as_encoded_bytes())
                .map_err(|_| MluLoadError::LibraryOpen {
                    library: library.id.clone(),
                    reason: "retained image path contains NUL".to_owned(),
                })?;
            let raw = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            NonNull::new(raw)
                .map(Handle)
                .ok_or_else(|| MluLoadError::LibraryOpen {
                    library: library.id.clone(),
                    reason: last_dl_error(),
                })
        }
    }

    fn symbol_address(
        handle: &Handle,
        library: &LibraryContract,
        symbol: &str,
    ) -> Result<NonNull<c_void>, MluLoadError> {
        let name = CString::new(symbol).map_err(|_| MluLoadError::MissingSymbol {
            library: library.id.clone(),
            symbol: symbol.to_owned(),
        })?;
        let address = unsafe { libc::dlsym(handle.0.as_ptr(), name.as_ptr()) };
        NonNull::new(address).ok_or_else(|| MluLoadError::MissingSymbol {
            library: library.id.clone(),
            symbol: symbol.to_owned(),
        })
    }

    fn last_dl_error() -> String {
        let error = unsafe { libc::dlerror() };
        if error.is_null() {
            "dynamic loader returned no diagnostic".to_owned()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        }
    }

    pub(super) fn load_runtime(
        images: &CertifiedMluImages<'_>,
    ) -> Result<(RetainedHandles, MluSymbols), MluLoadError> {
        let manifest =
            AbiManifest::embedded().map_err(|error| MluLoadError::Manifest(error.to_string()))?;
        let cnrt_contract = manifest
            .libraries
            .iter()
            .find(|library| library.id == "cnrt")
            .ok_or_else(|| MluLoadError::Manifest("missing cnrt".to_owned()))?;
        let cnnl_contract = manifest
            .libraries
            .iter()
            .find(|library| library.id == "cnnl")
            .ok_or_else(|| MluLoadError::Manifest("missing cnnl".to_owned()))?;
        let mut system = System;
        let cnrt = system.open(cnrt_contract, images.image("cnrt")?)?;
        let cnnl = system.open(cnnl_contract, images.image("cnnl")?)?;

        macro_rules! resolve_cnrt {
            ($symbol:literal, $type:ty) => {{
                let address = symbol_address(&cnrt, cnrt_contract, $symbol)?;
                unsafe { std::mem::transmute::<*mut c_void, $type>(address.as_ptr()) }
            }};
        }
        macro_rules! resolve_cnnl {
            ($symbol:literal, $type:ty) => {{
                let address = symbol_address(&cnnl, cnnl_contract, $symbol)?;
                unsafe { std::mem::transmute::<*mut c_void, $type>(address.as_ptr()) }
            }};
        }

        let symbols = MluSymbols {
            cnrt_get_lib_version: resolve_cnrt!("cnrtGetLibVersion", CnrtGetLibVersion),
            cnrt_get_device_count: resolve_cnrt!("cnrtGetDeviceCount", CnrtGetDeviceCount),
            cnrt_set_device: resolve_cnrt!("cnrtSetDevice", CnrtSetDevice),
            cnrt_malloc: resolve_cnrt!("cnrtMalloc", CnrtMalloc),
            cnrt_free: resolve_cnrt!("cnrtFree", CnrtFree),
            cnrt_memcpy: resolve_cnrt!("cnrtMemcpy", CnrtMemcpy),
            cnrt_queue_create: resolve_cnrt!("cnrtQueueCreate", CnrtQueueCreate),
            cnrt_queue_destroy: resolve_cnrt!("cnrtQueueDestroy", CnrtQueueDestroy),
            cnrt_queue_sync: resolve_cnrt!("cnrtQueueSync", CnrtQueueSync),
            cnnl_get_lib_version: resolve_cnnl!("cnnlGetLibVersion", CnnlGetLibVersion),
            cnnl_create: resolve_cnnl!("cnnlCreate", CnnlCreate),
            cnnl_create_op_tensor_descriptor: resolve_cnnl!(
                "cnnlCreateOpTensorDescriptor",
                CnnlCreateOpTensorDescriptor
            ),
            cnnl_destroy: resolve_cnnl!("cnnlDestroy", CnnlDestroy),
            cnnl_destroy_op_tensor_descriptor: resolve_cnnl!(
                "cnnlDestroyOpTensorDescriptor",
                CnnlDestroyOpTensorDescriptor
            ),
            cnnl_set_queue: resolve_cnnl!("cnnlSetQueue", CnnlSetQueue),
            cnnl_create_tensor_descriptor: resolve_cnnl!(
                "cnnlCreateTensorDescriptor",
                CnnlCreateTensorDescriptor
            ),
            cnnl_destroy_tensor_descriptor: resolve_cnnl!(
                "cnnlDestroyTensorDescriptor",
                CnnlDestroyTensorDescriptor
            ),
            cnnl_set_tensor_descriptor: resolve_cnnl!(
                "cnnlSetTensorDescriptor",
                CnnlSetTensorDescriptor
            ),
            cnnl_set_op_tensor_descriptor: resolve_cnnl!(
                "cnnlSetOpTensorDescriptor",
                CnnlSetOpTensorDescriptor
            ),
            cnnl_op_tensor: resolve_cnnl!("cnnlOpTensor", CnnlOpTensor),
        };
        Ok((
            RetainedHandles {
                _cnrt: cnrt,
                _cnnl: cnnl,
            },
            symbols,
        ))
    }
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod platform {
    use super::*;

    pub(super) struct RetainedHandles;

    pub(super) fn load_runtime(
        _images: &CertifiedMluImages<'_>,
    ) -> Result<(RetainedHandles, MluSymbols), MluLoadError> {
        Err(MluLoadError::UnsupportedTarget {
            target: env!("SIM_MLU_TARGET").to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static QUEUE_SYNCHRONIZATIONS: AtomicUsize = AtomicUsize::new(0);
    static ALLOCATION_RELEASES: AtomicUsize = AtomicUsize::new(0);
    static CNNL_CONTEXT_RELEASES: AtomicUsize = AtomicUsize::new(0);
    static DESCRIPTOR_RELEASES: AtomicUsize = AtomicUsize::new(0);
    static OP_DESCRIPTOR_RELEASES: AtomicUsize = AtomicUsize::new(0);
    static OPERATOR_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn fake_cnrt_version(
        major: *mut i32,
        minor: *mut i32,
        patch: *mut i32,
    ) -> CnrtStatus {
        unsafe {
            major.write(6);
            minor.write(6);
            patch.write(0);
        }
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_device_count(count: *mut u32) -> CnrtStatus {
        unsafe { count.write(2) };
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_set_device(_device_id: i32) -> CnrtStatus {
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_malloc(pointer: *mut *mut c_void, _bytes: usize) -> CnrtStatus {
        unsafe { pointer.write(NonNull::<u8>::dangling().as_ptr().cast()) };
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_free(_pointer: *mut c_void) -> CnrtStatus {
        ALLOCATION_RELEASES.fetch_add(1, Ordering::SeqCst);
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_memcpy(
        _destination: *mut c_void,
        _source: *mut c_void,
        _bytes: usize,
        _direction: CnrtMemTransferDirection,
    ) -> CnrtStatus {
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_queue_create(queue: *mut *mut c_void) -> CnrtStatus {
        unsafe { queue.write(NonNull::<u16>::dangling().as_ptr().cast()) };
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_queue_destroy(_queue: *mut c_void) -> CnrtStatus {
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_queue_sync(_queue: *mut c_void) -> CnrtStatus {
        QUEUE_SYNCHRONIZATIONS.fetch_add(1, Ordering::SeqCst);
        CnrtStatus::SUCCESS
    }

    unsafe extern "C" fn fake_cnnl_version(major: *mut i32, minor: *mut i32, patch: *mut i32) {
        unsafe {
            major.write(1);
            minor.write(20);
            patch.write(4);
        }
    }

    unsafe extern "C" fn fake_cnnl_create(handle: *mut *mut c_void) -> CnnlStatus {
        unsafe { handle.write(NonNull::<u32>::dangling().as_ptr().cast()) };
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_cnnl_destroy(_handle: *mut c_void) -> CnnlStatus {
        CNNL_CONTEXT_RELEASES.fetch_add(1, Ordering::SeqCst);
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_op_descriptor_create(descriptor: *mut *mut c_void) -> CnnlStatus {
        unsafe { descriptor.write(NonNull::<u128>::dangling().as_ptr().cast()) };
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_op_descriptor_destroy(_descriptor: *mut c_void) -> CnnlStatus {
        OP_DESCRIPTOR_RELEASES.fetch_add(1, Ordering::SeqCst);
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_cnnl_set_queue(
        _handle: *mut c_void,
        _queue: *mut c_void,
    ) -> CnnlStatus {
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_descriptor_create(descriptor: *mut *mut c_void) -> CnnlStatus {
        unsafe { descriptor.write(NonNull::<u64>::dangling().as_ptr().cast()) };
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_descriptor_destroy(_descriptor: *mut c_void) -> CnnlStatus {
        DESCRIPTOR_RELEASES.fetch_add(1, Ordering::SeqCst);
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_set_tensor_descriptor(
        _descriptor: *mut c_void,
        _layout: CnnlTensorLayout,
        _data_type: CnnlDataType,
        _rank: i32,
        _dimensions: *const i32,
    ) -> CnnlStatus {
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_set_op_tensor_descriptor(
        _descriptor: *mut c_void,
        _operation: CnnlOpTensorDescription,
        _data_type: CnnlDataType,
        _nan: CnnlNanPropagation,
    ) -> CnnlStatus {
        CnnlStatus::SUCCESS
    }

    unsafe extern "C" fn fake_op_tensor(
        _handle: *mut c_void,
        _operation: *mut c_void,
        _alpha_left: *const c_void,
        _left_descriptor: *mut c_void,
        _left: *const c_void,
        _alpha_right: *const c_void,
        _right_descriptor: *mut c_void,
        _right: *const c_void,
        _workspace: *mut c_void,
        _workspace_bytes: usize,
        _beta: *const c_void,
        _output_descriptor: *mut c_void,
        _output: *mut c_void,
    ) -> CnnlStatus {
        OPERATOR_CALLS.fetch_add(1, Ordering::SeqCst);
        CnnlStatus::SUCCESS
    }

    fn fake_call_surface() -> MluCallSurface<'static> {
        MluCallSurface {
            symbols: MluSymbols {
                cnrt_get_lib_version: fake_cnrt_version,
                cnrt_get_device_count: fake_device_count,
                cnrt_set_device: fake_set_device,
                cnrt_malloc: fake_malloc,
                cnrt_free: fake_free,
                cnrt_memcpy: fake_memcpy,
                cnrt_queue_create: fake_queue_create,
                cnrt_queue_destroy: fake_queue_destroy,
                cnrt_queue_sync: fake_queue_sync,
                cnnl_get_lib_version: fake_cnnl_version,
                cnnl_create: fake_cnnl_create,
                cnnl_create_op_tensor_descriptor: fake_op_descriptor_create,
                cnnl_destroy: fake_cnnl_destroy,
                cnnl_destroy_op_tensor_descriptor: fake_op_descriptor_destroy,
                cnnl_set_queue: fake_cnnl_set_queue,
                cnnl_create_tensor_descriptor: fake_descriptor_create,
                cnnl_destroy_tensor_descriptor: fake_descriptor_destroy,
                cnnl_set_tensor_descriptor: fake_set_tensor_descriptor,
                cnnl_set_op_tensor_descriptor: fake_set_op_tensor_descriptor,
                cnnl_op_tensor: fake_op_tensor,
            },
            authority_lifetime: PhantomData,
        }
    }

    struct FakeSystem {
        trace: Vec<String>,
        missing_symbol: Option<String>,
        versions: BTreeMap<String, LibraryVersion>,
    }

    impl FakeSystem {
        fn passing() -> Self {
            Self {
                trace: Vec::new(),
                missing_symbol: None,
                versions: BTreeMap::from([
                    (
                        "cnrt".to_owned(),
                        LibraryVersion {
                            major: 6,
                            minor: 6,
                            patch: 0,
                        },
                    ),
                    (
                        "cnnl".to_owned(),
                        LibraryVersion {
                            major: 1,
                            minor: 20,
                            patch: 4,
                        },
                    ),
                ]),
            }
        }
    }

    impl ProbeSystem for FakeSystem {
        type Handle = String;

        fn open(
            &mut self,
            library: &LibraryContract,
            _image: &RegistryCertifiedImage,
        ) -> Result<Self::Handle, MluLoadError> {
            self.trace.push(format!("open:{}", library.id));
            Ok(library.id.clone())
        }

        fn require_symbol(
            &mut self,
            _handle: &Self::Handle,
            library: &LibraryContract,
            symbol: &str,
        ) -> Result<(), MluLoadError> {
            self.trace.push(format!("symbol:{}:{symbol}", library.id));
            if self.missing_symbol.as_deref() == Some(symbol) {
                Err(MluLoadError::MissingSymbol {
                    library: library.id.clone(),
                    symbol: symbol.to_owned(),
                })
            } else {
                Ok(())
            }
        }

        fn version(
            &mut self,
            _handle: &Self::Handle,
            library: &LibraryContract,
        ) -> Result<LibraryVersion, MluLoadError> {
            self.trace.push(format!("version:{}", library.id));
            self.versions
                .get(&library.id)
                .copied()
                .ok_or_else(|| MluLoadError::VersionCall {
                    library: library.id.clone(),
                    status: -1,
                })
        }
    }

    fn image(library: &LibraryContract, descriptor: u32) -> RegistryCertifiedImage {
        RegistryCertifiedImage {
            library_id: library.id.clone(),
            digest_sha256: "a".repeat(64),
            abi_version: ABI_FLOOR.to_owned(),
            required_symbols: library
                .symbols
                .iter()
                .map(|symbol| symbol.name.clone())
                .collect(),
            unsafe_owner: UNSAFE_OWNER.to_owned(),
            retained_image_path: PathBuf::from(format!("/proc/self/fd/{descriptor}")),
        }
    }

    fn images(certificate_session: &()) -> Result<CertifiedMluImages<'_>, MluLoadError> {
        let manifest =
            AbiManifest::embedded().map_err(|error| MluLoadError::Manifest(error.to_string()))?;
        let rows = manifest
            .libraries
            .iter()
            .enumerate()
            .map(|(index, library)| image(library, index as u32 + 10));
        unsafe { CertifiedMluImages::from_registry_certificates(certificate_session, rows) }
    }

    #[test]
    fn discovery_order_is_deterministic_and_deduplicated() -> Result<(), MluLoadError> {
        let plan = DiscoveryPlan::from_sources(
            Some(PathBuf::from("/sim")),
            Some(PathBuf::from("/neuware")),
            [PathBuf::from("/signed-a"), PathBuf::from("/sim")],
        )?;
        let candidates = plan.candidates();
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].0, "SIM_MLU_ROOT");
        assert_eq!(candidates[1].0, "NEUWARE_HOME");
        assert_eq!(candidates[2].0, "signed_package_roots");
        assert_eq!(candidates[0].1, PathBuf::from("/sim/lib64/libcnrt.so"));
        Ok(())
    }

    #[test]
    fn ordinary_paths_and_forged_symbol_sets_fail_closed() -> Result<(), Box<dyn std::error::Error>>
    {
        let manifest = AbiManifest::embedded()?;
        let certified_rows = || {
            manifest
                .libraries
                .iter()
                .enumerate()
                .map(|(index, library)| image(library, index as u32 + 10))
                .collect::<Vec<_>>()
        };
        let certificate_session = ();
        let missing_library = unsafe {
            CertifiedMluImages::from_registry_certificates(
                &certificate_session,
                certified_rows().into_iter().take(1),
            )
        };
        assert!(matches!(
            missing_library,
            Err(MluLoadError::MissingCertifiedLibrary { library }) if library == "cnrt"
        ));

        let mut rows = certified_rows();
        rows[0].retained_image_path = PathBuf::from("/opt/neuware/lib64/libcnnl.so");
        let ordinary =
            unsafe { CertifiedMluImages::from_registry_certificates(&certificate_session, rows) };
        assert!(matches!(
            ordinary,
            Err(MluLoadError::UnsealedImagePath { .. })
        ));

        let mut rows = certified_rows();
        rows[0].required_symbols.remove("cnnlCreate");
        let missing =
            unsafe { CertifiedMluImages::from_registry_certificates(&certificate_session, rows) };
        assert!(
            matches!(missing, Err(MluLoadError::CertificateMissingSymbol { symbol, .. }) if symbol == "cnnlCreate")
        );

        let mut rows = certified_rows();
        rows[0]
            .required_symbols
            .insert("unreviewedSymbol".to_owned());
        let extra =
            unsafe { CertifiedMluImages::from_registry_certificates(&certificate_session, rows) };
        assert!(matches!(
            extra,
            Err(MluLoadError::CertificateMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn probe_loads_dependencies_first_and_checks_every_symbol() -> Result<(), MluLoadError> {
        let certificate_session = ();
        let images = images(&certificate_session)?;
        let mut system = FakeSystem::passing();
        let probe = probe_with_system(&mut system, "x86_64-unknown-linux-gnu", &images)?;
        assert_eq!(probe.symbol_count, REVIEWED_SYMBOL_COUNT);
        assert_eq!(system.trace.first().map(String::as_str), Some("open:cnrt"));
        assert!(
            system.trace.iter().position(|row| row == "open:cnnl")
                > system.trace.iter().position(|row| row == "version:cnrt")
        );
        Ok(())
    }

    #[test]
    fn missing_symbol_and_incompatible_version_are_typed() -> Result<(), MluLoadError> {
        let certificate_session = ();
        let images = images(&certificate_session)?;
        let mut missing = FakeSystem::passing();
        missing.missing_symbol = Some("cnrtMalloc".to_owned());
        assert!(matches!(
            probe_with_system(&mut missing, "aarch64-unknown-linux-gnu", &images),
            Err(MluLoadError::MissingSymbol { symbol, .. }) if symbol == "cnrtMalloc"
        ));

        let mut old = FakeSystem::passing();
        old.versions.insert(
            "cnnl".to_owned(),
            LibraryVersion {
                major: 1,
                minor: 19,
                patch: 9,
            },
        );
        assert!(matches!(
            probe_with_system(&mut old, "x86_64-unknown-linux-gnu", &images),
            Err(MluLoadError::Version { library, actual, .. }) if library == "cnnl" && actual == "1.19.9"
        ));
        Ok(())
    }

    #[test]
    fn unsupported_target_is_reported_before_loading() -> Result<(), MluLoadError> {
        let certificate_session = ();
        let images = images(&certificate_session)?;
        let mut system = FakeSystem::passing();
        assert!(matches!(
            probe_with_system(&mut system, "x86_64-apple-darwin", &images),
            Err(MluLoadError::UnsupportedTarget { target }) if target == "x86_64-apple-darwin"
        ));
        assert!(system.trace.is_empty());
        Ok(())
    }

    #[test]
    fn focused_call_surface_owns_resource_lifetimes_and_vendor_calls() -> Result<(), MluLoadError> {
        QUEUE_SYNCHRONIZATIONS.store(0, Ordering::SeqCst);
        ALLOCATION_RELEASES.store(0, Ordering::SeqCst);
        CNNL_CONTEXT_RELEASES.store(0, Ordering::SeqCst);
        DESCRIPTOR_RELEASES.store(0, Ordering::SeqCst);
        OP_DESCRIPTOR_RELEASES.store(0, Ordering::SeqCst);
        OPERATOR_CALLS.store(0, Ordering::SeqCst);
        let calls = fake_call_surface();
        assert_eq!(
            calls.probe("x86_64-unknown-linux-gnu")?.symbol_count,
            REVIEWED_SYMBOL_COUNT
        );
        assert_eq!(calls.device_count()?, 2);
        calls.set_device(0)?;
        {
            let mut destination = calls.allocate(32)?;
            let source = calls.allocate(32)?;
            calls.copy(
                &mut destination,
                &source,
                32,
                CnrtMemTransferDirection::DeviceToDevice,
            )?;
            let queue = calls.create_queue()?;
            queue.synchronize()?;
            let mut context = calls.create_cnnl_context()?;
            {
                let binding = context.bind_queue(&queue)?;
                let left_descriptor = binding.create_f32_tensor_descriptor(&[2, 4])?;
                let right_descriptor = binding.create_f32_tensor_descriptor(&[2, 4])?;
                let output_descriptor = binding.create_f32_tensor_descriptor(&[2, 4])?;
                let operation = binding.create_f32_add_descriptor()?;
                binding.add_f32(
                    &operation,
                    &left_descriptor,
                    &source,
                    &right_descriptor,
                    &source,
                    &output_descriptor,
                    &mut destination,
                )?;
            }
        }
        assert_eq!(QUEUE_SYNCHRONIZATIONS.load(Ordering::SeqCst), 1);
        assert_eq!(ALLOCATION_RELEASES.load(Ordering::SeqCst), 2);
        assert_eq!(CNNL_CONTEXT_RELEASES.load(Ordering::SeqCst), 1);
        assert_eq!(DESCRIPTOR_RELEASES.load(Ordering::SeqCst), 3);
        assert_eq!(OP_DESCRIPTOR_RELEASES.load(Ordering::SeqCst), 1);
        assert_eq!(OPERATOR_CALLS.load(Ordering::SeqCst), 1);
        assert!(matches!(
            calls.allocate(0),
            Err(MluLoadError::InvalidArgument {
                operation: "cnrtMalloc",
                ..
            })
        ));
        Ok(())
    }
}
