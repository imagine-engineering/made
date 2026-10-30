use crate::{
    CertifiedNativeFfi, CudaPackageVerificationKey, NativeFfiContract, NativeFfiRegistry,
    TrustError,
    trust::{
        NativeLibraryImageError, NativePackageAdmissionError, NativePackagePayloadLimit,
        RetainedNativeLibraryImage, capture_native_library_image, capture_native_package,
        validate_native_package_coverage,
    },
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_backend_cuda::{
    ABI_FLOOR, AbiManifest, CudaExecutionError, CudaExecutionSession, CudaLibraryCandidates,
    DiscoveryEnvironment, RegistryCertifiedCudaImages, SignedPackageRoot, discovery_candidates,
};
use sim_model::ArtifactRoot;
use sim_types::{BackendUnavailable, CancellationError, CancellationToken, DeviceKind};
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
use std::os::windows::ffi::OsStrExt;
use std::{collections::BTreeMap, path::Path, ptr::NonNull, sync::Arc};
#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use std::{ffi::CString, os::unix::ffi::OsStrExt};
use thiserror::Error;

const PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/cuda/package-policy.json");
const ABI_MANIFEST: &str = sim_backend_cuda::ABI_MANIFEST_JSON;
const KERNEL_MANIFEST: &str = include_str!("../../sim_backend_cuda/kernels/manifest-v1.json");
const CORE_PTX: &[u8] = include_bytes!("../../sim_backend_cuda/kernels/core-v1.ptx");
const UNSAFE_OWNER: &str = "sim_backend_cuda::loader";
const MAX_PACKAGE_FILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_COVERAGE_BYTES: usize = 64 * 1024;
const MAX_PACKAGE_PAYLOADS: usize = 16;
const MAX_PACKAGE_BYTES: usize = 16 * 1024 * 1024;
const PACKAGE_PAYLOAD_LIMITS: [NativePackagePayloadLimit; 9] = [
    NativePackagePayloadLimit::new("LICENSES", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("abi/symbols-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.sig", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("ffi-contracts-v1.json", MAX_CATALOG_BYTES),
    NativePackagePayloadLimit::new("kernels/core-v1.ptx", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("kernels/manifest-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("package-coverage.sha256", MAX_COVERAGE_BYTES),
    NativePackagePayloadLimit::new("package-policy.json", MAX_PACKAGE_FILE_BYTES),
];
const COVERAGE_EXCLUDES: [&str; 2] = ["adapter-manifest.sig", "package-coverage.sha256"];
const SUPPORTED_TARGETS: [&str; 3] = [
    "aarch64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CudaFfiContractCatalogDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    abi_manifest_sha256: String,
    package_policy_sha256: String,
    libraries: Vec<CudaFfiContractDto>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CudaFfiContractDto {
    identity: String,
    filename: String,
    sha256: String,
    abi: String,
    required_symbols: Vec<String>,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CudaPackageManifestDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    libraries: Vec<String>,
    discovery_order: Vec<String>,
    abi_manifest_sha256: String,
    kernel_manifest_sha256: String,
    core_ptx_sha256: String,
    ffi_contracts_sha256: String,
    package_policy_sha256: String,
    redistributes_driver: bool,
    approved_vendor_redistributables: Vec<String>,
    license_approval_required_for_vendor_runtime: bool,
    runtime_compilation_for_core_kernels: bool,
    signer: String,
    signature_algorithm: String,
    signature_domain: String,
    signature_coverage: String,
    certificate_owner: String,
    unsafe_owner: String,
    structural_receipt_is_authorization: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaFfiLibraryIdentity {
    filename: String,
}

impl CudaFfiLibraryIdentity {
    pub fn filename(&self) -> &str {
        &self.filename
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedCudaFfiContracts {
    package_root: ArtifactRoot,
    target: String,
    registry: NativeFfiRegistry,
    identities: BTreeMap<String, CudaFfiLibraryIdentity>,
}

impl VerifiedCudaFfiContracts {
    pub fn package_root(&self) -> &ArtifactRoot {
        &self.package_root
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn registry(&self) -> &NativeFfiRegistry {
        &self.registry
    }

    pub fn identities(&self) -> &BTreeMap<String, CudaFfiLibraryIdentity> {
        &self.identities
    }
}

#[derive(Debug, Error)]
pub enum CudaPackageContractError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("CUDA package tree is unsafe or incomplete: {0}")]
    UnsafePackage(String),
    #[error("signed CUDA package metadata is invalid: {0}")]
    InvalidPackage(String),
    #[error("signed CUDA FFI contract catalog is invalid: {0}")]
    InvalidCatalog(String),
}

#[derive(Debug, Error)]
pub enum CudaCertificationError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("no complete CUDA library root was discovered")]
    MissingLibraryRoot,
    #[error("CUDA library root is incomplete or unsafe: {0}")]
    UnsafeLibraryRoot(String),
    #[error("CUDA library image {library_id} is invalid: {reason}")]
    InvalidImage { library_id: String, reason: String },
    #[error("sealed CUDA images require a supported x86_64 Linux or Windows target")]
    UnsupportedPlatform,
}

struct CudaCertificationRetention {
    _verified: VerifiedCudaFfiContracts,
    _certificates: Vec<CertifiedNativeFfi>,
    _sealed_images: BTreeMap<String, RetainedNativeLibraryImage>,
    _library_handles: NativeCudaLibraryHandles,
}

struct NativeCudaLibraryHandles {
    driver: NonNull<std::ffi::c_void>,
    nvrtc: NonNull<std::ffi::c_void>,
    cublaslt: NonNull<std::ffi::c_void>,
    cudnn: NonNull<std::ffi::c_void>,
}

unsafe impl Send for NativeCudaLibraryHandles {}
unsafe impl Sync for NativeCudaLibraryHandles {}

impl Drop for NativeCudaLibraryHandles {
    fn drop(&mut self) {
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        unsafe {
            libc::dlclose(self.cudnn.as_ptr());
            libc::dlclose(self.cublaslt.as_ptr());
            libc::dlclose(self.nvrtc.as_ptr());
            libc::dlclose(self.driver.as_ptr());
        }
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        unsafe {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
            }
            for (library, handle) in [
                ("cuDNN", self.cudnn),
                ("cuBLASLt", self.cublaslt),
                ("NVRTC", self.nvrtc),
                ("CUDA driver", self.driver),
            ] {
                if FreeLibrary(handle.as_ptr()) == 0 {
                    eprintln!(
                        "failed to release the retained {library} image: {}",
                        std::io::Error::last_os_error()
                    );
                }
            }
        }
    }
}

pub struct CertifiedCudaLoad {
    retention: Arc<CudaCertificationRetention>,
    certificates: Vec<CertifiedNativeFfi>,
}

impl CertifiedCudaLoad {
    pub fn certificates(&self) -> &[CertifiedNativeFfi] {
        &self.certificates
    }

    pub fn load_execution_runtime(
        self,
        device_ordinal: usize,
    ) -> Result<CudaExecutionSession, CudaExecutionError> {
        let certification: Arc<dyn std::any::Any + Send + Sync> = self.retention.clone();
        let images = unsafe {
            RegistryCertifiedCudaImages::from_registry_certified_images(
                certification,
                self.retention._library_handles.driver.as_ptr(),
                self.retention._library_handles.nvrtc.as_ptr(),
                self.retention._library_handles.cublaslt.as_ptr(),
                self.retention._library_handles.cudnn.as_ptr(),
                CORE_PTX,
                sim_backend_cuda::CORE_PTX_SHA256,
            )
        }?;
        CudaExecutionSession::from_registry_certified_images(images, device_ordinal)
    }
}

pub fn verify_cuda_package_contracts(
    package_root: &Path,
    verification_key: &CudaPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedCudaFfiContracts, CudaPackageContractError> {
    cancellation.check()?;
    let root = ArtifactRoot::canonical(
        "sim-cuda-package",
        "native-ffi-package",
        package_root,
        std::iter::empty::<String>(),
    )
    .map_err(|error| CudaPackageContractError::UnsafePackage(error.to_string()))?;
    let payloads = capture_native_package(
        &root,
        &PACKAGE_PAYLOAD_LIMITS,
        MAX_PACKAGE_PAYLOADS,
        MAX_PACKAGE_BYTES,
        cancellation,
    )
    .map_err(map_package_admission_error)?;

    let coverage = required_payload(&payloads, "package-coverage.sha256")?;
    validate_native_package_coverage(coverage, &payloads, &COVERAGE_EXCLUDES, MAX_COVERAGE_BYTES)
        .map_err(map_package_admission_error)?;
    let packaged_policy: Value = parse_strict_json(
        required_payload(&payloads, "package-policy.json")?,
        "package policy",
    )?;
    let reviewed_policy: Value = parse_strict_json(PACKAGE_POLICY.as_bytes(), "reviewed policy")?;
    if packaged_policy != reviewed_policy {
        return Err(CudaPackageContractError::InvalidPackage(
            "installed package policy differs from the compiled reviewed policy".to_owned(),
        ));
    }
    let manifest: CudaPackageManifestDto = parse_strict_json(
        required_payload(&payloads, "adapter-manifest.json")?,
        "adapter manifest",
    )?;
    validate_manifest(&manifest, &payloads)?;
    verification_key.verify_package(
        &manifest.signer,
        coverage,
        required_payload(&payloads, "adapter-manifest.sig")?,
    )?;
    cancellation.check()?;

    let catalog: CudaFfiContractCatalogDto = parse_strict_json(
        required_payload(&payloads, "ffi-contracts-v1.json")?,
        "FFI contract catalog",
    )?;
    let (registry, identities) = validate_and_map_catalog(&catalog, &manifest)?;
    cancellation.check()?;
    Ok(VerifiedCudaFfiContracts {
        package_root: root,
        target: catalog.target,
        registry,
        identities,
    })
}

pub fn certify_cuda_library_images(
    verified: VerifiedCudaFfiContracts,
    candidates: &[CudaLibraryCandidates],
    cancellation: &CancellationToken,
) -> Result<CertifiedCudaLoad, CudaCertificationError> {
    cancellation.check()?;
    if !host_target_matches(verified.target()) {
        return Err(CudaCertificationError::UnsafeLibraryRoot(format!(
            "signed target {} does not match this process",
            verified.target()
        )));
    }
    certify_cuda_library_images_after_target_check(verified, candidates, cancellation)
}

fn certify_cuda_library_images_after_target_check(
    verified: VerifiedCudaFfiContracts,
    candidates: &[CudaLibraryCandidates],
    cancellation: &CancellationToken,
) -> Result<CertifiedCudaLoad, CudaCertificationError> {
    let paths = candidates
        .iter()
        .find_map(|candidate| {
            (candidate.libraries.len() == verified.identities().len()
                && verified.identities().keys().all(|library_id| {
                    candidate
                        .libraries
                        .get(library_id)
                        .is_some_and(|path| path.is_absolute() && path.is_file())
                }))
            .then(|| candidate.libraries.clone())
        })
        .ok_or(CudaCertificationError::MissingLibraryRoot)?;
    let mut certificates = Vec::with_capacity(paths.len());
    let mut retained_files = BTreeMap::new();
    for (library_id, identity) in verified.identities() {
        cancellation.check()?;
        let path = paths
            .get(library_id)
            .ok_or_else(|| CudaCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "catalog identity has no reviewed discovery path".to_owned(),
            })?;
        if path.file_name().and_then(|name| name.to_str()) != Some(identity.filename()) {
            return Err(CudaCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "discovered filename differs from the signed contract".to_owned(),
            });
        }
        let captured = capture_native_library_image(path, cancellation)
            .map_err(|error| map_native_library_image_error(library_id, error))?;
        let symbols =
            verified
                .registry()
                .required_symbols_for(library_id, ABI_FLOOR, UNSAFE_OWNER)?;
        let certificate = verified.registry().authorize(
            library_id,
            captured.digest_sha256(),
            ABI_FLOOR,
            &symbols,
        )?;
        let retained = captured
            .seal(&format!("cuda-{library_id}"), cancellation)
            .map_err(|error| map_native_library_image_error(library_id, error))?;
        certificates.push(certificate);
        retained_files.insert(library_id.clone(), retained);
    }
    let library_handles = load_retained_cuda_images(&retained_files)?;
    let retained_certificates = certificates.clone();
    let retention = Arc::new(CudaCertificationRetention {
        _verified: verified,
        _certificates: retained_certificates,
        _sealed_images: retained_files,
        _library_handles: library_handles,
    });
    Ok(CertifiedCudaLoad {
        retention,
        certificates,
    })
}

fn load_retained_cuda_images(
    retained_images: &BTreeMap<String, RetainedNativeLibraryImage>,
) -> Result<NativeCudaLibraryHandles, CudaCertificationError> {
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        let _retained_images = retained_images;
        Err(CudaCertificationError::UnsupportedPlatform)
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    {
        let driver = required_retained_image(retained_images, "driver")?;
        let nvrtc = required_retained_image(retained_images, "nvrtc")?;
        let cublaslt = required_retained_image(retained_images, "cublaslt")?;
        let cudnn = required_retained_image(retained_images, "cudnn")?;
        let driver_handle =
            open_retained_image(driver.loader_path(), libc::RTLD_NOW | libc::RTLD_GLOBAL)?;
        let nvrtc_handle = open_retained_image_after(
            nvrtc.loader_path(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
            &[driver_handle],
        )?;
        let cublaslt_handle = open_retained_image_after(
            cublaslt.loader_path(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
            &[nvrtc_handle, driver_handle],
        )?;
        let cudnn_handle = open_retained_image_after(
            cudnn.loader_path(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
            &[cublaslt_handle, nvrtc_handle, driver_handle],
        )?;
        Ok(NativeCudaLibraryHandles {
            driver: driver_handle,
            nvrtc: nvrtc_handle,
            cublaslt: cublaslt_handle,
            cudnn: cudnn_handle,
        })
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        let driver_handle = open_retained_windows_image(
            required_retained_image(retained_images, "driver")?.loader_path(),
        )?;
        let nvrtc_handle = open_retained_windows_image_after(
            required_retained_image(retained_images, "nvrtc")?.loader_path(),
            &[driver_handle],
        )?;
        let cublaslt_handle = open_retained_windows_image_after(
            required_retained_image(retained_images, "cublaslt")?.loader_path(),
            &[nvrtc_handle, driver_handle],
        )?;
        let cudnn_handle = open_retained_windows_image_after(
            required_retained_image(retained_images, "cudnn")?.loader_path(),
            &[cublaslt_handle, nvrtc_handle, driver_handle],
        )?;
        Ok(NativeCudaLibraryHandles {
            driver: driver_handle,
            nvrtc: nvrtc_handle,
            cublaslt: cublaslt_handle,
            cudnn: cudnn_handle,
        })
    }
}

#[cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ),
    all(target_os = "windows", target_arch = "x86_64")
))]
fn required_retained_image<'a>(
    images: &'a BTreeMap<String, RetainedNativeLibraryImage>,
    library_id: &str,
) -> Result<&'a RetainedNativeLibraryImage, CudaCertificationError> {
    images.get(library_id).ok_or_else(|| {
        CudaCertificationError::UnsafeLibraryRoot(format!(
            "certified {library_id} image is missing"
        ))
    })
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn open_retained_image(
    path: &Path,
    flags: libc::c_int,
) -> Result<NonNull<std::ffi::c_void>, CudaCertificationError> {
    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        CudaCertificationError::UnsafeLibraryRoot(
            "sealed CUDA loader path contains an interior NUL".to_owned(),
        )
    })?;
    let handle = unsafe { libc::dlopen(path.as_ptr(), flags) };
    NonNull::new(handle).ok_or_else(|| {
        CudaCertificationError::UnsafeLibraryRoot(
            "the exact sealed CUDA image could not be loaded".to_owned(),
        )
    })
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn open_retained_image_after(
    path: &Path,
    flags: libc::c_int,
    previously_opened: &[NonNull<std::ffi::c_void>],
) -> Result<NonNull<std::ffi::c_void>, CudaCertificationError> {
    match open_retained_image(path, flags) {
        Ok(handle) => Ok(handle),
        Err(error) => {
            for handle in previously_opened {
                unsafe {
                    libc::dlclose(handle.as_ptr());
                }
            }
            Err(error)
        }
    }
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn open_retained_windows_image(
    path: &Path,
) -> Result<NonNull<std::ffi::c_void>, CudaCertificationError> {
    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(
            file_name: *const u16,
            file: *mut std::ffi::c_void,
            flags: u32,
        ) -> *mut std::ffi::c_void;
    }

    let mut wide_path = path.as_os_str().encode_wide().collect::<Vec<_>>();
    if wide_path.contains(&0) {
        return Err(CudaCertificationError::UnsafeLibraryRoot(
            "sealed CUDA loader path contains an interior NUL".to_owned(),
        ));
    }
    wide_path.push(0);
    let handle = unsafe {
        LoadLibraryExW(
            wide_path.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    NonNull::new(handle).ok_or_else(|| {
        CudaCertificationError::UnsafeLibraryRoot(format!(
            "the exact sealed CUDA image could not be loaded: {}",
            std::io::Error::last_os_error()
        ))
    })
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn open_retained_windows_image_after(
    path: &Path,
    previously_opened: &[NonNull<std::ffi::c_void>],
) -> Result<NonNull<std::ffi::c_void>, CudaCertificationError> {
    match open_retained_windows_image(path) {
        Ok(handle) => Ok(handle),
        Err(error) => {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
            }
            for handle in previously_opened {
                unsafe {
                    if FreeLibrary(handle.as_ptr()) == 0 {
                        eprintln!(
                            "failed to release a retained CUDA image after load failure: {}",
                            std::io::Error::last_os_error()
                        );
                    }
                }
            }
            Err(error)
        }
    }
}

pub fn initialize_certified_cuda_runtime(
    settings: &crate::NativeCudaPackageSettings,
    device_ordinal: usize,
    cancellation: &CancellationToken,
) -> Result<CudaExecutionSession, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Cuda, reason);
    let verified = verify_cuda_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed package or contract verification failed"))?;
    let signed_root = unsafe {
        SignedPackageRoot::from_runtime_verified_path(
            &verified,
            verified.package_root().canonical_path().to_path_buf(),
        )
    }
    .map_err(|_| unavailable("signed CUDA package root projection failed"))?;
    let candidates = discovery_candidates(
        verified.target(),
        &DiscoveryEnvironment::from_process(),
        &[signed_root],
    )
    .map_err(|_| unavailable("CUDA discovery plan is invalid"))?;
    initialize_certified_cuda_runtime_from_verified(
        verified,
        &candidates,
        device_ordinal,
        cancellation,
    )
}

pub fn initialize_certified_cuda_runtime_with_candidates(
    settings: &crate::NativeCudaPackageSettings,
    candidates: &[CudaLibraryCandidates],
    device_ordinal: usize,
    cancellation: &CancellationToken,
) -> Result<CudaExecutionSession, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Cuda, reason);
    let verified = verify_cuda_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed package or contract verification failed"))?;
    initialize_certified_cuda_runtime_from_verified(
        verified,
        candidates,
        device_ordinal,
        cancellation,
    )
}

fn initialize_certified_cuda_runtime_from_verified(
    verified: VerifiedCudaFfiContracts,
    candidates: &[CudaLibraryCandidates],
    device_ordinal: usize,
    cancellation: &CancellationToken,
) -> Result<CudaExecutionSession, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Cuda, reason);
    let certified = certify_cuda_library_images(verified, candidates, cancellation)
        .map_err(|_| unavailable("exact CUDA library certification failed"))?;
    certified
        .load_execution_runtime(device_ordinal)
        .map_err(|_| unavailable("certified CUDA loader or ABI probe failed"))
}

fn validate_manifest(
    manifest: &CudaPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), CudaPackageContractError> {
    let target_supported = SUPPORTED_TARGETS.contains(&manifest.target.as_str());
    if manifest.schema_version != 1
        || manifest.backend != "cuda"
        || manifest.abi_floor != ABI_FLOOR
        || !target_supported
        || manifest.libraries != ["cublaslt", "cudnn", "driver", "nvrtc"]
        || manifest.discovery_order
            != [
                "SIM_CUDA_ROOT",
                "CUDA_PATH",
                "signed_package_roots",
                "installed_driver_library",
            ]
        || manifest.abi_manifest_sha256
            != sha256_hex(required_payload(payloads, "abi/symbols-v1.json")?)
        || manifest.ffi_contracts_sha256
            != sha256_hex(required_payload(payloads, "ffi-contracts-v1.json")?)
        || manifest.kernel_manifest_sha256
            != sha256_hex(required_payload(payloads, "kernels/manifest-v1.json")?)
        || manifest.core_ptx_sha256
            != sha256_hex(required_payload(payloads, "kernels/core-v1.ptx")?)
        || manifest.package_policy_sha256
            != sha256_hex(required_payload(payloads, "package-policy.json")?)
        || manifest.redistributes_driver
        || !manifest.approved_vendor_redistributables.is_empty()
        || !manifest.license_approval_required_for_vendor_runtime
        || manifest.runtime_compilation_for_core_kernels
        || manifest.signature_algorithm != "ed25519"
        || manifest.signature_domain != "zed-sim-cuda-package-v1"
        || manifest.signature_coverage != "package-coverage-v1"
        || manifest.certificate_owner != "sim_runtime::NativeFfiRegistry"
        || manifest.unsafe_owner != UNSAFE_OWNER
        || manifest.structural_receipt_is_authorization
        || !valid_signer(&manifest.signer)
    {
        return Err(CudaPackageContractError::InvalidPackage(
            "adapter manifest differs from the reviewed CUDA package contract".to_owned(),
        ));
    }
    if required_payload(payloads, "abi/symbols-v1.json")? != ABI_MANIFEST.as_bytes() {
        return Err(CudaPackageContractError::InvalidPackage(
            "packaged ABI manifest differs from the compiled reviewed manifest".to_owned(),
        ));
    }
    if required_payload(payloads, "kernels/manifest-v1.json")? != KERNEL_MANIFEST.as_bytes()
        || required_payload(payloads, "kernels/core-v1.ptx")? != CORE_PTX
        || manifest.core_ptx_sha256 != sim_backend_cuda::CORE_PTX_SHA256
    {
        return Err(CudaPackageContractError::InvalidPackage(
            "packaged reviewed CUDA kernels differ from the compiled evidence".to_owned(),
        ));
    }
    Ok(())
}

fn validate_and_map_catalog(
    catalog: &CudaFfiContractCatalogDto,
    manifest: &CudaPackageManifestDto,
) -> Result<(NativeFfiRegistry, BTreeMap<String, CudaFfiLibraryIdentity>), CudaPackageContractError>
{
    let abi = AbiManifest::embedded()
        .map_err(|error| CudaPackageContractError::InvalidCatalog(error.to_string()))?;
    if catalog.schema_version != 1
        || catalog.backend != "cuda"
        || catalog.abi_floor != ABI_FLOOR
        || catalog.target != manifest.target
        || catalog.abi_manifest_sha256 != manifest.abi_manifest_sha256
        || catalog.package_policy_sha256 != manifest.package_policy_sha256
        || catalog.libraries.len() != abi.libraries.len()
    {
        return Err(CudaPackageContractError::InvalidCatalog(
            "catalog envelope is unsupported, target-mismatched, or incomplete".to_owned(),
        ));
    }
    let mut previous_identity: Option<&str> = None;
    let mut contracts = Vec::with_capacity(catalog.libraries.len());
    let mut identities = BTreeMap::new();
    for row in &catalog.libraries {
        if previous_identity.is_some_and(|previous| previous >= row.identity.as_str()) {
            return Err(CudaPackageContractError::InvalidCatalog(
                "library identities must be sorted and unique".to_owned(),
            ));
        }
        previous_identity = Some(&row.identity);
        let reviewed = abi
            .libraries
            .iter()
            .find(|library| library.id == row.identity)
            .ok_or_else(|| {
                CudaPackageContractError::InvalidCatalog(format!(
                    "unknown library identity {}",
                    row.identity
                ))
            })?;
        let mut expected_symbols = reviewed
            .symbols
            .iter()
            .map(|symbol| symbol.name.clone())
            .collect::<Vec<_>>();
        expected_symbols.sort();
        let expected_filename = if catalog.target.contains("windows") {
            &reviewed.windows_filename
        } else {
            &reviewed.linux_filename
        };
        if &row.filename != expected_filename
            || !valid_lower_hex_digest(&row.sha256)
            || row.abi != ABI_FLOOR
            || row.required_symbols != expected_symbols
            || row.required_symbols.is_empty()
            || row
                .required_symbols
                .windows(2)
                .any(|pair| matches!(pair, [first, second] if first >= second))
            || row.unsafe_owner != UNSAFE_OWNER
        {
            return Err(CudaPackageContractError::InvalidCatalog(format!(
                "library contract {} differs from the reviewed ABI",
                row.identity
            )));
        }
        contracts.push(NativeFfiContract::new(
            row.identity.clone(),
            row.sha256.clone(),
            row.abi.clone(),
            row.required_symbols.clone(),
            row.unsafe_owner.clone(),
        )?);
        identities.insert(
            row.identity.clone(),
            CudaFfiLibraryIdentity {
                filename: row.filename.clone(),
            },
        );
    }
    if identities.len() != abi.libraries.len() {
        return Err(CudaPackageContractError::InvalidCatalog(
            "catalog does not cover every reviewed library".to_owned(),
        ));
    }
    Ok((NativeFfiRegistry::new(contracts)?, identities))
}

fn map_package_admission_error(error: NativePackageAdmissionError) -> CudaPackageContractError {
    match error {
        NativePackageAdmissionError::Cancelled => CancellationError.into(),
        NativePackageAdmissionError::UnsafePackage(reason) => {
            CudaPackageContractError::UnsafePackage(reason)
        }
        NativePackageAdmissionError::InvalidCoverage(reason) => {
            CudaPackageContractError::InvalidPackage(reason)
        }
    }
}

fn required_payload<'a>(
    payloads: &'a BTreeMap<String, Vec<u8>>,
    path: &str,
) -> Result<&'a [u8], CudaPackageContractError> {
    payloads.get(path).map(Vec::as_slice).ok_or_else(|| {
        CudaPackageContractError::UnsafePackage(format!("required payload is missing: {path}"))
    })
}

fn parse_strict_json<T: DeserializeOwned>(
    bytes: &[u8],
    label: &str,
) -> Result<T, CudaPackageContractError> {
    let strict = crate::trust::parse_strict_json_value(bytes).map_err(|error| {
        CudaPackageContractError::InvalidPackage(format!("{label} is not strict JSON: {error}"))
    })?;
    serde_json::from_value(strict).map_err(|error| {
        CudaPackageContractError::InvalidPackage(format!("{label} is invalid: {error}"))
    })
}

fn valid_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn valid_signer(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn host_target_matches(target: &str) -> bool {
    matches!(
        (std::env::consts::ARCH, std::env::consts::OS, target),
        ("aarch64", "linux", "aarch64-unknown-linux-gnu")
            | ("x86_64", "linux", "x86_64-unknown-linux-gnu")
            | ("x86_64", "windows", "x86_64-pc-windows-msvc")
    )
}

fn map_native_library_image_error(
    library_id: &str,
    error: NativeLibraryImageError,
) -> CudaCertificationError {
    match error {
        NativeLibraryImageError::Cancelled => CudaCertificationError::Cancelled(CancellationError),
        NativeLibraryImageError::UnsupportedPlatform => CudaCertificationError::UnsupportedPlatform,
        NativeLibraryImageError::Invalid(reason) => CudaCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use serde_json::{Value, json};
    use std::{fs, io};

    const SIGNING_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
    const FIXTURE_DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn signed_receipt(
        signer: &str,
        coverage: &[u8],
        key_pair: &Ed25519KeyPair,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let payload = crate::cuda_package_signing_payload(signer, coverage)?;
        Ok(format!(
            "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{}\"}}\n",
            key_pair
                .sign(&payload)
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
        .into_bytes())
    }

    fn fixture_catalog(target: &str) -> Value {
        let abi = AbiManifest::embedded().expect("embedded CUDA ABI is valid");
        let libraries = abi
            .libraries
            .iter()
            .map(|library| {
                let mut required_symbols = library
                    .symbols
                    .iter()
                    .map(|symbol| symbol.name.clone())
                    .collect::<Vec<_>>();
                required_symbols.sort();
                json!({
                    "identity": library.id,
                    "filename": if target.contains("windows") {
                        &library.windows_filename
                    } else {
                        &library.linux_filename
                    },
                    "sha256": FIXTURE_DIGEST,
                    "abi": ABI_FLOOR,
                    "required_symbols": required_symbols,
                    "unsafe_owner": UNSAFE_OWNER,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": 1,
            "backend": "cuda",
            "abi_floor": ABI_FLOOR,
            "target": target,
            "abi_manifest_sha256": sha256_hex(ABI_MANIFEST.as_bytes()),
            "package_policy_sha256": sha256_hex(PACKAGE_POLICY.as_bytes()),
            "libraries": libraries,
        })
    }

    fn write_fixture_package(
        root: &Path,
        manifest_target: &str,
        catalog: &Value,
        key_pair: &Ed25519KeyPair,
    ) -> Result<CudaPackageVerificationKey, Box<dyn std::error::Error>> {
        fs::create_dir_all(root.join("abi"))?;
        fs::create_dir_all(root.join("kernels"))?;
        fs::write(root.join("LICENSES"), sim_backend_cuda::PACKAGE_LICENSES)?;
        fs::write(root.join("abi/symbols-v1.json"), ABI_MANIFEST)?;
        fs::write(root.join("kernels/manifest-v1.json"), KERNEL_MANIFEST)?;
        fs::write(root.join("kernels/core-v1.ptx"), CORE_PTX)?;
        fs::write(root.join("package-policy.json"), PACKAGE_POLICY)?;
        let mut catalog_bytes = serde_json::to_vec_pretty(catalog)?;
        catalog_bytes.push(b'\n');
        fs::write(root.join("ffi-contracts-v1.json"), &catalog_bytes)?;
        let manifest = json!({
            "schema_version": 1,
            "backend": "cuda",
            "abi_floor": ABI_FLOOR,
            "target": manifest_target,
            "libraries": ["cublaslt", "cudnn", "driver", "nvrtc"],
            "discovery_order": ["SIM_CUDA_ROOT", "CUDA_PATH", "signed_package_roots", "installed_driver_library"],
            "abi_manifest_sha256": sha256_hex(ABI_MANIFEST.as_bytes()),
            "kernel_manifest_sha256": sha256_hex(KERNEL_MANIFEST.as_bytes()),
            "core_ptx_sha256": sha256_hex(CORE_PTX),
            "ffi_contracts_sha256": sha256_hex(&catalog_bytes),
            "package_policy_sha256": sha256_hex(PACKAGE_POLICY.as_bytes()),
            "redistributes_driver": false,
            "approved_vendor_redistributables": [],
            "license_approval_required_for_vendor_runtime": true,
            "runtime_compilation_for_core_kernels": false,
            "signer": "cuda.release",
            "signature_algorithm": "ed25519",
            "signature_domain": "zed-sim-cuda-package-v1",
            "signature_coverage": "package-coverage-v1",
            "certificate_owner": "sim_runtime::NativeFfiRegistry",
            "unsafe_owner": UNSAFE_OWNER,
            "structural_receipt_is_authorization": false,
        });
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        manifest_bytes.push(b'\n');
        fs::write(root.join("adapter-manifest.json"), manifest_bytes)?;
        fs::write(root.join("adapter-manifest.sig"), b"pending\n")?;
        fs::write(root.join("package-coverage.sha256"), b"pending\n")?;

        let mut payloads = BTreeMap::new();
        for path in PACKAGE_PAYLOAD_LIMITS
            .iter()
            .map(NativePackagePayloadLimit::path)
        {
            payloads.insert(path.to_owned(), fs::read(root.join(path))?);
        }
        let mut coverage = Vec::new();
        for (path, bytes) in &payloads {
            if !COVERAGE_EXCLUDES.contains(&path.as_str()) {
                coverage.extend_from_slice(
                    format!("{} {}  {path}\n", sha256_hex(bytes), bytes.len()).as_bytes(),
                );
            }
        }
        fs::write(root.join("package-coverage.sha256"), &coverage)?;
        fs::write(
            root.join("adapter-manifest.sig"),
            signed_receipt("cuda.release", &coverage, key_pair)?,
        )?;
        CudaPackageVerificationKey::new("cuda.release", key_pair.public_key().as_ref())
            .map_err(|error| io::Error::other(error.to_string()).into())
    }

    #[test]
    fn signed_catalog_maps_once_to_the_canonical_registry_and_rejects_tampering()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let root = tempfile::tempdir()?;
        let catalog = fixture_catalog("x86_64-unknown-linux-gnu");
        let key =
            write_fixture_package(root.path(), "x86_64-unknown-linux-gnu", &catalog, &key_pair)?;
        let verified =
            verify_cuda_package_contracts(root.path(), &key, &CancellationToken::default())?;
        assert_eq!(verified.identities().len(), 4);
        assert_eq!(
            verified
                .registry()
                .required_symbols_for("driver", ABI_FLOOR, UNSAFE_OWNER)?
                .len(),
            23
        );
        assert_eq!(verified.identities()["cudnn"].filename(), "libcudnn.so.9");

        fs::write(root.path().join("kernels/core-v1.ptx"), b"tampered\n")?;
        assert!(
            verify_cuda_package_contracts(root.path(), &key, &CancellationToken::default())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn cuda_signature_domain_cannot_be_cross_authorized() -> Result<(), Box<dyn std::error::Error>>
    {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let coverage = b"fixture coverage\n";
        let receipt = signed_receipt("cuda.release", coverage, &key_pair)?;
        let cuda = CudaPackageVerificationKey::new("cuda.release", key_pair.public_key().as_ref())?;
        cuda.verify_package("cuda.release", coverage, &receipt)?;
        let plugin_domain =
            crate::XpuPackageVerificationKey::new("cuda.release", key_pair.public_key().as_ref())?;
        assert_eq!(
            plugin_domain.verify_package("cuda.release", coverage, &receipt),
            Err(TrustError::InvalidXpuPackageSignature)
        );
        Ok(())
    }

    #[test]
    fn catalog_rejects_unsorted_empty_mismatched_and_unknown_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        for mutation in [
            "reverse", "empty", "owner", "target", "filename", "digest", "abi", "symbols",
            "unknown",
        ] {
            let mut catalog = fixture_catalog("x86_64-unknown-linux-gnu");
            match mutation {
                "reverse" => catalog["libraries"]
                    .as_array_mut()
                    .expect("libraries are an array")
                    .reverse(),
                "empty" => catalog["libraries"][0]["required_symbols"] = json!([]),
                "owner" => catalog["libraries"][0]["unsafe_owner"] = json!("other::loader"),
                "target" => catalog["target"] = json!("aarch64-unknown-linux-gnu"),
                "filename" => catalog["libraries"][0]["filename"] = json!("other.so"),
                "digest" => catalog["libraries"][0]["sha256"] = json!("0"),
                "abi" => catalog["libraries"][0]["abi"] = json!("other-abi"),
                "symbols" => catalog["libraries"][0]["required_symbols"][0] = json!("other"),
                "unknown" => catalog["libraries"][0]["grants_availability"] = json!(true),
                _ => unreachable!(),
            }
            let root = tempfile::tempdir()?;
            let key = write_fixture_package(
                root.path(),
                "x86_64-unknown-linux-gnu",
                &catalog,
                &key_pair,
            )?;
            assert!(
                verify_cuda_package_contracts(root.path(), &key, &CancellationToken::default())
                    .is_err(),
                "mutation {mutation} was accepted"
            );
        }
        Ok(())
    }

    #[test]
    fn duplicate_json_cancelled_verification_wrong_key_and_settings_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(parse_strict_json::<Value>(b"{\"a\":1,\"a\":2}", "duplicate").is_err());
        let signer = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let other = Ed25519KeyPair::from_seed_unchecked(b"abcdef0123456789abcdef0123456789")
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let root = tempfile::tempdir()?;
        let catalog = fixture_catalog("x86_64-unknown-linux-gnu");
        write_fixture_package(root.path(), "x86_64-unknown-linux-gnu", &catalog, &signer)?;
        let wrong_key =
            CudaPackageVerificationKey::new("cuda.release", other.public_key().as_ref())?;
        assert_eq!(
            verify_cuda_package_contracts(root.path(), &wrong_key, &CancellationToken::default())
                .expect_err("wrong key must fail")
                .to_string(),
            TrustError::InvalidCudaPackageSignature.to_string()
        );

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            verify_cuda_package_contracts(root.path(), &wrong_key, &cancellation),
            Err(CudaPackageContractError::Cancelled(_))
        ));

        let settings = crate::NativeCudaPackageSettings::from_public_authority(
            "/opt/zed/cuda-package",
            "cuda.release",
            &signer
                .public_key()
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        )?;
        let encoded = serde_json::to_vec(&settings)?;
        let decoded: crate::NativeCudaPackageSettings = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, settings);
        assert!(!String::from_utf8(encoded)?.contains("private"));
        Ok(())
    }
}
