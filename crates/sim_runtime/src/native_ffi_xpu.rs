use crate::{
    CertifiedNativeFfi, NativeFfiContract, NativeFfiRegistry, TrustError,
    XpuPackageVerificationKey,
    trust::{
        NativeLibraryImageError, NativePackageAdmissionError, NativePackagePayloadLimit,
        RetainedNativeLibraryImage, capture_native_library_image, capture_native_package,
        validate_native_package_coverage,
    },
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_backend_xpu::{
    ABI_FLOOR, AbiManifest, DiscoveryPlan, LibraryLocation, RegistryCertifiedXpuImages,
    XpuExecutionError, XpuExecutionSession,
};
use sim_model::ArtifactRoot;
use sim_types::{BackendUnavailable, CancellationError, CancellationToken, DeviceKind};
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
use std::os::windows::ffi::OsStrExt;
use std::{collections::BTreeMap, path::Path, ptr::NonNull, sync::Arc};
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use std::{ffi::CString, os::unix::ffi::OsStrExt};
use thiserror::Error;

const PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/xpu/package-policy.json");
const ABI_MANIFEST: &str = sim_backend_xpu::ABI_MANIFEST;
const REVIEWED_EXECUTION_BINDINGS: &str =
    include_str!("../../sim_backend_xpu/abi/reviewed-execution-bindings-v1.txt");
const EXECUTION_VERIFIER: &[u8] =
    include_bytes!("../../sim_backend_xpu/abi/verify-execution-bindings.c");
const UNSAFE_OWNER: &str = "sim_backend_xpu::loader";
const MAX_PACKAGE_FILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_COVERAGE_BYTES: usize = 64 * 1024;
const MAX_PACKAGE_PAYLOADS: usize = 16;
const MAX_PACKAGE_BYTES: usize = 16 * 1024 * 1024;
const PACKAGE_PAYLOAD_LIMITS: [NativePackagePayloadLimit; 9] = [
    NativePackagePayloadLimit::new("LICENSES", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new(
        "abi/reviewed-execution-bindings-v1.txt",
        MAX_PACKAGE_FILE_BYTES,
    ),
    NativePackagePayloadLimit::new("abi/symbols-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("abi/verify-execution-bindings.c", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.sig", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("ffi-contracts-v1.json", MAX_CATALOG_BYTES),
    NativePackagePayloadLimit::new("package-coverage.sha256", MAX_COVERAGE_BYTES),
    NativePackagePayloadLimit::new("package-policy.json", MAX_PACKAGE_FILE_BYTES),
];
const COVERAGE_EXCLUDES: [&str; 2] = ["adapter-manifest.sig", "package-coverage.sha256"];
const SUPPORTED_TARGETS: [&str; 2] = ["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct XpuFfiContractCatalogDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    abi_manifest_sha256: String,
    package_policy_sha256: String,
    libraries: Vec<XpuFfiContractDto>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct XpuFfiContractDto {
    identity: String,
    filename: String,
    sha256: String,
    abi: String,
    required_symbols: Vec<String>,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct XpuPackageManifestDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    libraries: Vec<String>,
    discovery_order: Vec<String>,
    abi_manifest_sha256: String,
    reviewed_execution_bindings_sha256: String,
    execution_verifier_sha256: String,
    ffi_contracts_sha256: String,
    package_policy_sha256: String,
    redistributes_vendor_runtime: bool,
    license_approval_required_for_vendor_runtime: bool,
    runtime_compilation_forbidden: bool,
    signer: String,
    signature_algorithm: String,
    signature_domain: String,
    signature_coverage: String,
    certificate_owner: String,
    unsafe_owner: String,
    structural_receipt_is_authorization: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct XpuFfiLibraryIdentity {
    filename: String,
}

impl XpuFfiLibraryIdentity {
    pub fn filename(&self) -> &str {
        &self.filename
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedXpuFfiContracts {
    package_root: ArtifactRoot,
    target: String,
    registry: NativeFfiRegistry,
    identities: BTreeMap<String, XpuFfiLibraryIdentity>,
}

impl VerifiedXpuFfiContracts {
    pub fn package_root(&self) -> &ArtifactRoot {
        &self.package_root
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn registry(&self) -> &NativeFfiRegistry {
        &self.registry
    }

    pub fn identities(&self) -> &BTreeMap<String, XpuFfiLibraryIdentity> {
        &self.identities
    }
}

#[derive(Debug, Error)]
pub enum XpuPackageContractError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("XPU package tree is unsafe or incomplete: {0}")]
    UnsafePackage(String),
    #[error("signed XPU package metadata is invalid: {0}")]
    InvalidPackage(String),
    #[error("signed XPU FFI contract catalog is invalid: {0}")]
    InvalidCatalog(String),
}

#[derive(Debug, Error)]
pub enum XpuCertificationError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("no complete XPU library root was discovered")]
    MissingLibraryRoot,
    #[error("XPU library root is incomplete or unsafe: {0}")]
    UnsafeLibraryRoot(String),
    #[error("XPU library image {library_id} is invalid: {reason}")]
    InvalidImage { library_id: String, reason: String },
    #[error("sealed XPU images require a supported x86_64 Linux or Windows target")]
    UnsupportedPlatform,
}

struct XpuCertificationRetention {
    _verified: VerifiedXpuFfiContracts,
    _certificates: Vec<CertifiedNativeFfi>,
    _sealed_images: Vec<RetainedNativeLibraryImage>,
    _library_handles: NativeXpuLibraryHandles,
}

struct NativeXpuLibraryHandles {
    level_zero: NonNull<std::ffi::c_void>,
    onednn: NonNull<std::ffi::c_void>,
}

unsafe impl Send for NativeXpuLibraryHandles {}
unsafe impl Sync for NativeXpuLibraryHandles {}

impl Drop for NativeXpuLibraryHandles {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        unsafe {
            libc::dlclose(self.onednn.as_ptr());
            libc::dlclose(self.level_zero.as_ptr());
        }
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        unsafe {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
            }
            if FreeLibrary(self.onednn.as_ptr()) == 0 {
                eprintln!(
                    "failed to release the retained oneDNN image: {}",
                    std::io::Error::last_os_error()
                );
            }
            if FreeLibrary(self.level_zero.as_ptr()) == 0 {
                eprintln!(
                    "failed to release the retained Level Zero image: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

pub struct CertifiedXpuLoad {
    retention: Arc<XpuCertificationRetention>,
    certificates: Vec<CertifiedNativeFfi>,
}

impl CertifiedXpuLoad {
    pub fn certificates(&self) -> &[CertifiedNativeFfi] {
        &self.certificates
    }

    pub fn load_execution_runtime(
        self,
        device_ordinal: usize,
    ) -> Result<XpuExecutionSession, XpuExecutionError> {
        let certification: Arc<dyn std::any::Any + Send + Sync> = self.retention.clone();
        let images = unsafe {
            RegistryCertifiedXpuImages::from_registry_certified_images(
                certification,
                self.retention._library_handles.level_zero.as_ptr(),
                self.retention._library_handles.onednn.as_ptr(),
            )
        }?;
        XpuExecutionSession::from_registry_certified_images(images, device_ordinal)
    }
}

pub fn verify_xpu_package_contracts(
    package_root: &Path,
    verification_key: &XpuPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedXpuFfiContracts, XpuPackageContractError> {
    cancellation.check()?;
    let root = ArtifactRoot::canonical(
        "sim-xpu-package",
        "native-ffi-package",
        package_root,
        std::iter::empty::<String>(),
    )
    .map_err(|error| XpuPackageContractError::UnsafePackage(error.to_string()))?;
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
        return Err(XpuPackageContractError::InvalidPackage(
            "installed package policy differs from the compiled reviewed policy".to_owned(),
        ));
    }
    let manifest: XpuPackageManifestDto = parse_strict_json(
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

    let catalog: XpuFfiContractCatalogDto = parse_strict_json(
        required_payload(&payloads, "ffi-contracts-v1.json")?,
        "FFI contract catalog",
    )?;
    let (registry, identities) = validate_and_map_catalog(&catalog, &manifest)?;
    cancellation.check()?;
    Ok(VerifiedXpuFfiContracts {
        package_root: root,
        target: catalog.target,
        registry,
        identities,
    })
}

pub fn certify_xpu_library_images(
    verified: VerifiedXpuFfiContracts,
    discovery: &DiscoveryPlan,
    cancellation: &CancellationToken,
) -> Result<CertifiedXpuLoad, XpuCertificationError> {
    cancellation.check()?;
    if !host_target_matches(verified.target()) {
        return Err(XpuCertificationError::UnsafeLibraryRoot(format!(
            "signed target {} does not match this process",
            verified.target()
        )));
    }
    certify_xpu_library_images_after_target_check(verified, discovery, cancellation)
}

fn certify_xpu_library_images_after_target_check(
    verified: VerifiedXpuFfiContracts,
    discovery: &DiscoveryPlan,
    cancellation: &CancellationToken,
) -> Result<CertifiedXpuLoad, XpuCertificationError> {
    if discovery.target() != verified.target() {
        return Err(XpuCertificationError::UnsafeLibraryRoot(
            "discovery target differs from the signed contract".to_owned(),
        ));
    }
    let paths = discovery
        .candidates()
        .iter()
        .find_map(|candidate| {
            let LibraryLocation::AbsolutePath(level_zero) = candidate.level_zero() else {
                return None;
            };
            let LibraryLocation::AbsolutePath(onednn) = candidate.onednn()? else {
                return None;
            };
            (level_zero.is_file() && onednn.is_file()).then(|| {
                BTreeMap::from([
                    ("level_zero".to_owned(), level_zero.clone()),
                    ("onednn".to_owned(), onednn.clone()),
                ])
            })
        })
        .ok_or(XpuCertificationError::MissingLibraryRoot)?;
    let mut certificates = Vec::with_capacity(paths.len());
    let mut retained_files = Vec::with_capacity(paths.len());
    for (library_id, identity) in verified.identities() {
        cancellation.check()?;
        let path = paths
            .get(library_id)
            .ok_or_else(|| XpuCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "catalog identity has no reviewed discovery path".to_owned(),
            })?;
        if path.file_name().and_then(|name| name.to_str()) != Some(identity.filename()) {
            return Err(XpuCertificationError::InvalidImage {
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
            .seal(&format!("xpu-{library_id}"), cancellation)
            .map_err(|error| map_native_library_image_error(library_id, error))?;
        certificates.push(certificate);
        retained_files.push(retained);
    }
    let library_handles = load_retained_xpu_images(&retained_files)?;
    let retained_certificates = certificates.clone();
    let retention = Arc::new(XpuCertificationRetention {
        _verified: verified,
        _certificates: retained_certificates,
        _sealed_images: retained_files,
        _library_handles: library_handles,
    });
    Ok(CertifiedXpuLoad {
        retention,
        certificates,
    })
}

fn load_retained_xpu_images(
    retained_images: &[RetainedNativeLibraryImage],
) -> Result<NativeXpuLibraryHandles, XpuCertificationError> {
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        let _retained_images = retained_images;
        Err(XpuCertificationError::UnsupportedPlatform)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let level_zero = retained_images.first().ok_or_else(|| {
            XpuCertificationError::UnsafeLibraryRoot(
                "certified Level Zero image is missing".to_owned(),
            )
        })?;
        let onednn = retained_images.get(1).ok_or_else(|| {
            XpuCertificationError::UnsafeLibraryRoot("certified oneDNN image is missing".to_owned())
        })?;
        let level_zero_handle =
            open_retained_image(level_zero.loader_path(), libc::RTLD_NOW | libc::RTLD_GLOBAL)?;
        let onednn_handle =
            match open_retained_image(onednn.loader_path(), libc::RTLD_NOW | libc::RTLD_LOCAL) {
                Ok(handle) => handle,
                Err(error) => {
                    unsafe {
                        libc::dlclose(level_zero_handle.as_ptr());
                    }
                    return Err(error);
                }
            };
        Ok(NativeXpuLibraryHandles {
            level_zero: level_zero_handle,
            onednn: onednn_handle,
        })
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        let level_zero = retained_images.first().ok_or_else(|| {
            XpuCertificationError::UnsafeLibraryRoot(
                "certified Level Zero image is missing".to_owned(),
            )
        })?;
        let onednn = retained_images.get(1).ok_or_else(|| {
            XpuCertificationError::UnsafeLibraryRoot("certified oneDNN image is missing".to_owned())
        })?;
        let level_zero_handle = open_retained_windows_image(level_zero.loader_path())?;
        let onednn_handle = match open_retained_windows_image(onednn.loader_path()) {
            Ok(handle) => handle,
            Err(error) => {
                unsafe {
                    #[link(name = "kernel32")]
                    unsafe extern "system" {
                        fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
                    }
                    if FreeLibrary(level_zero_handle.as_ptr()) == 0 {
                        eprintln!(
                            "failed to release the retained Level Zero image after oneDNN load failure: {}",
                            std::io::Error::last_os_error()
                        );
                    }
                }
                return Err(error);
            }
        };
        Ok(NativeXpuLibraryHandles {
            level_zero: level_zero_handle,
            onednn: onednn_handle,
        })
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn open_retained_image(
    path: &Path,
    flags: libc::c_int,
) -> Result<NonNull<std::ffi::c_void>, XpuCertificationError> {
    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        XpuCertificationError::UnsafeLibraryRoot(
            "sealed XPU loader path contains an interior NUL".to_owned(),
        )
    })?;
    let handle = unsafe { libc::dlopen(path.as_ptr(), flags) };
    NonNull::new(handle).ok_or_else(|| {
        XpuCertificationError::UnsafeLibraryRoot(
            "the exact sealed XPU image could not be loaded".to_owned(),
        )
    })
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn open_retained_windows_image(
    path: &Path,
) -> Result<NonNull<std::ffi::c_void>, XpuCertificationError> {
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
        return Err(XpuCertificationError::UnsafeLibraryRoot(
            "sealed XPU loader path contains an interior NUL".to_owned(),
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
        XpuCertificationError::UnsafeLibraryRoot(format!(
            "the exact sealed XPU image could not be loaded: {}",
            std::io::Error::last_os_error()
        ))
    })
}

pub fn initialize_certified_xpu_runtime(
    settings: &crate::NativeXpuPackageSettings,
    device_ordinal: usize,
    cancellation: &CancellationToken,
) -> Result<XpuExecutionSession, BackendUnavailable> {
    let discovery =
        DiscoveryPlan::from_environment(current_target(), [settings.package_root().to_path_buf()])
            .map_err(|_| {
                BackendUnavailable::new(DeviceKind::Xpu, "XPU discovery plan is invalid")
            })?;
    initialize_certified_xpu_runtime_with_discovery(
        settings,
        &discovery,
        device_ordinal,
        cancellation,
    )
}

pub fn initialize_certified_xpu_runtime_with_discovery(
    settings: &crate::NativeXpuPackageSettings,
    discovery: &DiscoveryPlan,
    device_ordinal: usize,
    cancellation: &CancellationToken,
) -> Result<XpuExecutionSession, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Xpu, reason);
    let verified = verify_xpu_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed package or contract verification failed"))?;
    let certified = certify_xpu_library_images(verified, discovery, cancellation)
        .map_err(|_| unavailable("exact XPU library certification failed"))?;
    certified
        .load_execution_runtime(device_ordinal)
        .map_err(|_| unavailable("certified XPU loader or ABI probe failed"))
}

fn validate_manifest(
    manifest: &XpuPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), XpuPackageContractError> {
    let target_supported = SUPPORTED_TARGETS.contains(&manifest.target.as_str());
    if manifest.schema_version != 1
        || manifest.backend != "xpu"
        || manifest.abi_floor != ABI_FLOOR
        || !target_supported
        || manifest.libraries != ["level_zero", "onednn"]
        || manifest.discovery_order
            != [
                "SIM_XPU_ROOT",
                "ONEAPI_ROOT",
                "signed_package_roots",
                "system_level_zero_loader",
            ]
        || manifest.abi_manifest_sha256
            != sha256_hex(required_payload(payloads, "abi/symbols-v1.json")?)
        || manifest.ffi_contracts_sha256
            != sha256_hex(required_payload(payloads, "ffi-contracts-v1.json")?)
        || manifest.reviewed_execution_bindings_sha256
            != sha256_hex(required_payload(
                payloads,
                "abi/reviewed-execution-bindings-v1.txt",
            )?)
        || manifest.execution_verifier_sha256
            != sha256_hex(required_payload(
                payloads,
                "abi/verify-execution-bindings.c",
            )?)
        || manifest.package_policy_sha256
            != sha256_hex(required_payload(payloads, "package-policy.json")?)
        || manifest.redistributes_vendor_runtime
        || !manifest.license_approval_required_for_vendor_runtime
        || !manifest.runtime_compilation_forbidden
        || manifest.signature_algorithm != "ed25519"
        || manifest.signature_domain != "zed-sim-xpu-package-v1"
        || manifest.signature_coverage != "package-coverage-v1"
        || manifest.certificate_owner != "sim_runtime::NativeFfiRegistry"
        || manifest.unsafe_owner != UNSAFE_OWNER
        || manifest.structural_receipt_is_authorization
        || !valid_signer(&manifest.signer)
    {
        return Err(XpuPackageContractError::InvalidPackage(
            "adapter manifest differs from the reviewed XPU package contract".to_owned(),
        ));
    }
    if required_payload(payloads, "abi/symbols-v1.json")? != ABI_MANIFEST.as_bytes() {
        return Err(XpuPackageContractError::InvalidPackage(
            "packaged ABI manifest differs from the compiled reviewed manifest".to_owned(),
        ));
    }
    if required_payload(payloads, "abi/reviewed-execution-bindings-v1.txt")?
        != REVIEWED_EXECUTION_BINDINGS.as_bytes()
        || required_payload(payloads, "abi/verify-execution-bindings.c")? != EXECUTION_VERIFIER
    {
        return Err(XpuPackageContractError::InvalidPackage(
            "packaged reviewed execution evidence differs from the compiled evidence".to_owned(),
        ));
    }
    Ok(())
}

fn validate_and_map_catalog(
    catalog: &XpuFfiContractCatalogDto,
    manifest: &XpuPackageManifestDto,
) -> Result<(NativeFfiRegistry, BTreeMap<String, XpuFfiLibraryIdentity>), XpuPackageContractError> {
    let abi = AbiManifest::embedded()
        .map_err(|error| XpuPackageContractError::InvalidCatalog(error.to_string()))?;
    if catalog.schema_version != 1
        || catalog.backend != "xpu"
        || catalog.abi_floor != ABI_FLOOR
        || catalog.target != manifest.target
        || catalog.abi_manifest_sha256 != manifest.abi_manifest_sha256
        || catalog.package_policy_sha256 != manifest.package_policy_sha256
        || catalog.libraries.len() != abi.libraries.len()
    {
        return Err(XpuPackageContractError::InvalidCatalog(
            "catalog envelope is unsupported, target-mismatched, or incomplete".to_owned(),
        ));
    }
    let mut previous_identity: Option<&str> = None;
    let mut contracts = Vec::with_capacity(catalog.libraries.len());
    let mut identities = BTreeMap::new();
    for row in &catalog.libraries {
        if previous_identity.is_some_and(|previous| previous >= row.identity.as_str()) {
            return Err(XpuPackageContractError::InvalidCatalog(
                "library identities must be sorted and unique".to_owned(),
            ));
        }
        previous_identity = Some(&row.identity);
        let reviewed = abi
            .libraries
            .iter()
            .find(|library| library.id == row.identity)
            .ok_or_else(|| {
                XpuPackageContractError::InvalidCatalog(format!(
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
            &reviewed.filenames.windows
        } else {
            &reviewed.filenames.linux
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
            return Err(XpuPackageContractError::InvalidCatalog(format!(
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
            XpuFfiLibraryIdentity {
                filename: row.filename.clone(),
            },
        );
    }
    if identities.len() != abi.libraries.len() {
        return Err(XpuPackageContractError::InvalidCatalog(
            "catalog does not cover every reviewed library".to_owned(),
        ));
    }
    Ok((NativeFfiRegistry::new(contracts)?, identities))
}

fn map_package_admission_error(error: NativePackageAdmissionError) -> XpuPackageContractError {
    match error {
        NativePackageAdmissionError::Cancelled => CancellationError.into(),
        NativePackageAdmissionError::UnsafePackage(reason) => {
            XpuPackageContractError::UnsafePackage(reason)
        }
        NativePackageAdmissionError::InvalidCoverage(reason) => {
            XpuPackageContractError::InvalidPackage(reason)
        }
    }
}

fn required_payload<'a>(
    payloads: &'a BTreeMap<String, Vec<u8>>,
    path: &str,
) -> Result<&'a [u8], XpuPackageContractError> {
    payloads.get(path).map(Vec::as_slice).ok_or_else(|| {
        XpuPackageContractError::UnsafePackage(format!("required payload is missing: {path}"))
    })
}

fn parse_strict_json<T: DeserializeOwned>(
    bytes: &[u8],
    label: &str,
) -> Result<T, XpuPackageContractError> {
    let strict = crate::trust::parse_strict_json_value(bytes).map_err(|error| {
        XpuPackageContractError::InvalidPackage(format!("{label} is not strict JSON: {error}"))
    })?;
    serde_json::from_value(strict).map_err(|error| {
        XpuPackageContractError::InvalidPackage(format!("{label} is invalid: {error}"))
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
        ("x86_64", "linux", "x86_64-unknown-linux-gnu")
            | ("x86_64", "windows", "x86_64-pc-windows-msvc")
    )
}

fn current_target() -> &'static str {
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu",
        ("x86_64", "windows") => "x86_64-pc-windows-msvc",
        _ => "unsupported-xpu-host",
    }
}

fn map_native_library_image_error(
    library_id: &str,
    error: NativeLibraryImageError,
) -> XpuCertificationError {
    match error {
        NativeLibraryImageError::Cancelled => XpuCertificationError::Cancelled(CancellationError),
        NativeLibraryImageError::UnsupportedPlatform => XpuCertificationError::UnsupportedPlatform,
        NativeLibraryImageError::Invalid(reason) => XpuCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use serde_json::json;
    use std::{fs, io};

    const SIGNING_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";

    fn signed_receipt(
        signer: &str,
        coverage: &[u8],
        key_pair: &Ed25519KeyPair,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let payload = crate::xpu_package_signing_payload(signer, coverage)?;
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

    fn fixture_catalog(target: &str, level_zero_digest: &str, onednn_digest: &str) -> Value {
        let abi: Value = serde_json::from_str(ABI_MANIFEST).expect("embedded ABI is valid");
        let platform = if target.contains("windows") {
            "windows"
        } else {
            "linux"
        };
        let libraries = abi["libraries"]
            .as_array()
            .expect("libraries are an array")
            .iter()
            .map(|library| {
                let identity = library["id"].as_str().expect("id is text");
                let mut required_symbols = library["symbols"]
                    .as_array()
                    .expect("symbols")
                    .iter()
                    .map(|symbol| symbol["name"].clone())
                    .collect::<Vec<_>>();
                required_symbols.sort_by(|left, right| {
                    left.as_str()
                        .unwrap_or_default()
                        .cmp(right.as_str().unwrap_or_default())
                });
                json!({
                    "identity": identity,
                    "filename": library["filenames"][platform],
                    "sha256": if identity == "level_zero" { level_zero_digest } else { onednn_digest },
                    "abi": ABI_FLOOR,
                    "required_symbols": required_symbols,
                    "unsafe_owner": UNSAFE_OWNER,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": 1,
            "backend": "xpu",
            "abi_floor": ABI_FLOOR,
            "target": target,
            "abi_manifest_sha256": sha256_hex(ABI_MANIFEST.as_bytes()),
            "package_policy_sha256": sha256_hex(PACKAGE_POLICY.as_bytes()),
            "libraries": libraries,
        })
    }

    fn write_fixture_package(
        root: &Path,
        target: &str,
        catalog: &Value,
        key_pair: &Ed25519KeyPair,
    ) -> Result<XpuPackageVerificationKey, Box<dyn std::error::Error>> {
        fs::create_dir_all(root.join("abi"))?;
        fs::write(root.join("LICENSES"), sim_backend_xpu::PACKAGE_LICENSES)?;
        fs::write(root.join("abi/symbols-v1.json"), ABI_MANIFEST)?;
        fs::write(
            root.join("abi/reviewed-execution-bindings-v1.txt"),
            REVIEWED_EXECUTION_BINDINGS,
        )?;
        fs::write(
            root.join("abi/verify-execution-bindings.c"),
            EXECUTION_VERIFIER,
        )?;
        fs::write(root.join("package-policy.json"), PACKAGE_POLICY)?;
        let mut catalog_bytes = serde_json::to_vec_pretty(catalog)?;
        catalog_bytes.push(b'\n');
        fs::write(root.join("ffi-contracts-v1.json"), &catalog_bytes)?;
        let manifest = json!({
            "schema_version": 1,
            "backend": "xpu",
            "abi_floor": ABI_FLOOR,
            "target": target,
            "libraries": ["level_zero", "onednn"],
            "discovery_order": ["SIM_XPU_ROOT", "ONEAPI_ROOT", "signed_package_roots", "system_level_zero_loader"],
            "abi_manifest_sha256": sha256_hex(ABI_MANIFEST.as_bytes()),
            "reviewed_execution_bindings_sha256": sha256_hex(REVIEWED_EXECUTION_BINDINGS.as_bytes()),
            "execution_verifier_sha256": sha256_hex(EXECUTION_VERIFIER),
            "ffi_contracts_sha256": sha256_hex(&catalog_bytes),
            "package_policy_sha256": sha256_hex(PACKAGE_POLICY.as_bytes()),
            "redistributes_vendor_runtime": false,
            "license_approval_required_for_vendor_runtime": true,
            "runtime_compilation_forbidden": true,
            "signer": "xpu.release",
            "signature_algorithm": "ed25519",
            "signature_domain": "zed-sim-xpu-package-v1",
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
            signed_receipt("xpu.release", &coverage, key_pair)?,
        )?;
        XpuPackageVerificationKey::new("xpu.release", key_pair.public_key().as_ref())
            .map_err(|error| io::Error::other(error.to_string()).into())
    }

    #[test]
    fn signed_catalog_maps_once_to_the_canonical_registry_and_rejects_tampering()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let root = tempfile::tempdir()?;
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let catalog = fixture_catalog("x86_64-unknown-linux-gnu", digest, digest);
        let key =
            write_fixture_package(root.path(), "x86_64-unknown-linux-gnu", &catalog, &key_pair)?;
        let verified =
            verify_xpu_package_contracts(root.path(), &key, &CancellationToken::default())?;
        assert_eq!(verified.identities().len(), 2);
        assert_eq!(
            verified
                .registry()
                .required_symbols_for("level_zero", ABI_FLOOR, UNSAFE_OWNER)?
                .len(),
            12
        );
        assert_eq!(verified.identities()["onednn"].filename(), "libdnnl.so.3");

        fs::write(root.path().join("ffi-contracts-v1.json"), b"{}\n")?;
        assert!(
            verify_xpu_package_contracts(root.path(), &key, &CancellationToken::default()).is_err()
        );
        Ok(())
    }

    #[test]
    fn xpu_signature_domain_cannot_be_cross_authorized() -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let coverage = b"fixture coverage\n";
        let receipt = signed_receipt("xpu.release", coverage, &key_pair)?;
        let xpu = XpuPackageVerificationKey::new("xpu.release", key_pair.public_key().as_ref())?;
        xpu.verify_package("xpu.release", coverage, &receipt)?;
        let metal =
            crate::MetalPackageVerificationKey::new("xpu.release", key_pair.public_key().as_ref())?;
        assert_eq!(
            metal.verify_package("xpu.release", coverage, &receipt),
            Err(TrustError::InvalidMetalPackageSignature)
        );
        Ok(())
    }

    #[test]
    fn catalog_rejects_unsorted_empty_and_mismatched_rows() -> Result<(), Box<dyn std::error::Error>>
    {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        for mutate in [
            "reverse", "empty", "owner", "target", "filename", "digest", "abi", "symbols",
            "unknown",
        ]
        .into_iter()
        {
            let mut catalog = fixture_catalog("x86_64-unknown-linux-gnu", digest, digest);
            match mutate {
                "reverse" => catalog["libraries"]
                    .as_array_mut()
                    .expect("array")
                    .reverse(),
                "empty" => catalog["libraries"][0]["required_symbols"] = json!([]),
                "owner" => catalog["libraries"][0]["unsafe_owner"] = json!("other::loader"),
                "target" => catalog["target"] = json!("aarch64-unknown-linux-gnu"),
                "filename" => catalog["libraries"][0]["filename"] = json!("ze_loader.dll"),
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
            let result =
                verify_xpu_package_contracts(root.path(), &key, &CancellationToken::default());
            assert!(result.is_err(), "mutation {mutate} was accepted");
        }
        Ok(())
    }

    #[test]
    fn duplicate_json_unknown_key_and_cancelled_verification_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(parse_strict_json::<Value>(b"{\"a\":1,\"a\":2}", "duplicate").is_err());
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let root = tempfile::tempdir()?;
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let catalog = fixture_catalog("x86_64-unknown-linux-gnu", digest, digest);
        let key =
            write_fixture_package(root.path(), "x86_64-unknown-linux-gnu", &catalog, &key_pair)?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            verify_xpu_package_contracts(root.path(), &key, &cancellation),
            Err(XpuPackageContractError::Cancelled(_))
        ));
        Ok(())
    }

    #[test]
    fn package_signed_by_an_unconfigured_key_is_rejected() -> Result<(), Box<dyn std::error::Error>>
    {
        let signer = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let other = Ed25519KeyPair::from_seed_unchecked(b"abcdef0123456789abcdef0123456789")
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let root = tempfile::tempdir()?;
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let catalog = fixture_catalog("x86_64-unknown-linux-gnu", digest, digest);
        write_fixture_package(root.path(), "x86_64-unknown-linux-gnu", &catalog, &signer)?;
        let key = XpuPackageVerificationKey::new("xpu.release", other.public_key().as_ref())?;
        assert_eq!(
            verify_xpu_package_contracts(root.path(), &key, &CancellationToken::default())
                .expect_err("wrong verification key must fail")
                .to_string(),
            TrustError::InvalidXpuPackageSignature.to_string()
        );
        Ok(())
    }

    #[test]
    fn xpu_settings_round_trip_contains_only_public_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let settings = crate::NativeXpuPackageSettings::from_public_authority(
            "/opt/zed/xpu-package",
            "xpu.release",
            &key_pair
                .public_key()
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        )?;
        let encoded = serde_json::to_vec(&settings)?;
        let decoded: crate::NativeXpuPackageSettings = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, settings);
        assert!(!String::from_utf8(encoded)?.contains("private"));
        Ok(())
    }
}
