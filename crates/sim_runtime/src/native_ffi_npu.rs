use crate::{
    CertifiedNativeFfi, NativeFfiContract, NativeFfiRegistry, NpuPackageVerificationKey,
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
use sim_backend_npu::{
    ABI_FLOOR, AbiManifest, DiscoveryEnvironment, NpuExecutionError, NpuExecutionSession,
    RegistryCertifiedNpuImages, discover_installed_libraries,
};
use sim_model::ArtifactRoot;
use sim_types::{BackendUnavailable, CancellationError, CancellationToken, DeviceKind};
use std::{collections::BTreeMap, path::Path, ptr::NonNull, sync::Arc};
#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use std::{ffi::CString, os::unix::ffi::OsStrExt};
use thiserror::Error;

#[cfg(test)]
use std::fs;

const PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/npu/package-policy.json");
const ABI_MANIFEST: &str = sim_backend_npu::ABI_MANIFEST_JSON;
const UNSAFE_OWNER: &str = "sim_backend_npu::loader";
const MAX_PACKAGE_FILE_BYTES: usize = 4 * 1024 * 1024;
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_COVERAGE_BYTES: usize = 64 * 1024;
const MAX_PACKAGE_PAYLOADS: usize = 16;
const MAX_PACKAGE_BYTES: usize = 16 * 1024 * 1024;
const PACKAGE_PAYLOAD_LIMITS: [NativePackagePayloadLimit; 7] = [
    NativePackagePayloadLimit::new("LICENSES", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("abi/symbols-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.sig", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("ffi-contracts-v1.json", MAX_CATALOG_BYTES),
    NativePackagePayloadLimit::new("package-coverage.sha256", MAX_COVERAGE_BYTES),
    NativePackagePayloadLimit::new("package-policy.json", MAX_PACKAGE_FILE_BYTES),
];
const COVERAGE_EXCLUDES: [&str; 2] = ["adapter-manifest.sig", "package-coverage.sha256"];
const SUPPORTED_TARGETS: [&str; 2] = ["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"];

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NpuFfiContractCatalogDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    abi_manifest_sha256: String,
    package_policy_sha256: String,
    libraries: Vec<NpuFfiContractDto>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NpuFfiContractDto {
    identity: String,
    filename: String,
    sha256: String,
    abi: String,
    required_symbols: Vec<String>,
    required_by: Option<String>,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NpuPackageManifestDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    libraries: Vec<String>,
    discovery_order: Vec<String>,
    abi_manifest_sha256: String,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NpuFfiLibraryIdentity {
    filename: String,
}

impl NpuFfiLibraryIdentity {
    pub fn filename(&self) -> &str {
        &self.filename
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedNpuFfiContracts {
    package_root: ArtifactRoot,
    target: String,
    registry: NativeFfiRegistry,
    identities: BTreeMap<String, NpuFfiLibraryIdentity>,
}

impl VerifiedNpuFfiContracts {
    pub fn package_root(&self) -> &ArtifactRoot {
        &self.package_root
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn registry(&self) -> &NativeFfiRegistry {
        &self.registry
    }

    pub fn identities(&self) -> &BTreeMap<String, NpuFfiLibraryIdentity> {
        &self.identities
    }
}

#[derive(Debug, Error)]
pub enum NpuPackageContractError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("NPU package tree is unsafe or incomplete: {0}")]
    UnsafePackage(String),
    #[error("signed NPU package metadata is invalid: {0}")]
    InvalidPackage(String),
    #[error("signed NPU FFI contract catalog is invalid: {0}")]
    InvalidCatalog(String),
}

#[derive(Debug, Error)]
pub enum NpuCertificationError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("no complete NPU library root was discovered")]
    MissingLibraryRoot,
    #[error("NPU library root is incomplete or unsafe: {0}")]
    UnsafeLibraryRoot(String),
    #[error("NPU library image {library_id} is invalid: {reason}")]
    InvalidImage { library_id: String, reason: String },
    #[error("sealed NPU images require Linux")]
    UnsupportedPlatform,
}

struct NpuCertificationRetention {
    _verified: VerifiedNpuFfiContracts,
    _certificates: Vec<CertifiedNativeFfi>,
    _sealed_images: Vec<RetainedNativeLibraryImage>,
    _library_handles: NativeNpuLibraryHandles,
}

struct NativeNpuLibraryHandles {
    ascendcl: NonNull<std::ffi::c_void>,
    runtime: NonNull<std::ffi::c_void>,
}

unsafe impl Send for NativeNpuLibraryHandles {}
unsafe impl Sync for NativeNpuLibraryHandles {}

impl Drop for NativeNpuLibraryHandles {
    fn drop(&mut self) {
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        unsafe {
            libc::dlclose(self.ascendcl.as_ptr());
            libc::dlclose(self.runtime.as_ptr());
        }
    }
}

pub struct CertifiedNpuLoad {
    retention: Arc<NpuCertificationRetention>,
    certificates: Vec<CertifiedNativeFfi>,
}

impl CertifiedNpuLoad {
    pub fn certificates(&self) -> &[CertifiedNativeFfi] {
        &self.certificates
    }

    pub fn load_execution_runtime(
        self,
        device_ordinal: u32,
    ) -> Result<NpuExecutionSession, NpuExecutionError> {
        let certification: Arc<dyn std::any::Any + Send + Sync> = self.retention.clone();
        let images = unsafe {
            RegistryCertifiedNpuImages::from_registry_certified_handles(
                certification,
                self.retention._library_handles.ascendcl.as_ptr(),
                self.retention._library_handles.runtime.as_ptr(),
            )
        }?;
        NpuExecutionSession::from_registry_certified_images(images, device_ordinal)
    }
}

pub fn verify_npu_package_contracts(
    package_root: &Path,
    verification_key: &NpuPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedNpuFfiContracts, NpuPackageContractError> {
    cancellation.check()?;
    let root = ArtifactRoot::canonical(
        "sim-npu-package",
        "native-ffi-package",
        package_root,
        std::iter::empty::<String>(),
    )
    .map_err(|error| NpuPackageContractError::UnsafePackage(error.to_string()))?;
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
        return Err(NpuPackageContractError::InvalidPackage(
            "installed package policy differs from the compiled reviewed policy".to_owned(),
        ));
    }
    let manifest: NpuPackageManifestDto = parse_strict_json(
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

    let catalog: NpuFfiContractCatalogDto = parse_strict_json(
        required_payload(&payloads, "ffi-contracts-v1.json")?,
        "FFI contract catalog",
    )?;
    let (registry, identities) = validate_and_map_catalog(&catalog, &manifest)?;
    cancellation.check()?;
    Ok(VerifiedNpuFfiContracts {
        package_root: root,
        target: catalog.target,
        registry,
        identities,
    })
}

pub fn certify_npu_library_images(
    verified: VerifiedNpuFfiContracts,
    discovery: &DiscoveryEnvironment,
    cancellation: &CancellationToken,
) -> Result<CertifiedNpuLoad, NpuCertificationError> {
    cancellation.check()?;
    if !host_target_matches(verified.target()) {
        return Err(NpuCertificationError::UnsafeLibraryRoot(format!(
            "signed target {} does not match this process",
            verified.target()
        )));
    }
    certify_npu_library_images_after_target_check(verified, discovery, cancellation)
}

fn certify_npu_library_images_after_target_check(
    verified: VerifiedNpuFfiContracts,
    discovery: &DiscoveryEnvironment,
    cancellation: &CancellationToken,
) -> Result<CertifiedNpuLoad, NpuCertificationError> {
    let discovered = discover_installed_libraries(discovery, &[])
        .map_err(|error| NpuCertificationError::UnsafeLibraryRoot(error.to_string()))?;
    let paths = BTreeMap::from([
        ("ascendcl".to_owned(), discovered.ascendcl),
        ("runtime".to_owned(), discovered.runtime),
    ]);
    let mut certificates = Vec::with_capacity(paths.len());
    let mut retained_files = Vec::with_capacity(paths.len());
    for (library_id, identity) in verified.identities() {
        cancellation.check()?;
        let path = paths
            .get(library_id)
            .ok_or_else(|| NpuCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "catalog identity has no reviewed discovery path".to_owned(),
            })?;
        if path.file_name().and_then(|name| name.to_str()) != Some(identity.filename()) {
            return Err(NpuCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "discovered filename differs from the signed contract".to_owned(),
            });
        }
        let captured = capture_native_library_image(path, cancellation)
            .map_err(|error| map_native_library_image_error(library_id, error))?;
        let certificate = if library_id == "runtime" {
            verified.registry().authorize_dependency(
                library_id,
                captured.digest_sha256(),
                ABI_FLOOR,
                "ascendcl",
            )?
        } else {
            let symbols =
                verified
                    .registry()
                    .required_symbols_for(library_id, ABI_FLOOR, UNSAFE_OWNER)?;
            verified.registry().authorize(
                library_id,
                captured.digest_sha256(),
                ABI_FLOOR,
                &symbols,
            )?
        };
        let retained = captured
            .seal(&format!("npu-{library_id}"), cancellation)
            .map_err(|error| map_native_library_image_error(library_id, error))?;
        certificates.push(certificate);
        retained_files.push(retained);
    }
    let library_handles = load_retained_npu_images(&retained_files)?;
    let retained_certificates = certificates.clone();
    let retention = Arc::new(NpuCertificationRetention {
        _verified: verified,
        _certificates: retained_certificates,
        _sealed_images: retained_files,
        _library_handles: library_handles,
    });
    Ok(CertifiedNpuLoad {
        retention,
        certificates,
    })
}

fn load_retained_npu_images(
    retained_images: &[RetainedNativeLibraryImage],
) -> Result<NativeNpuLibraryHandles, NpuCertificationError> {
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    {
        let _retained_images = retained_images;
        Err(NpuCertificationError::UnsupportedPlatform)
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    {
        let ascendcl = retained_images.first().ok_or_else(|| {
            NpuCertificationError::UnsafeLibraryRoot(
                "certified AscendCL image is missing".to_owned(),
            )
        })?;
        let runtime = retained_images.get(1).ok_or_else(|| {
            NpuCertificationError::UnsafeLibraryRoot(
                "certified Ascend runtime image is missing".to_owned(),
            )
        })?;
        let runtime_handle =
            open_retained_image(runtime.loader_path(), libc::RTLD_NOW | libc::RTLD_GLOBAL)?;
        let ascendcl_handle =
            match open_retained_image(ascendcl.loader_path(), libc::RTLD_NOW | libc::RTLD_LOCAL) {
                Ok(handle) => handle,
                Err(error) => {
                    unsafe {
                        libc::dlclose(runtime_handle.as_ptr());
                    }
                    return Err(error);
                }
            };
        Ok(NativeNpuLibraryHandles {
            ascendcl: ascendcl_handle,
            runtime: runtime_handle,
        })
    }
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn open_retained_image(
    path: &Path,
    flags: libc::c_int,
) -> Result<NonNull<std::ffi::c_void>, NpuCertificationError> {
    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        NpuCertificationError::UnsafeLibraryRoot(
            "sealed NPU loader path contains an interior NUL".to_owned(),
        )
    })?;
    let handle = unsafe { libc::dlopen(path.as_ptr(), flags) };
    NonNull::new(handle).ok_or_else(|| {
        NpuCertificationError::UnsafeLibraryRoot(
            "the exact sealed NPU image could not be loaded".to_owned(),
        )
    })
}

pub fn initialize_certified_npu_runtime(
    settings: &crate::NativeNpuPackageSettings,
    device_ordinal: u32,
    cancellation: &CancellationToken,
) -> Result<NpuExecutionSession, BackendUnavailable> {
    let discovery = DiscoveryEnvironment::from_process();
    initialize_certified_npu_runtime_with_discovery(
        settings,
        &discovery,
        device_ordinal,
        cancellation,
    )
}

pub fn initialize_certified_npu_runtime_with_discovery(
    settings: &crate::NativeNpuPackageSettings,
    discovery: &DiscoveryEnvironment,
    device_ordinal: u32,
    cancellation: &CancellationToken,
) -> Result<NpuExecutionSession, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Npu, reason);
    let verified = verify_npu_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed package or contract verification failed"))?;
    let certified = certify_npu_library_images(verified, discovery, cancellation)
        .map_err(|_| unavailable("exact NPU library certification failed"))?;
    certified
        .load_execution_runtime(device_ordinal)
        .map_err(|_| unavailable("certified NPU loader or ABI probe failed"))
}

fn validate_manifest(
    manifest: &NpuPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), NpuPackageContractError> {
    let target_supported = SUPPORTED_TARGETS.contains(&manifest.target.as_str());
    if manifest.schema_version != 1
        || manifest.backend != "npu"
        || manifest.abi_floor != ABI_FLOOR
        || !target_supported
        || manifest.libraries != ["libascendcl.so", "libruntime.so"]
        || manifest.discovery_order
            != [
                "SIM_ASCEND_ROOT",
                "ASCEND_HOME_PATH",
                "signed_package_roots",
            ]
        || manifest.abi_manifest_sha256
            != sha256_hex(required_payload(payloads, "abi/symbols-v1.json")?)
        || manifest.ffi_contracts_sha256
            != sha256_hex(required_payload(payloads, "ffi-contracts-v1.json")?)
        || manifest.package_policy_sha256
            != sha256_hex(required_payload(payloads, "package-policy.json")?)
        || manifest.redistributes_vendor_runtime
        || !manifest.license_approval_required_for_vendor_runtime
        || !manifest.runtime_compilation_forbidden
        || manifest.signature_algorithm != "ed25519"
        || manifest.signature_domain != "zed-sim-npu-package-v1"
        || manifest.signature_coverage != "package-coverage-v1"
        || manifest.certificate_owner != "sim_runtime::NativeFfiRegistry"
        || manifest.unsafe_owner != UNSAFE_OWNER
        || !valid_signer(&manifest.signer)
    {
        return Err(NpuPackageContractError::InvalidPackage(
            "adapter manifest differs from the reviewed NPU package contract".to_owned(),
        ));
    }
    if required_payload(payloads, "abi/symbols-v1.json")? != ABI_MANIFEST.as_bytes() {
        return Err(NpuPackageContractError::InvalidPackage(
            "packaged ABI manifest differs from the compiled reviewed manifest".to_owned(),
        ));
    }
    Ok(())
}

fn validate_and_map_catalog(
    catalog: &NpuFfiContractCatalogDto,
    manifest: &NpuPackageManifestDto,
) -> Result<(NativeFfiRegistry, BTreeMap<String, NpuFfiLibraryIdentity>), NpuPackageContractError> {
    let abi = AbiManifest::embedded()
        .map_err(|error| NpuPackageContractError::InvalidCatalog(error.to_string()))?;
    if catalog.schema_version != 1
        || catalog.backend != "npu"
        || catalog.abi_floor != ABI_FLOOR
        || catalog.target != manifest.target
        || catalog.abi_manifest_sha256 != manifest.abi_manifest_sha256
        || catalog.package_policy_sha256 != manifest.package_policy_sha256
        || catalog.libraries.len() != abi.libraries.len()
    {
        return Err(NpuPackageContractError::InvalidCatalog(
            "catalog envelope is unsupported, target-mismatched, or incomplete".to_owned(),
        ));
    }
    let mut previous_identity: Option<&str> = None;
    let mut contracts = Vec::with_capacity(catalog.libraries.len());
    let mut identities = BTreeMap::new();
    for row in &catalog.libraries {
        if previous_identity.is_some_and(|previous| previous >= row.identity.as_str()) {
            return Err(NpuPackageContractError::InvalidCatalog(
                "library identities must be sorted and unique".to_owned(),
            ));
        }
        previous_identity = Some(&row.identity);
        let reviewed = abi
            .libraries
            .iter()
            .find(|library| library.id == row.identity)
            .ok_or_else(|| {
                NpuPackageContractError::InvalidCatalog(format!(
                    "unknown library identity {}",
                    row.identity
                ))
            })?;
        let expected_symbols = reviewed
            .symbols
            .iter()
            .map(|symbol| symbol.name.clone())
            .collect::<Vec<_>>();
        let expected_required_by = (row.identity == "runtime").then_some("ascendcl");
        if row.filename != reviewed.filename
            || !valid_lower_hex_digest(&row.sha256)
            || row.abi != ABI_FLOOR
            || row.required_symbols != expected_symbols
            || row.required_by.as_deref() != expected_required_by
            || (row.identity != "runtime" && row.required_symbols.is_empty())
            || row
                .required_symbols
                .windows(2)
                .any(|pair| matches!(pair, [first, second] if first >= second))
            || row.unsafe_owner != UNSAFE_OWNER
        {
            return Err(NpuPackageContractError::InvalidCatalog(format!(
                "library contract {} differs from the reviewed ABI",
                row.identity
            )));
        }
        contracts.push(if let Some(required_by) = &row.required_by {
            NativeFfiContract::new_dependency(
                row.identity.clone(),
                row.sha256.clone(),
                row.abi.clone(),
                required_by.clone(),
                row.unsafe_owner.clone(),
            )?
        } else {
            NativeFfiContract::new(
                row.identity.clone(),
                row.sha256.clone(),
                row.abi.clone(),
                row.required_symbols.clone(),
                row.unsafe_owner.clone(),
            )?
        });
        identities.insert(
            row.identity.clone(),
            NpuFfiLibraryIdentity {
                filename: row.filename.clone(),
            },
        );
    }
    if identities.len() != abi.libraries.len() {
        return Err(NpuPackageContractError::InvalidCatalog(
            "catalog does not cover every reviewed library".to_owned(),
        ));
    }
    Ok((NativeFfiRegistry::new(contracts)?, identities))
}

fn map_package_admission_error(error: NativePackageAdmissionError) -> NpuPackageContractError {
    match error {
        NativePackageAdmissionError::Cancelled => CancellationError.into(),
        NativePackageAdmissionError::UnsafePackage(reason) => {
            NpuPackageContractError::UnsafePackage(reason)
        }
        NativePackageAdmissionError::InvalidCoverage(reason) => {
            NpuPackageContractError::InvalidPackage(reason)
        }
    }
}

fn required_payload<'a>(
    payloads: &'a BTreeMap<String, Vec<u8>>,
    path: &str,
) -> Result<&'a [u8], NpuPackageContractError> {
    payloads.get(path).map(Vec::as_slice).ok_or_else(|| {
        NpuPackageContractError::UnsafePackage(format!("required payload is missing: {path}"))
    })
}

fn parse_strict_json<T: DeserializeOwned>(
    bytes: &[u8],
    label: &str,
) -> Result<T, NpuPackageContractError> {
    let strict = crate::trust::parse_strict_json_value(bytes).map_err(|error| {
        NpuPackageContractError::InvalidPackage(format!("{label} is not strict JSON: {error}"))
    })?;
    serde_json::from_value(strict).map_err(|error| {
        NpuPackageContractError::InvalidPackage(format!("{label} is invalid: {error}"))
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
            | ("aarch64", "linux", "aarch64-unknown-linux-gnu")
    )
}

fn map_native_library_image_error(
    library_id: &str,
    error: NativeLibraryImageError,
) -> NpuCertificationError {
    match error {
        NativeLibraryImageError::Cancelled => NpuCertificationError::Cancelled(CancellationError),
        NativeLibraryImageError::UnsupportedPlatform => NpuCertificationError::UnsupportedPlatform,
        NativeLibraryImageError::Invalid(reason) => NpuCertificationError::InvalidImage {
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
    use std::io;

    const SIGNING_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";

    fn signed_receipt(
        signer: &str,
        coverage: &[u8],
        key_pair: &Ed25519KeyPair,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let payload = crate::npu_package_signing_payload(signer, coverage)?;
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

    fn fixture_catalog(target: &str, ascendcl_digest: &str, runtime_digest: &str) -> Value {
        let abi: Value = serde_json::from_str(ABI_MANIFEST).expect("embedded ABI is valid");
        let libraries = abi["libraries"]
            .as_array()
            .expect("libraries are an array")
            .iter()
            .map(|library| {
                let identity = library["id"].as_str().expect("id is text");
                let mut row = json!({
                    "identity": identity,
                    "filename": library["filename"],
                    "sha256": if identity == "ascendcl" { ascendcl_digest } else { runtime_digest },
                    "abi": ABI_FLOOR,
                    "required_symbols": library["symbols"].as_array().expect("symbols").iter().map(|symbol| symbol["name"].clone()).collect::<Vec<_>>(),
                    "unsafe_owner": UNSAFE_OWNER,
                });
                if identity == "runtime" {
                    row["required_by"] = json!("ascendcl");
                }
                row
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": 1,
            "backend": "npu",
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
    ) -> Result<NpuPackageVerificationKey, Box<dyn std::error::Error>> {
        fs::create_dir_all(root.join("abi"))?;
        fs::write(root.join("LICENSES"), sim_backend_npu::PACKAGE_LICENSES)?;
        fs::write(root.join("abi/symbols-v1.json"), ABI_MANIFEST)?;
        fs::write(root.join("package-policy.json"), PACKAGE_POLICY)?;
        let mut catalog_bytes = serde_json::to_vec_pretty(catalog)?;
        catalog_bytes.push(b'\n');
        fs::write(root.join("ffi-contracts-v1.json"), &catalog_bytes)?;
        let manifest = json!({
            "schema_version": 1,
            "backend": "npu",
            "abi_floor": ABI_FLOOR,
            "target": target,
            "libraries": ["libascendcl.so", "libruntime.so"],
            "discovery_order": ["SIM_ASCEND_ROOT", "ASCEND_HOME_PATH", "signed_package_roots"],
            "abi_manifest_sha256": sha256_hex(ABI_MANIFEST.as_bytes()),
            "ffi_contracts_sha256": sha256_hex(&catalog_bytes),
            "package_policy_sha256": sha256_hex(PACKAGE_POLICY.as_bytes()),
            "redistributes_vendor_runtime": false,
            "license_approval_required_for_vendor_runtime": true,
            "runtime_compilation_forbidden": true,
            "signer": "npu.release",
            "signature_algorithm": "ed25519",
            "signature_domain": "zed-sim-npu-package-v1",
            "signature_coverage": "package-coverage-v1",
            "certificate_owner": "sim_runtime::NativeFfiRegistry",
            "unsafe_owner": UNSAFE_OWNER,
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
            signed_receipt("npu.release", &coverage, key_pair)?,
        )?;
        NpuPackageVerificationKey::new("npu.release", key_pair.public_key().as_ref())
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
            verify_npu_package_contracts(root.path(), &key, &CancellationToken::default())?;
        assert_eq!(verified.identities().len(), 2);
        assert_eq!(
            verified
                .registry()
                .required_symbols_for("ascendcl", ABI_FLOOR, UNSAFE_OWNER)?
                .len(),
            28
        );
        let runtime_digest = verified.identities()["runtime"].filename();
        assert_eq!(runtime_digest, "libruntime.so");

        fs::write(root.path().join("ffi-contracts-v1.json"), b"{}\n")?;
        assert!(
            verify_npu_package_contracts(root.path(), &key, &CancellationToken::default()).is_err()
        );
        Ok(())
    }

    #[test]
    fn dependency_only_runtime_contract_cannot_authorize_a_callable_surface()
    -> Result<(), Box<dyn std::error::Error>> {
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let registry = NativeFfiRegistry::new([
            NativeFfiContract::new(
                "ascendcl",
                digest,
                ABI_FLOOR,
                ["aclInit".to_owned()],
                UNSAFE_OWNER,
            )?,
            NativeFfiContract::new_dependency(
                "runtime",
                digest,
                ABI_FLOOR,
                "ascendcl",
                UNSAFE_OWNER,
            )?,
        ])?;
        assert!(
            registry
                .required_symbols_for("runtime", ABI_FLOOR, UNSAFE_OWNER)
                .is_err()
        );
        registry.authorize_dependency("runtime", digest, ABI_FLOOR, "ascendcl")?;
        assert!(
            registry
                .authorize_dependency("runtime", digest, ABI_FLOOR, "other")
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn npu_signature_domain_cannot_be_cross_authorized() -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let coverage = b"fixture coverage\n";
        let receipt = signed_receipt("npu.release", coverage, &key_pair)?;
        let npu = NpuPackageVerificationKey::new("npu.release", key_pair.public_key().as_ref())?;
        npu.verify_package("npu.release", coverage, &receipt)?;
        let metal =
            crate::MetalPackageVerificationKey::new("npu.release", key_pair.public_key().as_ref())?;
        assert_eq!(
            metal.verify_package("npu.release", coverage, &receipt),
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
        for (index, mutate) in ["reverse", "empty", "owner", "target"]
            .into_iter()
            .enumerate()
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
                verify_npu_package_contracts(root.path(), &key, &CancellationToken::default());
            assert!(result.is_err(), "mutation {index} was accepted");
        }
        Ok(())
    }

    #[test]
    fn npu_settings_round_trip_contains_only_public_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let settings = crate::NativeNpuPackageSettings::from_public_authority(
            "/opt/zed/npu-package",
            "npu.release",
            &key_pair
                .public_key()
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        )?;
        let encoded = serde_json::to_vec(&settings)?;
        let decoded: crate::NativeNpuPackageSettings = serde_json::from_slice(&encoded)?;
        assert_eq!(decoded, settings);
        assert!(!String::from_utf8(encoded)?.contains("private"));
        Ok(())
    }
}
