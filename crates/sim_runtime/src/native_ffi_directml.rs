use crate::{
    CertifiedNativeFfi, DirectMlPackageVerificationKey, NativeFfiContract, NativeFfiRegistry,
    TrustError,
    trust::{
        NativePackageAdmissionError, NativePackagePayloadLimit, capture_native_package,
        validate_native_package_coverage,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_backend_directml::{
    ABI_FLOOR, ABI_MANIFEST_JSON, AbiManifest, DirectMlCandidateObservation, DirectMlDiscoveryPlan,
    DirectMlExecutionError, DirectMlExecutionSession, DirectMlLoadError, DiscoverySource,
    RegistryCertifiedDirectMlImage, RetainedDirectMlLibraryHandles, UNSAFE_OWNER,
    observe_directml_candidate, probe_certified, validate_candidate_observation,
};
use sim_model::ArtifactRoot;
use sim_types::{BackendUnavailable, CancellationError, CancellationToken, DeviceKind};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use std::{
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
};

const PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/directml/package-policy.json");
const MAX_PACKAGE_FILE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_COVERAGE_BYTES: usize = 64 * 1024;
const MAX_PACKAGE_PAYLOADS: usize = 24;
const MAX_PACKAGE_BYTES: usize = 96 * 1024 * 1024;
#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
const MAX_NATIVE_LIBRARY_BYTES: u64 = 512 * 1024 * 1024;
#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
const COPY_CHUNK_BYTES: usize = 64 * 1024;
const PACKAGE_PAYLOAD_LIMITS: [NativePackagePayloadLimit; 11] = [
    NativePackagePayloadLimit::new("DirectML.dll", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("LICENSE-CODE.txt", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("LICENSE.txt", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("LICENSES", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("ThirdPartyNotices.txt", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("abi/symbols-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.sig", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("ffi-contracts-v1.json", MAX_CATALOG_BYTES),
    NativePackagePayloadLimit::new("package-coverage.sha256", MAX_COVERAGE_BYTES),
    NativePackagePayloadLimit::new("package-policy.json", MAX_PACKAGE_FILE_BYTES),
];
const COVERAGE_EXCLUDES: [&str; 2] = ["adapter-manifest.sig", "package-coverage.sha256"];
const SUPPORTED_TARGETS: [&str; 2] = ["aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc"];
const D3D12_LIBRARY_ID: &str = "D3D12.dll";
const DIRECTML_LIBRARY_ID: &str = "DirectML.dll";
const DXGI_LIBRARY_ID: &str = "DXGI.dll";

#[cfg(test)]
thread_local! {
    static REVIEWED_DIRECTML_PAYLOAD_OVERRIDE:
        std::cell::RefCell<Option<(String, u64, String)>> = const {
            std::cell::RefCell::new(None)
        };
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DirectMlFfiContractCatalogDto {
    abi_floor: String,
    abi_manifest_sha256: String,
    backend: String,
    libraries: Vec<DirectMlFfiContractDto>,
    package_policy_sha256: String,
    schema_version: u16,
    target: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DirectMlFfiContractDto {
    abi: String,
    filename: String,
    identity: String,
    required_symbols: Vec<String>,
    sha256: String,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DirectMlPackageManifestDto {
    abi_floor: String,
    abi_manifest_sha256: String,
    backend: String,
    certificate_owner: String,
    directml_dll_byte_length: u64,
    directml_dll_file_version: String,
    directml_dll_sha256: String,
    ffi_contracts_sha256: String,
    package_policy_sha256: String,
    registry_authorization_required: bool,
    runtime_compilation_forbidden: bool,
    schema_version: u16,
    signature_algorithm: String,
    signature_coverage: String,
    signature_domain: String,
    signer: String,
    source_package: String,
    source_package_sha256: String,
    target: String,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlFfiLibraryIdentity {
    filename: String,
    digest_sha256: String,
    abi_version: String,
    required_symbols: BTreeSet<String>,
}

impl DirectMlFfiLibraryIdentity {
    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn abi_version(&self) -> &str {
        &self.abi_version
    }

    pub fn required_symbols(&self) -> &BTreeSet<String> {
        &self.required_symbols
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedDirectMlFfiContracts {
    package_root: ArtifactRoot,
    target: String,
    registry: NativeFfiRegistry,
    identities: BTreeMap<String, DirectMlFfiLibraryIdentity>,
    #[cfg(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    directml_image: Arc<[u8]>,
}

impl VerifiedDirectMlFfiContracts {
    pub fn package_root(&self) -> &ArtifactRoot {
        &self.package_root
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn registry(&self) -> &NativeFfiRegistry {
        &self.registry
    }

    pub fn identities(&self) -> &BTreeMap<String, DirectMlFfiLibraryIdentity> {
        &self.identities
    }
}

#[derive(Debug, Error)]
pub enum DirectMlPackageContractError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("DirectML package tree is unsafe or incomplete: {0}")]
    UnsafePackage(String),
    #[error("signed DirectML package metadata is invalid: {0}")]
    InvalidPackage(String),
    #[error("signed DirectML FFI contract catalog is invalid: {0}")]
    InvalidCatalog(String),
}

#[derive(Debug, Error)]
pub enum DirectMlCertificationError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error(transparent)]
    Loader(#[from] DirectMlLoadError),
    #[error("DirectML discovery plan differs from the signed package: {0}")]
    InvalidDiscovery(String),
    #[error("DirectML library image {library_id} is invalid: {reason}")]
    InvalidImage { library_id: String, reason: String },
    #[error("sealed DirectML images require a supported Windows MSVC target")]
    UnsupportedPlatform,
}

struct DirectMlCertificationRetention {
    _verified: VerifiedDirectMlFfiContracts,
    _certificates: Vec<CertifiedNativeFfi>,
    _sealed_images: SealedDirectMlImages,
}

struct SealedDirectMlImages {
    _files: Vec<File>,
    #[cfg(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    _directory: tempfile::TempDir,
}

pub struct CertifiedDirectMlLoad {
    handles: RetainedDirectMlLibraryHandles,
}

impl CertifiedDirectMlLoad {
    pub fn load_execution_session(
        self,
    ) -> Result<DirectMlExecutionSession, DirectMlExecutionError> {
        DirectMlExecutionSession::from_registry_certified_handles(self.handles)
    }
}

pub fn verify_directml_package_contracts(
    package_root: &Path,
    verification_key: &DirectMlPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedDirectMlFfiContracts, DirectMlPackageContractError> {
    cancellation.check()?;
    let root = ArtifactRoot::canonical(
        "sim-directml-package",
        "native-ffi-package",
        package_root,
        std::iter::empty::<String>(),
    )
    .map_err(|error| DirectMlPackageContractError::UnsafePackage(error.to_string()))?;
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
        return Err(DirectMlPackageContractError::InvalidPackage(
            "installed package policy differs from the compiled reviewed policy".to_owned(),
        ));
    }

    let manifest: DirectMlPackageManifestDto = parse_canonical_json(
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

    let catalog: DirectMlFfiContractCatalogDto = parse_canonical_json(
        required_payload(&payloads, "ffi-contracts-v1.json")?,
        "FFI contract catalog",
    )?;
    let (registry, identities) = validate_and_map_catalog(&catalog, &manifest)?;
    cancellation.check()?;
    Ok(VerifiedDirectMlFfiContracts {
        package_root: root,
        target: catalog.target,
        registry,
        identities,
        #[cfg(all(
            target_os = "windows",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        directml_image: Arc::from(required_payload(&payloads, DIRECTML_LIBRARY_ID)?),
    })
}

pub fn certify_directml_library_images(
    verified: VerifiedDirectMlFfiContracts,
    discovery: &DirectMlDiscoveryPlan,
    observation: &DirectMlCandidateObservation,
    cancellation: &CancellationToken,
) -> Result<CertifiedDirectMlLoad, DirectMlCertificationError> {
    cancellation.check()?;
    if discovery.target() != verified.target() {
        return Err(DirectMlCertificationError::InvalidDiscovery(
            "discovery target differs from the signed catalog".to_owned(),
        ));
    }
    discovery.validate_system_directory()?;
    let package_candidate = discovery.candidates().first().ok_or_else(|| {
        DirectMlCertificationError::InvalidDiscovery(
            "application-package candidate is missing".to_owned(),
        )
    })?;
    let expected_package_path = verified
        .package_root()
        .canonical_path()
        .join(DIRECTML_LIBRARY_ID);
    if package_candidate.source() != DiscoverySource::SignedApplicationPackage
        || package_candidate.path() != expected_package_path
        || observation.source() != DiscoverySource::SignedApplicationPackage
        || observation.path() != expected_package_path
    {
        return Err(DirectMlCertificationError::InvalidDiscovery(
            "application-package candidate or observation is outside the verified package root"
                .to_owned(),
        ));
    }
    let system_candidate = discovery.candidates().get(1).ok_or_else(|| {
        DirectMlCertificationError::InvalidDiscovery("system candidate is missing".to_owned())
    })?;
    if system_candidate.source() != DiscoverySource::CompatibleSystemComponent {
        return Err(DirectMlCertificationError::InvalidDiscovery(
            "second candidate is not the compatible system component".to_owned(),
        ));
    }
    let system_directory = system_candidate.path().parent().ok_or_else(|| {
        DirectMlCertificationError::InvalidDiscovery(
            "system DirectML candidate has no parent directory".to_owned(),
        )
    })?;
    let directml_identity = verified
        .identities()
        .get(DIRECTML_LIBRARY_ID)
        .ok_or_else(|| DirectMlCertificationError::InvalidImage {
            library_id: DIRECTML_LIBRARY_ID.to_owned(),
            reason: "signed DirectML identity is missing".to_owned(),
        })?;
    validate_candidate_observation(discovery, observation, directml_identity.digest_sha256())?;
    let (sealed_images, retained_paths) =
        seal_directml_images(&verified, system_directory, cancellation)?;

    let mut certificates = Vec::with_capacity(verified.identities().len());
    let mut projected_images = Vec::with_capacity(verified.identities().len());
    for (library_id, identity) in verified.identities() {
        cancellation.check()?;
        let path = retained_paths.get(library_id).ok_or_else(|| {
            DirectMlCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "sealed image path is missing".to_owned(),
            }
        })?;
        let certificate = verified.registry().authorize(
            library_id,
            identity.digest_sha256(),
            identity.abi_version(),
            identity.required_symbols(),
        )?;
        let image = unsafe {
            RegistryCertifiedDirectMlImage::load_from_registry_certificate(
                certificate.library_id(),
                certificate.digest_sha256(),
                certificate.abi_version(),
                certificate.required_symbols().clone(),
                certificate.unsafe_owner(),
                path,
            )
        }?;
        certificates.push(certificate);
        projected_images.push(image);
    }
    let retained_certificates = certificates.clone();
    let retention = Arc::new(DirectMlCertificationRetention {
        _verified: verified,
        _certificates: retained_certificates,
        _sealed_images: sealed_images,
    });
    let handles = unsafe {
        RetainedDirectMlLibraryHandles::from_registry_certificates(retention, projected_images)
    }?;
    probe_certified(&handles, discovery, observation)?;
    cancellation.check()?;
    Ok(CertifiedDirectMlLoad { handles })
}

pub fn initialize_certified_directml_runtime(
    settings: &crate::NativeDirectMlPackageSettings,
    cancellation: &CancellationToken,
) -> Result<DirectMlExecutionSession, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::DirectMl, reason);
    let verified = verify_directml_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed DirectML package or contract verification failed"))?;
    let directml_digest = verified
        .identities()
        .get(DIRECTML_LIBRARY_ID)
        .map(|identity| identity.digest_sha256().to_owned())
        .ok_or_else(|| unavailable("signed DirectML library identity is missing"))?;
    let discovery = DirectMlDiscoveryPlan::for_current_system(
        verified.target(),
        verified.package_root().canonical_path(),
    )
    .map_err(|_| unavailable("exact DirectML system discovery failed"))?;
    cancellation
        .check()
        .map_err(|_| unavailable("DirectML initialization was cancelled"))?;
    let observation = observe_directml_candidate(&discovery, directml_digest)
        .map_err(|_| unavailable("DirectML host trust or version observation failed"))?;
    cancellation
        .check()
        .map_err(|_| unavailable("DirectML initialization was cancelled"))?;
    let certified =
        certify_directml_library_images(verified, &discovery, &observation, cancellation)
            .map_err(|_| unavailable("exact DirectML library certification failed"))?;
    certified
        .load_execution_session()
        .map_err(|_| unavailable("certified DirectML device session initialization failed"))
}

fn validate_manifest(
    manifest: &DirectMlPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), DirectMlPackageContractError> {
    let abi = AbiManifest::embedded()
        .map_err(|error| DirectMlPackageContractError::InvalidPackage(error.to_string()))?;
    let architecture = abi
        .redistributable
        .architectures
        .iter()
        .find(|architecture| architecture.target == manifest.target)
        .ok_or_else(|| {
            DirectMlPackageContractError::InvalidPackage(
                "adapter target has no reviewed redistributable".to_owned(),
            )
        })?;
    let directml = required_payload(payloads, DIRECTML_LIBRARY_ID)?;
    let (reviewed_byte_length, reviewed_digest) = reviewed_directml_payload(
        &manifest.target,
        architecture.byte_length,
        &architecture.sha256,
    );
    if manifest.schema_version != 2
        || manifest.backend != "directml"
        || manifest.abi_floor != ABI_FLOOR
        || !SUPPORTED_TARGETS.contains(&manifest.target.as_str())
        || manifest.abi_manifest_sha256
            != sha256_hex(required_payload(payloads, "abi/symbols-v1.json")?)
        || manifest.ffi_contracts_sha256
            != sha256_hex(required_payload(payloads, "ffi-contracts-v1.json")?)
        || manifest.package_policy_sha256
            != sha256_hex(required_payload(payloads, "package-policy.json")?)
        || manifest.directml_dll_byte_length != u64::try_from(directml.len()).unwrap_or(u64::MAX)
        || manifest.directml_dll_byte_length != reviewed_byte_length
        || manifest.directml_dll_file_version != abi.redistributable.file_version
        || manifest.directml_dll_sha256 != sha256_hex(directml)
        || manifest.directml_dll_sha256 != reviewed_digest
        || !manifest.registry_authorization_required
        || !manifest.runtime_compilation_forbidden
        || manifest.signature_algorithm != "ed25519"
        || manifest.signature_coverage != "package-coverage-v1"
        || manifest.signature_domain != "zed-sim-directml-package-v1"
        || manifest.certificate_owner != "sim_runtime::NativeFfiRegistry"
        || manifest.unsafe_owner != UNSAFE_OWNER
        || manifest.source_package != "Microsoft.AI.DirectML/1.13.1"
        || manifest.source_package_sha256 != abi.reviewed_package.nupkg_sha256
        || !valid_signer(&manifest.signer)
    {
        return Err(DirectMlPackageContractError::InvalidPackage(
            "adapter manifest differs from the reviewed DirectML package contract".to_owned(),
        ));
    }
    if required_payload(payloads, "abi/symbols-v1.json")? != ABI_MANIFEST_JSON.as_bytes() {
        return Err(DirectMlPackageContractError::InvalidPackage(
            "packaged ABI manifest differs from the compiled reviewed manifest".to_owned(),
        ));
    }
    Ok(())
}

fn reviewed_directml_payload(_target: &str, byte_length: u64, digest: &str) -> (u64, String) {
    #[cfg(test)]
    if let Some((reviewed_target, reviewed_length, reviewed_digest)) =
        REVIEWED_DIRECTML_PAYLOAD_OVERRIDE.with(|slot| slot.borrow().clone())
        && reviewed_target == _target
    {
        return (reviewed_length, reviewed_digest);
    }
    (byte_length, digest.to_owned())
}

fn validate_and_map_catalog(
    catalog: &DirectMlFfiContractCatalogDto,
    manifest: &DirectMlPackageManifestDto,
) -> Result<
    (
        NativeFfiRegistry,
        BTreeMap<String, DirectMlFfiLibraryIdentity>,
    ),
    DirectMlPackageContractError,
> {
    let abi = AbiManifest::embedded()
        .map_err(|error| DirectMlPackageContractError::InvalidCatalog(error.to_string()))?;
    if catalog.schema_version != 1
        || catalog.backend != "directml"
        || catalog.abi_floor != ABI_FLOOR
        || catalog.target != manifest.target
        || catalog.abi_manifest_sha256 != manifest.abi_manifest_sha256
        || catalog.package_policy_sha256 != manifest.package_policy_sha256
        || catalog.libraries.len() != abi.libraries.len()
    {
        return Err(DirectMlPackageContractError::InvalidCatalog(
            "catalog envelope is unsupported, target-mismatched, or incomplete".to_owned(),
        ));
    }

    let mut previous_identity: Option<&str> = None;
    let mut contracts = Vec::with_capacity(catalog.libraries.len());
    let mut identities = BTreeMap::new();
    for row in &catalog.libraries {
        if previous_identity.is_some_and(|previous| previous >= row.identity.as_str()) {
            return Err(DirectMlPackageContractError::InvalidCatalog(
                "library identities must be sorted and unique".to_owned(),
            ));
        }
        previous_identity = Some(&row.identity);
        let reviewed = abi
            .libraries
            .iter()
            .find(|library| library.name == row.identity)
            .ok_or_else(|| {
                DirectMlPackageContractError::InvalidCatalog(format!(
                    "unknown library identity {}",
                    row.identity
                ))
            })?;
        let expected_symbols = reviewed
            .symbols
            .iter()
            .map(|symbol| symbol.name.clone())
            .collect::<Vec<_>>();
        if row.filename != reviewed.name
            || !valid_lower_hex_digest(&row.sha256)
            || row.abi != reviewed.abi_version
            || row.required_symbols != expected_symbols
            || row.required_symbols.is_empty()
            || row
                .required_symbols
                .windows(2)
                .any(|pair| matches!(pair, [first, second] if first >= second))
            || row.unsafe_owner != UNSAFE_OWNER
            || row.identity == DIRECTML_LIBRARY_ID && row.sha256 != manifest.directml_dll_sha256
        {
            return Err(DirectMlPackageContractError::InvalidCatalog(format!(
                "library contract {} differs from the reviewed ABI",
                row.identity
            )));
        }
        let required_symbols = row
            .required_symbols
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        contracts.push(NativeFfiContract::new(
            row.identity.clone(),
            row.sha256.clone(),
            row.abi.clone(),
            required_symbols.clone(),
            row.unsafe_owner.clone(),
        )?);
        identities.insert(
            row.identity.clone(),
            DirectMlFfiLibraryIdentity {
                filename: row.filename.clone(),
                digest_sha256: row.sha256.clone(),
                abi_version: row.abi.clone(),
                required_symbols,
            },
        );
    }
    if identities.len() != abi.libraries.len()
        || ![D3D12_LIBRARY_ID, DIRECTML_LIBRARY_ID, DXGI_LIBRARY_ID]
            .iter()
            .all(|identity| identities.contains_key(*identity))
    {
        return Err(DirectMlPackageContractError::InvalidCatalog(
            "catalog does not cover every reviewed library".to_owned(),
        ));
    }
    Ok((NativeFfiRegistry::new(contracts)?, identities))
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn seal_directml_images(
    verified: &VerifiedDirectMlFfiContracts,
    system_directory: &Path,
    cancellation: &CancellationToken,
) -> Result<(SealedDirectMlImages, BTreeMap<String, PathBuf>), DirectMlCertificationError> {
    let mut bytes = BTreeMap::from([(
        DIRECTML_LIBRARY_ID.to_owned(),
        verified.directml_image.to_vec(),
    )]);
    for library_id in [D3D12_LIBRARY_ID, DXGI_LIBRARY_ID] {
        cancellation.check()?;
        bytes.insert(
            library_id.to_owned(),
            read_stable_windows_image(
                library_id,
                &system_directory.join(library_id),
                cancellation,
            )?,
        );
    }
    for (library_id, identity) in verified.identities() {
        let image =
            bytes
                .get(library_id)
                .ok_or_else(|| DirectMlCertificationError::InvalidImage {
                    library_id: library_id.clone(),
                    reason: "captured image is missing".to_owned(),
                })?;
        if sha256_hex_cancellable(image, cancellation)? != identity.digest_sha256() {
            return Err(DirectMlCertificationError::InvalidImage {
                library_id: library_id.clone(),
                reason: "image digest differs from the signed contract".to_owned(),
            });
        }
    }

    let directory =
        tempfile::tempdir().map_err(|error| DirectMlCertificationError::InvalidImage {
            library_id: "sealed-image-directory".to_owned(),
            reason: error.to_string(),
        })?;
    let mut retained_files = Vec::with_capacity(bytes.len());
    let mut retained_paths = BTreeMap::new();
    for (library_id, image) in bytes {
        cancellation.check()?;
        let path = directory.path().join(&library_id);
        let file = create_retained_windows_snapshot(&library_id, &path, &image, cancellation)?;
        retained_files.push(file);
        retained_paths.insert(library_id, path);
    }
    Ok((
        SealedDirectMlImages {
            _directory: directory,
            _files: retained_files,
        },
        retained_paths,
    ))
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn seal_directml_images(
    _verified: &VerifiedDirectMlFfiContracts,
    _system_directory: &Path,
    _cancellation: &CancellationToken,
) -> Result<(SealedDirectMlImages, BTreeMap<String, PathBuf>), DirectMlCertificationError> {
    Err(DirectMlCertificationError::UnsupportedPlatform)
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn read_stable_windows_image(
    library_id: &str,
    path: &Path,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, DirectMlCertificationError> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

    let mut file = OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN)
        .open(path)
        .map_err(|error| DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: error.to_string(),
        })?;
    let metadata = file
        .metadata()
        .map_err(|error| DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: error.to_string(),
        })?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > MAX_NATIVE_LIBRARY_BYTES
    {
        return Err(DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: "image is not a bounded nonempty regular file".to_owned(),
        });
    }
    let first = read_bounded_file(library_id, &mut file, metadata.len(), cancellation)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: error.to_string(),
        })?;
    let second = read_bounded_file(library_id, &mut file, metadata.len(), cancellation)?;
    if first != second {
        return Err(DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: "image changed while it was captured".to_owned(),
        });
    }
    Ok(first)
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn create_retained_windows_snapshot(
    library_id: &str,
    path: &Path,
    bytes: &[u8],
    cancellation: &CancellationToken,
) -> Result<File, DirectMlCertificationError> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x0000_0100;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_SEQUENTIAL_SCAN)
        .open(path)
        .map_err(|error| DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: error.to_string(),
        })?;
    for chunk in bytes.chunks(COPY_CHUNK_BYTES) {
        cancellation.check()?;
        file.write_all(chunk)
            .map_err(|error| DirectMlCertificationError::InvalidImage {
                library_id: library_id.to_owned(),
                reason: error.to_string(),
            })?;
    }
    file.sync_all()
        .map_err(|error| DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: error.to_string(),
        })?;
    cancellation.check()?;
    Ok(file)
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn read_bounded_file(
    library_id: &str,
    file: &mut File,
    expected_bytes: u64,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, DirectMlCertificationError> {
    let capacity =
        usize::try_from(expected_bytes).map_err(|_| DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: "image length exceeds addressable memory".to_owned(),
        })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|error| {
        DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: format!("image buffer allocation failed: {error}"),
        }
    })?;
    let mut chunk = [0_u8; COPY_CHUNK_BYTES];
    loop {
        cancellation.check()?;
        let count =
            file.read(&mut chunk)
                .map_err(|error| DirectMlCertificationError::InvalidImage {
                    library_id: library_id.to_owned(),
                    reason: error.to_string(),
                })?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_NATIVE_LIBRARY_BYTES {
            return Err(DirectMlCertificationError::InvalidImage {
                library_id: library_id.to_owned(),
                reason: "image exceeds the native-library byte bound".to_owned(),
            });
        }
    }
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != expected_bytes {
        return Err(DirectMlCertificationError::InvalidImage {
            library_id: library_id.to_owned(),
            reason: "image length changed while it was captured".to_owned(),
        });
    }
    Ok(bytes)
}

fn map_package_admission_error(error: NativePackageAdmissionError) -> DirectMlPackageContractError {
    match error {
        NativePackageAdmissionError::Cancelled => CancellationError.into(),
        NativePackageAdmissionError::UnsafePackage(reason) => {
            DirectMlPackageContractError::UnsafePackage(reason)
        }
        NativePackageAdmissionError::InvalidCoverage(reason) => {
            DirectMlPackageContractError::InvalidPackage(reason)
        }
    }
}

fn required_payload<'a>(
    payloads: &'a BTreeMap<String, Vec<u8>>,
    path: &str,
) -> Result<&'a [u8], DirectMlPackageContractError> {
    payloads.get(path).map(Vec::as_slice).ok_or_else(|| {
        DirectMlPackageContractError::UnsafePackage(format!("required payload is missing: {path}"))
    })
}

fn parse_strict_json<T: DeserializeOwned>(
    bytes: &[u8],
    label: &str,
) -> Result<T, DirectMlPackageContractError> {
    let strict = crate::trust::parse_strict_json_value(bytes).map_err(|error| {
        DirectMlPackageContractError::InvalidPackage(format!("{label} is not strict JSON: {error}"))
    })?;
    serde_json::from_value(strict).map_err(|error| {
        DirectMlPackageContractError::InvalidPackage(format!("{label} is invalid: {error}"))
    })
}

fn parse_canonical_json<T: DeserializeOwned + Serialize>(
    bytes: &[u8],
    label: &str,
) -> Result<T, DirectMlPackageContractError> {
    let value = parse_strict_json(bytes, label)?;
    let mut canonical = serde_json::to_vec_pretty(&value).map_err(|error| {
        DirectMlPackageContractError::InvalidPackage(format!(
            "{label} cannot be encoded canonically: {error}"
        ))
    })?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(DirectMlPackageContractError::InvalidPackage(format!(
            "{label} is not canonical sorted JSON"
        )));
    }
    Ok(value)
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

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn sha256_hex_cancellable(
    bytes: &[u8],
    cancellation: &CancellationToken,
) -> Result<String, CancellationError> {
    let mut hasher = Sha256::new();
    for chunk in bytes.chunks(COPY_CHUNK_BYTES) {
        cancellation.check()?;
        hasher.update(chunk);
    }
    cancellation.check()?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use std::fs;

    const SIGNING_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
    const TARGET: &str = "x86_64-pc-windows-msvc";

    struct PayloadOverride;

    impl Drop for PayloadOverride {
        fn drop(&mut self) {
            REVIEWED_DIRECTML_PAYLOAD_OVERRIDE.with(|slot| {
                *slot.borrow_mut() = None;
            });
        }
    }

    fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut bytes = serde_json::to_vec_pretty(value)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn fixture_catalog(
        directml_digest: &str,
        unsorted: bool,
    ) -> Result<DirectMlFfiContractCatalogDto, Box<dyn std::error::Error>> {
        let abi = AbiManifest::embedded()?;
        let mut identities = [D3D12_LIBRARY_ID, DXGI_LIBRARY_ID, DIRECTML_LIBRARY_ID];
        if unsorted {
            identities.swap(0, 1);
        }
        let libraries = identities
            .iter()
            .map(|identity| {
                let reviewed = abi
                    .libraries
                    .iter()
                    .find(|library| library.name == *identity)
                    .ok_or("reviewed library is missing")?;
                Ok(DirectMlFfiContractDto {
                    abi: reviewed.abi_version.clone(),
                    filename: reviewed.name.clone(),
                    identity: reviewed.name.clone(),
                    required_symbols: reviewed
                        .symbols
                        .iter()
                        .map(|symbol| symbol.name.clone())
                        .collect(),
                    sha256: if *identity == DIRECTML_LIBRARY_ID {
                        directml_digest.to_owned()
                    } else if *identity == D3D12_LIBRARY_ID {
                        "1111111111111111111111111111111111111111111111111111111111111111"
                            .to_owned()
                    } else {
                        "2222222222222222222222222222222222222222222222222222222222222222"
                            .to_owned()
                    },
                    unsafe_owner: UNSAFE_OWNER.to_owned(),
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        Ok(DirectMlFfiContractCatalogDto {
            abi_floor: ABI_FLOOR.to_owned(),
            abi_manifest_sha256: sha256_hex(ABI_MANIFEST_JSON.as_bytes()),
            backend: "directml".to_owned(),
            libraries,
            package_policy_sha256: sha256_hex(PACKAGE_POLICY.as_bytes()),
            schema_version: 1,
            target: TARGET.to_owned(),
        })
    }

    fn coverage_for(
        root: &Path,
    ) -> Result<(Vec<u8>, BTreeMap<String, Vec<u8>>), Box<dyn std::error::Error>> {
        let mut payloads = BTreeMap::new();
        for limit in PACKAGE_PAYLOAD_LIMITS {
            payloads.insert(limit.path().to_owned(), fs::read(root.join(limit.path()))?);
        }
        let mut coverage = Vec::new();
        for (path, payload) in &payloads {
            if COVERAGE_EXCLUDES.binary_search(&path.as_str()).is_err() {
                coverage.extend_from_slice(
                    format!("{} {}  {path}\n", sha256_hex(payload), payload.len()).as_bytes(),
                );
            }
        }
        Ok((coverage, payloads))
    }

    fn write_signed_package(
        root: &Path,
        unsorted_catalog: bool,
        wrong_domain: bool,
    ) -> Result<DirectMlPackageVerificationKey, Box<dyn std::error::Error>> {
        let directml = b"deterministic-directml-fixture";
        let directml_digest = sha256_hex(directml);
        REVIEWED_DIRECTML_PAYLOAD_OVERRIDE.with(|slot| {
            *slot.borrow_mut() = Some((
                TARGET.to_owned(),
                u64::try_from(directml.len()).unwrap_or(u64::MAX),
                directml_digest.clone(),
            ));
        });
        fs::create_dir_all(root.join("abi"))?;
        fs::write(root.join(DIRECTML_LIBRARY_ID), directml)?;
        for path in [
            "LICENSE-CODE.txt",
            "LICENSE.txt",
            "LICENSES",
            "ThirdPartyNotices.txt",
        ] {
            fs::write(root.join(path), format!("{path}\n"))?;
        }
        fs::write(root.join("abi/symbols-v1.json"), ABI_MANIFEST_JSON)?;
        fs::write(root.join("package-policy.json"), PACKAGE_POLICY)?;
        let catalog = fixture_catalog(&directml_digest, unsorted_catalog)?;
        let catalog_bytes = canonical_json(&catalog)?;
        fs::write(root.join("ffi-contracts-v1.json"), &catalog_bytes)?;

        let manifest = DirectMlPackageManifestDto {
            abi_floor: ABI_FLOOR.to_owned(),
            abi_manifest_sha256: sha256_hex(ABI_MANIFEST_JSON.as_bytes()),
            backend: "directml".to_owned(),
            certificate_owner: "sim_runtime::NativeFfiRegistry".to_owned(),
            directml_dll_byte_length: u64::try_from(directml.len()).unwrap_or(u64::MAX),
            directml_dll_file_version: "1.13.1.0".to_owned(),
            directml_dll_sha256: directml_digest,
            ffi_contracts_sha256: sha256_hex(&catalog_bytes),
            package_policy_sha256: sha256_hex(PACKAGE_POLICY.as_bytes()),
            registry_authorization_required: true,
            runtime_compilation_forbidden: true,
            schema_version: 2,
            signature_algorithm: "ed25519".to_owned(),
            signature_coverage: "package-coverage-v1".to_owned(),
            signature_domain: "zed-sim-directml-package-v1".to_owned(),
            signer: "zed.release.directml-v1".to_owned(),
            source_package: "Microsoft.AI.DirectML/1.13.1".to_owned(),
            source_package_sha256: AbiManifest::embedded()?.reviewed_package.nupkg_sha256,
            target: TARGET.to_owned(),
            unsafe_owner: UNSAFE_OWNER.to_owned(),
        };
        fs::write(
            root.join("adapter-manifest.json"),
            canonical_json(&manifest)?,
        )?;
        fs::write(root.join("adapter-manifest.sig"), b"pending\n")?;
        fs::write(root.join("package-coverage.sha256"), b"pending\n")?;
        let (coverage, _) = coverage_for(root)?;
        fs::write(root.join("package-coverage.sha256"), &coverage)?;

        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| std::io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let signing_payload = if wrong_domain {
            crate::mlu_package_signing_payload(&manifest.signer, &coverage)?
        } else {
            crate::directml_package_signing_payload(&manifest.signer, &coverage)?
        };
        let signature = key_pair
            .sign(&signing_payload)
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        fs::write(
            root.join("adapter-manifest.sig"),
            format!(
                "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{signature}\"}}\n"
            ),
        )?;
        DirectMlPackageVerificationKey::new(manifest.signer, key_pair.public_key().as_ref())
            .map_err(Into::into)
    }

    fn refresh_signed_envelope(
        root: &Path,
    ) -> Result<DirectMlPackageVerificationKey, Box<dyn std::error::Error>> {
        let mut manifest: DirectMlPackageManifestDto =
            serde_json::from_slice(&fs::read(root.join("adapter-manifest.json"))?)?;
        manifest.abi_manifest_sha256 = sha256_hex(&fs::read(root.join("abi/symbols-v1.json"))?);
        manifest.ffi_contracts_sha256 = sha256_hex(&fs::read(root.join("ffi-contracts-v1.json"))?);
        manifest.package_policy_sha256 = sha256_hex(&fs::read(root.join("package-policy.json"))?);
        fs::write(
            root.join("adapter-manifest.json"),
            canonical_json(&manifest)?,
        )?;
        let (coverage, _) = coverage_for(root)?;
        fs::write(root.join("package-coverage.sha256"), &coverage)?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(SIGNING_KEY)
            .map_err(|error| std::io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let signing_payload = crate::directml_package_signing_payload(&manifest.signer, &coverage)?;
        let signature = key_pair
            .sign(&signing_payload)
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        fs::write(
            root.join("adapter-manifest.sig"),
            format!(
                "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{signature}\"}}\n"
            ),
        )?;
        DirectMlPackageVerificationKey::new(manifest.signer, key_pair.public_key().as_ref())
            .map_err(Into::into)
    }

    fn new_fixture() -> Result<
        (
            tempfile::TempDir,
            DirectMlPackageVerificationKey,
            PayloadOverride,
        ),
        Box<dyn std::error::Error>,
    > {
        let package = tempfile::tempdir()?;
        let key = write_signed_package(package.path(), false, false)?;
        Ok((package, key, PayloadOverride))
    }

    #[test]
    fn signed_catalog_constructs_exact_registry_only_after_package_verification()
    -> Result<(), Box<dyn std::error::Error>> {
        let _override = PayloadOverride;
        let package = tempfile::tempdir()?;
        let key = write_signed_package(package.path(), false, false)?;
        let verified =
            verify_directml_package_contracts(package.path(), &key, &CancellationToken::default())?;
        assert_eq!(verified.target(), TARGET);
        assert_eq!(
            verified
                .identities()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [D3D12_LIBRARY_ID, DXGI_LIBRARY_ID, DIRECTML_LIBRARY_ID]
        );
        for (library_id, identity) in verified.identities() {
            verified.registry().authorize(
                library_id,
                identity.digest_sha256(),
                identity.abi_version(),
                identity.required_symbols(),
            )?;
        }
        Ok(())
    }

    #[test]
    fn production_initializer_orders_trust_observation_certification_and_session() {
        let source = include_str!("native_ffi_directml.rs");
        let initializer = source
            .split_once("pub fn initialize_certified_directml_runtime(")
            .and_then(|(_, initializer)| initializer.split_once("\nfn validate_manifest("))
            .map(|(initializer, _)| initializer)
            .expect("production DirectML initializer is present");
        let verification = initializer
            .find("verify_directml_package_contracts(")
            .expect("package verification is present");
        let observation = initializer
            .find("observe_directml_candidate(")
            .expect("real host observation is present");
        let certification = initializer
            .find("certify_directml_library_images(")
            .expect("registry certification is present");
        let session = initializer
            .find(".load_execution_session()")
            .expect("session construction is present");
        assert!(verification < observation);
        assert!(observation < certification);
        assert!(certification < session);
        assert!(!initializer.contains("DirectMlCandidateObservation {"));
        assert!(!initializer.contains("authenticode_trusted"));
    }

    #[cfg(not(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    #[test]
    fn production_initializer_is_typed_unavailable_without_windows_observation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (package, key, _override) = new_fixture()?;
        let public_key = key
            .public_key_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let settings = crate::NativeDirectMlPackageSettings::from_public_authority(
            package.path(),
            key.signer(),
            &public_key,
        )?;
        let unavailable =
            initialize_certified_directml_runtime(&settings, &CancellationToken::default())
                .expect_err("non-Windows host cannot synthesize DirectML observation");
        assert_eq!(unavailable.device(), DeviceKind::DirectMl);
        assert_eq!(
            unavailable.reason(),
            "exact DirectML system discovery failed"
        );
        Ok(())
    }

    #[test]
    fn signed_catalog_rejects_wrong_domain_unsorted_rows_and_tamper()
    -> Result<(), Box<dyn std::error::Error>> {
        let _override = PayloadOverride;
        let wrong_domain = tempfile::tempdir()?;
        let key = write_signed_package(wrong_domain.path(), false, true)?;
        assert!(matches!(
            verify_directml_package_contracts(
                wrong_domain.path(),
                &key,
                &CancellationToken::default()
            ),
            Err(DirectMlPackageContractError::Trust(
                TrustError::InvalidDirectMlPackageSignature
            ))
        ));

        let unsorted = tempfile::tempdir()?;
        let key = write_signed_package(unsorted.path(), true, false)?;
        assert!(matches!(
            verify_directml_package_contracts(unsorted.path(), &key, &CancellationToken::default()),
            Err(DirectMlPackageContractError::InvalidCatalog(_))
        ));

        let tampered = tempfile::tempdir()?;
        let key = write_signed_package(tampered.path(), false, false)?;
        fs::write(
            tampered.path().join("ffi-contracts-v1.json"),
            b"{\"tampered\":true}\n",
        )?;
        assert!(
            verify_directml_package_contracts(tampered.path(), &key, &CancellationToken::default())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn package_and_catalog_admission_fail_closed_for_structural_and_policy_mutations()
    -> Result<(), Box<dyn std::error::Error>> {
        let assert_rejected = |root: &Path,
                               key: &DirectMlPackageVerificationKey,
                               cancellation: &CancellationToken| {
            assert!(
                verify_directml_package_contracts(root, key, cancellation).is_err(),
                "mutated package was admitted: {}",
                root.display()
            );
        };

        let (package, key, _override) = new_fixture()?;
        fs::remove_file(package.path().join("LICENSES"))?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, key, _override) = new_fixture()?;
        fs::write(package.path().join("unexpected.bin"), b"extra")?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, key, _override) = new_fixture()?;
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_rejected(package.path(), &key, &cancellation);

        let (package, _, _override) = new_fixture()?;
        let unknown_signer =
            DirectMlPackageVerificationKey::new("different.authority", SIGNING_KEY)?;
        assert!(matches!(
            verify_directml_package_contracts(
                package.path(),
                &unknown_signer,
                &CancellationToken::default()
            ),
            Err(DirectMlPackageContractError::Trust(
                TrustError::UnknownDirectMlPackageSigner
            ))
        ));

        let (package, _, _override) = new_fixture()?;
        let mut policy: Value =
            serde_json::from_slice(&fs::read(package.path().join("package-policy.json"))?)?;
        policy["minimum_windows_build"] = Value::from(19_042);
        fs::write(
            package.path().join("package-policy.json"),
            canonical_json(&policy)?,
        )?;
        let key = refresh_signed_envelope(package.path())?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, _, _override) = new_fixture()?;
        let mut catalog: DirectMlFfiContractCatalogDto =
            serde_json::from_slice(&fs::read(package.path().join("ffi-contracts-v1.json"))?)?;
        catalog.target = "aarch64-pc-windows-msvc".to_owned();
        fs::write(
            package.path().join("ffi-contracts-v1.json"),
            canonical_json(&catalog)?,
        )?;
        let key = refresh_signed_envelope(package.path())?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, _, _override) = new_fixture()?;
        let mut catalog: DirectMlFfiContractCatalogDto =
            serde_json::from_slice(&fs::read(package.path().join("ffi-contracts-v1.json"))?)?;
        catalog.libraries.pop();
        fs::write(
            package.path().join("ffi-contracts-v1.json"),
            canonical_json(&catalog)?,
        )?;
        let key = refresh_signed_envelope(package.path())?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, _, _override) = new_fixture()?;
        let mut catalog: DirectMlFfiContractCatalogDto =
            serde_json::from_slice(&fs::read(package.path().join("ffi-contracts-v1.json"))?)?;
        let directml = catalog
            .libraries
            .iter_mut()
            .find(|row| row.identity == DIRECTML_LIBRARY_ID)
            .ok_or("DirectML fixture row is missing")?;
        directml.required_symbols.clear();
        fs::write(
            package.path().join("ffi-contracts-v1.json"),
            canonical_json(&catalog)?,
        )?;
        let key = refresh_signed_envelope(package.path())?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, _, _override) = new_fixture()?;
        let catalog = fs::read_to_string(package.path().join("ffi-contracts-v1.json"))?;
        let duplicate = catalog.replacen("{\n", "{\n  \"abi_floor\": \"1.13.1\",\n", 1);
        fs::write(package.path().join("ffi-contracts-v1.json"), duplicate)?;
        let key = refresh_signed_envelope(package.path())?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        let (package, _, _override) = new_fixture()?;
        let catalog = fs::read_to_string(package.path().join("ffi-contracts-v1.json"))?;
        let unknown_field = catalog.replacen("{\n", "{\n  \"additional_authority\": false,\n", 1);
        fs::write(package.path().join("ffi-contracts-v1.json"), unknown_field)?;
        let key = refresh_signed_envelope(package.path())?;
        assert_rejected(package.path(), &key, &CancellationToken::default());

        Ok(())
    }
}
