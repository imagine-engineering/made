use crate::{
    CertifiedNativeFfi, NativeFfiContract, NativeFfiRegistry, RocmPackageVerificationKey,
    TrustError,
    native_ffi_elf::{NativeElfDynamicContract, inspect_elf64_dynamic_contract},
    trust::{
        CapturedNativeLibraryImage, NativeLibraryImageError, RetainedNativeLibraryImage,
        capture_native_library_image_with_check,
    },
};
use serde::Deserialize;
#[cfg(test)]
use sha2::{Digest, Sha256};
use sim_backend_rocm::{
    DiscoveryRoot, PackageSignatureContract, PlatformPackageVerifier, RocmDependencyCandidate,
    RocmDependencyEdge, RocmLibrarySet, RocmLoadError, RocmRuntime, discover_library_set,
    verify_signed_package_root,
};
use sim_types::{BackendUnavailable, CancellationError, CancellationToken, DeviceKind};
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

#[cfg(test)]
thread_local! {
    static INJECT_CANCELLATION_AFTER_CHECKS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
use std::fs::OpenOptions;

const ROCM_ABI_VERSION: &str = "6.1.0";
const ROCM_UNSAFE_OWNER: &str = "sim_backend_rocm::loader";
#[cfg(test)]
const CANCELLATION_CHUNK_BYTES: usize = 64 * 1024;
const MAX_ROCM_FFI_CONTRACT_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_ROCM_FFI_CONTRACTS: usize = 256;
const MAX_ROCM_FFI_SYMBOLS_PER_CONTRACT: usize = 4096;
const PRIMARY_ROCM_LIBRARIES: &[(&str, &str)] = &[
    ("libMIOpen", "libMIOpen.so"),
    ("libamdhip64", "libamdhip64.so"),
    ("libhiprtc", "libhiprtc.so"),
    ("librocblas", "librocblas.so"),
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum RocmFfiContractRole {
    Primary,
    RecursiveDependency,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RocmFfiContractCatalogDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    libraries: Vec<RocmFfiContractDto>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RocmFfiContractDto {
    identity: String,
    role: RocmFfiContractRole,
    filename: String,
    soname: String,
    sha256: String,
    abi: String,
    required_symbols: Vec<String>,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RocmFfiLibraryIdentity {
    role: &'static str,
    filename: String,
    soname: String,
}

impl RocmFfiLibraryIdentity {
    pub fn role(&self) -> &'static str {
        self.role
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn soname(&self) -> &str {
        &self.soname
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRocmFfiContracts {
    discovery_root: DiscoveryRoot,
    registry: NativeFfiRegistry,
    identities: BTreeMap<String, RocmFfiLibraryIdentity>,
}

impl VerifiedRocmFfiContracts {
    pub fn discovery_root(&self) -> &DiscoveryRoot {
        &self.discovery_root
    }

    pub fn registry(&self) -> &NativeFfiRegistry {
        &self.registry
    }

    pub fn identities(&self) -> &BTreeMap<String, RocmFfiLibraryIdentity> {
        &self.identities
    }
}

#[derive(Debug, Error)]
pub enum RocmPackageContractError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Loader(#[from] RocmLoadError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("signed ROCm FFI contract catalog is invalid: {0}")]
    InvalidCatalog(String),
}

struct NativeRocmPackageVerifier<'a> {
    key: &'a RocmPackageVerificationKey,
    cancellation: &'a CancellationToken,
}

impl PlatformPackageVerifier for NativeRocmPackageVerifier<'_> {
    fn verify_rocm_package(
        &self,
        _package_root: &Path,
        contract: &PackageSignatureContract,
    ) -> Result<(), String> {
        check_rocm_cancellation(self.cancellation).map_err(|error| error.to_string())?;
        self.key
            .verify_package(
                contract.signer(),
                contract.package_coverage_bytes(),
                contract.signature_receipt_bytes(),
            )
            .map_err(|error| error.to_string())?;
        check_rocm_cancellation(self.cancellation).map_err(|error| error.to_string())
    }
}

pub fn verify_rocm_package_contracts(
    package_root: &Path,
    verification_key: &RocmPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedRocmFfiContracts, RocmPackageContractError> {
    check_rocm_cancellation(cancellation)?;
    let verifier = NativeRocmPackageVerifier {
        key: verification_key,
        cancellation,
    };
    let verified = verify_signed_package_root(package_root, &verifier)?;
    check_rocm_cancellation(cancellation)?;
    let (registry, identities) =
        parse_rocm_ffi_contract_catalog(verified.signature_contract().ffi_contracts_bytes())?;
    check_rocm_cancellation(cancellation)?;
    Ok(VerifiedRocmFfiContracts {
        discovery_root: verified.discovery_root().clone(),
        registry,
        identities,
    })
}

fn parse_rocm_ffi_contract_catalog(
    bytes: &[u8],
) -> Result<(NativeFfiRegistry, BTreeMap<String, RocmFfiLibraryIdentity>), RocmPackageContractError>
{
    if bytes.is_empty() || bytes.len() > MAX_ROCM_FFI_CONTRACT_CATALOG_BYTES {
        return Err(RocmPackageContractError::InvalidCatalog(
            "catalog byte length is outside the bounded range".to_owned(),
        ));
    }
    let catalog: RocmFfiContractCatalogDto = serde_json::from_slice(bytes)
        .map_err(|error| RocmPackageContractError::InvalidCatalog(error.to_string()))?;
    if catalog.schema_version != 1
        || catalog.backend != "rocm"
        || catalog.abi_floor != ROCM_ABI_VERSION
        || catalog.libraries.len() < PRIMARY_ROCM_LIBRARIES.len()
        || catalog.libraries.len() > MAX_ROCM_FFI_CONTRACTS
    {
        return Err(RocmPackageContractError::InvalidCatalog(
            "catalog envelope is unsupported or incomplete".to_owned(),
        ));
    }

    let mut previous_identity: Option<String> = None;
    let mut primary_identities = BTreeSet::new();
    let mut filenames = BTreeSet::new();
    let mut sonames = BTreeSet::new();
    let mut identities = BTreeMap::new();
    let mut contracts = Vec::new();
    contracts
        .try_reserve_exact(catalog.libraries.len())
        .map_err(|_| {
            RocmPackageContractError::InvalidCatalog("catalog allocation failed".to_owned())
        })?;
    for row in catalog.libraries {
        if previous_identity
            .as_deref()
            .is_some_and(|previous| previous >= row.identity.as_str())
        {
            return Err(RocmPackageContractError::InvalidCatalog(
                "library identities must be sorted and unique".to_owned(),
            ));
        }
        previous_identity = Some(row.identity.clone());
        if !valid_rocm_library_filename(&row.filename)
            || !valid_rocm_library_filename(&row.soname)
            || row.filename != row.soname
            || !filenames.insert(row.filename.clone())
            || !sonames.insert(row.soname.clone())
            || !valid_lower_hex_digest(&row.sha256)
            || row.abi != ROCM_ABI_VERSION
            || row.unsafe_owner != ROCM_UNSAFE_OWNER
            || row.required_symbols.is_empty()
            || row.required_symbols.len() > MAX_ROCM_FFI_SYMBOLS_PER_CONTRACT
            || row
                .required_symbols
                .windows(2)
                .any(|pair| matches!(pair, [first, second] if first >= second))
        {
            return Err(RocmPackageContractError::InvalidCatalog(format!(
                "library contract {} has invalid identity, ABI, digest, owner, or symbols",
                row.identity
            )));
        }
        let role = match row.role {
            RocmFfiContractRole::Primary => {
                let expected_filename = PRIMARY_ROCM_LIBRARIES
                    .iter()
                    .find_map(|(identity, filename)| {
                        (*identity == row.identity).then_some(*filename)
                    })
                    .ok_or_else(|| {
                        RocmPackageContractError::InvalidCatalog(format!(
                            "unknown primary ROCm identity {}",
                            row.identity
                        ))
                    })?;
                if row.filename != expected_filename {
                    return Err(RocmPackageContractError::InvalidCatalog(format!(
                        "primary ROCm identity {} has the wrong filename",
                        row.identity
                    )));
                }
                primary_identities.insert(row.identity.clone());
                "primary"
            }
            RocmFfiContractRole::RecursiveDependency => {
                if row.identity != format!("rocm-dependency:{}", row.soname) {
                    return Err(RocmPackageContractError::InvalidCatalog(format!(
                        "recursive dependency {} does not match its SONAME",
                        row.identity
                    )));
                }
                "recursive_dependency"
            }
        };
        let contract = NativeFfiContract::new(
            row.identity.clone(),
            row.sha256,
            row.abi,
            row.required_symbols,
            row.unsafe_owner,
        )?;
        identities.insert(
            row.identity,
            RocmFfiLibraryIdentity {
                role,
                filename: row.filename,
                soname: row.soname,
            },
        );
        contracts.push(contract);
    }
    let expected_primaries = PRIMARY_ROCM_LIBRARIES
        .iter()
        .map(|(identity, _)| (*identity).to_owned())
        .collect::<BTreeSet<_>>();
    if primary_identities != expected_primaries {
        return Err(RocmPackageContractError::InvalidCatalog(
            "catalog does not contain the exact four primary ROCm contracts".to_owned(),
        ));
    }
    Ok((NativeFfiRegistry::new(contracts)?, identities))
}

fn valid_rocm_library_filename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value != "."
        && value != ".."
        && !value
            .chars()
            .any(|character| matches!(character, '/' | '\\' | '\0'))
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'+' | b'-'))
}

fn valid_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
struct CancellableChunks<'a> {
    chunks: std::slice::Chunks<'a, u8>,
    cancellation: &'a CancellationToken,
}

#[cfg(test)]
impl<'a> CancellableChunks<'a> {
    fn new(bytes: &'a [u8], chunk_size: usize, cancellation: &'a CancellationToken) -> Self {
        Self {
            chunks: bytes.chunks(chunk_size.max(1)),
            cancellation,
        }
    }
}

#[cfg(test)]
impl<'a> Iterator for CancellableChunks<'a> {
    type Item = Result<&'a [u8], CancellationError>;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk = self.chunks.next()?;
        Some(check_rocm_cancellation(self.cancellation).map(|()| chunk))
    }
}

fn check_rocm_cancellation(cancellation: &CancellationToken) -> Result<(), CancellationError> {
    #[cfg(test)]
    INJECT_CANCELLATION_AFTER_CHECKS.with(|remaining| {
        if let Some(checks) = remaining.get() {
            if checks == 0 {
                remaining.set(None);
                cancellation.cancel();
            } else {
                remaining.set(Some(checks - 1));
            }
        }
    });
    cancellation.check()
}

#[cfg(test)]
fn inject_cancellation_after_successful_checks(checks: usize) {
    INJECT_CANCELLATION_AFTER_CHECKS.with(|remaining| remaining.set(Some(checks)));
}

#[derive(Debug, Error)]
pub enum RocmCertificationError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error("ROCm certification is supported only on x86_64 Linux")]
    UnsupportedTarget,
    #[error("ROCm library {library_id} has invalid identity: {reason}")]
    InvalidIdentity { library_id: String, reason: String },
    #[error("ROCm libraries do not belong to one discovery root")]
    MixedRoots,
    #[error("ROCm library {library_id} could not be read safely: {reason}")]
    File { library_id: String, reason: String },
    #[error("ROCm library {library_id} is not a valid certified ELF object: {reason}")]
    InvalidElf { library_id: String, reason: String },
    #[error("ROCm library {library_id} is missing required symbol {symbol}")]
    MissingSymbol { library_id: String, symbol: String },
    #[error("ROCm library {library_id} declares unaccounted ELF dependency {dependency}")]
    UnaccountedDependency {
        library_id: String,
        dependency: String,
    },
    #[error("ROCm dependency graph is invalid: {reason}")]
    DependencyGraph { reason: String },
    #[error("ROCm library {library_id} was rejected by the canonical native-FFI registry")]
    Registry {
        library_id: String,
        #[source]
        error: TrustError,
    },
    #[error("ROCm certified loader failed: {0}")]
    Loader(#[from] RocmLoadError),
}

pub struct CertifiedRocmRuntime {
    runtime: RocmRuntime,
    certificates: Vec<CertifiedNativeFfi>,
}

impl CertifiedRocmRuntime {
    pub fn runtime(&self) -> &RocmRuntime {
        &self.runtime
    }

    pub fn certificates(&self) -> &[CertifiedNativeFfi] {
        &self.certificates
    }

    pub fn into_runtime(self) -> RocmRuntime {
        self.runtime
    }
}

pub fn initialize_certified_rocm_runtime(
    settings: &crate::NativeRocmPackageSettings,
    cancellation: &CancellationToken,
) -> Result<CertifiedRocmRuntime, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Rocm, reason);
    let verified = verify_rocm_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed package or contract verification failed"))?;
    let library_set = discover_library_set(std::slice::from_ref(verified.discovery_root()))
        .map_err(|_| unavailable("exact library discovery failed"))?;
    load_certified_rocm_runtime(verified.registry(), &library_set, cancellation)
        .map_err(|_| unavailable("recursive native-FFI certification or retained loading failed"))
}

#[derive(Clone)]
struct CandidateInput {
    library_id: String,
    path: PathBuf,
    abi_version: String,
    required_symbols: BTreeSet<String>,
    unsafe_owner: String,
}

struct CertifiedLoadInputs {
    remapped_set: RocmLibrarySet,
    certificates: Vec<CertifiedNativeFfi>,
    snapshots: Vec<RetainedNativeLibraryImage>,
}

struct RocmCertificationRetention {
    _certificates: Vec<CertifiedNativeFfi>,
    _snapshots: Vec<RetainedNativeLibraryImage>,
}

struct InspectedCandidate {
    input: CandidateInput,
    image: CapturedNativeLibraryImage,
    dynamic: NativeElfDynamicContract,
}

const TRUSTED_SYSTEM_ELF_DEPENDENCIES: &[&str] = &[
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libstdc++.so.6",
];

trait UnsafeRocmLoader {
    type Runtime;

    unsafe fn load(
        library_set: &RocmLibrarySet,
        certification_retention: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self::Runtime, RocmLoadError>;
}

struct BackendLoader;

impl UnsafeRocmLoader for BackendLoader {
    type Runtime = RocmRuntime;

    unsafe fn load(
        library_set: &RocmLibrarySet,
        certification_retention: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self::Runtime, RocmLoadError> {
        // SAFETY: `prepare_certified_load` supplies only sealed in-memory snapshots for which it
        // retains one registry-issued certificate per exact digest, ABI, symbol set, and owner.
        unsafe { RocmRuntime::load_certified(library_set, certification_retention) }
    }
}

pub fn load_certified_rocm_runtime(
    registry: &NativeFfiRegistry,
    library_set: &RocmLibrarySet,
    cancellation: &CancellationToken,
) -> Result<CertifiedRocmRuntime, RocmCertificationError> {
    check_rocm_cancellation(cancellation)?;
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err(RocmCertificationError::UnsupportedTarget);
    }
    let inputs = candidate_inputs(library_set);
    let prepared = prepare_certified_load(registry, library_set, &inputs, cancellation)?;
    check_rocm_cancellation(cancellation)?;
    // SAFETY: no loader call is reachable until all candidates have been copied into sealed
    // snapshots and certified by NativeFfiRegistry. The snapshots and certificates are moved
    // into the returned value and therefore outlive every backend function pointer.
    let certificates = prepared.certificates.clone();
    let retention: Arc<dyn Any + Send + Sync> = Arc::new(RocmCertificationRetention {
        _certificates: prepared.certificates,
        _snapshots: prepared.snapshots,
    });
    let runtime = unsafe { BackendLoader::load(&prepared.remapped_set, retention) }?;
    Ok(CertifiedRocmRuntime {
        runtime,
        certificates,
    })
}

fn candidate_inputs(library_set: &RocmLibrarySet) -> Vec<CandidateInput> {
    library_set
        .libraries()
        .iter()
        .map(|candidate| CandidateInput {
            library_id: candidate.library_id().to_owned(),
            path: candidate.path().to_owned(),
            abi_version: candidate.abi_version().to_owned(),
            required_symbols: candidate
                .required_symbols()
                .iter()
                .map(|symbol| (*symbol).to_owned())
                .collect(),
            unsafe_owner: candidate.unsafe_owner().to_owned(),
        })
        .collect()
}

fn prepare_certified_load(
    registry: &NativeFfiRegistry,
    library_set: &RocmLibrarySet,
    candidates: &[CandidateInput],
    cancellation: &CancellationToken,
) -> Result<CertifiedLoadInputs, RocmCertificationError> {
    check_rocm_cancellation(cancellation)?;
    if !library_set.dependencies().is_empty() || !library_set.dependency_edges().is_empty() {
        return Err(RocmCertificationError::DependencyGraph {
            reason: "the dependency closure must be derived by the canonical certifier".to_owned(),
        });
    }
    let primary_paths = candidates
        .iter()
        .map(checked_candidate_path)
        .collect::<Result<Vec<_>, _>>()?;
    let discovery_root = validate_one_root(candidates, &primary_paths)?;

    let mut primary_ids = BTreeSet::new();
    let mut soname_to_id = BTreeMap::new();
    let mut pending = candidates.iter().cloned().collect::<Vec<_>>();
    pending.sort_by(|left, right| left.library_id.cmp(&right.library_id));
    let mut pending = VecDeque::from(pending);
    for candidate in candidates {
        primary_ids.insert(candidate.library_id.clone());
        let soname = candidate_soname(candidate)?;
        if soname_to_id
            .insert(soname.to_owned(), candidate.library_id.clone())
            .is_some()
        {
            return Err(RocmCertificationError::DependencyGraph {
                reason: format!("multiple primary objects claim SONAME {soname}"),
            });
        }
    }

    let mut inspected = BTreeMap::new();
    let mut dependency_edges = BTreeSet::new();
    while let Some(candidate) = pending.pop_front() {
        if inspected.contains_key(&candidate.library_id) {
            continue;
        }
        check_rocm_cancellation(cancellation)?;
        validate_candidate_identity(&candidate)?;
        let checked_path = checked_candidate_path(&candidate)?;
        validate_path_root(&candidate, &checked_path, &discovery_root)?;
        let image = capture_native_library_image_with_check(&checked_path, || {
            check_rocm_cancellation(cancellation)
        })
        .map_err(|error| map_native_library_image_error(&candidate.library_id, error))?;
        check_rocm_cancellation(cancellation)?;
        let dynamic =
            inspect_elf64_dynamic_contract(image.bytes(), 62, cancellation).map_err(|error| {
                RocmCertificationError::InvalidElf {
                    library_id: candidate.library_id.clone(),
                    reason: error.to_string(),
                }
            })?;
        let expected_soname = candidate_soname(&candidate)?;
        if dynamic.soname() != Some(expected_soname) {
            return Err(RocmCertificationError::InvalidIdentity {
                library_id: candidate.library_id.clone(),
                reason: format!(
                    "ELF SONAME must be the exact discovered filename {expected_soname}"
                ),
            });
        }
        for dependency_soname in dynamic
            .needed()
            .iter()
            .filter(|dependency| !TRUSTED_SYSTEM_ELF_DEPENDENCIES.contains(&dependency.as_str()))
        {
            check_rocm_cancellation(cancellation)?;
            let dependency_id = if let Some(library_id) = soname_to_id.get(dependency_soname) {
                library_id.clone()
            } else {
                let library_id = dependency_library_id(dependency_soname)?;
                let path = resolve_dependency_path(
                    &candidate.library_id,
                    dependency_soname,
                    &discovery_root,
                )?;
                let required_symbols = registry
                    .required_symbols_for(&library_id, ROCM_ABI_VERSION, ROCM_UNSAFE_OWNER)
                    .map_err(|error| RocmCertificationError::Registry {
                        library_id: library_id.clone(),
                        error,
                    })?;
                soname_to_id.insert(dependency_soname.clone(), library_id.clone());
                pending.push_back(CandidateInput {
                    library_id: library_id.clone(),
                    path,
                    abi_version: ROCM_ABI_VERSION.to_owned(),
                    required_symbols,
                    unsafe_owner: ROCM_UNSAFE_OWNER.to_owned(),
                });
                library_id
            };
            dependency_edges.insert(RocmDependencyEdge::new(
                candidate.library_id.clone(),
                dependency_id,
            ));
        }

        inspected.insert(
            candidate.library_id.clone(),
            InspectedCandidate {
                input: candidate,
                image,
                dynamic,
            },
        );
    }

    let dependencies = inspected
        .values()
        .filter(|candidate| !primary_ids.contains(&candidate.input.library_id))
        .map(|candidate| {
            Ok(RocmDependencyCandidate::new(
                candidate.input.library_id.clone(),
                candidate_soname(&candidate.input)?,
                candidate.input.path.clone(),
                candidate.input.abi_version.clone(),
                candidate.input.required_symbols.iter().cloned().collect(),
                candidate.input.unsafe_owner.clone(),
            ))
        })
        .collect::<Result<Vec<_>, RocmCertificationError>>()?;
    let expanded_set = library_set
        .clone()
        .with_certified_dependency_closure(dependencies, dependency_edges.into_iter().collect())
        .map_err(map_dependency_graph_error)?;
    let load_order = expanded_set
        .load_order()
        .map_err(map_dependency_graph_error)?;

    let mut remapped_paths = BTreeMap::new();
    let mut certificates = Vec::with_capacity(inspected.len());
    let mut snapshots = Vec::with_capacity(inspected.len());
    for library_id in load_order {
        check_rocm_cancellation(cancellation)?;
        let candidate = inspected.remove(&library_id).ok_or_else(|| {
            RocmCertificationError::DependencyGraph {
                reason: format!("load order references uninspected object {library_id}"),
            }
        })?;
        if let Some(symbol) = candidate
            .input
            .required_symbols
            .iter()
            .find(|symbol| !candidate.dynamic.symbols().contains(*symbol))
        {
            return Err(RocmCertificationError::MissingSymbol {
                library_id: candidate.input.library_id.clone(),
                symbol: symbol.clone(),
            });
        }
        let certificate = registry
            .authorize(
                &candidate.input.library_id,
                candidate.image.digest_sha256(),
                &candidate.input.abi_version,
                candidate.dynamic.symbols(),
            )
            .map_err(|error| RocmCertificationError::Registry {
                library_id: candidate.input.library_id.clone(),
                error,
            })?;
        if certificate.abi_version() != candidate.input.abi_version
            || certificate.unsafe_owner() != candidate.input.unsafe_owner
            || certificate.library_id() != candidate.input.library_id
        {
            return Err(RocmCertificationError::InvalidIdentity {
                library_id: candidate.input.library_id,
                reason:
                    "registry certificate does not preserve ABI, library, and unsafe-owner identity"
                        .to_owned(),
            });
        }
        let candidate_library_id = candidate.input.library_id.clone();
        let retained = candidate
            .image
            .seal_with_check(&format!("rocm-{candidate_library_id}"), || {
                check_rocm_cancellation(cancellation)
            })
            .map_err(|error| map_native_library_image_error(&candidate_library_id, error))?;
        certificates.push(certificate);
        remapped_paths.insert(candidate_library_id, retained.loader_path().to_path_buf());
        snapshots.push(retained);
    }
    let remapped_set = expanded_set
        .remap_to_retained_descriptors(remapped_paths)
        .map_err(RocmCertificationError::Loader)?;
    check_rocm_cancellation(cancellation)?;
    Ok(CertifiedLoadInputs {
        remapped_set,
        certificates,
        snapshots,
    })
}

fn map_dependency_graph_error(error: RocmLoadError) -> RocmCertificationError {
    match error {
        RocmLoadError::DependencyGraph { reason } => {
            RocmCertificationError::DependencyGraph { reason }
        }
        error => RocmCertificationError::Loader(error),
    }
}

fn candidate_soname(candidate: &CandidateInput) -> Result<&str, RocmCertificationError> {
    candidate
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| RocmCertificationError::InvalidIdentity {
            library_id: candidate.library_id.clone(),
            reason: "library filename is not UTF-8".to_owned(),
        })
}

fn dependency_library_id(soname: &str) -> Result<String, RocmCertificationError> {
    if soname.is_empty()
        || soname.len() > 255
        || soname.contains('/')
        || soname.contains('\0')
        || soname == "."
        || soname == ".."
    {
        return Err(RocmCertificationError::DependencyGraph {
            reason: format!("invalid DT_NEEDED filename {soname:?}"),
        });
    }
    Ok(format!("rocm-dependency:{soname}"))
}

fn resolve_dependency_path(
    consumer_id: &str,
    soname: &str,
    discovery_root: &Path,
) -> Result<PathBuf, RocmCertificationError> {
    dependency_library_id(soname)?;
    let mut resolved = None;
    for directory_name in ["lib", "lib64"] {
        let candidate = discovery_root.join(directory_name).join(soname);
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(RocmCertificationError::File {
                    library_id: format!("rocm-dependency:{soname}"),
                    reason: "dependency final component must be a non-symlink regular file"
                        .to_owned(),
                });
            }
            Ok(metadata) if metadata.file_type().is_file() => {
                if resolved.replace(candidate).is_some() {
                    return Err(RocmCertificationError::DependencyGraph {
                        reason: format!(
                            "dependency SONAME {soname} is ambiguous between lib and lib64"
                        ),
                    });
                }
            }
            Ok(_) => {
                return Err(RocmCertificationError::File {
                    library_id: format!("rocm-dependency:{soname}"),
                    reason: "dependency final component must be a non-symlink regular file"
                        .to_owned(),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(RocmCertificationError::File {
                    library_id: format!("rocm-dependency:{soname}"),
                    reason: error.to_string(),
                });
            }
        }
    }
    resolved.ok_or_else(|| RocmCertificationError::UnaccountedDependency {
        library_id: consumer_id.to_owned(),
        dependency: soname.to_owned(),
    })
}

fn checked_candidate_path(candidate: &CandidateInput) -> Result<PathBuf, RocmCertificationError> {
    let file_name =
        candidate
            .path
            .file_name()
            .ok_or_else(|| RocmCertificationError::InvalidIdentity {
                library_id: candidate.library_id.clone(),
                reason: "library path has no final filename".to_owned(),
            })?;
    let parent =
        candidate
            .path
            .parent()
            .ok_or_else(|| RocmCertificationError::InvalidIdentity {
                library_id: candidate.library_id.clone(),
                reason: "library path has no parent directory".to_owned(),
            })?;
    let canonical_parent = parent
        .canonicalize()
        .map_err(|error| RocmCertificationError::File {
            library_id: candidate.library_id.clone(),
            reason: error.to_string(),
        })?;
    let checked = canonical_parent.join(file_name);
    let metadata =
        fs::symlink_metadata(&checked).map_err(|error| RocmCertificationError::File {
            library_id: candidate.library_id.clone(),
            reason: error.to_string(),
        })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(RocmCertificationError::File {
            library_id: candidate.library_id.clone(),
            reason: "candidate final component must be a non-symlink regular file".to_owned(),
        });
    }
    Ok(checked)
}

fn validate_candidate_identity(candidate: &CandidateInput) -> Result<(), RocmCertificationError> {
    if candidate.abi_version != ROCM_ABI_VERSION || candidate.unsafe_owner != ROCM_UNSAFE_OWNER {
        return Err(RocmCertificationError::InvalidIdentity {
            library_id: candidate.library_id.clone(),
            reason: format!("expected ABI {ROCM_ABI_VERSION} and unsafe owner {ROCM_UNSAFE_OWNER}"),
        });
    }
    Ok(())
}

fn validate_one_root(
    candidates: &[CandidateInput],
    canonical_paths: &[PathBuf],
) -> Result<PathBuf, RocmCertificationError> {
    let mut expected_root: Option<PathBuf> = None;
    for (candidate, path) in candidates.iter().zip(canonical_paths) {
        let directory = path
            .parent()
            .ok_or_else(|| RocmCertificationError::InvalidIdentity {
                library_id: candidate.library_id.clone(),
                reason: "library has no parent directory".to_owned(),
            })?;
        let directory_name = directory.file_name().and_then(|name| name.to_str());
        if !matches!(directory_name, Some("lib" | "lib64")) {
            return Err(RocmCertificationError::InvalidIdentity {
                library_id: candidate.library_id.clone(),
                reason: "library is not below the fixed lib or lib64 discovery directory"
                    .to_owned(),
            });
        }
        let root = directory
            .parent()
            .ok_or_else(|| RocmCertificationError::InvalidIdentity {
                library_id: candidate.library_id.clone(),
                reason: "library discovery directory has no root".to_owned(),
            })?;
        if expected_root
            .as_deref()
            .is_some_and(|expected| expected != root)
        {
            return Err(RocmCertificationError::MixedRoots);
        }
        expected_root.get_or_insert_with(|| root.to_owned());
    }
    expected_root.ok_or_else(|| RocmCertificationError::DependencyGraph {
        reason: "ROCm certification requires at least one primary library".to_owned(),
    })
}

fn validate_path_root(
    candidate: &CandidateInput,
    path: &Path,
    expected_root: &Path,
) -> Result<(), RocmCertificationError> {
    let directory = path
        .parent()
        .ok_or_else(|| RocmCertificationError::InvalidIdentity {
            library_id: candidate.library_id.clone(),
            reason: "library has no parent directory".to_owned(),
        })?;
    let directory_name = directory.file_name().and_then(|name| name.to_str());
    let root = directory
        .parent()
        .ok_or_else(|| RocmCertificationError::InvalidIdentity {
            library_id: candidate.library_id.clone(),
            reason: "library discovery directory has no root".to_owned(),
        })?;
    if !matches!(directory_name, Some("lib" | "lib64")) || root != expected_root {
        return Err(RocmCertificationError::MixedRoots);
    }
    Ok(())
}

fn map_native_library_image_error(
    library_id: &str,
    error: NativeLibraryImageError,
) -> RocmCertificationError {
    match error {
        NativeLibraryImageError::Cancelled => RocmCertificationError::Cancelled(CancellationError),
        NativeLibraryImageError::UnsupportedPlatform => RocmCertificationError::File {
            library_id: library_id.to_owned(),
            reason: "retained native-library descriptors require Linux".to_owned(),
        },
        NativeLibraryImageError::Invalid(reason) => RocmCertificationError::File {
            library_id: library_id.to_owned(),
            reason,
        },
    }
}

#[cfg(test)]
fn hex_sha256(bytes: &[u8], cancellation: &CancellationToken) -> Result<String, CancellationError> {
    let mut hasher = Sha256::new();
    for chunk in CancellableChunks::new(bytes, CANCELLATION_CHUNK_BYTES, cancellation) {
        hasher.update(chunk?);
    }
    check_rocm_cancellation(cancellation)?;
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
type ElfDynamicContract = NativeElfDynamicContract;

#[cfg(test)]
fn elf64_dynamic_contract(
    bytes: &[u8],
    cancellation: &CancellationToken,
) -> Result<ElfDynamicContract, String> {
    inspect_elf64_dynamic_contract(bytes, 62, cancellation).map_err(|error| error.to_string())
}

#[cfg(test)]
fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| "ELF range is out of bounds".to_owned())?;
    let value = &bytes[offset..end];
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeFfiContract, trust::rocm_package_signing_payload};
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use sim_backend_rocm::discover_library_set;
    use std::cell::Cell;

    #[cfg(target_os = "linux")]
    struct CountingLoader;

    thread_local! {
        static LOADER_CALLS: Cell<usize> = const { Cell::new(0) };
        static REGISTRY_PIPELINE_ENTRIES: Cell<usize> = const { Cell::new(0) };
        static RETAINED_OBJECT_COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    #[cfg(target_os = "linux")]
    impl UnsafeRocmLoader for CountingLoader {
        type Runtime = ();

        unsafe fn load(
            _library_set: &RocmLibrarySet,
            certification_retention: Arc<dyn Any + Send + Sync>,
        ) -> Result<Self::Runtime, RocmLoadError> {
            LOADER_CALLS.with(|calls| calls.set(calls.get() + 1));
            let retention = certification_retention
                .as_ref()
                .downcast_ref::<RocmCertificationRetention>()
                .ok_or_else(|| RocmLoadError::CertifiedPathRemap {
                    reason: "loader did not receive canonical certification retention".to_owned(),
                })?;
            RETAINED_OBJECT_COUNTS.with(|counts| {
                counts.set((retention._certificates.len(), retention._snapshots.len()));
            });
            Ok(())
        }
    }

    fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn elf_fixture(symbols: &BTreeSet<String>, soname: &str) -> Vec<u8> {
        elf_fixture_with_dynamic(symbols, &[], None, soname)
    }

    fn elf_fixture_with_dynamic(
        symbols: &BTreeSet<String>,
        needed: &[&str],
        runpath: Option<&str>,
        soname: &str,
    ) -> Vec<u8> {
        let mut strings = vec![0_u8];
        let mut names = Vec::new();
        for symbol in symbols {
            names.push(u32::try_from(strings.len()).unwrap_or_default());
            strings.extend_from_slice(symbol.as_bytes());
            strings.push(0);
        }
        let mut needed_offsets = Vec::new();
        for dependency in needed {
            needed_offsets.push(u64::try_from(strings.len()).unwrap_or_default());
            strings.extend_from_slice(dependency.as_bytes());
            strings.push(0);
        }
        let runpath_offset = runpath.map(|path| {
            let offset = u64::try_from(strings.len()).unwrap_or_default();
            strings.extend_from_slice(path.as_bytes());
            strings.push(0);
            offset
        });
        let soname_offset = u64::try_from(strings.len()).unwrap_or_default();
        strings.extend_from_slice(soname.as_bytes());
        strings.push(0);
        let program_offset = 64;
        let program_entry_size = 56;
        let program_count = 2;
        let string_offset = 192;
        let symbol_offset = (string_offset + strings.len() + 7) & !7;
        let symbol_size = (symbols.len() + 1) * 24;
        let dynamic_offset = (symbol_offset + symbol_size + 7) & !7;
        let dynamic_entries = 6 + needed_offsets.len() + usize::from(runpath_offset.is_some());
        let dynamic_size = dynamic_entries * 16;
        let section_offset = (dynamic_offset + dynamic_size + 7) & !7;
        let mut bytes = vec![0_u8; section_offset + 4 * 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        write_u16(&mut bytes, 16, 3);
        write_u16(&mut bytes, 18, 62);
        write_u64(
            &mut bytes,
            32,
            u64::try_from(program_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            40,
            u64::try_from(section_offset).unwrap_or_default(),
        );
        write_u16(&mut bytes, 54, program_entry_size);
        write_u16(&mut bytes, 56, program_count);
        write_u16(&mut bytes, 58, 64);
        write_u16(&mut bytes, 60, 4);
        let load_program = program_offset;
        let file_length = u64::try_from(bytes.len()).unwrap_or_default();
        write_u32(&mut bytes, load_program, 1);
        write_u32(&mut bytes, load_program + 4, 4);
        write_u64(&mut bytes, load_program + 32, file_length);
        write_u64(&mut bytes, load_program + 40, file_length);
        write_u64(&mut bytes, load_program + 48, 8);
        let dynamic_program = program_offset + usize::from(program_entry_size);
        write_u32(&mut bytes, dynamic_program, 2);
        write_u32(&mut bytes, dynamic_program + 4, 4);
        write_u64(
            &mut bytes,
            dynamic_program + 8,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_program + 16,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_program + 32,
            u64::try_from(dynamic_size).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_program + 40,
            u64::try_from(dynamic_size).unwrap_or_default(),
        );
        write_u64(&mut bytes, dynamic_program + 48, 8);
        bytes[string_offset..string_offset + strings.len()].copy_from_slice(&strings);
        for (index, name_offset) in names.into_iter().enumerate() {
            let entry = symbol_offset + (index + 1) * 24;
            write_u32(&mut bytes, entry, name_offset);
            bytes[entry + 4] = 0x12;
            write_u16(&mut bytes, entry + 6, 1);
        }
        let strings_header = section_offset + 64;
        write_u32(&mut bytes, strings_header + 4, 3);
        write_u64(
            &mut bytes,
            strings_header + 16,
            u64::try_from(string_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            strings_header + 24,
            u64::try_from(string_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            strings_header + 32,
            u64::try_from(strings.len()).unwrap_or_default(),
        );
        let symbols_header = section_offset + 128;
        write_u32(&mut bytes, symbols_header + 4, 11);
        write_u64(&mut bytes, symbols_header + 8, 2);
        write_u64(
            &mut bytes,
            symbols_header + 16,
            u64::try_from(symbol_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            symbols_header + 24,
            u64::try_from(symbol_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            symbols_header + 32,
            u64::try_from(symbol_size).unwrap_or_default(),
        );
        write_u32(&mut bytes, symbols_header + 40, 1);
        write_u64(&mut bytes, symbols_header + 56, 24);
        let mut dynamic_index = 0;
        for (tag, value) in [
            (5, u64::try_from(string_offset).unwrap_or_default()),
            (10, u64::try_from(strings.len()).unwrap_or_default()),
            (6, u64::try_from(symbol_offset).unwrap_or_default()),
            (11, 24),
            (14, soname_offset),
        ] {
            let entry = dynamic_offset + dynamic_index * 16;
            write_u64(&mut bytes, entry, tag);
            write_u64(&mut bytes, entry + 8, value);
            dynamic_index += 1;
        }
        for offset in needed_offsets {
            let entry = dynamic_offset + dynamic_index * 16;
            write_u64(&mut bytes, entry, 1);
            write_u64(&mut bytes, entry + 8, offset);
            dynamic_index += 1;
        }
        if let Some(offset) = runpath_offset {
            let entry = dynamic_offset + dynamic_index * 16;
            write_u64(&mut bytes, entry, 29);
            write_u64(&mut bytes, entry + 8, offset);
            dynamic_index += 1;
        }
        let null_entry = dynamic_offset + dynamic_index * 16;
        write_u64(&mut bytes, null_entry, 0);
        let dynamic_header = section_offset + 192;
        write_u32(&mut bytes, dynamic_header + 4, 6);
        write_u64(
            &mut bytes,
            dynamic_header + 16,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_header + 24,
            u64::try_from(dynamic_offset).unwrap_or_default(),
        );
        write_u64(
            &mut bytes,
            dynamic_header + 32,
            u64::try_from(dynamic_size).unwrap_or_default(),
        );
        write_u32(&mut bytes, dynamic_header + 40, 1);
        write_u64(&mut bytes, dynamic_header + 56, 16);
        bytes
    }

    fn fixture_sha256(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn fixture_ffi_contract_catalog() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "backend": "rocm",
            "abi_floor": "6.1.0",
            "libraries": [
                {
                    "identity": "libMIOpen",
                    "role": "primary",
                    "filename": "libMIOpen.so",
                    "soname": "libMIOpen.so",
                    "sha256": "11".repeat(32),
                    "abi": "6.1.0",
                    "required_symbols": ["miopenCreate"],
                    "unsafe_owner": ROCM_UNSAFE_OWNER,
                },
                {
                    "identity": "libamdhip64",
                    "role": "primary",
                    "filename": "libamdhip64.so",
                    "soname": "libamdhip64.so",
                    "sha256": "22".repeat(32),
                    "abi": "6.1.0",
                    "required_symbols": ["hipInit"],
                    "unsafe_owner": ROCM_UNSAFE_OWNER,
                },
                {
                    "identity": "libhiprtc",
                    "role": "primary",
                    "filename": "libhiprtc.so",
                    "soname": "libhiprtc.so",
                    "sha256": "33".repeat(32),
                    "abi": "6.1.0",
                    "required_symbols": ["hiprtcCreateProgram"],
                    "unsafe_owner": ROCM_UNSAFE_OWNER,
                },
                {
                    "identity": "librocblas",
                    "role": "primary",
                    "filename": "librocblas.so",
                    "soname": "librocblas.so",
                    "sha256": "44".repeat(32),
                    "abi": "6.1.0",
                    "required_symbols": ["rocblas_create_handle"],
                    "unsafe_owner": ROCM_UNSAFE_OWNER,
                },
                {
                    "identity": "rocm-dependency:libhsa-runtime64.so.1",
                    "role": "recursive_dependency",
                    "filename": "libhsa-runtime64.so.1",
                    "soname": "libhsa-runtime64.so.1",
                    "sha256": "55".repeat(32),
                    "abi": "6.1.0",
                    "required_symbols": ["hsa_init"],
                    "unsafe_owner": ROCM_UNSAFE_OWNER,
                },
            ],
        })
    }

    #[test]
    fn signed_contract_catalog_maps_strict_dtos_to_the_canonical_registry()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = fixture_ffi_contract_catalog();
        let bytes = serde_json::to_vec(&catalog)?;
        let (registry, identities) = parse_rocm_ffi_contract_catalog(&bytes)?;
        assert_eq!(identities.len(), 5);
        assert_eq!(
            identities
                .get("rocm-dependency:libhsa-runtime64.so.1")
                .map(RocmFfiLibraryIdentity::role),
            Some("recursive_dependency")
        );
        let available_symbols = BTreeSet::from(["hipInit".to_owned()]);
        assert_eq!(
            registry
                .authorize(
                    "libamdhip64",
                    &"22".repeat(32),
                    ROCM_ABI_VERSION,
                    &available_symbols,
                )?
                .unsafe_owner(),
            ROCM_UNSAFE_OWNER
        );

        let mut invalid_cases = Vec::new();
        let mut unknown_field = catalog.clone();
        unknown_field
            .as_object_mut()
            .ok_or("fixture catalog is not an object")?
            .insert("plugin_key_id".to_owned(), serde_json::json!("forbidden"));
        invalid_cases.push(unknown_field);
        let mut missing_primary = catalog.clone();
        missing_primary
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture libraries are not an array")?
            .remove(0);
        invalid_cases.push(missing_primary);
        let mut unsorted = catalog.clone();
        unsorted
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture libraries are not an array")?
            .swap(0, 1);
        invalid_cases.push(unsorted);
        let mut wrong_owner = catalog.clone();
        wrong_owner
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|rows| rows.first_mut())
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("fixture first row is not an object")?
            .insert("unsafe_owner".to_owned(), serde_json::json!("plugin_host"));
        invalid_cases.push(wrong_owner);
        let mut duplicate_symbol = catalog;
        duplicate_symbol
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|rows| rows.first_mut())
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("fixture first row is not an object")?
            .insert(
                "required_symbols".to_owned(),
                serde_json::json!(["miopenCreate", "miopenCreate"]),
            );
        invalid_cases.push(duplicate_symbol);
        for (field, value) in [
            ("filename", serde_json::json!("libwrong.so")),
            ("sha256", serde_json::json!("AA".repeat(32))),
            ("abi", serde_json::json!("6.0.0")),
            ("required_symbols", serde_json::json!([])),
        ] {
            let mut invalid_row = fixture_ffi_contract_catalog();
            invalid_row
                .get_mut("libraries")
                .and_then(serde_json::Value::as_array_mut)
                .and_then(|rows| rows.first_mut())
                .and_then(serde_json::Value::as_object_mut)
                .ok_or("fixture first row is not an object")?
                .insert(field.to_owned(), value);
            invalid_cases.push(invalid_row);
        }
        let mut wrong_dependency_identity = fixture_ffi_contract_catalog();
        wrong_dependency_identity
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|rows| rows.last_mut())
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("fixture dependency row is not an object")?
            .insert(
                "identity".to_owned(),
                serde_json::json!("rocm-dependency:libwrong.so.1"),
            );
        invalid_cases.push(wrong_dependency_identity);
        let mut duplicate_row = fixture_ffi_contract_catalog();
        let first_row = duplicate_row
            .get("libraries")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| rows.first())
            .cloned()
            .ok_or("fixture first row is absent")?;
        duplicate_row
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture libraries are not an array")?
            .insert(1, first_row);
        invalid_cases.push(duplicate_row);
        for invalid in invalid_cases {
            assert!(matches!(
                parse_rocm_ffi_contract_catalog(&serde_json::to_vec(&invalid)?),
                Err(RocmPackageContractError::InvalidCatalog(_))
            ));
        }
        assert!(matches!(
            parse_rocm_ffi_contract_catalog(&vec![b' '; MAX_ROCM_FFI_CONTRACT_CATALOG_BYTES + 1]),
            Err(RocmPackageContractError::InvalidCatalog(_))
        ));
        Ok(())
    }

    fn write_fixture_package(
        package: &Path,
        sdk: &Path,
        contract_catalog: serde_json::Value,
    ) -> Result<RocmPackageVerificationKey, Box<dyn std::error::Error>> {
        write_fixture_package_bytes(package, sdk, &serde_json::to_vec(&contract_catalog)?)
    }

    fn write_fixture_package_bytes(
        package: &Path,
        sdk: &Path,
        ffi_contracts: &[u8],
    ) -> Result<RocmPackageVerificationKey, Box<dyn std::error::Error>> {
        const ABI_MANIFEST: &str = include_str!("../../sim_backend_rocm/abi/symbols-v1.json");
        const PACKAGE_POLICY: &str =
            include_str!("../../../nix/sim-backends/rocm/package-policy.json");

        fs::create_dir_all(package.join("abi"))?;
        fs::write(package.join("abi/symbols-v1.json"), ABI_MANIFEST)?;
        fs::write(package.join("LICENSES"), "fixture notices\n")?;
        fs::write(package.join("package-policy.json"), PACKAGE_POLICY)?;
        let ffi_contracts_sha256 = fixture_sha256(ffi_contracts);
        fs::write(package.join("ffi-contracts-v1.json"), ffi_contracts)?;
        fs::write(
            package.join("adapter-manifest.json"),
            format!(
                r#"{{
                "schema_version":2,
                "backend":"rocm",
                "abi_floor":"6.1.0",
                "abi_manifest_sha256":"3259ee5fc5657e3d06597d8b1782a04024540287135b9d6ec7edb10935e83d8c",
                "ffi_contracts_sha256":"{ffi_contracts_sha256}",
                "redistributes_amd_runtime":false,
                "signer":"zed.release",
                "signature_algorithm":"ed25519",
                "signature_domain":"zed-sim-rocm-package-v1",
                "signature_coverage":"package-coverage-v1",
                "runtime_root":{}
            }}"#,
                serde_json::to_string(sdk)?
            ),
        )?;
        let mut coverage = BTreeMap::new();
        for relative in [
            "LICENSES",
            "abi/symbols-v1.json",
            "adapter-manifest.json",
            "ffi-contracts-v1.json",
            "package-policy.json",
        ] {
            let bytes = fs::read(package.join(relative))?;
            coverage.insert(relative, (fixture_sha256(&bytes), bytes.len()));
        }
        let coverage = coverage
            .into_iter()
            .map(|(path, (digest, size))| format!("{digest} {size}  {path}\n"))
            .collect::<String>();
        fs::write(package.join("package-coverage.sha256"), &coverage)?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(&[7_u8; 32])
            .map_err(|error| std::io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let signing_payload = rocm_package_signing_payload("zed.release", coverage.as_bytes())?;
        let signature = key_pair.sign(&signing_payload);
        let receipt = format!(
            "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{}\"}}\n",
            signature
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        fs::write(package.join("adapter-manifest.sig"), receipt)?;
        Ok(RocmPackageVerificationKey::new(
            "zed.release",
            key_pair.public_key().as_ref(),
        )?)
    }

    fn signed_fixture_with_catalog_bytes(
        ffi_contracts: &[u8],
    ) -> Result<(tempfile::TempDir, RocmPackageVerificationKey), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let sdk = root.path().join("sdk");
        let package = root.path().join("package");
        let libraries = sdk.join("lib");
        fs::create_dir(&sdk)?;
        fs::create_dir(&package)?;
        fs::create_dir(&libraries)?;
        for filename in [
            "libamdhip64.so",
            "libhiprtc.so",
            "librocblas.so",
            "libMIOpen.so",
        ] {
            fs::write(
                libraries.join(filename),
                elf_fixture(&BTreeSet::new(), filename),
            )?;
        }
        let verification_key = write_fixture_package_bytes(&package, &sdk, ffi_contracts)?;
        Ok((root, verification_key))
    }

    fn fixture_signing_payload(
        domain: &[u8],
        signer: &str,
        coverage: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let signer_length = u64::try_from(signer.len())?;
        let coverage_length = u64::try_from(coverage.len())?;
        let mut payload = Vec::new();
        payload.extend_from_slice(domain);
        payload.extend_from_slice(&signer_length.to_be_bytes());
        payload.extend_from_slice(signer.as_bytes());
        payload.extend_from_slice(&coverage_length.to_be_bytes());
        payload.extend_from_slice(coverage);
        Ok(payload)
    }

    fn rewrite_fixture_receipt(
        package: &Path,
        domain: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let coverage = fs::read(package.join("package-coverage.sha256"))?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(&[7_u8; 32])
            .map_err(|error| std::io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let signature = key_pair.sign(&fixture_signing_payload(domain, "zed.release", &coverage)?);
        let receipt = format!(
            "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{}\"}}\n",
            signature
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        fs::write(package.join("adapter-manifest.sig"), receipt)?;
        Ok(())
    }

    fn run_signed_fixture_pipeline(
        package: &Path,
        verification_key: &RocmPackageVerificationKey,
    ) -> Result<(), String> {
        let cancellation = CancellationToken::default();
        let verified = verify_rocm_package_contracts(package, verification_key, &cancellation)
            .map_err(|error| error.to_string())?;
        REGISTRY_PIPELINE_ENTRIES.with(|entries| entries.set(entries.get() + 1));
        let set = discover_library_set(&[verified.discovery_root().clone()])
            .map_err(|error| error.to_string())?;
        let inputs = candidate_inputs(&set);
        let prepared = prepare_certified_load(verified.registry(), &set, &inputs, &cancellation)
            .map_err(|error| error.to_string())?;
        #[cfg(target_os = "linux")]
        {
            let retention: Arc<dyn Any + Send + Sync> = Arc::new(RocmCertificationRetention {
                _certificates: prepared.certificates,
                _snapshots: prepared.snapshots,
            });
            // SAFETY: the counting loader performs no FFI and only records that every preceding
            // trust and certification stage succeeded.
            unsafe { CountingLoader::load(&prepared.remapped_set, retention) }
                .map_err(|error| error.to_string())?;
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _prepared = prepared;
            LOADER_CALLS.with(|calls| calls.set(calls.get() + 1));
        }
        Ok(())
    }

    fn assert_signed_fixture_rejected(
        package: &Path,
        verification_key: &RocmPackageVerificationKey,
        expected_registry_pipeline_entries: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        LOADER_CALLS.with(|calls| calls.set(0));
        REGISTRY_PIPELINE_ENTRIES.with(|entries| entries.set(0));
        let result = run_signed_fixture_pipeline(package, verification_key);
        if result.is_ok() {
            return Err(std::io::Error::other("invalid signed fixture reached the loader").into());
        }
        assert_eq!(LOADER_CALLS.with(|calls| calls.get()), 0);
        assert_eq!(
            REGISTRY_PIPELINE_ENTRIES.with(|entries| entries.get()),
            expected_registry_pipeline_entries
        );
        Ok(())
    }

    fn mutate_first_catalog_row(
        field: &str,
        value: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut catalog = fixture_ffi_contract_catalog();
        catalog
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|rows| rows.first_mut())
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("fixture first row is not an object")?
            .insert(field.to_owned(), value);
        Ok(catalog)
    }

    #[test]
    fn signed_package_rejection_matrix_stops_before_registry_or_loader_entry()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut invalid_catalogs = Vec::new();

        let mut unknown_field = fixture_ffi_contract_catalog();
        unknown_field
            .as_object_mut()
            .ok_or("fixture catalog is not an object")?
            .insert("plugin_key_id".to_owned(), serde_json::json!("forbidden"));
        invalid_catalogs.push(("unknown-field", serde_json::to_vec(&unknown_field)?));

        let mut missing_primary = fixture_ffi_contract_catalog();
        missing_primary
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture libraries are not an array")?
            .remove(0);
        invalid_catalogs.push(("missing-primary", serde_json::to_vec(&missing_primary)?));

        let mut unsorted = fixture_ffi_contract_catalog();
        unsorted
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture libraries are not an array")?
            .swap(0, 1);
        invalid_catalogs.push(("unsorted", serde_json::to_vec(&unsorted)?));

        let mut duplicate = fixture_ffi_contract_catalog();
        let first_row = duplicate
            .get("libraries")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| rows.first())
            .cloned()
            .ok_or("fixture first row is absent")?;
        duplicate
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture libraries are not an array")?
            .insert(1, first_row);
        invalid_catalogs.push(("duplicate", serde_json::to_vec(&duplicate)?));

        for (name, catalog) in [
            (
                "digest",
                mutate_first_catalog_row("sha256", serde_json::json!("AA".repeat(32)))?,
            ),
            (
                "filename",
                mutate_first_catalog_row("filename", serde_json::json!("libwrong.so"))?,
            ),
            (
                "soname",
                mutate_first_catalog_row("soname", serde_json::json!("libwrong.so"))?,
            ),
            (
                "abi",
                mutate_first_catalog_row("abi", serde_json::json!("6.0.0"))?,
            ),
            (
                "symbols",
                mutate_first_catalog_row("required_symbols", serde_json::json!([]))?,
            ),
            (
                "owner",
                mutate_first_catalog_row("unsafe_owner", serde_json::json!("plugin_host"))?,
            ),
        ] {
            invalid_catalogs.push((name, serde_json::to_vec(&catalog)?));
        }
        invalid_catalogs.push(("malformed", b"{".to_vec()));
        invalid_catalogs.push((
            "oversized",
            vec![b' '; MAX_ROCM_FFI_CONTRACT_CATALOG_BYTES + 1],
        ));

        for (name, catalog) in invalid_catalogs {
            let (root, verification_key) = signed_fixture_with_catalog_bytes(&catalog)?;
            assert_signed_fixture_rejected(&root.path().join("package"), &verification_key, 0)
                .map_err(|error| std::io::Error::other(format!("{name}: {error}")))?;
        }

        let valid_catalog = serde_json::to_vec(&fixture_ffi_contract_catalog())?;

        let (unsigned_root, unsigned_key) = signed_fixture_with_catalog_bytes(&valid_catalog)?;
        fs::remove_file(unsigned_root.path().join("package/adapter-manifest.sig"))?;
        assert_signed_fixture_rejected(&unsigned_root.path().join("package"), &unsigned_key, 0)?;

        let (missing_root, missing_key) = signed_fixture_with_catalog_bytes(&valid_catalog)?;
        fs::remove_file(missing_root.path().join("package/ffi-contracts-v1.json"))?;
        assert_signed_fixture_rejected(&missing_root.path().join("package"), &missing_key, 0)?;

        let (tampered_root, tampered_key) = signed_fixture_with_catalog_bytes(&valid_catalog)?;
        fs::write(
            tampered_root.path().join("package/ffi-contracts-v1.json"),
            b"{}",
        )?;
        assert_signed_fixture_rejected(&tampered_root.path().join("package"), &tampered_key, 0)?;

        let (wrong_domain_root, wrong_domain_key) =
            signed_fixture_with_catalog_bytes(&valid_catalog)?;
        rewrite_fixture_receipt(
            &wrong_domain_root.path().join("package"),
            b"zed-sim-rocm-package-v0\0",
        )?;
        assert_signed_fixture_rejected(
            &wrong_domain_root.path().join("package"),
            &wrong_domain_key,
            0,
        )?;

        let (unknown_key_root, _known_key) = signed_fixture_with_catalog_bytes(&valid_catalog)?;
        let unknown_key_pair = Ed25519KeyPair::from_seed_unchecked(&[8_u8; 32])
            .map_err(|error| std::io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let unknown_key =
            RocmPackageVerificationKey::new("zed.release", unknown_key_pair.public_key().as_ref())?;
        assert_signed_fixture_rejected(&unknown_key_root.path().join("package"), &unknown_key, 0)?;

        let (uncovered_root, uncovered_key) = signed_fixture_with_catalog_bytes(&valid_catalog)?;
        let uncovered_package = uncovered_root.path().join("package");
        let coverage = fs::read_to_string(uncovered_package.join("package-coverage.sha256"))?;
        let uncovered_coverage = coverage
            .lines()
            .filter(|line| !line.ends_with("ffi-contracts-v1.json"))
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        fs::write(
            uncovered_package.join("package-coverage.sha256"),
            uncovered_coverage,
        )?;
        rewrite_fixture_receipt(&uncovered_package, b"zed-sim-rocm-package-v1\0")?;
        assert_signed_fixture_rejected(&uncovered_package, &uncovered_key, 0)?;

        Ok(())
    }

    fn fixture_set() -> Result<(tempfile::TempDir, RocmLibrarySet), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let sdk = root.path().join("sdk");
        let package = root.path().join("package");
        let libraries = sdk.join("lib");
        fs::create_dir(&sdk)?;
        fs::create_dir(&package)?;
        fs::create_dir(&libraries)?;
        for filename in [
            "libamdhip64.so",
            "libhiprtc.so",
            "librocblas.so",
            "libMIOpen.so",
        ] {
            fs::write(
                libraries.join(filename),
                elf_fixture(&BTreeSet::new(), filename),
            )?;
        }
        let verification_key =
            write_fixture_package(&package, &sdk, fixture_ffi_contract_catalog())?;
        let verified = verify_rocm_package_contracts(
            &package,
            &verification_key,
            &CancellationToken::default(),
        )?;
        let set = discover_library_set(&[verified.discovery_root().clone()])?;
        for candidate in set.libraries() {
            fs::write(
                candidate.path(),
                elf_fixture(
                    &candidate
                        .required_symbols()
                        .iter()
                        .map(|symbol| (*symbol).to_owned())
                        .collect(),
                    candidate
                        .path()
                        .file_name()
                        .and_then(|name| name.to_str())
                        .ok_or("candidate filename is not UTF-8")?,
                ),
            )?;
        }
        Ok((root, set))
    }

    fn exact_fixture_contract_catalog(
        library_set: &RocmLibrarySet,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut libraries = library_set
            .libraries()
            .iter()
            .map(|candidate| {
                let bytes = fs::read(candidate.path())?;
                let filename = candidate
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or("fixture library filename is not UTF-8")?;
                let mut required_symbols = candidate
                    .required_symbols()
                    .iter()
                    .map(|symbol| (*symbol).to_owned())
                    .collect::<Vec<_>>();
                required_symbols.sort();
                Ok::<_, Box<dyn std::error::Error>>(serde_json::json!({
                    "identity": candidate.library_id(),
                    "role": "primary",
                    "filename": filename,
                    "soname": filename,
                    "sha256": fixture_sha256(&bytes),
                    "abi": candidate.abi_version(),
                    "required_symbols": required_symbols,
                    "unsafe_owner": candidate.unsafe_owner(),
                }))
            })
            .collect::<Result<Vec<_>, _>>()?;
        libraries.sort_by(|left, right| {
            left.get("identity")
                .and_then(serde_json::Value::as_str)
                .cmp(&right.get("identity").and_then(serde_json::Value::as_str))
        });
        Ok(serde_json::json!({
            "schema_version": 1,
            "backend": "rocm",
            "abi_floor": ROCM_ABI_VERSION,
            "libraries": libraries,
        }))
    }

    #[test]
    fn signed_registry_certification_rejects_digest_symbol_and_missing_transitive_contracts()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, set) = fixture_set()?;
        let package = root.path().join("package");
        let libraries = root.path().join("sdk/lib");

        let exact_catalog = exact_fixture_contract_catalog(&set)?;
        let mut wrong_digest = exact_catalog.clone();
        wrong_digest
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|rows| rows.first_mut())
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("fixture first row is not an object")?
            .insert("sha256".to_owned(), serde_json::json!("00".repeat(32)));
        let wrong_digest_key =
            write_fixture_package(&package, root.path().join("sdk").as_path(), wrong_digest)?;
        assert_signed_fixture_rejected(&package, &wrong_digest_key, 1)?;

        let mut missing_symbol = exact_catalog;
        let required_symbols = missing_symbol
            .get_mut("libraries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|rows| rows.first_mut())
            .and_then(|row| row.get_mut("required_symbols"))
            .and_then(serde_json::Value::as_array_mut)
            .ok_or("fixture symbols are not an array")?;
        required_symbols.push(serde_json::json!("zz_missing_symbol"));
        let missing_symbol_key =
            write_fixture_package(&package, root.path().join("sdk").as_path(), missing_symbol)?;
        assert_signed_fixture_rejected(&package, &missing_symbol_key, 1)?;

        let dependency_soname = "libfixturedep.so.1";
        let primary = set
            .libraries()
            .iter()
            .find(|candidate| candidate.library_id() == "libamdhip64")
            .ok_or("fixture HIP primary is absent")?;
        let primary_symbols = primary
            .required_symbols()
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect();
        fs::write(
            primary.path(),
            elf_fixture_with_dynamic(
                &primary_symbols,
                &[dependency_soname],
                None,
                "libamdhip64.so",
            ),
        )?;
        fs::write(
            libraries.join(dependency_soname),
            elf_fixture(
                &BTreeSet::from(["fixture_dependency_symbol".to_owned()]),
                dependency_soname,
            ),
        )?;
        let missing_transitive_catalog = exact_fixture_contract_catalog(&set)?;
        let missing_transitive_key = write_fixture_package(
            &package,
            root.path().join("sdk").as_path(),
            missing_transitive_catalog,
        )?;
        assert_signed_fixture_rejected(&package, &missing_transitive_key, 1)?;
        Ok(())
    }

    #[test]
    fn complete_signed_fixture_drives_exact_registry_certification_closure()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let sdk = root.path().join("sdk");
        let package = root.path().join("package");
        let libraries = sdk.join("lib");
        fs::create_dir(&sdk)?;
        fs::create_dir(&package)?;
        fs::create_dir(&libraries)?;
        for filename in [
            "libamdhip64.so",
            "libhiprtc.so",
            "librocblas.so",
            "libMIOpen.so",
        ] {
            fs::write(
                libraries.join(filename),
                elf_fixture(&BTreeSet::new(), filename),
            )?;
        }
        let bootstrap_key = write_fixture_package(&package, &sdk, fixture_ffi_contract_catalog())?;
        let bootstrap_verified =
            verify_rocm_package_contracts(&package, &bootstrap_key, &CancellationToken::default())?;
        let bootstrap_set = discover_library_set(&[bootstrap_verified.discovery_root().clone()])?;
        for candidate in bootstrap_set.libraries() {
            let symbols = candidate
                .required_symbols()
                .iter()
                .map(|symbol| (*symbol).to_owned())
                .collect();
            let filename = candidate
                .path()
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("fixture library filename is not UTF-8")?;
            fs::write(candidate.path(), elf_fixture(&symbols, filename))?;
        }
        let catalog = exact_fixture_contract_catalog(&bootstrap_set)?;
        let verification_key = write_fixture_package(&package, &sdk, catalog)?;
        let cancellation = CancellationToken::default();
        let verified = verify_rocm_package_contracts(&package, &verification_key, &cancellation)?;
        let certified_set = discover_library_set(&[verified.discovery_root().clone()])?;
        let inputs = candidate_inputs(&certified_set);
        let prepared =
            prepare_certified_load(verified.registry(), &certified_set, &inputs, &cancellation);
        assert_eq!(verified.identities().len(), 4);
        #[cfg(target_os = "linux")]
        {
            let prepared = prepared?;
            assert_eq!(prepared.certificates.len(), 4);
            assert_eq!(prepared.snapshots.len(), 4);
            assert_eq!(prepared.remapped_set.libraries().len(), 4);
            LOADER_CALLS.with(|calls| calls.set(0));
            let retention: Arc<dyn Any + Send + Sync> = Arc::new(RocmCertificationRetention {
                _certificates: prepared.certificates,
                _snapshots: prepared.snapshots,
            });
            // SAFETY: the counting loader performs no FFI and records only that the exact signed
            // registry and certification closure were retained through loader entry.
            unsafe { CountingLoader::load(&prepared.remapped_set, retention) }?;
            assert_eq!(LOADER_CALLS.with(|calls| calls.get()), 1);
        }
        #[cfg(not(target_os = "linux"))]
        assert!(matches!(
            prepared,
            Err(RocmCertificationError::Loader(
                RocmLoadError::CertifiedPathRemap { .. }
            ))
        ));
        for input in inputs {
            let bytes = fs::read(&input.path)?;
            let digest = fixture_sha256(&bytes);
            let certificate = verified.registry().authorize(
                &input.library_id,
                &digest,
                &input.abi_version,
                &input.required_symbols,
            )?;
            assert_eq!(certificate.library_id(), input.library_id);
            assert_eq!(certificate.unsafe_owner(), ROCM_UNSAFE_OWNER);
        }
        Ok(())
    }

    fn registry_for(
        inputs: &[CandidateInput],
        digest_override: Option<(&str, String)>,
    ) -> Result<NativeFfiRegistry, TrustError> {
        let contracts = inputs
            .iter()
            .map(|input| {
                let bytes = fs::read(&input.path).map_err(|_| TrustError::UncertifiedFfi)?;
                let digest = digest_override
                    .as_ref()
                    .filter(|(library_id, _)| *library_id == input.library_id)
                    .map(|(_, digest)| digest.clone())
                    .map(Ok)
                    .unwrap_or_else(|| {
                        hex_sha256(&bytes, &CancellationToken::default())
                            .map_err(|_| TrustError::UncertifiedFfi)
                    })?;
                NativeFfiContract::new(
                    input.library_id.clone(),
                    digest,
                    input.abi_version.clone(),
                    input.required_symbols.iter().cloned(),
                    input.unsafe_owner.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        NativeFfiRegistry::new(contracts)
    }

    fn registry_with_dependencies(
        inputs: &[CandidateInput],
        dependencies: &[(&str, &Path, &BTreeSet<String>)],
    ) -> Result<NativeFfiRegistry, TrustError> {
        let mut contracts = Vec::new();
        for input in inputs {
            let bytes = fs::read(&input.path).map_err(|_| TrustError::UncertifiedFfi)?;
            contracts.push(NativeFfiContract::new(
                input.library_id.clone(),
                hex_sha256(&bytes, &CancellationToken::default())
                    .map_err(|_| TrustError::UncertifiedFfi)?,
                input.abi_version.clone(),
                input.required_symbols.iter().cloned(),
                input.unsafe_owner.clone(),
            )?);
        }
        for (soname, path, symbols) in dependencies {
            let bytes = fs::read(path).map_err(|_| TrustError::UncertifiedFfi)?;
            contracts.push(NativeFfiContract::new(
                dependency_library_id(soname).map_err(|_| TrustError::InvalidFfiContract)?,
                hex_sha256(&bytes, &CancellationToken::default())
                    .map_err(|_| TrustError::UncertifiedFfi)?,
                ROCM_ABI_VERSION,
                symbols.iter().cloned(),
                ROCM_UNSAFE_OWNER,
            )?);
        }
        NativeFfiRegistry::new(contracts)
    }

    #[cfg(target_os = "linux")]
    fn prepare_with_counting_loader(
        registry: &NativeFfiRegistry,
        set: &RocmLibrarySet,
        inputs: &[CandidateInput],
    ) -> Result<(usize, usize, usize), RocmCertificationError> {
        LOADER_CALLS.with(|calls| calls.set(0));
        RETAINED_OBJECT_COUNTS.with(|counts| counts.set((0, 0)));
        let prepared =
            prepare_certified_load(registry, set, inputs, &CancellationToken::default())?;
        let retention: Arc<dyn Any + Send + Sync> = Arc::new(RocmCertificationRetention {
            _certificates: prepared.certificates,
            _snapshots: prepared.snapshots,
        });
        // SAFETY: the test loader does not inspect or call through the backend paths.
        unsafe { CountingLoader::load(&prepared.remapped_set, retention) }?;
        let loader_calls = LOADER_CALLS.with(|calls| calls.get());
        let (certificates, snapshots) = RETAINED_OBJECT_COUNTS.with(|counts| counts.get());
        Ok((loader_calls, certificates, snapshots))
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn exact_files_symbols_and_registry_certificates_precede_the_loader()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let registry = registry_for(&inputs, None)?;
        assert_eq!(
            prepare_with_counting_loader(&registry, &set, &inputs)?,
            (1, inputs.len(), inputs.len())
        );
        Ok(())
    }

    #[test]
    fn tampered_digest_and_fabricated_metadata_never_reach_the_loader()
    -> Result<(), Box<dyn std::error::Error>> {
        let (root, set) = fixture_set()?;
        fs::write(
            root.path().join("adapter-manifest.json"),
            br#"{"schema_version":1,"backend":"rocm","signed":true}"#,
        )?;
        let inputs = candidate_inputs(&set);
        let registry = registry_for(&inputs, Some((&inputs[0].library_id, "00".repeat(32))))?;
        LOADER_CALLS.with(|calls| calls.set(0));
        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::Registry { .. })
        ));
        assert_eq!(LOADER_CALLS.with(|calls| calls.get()), 0);
        Ok(())
    }

    #[test]
    fn missing_symbol_never_reaches_the_loader() -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let missing = inputs[0]
            .required_symbols
            .iter()
            .next()
            .cloned()
            .ok_or("missing fixture symbol")?;
        let remaining = inputs[0]
            .required_symbols
            .iter()
            .filter(|symbol| **symbol != missing)
            .cloned()
            .collect();
        let filename = inputs[0]
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("candidate filename is not UTF-8")?;
        fs::write(&inputs[0].path, elf_fixture(&remaining, filename))?;
        let registry = registry_for(&inputs, None)?;
        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::MissingSymbol { .. })
        ));
        Ok(())
    }

    #[test]
    fn unaccounted_dependencies_and_embedded_search_paths_never_reach_the_loader()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &["libunaccounted-vendor.so"],
                None,
                inputs[0]
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or("candidate filename is not UTF-8")?,
            ),
        )?;
        let registry = registry_for(&inputs, None)?;
        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::UnaccountedDependency { dependency, .. })
                if dependency == "libunaccounted-vendor.so"
        ));
        assert!(
            elf64_dynamic_contract(
                &elf_fixture_with_dynamic(
                    &BTreeSet::new(),
                    &["libc.so.6"],
                    Some("/untrusted"),
                    "fixture.so",
                ),
                &CancellationToken::default(),
            )
            .is_err()
        );
        let system_contract = elf64_dynamic_contract(
            &elf_fixture_with_dynamic(&BTreeSet::new(), &["libc.so.6"], None, "fixture.so"),
            &CancellationToken::default(),
        )?;
        assert_eq!(
            system_contract.needed,
            BTreeSet::from(["libc.so.6".to_owned()])
        );
        Ok(())
    }

    #[test]
    fn discovered_dependency_without_a_preprovisioned_contract_is_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let dependency_soname = "libfixture-unprovisioned.so";
        let dependency_path = inputs[0]
            .path
            .parent()
            .ok_or("primary has no parent")?
            .join(dependency_soname);
        fs::write(
            &dependency_path,
            elf_fixture(
                &BTreeSet::from(["dependency_symbol".to_owned()]),
                dependency_soname,
            ),
        )?;
        let primary_soname = candidate_soname(&inputs[0])?;
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &[dependency_soname],
                None,
                primary_soname,
            ),
        )?;
        let registry = registry_for(&inputs, None)?;

        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::Registry { library_id, .. })
                if library_id == "rocm-dependency:libfixture-unprovisioned.so"
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn dependency_symlinks_are_rejected_even_when_the_target_is_inside_the_sdk()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let dependency_soname = "libfixture-link.so";
        let directory = inputs[0].path.parent().ok_or("primary has no parent")?;
        let target = directory.join("libfixture-link-real.so");
        fs::write(
            &target,
            elf_fixture(
                &BTreeSet::from(["dependency_symbol".to_owned()]),
                dependency_soname,
            ),
        )?;
        symlink(&target, directory.join(dependency_soname))?;
        let primary_soname = candidate_soname(&inputs[0])?;
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &[dependency_soname],
                None,
                primary_soname,
            ),
        )?;
        let registry = registry_for(&inputs, None)?;

        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::File { reason, .. })
                if reason.contains("dependency final component")
        ));
        Ok(())
    }

    #[test]
    fn dependency_search_never_escapes_the_approved_sdk_root()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let (other_root, _other_set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let dependency_soname = "libfixture-other-root.so";
        fs::write(
            other_root.path().join("sdk/lib").join(dependency_soname),
            elf_fixture(
                &BTreeSet::from(["dependency_symbol".to_owned()]),
                dependency_soname,
            ),
        )?;
        let primary_soname = candidate_soname(&inputs[0])?;
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &[dependency_soname],
                None,
                primary_soname,
            ),
        )?;
        let registry = registry_for(&inputs, None)?;

        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::UnaccountedDependency { dependency, .. })
                if dependency == dependency_soname
        ));
        Ok(())
    }

    #[test]
    fn cyclic_vendor_dependency_graph_is_rejected_with_a_typed_error()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let dependency_soname = "libfixture-cycle.so";
        let dependency_symbols = BTreeSet::from(["dependency_symbol".to_owned()]);
        let primary_soname = candidate_soname(&inputs[0])?.to_owned();
        let dependency_path = inputs[0]
            .path
            .parent()
            .ok_or("primary has no parent")?
            .join(dependency_soname);
        fs::write(
            &dependency_path,
            elf_fixture_with_dynamic(
                &dependency_symbols,
                &[&primary_soname],
                None,
                dependency_soname,
            ),
        )?;
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &[dependency_soname],
                None,
                &primary_soname,
            ),
        )?;
        let registry = registry_with_dependencies(
            &inputs,
            &[(dependency_soname, &dependency_path, &dependency_symbols)],
        )?;

        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::DependencyGraph { reason })
                if reason.contains("cycle")
        ));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn complete_dependency_closure_is_certified_and_snapshotted_leaves_first()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let first_soname = "libfixture-first.so";
        let leaf_soname = "libfixture-leaf.so";
        let dependency_symbols = BTreeSet::from(["dependency_symbol".to_owned()]);
        let directory = inputs[0].path.parent().ok_or("primary has no parent")?;
        let first_path = directory.join(first_soname);
        let leaf_path = directory.join(leaf_soname);
        fs::write(
            &first_path,
            elf_fixture_with_dynamic(&dependency_symbols, &[leaf_soname], None, first_soname),
        )?;
        fs::write(&leaf_path, elf_fixture(&dependency_symbols, leaf_soname))?;
        let primary_soname = candidate_soname(&inputs[0])?;
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &[first_soname],
                None,
                primary_soname,
            ),
        )?;
        let registry = registry_with_dependencies(
            &inputs,
            &[
                (first_soname, &first_path, &dependency_symbols),
                (leaf_soname, &leaf_path, &dependency_symbols),
            ],
        )?;

        let prepared =
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default())?;
        let order = prepared.remapped_set.load_order()?;
        let primary_id = &inputs[0].library_id;
        let first_id = dependency_library_id(first_soname)?;
        let leaf_id = dependency_library_id(leaf_soname)?;
        let position = |library_id: &str| {
            order
                .iter()
                .position(|candidate| candidate == library_id)
                .ok_or("library missing from load order")
        };
        assert!(position(&leaf_id)? < position(&first_id)?);
        assert!(position(&first_id)? < position(primary_id)?);
        assert_eq!(
            prepared
                .certificates
                .iter()
                .map(CertifiedNativeFfi::library_id)
                .collect::<Vec<_>>(),
            order.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(prepared.snapshots.len(), order.len());
        Ok(())
    }

    #[test]
    fn loader_consumed_dynamic_segment_cannot_be_hidden_by_section_headers()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut with_runpath = elf_fixture_with_dynamic(
            &BTreeSet::new(),
            &["libc.so.6"],
            Some("/ambient"),
            "fixture.so",
        );
        let section_offset = usize::try_from(read_u64(&with_runpath, 40)?)?;
        let dynamic_section = section_offset + 3 * 64;
        write_u64(&mut with_runpath, dynamic_section + 24, 0);
        write_u64(&mut with_runpath, dynamic_section + 32, 0);
        assert!(
            elf64_dynamic_contract(&with_runpath, &CancellationToken::default())
                .is_err_and(|reason| reason.contains("RPATH"))
        );

        let mut section_only =
            elf_fixture_with_dynamic(&BTreeSet::new(), &["libc.so.6"], None, "fixture.so");
        write_u32(&mut section_only, 64 + 56, 0);
        assert!(
            elf64_dynamic_contract(&section_only, &CancellationToken::default())
                .is_err_and(|reason| reason.contains("exactly one PT_DYNAMIC"))
        );
        Ok(())
    }

    #[test]
    fn exact_soname_and_system_only_dependencies_prevent_ambient_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        fs::write(
            &inputs[0].path,
            elf_fixture(&inputs[0].required_symbols, "forged-vendor.so"),
        )?;
        let registry = registry_for(&inputs, None)?;
        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::InvalidIdentity { reason, .. })
                if reason.contains("exact discovered filename")
        ));

        let expected_soname = inputs[0]
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("candidate filename is not UTF-8")?;
        let other_soname = inputs[1]
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("candidate filename is not UTF-8")?;
        fs::write(
            &inputs[0].path,
            elf_fixture_with_dynamic(
                &inputs[0].required_symbols,
                &[other_soname],
                None,
                expected_soname,
            ),
        )?;
        let registry = registry_for(&inputs, None)?;
        let result =
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default());
        #[cfg(target_os = "linux")]
        assert!(result.is_ok());
        #[cfg(not(target_os = "linux"))]
        assert!(matches!(
            result,
            Err(RocmCertificationError::Loader(
                RocmLoadError::CertifiedPathRemap { .. }
            ))
        ));
        Ok(())
    }

    #[test]
    fn cancellation_is_injected_during_read_hash_parse_and_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let input = candidate_inputs(&set).remove(0);
        let filename = input
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("candidate filename is not UTF-8")?;
        let mut large_elf = elf_fixture(&input.required_symbols, filename);
        large_elf.resize(CANCELLATION_CHUNK_BYTES * 3, 0);
        fs::write(&input.path, &large_elf)?;

        let cancellation = CancellationToken::default();
        inject_cancellation_after_successful_checks(1);
        assert!(matches!(
            capture_native_library_image_with_check(&input.path, || {
                check_rocm_cancellation(&cancellation)
            }),
            Err(NativeLibraryImageError::Cancelled)
        ));

        let cancellation = CancellationToken::default();
        inject_cancellation_after_successful_checks(1);
        assert!(matches!(
            hex_sha256(&large_elf, &cancellation),
            Err(CancellationError)
        ));

        let cancellation = CancellationToken::default();
        inject_cancellation_after_successful_checks(2);
        assert!(
            elf64_dynamic_contract(
                &elf_fixture_with_dynamic(&input.required_symbols, &["libc.so.6"], None, filename,),
                &cancellation,
            )
            .is_err_and(|reason| reason.contains("cancelled"))
        );

        let cancellation = CancellationToken::default();
        let captured = capture_native_library_image_with_check(&input.path, || Ok(()))?;
        inject_cancellation_after_successful_checks(1);
        assert!(matches!(
            captured.seal_with_check("fixture", || check_rocm_cancellation(&cancellation)),
            Err(NativeLibraryImageError::Cancelled)
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn final_library_symlink_is_rejected_before_opening() -> Result<(), Box<dyn std::error::Error>>
    {
        use std::os::unix::fs::symlink;

        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let replacement = inputs[0].path.with_extension("real");
        fs::rename(&inputs[0].path, &replacement)?;
        symlink(&replacement, &inputs[0].path)?;
        let registry = registry_for(&inputs, None)?;
        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::File { reason, .. })
                if reason.contains("non-symlink regular file")
        ));
        Ok(())
    }

    #[test]
    fn cancellation_precedes_candidate_validation_and_allocation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        fs::write(&inputs[0].path, b"not an ELF object")?;
        let registry = registry_for(&inputs, None)?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            prepare_certified_load(&registry, &set, &inputs, &cancellation),
            Err(RocmCertificationError::Cancelled(_))
        ));
        Ok(())
    }

    #[test]
    fn wrong_abi_or_unsafe_owner_never_reaches_the_loader() -> Result<(), Box<dyn std::error::Error>>
    {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let registry = registry_for(&inputs, None)?;
        for mutate in [0_u8, 1] {
            let mut changed = inputs.clone();
            if mutate == 0 {
                changed[0].abi_version = "6.0.0".to_owned();
            } else {
                changed[0].unsafe_owner = "sim_runtime".to_owned();
            }
            assert!(matches!(
                prepare_certified_load(&registry, &set, &changed, &CancellationToken::default(),),
                Err(RocmCertificationError::InvalidIdentity { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn mixed_discovery_roots_never_reach_the_loader() -> Result<(), Box<dyn std::error::Error>> {
        let (_root_a, set_a) = fixture_set()?;
        let (_root_b, set_b) = fixture_set()?;
        let mut inputs = candidate_inputs(&set_a);
        inputs[0].path = set_b.libraries()[0].path().to_owned();
        let registry = registry_for(&inputs, None)?;
        assert!(matches!(
            prepare_certified_load(&registry, &set_a, &inputs, &CancellationToken::default()),
            Err(RocmCertificationError::MixedRoots)
        ));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn certified_load_uses_an_immutable_snapshot_not_the_mutable_source_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let inputs = candidate_inputs(&set);
        let registry = registry_for(&inputs, None)?;
        let prepared =
            prepare_certified_load(&registry, &set, &inputs, &CancellationToken::default())?;
        fs::write(&inputs[0].path, b"replaced after certification")?;
        let retained = fs::read(prepared.snapshots[0].loader_path())?;
        assert_eq!(
            hex_sha256(&retained, &CancellationToken::default())?,
            prepared.certificates[0].digest_sha256()
        );
        // SAFETY: the counting loader performs no FFI and observes only that the certified stage
        // has completed.
        let retention: Arc<dyn Any + Send + Sync> = Arc::new(RocmCertificationRetention {
            _certificates: prepared.certificates,
            _snapshots: prepared.snapshots,
        });
        unsafe { CountingLoader::load(&prepared.remapped_set, retention) }?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sealed_snapshot_rejects_in_place_mutation() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("fixture.so");
        fs::write(&source, b"immutable")?;
        let snapshot = capture_native_library_image_with_check(&source, || Ok(()))?
            .seal("fixture", &CancellationToken::default())?;
        assert!(
            OpenOptions::new()
                .write(true)
                .open(snapshot.loader_path())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn oversized_library_is_rejected_before_allocation() -> Result<(), Box<dyn std::error::Error>> {
        let (_root, set) = fixture_set()?;
        let input = candidate_inputs(&set).remove(0);
        OpenOptions::new()
            .write(true)
            .open(&input.path)?
            .set_len(2 * 1024 * 1024 * 1024 + 1)?;
        let canonical = fs::canonicalize(&input.path)?;
        assert!(matches!(
            capture_native_library_image_with_check(&canonical, || Ok(())),
            Err(NativeLibraryImageError::Invalid(reason)) if reason.contains("nonempty regular file")
        ));
        Ok(())
    }
}
