use crate::abi;
use serde::Deserialize;
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    env, fmt, fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
use thiserror::Error;

const ABI_MANIFEST: &str = include_str!("../abi/symbols-v1.json");
const PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/rocm/package-policy.json");
const ABI_MANIFEST_SHA256: &str =
    "3259ee5fc5657e3d06597d8b1782a04024540287135b9d6ec7edb10935e83d8c";
const MAX_PACKAGE_ENTRIES: usize = 4_096;
const MAX_PACKAGE_PATH_BYTES: usize = 4_096;
const MAX_PACKAGE_METADATA_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscoverySource {
    SimRocmRoot,
    RocmPath,
    SignedPackage,
}

impl fmt::Display for DiscoverySource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SimRocmRoot => formatter.write_str("SIM_ROCM_ROOT"),
            Self::RocmPath => formatter.write_str("ROCM_PATH"),
            Self::SignedPackage => formatter.write_str("signed package root"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryRoot {
    source: DiscoverySource,
    path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RocmLibraryCandidate {
    library_id: &'static str,
    path: PathBuf,
    abi_version: &'static str,
    required_symbols: Vec<&'static str>,
    unsafe_owner: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RocmLibraryRole {
    Primary,
    Dependency,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RocmDependencyCandidate {
    library_id: String,
    soname: String,
    path: PathBuf,
    abi_version: String,
    required_symbols: Vec<String>,
    unsafe_owner: String,
}

impl RocmDependencyCandidate {
    pub fn new(
        library_id: impl Into<String>,
        soname: impl Into<String>,
        path: impl Into<PathBuf>,
        abi_version: impl Into<String>,
        required_symbols: Vec<String>,
        unsafe_owner: impl Into<String>,
    ) -> Self {
        Self {
            library_id: library_id.into(),
            soname: soname.into(),
            path: path.into(),
            abi_version: abi_version.into(),
            required_symbols,
            unsafe_owner: unsafe_owner.into(),
        }
    }

    pub fn library_id(&self) -> &str {
        &self.library_id
    }

    pub fn soname(&self) -> &str {
        &self.soname
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn abi_version(&self) -> &str {
        &self.abi_version
    }

    pub fn required_symbols(&self) -> &[String] {
        &self.required_symbols
    }

    pub fn unsafe_owner(&self) -> &str {
        &self.unsafe_owner
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RocmDependencyEdge {
    consumer: String,
    dependency: String,
}

impl RocmDependencyEdge {
    pub fn new(consumer: impl Into<String>, dependency: impl Into<String>) -> Self {
        Self {
            consumer: consumer.into(),
            dependency: dependency.into(),
        }
    }

    pub fn consumer(&self) -> &str {
        &self.consumer
    }

    pub fn dependency(&self) -> &str {
        &self.dependency
    }
}

impl RocmLibraryCandidate {
    pub const fn library_id(&self) -> &'static str {
        self.library_id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn abi_version(&self) -> &'static str {
        self.abi_version
    }

    pub fn required_symbols(&self) -> &[&'static str] {
        &self.required_symbols
    }

    pub const fn unsafe_owner(&self) -> &'static str {
        self.unsafe_owner
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RocmLibrarySet {
    libraries: Vec<RocmLibraryCandidate>,
    dependencies: Vec<RocmDependencyCandidate>,
    dependency_edges: Vec<RocmDependencyEdge>,
}

impl RocmLibrarySet {
    pub fn libraries(&self) -> &[RocmLibraryCandidate] {
        &self.libraries
    }

    pub fn dependencies(&self) -> &[RocmDependencyCandidate] {
        &self.dependencies
    }

    pub fn dependency_edges(&self) -> &[RocmDependencyEdge] {
        &self.dependency_edges
    }

    pub fn library_role(&self, library_id: &str) -> Option<RocmLibraryRole> {
        if self
            .libraries
            .iter()
            .any(|library| library.library_id == library_id)
        {
            Some(RocmLibraryRole::Primary)
        } else if self
            .dependencies
            .iter()
            .any(|library| library.library_id == library_id)
        {
            Some(RocmLibraryRole::Dependency)
        } else {
            None
        }
    }

    pub fn with_certified_dependency_closure(
        mut self,
        mut dependencies: Vec<RocmDependencyCandidate>,
        mut dependency_edges: Vec<RocmDependencyEdge>,
    ) -> Result<Self, RocmLoadError> {
        dependencies.sort_by(|left, right| left.library_id.cmp(&right.library_id));
        dependency_edges.sort();
        if dependencies
            .windows(2)
            .any(|pair| pair[0].library_id == pair[1].library_id)
            || dependency_edges.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(RocmLoadError::DependencyGraph {
                reason: "dependency IDs and graph edges must be unique".to_owned(),
            });
        }
        self.dependencies = dependencies;
        self.dependency_edges = dependency_edges;
        self.validate_dependency_graph()?;
        Ok(self)
    }

    pub fn load_order(&self) -> Result<Vec<String>, RocmLoadError> {
        self.validate_dependency_graph()?;
        self.topological_load_order()
    }

    fn topological_load_order(&self) -> Result<Vec<String>, RocmLoadError> {
        let mut remaining = self.all_ids();
        let mut ordered = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let ready = remaining
                .iter()
                .filter(|candidate| {
                    self.dependency_edges.iter().all(|edge| {
                        edge.consumer.as_str() != candidate.as_str()
                            || !remaining.contains(&edge.dependency)
                    })
                })
                .cloned()
                .collect::<Vec<_>>();
            if ready.is_empty() {
                return Err(RocmLoadError::DependencyGraph {
                    reason: "vendor dependency graph contains a cycle".to_owned(),
                });
            }
            for candidate in ready {
                remaining.remove(&candidate);
                ordered.push(candidate);
            }
        }
        Ok(ordered)
    }

    pub fn remap_to_retained_descriptors(
        &self,
        mut descriptor_paths: BTreeMap<String, PathBuf>,
    ) -> Result<Self, RocmLoadError> {
        if descriptor_paths.len() != self.libraries.len() + self.dependencies.len() {
            return Err(RocmLoadError::CertifiedPathRemap {
                reason: "descriptor map must contain exactly one path per candidate".to_owned(),
            });
        }
        let libraries = self
            .libraries
            .iter()
            .map(|library| {
                let path = descriptor_paths.remove(library.library_id).ok_or_else(|| {
                    RocmLoadError::CertifiedPathRemap {
                        reason: format!(
                            "descriptor map is missing candidate {}",
                            library.library_id
                        ),
                    }
                })?;
                if !is_retained_descriptor_path(&path) {
                    return Err(RocmLoadError::CertifiedPathRemap {
                        reason: format!(
                            "candidate {} must use /proc/self/fd/<decimal descriptor>",
                            library.library_id
                        ),
                    });
                }
                let mut remapped = library.clone();
                remapped.path = path;
                Ok(remapped)
            })
            .collect::<Result<Vec<_>, RocmLoadError>>()?;
        let dependencies = self
            .dependencies
            .iter()
            .map(|dependency| {
                let path = descriptor_paths
                    .remove(&dependency.library_id)
                    .ok_or_else(|| RocmLoadError::CertifiedPathRemap {
                        reason: format!(
                            "descriptor map is missing dependency {}",
                            dependency.library_id
                        ),
                    })?;
                if !is_retained_descriptor_path(&path) {
                    return Err(RocmLoadError::CertifiedPathRemap {
                        reason: format!(
                            "dependency {} must use /proc/self/fd/<decimal descriptor>",
                            dependency.library_id
                        ),
                    });
                }
                let mut remapped = dependency.clone();
                remapped.path = path;
                Ok(remapped)
            })
            .collect::<Result<Vec<_>, RocmLoadError>>()?;
        if !descriptor_paths.is_empty() {
            return Err(RocmLoadError::CertifiedPathRemap {
                reason: "descriptor map contains an unknown library ID".to_owned(),
            });
        }
        let remapped = Self {
            libraries,
            dependencies,
            dependency_edges: self.dependency_edges.clone(),
        };
        validate_retained_descriptor_set(&remapped)?;
        Ok(remapped)
    }

    fn path_map(&self) -> BTreeMap<String, PathBuf> {
        self.libraries
            .iter()
            .map(|library| (library.library_id.to_owned(), library.path.clone()))
            .chain(
                self.dependencies
                    .iter()
                    .map(|library| (library.library_id.clone(), library.path.clone())),
            )
            .collect()
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn soname_map(&self) -> BTreeMap<String, String> {
        self.libraries
            .iter()
            .filter_map(|library| {
                abi::LIBRARIES
                    .iter()
                    .find(|(library_id, _)| *library_id == library.library_id)
                    .map(|(_, soname)| (library.library_id.to_owned(), (*soname).to_owned()))
            })
            .chain(
                self.dependencies
                    .iter()
                    .map(|library| (library.library_id.clone(), library.soname.clone())),
            )
            .collect()
    }

    fn all_ids(&self) -> BTreeSet<String> {
        self.libraries
            .iter()
            .map(|library| library.library_id.to_owned())
            .chain(
                self.dependencies
                    .iter()
                    .map(|library| library.library_id.clone()),
            )
            .collect()
    }

    fn validate_dependency_graph(&self) -> Result<(), RocmLoadError> {
        let primary_ids = self
            .libraries
            .iter()
            .map(|library| library.library_id.to_owned())
            .collect::<BTreeSet<_>>();
        let dependency_ids = self
            .dependencies
            .iter()
            .map(|library| library.library_id.clone())
            .collect::<BTreeSet<_>>();
        if primary_ids.intersection(&dependency_ids).next().is_some() {
            return Err(RocmLoadError::DependencyGraph {
                reason: "primary and dependency library IDs must be disjoint".to_owned(),
            });
        }
        for dependency in &self.dependencies {
            let valid_text = |value: &str| {
                !value.is_empty()
                    && value.len() <= 255
                    && !value.contains('/')
                    && !value.contains('\0')
            };
            if !valid_text(&dependency.library_id)
                || !valid_text(&dependency.soname)
                || dependency.abi_version != "6.1.0"
                || dependency.unsafe_owner != "sim_backend_rocm::loader"
                || (!is_retained_descriptor_path(&dependency.path)
                    && dependency.path.file_name().and_then(|name| name.to_str())
                        != Some(dependency.soname.as_str()))
            {
                return Err(RocmLoadError::DependencyGraph {
                    reason: format!(
                        "dependency {} does not match the fixed ROCm certification contract",
                        dependency.library_id
                    ),
                });
            }
        }
        let mut sonames = abi::LIBRARIES
            .iter()
            .map(|(_, soname)| (*soname).to_owned())
            .collect::<BTreeSet<_>>();
        if self
            .dependencies
            .iter()
            .any(|dependency| !sonames.insert(dependency.soname.clone()))
        {
            return Err(RocmLoadError::DependencyGraph {
                reason: "every certified ROCm object must have a unique SONAME".to_owned(),
            });
        }
        let all_ids = primary_ids
            .union(&dependency_ids)
            .cloned()
            .collect::<BTreeSet<_>>();
        for edge in &self.dependency_edges {
            if edge.consumer == edge.dependency
                || !all_ids.contains(&edge.consumer)
                || !all_ids.contains(&edge.dependency)
            {
                return Err(RocmLoadError::DependencyGraph {
                    reason: format!(
                        "invalid vendor dependency edge {} -> {}",
                        edge.consumer, edge.dependency
                    ),
                });
            }
        }
        let mut reachable = primary_ids;
        loop {
            let previous = reachable.len();
            for edge in &self.dependency_edges {
                if reachable.contains(&edge.consumer) {
                    reachable.insert(edge.dependency.clone());
                }
            }
            if reachable.len() == previous {
                break;
            }
        }
        if dependency_ids
            .iter()
            .any(|dependency| !reachable.contains(dependency))
        {
            return Err(RocmLoadError::DependencyGraph {
                reason:
                    "every certified vendor dependency must be reachable from a primary library"
                        .to_owned(),
            });
        }
        self.topological_load_order().map(|_| ())
    }
}

fn is_retained_descriptor_path(path: &Path) -> bool {
    retained_descriptor_number(path).is_some()
}

fn retained_descriptor_number(path: &Path) -> Option<u32> {
    let Some(path) = path.to_str() else {
        return None;
    };
    let Some(descriptor) = path.strip_prefix("/proc/self/fd/") else {
        return None;
    };
    let Ok(descriptor_number) = descriptor.parse::<u32>() else {
        return None;
    };
    (descriptor_number >= 3 && descriptor_number.to_string() == descriptor)
        .then_some(descriptor_number)
}

#[cfg(target_os = "linux")]
fn validate_sealed_memfd(path: &Path) -> Result<(), RocmLoadError> {
    use std::os::fd::AsRawFd;

    let descriptor =
        retained_descriptor_number(path).ok_or_else(|| RocmLoadError::CertifiedPathRemap {
            reason: format!(
                "{} is not a canonical retained descriptor path",
                path.display()
            ),
        })?;
    let file = fs::File::open(path).map_err(|error| RocmLoadError::CertifiedPathRemap {
        reason: format!("retained descriptor {descriptor} is not open: {error}"),
    })?;
    let metadata = file
        .metadata()
        .map_err(|error| RocmLoadError::CertifiedPathRemap {
            reason: format!("retained descriptor {descriptor} has no metadata: {error}"),
        })?;
    if !metadata.file_type().is_file() {
        return Err(RocmLoadError::CertifiedPathRemap {
            reason: format!("retained descriptor {descriptor} is not a regular file"),
        });
    }
    // SAFETY: F_GET_SEALS reads the seal bitmask from the live descriptor and has no pointer
    // argument. `file` owns the duplicated descriptor for this check.
    let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    let required = libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
    if seals < 0 || seals & required != required {
        return Err(RocmLoadError::CertifiedPathRemap {
            reason: format!(
                "retained descriptor {descriptor} is not an immutable sealed memfd image"
            ),
        });
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn validate_sealed_memfd(path: &Path) -> Result<(), RocmLoadError> {
    Err(RocmLoadError::CertifiedPathRemap {
        reason: format!(
            "sealed memfd descriptor validation for {} requires Linux",
            path.display()
        ),
    })
}

fn validate_retained_descriptor_set(library_set: &RocmLibrarySet) -> Result<(), RocmLoadError> {
    let mut paths = BTreeSet::new();
    for path in library_set
        .libraries
        .iter()
        .map(|library| &library.path)
        .chain(library_set.dependencies.iter().map(|library| &library.path))
    {
        if !is_retained_descriptor_path(path) || !paths.insert(path) {
            return Err(RocmLoadError::CertifiedPathRemap {
                reason:
                    "every candidate must use a distinct canonical /proc/self/fd/<descriptor> path"
                        .to_owned(),
            });
        }
        validate_sealed_memfd(path)?;
    }
    library_set.validate_dependency_graph()?;
    Ok(())
}

impl DiscoveryRoot {
    fn new(source: DiscoverySource, path: impl Into<PathBuf>) -> Self {
        Self {
            source,
            path: path.into(),
        }
    }

    pub const fn source(&self) -> DiscoverySource {
        self.source
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentVersions {
    pub hip_driver: String,
    pub hip_runtime: String,
    pub hiprtc: String,
    pub rocblas: String,
    pub miopen: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RocmDeviceProperties {
    name: String,
    total_memory_bytes: u64,
    major: u32,
    minor: u32,
    architecture: Option<String>,
    has_fp16: bool,
}

impl RocmDeviceProperties {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }

    pub const fn major(&self) -> u32 {
        self.major
    }

    pub const fn minor(&self) -> u32 {
        self.minor
    }

    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    pub const fn has_fp16(&self) -> bool {
        self.has_fp16
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackageSignatureContract {
    signer: String,
    abi_manifest_sha256: &'static str,
    signature_algorithm: String,
    signature_coverage: String,
    runtime_root: PathBuf,
    adapter_manifest: Vec<u8>,
    package_policy: Vec<u8>,
    package_coverage: Vec<u8>,
    signature_receipt: Vec<u8>,
    ffi_contracts: Vec<u8>,
}

impl PackageSignatureContract {
    pub fn signer(&self) -> &str {
        &self.signer
    }

    pub const fn abi_manifest_sha256(&self) -> &'static str {
        self.abi_manifest_sha256
    }

    pub fn signature_algorithm(&self) -> &str {
        &self.signature_algorithm
    }

    pub fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }

    pub fn signature_coverage(&self) -> &str {
        &self.signature_coverage
    }

    pub fn adapter_manifest_bytes(&self) -> &[u8] {
        &self.adapter_manifest
    }

    pub fn package_policy_bytes(&self) -> &[u8] {
        &self.package_policy
    }

    pub fn package_coverage_bytes(&self) -> &[u8] {
        &self.package_coverage
    }

    pub fn signature_receipt_bytes(&self) -> &[u8] {
        &self.signature_receipt
    }

    pub fn ffi_contracts_bytes(&self) -> &[u8] {
        &self.ffi_contracts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRocmPackageRoot {
    discovery_root: DiscoveryRoot,
    signature_contract: PackageSignatureContract,
}

impl VerifiedRocmPackageRoot {
    pub fn discovery_root(&self) -> &DiscoveryRoot {
        &self.discovery_root
    }

    pub fn signature_contract(&self) -> &PackageSignatureContract {
        &self.signature_contract
    }
}

pub trait PlatformPackageVerifier {
    fn verify_rocm_package(
        &self,
        package_root: &Path,
        contract: &PackageSignatureContract,
    ) -> Result<(), String>;
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RocmLoadError {
    #[error("ROCm requires target {required}; current target is {actual}")]
    UnsupportedTarget {
        required: &'static str,
        actual: String,
    },
    #[error("ROCm discovery root from {origin} is invalid ({path}): {reason}")]
    InvalidRoot {
        origin: DiscoverySource,
        path: String,
        reason: String,
    },
    #[error("ROCm required library {library} was not found in ordered roots: {searched}")]
    MissingLibrary {
        library: &'static str,
        searched: String,
    },
    #[error("ROCm library {library} at {path} could not be loaded: {reason}")]
    LibraryLoad {
        library: &'static str,
        path: String,
        reason: String,
    },
    #[error("ROCm library {library} is missing required symbol {symbol}")]
    MissingSymbol {
        library: &'static str,
        symbol: String,
    },
    #[error("ROCm {component} version {found} is below required {minimum}")]
    VersionTooOld {
        component: &'static str,
        found: String,
        minimum: String,
    },
    #[error("ROCm ABI manifest is invalid: {reason}")]
    AbiManifest { reason: String },
    #[error("ROCm package metadata is invalid at {path}: {reason}")]
    PackageMetadata { path: String, reason: String },
    #[error("ROCm package signature verification failed at {path}: {reason}")]
    PackageVerification { path: String, reason: String },
    #[error("ROCm certified library path remap is invalid: {reason}")]
    CertifiedPathRemap { reason: String },
    #[error("ROCm vendor dependency graph is invalid: {reason}")]
    DependencyGraph { reason: String },
    #[error("ROCm loader namespace binding proof failed: {reason}")]
    BindingProof { reason: String },
    #[error("ROCm runtime probe {function} failed with status {status}")]
    Probe { function: &'static str, status: i32 },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    libraries: Vec<ManifestLibrary>,
    layouts: Vec<ManifestLayout>,
    headers: Vec<ManifestHeader>,
    unsafe_owner: String,
    package: ManifestPackage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLibrary {
    name: String,
    file: String,
    symbols: Vec<ManifestSymbol>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestSymbol {
    name: String,
    signature: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLayout {
    name: String,
    size: usize,
    align: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestHeader {
    name: String,
    source: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestPackage {
    redistributes_amd_runtime: bool,
    signature_required: bool,
    notice_file: String,
}

pub fn discover_from_environment() -> Result<Vec<DiscoveryRoot>, RocmLoadError> {
    let mut roots = Vec::new();
    if let Some(root) = env::var_os("SIM_ROCM_ROOT") {
        roots.push(DiscoveryRoot::new(DiscoverySource::SimRocmRoot, root));
    }
    if let Some(root) = env::var_os("ROCM_PATH") {
        roots.push(DiscoveryRoot::new(DiscoverySource::RocmPath, root));
    }
    Ok(deduplicate_roots(roots))
}

pub fn discover_library_set(roots: &[DiscoveryRoot]) -> Result<RocmLibrarySet, RocmLoadError> {
    validate_manifest()?;
    resolve_library_set(roots)
}

pub fn admit_signed_package_root(
    root: impl Into<PathBuf>,
    verifier: &dyn PlatformPackageVerifier,
) -> Result<DiscoveryRoot, RocmLoadError> {
    Ok(verify_signed_package_root(root, verifier)?.discovery_root)
}

pub fn verify_signed_package_root(
    root: impl Into<PathBuf>,
    verifier: &dyn PlatformPackageVerifier,
) -> Result<VerifiedRocmPackageRoot, RocmLoadError> {
    let root = DiscoveryRoot::new(DiscoverySource::SignedPackage, root);
    let canonical_package_root = checked_sdk_root(&root)?;
    let contract = validate_signed_package(&canonical_package_root)?;
    verifier
        .verify_rocm_package(&canonical_package_root, &contract)
        .map_err(|reason| RocmLoadError::PackageVerification {
            path: canonical_package_root.display().to_string(),
            reason,
        })?;
    let revalidated_contract = validate_signed_package(&canonical_package_root)?;
    if revalidated_contract != contract {
        return Err(RocmLoadError::PackageMetadata {
            path: canonical_package_root.display().to_string(),
            reason: "signed package changed during platform verification".to_owned(),
        });
    }
    let runtime_root = checked_sdk_root(&DiscoveryRoot::new(
        DiscoverySource::SignedPackage,
        revalidated_contract.runtime_root(),
    ))?;
    if runtime_root.starts_with(&canonical_package_root) {
        return Err(RocmLoadError::PackageMetadata {
            path: canonical_package_root.display().to_string(),
            reason: "signed adapter packages must reference a separately installed ROCm SDK root"
                .to_owned(),
        });
    }
    Ok(VerifiedRocmPackageRoot {
        discovery_root: DiscoveryRoot::new(DiscoverySource::SignedPackage, runtime_root),
        signature_contract: revalidated_contract,
    })
}

pub fn discover_with_verified_package_roots(
    package_roots: impl IntoIterator<Item = PathBuf>,
    verifier: &dyn PlatformPackageVerifier,
) -> Result<Vec<DiscoveryRoot>, RocmLoadError> {
    let mut roots = discover_from_environment()?;
    for package_root in package_roots {
        roots.push(admit_signed_package_root(package_root, verifier)?);
    }
    Ok(deduplicate_roots(roots))
}

fn target_gate() -> Result<(), RocmLoadError> {
    if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        Ok(())
    } else {
        Err(RocmLoadError::UnsupportedTarget {
            required: abi::REQUIRED_TARGET,
            actual: env!("SIM_ROCM_BUILD_TARGET").to_owned(),
        })
    }
}

fn deduplicate_roots(roots: Vec<DiscoveryRoot>) -> Vec<DiscoveryRoot> {
    let mut seen = BTreeSet::new();
    roots
        .into_iter()
        .filter(|root| seen.insert(root.path.clone()))
        .collect()
}

fn checked_sdk_root(root: &DiscoveryRoot) -> Result<PathBuf, RocmLoadError> {
    // ArtifactRoot remains the generic path-capability owner. This adapter resolves only the four
    // fixed vendor library names and never returns a caller-selected relative path capability.
    if !root.path.is_absolute() {
        return Err(invalid_root(root, "root must be absolute"));
    }
    let canonical = root
        .path
        .canonicalize()
        .map_err(|error| invalid_root(root, &error.to_string()))?;
    if !canonical.is_dir() {
        return Err(invalid_root(root, "root is not a directory"));
    }
    Ok(canonical)
}

fn invalid_root(root: &DiscoveryRoot, reason: &str) -> RocmLoadError {
    RocmLoadError::InvalidRoot {
        origin: root.source,
        path: root.path.display().to_string(),
        reason: reason.to_owned(),
    }
}

fn library_path(root: &Path, filename: &str) -> Option<PathBuf> {
    [
        root.join("lib").join(filename),
        root.join("lib64").join(filename),
    ]
    .into_iter()
    .find_map(|candidate| {
        let parent = candidate.parent()?.canonicalize().ok()?;
        if !parent.starts_with(root) {
            return None;
        }
        let checked = parent.join(candidate.file_name()?);
        let metadata = fs::symlink_metadata(&checked).ok()?;
        (metadata.file_type().is_file() && !metadata.file_type().is_symlink()).then_some(checked)
    })
}

fn resolve_library_set(roots: &[DiscoveryRoot]) -> Result<RocmLibrarySet, RocmLoadError> {
    let mut searched = Vec::new();
    let mut first_missing = abi::LIBRARIES[0].0;
    for root in roots {
        let canonical = checked_sdk_root(root)?;
        searched.push(format!("{}={}", root.source, canonical.display()));
        let mut paths = BTreeMap::new();
        let mut complete = true;
        for &(library, filename) in abi::LIBRARIES {
            if let Some(path) = library_path(&canonical, filename) {
                paths.insert(library, path);
            } else {
                first_missing = library;
                complete = false;
                break;
            }
        }
        if complete {
            let libraries = abi::LIBRARIES
                .iter()
                .map(|&(library_id, _)| {
                    let path = paths
                        .remove(library_id)
                        .ok_or(RocmLoadError::MissingLibrary {
                            library: library_id,
                            searched: "resolved root did not contain a complete library set"
                                .to_owned(),
                        })?;
                    Ok(RocmLibraryCandidate {
                        library_id,
                        path,
                        abi_version: "6.1.0",
                        required_symbols: abi::SYMBOLS
                            .iter()
                            .filter(|symbol| symbol.library == library_id)
                            .map(|symbol| symbol.name)
                            .collect(),
                        unsafe_owner: "sim_backend_rocm::loader",
                    })
                })
                .collect::<Result<Vec<_>, RocmLoadError>>()?;
            return Ok(RocmLibrarySet {
                libraries,
                dependencies: Vec::new(),
                dependency_edges: Vec::new(),
            });
        }
    }
    Err(RocmLoadError::MissingLibrary {
        library: first_missing,
        searched: if searched.is_empty() {
            "<none>; set SIM_ROCM_ROOT or ROCM_PATH, or install the signed Zed adapter package"
                .to_owned()
        } else {
            searched.join(", ")
        },
    })
}

fn validate_manifest() -> Result<Manifest, RocmLoadError> {
    let manifest: Manifest =
        serde_json::from_str(ABI_MANIFEST).map_err(|error| RocmLoadError::AbiManifest {
            reason: error.to_string(),
        })?;
    if manifest.schema_version != abi::ABI_SCHEMA_VERSION
        || manifest.backend != "rocm"
        || manifest.abi_floor != "6.1.0"
        || manifest.target != abi::REQUIRED_TARGET
        || manifest.unsafe_owner != "sim_backend_rocm::loader"
    {
        return Err(RocmLoadError::AbiManifest {
            reason: "identity, floor, target, or unsafe owner differs from the compiled contract"
                .to_owned(),
        });
    }
    if manifest.package.redistributes_amd_runtime
        || !manifest.package.signature_required
        || manifest.package.notice_file != "LICENSES"
    {
        return Err(RocmLoadError::AbiManifest {
            reason: "package redistribution, signature, or notice policy differs from the compiled contract"
                .to_owned(),
        });
    }
    if manifest.headers.len() != 9
        || manifest.headers.iter().any(|header| {
            header.name.is_empty()
                || !header.source.starts_with("https://github.com/ROCm/")
                || header.sha256.len() != 64
                || !header.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
    {
        return Err(RocmLoadError::AbiManifest {
            reason: "reviewed header provenance or SHA-256 digest set is incomplete".to_owned(),
        });
    }

    let manifest_symbols: BTreeSet<_> = manifest
        .libraries
        .iter()
        .flat_map(|library| {
            library.symbols.iter().map(move |symbol| {
                (
                    library.name.as_str(),
                    symbol.name.as_str(),
                    symbol.signature.as_str(),
                )
            })
        })
        .collect();
    let compiled_symbols: BTreeSet<_> = abi::SYMBOLS
        .iter()
        .map(|symbol| (symbol.library, symbol.name, symbol.signature))
        .collect();
    if manifest_symbols != compiled_symbols {
        return Err(RocmLoadError::AbiManifest {
            reason: "symbol names or C signatures differ from the reviewed declarations".to_owned(),
        });
    }
    let manifest_libraries: BTreeSet<_> = manifest
        .libraries
        .iter()
        .map(|library| (library.name.as_str(), library.file.as_str()))
        .collect();
    if manifest_libraries != abi::LIBRARIES.iter().copied().collect() {
        return Err(RocmLoadError::AbiManifest {
            reason: "library names differ from the compiled discovery contract".to_owned(),
        });
    }
    let layouts: BTreeMap<_, _> = manifest
        .layouts
        .iter()
        .map(|layout| (layout.name.as_str(), (layout.size, layout.align)))
        .collect();
    let expected = [
        (
            "hipUUID",
            (
                std::mem::size_of::<abi::HipUuid>(),
                std::mem::align_of::<abi::HipUuid>(),
            ),
        ),
        (
            "hipIpcMemHandle_t",
            (
                std::mem::size_of::<abi::HipIpcMemHandle>(),
                std::mem::align_of::<abi::HipIpcMemHandle>(),
            ),
        ),
        (
            "miopenConvAlgoPerf_t",
            (
                std::mem::size_of::<abi::MiopenConvAlgoPerf>(),
                std::mem::align_of::<abi::MiopenConvAlgoPerf>(),
            ),
        ),
    ];
    if layouts != expected.into_iter().collect() {
        return Err(RocmLoadError::AbiManifest {
            reason: "C struct size or alignment differs from the compiled target".to_owned(),
        });
    }
    Ok(manifest)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageManifest {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    abi_manifest_sha256: String,
    ffi_contracts_sha256: String,
    redistributes_amd_runtime: bool,
    signer: String,
    signature_algorithm: String,
    signature_domain: String,
    signature_coverage: String,
    runtime_root: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PackagePolicy {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    redistributes_amd_runtime: bool,
    allowed_payloads: Vec<String>,
    required_payloads: Vec<String>,
    forbidden_payloads: Vec<String>,
    signature_algorithm: String,
    signature_domain: String,
    signature_payload_format: String,
    signature_receipt_format: String,
    signature_coverage: String,
    signature_coverage_format: String,
    signature_coverage_excludes: Vec<String>,
    signature_verifier: String,
    runtime_root_policy: String,
    signature_required_before_install: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedPayload {
    size: u64,
    sha256: String,
    captured: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CoverageEntry {
    size: u64,
    sha256: String,
}

fn validate_signed_package(root: &Path) -> Result<PackageSignatureContract, RocmLoadError> {
    let (directories, payloads) = inspect_package_tree(root)?;
    let policy_bytes = captured_payload(root, &payloads, "package-policy.json")?;
    let policy: PackagePolicy = parse_package_json(root, "package-policy.json", policy_bytes)?;
    let compiled_policy: PackagePolicy =
        serde_json::from_str(PACKAGE_POLICY).map_err(|error| RocmLoadError::PackageMetadata {
            path: "compiled ROCm package policy".to_owned(),
            reason: error.to_string(),
        })?;
    validate_policy_contract(root, &policy)?;
    if policy != compiled_policy {
        return Err(package_metadata_error(
            root,
            "package-policy.json",
            "installed policy differs from the compiled reviewed policy",
        ));
    }
    validate_tree_membership(root, &policy, &directories, &payloads)?;

    let coverage_bytes = captured_payload(root, &payloads, "package-coverage.sha256")?;
    let coverage = parse_coverage(root, coverage_bytes)?;
    validate_coverage(root, &policy, &payloads, &coverage)?;

    let ffi_contracts = captured_payload(root, &payloads, "ffi-contracts-v1.json")?;
    let manifest_bytes = captured_payload(root, &payloads, "adapter-manifest.json")?;
    let manifest: PackageManifest =
        parse_package_json(root, "adapter-manifest.json", manifest_bytes)?;
    let ffi_contracts_sha256 = sha256_hex(ffi_contracts)?;
    if manifest.schema_version != 2
        || manifest.backend != "rocm"
        || manifest.abi_floor != "6.1.0"
        || manifest.redistributes_amd_runtime
        || manifest.abi_manifest_sha256 != ABI_MANIFEST_SHA256
        || manifest.ffi_contracts_sha256 != ffi_contracts_sha256
        || manifest.signer.is_empty()
        || manifest.signature_algorithm != "ed25519"
        || manifest.signature_domain != "zed-sim-rocm-package-v1"
        || manifest.signature_coverage != "package-coverage-v1"
        || !manifest.runtime_root.is_absolute()
    {
        return Err(package_metadata_error(
            root,
            "adapter-manifest.json",
            "signature-covered adapter policy is incomplete or permits AMD runtime redistribution",
        ));
    }
    if manifest.signature_algorithm != policy.signature_algorithm
        || manifest.signature_domain != policy.signature_domain
        || manifest.signature_coverage != policy.signature_coverage
    {
        return Err(package_metadata_error(
            root,
            "adapter-manifest.json",
            "manifest signature contract differs from package-policy.json",
        ));
    }
    let receipt = captured_payload(root, &payloads, "adapter-manifest.sig")?;
    if receipt.is_empty() {
        return Err(package_metadata_error(
            root,
            "adapter-manifest.sig",
            "native Ed25519 signature receipt must not be empty",
        ));
    }
    Ok(PackageSignatureContract {
        signer: manifest.signer,
        abi_manifest_sha256: ABI_MANIFEST_SHA256,
        signature_algorithm: manifest.signature_algorithm,
        signature_coverage: manifest.signature_coverage,
        runtime_root: manifest.runtime_root,
        adapter_manifest: manifest_bytes.to_vec(),
        package_policy: policy_bytes.to_vec(),
        package_coverage: coverage_bytes.to_vec(),
        signature_receipt: receipt.to_vec(),
        ffi_contracts: ffi_contracts.to_vec(),
    })
}

fn parse_package_json<'a, T: Deserialize<'a>>(
    root: &Path,
    relative: &str,
    bytes: &'a [u8],
) -> Result<T, RocmLoadError> {
    serde_json::from_slice(bytes)
        .map_err(|error| package_metadata_error(root, relative, &error.to_string()))
}

fn validate_policy_contract(root: &Path, policy: &PackagePolicy) -> Result<(), RocmLoadError> {
    const ALLOWED: &[&str] = &[
        "abi/symbols-v1.json",
        "ffi-contracts-v1.json",
        "LICENSES",
        "package-policy.json",
        "adapter-manifest.json",
        "adapter-manifest.sig",
        "package-coverage.sha256",
        "kernels/manifest.json",
        "kernels/*.hsaco",
    ];
    const REQUIRED: &[&str] = &[
        "abi/symbols-v1.json",
        "ffi-contracts-v1.json",
        "LICENSES",
        "package-policy.json",
        "adapter-manifest.json",
        "adapter-manifest.sig",
        "package-coverage.sha256",
    ];
    const FORBIDDEN: &[&str] = &[
        "libamdhip64.so*",
        "libhiprtc.so*",
        "librocblas.so*",
        "libMIOpen.so*",
    ];
    const EXCLUDED: &[&str] = &["adapter-manifest.sig", "package-coverage.sha256"];

    let exact = |actual: &[String], expected: &[&str]| {
        actual
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
    };
    if policy.schema_version != 2
        || policy.backend != "rocm"
        || policy.abi_floor != "6.1.0"
        || policy.target != abi::REQUIRED_TARGET
        || policy.redistributes_amd_runtime
        || !exact(&policy.allowed_payloads, ALLOWED)
        || !exact(&policy.required_payloads, REQUIRED)
        || !exact(&policy.forbidden_payloads, FORBIDDEN)
        || policy.signature_algorithm != "ed25519"
        || policy.signature_domain != "zed-sim-rocm-package-v1"
        || policy.signature_payload_format != "domain-nul-u64be-signer-u64be-coverage"
        || policy.signature_receipt_format != "canonical-json-v1"
        || policy.signature_coverage != "package-coverage-v1"
        || policy.signature_coverage_format != "sha256-decimal-size-two-space-path"
        || !exact(&policy.signature_coverage_excludes, EXCLUDED)
        || policy.signature_verifier != "sim_runtime-native-rust-ed25519"
        || policy.runtime_root_policy != "absolute-existing-directory-outside-package-output"
        || !policy.signature_required_before_install
    {
        return Err(package_metadata_error(
            root,
            "package-policy.json",
            "package policy differs from the native ROCm admission contract",
        ));
    }
    Ok(())
}

fn inspect_package_tree(
    root: &Path,
) -> Result<(BTreeSet<String>, BTreeMap<String, ObservedPayload>), RocmLoadError> {
    let mut directories = BTreeSet::new();
    let mut payloads = BTreeMap::new();
    let mut pending = vec![(root.to_path_buf(), String::new())];
    let mut entry_count = 0_usize;
    while let Some((directory, prefix)) = pending.pop() {
        let entries = fs::read_dir(&directory).map_err(|error| RocmLoadError::PackageMetadata {
            path: directory.display().to_string(),
            reason: error.to_string(),
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| RocmLoadError::PackageMetadata {
                path: directory.display().to_string(),
                reason: error.to_string(),
            })?;
            entry_count = entry_count.saturating_add(1);
            if entry_count > MAX_PACKAGE_ENTRIES {
                return Err(package_metadata_error(
                    root,
                    "",
                    "package tree exceeds the bounded entry count",
                ));
            }
            let name = entry.file_name().into_string().map_err(|_| {
                package_metadata_error(root, "", "package paths must be valid UTF-8")
            })?;
            let relative = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            validate_relative_package_path(root, &relative)?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| package_metadata_error(root, &relative, &error.to_string()))?;
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                return Err(package_metadata_error(
                    root,
                    &relative,
                    "symbolic links are forbidden in signed packages",
                ));
            }
            if file_type.is_dir() {
                directories.insert(relative.clone());
                pending.push((path, relative));
            } else if file_type.is_file() {
                let capture = matches!(
                    relative.as_str(),
                    "adapter-manifest.json"
                        | "adapter-manifest.sig"
                        | "ffi-contracts-v1.json"
                        | "package-policy.json"
                        | "package-coverage.sha256"
                );
                let payload = inspect_package_file(root, &relative, &path, capture)?;
                payloads.insert(relative, payload);
            } else {
                return Err(package_metadata_error(
                    root,
                    &relative,
                    "only regular files and declared directories are permitted",
                ));
            }
        }
    }
    Ok((directories, payloads))
}

fn inspect_package_file(
    root: &Path,
    relative: &str,
    path: &Path,
    capture: bool,
) -> Result<ObservedPayload, RocmLoadError> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .map_err(|error| package_metadata_error(root, relative, &error.to_string()))?;
    let metadata = file
        .metadata()
        .map_err(|error| package_metadata_error(root, relative, &error.to_string()))?;
    if !metadata.is_file() {
        return Err(package_metadata_error(
            root,
            relative,
            "package payload is not a regular file",
        ));
    }
    if capture && metadata.len() > MAX_PACKAGE_METADATA_BYTES {
        return Err(package_metadata_error(
            root,
            relative,
            "package metadata exceeds the bounded size",
        ));
    }
    let mut sha256 = Sha256::new();
    let mut size = 0_u64;
    let mut bytes = capture.then(|| Vec::with_capacity(metadata.len() as usize));
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| package_metadata_error(root, relative, &error.to_string()))?;
        if read == 0 {
            break;
        }
        size = size.checked_add(read as u64).ok_or_else(|| {
            package_metadata_error(root, relative, "package payload size overflow")
        })?;
        sha256.update(&buffer[..read]);
        if let Some(bytes) = &mut bytes {
            bytes.extend_from_slice(&buffer[..read]);
        }
    }
    if size != metadata.len() {
        return Err(package_metadata_error(
            root,
            relative,
            "package payload changed while it was inspected",
        ));
    }
    Ok(ObservedPayload {
        size,
        sha256: sha256.finish_hex(size)?,
        captured: bytes,
    })
}

fn validate_relative_package_path(root: &Path, relative: &str) -> Result<(), RocmLoadError> {
    if relative.is_empty()
        || relative.len() > MAX_PACKAGE_PATH_BYTES
        || relative.starts_with('/')
        || relative.ends_with('/')
        || relative.contains("//")
        || relative.contains('\\')
        || relative
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(package_metadata_error(
            root,
            relative,
            "package path is not canonical",
        ));
    }
    Ok(())
}

fn validate_tree_membership(
    root: &Path,
    policy: &PackagePolicy,
    directories: &BTreeSet<String>,
    payloads: &BTreeMap<String, ObservedPayload>,
) -> Result<(), RocmLoadError> {
    for directory in directories {
        if !policy
            .allowed_payloads
            .iter()
            .any(|pattern| pattern.starts_with(&format!("{directory}/")))
        {
            return Err(package_metadata_error(
                root,
                directory,
                "package directory is not declared by the allowlist",
            ));
        }
    }
    for relative in payloads.keys() {
        let basename = relative.rsplit('/').next().unwrap_or(relative);
        if policy
            .forbidden_payloads
            .iter()
            .any(|pattern| glob_matches(basename, pattern) || glob_matches(relative, pattern))
        {
            return Err(package_metadata_error(
                root,
                relative,
                "AMD runtime redistribution is forbidden",
            ));
        }
        if !policy
            .allowed_payloads
            .iter()
            .any(|pattern| glob_matches(relative, pattern))
        {
            return Err(package_metadata_error(
                root,
                relative,
                "package payload is not declared by the allowlist",
            ));
        }
    }
    for required in &policy.required_payloads {
        if !payloads.contains_key(required) {
            return Err(package_metadata_error(
                root,
                required,
                "required package payload is missing",
            ));
        }
    }
    Ok(())
}

fn glob_matches(value: &str, pattern: &str) -> bool {
    let value = value.as_bytes();
    let pattern = pattern.as_bytes();
    let mut matches = vec![false; value.len() + 1];
    matches[0] = true;
    for pattern_byte in pattern {
        let mut next = vec![false; value.len() + 1];
        if *pattern_byte == b'*' {
            for index in 0..=value.len() {
                next[index] =
                    matches[index] || (index > 0 && value[index - 1] != b'/' && next[index - 1]);
            }
        } else {
            for index in 1..=value.len() {
                next[index] = matches[index - 1]
                    && ((*pattern_byte == b'?' && value[index - 1] != b'/')
                        || *pattern_byte == value[index - 1]);
            }
        }
        matches = next;
    }
    matches[value.len()]
}

fn parse_coverage(
    root: &Path,
    bytes: &[u8],
) -> Result<BTreeMap<String, CoverageEntry>, RocmLoadError> {
    if bytes.is_empty() || !bytes.ends_with(b"\n") {
        return Err(package_metadata_error(
            root,
            "package-coverage.sha256",
            "coverage must be nonempty and newline terminated",
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|error| {
        package_metadata_error(root, "package-coverage.sha256", &error.to_string())
    })?;
    let mut entries = BTreeMap::new();
    let mut previous_path: Option<&str> = None;
    for line in text.lines() {
        let (fields, relative) = line.split_once("  ").ok_or_else(|| {
            package_metadata_error(
                root,
                "package-coverage.sha256",
                "coverage line must contain digest, size, and path",
            )
        })?;
        let (sha256, size) = fields.split_once(' ').ok_or_else(|| {
            package_metadata_error(
                root,
                "package-coverage.sha256",
                "coverage line must contain one digest-size separator",
            )
        })?;
        if size.contains(' ')
            || sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || size.is_empty()
            || !size.bytes().all(|byte| byte.is_ascii_digit())
            || (size.len() > 1 && size.starts_with('0'))
        {
            return Err(package_metadata_error(
                root,
                "package-coverage.sha256",
                "coverage digest or size is not canonical",
            ));
        }
        validate_relative_package_path(root, relative)?;
        if previous_path.is_some_and(|previous| previous >= relative) {
            return Err(package_metadata_error(
                root,
                "package-coverage.sha256",
                "coverage paths must be unique and strictly sorted",
            ));
        }
        previous_path = Some(relative);
        let size = size.parse::<u64>().map_err(|error| {
            package_metadata_error(root, "package-coverage.sha256", &error.to_string())
        })?;
        entries.insert(
            relative.to_owned(),
            CoverageEntry {
                size,
                sha256: sha256.to_owned(),
            },
        );
    }
    Ok(entries)
}

fn validate_coverage(
    root: &Path,
    policy: &PackagePolicy,
    payloads: &BTreeMap<String, ObservedPayload>,
    coverage: &BTreeMap<String, CoverageEntry>,
) -> Result<(), RocmLoadError> {
    let excluded: BTreeSet<_> = policy.signature_coverage_excludes.iter().collect();
    let expected_paths: BTreeSet<_> = payloads
        .keys()
        .filter(|path| !excluded.contains(path))
        .cloned()
        .collect();
    if coverage.keys().cloned().collect::<BTreeSet<_>>() != expected_paths {
        return Err(package_metadata_error(
            root,
            "package-coverage.sha256",
            "coverage membership differs from the exact package payload tree",
        ));
    }
    for (relative, entry) in coverage {
        let observed = payloads.get(relative).ok_or_else(|| {
            package_metadata_error(root, relative, "coverage references a missing payload")
        })?;
        if entry.size != observed.size || entry.sha256 != observed.sha256 {
            return Err(package_metadata_error(
                root,
                relative,
                "coverage digest or size does not match the package payload",
            ));
        }
    }
    Ok(())
}

fn captured_payload<'a>(
    root: &Path,
    payloads: &'a BTreeMap<String, ObservedPayload>,
    relative: &str,
) -> Result<&'a [u8], RocmLoadError> {
    payloads
        .get(relative)
        .and_then(|payload| payload.captured.as_deref())
        .ok_or_else(|| {
            package_metadata_error(root, relative, "required package metadata is missing")
        })
}

fn package_metadata_error(root: &Path, relative: &str, reason: &str) -> RocmLoadError {
    let path = if relative.is_empty() {
        root.to_path_buf()
    } else {
        root.join(relative)
    };
    RocmLoadError::PackageMetadata {
        path: path.display().to_string(),
        reason: reason.to_owned(),
    }
}

fn sha256_hex(bytes: &[u8]) -> Result<String, RocmLoadError> {
    let byte_length = u64::try_from(bytes.len()).map_err(|_| RocmLoadError::PackageMetadata {
        path: "ffi-contracts-v1.json".to_owned(),
        reason: "catalog length exceeds the supported digest range".to_owned(),
    })?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest.finish_hex(byte_length)
}

struct Sha256 {
    hash: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
}

impl Sha256 {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    fn new() -> Self {
        Self {
            hash: Self::INITIAL,
            buffer: [0; 64],
            buffered: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        if self.buffered > 0 {
            let copied = (64 - self.buffered).min(input.len());
            self.buffer[self.buffered..self.buffered + copied].copy_from_slice(&input[..copied]);
            self.buffered += copied;
            input = &input[copied..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        let mut chunks = input.chunks_exact(64);
        for chunk in &mut chunks {
            let mut block = [0_u8; 64];
            block.copy_from_slice(chunk);
            self.compress(&block);
        }
        let remainder = chunks.remainder();
        self.buffer[..remainder.len()].copy_from_slice(remainder);
        self.buffered = remainder.len();
    }

    fn finish_hex(mut self, byte_length: u64) -> Result<String, RocmLoadError> {
        let bit_length =
            byte_length
                .checked_mul(8)
                .ok_or_else(|| RocmLoadError::PackageMetadata {
                    path: "signed package payload".to_owned(),
                    reason: "package payload bit length overflow".to_owned(),
                })?;
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_length.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        Ok(self.hash.iter().map(|word| format!("{word:08x}")).collect())
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut schedule = [0_u32; 64];
        for (index, bytes) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let small_zero = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let small_one = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small_zero)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(small_one);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.hash;
        for (index, round) in Self::ROUND.iter().enumerate() {
            let temporary_one = h
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ ((!e) & g))
                .wrapping_add(*round)
                .wrapping_add(schedule[index]);
            let temporary_two = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary_one);
            d = c;
            c = b;
            b = a;
            a = temporary_one.wrapping_add(temporary_two);
        }
        for (word, compressed) in self.hash.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *word = word.wrapping_add(compressed);
        }
    }
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum RocmExecutionError {
    #[error("ROCm device {device} is unavailable (device count: {device_count})")]
    InvalidDevice { device: u32, device_count: u32 },
    #[error("ROCm allocation of {bytes} bytes failed on device {device}: {message}")]
    OutOfMemory {
        device: u32,
        bytes: usize,
        message: String,
    },
    #[error("ROCm device {device} was lost during {operation}: {message}")]
    DeviceLost {
        device: u32,
        operation: &'static str,
        message: String,
    },
    #[error("invalid ROCm operation argument: {reason}")]
    InvalidArgument { reason: String },
    #[error("ROCm {operation} failed with status {status}: {message}")]
    Status {
        operation: &'static str,
        status: i32,
        message: String,
    },
    #[error("ROCm execution requires target {required}; current target is {actual}")]
    UnsupportedTarget {
        required: &'static str,
        actual: String,
    },
}

#[derive(Clone)]
pub struct RocmRuntime {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    libraries: Arc<linux::LoadedLibraries>,
    versions: ComponentVersions,
    device_count: u32,
    _certification_retention: Arc<dyn Any + Send + Sync>,
}

pub struct RocmAllocation {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    allocation: linux::DeviceAllocation,
    device: u32,
    byte_length: usize,
    runtime_identity: usize,
    _runtime: RocmRuntime,
}

impl RocmAllocation {
    pub const fn device(&self) -> u32 {
        self.device
    }

    pub const fn byte_length(&self) -> usize {
        self.byte_length
    }

    pub fn try_clone(&self) -> Result<Self, RocmExecutionError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let clone = self._runtime.allocate(self.device, self.byte_length)?;
            let stream = self._runtime.create_stream(self.device)?;
            self._runtime
                .copy_device_to_device(&stream, &clone, 0, self, 0, self.byte_length)?;
            stream.synchronize()?;
            return Ok(clone);
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }
}

pub struct RocmStream {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    stream: linux::ExecutionStream,
    device: u32,
    runtime_identity: usize,
    _runtime: RocmRuntime,
}

impl RocmStream {
    pub const fn device(&self) -> u32 {
        self.device
    }

    pub fn synchronize(&self) -> Result<(), RocmExecutionError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            return self.stream.synchronize();
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }
}

pub struct RocmEvent {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    event: linux::ExecutionEvent,
    device: u32,
    runtime_identity: usize,
    _runtime: RocmRuntime,
}

impl RocmEvent {
    pub const fn device(&self) -> u32 {
        self.device
    }

    pub fn synchronize(&self) -> Result<(), RocmExecutionError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            return self.event.synchronize();
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn unsupported_execution_target() -> RocmExecutionError {
    RocmExecutionError::UnsupportedTarget {
        required: abi::REQUIRED_TARGET,
        actual: env!("SIM_ROCM_BUILD_TARGET").to_owned(),
    }
}

impl RocmRuntime {
    /// Loads and probes the exact libraries in `library_set`.
    ///
    /// # Safety
    ///
    /// Immediately before this call, every candidate must have been authorized by
    /// `sim_runtime::NativeFfiRegistry` using the SHA-256 digest of the exact file, the
    /// candidate ABI version, and the actual ELF dynamic-symbol set. Each returned
    /// `CertifiedNativeFfi` must name the candidate library and
    /// `sim_backend_rocm::loader` as its unsafe owner. The caller must retain those
    /// certificates and descriptor owners in `certification_retention`; it must own every
    /// certificate and retained descriptor for its complete lifetime. The supplied set must have
    /// been produced by `RocmLibrarySet::remap_to_retained_descriptors`, and the underlying file
    /// contents must remain unchanged from certification through this call.
    pub unsafe fn load_certified(
        library_set: &RocmLibrarySet,
        certification_retention: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, RocmLoadError> {
        validate_retained_descriptor_set(library_set)?;
        target_gate()?;
        validate_manifest()?;
        let paths = library_set.path_map();
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let load_order = library_set.load_order()?;
            let sonames = library_set.soname_map();
            let libraries = Arc::new(linux::LoadedLibraries::load(
                &paths,
                &sonames,
                &load_order,
                library_set.dependencies(),
                library_set.dependency_edges(),
            )?);
            let (versions, device_count) = libraries.probe()?;
            return Ok(Self {
                libraries,
                versions,
                device_count,
                _certification_retention: certification_retention,
            });
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            drop(certification_retention);
            drop(paths);
            Err(RocmLoadError::UnsupportedTarget {
                required: abi::REQUIRED_TARGET,
                actual: env!("SIM_ROCM_BUILD_TARGET").to_owned(),
            })
        }
    }

    pub fn versions(&self) -> &ComponentVersions {
        &self.versions
    }

    pub const fn device_count(&self) -> u32 {
        self.device_count
    }

    pub fn loaded_library_count(&self) -> usize {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.libraries.library_count()
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            0
        }
    }

    fn runtime_identity(&self) -> usize {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            Arc::as_ptr(&self.libraries) as usize
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            0
        }
    }

    pub fn select_device(&self, device: u32) -> Result<(), RocmExecutionError> {
        if device >= self.device_count {
            return Err(RocmExecutionError::InvalidDevice {
                device,
                device_count: self.device_count,
            });
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            return self.libraries.select_device(device);
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    pub fn device_properties(
        &self,
        device: u32,
    ) -> Result<RocmDeviceProperties, RocmExecutionError> {
        self.select_device(device)?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            return self.libraries.device_properties(device);
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    pub fn allocate(
        &self,
        device: u32,
        byte_length: usize,
    ) -> Result<RocmAllocation, RocmExecutionError> {
        self.select_device(device)?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let allocation = self.libraries.allocate(device, byte_length)?;
            return Ok(RocmAllocation {
                allocation,
                device,
                byte_length,
                runtime_identity: self.runtime_identity(),
                _runtime: self.clone(),
            });
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = byte_length;
            Err(unsupported_execution_target())
        }
    }

    pub fn create_stream(&self, device: u32) -> Result<RocmStream, RocmExecutionError> {
        self.select_device(device)?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            return Ok(RocmStream {
                stream: self.libraries.create_stream(device)?,
                device,
                runtime_identity: self.runtime_identity(),
                _runtime: self.clone(),
            });
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    pub fn copy_host_to_device(
        &self,
        stream: &RocmStream,
        destination: &RocmAllocation,
        destination_offset: usize,
        source: &[u8],
    ) -> Result<(), RocmExecutionError> {
        self.validate_stream_allocation(stream, destination, destination_offset, source.len())?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.select_device(stream.device)?;
            return self.libraries.copy_host_to_device(
                &stream.stream,
                &destination.allocation,
                destination_offset,
                source,
            );
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    pub fn copy_device_to_host(
        &self,
        stream: &RocmStream,
        destination: &mut [u8],
        source: &RocmAllocation,
        source_offset: usize,
    ) -> Result<(), RocmExecutionError> {
        self.validate_stream_allocation(stream, source, source_offset, destination.len())?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.select_device(stream.device)?;
            return self.libraries.copy_device_to_host(
                &stream.stream,
                destination,
                &source.allocation,
                source_offset,
            );
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    pub fn copy_device_to_device(
        &self,
        stream: &RocmStream,
        destination: &RocmAllocation,
        destination_offset: usize,
        source: &RocmAllocation,
        source_offset: usize,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        self.validate_stream_allocation(stream, destination, destination_offset, byte_length)?;
        self.validate_stream_allocation(stream, source, source_offset, byte_length)?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.select_device(stream.device)?;
            return self.libraries.copy_device_to_device(
                &stream.stream,
                &destination.allocation,
                destination_offset,
                &source.allocation,
                source_offset,
                byte_length,
            );
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    pub fn memset(
        &self,
        stream: &RocmStream,
        allocation: &RocmAllocation,
        offset: usize,
        value: u8,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        self.validate_stream_allocation(stream, allocation, offset, byte_length)?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.select_device(stream.device)?;
            return self.libraries.memset(
                &stream.stream,
                &allocation.allocation,
                offset,
                value,
                byte_length,
            );
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = value;
            Err(unsupported_execution_target())
        }
    }

    pub fn record_event(&self, stream: &RocmStream) -> Result<RocmEvent, RocmExecutionError> {
        self.validate_runtime(stream.runtime_identity, "stream")?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.select_device(stream.device)?;
            return Ok(RocmEvent {
                event: self.libraries.record_event(&stream.stream)?,
                device: stream.device,
                runtime_identity: self.runtime_identity(),
                _runtime: self.clone(),
            });
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    /// Computes column-major `output = left * right` for dimensions `[rows, columns, inner]`.
    pub fn sgemm_f32(
        &self,
        stream: &RocmStream,
        dimensions: [usize; 3],
        left: &RocmAllocation,
        right: &RocmAllocation,
        output: &RocmAllocation,
    ) -> Result<(), RocmExecutionError> {
        let [rows, columns, inner] = dimensions;
        let left_bytes = rows
            .checked_mul(inner)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| invalid_execution_argument("SGEMM left size overflow"))?;
        let right_bytes = inner
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| invalid_execution_argument("SGEMM right size overflow"))?;
        let output_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| invalid_execution_argument("SGEMM output size overflow"))?;
        self.validate_stream_allocation(stream, left, 0, left_bytes)?;
        self.validate_stream_allocation(stream, right, 0, right_bytes)?;
        self.validate_stream_allocation(stream, output, 0, output_bytes)?;
        if rows == 0 || columns == 0 || inner == 0 {
            return Ok(());
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.select_device(stream.device)?;
            return self.libraries.sgemm_f32(
                &stream.stream,
                dimensions,
                &left.allocation,
                &right.allocation,
                &output.allocation,
            );
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            Err(unsupported_execution_target())
        }
    }

    fn validate_stream_allocation(
        &self,
        stream: &RocmStream,
        allocation: &RocmAllocation,
        offset: usize,
        byte_length: usize,
    ) -> Result<(), RocmExecutionError> {
        self.validate_runtime(stream.runtime_identity, "stream")?;
        self.validate_runtime(allocation.runtime_identity, "allocation")?;
        if stream.device != allocation.device {
            return Err(invalid_execution_argument(
                "stream and allocation devices differ",
            ));
        }
        let end = offset
            .checked_add(byte_length)
            .ok_or_else(|| invalid_execution_argument("allocation range overflow"))?;
        if end > allocation.byte_length {
            return Err(invalid_execution_argument(
                "allocation range exceeds its byte length",
            ));
        }
        Ok(())
    }

    pub fn synchronize_event(&self, event: &RocmEvent) -> Result<(), RocmExecutionError> {
        self.validate_runtime(event.runtime_identity, "event")?;
        event.synchronize()
    }

    fn validate_runtime(
        &self,
        resource_identity: usize,
        resource: &str,
    ) -> Result<(), RocmExecutionError> {
        if resource_identity != self.runtime_identity() {
            Err(invalid_execution_argument(&format!(
                "{resource} belongs to a different certified ROCm runtime"
            )))
        } else {
            Ok(())
        }
    }
}

fn invalid_execution_argument(reason: &str) -> RocmExecutionError {
    RocmExecutionError::InvalidArgument {
        reason: reason.to_owned(),
    }
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn classify_hip_status(
    operation: &'static str,
    device: u32,
    allocation_bytes: usize,
    status: i32,
    message: String,
) -> Result<(), RocmExecutionError> {
    match status {
        abi::HIP_SUCCESS => Ok(()),
        abi::HIP_ERROR_OUT_OF_MEMORY => Err(RocmExecutionError::OutOfMemory {
            device,
            bytes: allocation_bytes,
            message,
        }),
        abi::HIP_ERROR_INVALID_CONTEXT
        | abi::HIP_ERROR_ILLEGAL_ADDRESS
        | abi::HIP_ERROR_CONTEXT_IS_DESTROYED
        | abi::HIP_ERROR_LAUNCH_FAILURE => Err(RocmExecutionError::DeviceLost {
            device,
            operation,
            message,
        }),
        _ => Err(RocmExecutionError::Status {
            operation,
            status,
            message,
        }),
    }
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn classify_rocblas_status(
    operation: &'static str,
    device: u32,
    requested_bytes: usize,
    status: i32,
) -> Result<(), RocmExecutionError> {
    match status {
        abi::ROCBLAS_STATUS_SUCCESS => Ok(()),
        abi::ROCBLAS_STATUS_MEMORY_ERROR => Err(RocmExecutionError::OutOfMemory {
            device,
            bytes: requested_bytes,
            message: format!("{operation} returned rocBLAS memory_error"),
        }),
        _ => Err(RocmExecutionError::Status {
            operation,
            status,
            message: format!("rocBLAS status {status}"),
        }),
    }
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn assemble_rocm_device_properties(
    name: String,
    total_memory_bytes: u64,
    major: u32,
    minor: u32,
) -> Result<RocmDeviceProperties, RocmExecutionError> {
    if name.is_empty() || name.len() > 255 || name.contains('\0') {
        return Err(RocmExecutionError::Status {
            operation: "hipDeviceGetName",
            status: -1,
            message: "HIP device name must contain 1..=255 non-NUL UTF-8 bytes".to_owned(),
        });
    }
    if total_memory_bytes == 0 {
        return Err(RocmExecutionError::Status {
            operation: "hipDeviceTotalMem",
            status: -1,
            message: "HIP returned zero total device memory".to_owned(),
        });
    }
    Ok(RocmDeviceProperties {
        name,
        total_memory_bytes,
        major,
        minor,
        architecture: Some(format!("hip-compute-{major}.{minor}")),
        // This is an execution-capability fact, not a guess from a marketing device name. The
        // current native adapter advertises and implements only F32 kernels.
        has_fp16: false,
    })
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn hip_version(raw: i32) -> abi::RocmVersion {
    let raw = u32::try_from(raw).unwrap_or_default();
    abi::RocmVersion::new(raw / 10_000_000, (raw / 100_000) % 100, raw % 100_000)
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn version_string(version: abi::RocmVersion) -> String {
    format!("{}.{}.{}", version.major, version.minor, version.patch)
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn check_probe_status(
    function: &'static str,
    status: i32,
    reviewed_success: i32,
) -> Result<(), RocmLoadError> {
    if status == reviewed_success {
        Ok(())
    } else {
        Err(RocmLoadError::Probe { function, status })
    }
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn enforce_hip_floor(component: &'static str, raw: i32) -> Result<String, RocmLoadError> {
    let found = hip_version(raw);
    if raw < abi::HIP_RUNTIME_FLOOR {
        Err(RocmLoadError::VersionTooOld {
            component,
            found: version_string(found),
            minimum: version_string(abi::ABI_FLOOR),
        })
    } else {
        Ok(version_string(found))
    }
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn leading_version(value: &str) -> Option<abi::RocmVersion> {
    let mut parts = value
        .trim_start_matches(|character: char| !character.is_ascii_digit())
        .split(|character: char| !character.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().and_then(|part| part.parse().ok()).unwrap_or(0);
    Some(abi::RocmVersion::new(major, minor, patch))
}

#[cfg(any(all(target_os = "linux", target_arch = "x86_64"), test))]
fn miopen_meets_floor(major: usize, minor: usize) -> bool {
    (major, minor) >= (3, 1)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux {
    use super::*;
    use std::{
        ffi::{CStr, CString, c_void},
        os::unix::ffi::OsStrExt,
        ptr::NonNull,
    };

    struct DynamicLibrary {
        handle: NonNull<c_void>,
        name: String,
        path: PathBuf,
    }

    type LoaderNamespace = libc::c_long;

    impl DynamicLibrary {
        fn open_in_namespace(
            name: &str,
            path: &Path,
            namespace: Option<LoaderNamespace>,
        ) -> Result<(Self, LoaderNamespace), RocmLoadError> {
            let path_bytes = path.as_os_str().as_bytes();
            let path_string = CString::new(path_bytes).map_err(|_| RocmLoadError::LibraryLoad {
                library: primary_error_name(name),
                path: path.display().to_string(),
                reason: "path contains a NUL byte".to_owned(),
            })?;
            #[cfg(target_env = "gnu")]
            // SAFETY: the path is a NUL-terminated retained descriptor path. A new isolated
            // namespace is created exactly once, then reused for every graph-ordered object.
            let handle = unsafe {
                libc::dlmopen(
                    namespace.unwrap_or(libc::LM_ID_NEWLM),
                    path_string.as_ptr(),
                    libc::RTLD_NOW | libc::RTLD_LOCAL,
                )
            };
            #[cfg(not(target_env = "gnu"))]
            let handle = {
                let _ = &path_string;
                std::ptr::null_mut()
            };
            let handle = NonNull::new(handle).ok_or_else(|| RocmLoadError::LibraryLoad {
                library: primary_error_name(name),
                path: path.display().to_string(),
                reason: if cfg!(target_env = "gnu") {
                    dlerror_string()
                } else {
                    "isolated dlmopen namespaces require the reviewed GNU target".to_owned()
                },
            })?;
            #[cfg(target_env = "gnu")]
            let mut actual_namespace = 0;
            #[cfg(not(target_env = "gnu"))]
            let actual_namespace = 0;
            #[cfg(target_env = "gnu")]
            // SAFETY: RTLD_DI_LMID writes one Lmid_t for this live handle.
            let namespace_status = unsafe {
                libc::dlinfo(
                    handle.as_ptr(),
                    libc::RTLD_DI_LMID,
                    std::ptr::addr_of_mut!(actual_namespace).cast(),
                )
            };
            #[cfg(not(target_env = "gnu"))]
            let namespace_status = -1;
            if namespace_status != 0
                || namespace.is_some_and(|expected| expected != actual_namespace)
            {
                // SAFETY: this error branch still uniquely owns the live loader handle.
                unsafe { libc::dlclose(handle.as_ptr()) };
                return Err(RocmLoadError::BindingProof {
                    reason: format!("could not prove isolated namespace for {name}"),
                });
            }
            Ok((
                Self {
                    handle,
                    name: name.to_owned(),
                    path: path.to_owned(),
                },
                actual_namespace,
            ))
        }

        fn symbol(&self, symbol: &str) -> Result<*mut c_void, RocmLoadError> {
            let symbol_string = CString::new(symbol).map_err(|_| RocmLoadError::MissingSymbol {
                library: primary_error_name(&self.name),
                symbol: symbol.to_owned(),
            })?;
            // SAFETY: the handle is live and the symbol name is a NUL-terminated C string.
            let pointer = unsafe { libc::dlsym(self.handle.as_ptr(), symbol_string.as_ptr()) };
            if pointer.is_null() {
                return Err(RocmLoadError::MissingSymbol {
                    library: primary_error_name(&self.name),
                    symbol: symbol.to_owned(),
                });
            }
            // SAFETY: dladdr initializes the plain Dl_info output for a live symbol address.
            let mut information = unsafe { std::mem::zeroed::<libc::Dl_info>() };
            // SAFETY: pointer is a live address returned by dlsym and information is writable.
            let status = unsafe { libc::dladdr(pointer.cast_const(), &mut information) };
            if status == 0 || information.dli_fname.is_null() {
                return Err(RocmLoadError::BindingProof {
                    reason: format!(
                        "loader could not prove the defining object for {}:{symbol}",
                        self.name
                    ),
                });
            }
            // SAFETY: successful dladdr returned a non-null loader-owned C string.
            let defining_path = unsafe { CStr::from_ptr(information.dli_fname) }.to_bytes();
            if defining_path != self.path.as_os_str().as_bytes() {
                return Err(RocmLoadError::BindingProof {
                    reason: format!(
                        "{}:{symbol} resolved from {} instead of certified object {}",
                        self.name,
                        String::from_utf8_lossy(defining_path),
                        self.path.display()
                    ),
                });
            }
            Ok(pointer)
        }
    }

    fn primary_error_name(name: &str) -> &'static str {
        abi::LIBRARIES
            .iter()
            .find(|(library, _)| *library == name)
            .map_or("vendor dependency", |(library, _)| *library)
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            // SAFETY: this object uniquely owns the live dlopen handle and all function pointers
            // are fields dropped before their backing library fields.
            unsafe {
                libc::dlclose(self.handle.as_ptr());
            }
        }
    }

    // SAFETY: a loaded library handle is immutable after construction; the ROCm C APIs provide
    // their own thread-safety contract and no Rust reference aliases vendor-owned memory.
    unsafe impl Send for DynamicLibrary {}
    // SAFETY: symbol resolution and immutable handle retention do not mutate Rust-owned state.
    unsafe impl Sync for DynamicLibrary {}

    fn dlerror_string() -> String {
        // SAFETY: dlerror returns either null or a NUL-terminated thread-local diagnostic.
        let error = unsafe { libc::dlerror() };
        if error.is_null() {
            "dynamic loader returned no diagnostic".to_owned()
        } else {
            // SAFETY: the non-null pointer returned by dlerror is a valid C string until the next
            // loader operation on this thread.
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        }
    }

    #[repr(C)]
    struct LinkMap {
        address: usize,
        name: *const libc::c_char,
        dynamic: *mut c_void,
        next: *mut LinkMap,
        previous: *mut LinkMap,
    }

    #[repr(C)]
    struct ElfDynamic {
        tag: i64,
        value: u64,
    }

    const TRUSTED_SYSTEM_SONAMES: &[&str] = &[
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libdl.so.2",
        "libgcc_s.so.1",
        "libm.so.6",
        "libpthread.so.0",
        "librt.so.1",
        "libstdc++.so.6",
    ];

    fn handle_link_map(library: &DynamicLibrary) -> Result<*mut LinkMap, RocmLoadError> {
        let mut link_map: *mut LinkMap = std::ptr::null_mut();
        // SAFETY: RTLD_DI_LINKMAP writes one live link_map pointer for this retained handle.
        let status = unsafe {
            libc::dlinfo(
                library.handle.as_ptr(),
                libc::RTLD_DI_LINKMAP,
                std::ptr::addr_of_mut!(link_map).cast(),
            )
        };
        if status != 0 || link_map.is_null() {
            Err(RocmLoadError::BindingProof {
                reason: format!("loader did not expose a link map for {}", library.name),
            })
        } else {
            Ok(link_map)
        }
    }

    unsafe fn bounded_dynamic_string(
        string_table: *const u8,
        offset: u64,
    ) -> Result<String, RocmLoadError> {
        let offset = usize::try_from(offset).map_err(|_| RocmLoadError::BindingProof {
            reason: "ELF dynamic string offset exceeds address space".to_owned(),
        })?;
        // SAFETY: certification validated that the loader-consumed dynamic string table and every
        // referenced offset are within a mapped PT_LOAD segment. The scan remains strictly bounded.
        let start = unsafe { string_table.add(offset) };
        let mut length = 0;
        while length <= 255 {
            // SAFETY: the certified dynamic string lies in the mapped, retained ELF image.
            if unsafe { *start.add(length) } == 0 {
                // SAFETY: the preceding bounded scan proved all bytes through the terminator live.
                let bytes = unsafe { std::slice::from_raw_parts(start, length) };
                return std::str::from_utf8(bytes).map(str::to_owned).map_err(|_| {
                    RocmLoadError::BindingProof {
                        reason: "ELF dynamic string is not UTF-8".to_owned(),
                    }
                });
            }
            length += 1;
        }
        Err(RocmLoadError::BindingProof {
            reason: "ELF dynamic string exceeds 255 bytes".to_owned(),
        })
    }

    unsafe fn dynamic_identity(
        link_map: *mut LinkMap,
    ) -> Result<(String, BTreeSet<String>), RocmLoadError> {
        // SAFETY: the pointer is from dlinfo for a retained loader handle.
        let dynamic = unsafe { (*link_map).dynamic.cast::<ElfDynamic>() };
        if dynamic.is_null() {
            return Err(RocmLoadError::BindingProof {
                reason: "loaded vendor object has no dynamic table".to_owned(),
            });
        }
        let mut string_table = std::ptr::null();
        let mut soname_offset = None;
        let mut needed_offsets = Vec::new();
        let mut terminated = false;
        for index in 0..65_536 {
            // SAFETY: certification bounded the sole PT_DYNAMIC segment and required DT_NULL.
            let entry = unsafe { &*dynamic.add(index) };
            match entry.tag {
                0 => {
                    terminated = true;
                    break;
                }
                1 => needed_offsets.push(entry.value),
                5 => string_table = entry.value as *const u8,
                14 => soname_offset = Some(entry.value),
                _ => {}
            }
        }
        if !terminated || string_table.is_null() {
            return Err(RocmLoadError::BindingProof {
                reason: "loaded vendor object has an invalid dynamic string table".to_owned(),
            });
        }
        let soname_offset = soname_offset.ok_or_else(|| RocmLoadError::BindingProof {
            reason: "loaded vendor object has no DT_SONAME".to_owned(),
        })?;
        // SAFETY: offsets and table provenance are established above and by certification.
        let soname = unsafe { bounded_dynamic_string(string_table, soname_offset) }?;
        let mut needed = BTreeSet::new();
        for offset in needed_offsets {
            // SAFETY: same certified dynamic string-table invariant.
            let dependency = unsafe { bounded_dynamic_string(string_table, offset) }?;
            if !needed.insert(dependency) {
                return Err(RocmLoadError::BindingProof {
                    reason: format!("{soname} repeats a DT_NEEDED entry"),
                });
            }
        }
        Ok((soname, needed))
    }

    fn prove_exact_bindings(
        libraries: &[DynamicLibrary],
        paths: &BTreeMap<String, PathBuf>,
        sonames: &BTreeMap<String, String>,
        dependencies: &[RocmDependencyCandidate],
        dependency_edges: &[RocmDependencyEdge],
    ) -> Result<(), RocmLoadError> {
        let mut unique_sonames = BTreeSet::new();
        if sonames
            .values()
            .any(|soname| !unique_sonames.insert(soname))
        {
            return Err(RocmLoadError::BindingProof {
                reason: "certified ROCm objects must have unique SONAMEs".to_owned(),
            });
        }
        for dependency in dependencies {
            if sonames.get(dependency.library_id()) != Some(&dependency.soname) {
                return Err(RocmLoadError::BindingProof {
                    reason: format!(
                        "dependency {} differs from the certified SONAME registry",
                        dependency.library_id
                    ),
                });
            }
        }
        let expected_paths = paths
            .iter()
            .map(|(id, path)| (path.as_os_str().as_bytes().to_vec(), id.as_str()))
            .collect::<BTreeMap<_, _>>();
        let expected_by_soname = sonames
            .iter()
            .map(|(id, soname)| (soname.as_str(), id.as_str()))
            .collect::<BTreeMap<_, _>>();
        let expected_edges = dependency_edges
            .iter()
            .map(|edge| {
                let dependency_soname =
                    sonames
                        .get(edge.dependency())
                        .ok_or_else(|| RocmLoadError::BindingProof {
                            reason: format!(
                                "declared edge {} -> {} has no registered dependency SONAME",
                                edge.consumer(),
                                edge.dependency()
                            ),
                        })?;
                Ok((edge.consumer(), dependency_soname.as_str()))
            })
            .collect::<Result<BTreeSet<_>, RocmLoadError>>()?;

        let mut maps_by_id = BTreeMap::new();
        for library in libraries {
            let expected = paths
                .get(&library.name)
                .ok_or_else(|| RocmLoadError::BindingProof {
                    reason: format!("loaded unregistered ROCm object {}", library.name),
                })?;
            let link_map = handle_link_map(library)?;
            // SAFETY: dlinfo returned a non-null link_map for the live handle. l_name remains
            // valid for the handle lifetime and is checked for null before conversion.
            let actual_name = unsafe { (*link_map).name };
            if actual_name.is_null() {
                return Err(RocmLoadError::BindingProof {
                    reason: format!("loader returned no object path for {}", library.name),
                });
            }
            // SAFETY: non-null l_name is a loader-owned NUL-terminated byte string.
            let actual = unsafe { CStr::from_ptr(actual_name) }.to_bytes();
            if actual != expected.as_os_str().as_bytes() {
                return Err(RocmLoadError::BindingProof {
                    reason: format!(
                        "{} resolved to {} instead of retained descriptor {}",
                        library.name,
                        String::from_utf8_lossy(actual),
                        expected.display()
                    ),
                });
            }
            if maps_by_id.insert(library.name.as_str(), link_map).is_some() {
                return Err(RocmLoadError::BindingProof {
                    reason: format!("loader returned duplicate handle for {}", library.name),
                });
            }
        }

        let mut head = *maps_by_id
            .values()
            .next()
            .ok_or_else(|| RocmLoadError::BindingProof {
                reason: "certified ROCm library set is empty".to_owned(),
            })?;
        let mut walked = BTreeSet::new();
        while !head.is_null() {
            if !walked.insert(head as usize) || walked.len() > 4_096 {
                return Err(RocmLoadError::BindingProof {
                    reason: "loader namespace link map is cyclic or exceeds 4096 objects"
                        .to_owned(),
                });
            }
            // SAFETY: each previous pointer belongs to the retained namespace link-map chain.
            let previous = unsafe { (*head).previous };
            if previous.is_null() {
                break;
            }
            head = previous;
        }

        walked.clear();
        let mut observed_explicit = BTreeSet::new();
        let mut observed_system = BTreeSet::new();
        let mut current = head;
        while !current.is_null() {
            if !walked.insert(current as usize) || walked.len() > 4_096 {
                return Err(RocmLoadError::BindingProof {
                    reason: "loader namespace link map is cyclic or exceeds 4096 objects"
                        .to_owned(),
                });
            }
            // SAFETY: current belongs to the retained namespace chain.
            let name_pointer = unsafe { (*current).name };
            if name_pointer.is_null() {
                return Err(RocmLoadError::BindingProof {
                    reason: "loader namespace contains an unnamed object".to_owned(),
                });
            }
            // SAFETY: l_name is a loader-owned C string for the namespace lifetime.
            let loaded_path = unsafe { CStr::from_ptr(name_pointer) }.to_bytes();
            if let Some(id) = expected_paths.get(loaded_path) {
                if !observed_explicit.insert(*id) {
                    return Err(RocmLoadError::BindingProof {
                        reason: format!("certified object {id} appears more than once"),
                    });
                }
            } else {
                let basename = loaded_path
                    .rsplit(|byte| *byte == b'/')
                    .next()
                    .and_then(|bytes| std::str::from_utf8(bytes).ok())
                    .unwrap_or_default();
                if expected_by_soname.contains_key(basename) {
                    return Err(RocmLoadError::BindingProof {
                        reason: format!(
                            "ambient object {} duplicates a certified vendor SONAME",
                            String::from_utf8_lossy(loaded_path)
                        ),
                    });
                }
                if !TRUSTED_SYSTEM_SONAMES.contains(&basename) {
                    return Err(RocmLoadError::BindingProof {
                        reason: format!(
                            "loader namespace contains undeclared object {}",
                            String::from_utf8_lossy(loaded_path)
                        ),
                    });
                }
                observed_system.insert(basename);
            }
            // SAFETY: current belongs to the retained namespace chain.
            current = unsafe { (*current).next };
        }
        if observed_explicit.len() != paths.len() {
            return Err(RocmLoadError::BindingProof {
                reason: "loader namespace omits a certified vendor object".to_owned(),
            });
        }

        for (id, link_map) in maps_by_id {
            // SAFETY: each map was returned for a retained explicit handle and certification
            // validated the mapped dynamic metadata before loading.
            let (actual_soname, needed) = unsafe { dynamic_identity(link_map) }?;
            let expected_soname = sonames.get(id).ok_or_else(|| RocmLoadError::BindingProof {
                reason: format!("loaded object {id} has no certified SONAME"),
            })?;
            if &actual_soname != expected_soname {
                return Err(RocmLoadError::BindingProof {
                    reason: format!(
                        "loaded object {id} reports SONAME {actual_soname} instead of {expected_soname}"
                    ),
                });
            }
            let mut actual_vendor_edges = BTreeSet::new();
            for needed_soname in needed {
                if expected_by_soname.contains_key(needed_soname.as_str()) {
                    actual_vendor_edges.insert((id, needed_soname.as_str().to_owned()));
                } else if !TRUSTED_SYSTEM_SONAMES.contains(&needed_soname.as_str()) {
                    return Err(RocmLoadError::BindingProof {
                        reason: format!(
                            "{id} requires undeclared non-system SONAME {needed_soname}"
                        ),
                    });
                }
            }
            let expected_vendor_edges = expected_edges
                .iter()
                .filter(|(consumer, _)| *consumer == id)
                .map(|(_, dependency)| (id, (*dependency).to_owned()))
                .collect::<BTreeSet<_>>();
            if actual_vendor_edges != expected_vendor_edges {
                return Err(RocmLoadError::BindingProof {
                    reason: format!("loaded DT_NEEDED edges differ for {id}"),
                });
            }
        }
        Ok(())
    }

    pub(super) struct LoadedLibraries {
        init: abi::HipInit,
        driver_version: abi::HipDriverGetVersion,
        runtime_version: abi::HipRuntimeGetVersion,
        device_count: abi::HipGetDeviceCount,
        device_name: abi::HipDeviceGetName,
        device_total_memory: abi::HipDeviceTotalMem,
        device_attribute: abi::HipDeviceGetAttribute,
        hiprtc_version: abi::HipRtcVersion,
        rocblas_version_size: abi::RocblasGetVersionStringSize,
        rocblas_version: abi::RocblasGetVersionString,
        miopen_version: abi::MiopenGetVersion,
        set_device: abi::HipSetDevice,
        malloc: abi::HipMalloc,
        free: abi::HipFree,
        memcpy_async: abi::HipMemcpyAsync,
        memset_async: abi::HipMemsetAsync,
        stream_create: abi::HipStreamCreateWithFlags,
        stream_destroy: abi::HipStreamDestroy,
        stream_synchronize: abi::HipStreamSynchronize,
        event_create: abi::HipEventCreateWithFlags,
        event_record: abi::HipEventRecord,
        event_synchronize: abi::HipEventSynchronize,
        event_destroy: abi::HipEventDestroy,
        get_error_string: abi::HipGetErrorString,
        rocblas_create: abi::RocblasCreateHandle,
        rocblas_destroy: abi::RocblasDestroyHandle,
        rocblas_set_stream: abi::RocblasSetStream,
        rocblas_sgemm: abi::RocblasSgemm,
        libraries: Vec<DynamicLibrary>,
    }

    impl LoadedLibraries {
        pub(super) fn load(
            paths: &BTreeMap<String, PathBuf>,
            sonames: &BTreeMap<String, String>,
            load_order: &[String],
            dependencies: &[RocmDependencyCandidate],
            dependency_edges: &[RocmDependencyEdge],
        ) -> Result<Self, RocmLoadError> {
            if paths.len() != load_order.len() || sonames.len() != load_order.len() {
                return Err(RocmLoadError::BindingProof {
                    reason: "load order, candidate paths, and SONAME registry differ".to_owned(),
                });
            }
            let mut libraries = Vec::with_capacity(load_order.len());
            let mut namespace = None;
            for name in load_order {
                let path = paths.get(name).ok_or_else(|| RocmLoadError::BindingProof {
                    reason: format!("load order references unknown candidate {name}"),
                })?;
                let (library, loaded_namespace) =
                    DynamicLibrary::open_in_namespace(name, path, namespace)?;
                namespace = Some(loaded_namespace);
                libraries.push(library);
            }
            prove_exact_bindings(&libraries, paths, sonames, dependencies, dependency_edges)?;
            for contract in abi::SYMBOLS {
                let library = libraries
                    .iter()
                    .find(|library| library.name == contract.library)
                    .ok_or(RocmLoadError::MissingLibrary {
                        library: contract.library,
                        searched: "checked ABI manifest library set".to_owned(),
                    })?;
                library.symbol(contract.name)?;
            }

            let symbol = |library_name: &'static str, name: &str| {
                libraries
                    .iter()
                    .find(|library| library.name == library_name)
                    .ok_or(RocmLoadError::MissingLibrary {
                        library: library_name,
                        searched: "loaded ABI library set".to_owned(),
                    })?
                    .symbol(name)
            };
            let init_pointer = symbol("libamdhip64", "hipInit")?;
            let driver_version_pointer = symbol("libamdhip64", "hipDriverGetVersion")?;
            let runtime_version_pointer = symbol("libamdhip64", "hipRuntimeGetVersion")?;
            let device_count_pointer = symbol("libamdhip64", "hipGetDeviceCount")?;
            let device_name_pointer = symbol("libamdhip64", "hipDeviceGetName")?;
            let device_total_memory_pointer = symbol("libamdhip64", "hipDeviceTotalMem")?;
            let device_attribute_pointer = symbol("libamdhip64", "hipDeviceGetAttribute")?;
            let hiprtc_version_pointer = symbol("libhiprtc", "hiprtcVersion")?;
            let rocblas_size_pointer = symbol("librocblas", "rocblas_get_version_string_size")?;
            let rocblas_version_pointer = symbol("librocblas", "rocblas_get_version_string")?;
            let miopen_version_pointer = symbol("libMIOpen", "miopenGetVersion")?;
            let set_device_pointer = symbol("libamdhip64", "hipSetDevice")?;
            let malloc_pointer = symbol("libamdhip64", "hipMalloc")?;
            let free_pointer = symbol("libamdhip64", "hipFree")?;
            let memcpy_async_pointer = symbol("libamdhip64", "hipMemcpyAsync")?;
            let memset_async_pointer = symbol("libamdhip64", "hipMemsetAsync")?;
            let stream_create_pointer = symbol("libamdhip64", "hipStreamCreateWithFlags")?;
            let stream_destroy_pointer = symbol("libamdhip64", "hipStreamDestroy")?;
            let stream_synchronize_pointer = symbol("libamdhip64", "hipStreamSynchronize")?;
            let event_create_pointer = symbol("libamdhip64", "hipEventCreateWithFlags")?;
            let event_record_pointer = symbol("libamdhip64", "hipEventRecord")?;
            let event_synchronize_pointer = symbol("libamdhip64", "hipEventSynchronize")?;
            let event_destroy_pointer = symbol("libamdhip64", "hipEventDestroy")?;
            let get_error_string_pointer = symbol("libamdhip64", "hipGetErrorString")?;
            let rocblas_create_pointer = symbol("librocblas", "rocblas_create_handle")?;
            let rocblas_destroy_pointer = symbol("librocblas", "rocblas_destroy_handle")?;
            let rocblas_set_stream_pointer = symbol("librocblas", "rocblas_set_stream")?;
            let rocblas_sgemm_pointer = symbol("librocblas", "rocblas_sgemm")?;

            // SAFETY: every address was resolved from the exact library/name pair and its C
            // signature is checked against the embedded reviewed ABI manifest before this point.
            let runtime_version = unsafe {
                std::mem::transmute::<*mut c_void, abi::HipRuntimeGetVersion>(
                    runtime_version_pointer,
                )
            };
            // SAFETY: same checked symbol/signature invariant as above.
            let device_count = unsafe {
                std::mem::transmute::<*mut c_void, abi::HipGetDeviceCount>(device_count_pointer)
            };
            // SAFETY: same checked symbol/signature invariant as above.
            let hiprtc_version = unsafe {
                std::mem::transmute::<*mut c_void, abi::HipRtcVersion>(hiprtc_version_pointer)
            };
            // SAFETY: same checked symbol/signature invariant as above.
            let rocblas_version_size = unsafe {
                std::mem::transmute::<*mut c_void, abi::RocblasGetVersionStringSize>(
                    rocblas_size_pointer,
                )
            };
            // SAFETY: same checked symbol/signature invariant as above.
            let rocblas_version = unsafe {
                std::mem::transmute::<*mut c_void, abi::RocblasGetVersionString>(
                    rocblas_version_pointer,
                )
            };
            // SAFETY: same checked symbol/signature invariant as above.
            let miopen_version = unsafe {
                std::mem::transmute::<*mut c_void, abi::MiopenGetVersion>(miopen_version_pointer)
            };
            macro_rules! checked_function {
                ($pointer:ident, $function_type:ty) => {{
                    // SAFETY: the address was resolved from the manifest-checked library/name
                    // pair and the destination is its reviewed generated C ABI function type.
                    unsafe { std::mem::transmute::<*mut c_void, $function_type>($pointer) }
                }};
            }
            Ok(Self {
                init: checked_function!(init_pointer, abi::HipInit),
                driver_version: checked_function!(driver_version_pointer, abi::HipDriverGetVersion),
                runtime_version,
                device_count,
                device_name: checked_function!(device_name_pointer, abi::HipDeviceGetName),
                device_total_memory: checked_function!(
                    device_total_memory_pointer,
                    abi::HipDeviceTotalMem
                ),
                device_attribute: checked_function!(
                    device_attribute_pointer,
                    abi::HipDeviceGetAttribute
                ),
                hiprtc_version,
                rocblas_version_size,
                rocblas_version,
                miopen_version,
                set_device: checked_function!(set_device_pointer, abi::HipSetDevice),
                malloc: checked_function!(malloc_pointer, abi::HipMalloc),
                free: checked_function!(free_pointer, abi::HipFree),
                memcpy_async: checked_function!(memcpy_async_pointer, abi::HipMemcpyAsync),
                memset_async: checked_function!(memset_async_pointer, abi::HipMemsetAsync),
                stream_create: checked_function!(
                    stream_create_pointer,
                    abi::HipStreamCreateWithFlags
                ),
                stream_destroy: checked_function!(stream_destroy_pointer, abi::HipStreamDestroy),
                stream_synchronize: checked_function!(
                    stream_synchronize_pointer,
                    abi::HipStreamSynchronize
                ),
                event_create: checked_function!(event_create_pointer, abi::HipEventCreateWithFlags),
                event_record: checked_function!(event_record_pointer, abi::HipEventRecord),
                event_synchronize: checked_function!(
                    event_synchronize_pointer,
                    abi::HipEventSynchronize
                ),
                event_destroy: checked_function!(event_destroy_pointer, abi::HipEventDestroy),
                get_error_string: checked_function!(
                    get_error_string_pointer,
                    abi::HipGetErrorString
                ),
                rocblas_create: checked_function!(rocblas_create_pointer, abi::RocblasCreateHandle),
                rocblas_destroy: checked_function!(
                    rocblas_destroy_pointer,
                    abi::RocblasDestroyHandle
                ),
                rocblas_set_stream: checked_function!(
                    rocblas_set_stream_pointer,
                    abi::RocblasSetStream
                ),
                rocblas_sgemm: checked_function!(rocblas_sgemm_pointer, abi::RocblasSgemm),
                libraries,
            })
        }

        pub(super) fn probe(&self) -> Result<(ComponentVersions, u32), RocmLoadError> {
            // SAFETY: zero is the only reviewed HIP initialization flag.
            check_probe_status(
                "hipInit",
                unsafe { (self.init)(abi::HIP_INIT_FLAGS_ZERO) },
                abi::HIP_SUCCESS,
            )?;
            let mut driver_raw = 0;
            // SAFETY: the checked function pointer writes one initialized integer.
            check_probe_status(
                "hipDriverGetVersion",
                unsafe { (self.driver_version)(&mut driver_raw) },
                abi::HIP_SUCCESS,
            )?;
            let hip_driver = enforce_hip_floor("HIP driver", driver_raw)?;
            let mut runtime_raw = 0;
            // SAFETY: the checked function pointer writes one initialized integer.
            check_probe_status(
                "hipRuntimeGetVersion",
                unsafe { (self.runtime_version)(&mut runtime_raw) },
                abi::HIP_SUCCESS,
            )?;
            let hip_runtime = enforce_hip_floor("HIP runtime", runtime_raw)?;
            let mut count = 0;
            // SAFETY: the checked function pointer writes one initialized integer.
            check_probe_status(
                "hipGetDeviceCount",
                unsafe { (self.device_count)(&mut count) },
                abi::HIP_SUCCESS,
            )?;
            let device_count = u32::try_from(count).map_err(|_| RocmLoadError::Probe {
                function: "hipGetDeviceCount",
                status: count,
            })?;

            let mut hiprtc_major = 0;
            let mut hiprtc_minor = 0;
            // SAFETY: the checked function pointer writes two initialized integers.
            check_probe_status(
                "hiprtcVersion",
                unsafe { (self.hiprtc_version)(&mut hiprtc_major, &mut hiprtc_minor) },
                abi::HIPRTC_SUCCESS,
            )?;
            if (hiprtc_major, hiprtc_minor) < (6, 1) {
                return Err(RocmLoadError::VersionTooOld {
                    component: "hipRTC",
                    found: format!("{hiprtc_major}.{hiprtc_minor}.0"),
                    minimum: "6.1.0".to_owned(),
                });
            }

            let mut rocblas_size = 0;
            // SAFETY: the checked function pointer writes one size value.
            check_probe_status(
                "rocblas_get_version_string_size",
                unsafe { (self.rocblas_version_size)(&mut rocblas_size) },
                abi::ROCBLAS_STATUS_SUCCESS,
            )?;
            if !(2..=256).contains(&rocblas_size) {
                return Err(RocmLoadError::Probe {
                    function: "rocblas_get_version_string_size",
                    status: i32::try_from(rocblas_size).unwrap_or(i32::MAX),
                });
            }
            let mut rocblas_buffer = vec![0_i8; rocblas_size];
            // SAFETY: the buffer has exactly the length supplied to the checked C function.
            check_probe_status(
                "rocblas_get_version_string",
                unsafe {
                    (self.rocblas_version)(rocblas_buffer.as_mut_ptr(), rocblas_buffer.len())
                },
                abi::ROCBLAS_STATUS_SUCCESS,
            )?;
            let rocblas = CStr::from_bytes_until_nul(
                &rocblas_buffer
                    .iter()
                    .map(|byte| byte.to_ne_bytes()[0])
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| RocmLoadError::Probe {
                function: "rocblas_get_version_string",
                status: -1,
            })?
            .to_string_lossy()
            .into_owned();
            let rocblas_version = leading_version(&rocblas).ok_or(RocmLoadError::Probe {
                function: "rocblas_get_version_string",
                status: -1,
            })?;
            if (rocblas_version.major, rocblas_version.minor) < (4, 1) {
                return Err(RocmLoadError::VersionTooOld {
                    component: "rocBLAS",
                    found: version_string(rocblas_version),
                    minimum: "4.1.0".to_owned(),
                });
            }

            let (mut miopen_major, mut miopen_minor, mut miopen_patch) = (0, 0, 0);
            // SAFETY: the checked function pointer writes three initialized size values.
            check_probe_status(
                "miopenGetVersion",
                unsafe {
                    (self.miopen_version)(&mut miopen_major, &mut miopen_minor, &mut miopen_patch)
                },
                abi::MIOPEN_STATUS_SUCCESS,
            )?;
            if !miopen_meets_floor(miopen_major, miopen_minor) {
                return Err(RocmLoadError::VersionTooOld {
                    component: "MIOpen",
                    found: format!("{miopen_major}.{miopen_minor}.{miopen_patch}"),
                    minimum: "3.1.0".to_owned(),
                });
            }

            Ok((
                ComponentVersions {
                    hip_driver,
                    hip_runtime,
                    hiprtc: format!("{hiprtc_major}.{hiprtc_minor}.0"),
                    rocblas,
                    miopen: format!("{miopen_major}.{miopen_minor}.{miopen_patch}"),
                },
                device_count,
            ))
        }

        pub(super) fn select_device(&self, device: u32) -> Result<(), RocmExecutionError> {
            let device_ordinal = i32::try_from(device).map_err(|_| {
                invalid_execution_argument("device index exceeds HIP integer range")
            })?;
            // SAFETY: the generated ABI accepts one integer device ordinal.
            let status = unsafe { (self.set_device)(device_ordinal) };
            self.check_hip_status("hipSetDevice", device, 0, status)
        }

        pub(super) fn device_properties(
            &self,
            device: u32,
        ) -> Result<RocmDeviceProperties, RocmExecutionError> {
            let device_ordinal = i32::try_from(device).map_err(|_| {
                invalid_execution_argument("device index exceeds HIP integer range")
            })?;
            let mut name = [0_i8; 256];
            // SAFETY: the checked function receives a writable 256-byte name buffer and a valid
            // ordinal already bounded by the public runtime.
            let name_status =
                unsafe { (self.device_name)(name.as_mut_ptr(), name.len() as i32, device_ordinal) };
            self.check_hip_status("hipDeviceGetName", device, 0, name_status)?;
            let name = CStr::from_bytes_until_nul(
                &name
                    .iter()
                    .map(|byte| byte.to_ne_bytes()[0])
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| RocmExecutionError::Status {
                operation: "hipDeviceGetName",
                status: -1,
                message: "HIP returned a device name without NUL termination".to_owned(),
            })?
            .to_str()
            .map_err(|_| RocmExecutionError::Status {
                operation: "hipDeviceGetName",
                status: -1,
                message: "HIP returned a non-UTF-8 device name".to_owned(),
            })?
            .to_owned();
            let mut total_memory = 0_usize;
            // SAFETY: the checked function writes one size value for the validated ordinal.
            let memory_status =
                unsafe { (self.device_total_memory)(&mut total_memory, device_ordinal) };
            self.check_hip_status("hipDeviceTotalMem", device, 0, memory_status)?;
            let total_memory_bytes = u64::try_from(total_memory).map_err(|_| {
                invalid_execution_argument("device memory size exceeds the canonical u64 range")
            })?;
            let attribute = |attribute: i32, operation: &'static str| {
                let mut value = 0;
                // SAFETY: the output is one integer and both attribute constants are taken from
                // the reviewed HIP 6.1.2 enum declaration.
                let status =
                    unsafe { (self.device_attribute)(&mut value, attribute, device_ordinal) };
                self.check_hip_status(operation, device, 0, status)?;
                u32::try_from(value).map_err(|_| RocmExecutionError::Status {
                    operation,
                    status: value,
                    message: "HIP returned a negative compute capability component".to_owned(),
                })
            };
            let major = attribute(
                abi::HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                "hipDeviceGetAttribute(compute-major)",
            )?;
            let minor = attribute(
                abi::HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                "hipDeviceGetAttribute(compute-minor)",
            )?;

            assemble_rocm_device_properties(name, total_memory_bytes, major, minor)
        }

        pub(super) fn allocate(
            self: &Arc<Self>,
            device: u32,
            byte_length: usize,
        ) -> Result<DeviceAllocation, RocmExecutionError> {
            self.select_device(device)?;
            let allocated_byte_length = byte_length.max(1);
            let mut pointer = std::ptr::null_mut();
            // SAFETY: HIP initializes the output pointer on success; the non-zero allocation
            // length is retained with the returned sole allocation owner.
            let status = unsafe { (self.malloc)(&mut pointer, allocated_byte_length) };
            self.check_hip_status("hipMalloc", device, byte_length, status)?;
            let pointer = NonNull::new(pointer).ok_or_else(|| RocmExecutionError::Status {
                operation: "hipMalloc",
                status: -1,
                message: "HIP returned success with a null allocation".to_owned(),
            })?;
            Ok(DeviceAllocation {
                pointer,
                device,
                libraries: self.clone(),
            })
        }

        pub(super) fn create_stream(
            self: &Arc<Self>,
            device: u32,
        ) -> Result<ExecutionStream, RocmExecutionError> {
            self.select_device(device)?;
            let mut stream = std::ptr::null_mut();
            // SAFETY: HIP initializes one stream handle on success. The reviewed non-blocking
            // flag avoids implicit synchronization with the null stream.
            let status = unsafe { (self.stream_create)(&mut stream, abi::HIP_STREAM_NON_BLOCKING) };
            self.check_hip_status("hipStreamCreateWithFlags", device, 0, status)?;
            let stream = NonNull::new(stream).ok_or_else(|| RocmExecutionError::Status {
                operation: "hipStreamCreateWithFlags",
                status: -1,
                message: "HIP returned success with a null stream".to_owned(),
            })?;
            Ok(ExecutionStream {
                stream,
                device,
                libraries: self.clone(),
            })
        }

        pub(super) fn copy_host_to_device(
            &self,
            stream: &ExecutionStream,
            destination: &DeviceAllocation,
            destination_offset: usize,
            source: &[u8],
        ) -> Result<(), RocmExecutionError> {
            // SAFETY: the public boundary validated destination bounds and resource provenance;
            // this call synchronizes before the borrowed host slice can expire.
            let destination = unsafe {
                destination
                    .pointer
                    .as_ptr()
                    .cast::<u8>()
                    .add(destination_offset)
            };
            let status = unsafe {
                (self.memcpy_async)(
                    destination.cast(),
                    source.as_ptr().cast(),
                    source.len(),
                    abi::HIP_MEMCPY_HOST_TO_DEVICE,
                    stream.stream.as_ptr(),
                )
            };
            self.check_hip_status("hipMemcpyAsync(host-to-device)", stream.device, 0, status)?;
            stream.synchronize()
        }

        pub(super) fn copy_device_to_host(
            &self,
            stream: &ExecutionStream,
            destination: &mut [u8],
            source: &DeviceAllocation,
            source_offset: usize,
        ) -> Result<(), RocmExecutionError> {
            // SAFETY: the public boundary validated source bounds and resource provenance; this
            // call synchronizes before the borrowed host slice can expire.
            let source = unsafe { source.pointer.as_ptr().cast::<u8>().add(source_offset) };
            let status = unsafe {
                (self.memcpy_async)(
                    destination.as_mut_ptr().cast(),
                    source.cast(),
                    destination.len(),
                    abi::HIP_MEMCPY_DEVICE_TO_HOST,
                    stream.stream.as_ptr(),
                )
            };
            self.check_hip_status("hipMemcpyAsync(device-to-host)", stream.device, 0, status)?;
            stream.synchronize()
        }

        pub(super) fn copy_device_to_device(
            &self,
            stream: &ExecutionStream,
            destination: &DeviceAllocation,
            destination_offset: usize,
            source: &DeviceAllocation,
            source_offset: usize,
            byte_length: usize,
        ) -> Result<(), RocmExecutionError> {
            // SAFETY: both ranges and provenance were validated by the public boundary.
            let destination = unsafe {
                destination
                    .pointer
                    .as_ptr()
                    .cast::<u8>()
                    .add(destination_offset)
            };
            // SAFETY: same validated allocation-range invariant.
            let source = unsafe { source.pointer.as_ptr().cast::<u8>().add(source_offset) };
            let status = unsafe {
                (self.memcpy_async)(
                    destination.cast(),
                    source.cast(),
                    byte_length,
                    abi::HIP_MEMCPY_DEVICE_TO_DEVICE,
                    stream.stream.as_ptr(),
                )
            };
            self.check_hip_status("hipMemcpyAsync(device-to-device)", stream.device, 0, status)
        }

        pub(super) fn memset(
            &self,
            stream: &ExecutionStream,
            allocation: &DeviceAllocation,
            offset: usize,
            value: u8,
            byte_length: usize,
        ) -> Result<(), RocmExecutionError> {
            // SAFETY: the allocation range and provenance were validated by the public boundary.
            let destination = unsafe { allocation.pointer.as_ptr().cast::<u8>().add(offset) };
            let status = unsafe {
                (self.memset_async)(
                    destination.cast(),
                    i32::from(value),
                    byte_length,
                    stream.stream.as_ptr(),
                )
            };
            self.check_hip_status("hipMemsetAsync", stream.device, 0, status)
        }

        pub(super) fn record_event(
            self: &Arc<Self>,
            stream: &ExecutionStream,
        ) -> Result<ExecutionEvent, RocmExecutionError> {
            let mut event = std::ptr::null_mut();
            // SAFETY: HIP initializes one event handle on success. The reviewed disable-timing
            // flag avoids device overhead that execution fences do not need.
            let status = unsafe { (self.event_create)(&mut event, abi::HIP_EVENT_DISABLE_TIMING) };
            self.check_hip_status("hipEventCreateWithFlags", stream.device, 0, status)?;
            let event = NonNull::new(event).ok_or_else(|| RocmExecutionError::Status {
                operation: "hipEventCreateWithFlags",
                status: -1,
                message: "HIP returned success with a null event".to_owned(),
            })?;
            // SAFETY: both handles are live and retained by their Rust owners.
            let status = unsafe { (self.event_record)(event.as_ptr(), stream.stream.as_ptr()) };
            if let Err(error) = self.check_hip_status("hipEventRecord", stream.device, 0, status) {
                // SAFETY: event is uniquely owned on this error path.
                let destroy_status = unsafe { (self.event_destroy)(event.as_ptr()) };
                if destroy_status != abi::HIP_SUCCESS {
                    eprintln!(
                        "ROCm hipEventDestroy failed after record error: status {destroy_status}"
                    );
                }
                return Err(error);
            }
            Ok(ExecutionEvent {
                event,
                device: stream.device,
                libraries: self.clone(),
            })
        }

        pub(super) fn sgemm_f32(
            self: &Arc<Self>,
            stream: &ExecutionStream,
            [rows, columns, inner]: [usize; 3],
            left: &DeviceAllocation,
            right: &DeviceAllocation,
            output: &DeviceAllocation,
        ) -> Result<(), RocmExecutionError> {
            let requested_bytes = rows
                .checked_mul(columns)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| invalid_execution_argument("SGEMM output size overflow"))?;
            let rows = i32::try_from(rows).map_err(|_| {
                invalid_execution_argument("SGEMM rows exceed rocBLAS integer range")
            })?;
            let columns = i32::try_from(columns).map_err(|_| {
                invalid_execution_argument("SGEMM columns exceed rocBLAS integer range")
            })?;
            let inner = i32::try_from(inner).map_err(|_| {
                invalid_execution_argument("SGEMM inner dimension exceeds rocBLAS integer range")
            })?;
            let mut handle = std::ptr::null_mut();
            // SAFETY: rocBLAS initializes one handle on success.
            let status = unsafe { (self.rocblas_create)(&mut handle) };
            check_rocblas_status(
                "rocblas_create_handle",
                stream.device,
                requested_bytes,
                status,
            )?;
            let handle = NonNull::new(handle).ok_or_else(|| RocmExecutionError::Status {
                operation: "rocblas_create_handle",
                status: -1,
                message: "rocBLAS returned success with a null handle".to_owned(),
            })?;
            let guard = RocblasHandleGuard {
                handle,
                libraries: self.clone(),
            };
            // SAFETY: handle and stream are live and belong to the same certified runtime.
            check_rocblas_status(
                "rocblas_set_stream",
                stream.device,
                requested_bytes,
                unsafe { (self.rocblas_set_stream)(guard.handle.as_ptr(), stream.stream.as_ptr()) },
            )?;
            let alpha = 1.0_f32;
            let beta = 0.0_f32;
            // Matrices are column-major and neither operand is transposed.
            let status = unsafe {
                (self.rocblas_sgemm)(
                    guard.handle.as_ptr(),
                    abi::ROCBLAS_OPERATION_NONE,
                    abi::ROCBLAS_OPERATION_NONE,
                    rows,
                    columns,
                    inner,
                    &alpha,
                    left.pointer.as_ptr().cast(),
                    rows,
                    right.pointer.as_ptr().cast(),
                    inner,
                    &beta,
                    output.pointer.as_ptr().cast(),
                    rows,
                )
            };
            check_rocblas_status("rocblas_sgemm", stream.device, requested_bytes, status)?;
            stream.synchronize()
        }

        fn check_hip_status(
            &self,
            operation: &'static str,
            device: u32,
            allocation_bytes: usize,
            status: i32,
        ) -> Result<(), RocmExecutionError> {
            if status == abi::HIP_SUCCESS {
                return Ok(());
            }
            classify_hip_status(
                operation,
                device,
                allocation_bytes,
                status,
                self.hip_error_message(status),
            )
        }

        fn hip_error_message(&self, status: i32) -> String {
            // SAFETY: this generated HIP function returns null or a static C string.
            let message = unsafe { (self.get_error_string)(status) };
            if message.is_null() {
                format!("HIP status {status}")
            } else {
                // SAFETY: the checked non-null vendor pointer names a static C string.
                unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .into_owned()
            }
        }

        pub(super) fn library_count(&self) -> usize {
            self.libraries.len()
        }
    }

    fn check_rocblas_status(
        operation: &'static str,
        device: u32,
        requested_bytes: usize,
        status: i32,
    ) -> Result<(), RocmExecutionError> {
        classify_rocblas_status(operation, device, requested_bytes, status)
    }

    pub(super) struct DeviceAllocation {
        pointer: NonNull<c_void>,
        device: u32,
        libraries: Arc<LoadedLibraries>,
    }

    impl Drop for DeviceAllocation {
        fn drop(&mut self) {
            if let Err(error) = self.libraries.select_device(self.device) {
                eprintln!("ROCm allocation cleanup could not select its device: {error}");
                return;
            }
            // SAFETY: this object uniquely owns the live allocation pointer.
            let status = unsafe { (self.libraries.free)(self.pointer.as_ptr()) };
            if status != abi::HIP_SUCCESS {
                eprintln!("ROCm hipFree failed during allocation cleanup: status {status}");
            }
        }
    }

    // SAFETY: the allocation is opaque device memory; access is sequenced by explicit HIP streams.
    unsafe impl Send for DeviceAllocation {}
    // SAFETY: immutable Rust references never directly dereference the device pointer.
    unsafe impl Sync for DeviceAllocation {}

    pub(super) struct ExecutionStream {
        stream: NonNull<c_void>,
        device: u32,
        libraries: Arc<LoadedLibraries>,
    }

    impl ExecutionStream {
        pub(super) fn synchronize(&self) -> Result<(), RocmExecutionError> {
            self.libraries.select_device(self.device)?;
            // SAFETY: the stream handle remains live for this call.
            let status = unsafe { (self.libraries.stream_synchronize)(self.stream.as_ptr()) };
            self.libraries
                .check_hip_status("hipStreamSynchronize", self.device, 0, status)
        }
    }

    impl Drop for ExecutionStream {
        fn drop(&mut self) {
            if let Err(error) = self.synchronize() {
                eprintln!("ROCm stream synchronization failed during cleanup: {error}");
            }
            // SAFETY: this object uniquely owns the live stream handle.
            let status = unsafe { (self.libraries.stream_destroy)(self.stream.as_ptr()) };
            if status != abi::HIP_SUCCESS {
                eprintln!("ROCm hipStreamDestroy failed during cleanup: status {status}");
            }
        }
    }

    // SAFETY: HIP stream handles may be moved between host threads; operations are vendor-serialized.
    unsafe impl Send for ExecutionStream {}
    // SAFETY: calls through shared references do not mutate Rust-owned memory.
    unsafe impl Sync for ExecutionStream {}

    pub(super) struct ExecutionEvent {
        event: NonNull<c_void>,
        device: u32,
        libraries: Arc<LoadedLibraries>,
    }

    impl ExecutionEvent {
        pub(super) fn synchronize(&self) -> Result<(), RocmExecutionError> {
            self.libraries.select_device(self.device)?;
            // SAFETY: the event handle remains live for this call.
            let status = unsafe { (self.libraries.event_synchronize)(self.event.as_ptr()) };
            self.libraries
                .check_hip_status("hipEventSynchronize", self.device, 0, status)
        }
    }

    impl Drop for ExecutionEvent {
        fn drop(&mut self) {
            if let Err(error) = self.synchronize() {
                eprintln!("ROCm event synchronization failed during cleanup: {error}");
            }
            // SAFETY: this object uniquely owns the live event handle.
            let status = unsafe { (self.libraries.event_destroy)(self.event.as_ptr()) };
            if status != abi::HIP_SUCCESS {
                eprintln!("ROCm hipEventDestroy failed during cleanup: status {status}");
            }
        }
    }

    // SAFETY: HIP event handles may be moved between host threads and are synchronized explicitly.
    unsafe impl Send for ExecutionEvent {}
    // SAFETY: shared references cannot directly mutate Rust-owned memory.
    unsafe impl Sync for ExecutionEvent {}

    struct RocblasHandleGuard {
        handle: NonNull<c_void>,
        libraries: Arc<LoadedLibraries>,
    }

    impl Drop for RocblasHandleGuard {
        fn drop(&mut self) {
            // SAFETY: this guard uniquely owns the live rocBLAS handle.
            let status = unsafe { (self.libraries.rocblas_destroy)(self.handle.as_ptr()) };
            if status != abi::ROCBLAS_STATUS_SUCCESS {
                eprintln!("ROCm rocblas_destroy_handle failed during cleanup: status {status}");
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::{
            process::Command,
            time::{SystemTime, UNIX_EPOCH},
        };

        #[test]
        #[allow(
            clippy::disallowed_methods,
            reason = "the Linux-only adversarial loader test must synchronously compile its two tiny ELF fixtures before dlmopen"
        )]
        fn ambient_same_soname_cannot_satisfy_a_certified_binding()
        -> Result<(), Box<dyn std::error::Error>> {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let root = env::temp_dir().join(format!(
                "sim-rocm-link-map-proof-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&root)?;
            let source = root.join("fixture.c");
            fs::write(&source, "int sim_rocm_fixture(void) { return 1; }\n")?;
            let certified = root.join("certified/libsame.so.1");
            let ambient = root.join("ambient/libsame.so.1");
            let certified_directory = certified
                .parent()
                .ok_or_else(|| std::io::Error::other("certified path has no parent"))?;
            let ambient_directory = ambient
                .parent()
                .ok_or_else(|| std::io::Error::other("ambient path has no parent"))?;
            fs::create_dir_all(certified_directory)?;
            fs::create_dir_all(ambient_directory)?;
            for output in [&certified, &ambient] {
                let result = Command::new("cc")
                    .arg("-shared")
                    .arg("-fPIC")
                    .arg("-Wl,-soname,libsame.so.1")
                    .arg(&source)
                    .arg("-o")
                    .arg(output)
                    .output()?;
                if !result.status.success() {
                    return Err(format!(
                        "fixture C compiler failed: {}",
                        String::from_utf8_lossy(&result.stderr)
                    )
                    .into());
                }
            }
            let consumer_source = root.join("consumer.c");
            fs::write(
                &consumer_source,
                "extern int sim_rocm_fixture(void);\nint sim_rocm_consumer(void) { return sim_rocm_fixture(); }\n",
            )?;
            let consumer = root.join("libconsumer.so.1");
            let result = Command::new("cc")
                .arg("-shared")
                .arg("-fPIC")
                .arg("-Wl,-soname,libconsumer.so.1")
                .arg(&consumer_source)
                .arg("-L")
                .arg(certified_directory)
                .arg("-l:libsame.so.1")
                .arg("-o")
                .arg(&consumer)
                .output()?;
            if !result.status.success() {
                return Err(format!(
                    "fixture consumer compiler failed: {}",
                    String::from_utf8_lossy(&result.stderr)
                )
                .into());
            }
            let (loaded_ambient, namespace) =
                DynamicLibrary::open_in_namespace("dependency", &ambient, None)?;
            let (loaded_consumer, loaded_namespace) =
                DynamicLibrary::open_in_namespace("consumer", &consumer, Some(namespace))?;
            assert_eq!(loaded_namespace, namespace);
            let result = prove_exact_bindings(
                &[loaded_ambient, loaded_consumer],
                &BTreeMap::from([
                    ("dependency".to_owned(), certified.clone()),
                    ("consumer".to_owned(), consumer),
                ]),
                &BTreeMap::from([
                    ("dependency".to_owned(), "libsame.so.1".to_owned()),
                    ("consumer".to_owned(), "libconsumer.so.1".to_owned()),
                ]),
                &[RocmDependencyCandidate::new(
                    "dependency",
                    "libsame.so.1",
                    certified,
                    "6.1.0",
                    vec!["sim_rocm_fixture".to_owned()],
                    "sim_backend_rocm::loader",
                )],
                &[RocmDependencyEdge::new("consumer", "dependency")],
            );
            assert!(
                matches!(result, Err(RocmLoadError::BindingProof { reason }) if reason.contains("instead of retained descriptor"))
            );
            fs::remove_dir_all(root)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::Cell,
        ffi::OsString,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temporary_directory(label: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!("sim-rocm-{label}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    fn valid_package_manifest(
        runtime_root: &Path,
        ffi_contracts_sha256: &str,
    ) -> Result<String, serde_json::Error> {
        Ok(format!(
            r#"{{
                "schema_version":2,
                "backend":"rocm",
                "abi_floor":"6.1.0",
                "abi_manifest_sha256":"{ABI_MANIFEST_SHA256}",
                "ffi_contracts_sha256":"{ffi_contracts_sha256}",
                "redistributes_amd_runtime":false,
                "signer":"zed.release",
                "signature_algorithm":"ed25519",
                "signature_domain":"zed-sim-rocm-package-v1",
                "signature_coverage":"package-coverage-v1",
                "runtime_root":{}
            }}"#,
            serde_json::to_string(runtime_root)?
        ))
    }

    fn write_package_coverage(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut payloads = BTreeMap::new();
        for relative in [
            "LICENSES",
            "abi/symbols-v1.json",
            "adapter-manifest.json",
            "ffi-contracts-v1.json",
            "package-policy.json",
        ] {
            let bytes = fs::read(root.join(relative))?;
            let mut sha256 = Sha256::new();
            sha256.update(&bytes);
            payloads.insert(
                relative,
                (sha256.finish_hex(bytes.len() as u64)?, bytes.len()),
            );
        }
        let coverage = payloads
            .into_iter()
            .map(|(relative, (digest, size))| format!("{digest} {size}  {relative}\n"))
            .collect::<String>();
        fs::write(root.join("package-coverage.sha256"), coverage)?;
        Ok(())
    }

    fn write_valid_package(
        root: &Path,
        runtime_root: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(root.join("abi"))?;
        fs::write(root.join("abi/symbols-v1.json"), ABI_MANIFEST)?;
        fs::write(root.join("LICENSES"), "fixture notices\n")?;
        fs::write(root.join("package-policy.json"), PACKAGE_POLICY)?;
        let ffi_contracts = b"{\"schema_version\":1,\"backend\":\"rocm\",\"abi_floor\":\"6.1.0\",\"libraries\":[]}\n";
        fs::write(root.join("ffi-contracts-v1.json"), ffi_contracts)?;
        let ffi_contracts_sha256 = sha256_hex(ffi_contracts)?;
        fs::write(
            root.join("adapter-manifest.json"),
            valid_package_manifest(runtime_root, &ffi_contracts_sha256)?,
        )?;
        fs::write(
            root.join("adapter-manifest.sig"),
            concat!(
                "{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"",
                "0000000000000000000000000000000000000000000000000000000000000000",
                "0000000000000000000000000000000000000000000000000000000000000000",
                "\"}\n"
            ),
        )?;
        write_package_coverage(root)
    }

    #[test]
    fn manifest_matches_compiled_symbols_layouts_and_reviewed_headers() -> Result<(), RocmLoadError>
    {
        let manifest = validate_manifest()?;
        assert_eq!(manifest.libraries.len(), abi::LIBRARIES.len());
        assert_eq!(manifest.layouts.len(), 3);
        assert_eq!(manifest.headers.len(), 9);
        Ok(())
    }

    #[test]
    fn discovery_order_is_explicit_and_never_adds_ambient_sonames() {
        let roots = deduplicate_roots(vec![
            DiscoveryRoot::new(DiscoverySource::SimRocmRoot, "/sdk/sim"),
            DiscoveryRoot::new(DiscoverySource::RocmPath, "/sdk/rocm"),
            DiscoveryRoot::new(DiscoverySource::SignedPackage, "/app/rocm"),
        ]);
        assert_eq!(
            roots.iter().map(DiscoveryRoot::source).collect::<Vec<_>>(),
            vec![
                DiscoverySource::SimRocmRoot,
                DiscoverySource::RocmPath,
                DiscoverySource::SignedPackage,
            ]
        );
        assert!(abi::LIBRARIES.iter().all(|(_, file)| file.contains(".so")));
    }

    #[test]
    fn safe_discovery_returns_certification_evidence_without_loading()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = temporary_directory("pure-discovery")?;
        fs::create_dir_all(root.join("lib"))?;
        for &(_, filename) in abi::LIBRARIES {
            fs::write(root.join("lib").join(filename), "not a dynamic library")?;
        }
        let library_set =
            discover_library_set(&[DiscoveryRoot::new(DiscoverySource::SimRocmRoot, &root)])?;
        assert_eq!(library_set.libraries().len(), abi::LIBRARIES.len());
        assert!(library_set.libraries().iter().all(|library| {
            library.abi_version() == "6.1.0"
                && library.unsafe_owner() == "sim_backend_rocm::loader"
                && !library.required_symbols().is_empty()
        }));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn loader_entry_point_requires_an_unsafe_call_site() {
        fn accept_unsafe_loader(
            _loader: unsafe fn(
                &RocmLibrarySet,
                Arc<dyn Any + Send + Sync>,
            ) -> Result<RocmRuntime, RocmLoadError>,
        ) {
        }

        accept_unsafe_loader(RocmRuntime::load_certified);
    }

    #[test]
    fn retained_descriptor_remap_requires_every_exact_library_id() {
        let library_set = RocmLibrarySet {
            libraries: abi::LIBRARIES
                .iter()
                .map(|&(library_id, filename)| RocmLibraryCandidate {
                    library_id,
                    path: PathBuf::from("/sdk/lib").join(filename),
                    abi_version: "6.1.0",
                    required_symbols: abi::SYMBOLS
                        .iter()
                        .filter(|symbol| symbol.library == library_id)
                        .map(|symbol| symbol.name)
                        .collect(),
                    unsafe_owner: "sim_backend_rocm::loader",
                })
                .collect(),
            dependencies: Vec::new(),
            dependency_edges: Vec::new(),
        };
        assert!(matches!(
            validate_retained_descriptor_set(&library_set),
            Err(RocmLoadError::CertifiedPathRemap { .. })
        ));
        let mut complete = BTreeMap::new();
        for (descriptor, library) in library_set.libraries().iter().enumerate() {
            complete.insert(
                library.library_id().to_owned(),
                PathBuf::from(format!("/proc/self/fd/{}", descriptor + 10)),
            );
        }
        let remapped = library_set.remap_to_retained_descriptors(complete.clone());
        assert!(matches!(
            remapped,
            Err(RocmLoadError::CertifiedPathRemap { .. })
        ));

        let mut incomplete = complete.clone();
        incomplete.remove("libMIOpen");
        assert!(matches!(
            library_set.remap_to_retained_descriptors(incomplete),
            Err(RocmLoadError::CertifiedPathRemap { .. })
        ));

        let mut unknown = complete;
        unknown.remove("libMIOpen");
        unknown.insert("libunknown".to_owned(), PathBuf::from("/proc/self/fd/99"));
        assert!(matches!(
            library_set.remap_to_retained_descriptors(unknown),
            Err(RocmLoadError::CertifiedPathRemap { .. })
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn retained_descriptor_remap_accepts_only_live_sealed_memfd_images()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::{
            ffi::CString,
            fs::File,
            os::fd::{AsRawFd, FromRawFd},
        };

        let library_set = RocmLibrarySet {
            libraries: abi::LIBRARIES
                .iter()
                .map(|&(library_id, filename)| RocmLibraryCandidate {
                    library_id,
                    path: PathBuf::from("/sdk/lib").join(filename),
                    abi_version: "6.1.0",
                    required_symbols: Vec::new(),
                    unsafe_owner: "sim_backend_rocm::loader",
                })
                .collect(),
            dependencies: Vec::new(),
            dependency_edges: Vec::new(),
        };
        let mut retained = Vec::new();
        let mut paths = BTreeMap::new();
        for library in library_set.libraries() {
            let name = CString::new(format!("zed-test-{}", library.library_id()))?;
            // SAFETY: the name is NUL terminated and a successful descriptor is transferred
            // immediately into its sole File owner.
            let descriptor = unsafe {
                libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
            };
            if descriptor < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            // SAFETY: descriptor is newly allocated and uniquely owned.
            let file = unsafe { File::from_raw_fd(descriptor) };
            let seals =
                libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
            // SAFETY: F_ADD_SEALS consumes only the integer seal mask for the live memfd.
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            paths.insert(
                library.library_id().to_owned(),
                PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd())),
            );
            retained.push(file);
        }
        let remapped = library_set.remap_to_retained_descriptors(paths)?;
        assert_eq!(remapped.libraries().len(), retained.len());
        Ok(())
    }

    #[test]
    fn absent_library_reports_exact_first_missing_contract()
    -> Result<(), Box<dyn std::error::Error>> {
        if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            return Ok(());
        }
        let root = temporary_directory("missing-library")?;
        fs::create_dir_all(root.join("lib"))?;
        let result =
            resolve_library_set(&[DiscoveryRoot::new(DiscoverySource::SimRocmRoot, &root)]);
        assert!(matches!(
            result,
            Err(RocmLoadError::MissingLibrary {
                library: "libamdhip64",
                ..
            })
        ));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn package_metadata_rejects_tampering_before_platform_verification()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = temporary_directory("package-policy")?;
        let runtime_root = temporary_directory("package-policy-runtime")?;
        write_valid_package(&root, &runtime_root)?;
        fs::write(
            root.join("adapter-manifest.json"),
            r#"{
            "schema_version":2,
            "backend":"rocm",
            "abi_floor":"6.1.0",
            "abi_manifest_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
            "ffi_contracts_sha256":"0000000000000000000000000000000000000000000000000000000000000000",
            "redistributes_amd_runtime":true,
            "signer":"fixture",
            "signature_algorithm":"ed25519",
            "signature_domain":"zed-sim-rocm-package-v1",
            "signature_coverage":"package-coverage-v1",
            "runtime_root":"/opt/rocm"
        }"#,
        )?;
        assert!(matches!(
            validate_signed_package(&root),
            Err(RocmLoadError::PackageMetadata { .. })
        ));
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(runtime_root)?;
        Ok(())
    }

    struct RejectingVerifier;

    impl PlatformPackageVerifier for RejectingVerifier {
        fn verify_rocm_package(
            &self,
            _package_root: &Path,
            _contract: &PackageSignatureContract,
        ) -> Result<(), String> {
            Err("fixture signature rejection".to_owned())
        }
    }

    struct ContractCheckingVerifier {
        called: Cell<bool>,
    }

    impl PlatformPackageVerifier for ContractCheckingVerifier {
        fn verify_rocm_package(
            &self,
            _package_root: &Path,
            contract: &PackageSignatureContract,
        ) -> Result<(), String> {
            if contract.signer() != "zed.release"
                || contract.abi_manifest_sha256() != ABI_MANIFEST_SHA256
                || contract.signature_algorithm() != "ed25519"
                || contract.signature_coverage() != "package-coverage-v1"
                || contract.package_policy_bytes() != PACKAGE_POLICY.as_bytes()
                || !contract
                    .ffi_contracts_bytes()
                    .windows("schema_version".len())
                    .any(|window| window == b"schema_version")
                || !contract
                    .package_coverage_bytes()
                    .windows("adapter-manifest.json".len())
                    .any(|window| window == b"adapter-manifest.json")
                || !contract
                    .adapter_manifest_bytes()
                    .windows("zed.release".len())
                    .any(|window| window == b"zed.release")
            {
                return Err("wrong signature contract".to_owned());
            }
            self.called.set(true);
            Ok(())
        }
    }

    #[test]
    fn structurally_valid_metadata_cannot_admit_a_package_without_verifier_approval()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = temporary_directory("signature-owner")?;
        let runtime_root = temporary_directory("signature-owner-runtime")?;
        write_valid_package(&root, &runtime_root)?;
        assert!(matches!(
            admit_signed_package_root(&root, &RejectingVerifier),
            Err(RocmLoadError::PackageVerification { reason, .. }) if reason == "fixture signature rejection"
        ));
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(runtime_root)?;
        Ok(())
    }

    #[test]
    fn signed_package_adapter_delegates_verification_before_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = temporary_directory("signature-delegation")?;
        let runtime_root = temporary_directory("signature-delegation-runtime")?;
        write_valid_package(&root, &runtime_root)?;
        let verifier = ContractCheckingVerifier {
            called: Cell::new(false),
        };
        let admitted = admit_signed_package_root(&root, &verifier)?;
        assert!(verifier.called.get());
        assert_eq!(admitted.source(), DiscoverySource::SignedPackage);
        assert_eq!(admitted.path(), runtime_root.canonicalize()?);
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(runtime_root)?;
        Ok(())
    }

    struct ApprovingVerifier {
        called: Cell<bool>,
    }

    impl PlatformPackageVerifier for ApprovingVerifier {
        fn verify_rocm_package(
            &self,
            _package_root: &Path,
            _contract: &PackageSignatureContract,
        ) -> Result<(), String> {
            self.called.set(true);
            Ok(())
        }
    }

    fn assert_rejected_before_approval(
        root: &Path,
        verifier: &ApprovingVerifier,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(matches!(
            admit_signed_package_root(root, verifier),
            Err(RocmLoadError::PackageMetadata { .. })
        ));
        assert!(!verifier.called.get());
        Ok(())
    }

    #[test]
    fn approving_verifier_cannot_admit_extra_or_forbidden_payloads()
    -> Result<(), Box<dyn std::error::Error>> {
        for (label, relative) in [
            ("extra", "unexpected.txt"),
            ("runtime", "kernels/libamdhip64.so.6"),
        ] {
            let root = temporary_directory(label)?;
            let runtime_root = temporary_directory(&format!("{label}-runtime"))?;
            write_valid_package(&root, &runtime_root)?;
            if let Some(parent) = root.join(relative).parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(root.join(relative), "untrusted payload")?;
            let verifier = ApprovingVerifier {
                called: Cell::new(false),
            };
            assert_rejected_before_approval(&root, &verifier)?;
            fs::remove_dir_all(root)?;
            fs::remove_dir_all(runtime_root)?;
        }
        Ok(())
    }

    #[test]
    fn approving_verifier_cannot_admit_coverage_omission_or_tamper()
    -> Result<(), Box<dyn std::error::Error>> {
        let omission_root = temporary_directory("coverage-omission")?;
        let omission_runtime = temporary_directory("coverage-omission-runtime")?;
        write_valid_package(&omission_root, &omission_runtime)?;
        let coverage = fs::read_to_string(omission_root.join("package-coverage.sha256"))?;
        let omitted = coverage
            .lines()
            .filter(|line| !line.ends_with("ffi-contracts-v1.json"))
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        fs::write(omission_root.join("package-coverage.sha256"), omitted)?;
        let verifier = ApprovingVerifier {
            called: Cell::new(false),
        };
        assert_rejected_before_approval(&omission_root, &verifier)?;
        fs::remove_dir_all(omission_root)?;
        fs::remove_dir_all(omission_runtime)?;

        let tamper_root = temporary_directory("coverage-tamper")?;
        let tamper_runtime = temporary_directory("coverage-tamper-runtime")?;
        write_valid_package(&tamper_root, &tamper_runtime)?;
        fs::write(
            tamper_root.join("ffi-contracts-v1.json"),
            "{\"tampered\":true}\n",
        )?;
        let verifier = ApprovingVerifier {
            called: Cell::new(false),
        };
        assert_rejected_before_approval(&tamper_root, &verifier)?;
        fs::remove_dir_all(tamper_root)?;
        fs::remove_dir_all(tamper_runtime)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn approving_verifier_cannot_admit_a_package_symlink() -> Result<(), Box<dyn std::error::Error>>
    {
        use std::os::unix::fs::symlink;

        let root = temporary_directory("package-symlink")?;
        let runtime_root = temporary_directory("package-symlink-runtime")?;
        write_valid_package(&root, &runtime_root)?;
        symlink(root.join("LICENSES"), root.join("unexpected-link"))?;
        let verifier = ApprovingVerifier {
            called: Cell::new(false),
        };
        assert_rejected_before_approval(&root, &verifier)?;
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(runtime_root)?;
        Ok(())
    }

    struct MutatingApprovingVerifier;

    impl PlatformPackageVerifier for MutatingApprovingVerifier {
        fn verify_rocm_package(
            &self,
            package_root: &Path,
            _contract: &PackageSignatureContract,
        ) -> Result<(), String> {
            fs::write(
                package_root.join("unexpected-after-verification"),
                "payload",
            )
            .map_err(|error| error.to_string())?;
            Ok(())
        }
    }

    #[test]
    fn verifier_time_package_mutation_is_revalidated_before_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = temporary_directory("verifier-mutation")?;
        let runtime_root = temporary_directory("verifier-mutation-runtime")?;
        write_valid_package(&root, &runtime_root)?;
        assert!(matches!(
            admit_signed_package_root(&root, &MutatingApprovingVerifier),
            Err(RocmLoadError::PackageMetadata { .. })
        ));
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(runtime_root)?;
        Ok(())
    }

    #[test]
    fn package_sha256_matches_standard_vector() -> Result<(), RocmLoadError> {
        let mut sha256 = Sha256::new();
        sha256.update(b"abc");
        assert_eq!(
            sha256.finish_hex(3)?,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn vendor_library_final_symlink_is_always_rejected() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let root = temporary_directory("contained-library")?;
        let outside = temporary_directory("outside-library")?;
        fs::create_dir_all(root.join("lib"))?;
        fs::write(outside.join("libamdhip64.so"), "not a library")?;
        symlink(
            outside.join("libamdhip64.so"),
            root.join("lib/libamdhip64.so"),
        )?;
        let canonical = root.canonicalize()?;
        assert_eq!(library_path(&canonical, "libamdhip64.so"), None);
        fs::remove_file(root.join("lib/libamdhip64.so"))?;
        fs::write(root.join("lib/libamdhip64-real.so"), "not a library")?;
        symlink(
            root.join("lib/libamdhip64-real.so"),
            root.join("lib/libamdhip64.so"),
        )?;
        assert_eq!(library_path(&canonical, "libamdhip64.so"), None);
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[test]
    fn version_encoding_enforces_the_rocm_floor() {
        assert_eq!(hip_version(60_100_000), abi::RocmVersion::new(6, 1, 0));
        assert!(59_900_000 < abi::HIP_RUNTIME_FLOOR);
        assert!(!miopen_meets_floor(3, 0));
        assert!(miopen_meets_floor(3, 1));
        assert_eq!(
            leading_version("rocBLAS 4.1.2.60102"),
            Some(abi::RocmVersion::new(4, 1, 2))
        );
        assert_eq!(
            enforce_hip_floor("HIP driver", 60_100_000),
            Ok("6.1.0".to_owned())
        );
        assert!(matches!(
            enforce_hip_floor("HIP driver", 60_099_999),
            Err(RocmLoadError::VersionTooOld {
                component: "HIP driver",
                minimum,
                ..
            }) if minimum == "6.1.0"
        ));
        assert_eq!(
            check_probe_status("hipInit", 17, abi::HIP_SUCCESS),
            Err(RocmLoadError::Probe {
                function: "hipInit",
                status: 17,
            })
        );
    }

    fn dependency_graph_fixture() -> RocmLibrarySet {
        RocmLibrarySet {
            libraries: abi::LIBRARIES
                .iter()
                .map(|&(library_id, filename)| RocmLibraryCandidate {
                    library_id,
                    path: PathBuf::from("/sdk/lib").join(filename),
                    abi_version: "6.1.0",
                    required_symbols: abi::SYMBOLS
                        .iter()
                        .filter(|symbol| symbol.library == library_id)
                        .map(|symbol| symbol.name)
                        .collect(),
                    unsafe_owner: "sim_backend_rocm::loader",
                })
                .collect(),
            dependencies: Vec::new(),
            dependency_edges: Vec::new(),
        }
    }

    fn dependency(id: &str, soname: &str) -> RocmDependencyCandidate {
        RocmDependencyCandidate::new(
            id,
            soname,
            PathBuf::from("/sdk/lib").join(soname),
            "6.1.0",
            Vec::new(),
            "sim_backend_rocm::loader",
        )
    }

    #[test]
    fn dependency_closure_orders_dependencies_before_consumers_deterministically()
    -> Result<(), RocmLoadError> {
        let set = dependency_graph_fixture().with_certified_dependency_closure(
            vec![
                dependency("hsa", "libhsa-runtime64.so.1"),
                dependency("numa", "libnuma.so.1"),
            ],
            vec![
                RocmDependencyEdge::new("libamdhip64", "hsa"),
                RocmDependencyEdge::new("hsa", "numa"),
            ],
        )?;
        let order = set.load_order()?;
        let position = |id: &str| {
            order
                .iter()
                .position(|candidate| candidate == id)
                .ok_or_else(|| RocmLoadError::DependencyGraph {
                    reason: format!("test order omitted {id}"),
                })
        };
        assert!(position("numa")? < position("hsa")?);
        assert!(position("hsa")? < position("libamdhip64")?);
        assert_eq!(set.dependencies()[0].library_id(), "hsa");
        assert_eq!(set.dependencies()[1].library_id(), "numa");
        Ok(())
    }

    #[test]
    fn dependency_closure_rejects_cycles_or_unreachable_and_duplicate_sonames() {
        let cycle = dependency_graph_fixture().with_certified_dependency_closure(
            vec![
                dependency("first", "libfirst.so.1"),
                dependency("second", "libsecond.so.1"),
            ],
            vec![
                RocmDependencyEdge::new("libamdhip64", "first"),
                RocmDependencyEdge::new("first", "second"),
                RocmDependencyEdge::new("second", "first"),
            ],
        );
        assert!(
            matches!(cycle, Err(RocmLoadError::DependencyGraph { reason }) if reason.contains("cycle"))
        );

        let unreachable = dependency_graph_fixture().with_certified_dependency_closure(
            vec![dependency("orphan", "liborphan.so.1")],
            Vec::new(),
        );
        assert!(
            matches!(unreachable, Err(RocmLoadError::DependencyGraph { reason }) if reason.contains("reachable"))
        );

        let duplicate = dependency_graph_fixture().with_certified_dependency_closure(
            vec![
                dependency("first", "libsame.so.1"),
                dependency("second", "libsame.so.1"),
            ],
            vec![
                RocmDependencyEdge::new("libamdhip64", "first"),
                RocmDependencyEdge::new("libamdhip64", "second"),
            ],
        );
        assert!(
            matches!(duplicate, Err(RocmLoadError::DependencyGraph { reason }) if reason.contains("unique SONAME"))
        );
    }

    #[test]
    fn hip_statuses_map_to_typed_oom_device_loss_and_status_errors() {
        assert!(matches!(
            classify_hip_status(
                "hipMalloc",
                3,
                4096,
                abi::HIP_ERROR_OUT_OF_MEMORY,
                "out".to_owned(),
            ),
            Err(RocmExecutionError::OutOfMemory {
                device: 3,
                bytes: 4096,
                ..
            })
        ));
        for status in [
            abi::HIP_ERROR_INVALID_CONTEXT,
            abi::HIP_ERROR_ILLEGAL_ADDRESS,
            abi::HIP_ERROR_CONTEXT_IS_DESTROYED,
            abi::HIP_ERROR_LAUNCH_FAILURE,
        ] {
            assert!(matches!(
                classify_hip_status("kernel", 2, 0, status, "lost".to_owned()),
                Err(RocmExecutionError::DeviceLost {
                    device: 2,
                    operation: "kernel",
                    ..
                })
            ));
        }
        assert!(matches!(
            classify_hip_status("copy", 0, 0, 17, "bad".to_owned()),
            Err(RocmExecutionError::Status {
                operation: "copy",
                status: 17,
                ..
            })
        ));
        assert_eq!(classify_hip_status("ok", 0, 0, 0, String::new()), Ok(()));
    }

    #[test]
    fn rocblas_memory_error_preserves_device_and_requested_bytes() {
        assert!(matches!(
            classify_rocblas_status(
                "rocblas_sgemm",
                7,
                16_384,
                abi::ROCBLAS_STATUS_MEMORY_ERROR,
            ),
            Err(RocmExecutionError::OutOfMemory {
                device: 7,
                bytes: 16_384,
                message,
            }) if message.contains("memory_error")
        ));
        assert!(matches!(
            classify_rocblas_status("rocblas_sgemm", 7, 16_384, 4),
            Err(RocmExecutionError::Status {
                operation: "rocblas_sgemm",
                status: 4,
                ..
            })
        ));
        assert_eq!(
            classify_rocblas_status("rocblas_sgemm", 0, 0, abi::ROCBLAS_STATUS_SUCCESS),
            Ok(())
        );
    }

    #[test]
    fn device_properties_are_bounded_and_derived_from_reviewed_probes()
    -> Result<(), RocmExecutionError> {
        let properties = assemble_rocm_device_properties(
            "AMD Instinct Fixture".to_owned(),
            64 * 1024 * 1024,
            9,
            4,
        )?;
        assert_eq!(properties.name(), "AMD Instinct Fixture");
        assert_eq!(properties.total_memory_bytes(), 64 * 1024 * 1024);
        assert_eq!((properties.major(), properties.minor()), (9, 4));
        assert_eq!(properties.architecture(), Some("hip-compute-9.4"));
        assert!(!properties.has_fp16());
        assert!(assemble_rocm_device_properties(String::new(), 1, 1, 0).is_err());
        assert!(assemble_rocm_device_properties("fixture".to_owned(), 0, 1, 0).is_err());
        assert!(assemble_rocm_device_properties("x".repeat(256), 1, 1, 0).is_err());
        Ok(())
    }

    #[test]
    fn non_target_error_names_required_and_actual_targets() {
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            return;
        }
        assert!(
            matches!(target_gate(), Err(RocmLoadError::UnsupportedTarget { required: "x86_64-unknown-linux-gnu", actual }) if !actual.is_empty())
        );
    }

    #[test]
    fn path_conversion_retains_non_utf8_capability() {
        assert_eq!(
            Path::new("/opt/rocm").as_os_str().to_owned(),
            OsString::from("/opt/rocm")
        );
    }
}
