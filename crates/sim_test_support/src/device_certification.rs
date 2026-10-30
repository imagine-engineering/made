use std::{
    collections::BTreeSet,
    fs::{self, File, Metadata},
    io::{self, Read, Seek},
    path::{Component, Path, PathBuf},
};

use ring::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const SIGNATURE_DOMAIN: &[u8] = b"zed-sim-device-certification-v2\0";
const MAX_ARTIFACT_BYTES: usize = 1024 * 1024;
const MAX_MATRIX_ROWS: usize = 4096;
const MAX_PROVENANCE_ROWS: usize = 4096;
const MAX_FACT_ROWS: usize = 256;
const MAX_IMPLEMENTATION_FILES: usize = 4096;
const MAX_IMPLEMENTATION_TREE_ENTRIES: usize = 8192;
const MAX_IMPLEMENTATION_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_IMPLEMENTATION_PATH_BYTES: usize = 4096;
const MAX_IMPLEMENTATION_COMPONENT_BYTES: usize = 255;
const MAX_SIGNING_KEY_BYTES: u64 = 16 * 1024;
const IMPLEMENTATION_MANIFEST_DOMAIN: &[u8] = b"zed-sim-device-certification-implementation-v1\0";

const CPU_IMPLEMENTATION_FILES: &[&str] = &[
    ".cargo/config.toml",
    "crates/sim_test_support/parity/catalogs/native-tensor-operation-contracts.csv",
    "Cargo.lock",
    "Cargo.toml",
    "crates/sim_tensor/Cargo.toml",
    "crates/sim_tensor/build.rs",
    "crates/sim_tensor/operation_contract_evidence.rs",
    "crates/sim_tensor/src/autograd.rs",
    "crates/sim_tensor/src/backends/cpu_sim_model_0016.rs",
    "crates/sim_tensor/src/sim_tensor.rs",
    "crates/sim_tensor/src/cpu_backend.rs",
    "crates/sim_tensor/src/dtypes.rs",
    "crates/sim_tensor/src/image_ops.rs",
    "crates/sim_tensor/src/operation.rs",
    "crates/sim_tensor/src/operation_contract_records.rs",
    "crates/sim_tensor/src/operation_contracts.rs",
    "crates/sim_tensor/src/promotion.rs",
    "crates/sim_tensor/src/rng.rs",
    "crates/sim_test_support/Cargo.toml",
    "crates/sim_test_support/src/sim_test_support.rs",
    "crates/sim_test_support/src/device_certification.rs",
    "crates/sim_test_support/tests/device_cpu_sim_model_0016.rs",
    "crates/sim_types/Cargo.toml",
    "crates/sim_types/src/cancellation.rs",
    "crates/sim_types/src/sim_types.rs",
    "crates/sim_types/src/worker_protocol.rs",
    "rust-toolchain.toml",
];

const CPU_IMPLEMENTATION_TREES: &[&str] = &[
    "crates/sim_tensor/src/autograd",
    "crates/sim_tensor/src/operation_resolutions",
    "crates/sim_tensor/src/ops",
    "crates/sim_tensor/src/rng_profiles",
];
const CPU_OPERATION_TREE: &str = "crates/sim_tensor/src/ops";
const CPU_BACKEND_OPERATION_MODULE: &str = "backend_cpu_sim_model_0016.rs";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificationStatus {
    Pass,
    Unsupported,
    Failure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationEnvironment {
    pub lab_id: String,
    pub hostname: String,
    pub os_name: String,
    pub os_version: String,
    pub os_build: String,
    pub architecture: String,
    pub rust_target: String,
    pub toolchain: Vec<CertificationFact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationFact {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationMemoryFact {
    pub name: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceEvidence {
    pub name: String,
    pub identifier: String,
    pub memory_model: String,
    pub observed_features: Vec<String>,
    pub memory: Vec<CertificationMemoryFact>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageEvidence {
    pub format: String,
    pub signer: String,
    pub signer_public_key: String,
    pub signature: String,
    pub coverage_sha256: String,
    pub manifest_sha256: String,
    pub contract_catalog_sha256: String,
    pub payloads: Vec<CertificationProvenance>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CertificationPackageEvidence {
    Signed(PackageEvidence),
    NotApplicable { reason: String },
}

impl CertificationPackageEvidence {
    pub fn signed(&self) -> Option<&PackageEvidence> {
        match self {
            Self::Signed(evidence) => Some(evidence),
            Self::NotApplicable { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractEvidence {
    pub abi_contract_sha256: String,
    pub abi_manifest_sha256: String,
    pub execution_abi_sha256: String,
    pub abi_floor: String,
    pub framework_count: usize,
    pub symbol_count: usize,
    pub class_count: usize,
    pub selector_count: usize,
    pub symbols: Vec<String>,
    pub package: CertificationPackageEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationMatrixRow {
    pub id: String,
    pub category: String,
    pub operation: String,
    pub dtype: Option<String>,
    pub layout: Option<String>,
    pub status: CertificationStatus,
    pub tolerance: String,
    pub evidence: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationProvenance {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationPayload {
    pub certification_id: String,
    pub task_id: String,
    pub feature_id: String,
    pub backend: String,
    pub target: String,
    pub observed_at_utc: String,
    pub environment: CertificationEnvironment,
    pub device: DeviceEvidence,
    pub contract: ContractEvidence,
    pub matrix: Vec<CertificationMatrixRow>,
    pub provenance: Vec<CertificationProvenance>,
    pub conclusion: CertificationStatus,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationSignature {
    pub algorithm: String,
    pub domain: String,
    pub signer: String,
    pub public_key: String,
    pub signature: String,
    pub payload_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedDeviceCertification {
    pub schema_version: u16,
    pub payload: CertificationPayload,
    pub attestation: CertificationSignature,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceCertificationTrustAnchor {
    signer: String,
    public_key: [u8; 32],
}

impl DeviceCertificationTrustAnchor {
    pub fn from_hex(
        signer: impl Into<String>,
        public_key: &str,
    ) -> Result<Self, CertificationArtifactError> {
        let signer = signer.into();
        validate_identifier(&signer, "approved attestation signer")?;
        let public_key = decode_hex(public_key).ok_or_else(|| {
            CertificationArtifactError::InvalidContract(
                "approved attestation public key is not 32-byte lowercase hexadecimal".to_owned(),
            )
        })?;
        Ok(Self { signer, public_key })
    }

    pub fn signer(&self) -> &str {
        &self.signer
    }

    pub fn public_key_hex(&self) -> String {
        encode_hex(&self.public_key)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertificationImplementationManifest {
    digest: String,
    provenance: Vec<CertificationProvenance>,
}

impl CertificationImplementationManifest {
    pub fn from_relative_paths(
        workspace: &Path,
        paths: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, CertificationImplementationManifestError> {
        validate_absolute_root(workspace)?;
        let mut paths = paths.into_iter().collect::<Vec<_>>();
        paths.sort();
        if paths.is_empty() || paths.len() > MAX_IMPLEMENTATION_FILES {
            return Err(CertificationImplementationManifestError::InvalidContract(
                "implementation manifest file count is outside the supported bound".to_owned(),
            ));
        }
        if paths.windows(2).any(|window| window[0] == window[1]) {
            return Err(CertificationImplementationManifestError::InvalidContract(
                "implementation manifest paths must be unique".to_owned(),
            ));
        }

        let mut provenance = Vec::new();
        provenance.try_reserve_exact(paths.len()).map_err(|_| {
            CertificationImplementationManifestError::InvalidContract(
                "implementation manifest allocation exceeds the supported bound".to_owned(),
            )
        })?;
        for relative_path in paths {
            validate_relative_source_path(&relative_path)?;
            let path = workspace.join(&relative_path);
            let bytes = read_stable_regular_file(&path, MAX_IMPLEMENTATION_FILE_BYTES)
                .map_err(|error| manifest_file_error(&relative_path, error))?;
            provenance.push(CertificationProvenance {
                path: relative_path
                    .to_str()
                    .ok_or_else(|| {
                        CertificationImplementationManifestError::InvalidContract(
                            "implementation manifest path is not UTF-8".to_owned(),
                        )
                    })?
                    .to_owned(),
                sha256: sha256_hex(&bytes),
            });
        }
        let digest = implementation_manifest_digest(&provenance)?;
        Ok(Self { digest, provenance })
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn provenance(&self) -> &[CertificationProvenance] {
        &self.provenance
    }

    pub fn into_provenance(self) -> Vec<CertificationProvenance> {
        self.provenance
    }
}

#[derive(Debug, Error)]
pub enum CertificationImplementationManifestError {
    #[error("device certification implementation manifest is invalid: {0}")]
    InvalidContract(String),
    #[error("read implementation source {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

#[derive(Debug, Error)]
pub enum CertificationSigningKeyError {
    #[error("device certification signing-key path is invalid: {0}")]
    InvalidPath(String),
    #[error("read device certification signing key: {0}")]
    Read(#[source] io::Error),
    #[error("device certification signing key is not valid Ed25519 PKCS#8: {0}")]
    InvalidKey(String),
}

#[derive(Debug, Error)]
pub enum CertificationArtifactError {
    #[error("device certification artifact exceeds its byte bound")]
    TooLarge,
    #[error("device certification artifact is not canonical strict JSON: {0}")]
    InvalidJson(String),
    #[error("device certification artifact contract is invalid: {0}")]
    InvalidContract(String),
    #[error("device certification artifact signature is invalid")]
    InvalidSignature,
    #[error("device certification artifact signer is not approved by the configured trust anchor")]
    UntrustedSigner,
}

pub struct CertificationArtifact;

impl CertificationArtifact {
    pub fn sign(
        payload: CertificationPayload,
        trust_anchor: &DeviceCertificationTrustAnchor,
        key_pair: &Ed25519KeyPair,
    ) -> Result<SignedDeviceCertification, CertificationArtifactError> {
        validate_payload(&payload)?;
        if key_pair.public_key().as_ref() != trust_anchor.public_key {
            return Err(CertificationArtifactError::UntrustedSigner);
        }
        let payload_bytes = canonical_payload_bytes(&payload)?;
        let signing_bytes = signing_bytes(&payload_bytes)?;
        let public_key = encode_hex(key_pair.public_key().as_ref());
        let signature = encode_hex(key_pair.sign(&signing_bytes).as_ref());
        let artifact = SignedDeviceCertification {
            schema_version: 2,
            payload,
            attestation: CertificationSignature {
                algorithm: "ed25519".to_owned(),
                domain: "zed-sim-device-certification-v2".to_owned(),
                signer: trust_anchor.signer.clone(),
                public_key,
                signature,
                payload_sha256: sha256_hex(&payload_bytes),
            },
        };
        validate_artifact(&artifact)?;
        Ok(artifact)
    }

    pub fn parse_and_verify(
        bytes: &[u8],
        trust_anchor: &DeviceCertificationTrustAnchor,
    ) -> Result<SignedDeviceCertification, CertificationArtifactError> {
        if bytes.is_empty() || bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CertificationArtifactError::TooLarge);
        }
        let artifact: SignedDeviceCertification = serde_json::from_slice(bytes)
            .map_err(|error| CertificationArtifactError::InvalidJson(error.to_string()))?;
        let canonical = Self::to_canonical_json(&artifact)?;
        if canonical != bytes {
            return Err(CertificationArtifactError::InvalidJson(
                "artifact is not canonical pretty-json-v1 or contains duplicate fields".to_owned(),
            ));
        }
        validate_artifact(&artifact)?;
        let public_key = decode_hex::<32>(&artifact.attestation.public_key)
            .ok_or(CertificationArtifactError::InvalidSignature)?;
        if artifact.attestation.signer != trust_anchor.signer
            || public_key != trust_anchor.public_key
        {
            return Err(CertificationArtifactError::UntrustedSigner);
        }
        let payload_bytes = canonical_payload_bytes(&artifact.payload)?;
        if sha256_hex(&payload_bytes) != artifact.attestation.payload_sha256 {
            return Err(CertificationArtifactError::InvalidSignature);
        }
        let signature = decode_hex::<64>(&artifact.attestation.signature)
            .ok_or(CertificationArtifactError::InvalidSignature)?;
        let signing_bytes = signing_bytes(&payload_bytes)?;
        UnparsedPublicKey::new(&ED25519, public_key)
            .verify(&signing_bytes, &signature)
            .map_err(|_| CertificationArtifactError::InvalidSignature)?;
        Ok(artifact)
    }

    pub fn to_canonical_json(
        artifact: &SignedDeviceCertification,
    ) -> Result<Vec<u8>, CertificationArtifactError> {
        let mut bytes = serde_json::to_vec_pretty(artifact)
            .map_err(|error| CertificationArtifactError::InvalidJson(error.to_string()))?;
        bytes.push(b'\n');
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CertificationArtifactError::TooLarge);
        }
        Ok(bytes)
    }
}

pub fn load_device_certification_signing_key(
    path: &Path,
) -> Result<Ed25519KeyPair, CertificationSigningKeyError> {
    if !path.is_absolute() {
        return Err(CertificationSigningKeyError::InvalidPath(
            "the signing-key path must be absolute".to_owned(),
        ));
    }
    validate_private_key_permissions(path)?;
    let bytes =
        read_stable_regular_file(path, MAX_SIGNING_KEY_BYTES).map_err(|error| match error {
            StableFileReadError::Invalid(reason) => {
                CertificationSigningKeyError::InvalidPath(reason)
            }
            StableFileReadError::Read(source) => CertificationSigningKeyError::Read(source),
        })?;
    validate_private_key_permissions(path)?;
    Ed25519KeyPair::from_pkcs8(&bytes)
        .map_err(|error| CertificationSigningKeyError::InvalidKey(format!("{error:?}")))
}

pub fn cpu_certification_implementation_manifest(
    workspace: &Path,
) -> Result<CertificationImplementationManifest, CertificationImplementationManifestError> {
    let mut paths = CPU_IMPLEMENTATION_FILES
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    for tree in CPU_IMPLEMENTATION_TREES {
        collect_rust_sources(workspace, Path::new(tree), &mut paths)?;
    }
    CertificationImplementationManifest::from_relative_paths(workspace, paths)
}

fn collect_rust_sources(
    workspace: &Path,
    relative_directory: &Path,
    paths: &mut Vec<PathBuf>,
) -> Result<(), CertificationImplementationManifestError> {
    validate_relative_source_path(relative_directory)?;
    let directory = workspace.join(relative_directory);
    reject_symbolic_link_components(&directory)
        .map_err(|error| manifest_file_error(relative_directory, error))?;
    let before = fs::symlink_metadata(&directory).map_err(|source| {
        CertificationImplementationManifestError::Read {
            path: relative_directory.to_owned(),
            source,
        }
    })?;
    if before.file_type().is_symlink() || !before.is_dir() {
        return Err(CertificationImplementationManifestError::InvalidContract(
            format!(
                "implementation source tree {} must be a regular directory",
                relative_directory.display()
            ),
        ));
    }
    let entries = fs::read_dir(&directory).map_err(|source| {
        CertificationImplementationManifestError::Read {
            path: relative_directory.to_owned(),
            source,
        }
    })?;
    let mut entries = entries.collect::<Result<Vec<_>, _>>().map_err(|source| {
        CertificationImplementationManifestError::Read {
            path: relative_directory.to_owned(),
            source,
        }
    })?;
    if entries.len() > MAX_IMPLEMENTATION_TREE_ENTRIES {
        return Err(CertificationImplementationManifestError::InvalidContract(
            format!(
                "implementation source tree {} exceeds the entry bound",
                relative_directory.display()
            ),
        ));
    }
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let file_name = entry.file_name();
        let component = file_name.to_str().ok_or_else(|| {
            CertificationImplementationManifestError::InvalidContract(format!(
                "implementation source tree {} contains a non-UTF-8 entry",
                relative_directory.display()
            ))
        })?;
        validate_source_component(component, relative_directory)?;
        let relative_path = relative_directory.join(file_name);
        let path = workspace.join(&relative_path);
        reject_symbolic_link_components(&path)
            .map_err(|error| manifest_file_error(&relative_path, error))?;
        let metadata = fs::symlink_metadata(&path).map_err(|source| {
            CertificationImplementationManifestError::Read {
                path: relative_path.clone(),
                source,
            }
        })?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(CertificationImplementationManifestError::InvalidContract(
                format!(
                    "implementation source tree contains symlink {}",
                    relative_path.display()
                ),
            ));
        }
        if file_type.is_dir() {
            collect_rust_sources(workspace, &relative_path, paths)?;
        } else if file_type.is_file()
            && relative_path
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("rs")
        {
            if !include_cpu_tree_source(&relative_path)? {
                continue;
            }
            if paths.len() >= MAX_IMPLEMENTATION_FILES {
                return Err(CertificationImplementationManifestError::InvalidContract(
                    "implementation manifest file count is outside the supported bound".to_owned(),
                ));
            }
            paths.push(relative_path);
        } else if file_type.is_file() {
            return Err(CertificationImplementationManifestError::InvalidContract(
                format!(
                    "implementation source tree contains an unclassified regular file {}",
                    relative_path.display()
                ),
            ));
        } else {
            return Err(CertificationImplementationManifestError::InvalidContract(
                format!(
                    "implementation source tree contains a non-regular entry {}",
                    relative_path.display()
                ),
            ));
        }
    }
    reject_symbolic_link_components(&directory)
        .map_err(|error| manifest_file_error(relative_directory, error))?;
    let after = fs::symlink_metadata(&directory).map_err(|source| {
        CertificationImplementationManifestError::Read {
            path: relative_directory.to_owned(),
            source,
        }
    })?;
    if !same_file_identity(&before, &after) {
        return Err(CertificationImplementationManifestError::InvalidContract(
            format!(
                "implementation source tree {} changed while it was scanned",
                relative_directory.display()
            ),
        ));
    }
    Ok(())
}

fn include_cpu_tree_source(
    relative_path: &Path,
) -> Result<bool, CertificationImplementationManifestError> {
    if relative_path.parent() != Some(Path::new(CPU_OPERATION_TREE)) {
        return Ok(true);
    }
    let file_name = relative_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            CertificationImplementationManifestError::InvalidContract(
                "CPU operation source has no canonical UTF-8 filename".to_owned(),
            )
        })?;
    if file_name == CPU_BACKEND_OPERATION_MODULE {
        return Ok(true);
    }
    if file_name.starts_with("backend_") {
        return Ok(false);
    }
    if file_name.starts_with("backend") {
        return Err(CertificationImplementationManifestError::InvalidContract(
            format!(
                "CPU operation source {} has an ambiguous backend classification",
                relative_path.display()
            ),
        ));
    }
    Ok(true)
}

fn validate_relative_source_path(
    path: &Path,
) -> Result<(), CertificationImplementationManifestError> {
    let path_text = path.to_str().ok_or_else(|| {
        CertificationImplementationManifestError::InvalidContract(
            "implementation manifest path is not UTF-8".to_owned(),
        )
    })?;
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path_text.len() > MAX_IMPLEMENTATION_PATH_BYTES
        || path.components().any(|component| match component {
            Component::Normal(value) => value.to_str().is_none_or(|value| {
                value.is_empty() || value.len() > MAX_IMPLEMENTATION_COMPONENT_BYTES
            }),
            _ => true,
        })
    {
        return Err(CertificationImplementationManifestError::InvalidContract(
            format!(
                "implementation source path {} is not a canonical relative path",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn validate_absolute_root(
    workspace: &Path,
) -> Result<(), CertificationImplementationManifestError> {
    if !workspace.is_absolute() {
        return Err(CertificationImplementationManifestError::InvalidContract(
            "implementation manifest workspace root must be absolute".to_owned(),
        ));
    }
    reject_symbolic_link_components(workspace)
        .map_err(|error| manifest_file_error(Path::new("."), error))?;
    let metadata = fs::symlink_metadata(workspace).map_err(|source| {
        CertificationImplementationManifestError::Read {
            path: PathBuf::from("."),
            source,
        }
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(CertificationImplementationManifestError::InvalidContract(
            "implementation manifest workspace root must be a non-symlink directory".to_owned(),
        ));
    }
    Ok(())
}

fn validate_source_component(
    component: &str,
    parent: &Path,
) -> Result<(), CertificationImplementationManifestError> {
    if component.is_empty()
        || component == "."
        || component == ".."
        || component.len() > MAX_IMPLEMENTATION_COMPONENT_BYTES
        || component.contains(['/', '\\'])
    {
        return Err(CertificationImplementationManifestError::InvalidContract(
            format!(
                "implementation source tree {} contains an unsafe path component",
                parent.display()
            ),
        ));
    }
    Ok(())
}

#[derive(Debug)]
enum StableFileReadError {
    Invalid(String),
    Read(io::Error),
}

fn manifest_file_error(
    path: &Path,
    error: StableFileReadError,
) -> CertificationImplementationManifestError {
    match error {
        StableFileReadError::Invalid(reason) => {
            CertificationImplementationManifestError::InvalidContract(format!(
                "implementation source {} {reason}",
                path.display()
            ))
        }
        StableFileReadError::Read(source) => CertificationImplementationManifestError::Read {
            path: path.to_owned(),
            source,
        },
    }
}

fn read_stable_regular_file(
    path: &Path,
    maximum_bytes: u64,
) -> Result<Vec<u8>, StableFileReadError> {
    read_stable_regular_file_with_hook(path, maximum_bytes, || Ok(()))
}

fn read_stable_regular_file_with_hook(
    path: &Path,
    maximum_bytes: u64,
    after_first_read: impl FnOnce() -> io::Result<()>,
) -> Result<Vec<u8>, StableFileReadError> {
    reject_symbolic_link_components(path)?;
    let before = fs::symlink_metadata(path).map_err(StableFileReadError::Read)?;
    validate_bounded_regular_file(&before, maximum_bytes)?;
    let mut file = File::open(path).map_err(StableFileReadError::Read)?;
    let opened = file.metadata().map_err(StableFileReadError::Read)?;
    if !same_file_identity(&before, &opened) {
        return Err(StableFileReadError::Invalid(
            "changed while it was opened".to_owned(),
        ));
    }
    let first = read_bounded(&mut file, maximum_bytes)?;
    after_first_read().map_err(StableFileReadError::Read)?;
    file.rewind().map_err(StableFileReadError::Read)?;
    let second = read_bounded(&mut file, maximum_bytes)?;
    let after = file.metadata().map_err(StableFileReadError::Read)?;
    reject_symbolic_link_components(path)?;
    let current = fs::symlink_metadata(path).map_err(StableFileReadError::Read)?;
    if u64::try_from(first.len()).ok() != Some(opened.len())
        || u64::try_from(second.len()).ok() != Some(opened.len())
        || first != second
        || !same_file_identity(&before, &after)
        || !same_file_identity(&after, &current)
    {
        return Err(StableFileReadError::Invalid(
            "changed while it was read".to_owned(),
        ));
    }
    Ok(first)
}

fn read_bounded(file: &mut File, maximum_bytes: u64) -> Result<Vec<u8>, StableFileReadError> {
    let limit = maximum_bytes
        .checked_add(1)
        .ok_or_else(|| StableFileReadError::Invalid("has an invalid byte bound".to_owned()))?;
    let mut bytes = Vec::new();
    let initial_capacity = usize::try_from(maximum_bytes.min(64 * 1024)).map_err(|_| {
        StableFileReadError::Invalid("has a byte bound that exceeds this platform".to_owned())
    })?;
    bytes.try_reserve_exact(initial_capacity).map_err(|_| {
        StableFileReadError::Invalid("could not allocate its bounded read buffer".to_owned())
    })?;
    file.take(limit)
        .read_to_end(&mut bytes)
        .map_err(StableFileReadError::Read)?;
    if bytes.is_empty()
        || u64::try_from(bytes.len())
            .ok()
            .is_none_or(|length| length > maximum_bytes)
    {
        return Err(StableFileReadError::Invalid(
            "must be a nonempty bounded regular file".to_owned(),
        ));
    }
    Ok(bytes)
}

fn validate_bounded_regular_file(
    metadata: &Metadata,
    maximum_bytes: u64,
) -> Result<(), StableFileReadError> {
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > maximum_bytes
    {
        return Err(StableFileReadError::Invalid(
            "must be a nonempty bounded regular file".to_owned(),
        ));
    }
    Ok(())
}

fn validate_private_key_permissions(path: &Path) -> Result<(), CertificationSigningKeyError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let metadata = fs::symlink_metadata(path).map_err(CertificationSigningKeyError::Read)?;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(CertificationSigningKeyError::InvalidPath(
                "the signing-key file must not grant group or world permissions".to_owned(),
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn reject_symbolic_link_components(path: &Path) -> Result<(), StableFileReadError> {
    if !path.is_absolute() {
        return Err(StableFileReadError::Invalid(
            "must use an absolute path".to_owned(),
        ));
    }
    let path_text = path.to_str().ok_or_else(|| {
        StableFileReadError::Invalid("contains a non-UTF-8 path component".to_owned())
    })?;
    if path_text.len() > MAX_IMPLEMENTATION_PATH_BYTES {
        return Err(StableFileReadError::Invalid(
            "exceeds the path byte bound".to_owned(),
        ));
    }
    let components = path.components().collect::<Vec<_>>();
    if components.is_empty() {
        return Err(StableFileReadError::Invalid(
            "has no path components".to_owned(),
        ));
    }
    let mut current = PathBuf::new();
    for (index, component) in components.iter().enumerate() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(component.as_os_str()),
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| {
                    StableFileReadError::Invalid("contains a non-UTF-8 path component".to_owned())
                })?;
                if value.is_empty()
                    || value.len() > MAX_IMPLEMENTATION_COMPONENT_BYTES
                    || value.contains(['/', '\\'])
                {
                    return Err(StableFileReadError::Invalid(
                        "contains an unsafe path component".to_owned(),
                    ));
                }
                current.push(value);
                let metadata = fs::symlink_metadata(&current).map_err(StableFileReadError::Read)?;
                if metadata.file_type().is_symlink() {
                    return Err(StableFileReadError::Invalid(
                        "contains a symbolic-link path component".to_owned(),
                    ));
                }
                if index + 1 < components.len() && !metadata.is_dir() {
                    return Err(StableFileReadError::Invalid(
                        "contains a non-directory intermediate component".to_owned(),
                    ));
                }
            }
            Component::CurDir | Component::ParentDir => {
                return Err(StableFileReadError::Invalid(
                    "contains a non-canonical path component".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn same_file_identity(left: &Metadata, right: &Metadata) -> bool {
    if left.len() != right.len()
        || left.modified().ok() != right.modified().ok()
        || left.file_type() != right.file_type()
    {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        left.dev() == right.dev() && left.ino() == right.ino() && left.mode() == right.mode()
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::fs::MetadataExt;
        matches!(
            (
                left.volume_serial_number(),
                left.file_index(),
                right.volume_serial_number(),
                right.file_index(),
            ),
            (Some(left_volume), Some(left_index), Some(right_volume), Some(right_index))
                if left_volume == right_volume && left_index == right_index
        )
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        false
    }
}

fn implementation_manifest_digest(
    provenance: &[CertificationProvenance],
) -> Result<String, CertificationImplementationManifestError> {
    let mut hasher = Sha256::new();
    hasher.update(IMPLEMENTATION_MANIFEST_DOMAIN);
    hasher.update(
        u64::try_from(provenance.len())
            .map_err(|_| {
                CertificationImplementationManifestError::InvalidContract(
                    "implementation manifest row count overflowed".to_owned(),
                )
            })?
            .to_be_bytes(),
    );
    for row in provenance {
        let path_length = u64::try_from(row.path.len()).map_err(|_| {
            CertificationImplementationManifestError::InvalidContract(
                "implementation manifest path length overflowed".to_owned(),
            )
        })?;
        hasher.update(path_length.to_be_bytes());
        hasher.update(row.path.as_bytes());
        let digest = decode_hex::<32>(&row.sha256).ok_or_else(|| {
            CertificationImplementationManifestError::InvalidContract(
                "implementation manifest contains an invalid digest".to_owned(),
            )
        })?;
        hasher.update(digest);
    }
    Ok(encode_hex(&hasher.finalize()))
}

fn validate_artifact(
    artifact: &SignedDeviceCertification,
) -> Result<(), CertificationArtifactError> {
    if artifact.schema_version != 2
        || artifact.attestation.algorithm != "ed25519"
        || artifact.attestation.domain != "zed-sim-device-certification-v2"
    {
        return Err(CertificationArtifactError::InvalidContract(
            "unsupported schema, algorithm, or signature domain".to_owned(),
        ));
    }
    validate_identifier(&artifact.attestation.signer, "attestation signer")?;
    validate_hex(
        &artifact.attestation.public_key,
        64,
        "attestation public key",
    )?;
    validate_hex(
        &artifact.attestation.signature,
        128,
        "attestation signature",
    )?;
    validate_digest(
        &artifact.attestation.payload_sha256,
        "attestation payload digest",
    )?;
    validate_payload(&artifact.payload)
}

fn validate_payload(payload: &CertificationPayload) -> Result<(), CertificationArtifactError> {
    validate_identifier(&payload.certification_id, "certification ID")?;
    validate_identifier(&payload.task_id, "task ID")?;
    validate_identifier(&payload.feature_id, "feature ID")?;
    validate_identifier(&payload.backend, "backend")?;
    validate_text(&payload.target, "target")?;
    validate_text(&payload.observed_at_utc, "observation time")?;
    for (value, label) in [
        (&payload.environment.lab_id, "lab ID"),
        (&payload.environment.hostname, "lab hostname"),
        (&payload.environment.os_name, "operating-system name"),
        (&payload.environment.os_version, "operating-system version"),
        (&payload.environment.os_build, "operating-system build"),
        (&payload.environment.architecture, "architecture"),
        (&payload.environment.rust_target, "Rust target"),
        (&payload.device.name, "device name"),
        (&payload.device.identifier, "device identifier"),
        (&payload.device.memory_model, "device memory model"),
        (&payload.contract.abi_floor, "ABI floor"),
    ] {
        validate_text(value, label)?;
    }
    validate_facts(&payload.environment.toolchain, "toolchain")?;
    validate_identifiers(&payload.device.observed_features, "observed device feature")?;
    validate_memory_facts(&payload.device.memory)?;
    let mut prior_symbol: Option<&str> = None;
    for symbol in &payload.contract.symbols {
        validate_text(symbol, "contract symbol")?;
        if prior_symbol.is_some_and(|prior| prior >= symbol.as_str()) {
            return invalid_contract("contract symbols must be unique and ascending");
        }
        prior_symbol = Some(symbol);
    }
    if payload.matrix.is_empty() || payload.matrix.len() > MAX_MATRIX_ROWS {
        return invalid_contract("matrix row count is outside the supported bound");
    }
    if payload.provenance.is_empty() || payload.provenance.len() > MAX_PROVENANCE_ROWS {
        return invalid_contract("provenance row count is outside the supported bound");
    }
    let mut prior_matrix_id: Option<&str> = None;
    let mut matrix_ids = BTreeSet::new();
    for row in &payload.matrix {
        validate_identifier(&row.id, "matrix row ID")?;
        validate_text(&row.category, "matrix category")?;
        validate_text(&row.operation, "matrix operation")?;
        validate_text(&row.tolerance, "matrix tolerance")?;
        validate_text(&row.evidence, "matrix evidence")?;
        if let Some(dtype) = &row.dtype {
            validate_text(dtype, "matrix dtype")?;
        }
        if let Some(layout) = &row.layout {
            validate_text(layout, "matrix layout")?;
        }
        if prior_matrix_id.is_some_and(|prior| prior >= row.id.as_str())
            || !matrix_ids.insert(row.id.as_str())
        {
            return invalid_contract("matrix rows must have unique ascending IDs");
        }
        prior_matrix_id = Some(&row.id);
    }
    let mut prior_path: Option<&str> = None;
    let mut paths = BTreeSet::new();
    for row in &payload.provenance {
        validate_text(&row.path, "provenance path")?;
        validate_digest(&row.sha256, "provenance digest")?;
        if prior_path.is_some_and(|prior| prior >= row.path.as_str())
            || !paths.insert(row.path.as_str())
        {
            return invalid_contract("provenance rows must have unique ascending paths");
        }
        prior_path = Some(&row.path);
    }
    for (digest, label) in [
        (&payload.contract.abi_contract_sha256, "ABI contract digest"),
        (&payload.contract.abi_manifest_sha256, "ABI manifest digest"),
        (
            &payload.contract.execution_abi_sha256,
            "execution ABI digest",
        ),
    ] {
        validate_digest(digest, label)?;
    }
    match &payload.contract.package {
        CertificationPackageEvidence::Signed(package) => {
            if payload.backend == "cpu" {
                return invalid_contract(
                    "CPU certification must not fabricate vendor package evidence",
                );
            }
            validate_text(&package.format, "package format")?;
            validate_identifier(&package.signer, "package signer")?;
            validate_hex(&package.signer_public_key, 64, "package public key")?;
            validate_hex(&package.signature, 128, "package signature")?;
            validate_digest(&package.coverage_sha256, "package coverage digest")?;
            validate_digest(&package.manifest_sha256, "package manifest digest")?;
            validate_digest(
                &package.contract_catalog_sha256,
                "package contract catalog digest",
            )?;
            validate_provenance(&package.payloads, "package payload")?;
        }
        CertificationPackageEvidence::NotApplicable { reason } => {
            if payload.backend != "cpu" {
                return invalid_contract(
                    "accelerator certification requires signed vendor package evidence",
                );
            }
            validate_text(reason, "package-not-applicable reason")?;
        }
    }
    let has_failure = payload
        .matrix
        .iter()
        .any(|row| row.status == CertificationStatus::Failure);
    if has_failure != (payload.conclusion == CertificationStatus::Failure)
        || payload.conclusion == CertificationStatus::Unsupported
    {
        return invalid_contract("conclusion does not match the recorded matrix statuses");
    }
    Ok(())
}

fn canonical_payload_bytes(
    payload: &CertificationPayload,
) -> Result<Vec<u8>, CertificationArtifactError> {
    serde_json::to_vec(payload)
        .map_err(|error| CertificationArtifactError::InvalidJson(error.to_string()))
}

fn signing_bytes(payload: &[u8]) -> Result<Vec<u8>, CertificationArtifactError> {
    let payload_length =
        u64::try_from(payload.len()).map_err(|_| CertificationArtifactError::TooLarge)?;
    let capacity = SIGNATURE_DOMAIN
        .len()
        .checked_add(8)
        .and_then(|length| length.checked_add(payload.len()))
        .ok_or(CertificationArtifactError::TooLarge)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| CertificationArtifactError::TooLarge)?;
    bytes.extend_from_slice(SIGNATURE_DOMAIN);
    bytes.extend_from_slice(&payload_length.to_be_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

fn validate_identifier(value: &str, label: &str) -> Result<(), CertificationArtifactError> {
    if value.is_empty()
        || value.len() > 256
        || value.starts_with(['.', '-', '_'])
        || value.ends_with(['.', '-', '_'])
        || value.contains("..")
        || value.contains("--")
        || value.contains("__")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return invalid_contract(&format!("{label} is not a canonical identifier"));
    }
    Ok(())
}

fn validate_identifiers(values: &[String], label: &str) -> Result<(), CertificationArtifactError> {
    if values.is_empty() || values.len() > MAX_FACT_ROWS {
        return invalid_contract(&format!("{label} rows are outside the supported bound"));
    }
    let mut prior: Option<&str> = None;
    for value in values {
        validate_identifier(value, label)?;
        if prior.is_some_and(|prior| prior >= value.as_str()) {
            return invalid_contract(&format!("{label} rows must be unique and ascending"));
        }
        prior = Some(value);
    }
    Ok(())
}

fn validate_facts(
    facts: &[CertificationFact],
    label: &str,
) -> Result<(), CertificationArtifactError> {
    if facts.is_empty() || facts.len() > MAX_FACT_ROWS {
        return invalid_contract(&format!("{label} rows are outside the supported bound"));
    }
    let mut prior: Option<&str> = None;
    for fact in facts {
        validate_identifier(&fact.name, label)?;
        validate_text(&fact.value, label)?;
        if prior.is_some_and(|prior| prior >= fact.name.as_str()) {
            return invalid_contract(&format!("{label} rows must be unique and ascending"));
        }
        prior = Some(&fact.name);
    }
    Ok(())
}

fn validate_memory_facts(
    facts: &[CertificationMemoryFact],
) -> Result<(), CertificationArtifactError> {
    if facts.is_empty() || facts.len() > MAX_FACT_ROWS {
        return invalid_contract("memory observations are outside the supported bound");
    }
    let mut prior: Option<&str> = None;
    for fact in facts {
        validate_identifier(&fact.name, "memory observation")?;
        if fact.bytes == 0 {
            return invalid_contract("memory observations must be non-zero");
        }
        if prior.is_some_and(|prior| prior >= fact.name.as_str()) {
            return invalid_contract("memory observations must be unique and ascending");
        }
        prior = Some(&fact.name);
    }
    Ok(())
}

fn validate_provenance(
    rows: &[CertificationProvenance],
    label: &str,
) -> Result<(), CertificationArtifactError> {
    if rows.is_empty() || rows.len() > MAX_PROVENANCE_ROWS {
        return invalid_contract(&format!("{label} rows are outside the supported bound"));
    }
    let mut prior: Option<&str> = None;
    for row in rows {
        validate_text(&row.path, label)?;
        validate_digest(&row.sha256, label)?;
        if prior.is_some_and(|prior| prior >= row.path.as_str()) {
            return invalid_contract(&format!("{label} rows must be unique and ascending"));
        }
        prior = Some(&row.path);
    }
    Ok(())
}

fn validate_text(value: &str, label: &str) -> Result<(), CertificationArtifactError> {
    if value.is_empty()
        || value.len() > 4096
        || value != value.trim()
        || value.chars().any(char::is_control)
    {
        return invalid_contract(&format!("{label} is empty, oversized, or non-canonical"));
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), CertificationArtifactError> {
    validate_hex(value, 64, label)
}

fn validate_hex(
    value: &str,
    expected_length: usize,
    label: &str,
) -> Result<(), CertificationArtifactError> {
    if value.len() != expected_length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid_contract(&format!("{label} is not lowercase hexadecimal"));
    }
    Ok(())
}

fn decode_hex<const LENGTH: usize>(value: &str) -> Option<[u8; LENGTH]> {
    if value.len() != LENGTH.checked_mul(2)? {
        return None;
    }
    let mut output = [0_u8; LENGTH];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_nibble(*pair.first()?)?;
        let low = decode_nibble(*pair.get(1)?)?;
        *output.get_mut(index)? = (high << 4) | low;
    }
    Some(output)
}

fn decode_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    encode_hex(&Sha256::digest(bytes))
}

fn invalid_contract<T>(reason: &str) -> Result<T, CertificationArtifactError> {
    Err(CertificationArtifactError::InvalidContract(
        reason.to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_certification_target_requires_explicit_development_feature() {
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains("default = [\"cpu\"]"));
        assert!(manifest.contains("hardware-certification = []"));
        assert!(manifest.contains("name = \"device_cpu_comfy_model_0016\""));
        assert!(manifest.contains("required-features = [\"hardware-certification\"]"));
    }

    fn payload() -> CertificationPayload {
        CertificationPayload {
            certification_id: "device-test-0001".to_owned(),
            task_id: "comfy-parity-certify-device-test-0001".to_owned(),
            feature_id: "COMFY-MODEL-0001".to_owned(),
            backend: "test".to_owned(),
            target: "aarch64-apple-darwin".to_owned(),
            observed_at_utc: "2026-07-26T00:00:00Z".to_owned(),
            environment: CertificationEnvironment {
                lab_id: "zed-test-lab".to_owned(),
                hostname: "test-host".to_owned(),
                os_name: "test-os".to_owned(),
                os_version: "1.0".to_owned(),
                os_build: "1A1".to_owned(),
                architecture: "aarch64".to_owned(),
                rust_target: "aarch64-apple-darwin".to_owned(),
                toolchain: vec![CertificationFact {
                    name: "sdk".to_owned(),
                    value: "test-sdk-1".to_owned(),
                }],
            },
            device: DeviceEvidence {
                name: "Test device".to_owned(),
                identifier: "test-device-1".to_owned(),
                memory_model: "unified".to_owned(),
                observed_features: vec!["test-capability".to_owned()],
                memory: vec![
                    CertificationMemoryFact {
                        name: "host-physical-memory".to_owned(),
                        bytes: 1024,
                    },
                    CertificationMemoryFact {
                        name: "recommended-working-set".to_owned(),
                        bytes: 512,
                    },
                ],
            },
            contract: ContractEvidence {
                abi_contract_sha256: "1".repeat(64),
                abi_manifest_sha256: "2".repeat(64),
                execution_abi_sha256: "3".repeat(64),
                abi_floor: "test-abi-1".to_owned(),
                framework_count: 1,
                symbol_count: 1,
                class_count: 0,
                selector_count: 0,
                symbols: vec!["test_symbol".to_owned()],
                package: CertificationPackageEvidence::Signed(PackageEvidence {
                    format: "test-package-v1".to_owned(),
                    signer: "test.package.signer".to_owned(),
                    signer_public_key: "4".repeat(64),
                    signature: "5".repeat(128),
                    coverage_sha256: "6".repeat(64),
                    manifest_sha256: "7".repeat(64),
                    contract_catalog_sha256: "8".repeat(64),
                    payloads: vec![CertificationProvenance {
                        path: "payload.bin".to_owned(),
                        sha256: "9".repeat(64),
                    }],
                }),
            },
            matrix: vec![CertificationMatrixRow {
                id: "001-test".to_owned(),
                category: "contract".to_owned(),
                operation: "test".to_owned(),
                dtype: None,
                layout: None,
                status: CertificationStatus::Pass,
                tolerance: "exact".to_owned(),
                evidence: "executed".to_owned(),
            }],
            provenance: vec![CertificationProvenance {
                path: "test/contract.json".to_owned(),
                sha256: "b".repeat(64),
            }],
            conclusion: CertificationStatus::Pass,
        }
    }

    fn key_pair() -> Result<Ed25519KeyPair, CertificationArtifactError> {
        Ed25519KeyPair::from_seed_unchecked(b"0123456789abcdef0123456789abcdef")
            .map_err(|_| CertificationArtifactError::InvalidSignature)
    }

    fn other_key_pair() -> Result<Ed25519KeyPair, CertificationArtifactError> {
        Ed25519KeyPair::from_seed_unchecked(b"abcdef0123456789abcdef0123456789")
            .map_err(|_| CertificationArtifactError::InvalidSignature)
    }

    fn fixed_pkcs8() -> Result<Vec<u8>, CertificationArtifactError> {
        let mut bytes = vec![
            0x30, 0x53, 0x02, 0x01, 0x01, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22,
            0x04, 0x20,
        ];
        bytes.extend_from_slice(b"0123456789abcdef0123456789abcdef");
        bytes.extend_from_slice(&[0xa1, 0x23, 0x03, 0x21, 0x00]);
        bytes.extend_from_slice(key_pair()?.public_key().as_ref());
        Ok(bytes)
    }

    fn restrict_key_permissions(path: &Path) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = fs::metadata(path)?.permissions();
            permissions.set_mode(0o600);
            fs::set_permissions(path, permissions)?;
        }
        #[cfg(not(unix))]
        let _ = path;
        Ok(())
    }

    fn trust_anchor(
        key_pair: &Ed25519KeyPair,
    ) -> Result<DeviceCertificationTrustAnchor, CertificationArtifactError> {
        DeviceCertificationTrustAnchor::from_hex(
            "test.lab.signer",
            &encode_hex(key_pair.public_key().as_ref()),
        )
    }

    #[test]
    fn canonical_signed_artifact_round_trips_and_rejects_tampering()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = key_pair()?;
        let trust_anchor = trust_anchor(&key_pair)?;
        let artifact = CertificationArtifact::sign(payload(), &trust_anchor, &key_pair)?;
        let bytes = CertificationArtifact::to_canonical_json(&artifact)?;
        assert_eq!(
            CertificationArtifact::parse_and_verify(&bytes, &trust_anchor)?,
            artifact
        );

        let mut tampered = bytes;
        let marker = b"Test device";
        let offset = tampered
            .windows(marker.len())
            .position(|window| window == marker)
            .ok_or("missing device marker")?;
        let replacement = tampered
            .get_mut(offset..offset + marker.len())
            .ok_or("invalid device marker range")?;
        replacement.copy_from_slice(b"Test Device");
        assert!(matches!(
            CertificationArtifact::parse_and_verify(&tampered, &trust_anchor),
            Err(CertificationArtifactError::InvalidSignature)
        ));
        Ok(())
    }

    #[test]
    fn artifact_rejects_self_signed_and_unknown_attestation_keys()
    -> Result<(), Box<dyn std::error::Error>> {
        let approved_key_pair = key_pair()?;
        let approved_anchor = trust_anchor(&approved_key_pair)?;
        let unknown_key_pair = other_key_pair()?;
        let unknown_anchor = DeviceCertificationTrustAnchor::from_hex(
            "test.lab.signer",
            &encode_hex(unknown_key_pair.public_key().as_ref()),
        )?;
        let self_signed =
            CertificationArtifact::sign(payload(), &unknown_anchor, &unknown_key_pair)?;
        let self_signed_bytes = CertificationArtifact::to_canonical_json(&self_signed)?;
        assert!(matches!(
            CertificationArtifact::parse_and_verify(&self_signed_bytes, &approved_anchor),
            Err(CertificationArtifactError::UntrustedSigner)
        ));

        let artifact =
            CertificationArtifact::sign(payload(), &approved_anchor, &approved_key_pair)?;
        let bytes = CertificationArtifact::to_canonical_json(&artifact)?;
        let unknown_signer_anchor = DeviceCertificationTrustAnchor::from_hex(
            "another.lab.signer",
            &approved_anchor.public_key_hex(),
        )?;
        assert!(matches!(
            CertificationArtifact::parse_and_verify(&bytes, &unknown_signer_anchor),
            Err(CertificationArtifactError::UntrustedSigner)
        ));
        assert!(matches!(
            CertificationArtifact::sign(payload(), &approved_anchor, &unknown_key_pair),
            Err(CertificationArtifactError::UntrustedSigner)
        ));
        Ok(())
    }

    #[test]
    fn artifact_rejects_unknown_fields_and_mismatched_conclusion()
    -> Result<(), Box<dyn std::error::Error>> {
        let key_pair = key_pair()?;
        let trust_anchor = trust_anchor(&key_pair)?;
        let artifact = CertificationArtifact::sign(payload(), &trust_anchor, &key_pair)?;
        let bytes = CertificationArtifact::to_canonical_json(&artifact)?;
        let unknown = String::from_utf8(bytes)?.replacen(
            "{\n  \"schema_version\"",
            "{\n  \"unknown\": true,\n  \"schema_version\"",
            1,
        );
        assert!(matches!(
            CertificationArtifact::parse_and_verify(unknown.as_bytes(), &trust_anchor),
            Err(CertificationArtifactError::InvalidJson(_))
        ));

        let mut invalid = payload();
        invalid
            .matrix
            .first_mut()
            .ok_or("missing matrix row")?
            .status = CertificationStatus::Failure;
        assert!(matches!(
            CertificationArtifact::sign(invalid, &trust_anchor, &key_pair),
            Err(CertificationArtifactError::InvalidContract(_))
        ));
        Ok(())
    }

    #[test]
    fn implementation_manifest_binds_omitted_and_changed_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        fs::create_dir(root.join("sources"))?;
        fs::write(root.join("sources/first.rs"), b"fn first() {}\n")?;
        fs::write(root.join("sources/second.rs"), b"fn second() {}\n")?;
        let both = CertificationImplementationManifest::from_relative_paths(
            &root,
            ["sources/first.rs", "sources/second.rs"]
                .into_iter()
                .map(PathBuf::from),
        )?;
        let omitted = CertificationImplementationManifest::from_relative_paths(
            &root,
            [PathBuf::from("sources/first.rs")],
        )?;
        assert_ne!(both.digest(), omitted.digest());
        assert_ne!(both.provenance(), omitted.provenance());

        fs::write(root.join("sources/second.rs"), b"fn second_changed() {}\n")?;
        let changed = CertificationImplementationManifest::from_relative_paths(
            &root,
            ["sources/first.rs", "sources/second.rs"]
                .into_iter()
                .map(PathBuf::from),
        )?;
        assert_ne!(both.digest(), changed.digest());
        assert_ne!(both.provenance(), changed.provenance());
        Ok(())
    }

    #[test]
    fn cpu_manifest_covers_the_exact_shared_and_cpu_source_closure()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()?;
        let manifest = cpu_certification_implementation_manifest(&workspace)?;
        let paths = manifest
            .provenance()
            .iter()
            .map(|row| row.path.as_str())
            .collect::<BTreeSet<_>>();
        for required in [
            ".cargo/config.toml",
            "Cargo.lock",
            "Cargo.toml",
            "crates/sim_tensor/Cargo.toml",
            "crates/sim_tensor/operation_contract_evidence.rs",
            "crates/sim_tensor/src/autograd.rs",
            "crates/sim_tensor/src/ops/backend_cpu_sim_model_0016.rs",
            "crates/sim_test_support/Cargo.toml",
            "crates/sim_test_support/src/sim_test_support.rs",
            "crates/sim_test_support/src/device_certification.rs",
            "crates/sim_test_support/tests/device_cpu_sim_model_0016.rs",
            "crates/sim_types/Cargo.toml",
            "crates/sim_types/src/cancellation.rs",
            "crates/sim_types/src/sim_types.rs",
            "crates/sim_types/src/worker_protocol.rs",
            "rust-toolchain.toml",
        ] {
            assert!(
                paths.contains(required),
                "missing CPU manifest source {required}"
            );
        }
        let included_backend_modules = paths
            .iter()
            .copied()
            .filter(|path| path.starts_with("crates/sim_tensor/src/ops/backend_"))
            .collect::<Vec<_>>();
        assert_eq!(
            included_backend_modules,
            vec!["crates/sim_tensor/src/ops/backend_cpu_sim_model_0016.rs"]
        );
        for excluded in [
            "backend_amd_rocm_sim_model_0014.rs",
            "backend_apple_metal_mps_sim_model_0015.rs",
            "backend_cambricon_mlu_sim_model_0017.rs",
            "backend_directml_sim_model_0018.rs",
            "backend_huawei_ascend_npu_sim_model_0019.rs",
        ] {
            assert!(!paths.contains(format!("crates/sim_tensor/src/ops/{excluded}").as_str()));
        }
        assert!(include_cpu_tree_source(Path::new(
            "crates/sim_tensor/src/ops/backend_cpu_sim_model_0016.rs"
        ))?);
        assert!(!include_cpu_tree_source(Path::new(
            "crates/sim_tensor/src/ops/backend_future_vendor.rs"
        ))?);
        assert!(include_cpu_tree_source(Path::new(
            "crates/sim_tensor/src/ops/reduction_01.rs"
        ))?);
        assert!(
            include_cpu_tree_source(Path::new("crates/sim_tensor/src/ops/backendambiguous.rs"))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn stable_source_read_rejects_same_length_mutation() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        let path = root.join("source.rs");
        fs::write(&path, b"fn first() {}\n")?;
        let result = read_stable_regular_file_with_hook(&path, 1024, || {
            fs::write(&path, b"fn other() {}\n")
        });
        assert!(
            matches!(result, Err(StableFileReadError::Invalid(reason)) if reason.contains("changed"))
        );
        Ok(())
    }

    #[test]
    fn manifest_tree_rejects_unclassified_regular_files() -> Result<(), Box<dyn std::error::Error>>
    {
        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        fs::create_dir(root.join("sources"))?;
        fs::write(root.join("sources/notes.txt"), b"unclassified input\n")?;
        assert!(matches!(
            collect_rust_sources(&root, Path::new("sources"), &mut Vec::new()),
            Err(CertificationImplementationManifestError::InvalidContract(reason))
                if reason.contains("unclassified regular file")
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn manifest_rejects_intermediate_symlinks_special_entries_and_non_utf8_entries()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::{ffi::CString, os::unix::ffi::OsStrExt, os::unix::fs::symlink};

        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        fs::create_dir(root.join("real"))?;
        fs::write(root.join("real/source.rs"), b"fn source() {}\n")?;
        symlink(root.join("real"), root.join("alias"))?;
        assert!(matches!(
            CertificationImplementationManifest::from_relative_paths(
                &root,
                [PathBuf::from("alias/source.rs")]
            ),
            Err(CertificationImplementationManifestError::InvalidContract(_))
        ));

        fs::create_dir(root.join("special"))?;
        let fifo_path = root.join("special/source.rs");
        let fifo_path_c = CString::new(fifo_path.as_os_str().as_bytes())?;
        // SAFETY: `fifo_path_c` is a live, NUL-terminated path and the mode contains only
        // permission bits. The temporary directory owns cleanup after the assertion.
        if unsafe { libc::mkfifo(fifo_path_c.as_ptr(), 0o600) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        assert!(matches!(
            collect_rust_sources(&root, Path::new("special"), &mut Vec::new()),
            Err(CertificationImplementationManifestError::InvalidContract(_))
        ));

        #[cfg(target_os = "linux")]
        {
            use std::{ffi::OsString, os::unix::ffi::OsStringExt};

            fs::create_dir(root.join("non-utf8"))?;
            let name = OsString::from_vec(vec![0xff, b'.', b'r', b's']);
            fs::write(root.join("non-utf8").join(name), b"fn source() {}\n")?;
            assert!(matches!(
                collect_rust_sources(&root, Path::new("non-utf8"), &mut Vec::new()),
                Err(CertificationImplementationManifestError::InvalidContract(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn signing_key_loader_accepts_only_bounded_absolute_regular_pkcs8_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        let key_path = root.join("key.pkcs8");
        fs::write(&key_path, fixed_pkcs8()?)?;
        restrict_key_permissions(&key_path)?;
        let loaded = load_device_certification_signing_key(&key_path)?;
        assert_eq!(
            loaded.public_key().as_ref(),
            key_pair()?.public_key().as_ref()
        );
        assert!(matches!(
            load_device_certification_signing_key(Path::new("relative-key.pkcs8")),
            Err(CertificationSigningKeyError::InvalidPath(_))
        ));
        let oversized = root.join("oversized.pkcs8");
        fs::write(
            &oversized,
            vec![1_u8; usize::try_from(MAX_SIGNING_KEY_BYTES)? + 1],
        )?;
        restrict_key_permissions(&oversized)?;
        assert!(matches!(
            load_device_certification_signing_key(&oversized),
            Err(CertificationSigningKeyError::InvalidPath(_))
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn signing_key_loader_rejects_symbolic_links() -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        let key_path = root.join("key.pkcs8");
        fs::write(&key_path, fixed_pkcs8()?)?;
        restrict_key_permissions(&key_path)?;
        let link_path = root.join("key-link.pkcs8");
        symlink(&key_path, &link_path)?;
        assert!(matches!(
            load_device_certification_signing_key(&link_path),
            Err(CertificationSigningKeyError::InvalidPath(_))
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn signing_key_loader_rejects_group_or_world_permissions()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir()?;
        let root = directory.path().canonicalize()?;
        let key_path = root.join("key.pkcs8");
        fs::write(&key_path, fixed_pkcs8()?)?;
        let mut permissions = fs::metadata(&key_path)?.permissions();
        permissions.set_mode(0o640);
        fs::set_permissions(&key_path, permissions)?;
        assert!(matches!(
            load_device_certification_signing_key(&key_path),
            Err(CertificationSigningKeyError::InvalidPath(reason))
                if reason.contains("group or world")
        ));
        Ok(())
    }
}
