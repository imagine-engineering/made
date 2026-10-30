use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::{ArtifactIndexError, ArtifactRoot};
use sim_tensor::CancellationToken;
use thiserror::Error;

use crate::{
    CertifiedNativeFfi, GENERAL_VIDEO_CODEC_ABI_MANIFEST_SHA256,
    GeneralVideoCodecPackageVerificationKey, NativeFfiContract, NativeFfiRegistry,
    NativeGeneralVideoCodecPackageSettings,
    native_ffi_elf::{
        NativeElfDynamicContract, NativeElfInspectionError, inspect_elf64_dynamic_contract,
    },
    native_video_codec_abi::{
        general_video_codec_library_contracts, general_video_codec_symbol_version_namespace,
    },
    trust::{
        GENERAL_VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN,
        GENERAL_VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN,
    },
    trust::{
        NativeLibraryImageError, NativePackageAdmissionError, NativePackageCoverageEntry,
        RetainedNativeLibraryImage, capture_native_library_bytes, parse_native_package_coverage,
    },
};

const GENERAL_VIDEO_TARGET: &str = "x86_64-unknown-linux-gnu";
const X86_64_ELF_MACHINE: u16 = 62;
const PACKAGE_COVERAGE_PATH: &str = "package-coverage.sha256";
const PACKAGE_RECEIPT_PATH: &str = "package-signature.json";
const PACKAGE_MANIFEST_PATH: &str = "package-manifest.json";
const DEPENDENCY_MANIFEST_PATH: &str = "dependency-contract-v1.json";
const DEPENDENCY_RECEIPT_PATH: &str = "dependency-contract-v1.signature.json";
const LICENSE_MANIFEST_PATH: &str = "license-manifest.json";
const SOURCE_BUILD_MANIFEST_PATH: &str = "source-build-manifest.json";
const MAXIMUM_PACKAGE_ENTRIES: usize = 512;
const MAXIMUM_COVERAGE_BYTES: usize = 1024 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 4 * 1024;
const MAXIMUM_METADATA_BYTES: usize = 4 * 1024 * 1024;
const MAXIMUM_METADATA_TOTAL_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_IMAGE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAXIMUM_IMAGE_TOTAL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const GENERAL_VIDEO_FFI_OWNER: &str = "sim_runtime::native_video_codec_ffi";
const FFMPEG_7_1_ARCHIVE_SHA256: &str =
    "40973d44970dbc83ef302b0609f2e74982be2d85916dd2ee7472d30678a7abe6";
const FFMPEG_7_1_SIGNATURE_SHA256: &str =
    "9bd1689dce76b109034dcc4765a406e84e8799a2fd857b000c0a4d9744b70617";
const FFMPEG_7_1_SIGNING_KEY_FINGERPRINT: &str = "FCF986EA15E6E293A5644F10B4322F04D67658D8";
const REVIEWED_SYSTEM_SONAMES: [&str; 5] = [
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
];
type CoverageEntry = NativePackageCoverageEntry;

#[derive(Debug, Error)]
pub enum GeneralVideoCodecPackageError {
    #[error("general video codec package admission was cancelled")]
    Cancelled,
    #[error("general video codec package is unsafe: {0}")]
    UnsafePackage(String),
    #[error("general video codec package signature is invalid: {0}")]
    InvalidSignature(String),
    #[error("general video codec package manifest is invalid: {0}")]
    InvalidManifest(String),
    #[error("general video codec package ELF contract is invalid: {0}")]
    InvalidElf(String),
    #[error("general video codec package FFI certification failed: {0}")]
    UncertifiedFfi(String),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoPackageManifest {
    schema_version: u16,
    signer: String,
    target: String,
    ffmpeg_release: String,
    source_archive_sha256: String,
    source_signature_sha256: String,
    source_signing_key_fingerprint: String,
    general_abi_sha256: String,
    dependency_contract_sha256: String,
    dependency_contract_receipt_sha256: String,
    license_manifest_sha256: String,
    source_build_manifest_sha256: String,
    libraries: Vec<GeneralVideoLibraryManifest>,
    support_files: Vec<String>,
    service_limits: GeneralVideoServiceLimits,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoLibraryManifest {
    identity: String,
    filename: String,
    sha256: String,
    abi_major: u16,
    soname: String,
    symbol_version_namespace: String,
    symbols: Vec<String>,
    needed: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoDependencyManifest {
    schema_version: u16,
    target: String,
    dependencies: Vec<GeneralVideoDependency>,
    edges: Vec<GeneralVideoDependencyEdge>,
    encoder_providers: BTreeMap<String, String>,
    reviewed_system_sonames: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoDependency {
    identity: String,
    filename: String,
    sha256: String,
    abi_version: String,
    soname: String,
    needed: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoDependencyEdge {
    consumer: String,
    dependency: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoServiceLimits {
    actor_capacity: u16,
    package_metadata_bytes: u64,
    retained_image_bytes: u64,
    codec_scratch_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoLicenseManifest {
    schema_version: u16,
    entries: Vec<GeneralVideoLicenseDisposition>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoLicenseDisposition {
    path: String,
    role: GeneralVideoLicenseRole,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum GeneralVideoLicenseRole {
    License,
    Notice,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoSourceBuildManifest {
    schema_version: u16,
    source_archive_sha256: String,
    source_signature_sha256: String,
    source_signing_key_fingerprint: String,
    source_signature_disposition: GeneralVideoSourceSignatureDisposition,
    runtime_compilation_forbidden: bool,
    entries: Vec<GeneralVideoSourceBuildDisposition>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum GeneralVideoSourceSignatureDisposition {
    VerifiedOfficialRelease,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralVideoSourceBuildDisposition {
    path: String,
    role: GeneralVideoSourceBuildRole,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum GeneralVideoSourceBuildRole {
    Source,
    BuildRecipe,
    BuildProvenance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralVideoCodecLibraryIdentity {
    identity: String,
    filename: String,
    digest_sha256: String,
    abi_major: u16,
    soname: String,
}

impl GeneralVideoCodecLibraryIdentity {
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn soname(&self) -> &str {
        &self.soname
    }

    pub fn abi_major(&self) -> u16 {
        self.abi_major
    }
}

pub struct VerifiedGeneralVideoCodecPackage {
    manifest: GeneralVideoPackageManifest,
    dependency_manifest: GeneralVideoDependencyManifest,
    coverage: BTreeMap<String, CoverageEntry>,
    coverage_bytes: Vec<u8>,
    receipt_bytes: Vec<u8>,
    metadata_bytes: BTreeMap<String, Vec<u8>>,
    semantic_identity: String,
}

pub struct InspectedGeneralVideoCodecPackage {
    verified: VerifiedGeneralVideoCodecPackage,
    primary_contracts: BTreeMap<String, NativeElfDynamicContract>,
    dependency_contracts: BTreeMap<String, NativeElfDynamicContract>,
    image_set: GeneralVideoCodecImageSet,
}

enum GeneralVideoCodecImageSet {
    Loadable(BTreeMap<String, RetainedNativeLibraryImage>),
    UnsupportedTarget,
}

pub struct CertifiedGeneralVideoCodecDependencyClosure {
    inspected: InspectedGeneralVideoCodecPackage,
    primary_certificates: BTreeMap<String, CertifiedNativeFfi>,
    dependency_certificates: BTreeMap<String, CertifiedNativeFfi>,
    libraries: BTreeMap<String, GeneralVideoCodecLibraryIdentity>,
    retained_image_bytes: u64,
    startup_resident_bytes: u64,
}

impl CertifiedGeneralVideoCodecDependencyClosure {
    pub fn semantic_identity(&self) -> &str {
        &self.inspected.verified.semantic_identity
    }

    pub fn libraries(&self) -> &BTreeMap<String, GeneralVideoCodecLibraryIdentity> {
        &self.libraries
    }

    pub fn retained_image_bytes(&self) -> u64 {
        self.retained_image_bytes
    }

    pub fn startup_resident_bytes(&self) -> u64 {
        self.startup_resident_bytes
    }

    pub fn codec_scratch_bytes(&self) -> u64 {
        self.inspected
            .verified
            .manifest
            .service_limits
            .codec_scratch_bytes
    }

    pub fn primary_certificate_count(&self) -> usize {
        self.primary_certificates.len()
    }

    pub fn dependency_certificate_count(&self) -> usize {
        self.dependency_certificates.len()
    }

    pub(crate) fn retained_loader_paths(&self) -> Option<BTreeMap<String, std::path::PathBuf>> {
        let GeneralVideoCodecImageSet::Loadable(images) = &self.inspected.image_set else {
            return None;
        };
        Some(
            images
                .iter()
                .map(|(identity, image)| (identity.clone(), image.loader_path().to_path_buf()))
                .collect(),
        )
    }

    pub(crate) fn dependency_first_order(
        &self,
    ) -> Result<Vec<String>, GeneralVideoCodecPackageError> {
        let edges = &self.inspected.verified.dependency_manifest.edges;
        let graph = edges
            .iter()
            .fold(BTreeMap::<&str, Vec<&str>>::new(), |mut graph, edge| {
                graph
                    .entry(edge.consumer.as_str())
                    .or_default()
                    .push(edge.dependency.as_str());
                graph
            });
        fn visit<'a>(
            identity: &'a str,
            graph: &BTreeMap<&'a str, Vec<&'a str>>,
            visited: &mut BTreeSet<&'a str>,
            ordered: &mut Vec<String>,
        ) {
            if !visited.insert(identity) {
                return;
            }
            if let Some(dependencies) = graph.get(identity) {
                for dependency in dependencies {
                    visit(dependency, graph, visited, ordered);
                }
            }
            ordered.push(identity.to_owned());
        }
        let mut identities = self
            .inspected
            .verified
            .manifest
            .libraries
            .iter()
            .map(|library| library.identity.as_str())
            .chain(
                self.inspected
                    .verified
                    .dependency_manifest
                    .dependencies
                    .iter()
                    .map(|dependency| dependency.identity.as_str()),
            )
            .collect::<Vec<_>>();
        identities.sort_unstable();
        let mut ordered = Vec::new();
        ordered.try_reserve_exact(identities.len()).map_err(|_| {
            GeneralVideoCodecPackageError::UncertifiedFfi(
                "dependency order allocation failed".to_owned(),
            )
        })?;
        let mut visited = BTreeSet::new();
        for identity in identities {
            visit(identity, &graph, &mut visited, &mut ordered);
        }
        Ok(ordered)
    }

    pub(crate) fn sonames(&self) -> BTreeMap<String, String> {
        self.inspected
            .verified
            .manifest
            .libraries
            .iter()
            .map(|library| (library.identity.clone(), library.soname.clone()))
            .chain(
                self.inspected
                    .verified
                    .dependency_manifest
                    .dependencies
                    .iter()
                    .map(|dependency| (dependency.identity.clone(), dependency.soname.clone())),
            )
            .collect()
    }

    pub(crate) fn needed(&self) -> BTreeMap<String, BTreeSet<String>> {
        self.inspected
            .primary_contracts
            .iter()
            .map(|(identity, contract)| (identity.clone(), contract.needed().clone()))
            .chain(
                self.inspected
                    .dependency_contracts
                    .iter()
                    .map(|(identity, contract)| (identity.clone(), contract.needed().clone())),
            )
            .collect()
    }

    pub(crate) fn reviewed_system_sonames(&self) -> BTreeSet<String> {
        self.inspected
            .verified
            .dependency_manifest
            .reviewed_system_sonames
            .iter()
            .cloned()
            .collect()
    }

    pub(crate) fn encoder_providers(&self) -> &BTreeMap<String, String> {
        &self
            .inspected
            .verified
            .dependency_manifest
            .encoder_providers
    }

    pub(crate) fn has_dependency(&self, identity: &str) -> bool {
        self.inspected
            .verified
            .dependency_manifest
            .dependencies
            .iter()
            .any(|dependency| dependency.identity == identity)
            && self.dependency_certificates.contains_key(identity)
    }

    pub(crate) fn has_dependency_edge(&self, consumer: &str, dependency: &str) -> bool {
        self.inspected
            .verified
            .dependency_manifest
            .edges
            .iter()
            .any(|edge| edge.consumer == consumer && edge.dependency == dependency)
    }

    pub(crate) fn primary_contracts(&self) -> &BTreeMap<String, NativeElfDynamicContract> {
        &self.inspected.primary_contracts
    }

    pub(crate) fn primary_certificates(&self) -> &BTreeMap<String, CertifiedNativeFfi> {
        &self.primary_certificates
    }
}

pub fn certify_general_video_codec_package(
    settings: &NativeGeneralVideoCodecPackageSettings,
    cancellation: &CancellationToken,
) -> Result<CertifiedGeneralVideoCodecDependencyClosure, GeneralVideoCodecPackageError> {
    let root = ArtifactRoot::canonical(
        "sim-general-video-codec-package",
        "native-general-video-codec-package",
        settings.package_root(),
        std::iter::empty::<String>(),
    )
    .map_err(|error| map_artifact_error(error, "<package-root>"))?;
    certify_general_video_codec_package_from_root(&root, settings.verification_key(), cancellation)
}

fn certify_general_video_codec_package_from_root(
    root: &ArtifactRoot,
    verification_key: &GeneralVideoCodecPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<CertifiedGeneralVideoCodecDependencyClosure, GeneralVideoCodecPackageError> {
    check_cancelled(cancellation)?;
    let coverage_capture = required_capture(
        root,
        PACKAGE_COVERAGE_PATH,
        MAXIMUM_COVERAGE_BYTES,
        cancellation,
    )?;
    let receipt_capture = required_capture(
        root,
        PACKAGE_RECEIPT_PATH,
        MAXIMUM_RECEIPT_BYTES,
        cancellation,
    )?;
    verification_key
        .verify_package(
            verification_key.signer(),
            coverage_capture.as_bytes(),
            receipt_capture.as_bytes(),
        )
        .map_err(|error| GeneralVideoCodecPackageError::InvalidSignature(error.to_string()))?;
    let coverage = parse_coverage(coverage_capture.as_bytes())?;
    let expected_tree = coverage
        .keys()
        .cloned()
        .chain([
            PACKAGE_COVERAGE_PATH.to_owned(),
            PACKAGE_RECEIPT_PATH.to_owned(),
        ])
        .collect::<BTreeSet<_>>();
    require_exact_tree(root, &expected_tree, cancellation)?;

    let fixed_metadata_paths = [
        PACKAGE_MANIFEST_PATH,
        DEPENDENCY_MANIFEST_PATH,
        DEPENDENCY_RECEIPT_PATH,
        LICENSE_MANIFEST_PATH,
        SOURCE_BUILD_MANIFEST_PATH,
    ];
    let mut metadata_bytes = BTreeMap::new();
    for path in fixed_metadata_paths {
        let bytes =
            capture_covered_file(root, path, MAXIMUM_METADATA_BYTES, &coverage, cancellation)?;
        metadata_bytes.insert(path.to_owned(), bytes);
    }
    let manifest: GeneralVideoPackageManifest = parse_canonical_json(
        metadata_bytes
            .get(PACKAGE_MANIFEST_PATH)
            .ok_or_else(|| invalid_manifest("package manifest is absent"))?,
        PACKAGE_MANIFEST_PATH,
    )?;
    let dependency_manifest: GeneralVideoDependencyManifest = parse_canonical_json(
        metadata_bytes
            .get(DEPENDENCY_MANIFEST_PATH)
            .ok_or_else(|| invalid_manifest("dependency manifest is absent"))?,
        DEPENDENCY_MANIFEST_PATH,
    )?;
    let dependency_receipt = metadata_bytes
        .get(DEPENDENCY_RECEIPT_PATH)
        .ok_or_else(|| invalid_manifest("dependency receipt is absent"))?;
    verification_key
        .verify_dependency_contract(
            &manifest.signer,
            metadata_bytes
                .get(DEPENDENCY_MANIFEST_PATH)
                .ok_or_else(|| invalid_manifest("dependency manifest is absent"))?,
            dependency_receipt,
        )
        .map_err(|error| GeneralVideoCodecPackageError::InvalidSignature(error.to_string()))?;
    let license_manifest: GeneralVideoLicenseManifest = parse_canonical_json(
        metadata_bytes
            .get(LICENSE_MANIFEST_PATH)
            .ok_or_else(|| invalid_manifest("license manifest is absent"))?,
        LICENSE_MANIFEST_PATH,
    )?;
    let source_build_manifest: GeneralVideoSourceBuildManifest = parse_canonical_json(
        metadata_bytes
            .get(SOURCE_BUILD_MANIFEST_PATH)
            .ok_or_else(|| invalid_manifest("source/build manifest is absent"))?,
        SOURCE_BUILD_MANIFEST_PATH,
    )?;
    validate_manifests(
        &manifest,
        &dependency_manifest,
        &license_manifest,
        &source_build_manifest,
        verification_key,
        &coverage,
        &metadata_bytes,
    )?;

    let image_paths = manifest
        .libraries
        .iter()
        .map(|library| library.filename.clone())
        .chain(
            dependency_manifest
                .dependencies
                .iter()
                .map(|dependency| dependency.filename.clone()),
        )
        .collect::<BTreeSet<_>>();
    let support_paths = license_manifest
        .entries
        .iter()
        .map(|entry| entry.path.clone())
        .chain(
            source_build_manifest
                .entries
                .iter()
                .map(|entry| entry.path.clone()),
        )
        .collect::<BTreeSet<_>>();
    if manifest.support_files.iter().ne(support_paths.iter()) {
        return Err(invalid_manifest(
            "package support-file list differs from reviewed license/source/build dispositions",
        ));
    }
    if support_paths.len()
        != license_manifest
            .entries
            .len()
            .checked_add(source_build_manifest.entries.len())
            .ok_or_else(|| invalid_manifest("support disposition count overflowed"))?
    {
        return Err(invalid_manifest(
            "license and source/build dispositions must name disjoint support files",
        ));
    }
    let expected_covered = fixed_metadata_paths
        .into_iter()
        .map(str::to_owned)
        .chain(image_paths.iter().cloned())
        .chain(support_paths.iter().cloned())
        .collect::<BTreeSet<_>>();
    if expected_covered.len() != coverage.len()
        || expected_covered.iter().ne(coverage.keys())
        || support_paths.iter().any(|path| image_paths.contains(path))
        || support_paths
            .iter()
            .any(|path| fixed_metadata_paths.contains(&path.as_str()))
    {
        return Err(invalid_manifest(
            "manifest roles do not cover the exact signed package tree",
        ));
    }
    let startup_capture_bytes = fixed_metadata_paths
        .iter()
        .copied()
        .chain(support_paths.iter().map(String::as_str))
        .try_fold(
            u64::try_from(coverage_capture.len())
                .ok()
                .and_then(|bytes| {
                    u64::try_from(receipt_capture.len())
                        .ok()
                        .and_then(|receipt| bytes.checked_add(receipt))
                })
                .ok_or_else(|| invalid_manifest("startup capture byte accounting overflowed"))?,
            |total, path| {
                total
                    .checked_add(required_coverage(&coverage, path)?.size)
                    .ok_or_else(|| invalid_manifest("startup capture byte accounting overflowed"))
            },
        )?;
    if startup_capture_bytes != manifest.service_limits.package_metadata_bytes
        || startup_capture_bytes > MAXIMUM_METADATA_TOTAL_BYTES
    {
        return Err(invalid_manifest(
            "signed metadata bytes differ from the package startup ceiling",
        ));
    }
    for path in &support_paths {
        let bytes =
            capture_covered_file(root, path, MAXIMUM_METADATA_BYTES, &coverage, cancellation)?;
        metadata_bytes.insert(path.clone(), bytes);
    }
    let metadata_total_bytes = metadata_bytes.values().try_fold(
        u64::try_from(coverage_capture.len())
            .ok()
            .and_then(|bytes| {
                u64::try_from(receipt_capture.len())
                    .ok()
                    .and_then(|receipt| bytes.checked_add(receipt))
            })
            .ok_or_else(|| invalid_manifest("metadata byte accounting overflowed"))?,
        |total, bytes| {
            total
                .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                    invalid_manifest("metadata length cannot be represented for accounting")
                })?)
                .ok_or_else(|| invalid_manifest("metadata byte accounting overflowed"))
        },
    )?;
    if metadata_total_bytes != startup_capture_bytes {
        return Err(invalid_manifest(
            "captured metadata bytes changed after the signed startup preflight",
        ));
    }

    let semantic_identity = semantic_identity(
        &manifest,
        &dependency_manifest,
        coverage_capture.digest_sha256(),
        verification_key,
    )?;
    let verified = VerifiedGeneralVideoCodecPackage {
        manifest,
        dependency_manifest,
        coverage,
        coverage_bytes: coverage_capture.into_bytes(),
        receipt_bytes: receipt_capture.into_bytes(),
        metadata_bytes,
        semantic_identity,
    };
    let inspected = inspect_and_seal_images(root, verified, cancellation)?;
    let certified = certify_inspected_package(inspected, cancellation)?;
    terminal_revalidate(
        root,
        verification_key,
        &certified,
        &expected_tree,
        cancellation,
    )?;
    Ok(certified)
}

fn inspect_and_seal_images(
    root: &ArtifactRoot,
    verified: VerifiedGeneralVideoCodecPackage,
    cancellation: &CancellationToken,
) -> Result<InspectedGeneralVideoCodecPackage, GeneralVideoCodecPackageError> {
    let mut primary_contracts = BTreeMap::new();
    let mut dependency_contracts = BTreeMap::new();
    let loadable_target = cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    ));
    let mut retained_images = BTreeMap::new();
    let mut total_image_bytes = 0_u64;

    let expected_image_bytes = verified
        .manifest
        .libraries
        .iter()
        .map(|library| library.filename.as_str())
        .chain(
            verified
                .dependency_manifest
                .dependencies
                .iter()
                .map(|dependency| dependency.filename.as_str()),
        )
        .try_fold(0_u64, |total, path| {
            checked_image_total(total, required_coverage(&verified.coverage, path)?.size)
        })?;
    if expected_image_bytes != verified.manifest.service_limits.retained_image_bytes {
        return Err(invalid_manifest(
            "signed native image sizes differ from the retained-image startup ceiling",
        ));
    }

    for library in &verified.manifest.libraries {
        check_cancelled(cancellation)?;
        let coverage = required_coverage(&verified.coverage, &library.filename)?;
        total_image_bytes = checked_image_total(total_image_bytes, coverage.size)?;
        let captured = required_capture(
            root,
            &library.filename,
            checked_image_limit(coverage.size)?,
            cancellation,
        )?;
        require_coverage_match(&library.filename, &captured, coverage)?;
        let contract =
            inspect_elf64_dynamic_contract(captured.as_bytes(), X86_64_ELF_MACHINE, cancellation)
                .map_err(map_native_elf_error)?;
        validate_primary_elf(library, &contract)?;
        let digest = captured.digest_sha256().to_owned();
        let captured =
            capture_native_library_bytes(captured, &digest).map_err(map_native_library_error)?;
        primary_contracts.insert(library.identity.clone(), contract);
        if loadable_target {
            let retained = captured
                .seal(&library.identity, cancellation)
                .map_err(map_native_library_error)?;
            retained_images.insert(library.identity.clone(), retained);
        }
    }
    for dependency in &verified.dependency_manifest.dependencies {
        check_cancelled(cancellation)?;
        let coverage = required_coverage(&verified.coverage, &dependency.filename)?;
        total_image_bytes = checked_image_total(total_image_bytes, coverage.size)?;
        let captured = required_capture(
            root,
            &dependency.filename,
            checked_image_limit(coverage.size)?,
            cancellation,
        )?;
        require_coverage_match(&dependency.filename, &captured, coverage)?;
        let contract =
            inspect_elf64_dynamic_contract(captured.as_bytes(), X86_64_ELF_MACHINE, cancellation)
                .map_err(map_native_elf_error)?;
        validate_dependency_elf(dependency, &contract)?;
        let digest = captured.digest_sha256().to_owned();
        let captured =
            capture_native_library_bytes(captured, &digest).map_err(map_native_library_error)?;
        dependency_contracts.insert(dependency.identity.clone(), contract);
        if loadable_target {
            let retained = captured
                .seal(&dependency.identity, cancellation)
                .map_err(map_native_library_error)?;
            retained_images.insert(dependency.identity.clone(), retained);
        }
    }
    if total_image_bytes != verified.manifest.service_limits.retained_image_bytes {
        return Err(invalid_manifest(
            "retained image bytes differ from the signed service limit",
        ));
    }
    Ok(InspectedGeneralVideoCodecPackage {
        verified,
        primary_contracts,
        dependency_contracts,
        image_set: if loadable_target {
            GeneralVideoCodecImageSet::Loadable(retained_images)
        } else {
            GeneralVideoCodecImageSet::UnsupportedTarget
        },
    })
}

fn certify_inspected_package(
    inspected: InspectedGeneralVideoCodecPackage,
    cancellation: &CancellationToken,
) -> Result<CertifiedGeneralVideoCodecDependencyClosure, GeneralVideoCodecPackageError> {
    let mut ffi_contracts = Vec::new();
    for library in &inspected.verified.manifest.libraries {
        ffi_contracts.push(
            NativeFfiContract::new(
                library.identity.clone(),
                library.sha256.clone(),
                library.abi_major.to_string(),
                library.symbols.clone(),
                GENERAL_VIDEO_FFI_OWNER,
            )
            .map_err(|error| GeneralVideoCodecPackageError::UncertifiedFfi(error.to_string()))?,
        );
    }
    let edge_sponsors = dependency_sponsors(&inspected.verified.dependency_manifest)?;
    for dependency in &inspected.verified.dependency_manifest.dependencies {
        ffi_contracts.push(
            NativeFfiContract::new_dependency(
                dependency.identity.clone(),
                dependency.sha256.clone(),
                dependency.abi_version.clone(),
                edge_sponsors
                    .get(&dependency.identity)
                    .ok_or_else(|| invalid_manifest("dependency sponsor is absent"))?
                    .clone(),
                GENERAL_VIDEO_FFI_OWNER,
            )
            .map_err(|error| GeneralVideoCodecPackageError::UncertifiedFfi(error.to_string()))?,
        );
    }
    let registry = NativeFfiRegistry::new(ffi_contracts)
        .map_err(|error| GeneralVideoCodecPackageError::UncertifiedFfi(error.to_string()))?;
    let mut primary_certificates = BTreeMap::new();
    let mut dependency_certificates = BTreeMap::new();
    let mut libraries = BTreeMap::new();
    for library in &inspected.verified.manifest.libraries {
        check_cancelled(cancellation)?;
        let contract = inspected
            .primary_contracts
            .get(&library.identity)
            .ok_or_else(|| invalid_manifest("inspected primary library is absent"))?;
        let certificate = registry
            .authorize(
                &library.identity,
                &library.sha256,
                &library.abi_major.to_string(),
                contract.symbols(),
            )
            .map_err(|error| GeneralVideoCodecPackageError::UncertifiedFfi(error.to_string()))?;
        primary_certificates.insert(library.identity.clone(), certificate);
        libraries.insert(
            library.identity.clone(),
            GeneralVideoCodecLibraryIdentity {
                identity: library.identity.clone(),
                filename: library.filename.clone(),
                digest_sha256: library.sha256.clone(),
                abi_major: library.abi_major,
                soname: library.soname.clone(),
            },
        );
    }
    for dependency in &inspected.verified.dependency_manifest.dependencies {
        check_cancelled(cancellation)?;
        let sponsor = edge_sponsors
            .get(&dependency.identity)
            .ok_or_else(|| invalid_manifest("dependency sponsor is absent"))?;
        let certificate = registry
            .authorize_dependency(
                &dependency.identity,
                &dependency.sha256,
                &dependency.abi_version,
                sponsor,
            )
            .map_err(|error| GeneralVideoCodecPackageError::UncertifiedFfi(error.to_string()))?;
        dependency_certificates.insert(dependency.identity.clone(), certificate);
    }
    let retained_image_bytes = inspected
        .verified
        .manifest
        .service_limits
        .retained_image_bytes;
    let startup_resident_bytes = retained_image_bytes
        .checked_add(
            inspected
                .verified
                .manifest
                .service_limits
                .package_metadata_bytes,
        )
        .ok_or_else(|| invalid_manifest("startup resident byte accounting overflowed"))?;
    Ok(CertifiedGeneralVideoCodecDependencyClosure {
        inspected,
        primary_certificates,
        dependency_certificates,
        libraries,
        retained_image_bytes,
        startup_resident_bytes,
    })
}

fn terminal_revalidate(
    root: &ArtifactRoot,
    verification_key: &GeneralVideoCodecPackageVerificationKey,
    closure: &CertifiedGeneralVideoCodecDependencyClosure,
    expected_tree: &BTreeSet<String>,
    cancellation: &CancellationToken,
) -> Result<(), GeneralVideoCodecPackageError> {
    require_exact_tree(root, expected_tree, cancellation)?;
    let coverage = required_capture(
        root,
        PACKAGE_COVERAGE_PATH,
        MAXIMUM_COVERAGE_BYTES,
        cancellation,
    )?;
    let receipt = required_capture(
        root,
        PACKAGE_RECEIPT_PATH,
        MAXIMUM_RECEIPT_BYTES,
        cancellation,
    )?;
    if coverage.as_bytes() != closure.inspected.verified.coverage_bytes
        || receipt.as_bytes() != closure.inspected.verified.receipt_bytes
    {
        return Err(GeneralVideoCodecPackageError::UnsafePackage(
            "coverage or receipt changed after initial verification".to_owned(),
        ));
    }
    verification_key
        .verify_package(
            verification_key.signer(),
            coverage.as_bytes(),
            receipt.as_bytes(),
        )
        .map_err(|error| GeneralVideoCodecPackageError::InvalidSignature(error.to_string()))?;
    for (path, bytes) in &closure.inspected.verified.metadata_bytes {
        let current = required_capture(root, path, MAXIMUM_METADATA_BYTES, cancellation)?;
        if current.as_bytes() != bytes {
            return Err(GeneralVideoCodecPackageError::UnsafePackage(format!(
                "covered metadata changed after capture: {path}"
            )));
        }
    }
    for (path, expected) in &closure.inspected.verified.coverage {
        if closure.inspected.verified.metadata_bytes.contains_key(path) {
            continue;
        }
        let (digest, size) = root
            .contained_file_digest(Path::new(path), cancellation)
            .map_err(|error| map_artifact_error(error, path))?
            .ok_or_else(|| {
                GeneralVideoCodecPackageError::UnsafePackage(format!(
                    "covered image disappeared: {path}"
                ))
            })?;
        if digest != expected.digest_sha256 || size != expected.size {
            return Err(GeneralVideoCodecPackageError::UnsafePackage(format!(
                "covered image changed after capture: {path}"
            )));
        }
    }
    check_cancelled(cancellation)
}

fn validate_manifests(
    manifest: &GeneralVideoPackageManifest,
    dependency: &GeneralVideoDependencyManifest,
    license: &GeneralVideoLicenseManifest,
    source_build: &GeneralVideoSourceBuildManifest,
    verification_key: &GeneralVideoCodecPackageVerificationKey,
    coverage: &BTreeMap<String, CoverageEntry>,
    metadata: &BTreeMap<String, Vec<u8>>,
) -> Result<(), GeneralVideoCodecPackageError> {
    if manifest.schema_version != 1
        || manifest.signer != verification_key.signer()
        || manifest.target != GENERAL_VIDEO_TARGET
        || manifest.ffmpeg_release != "7.1"
        || manifest.source_archive_sha256 != FFMPEG_7_1_ARCHIVE_SHA256
        || manifest.source_signature_sha256 != FFMPEG_7_1_SIGNATURE_SHA256
        || manifest.source_signing_key_fingerprint != FFMPEG_7_1_SIGNING_KEY_FINGERPRINT
        || manifest.general_abi_sha256 != GENERAL_VIDEO_CODEC_ABI_MANIFEST_SHA256
        || dependency.schema_version != 1
        || dependency.target != GENERAL_VIDEO_TARGET
        || manifest.service_limits.actor_capacity != 1
        || manifest.service_limits.package_metadata_bytes == 0
        || manifest.service_limits.package_metadata_bytes > MAXIMUM_METADATA_TOTAL_BYTES
        || manifest.service_limits.retained_image_bytes == 0
        || manifest.service_limits.retained_image_bytes > MAXIMUM_IMAGE_TOTAL_BYTES
        || manifest.service_limits.codec_scratch_bytes == 0
        || manifest
            .support_files
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid_manifest(
            "package identity, target, release, or service limits differ from the reviewed contract",
        ));
    }
    for (path, digest) in [
        (
            DEPENDENCY_MANIFEST_PATH,
            &manifest.dependency_contract_sha256,
        ),
        (
            DEPENDENCY_RECEIPT_PATH,
            &manifest.dependency_contract_receipt_sha256,
        ),
        (LICENSE_MANIFEST_PATH, &manifest.license_manifest_sha256),
        (
            SOURCE_BUILD_MANIFEST_PATH,
            &manifest.source_build_manifest_sha256,
        ),
    ] {
        require_sha256(digest)?;
        if format!(
            "{:x}",
            Sha256::digest(
                metadata
                    .get(path)
                    .ok_or_else(|| invalid_manifest("bound metadata is absent"))?
            )
        ) != *digest
        {
            return Err(invalid_manifest("bound metadata digest differs"));
        }
    }
    validate_primary_manifest(manifest, coverage)?;
    validate_dependency_manifest(dependency, manifest, coverage)?;
    validate_license_manifest(license, coverage)?;
    validate_source_build_manifest(source_build, coverage)?;
    Ok(())
}

fn validate_primary_manifest(
    manifest: &GeneralVideoPackageManifest,
    coverage: &BTreeMap<String, CoverageEntry>,
) -> Result<(), GeneralVideoCodecPackageError> {
    let expected = general_video_codec_library_contracts()
        .into_iter()
        .map(|(identity, abi_major, symbols)| (identity, (abi_major, symbols)))
        .collect::<BTreeMap<_, _>>();
    if manifest.libraries.len() != expected.len()
        || manifest
            .libraries
            .windows(2)
            .any(|pair| pair[0].identity >= pair[1].identity)
    {
        return Err(invalid_manifest(
            "primary library set is not exact and sorted",
        ));
    }
    let mut filenames = BTreeSet::new();
    let mut sonames = BTreeSet::new();
    for row in &manifest.libraries {
        let (abi_major, symbols) = expected
            .get(row.identity.as_str())
            .ok_or_else(|| invalid_manifest("primary library identity is not reviewed"))?;
        let expected_symbols = symbols
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect::<Vec<_>>();
        let (expected_filename, expected_soname) = reviewed_primary_library_paths(&row.identity)
            .ok_or_else(|| invalid_manifest("primary library file identity is not reviewed"))?;
        if row.abi_major != *abi_major
            || row.filename != expected_filename
            || row.soname != expected_soname
            || row.symbol_version_namespace
                != general_video_codec_symbol_version_namespace(&row.identity)
                    .ok_or_else(|| invalid_manifest("library namespace is absent"))?
            || row.symbols != expected_symbols
            || row.symbols.windows(2).any(|pair| pair[0] >= pair[1])
            || !valid_package_path(&row.filename)
            || !valid_filename(&row.soname)
            || row.needed.windows(2).any(|pair| pair[0] >= pair[1])
            || !filenames.insert(row.filename.as_str())
            || !sonames.insert(row.soname.as_str())
        {
            return Err(invalid_manifest("primary library contract differs"));
        }
        require_sha256(&row.sha256)?;
        if required_coverage(coverage, &row.filename)?.digest_sha256 != row.sha256 {
            return Err(invalid_manifest(
                "primary library digest differs from signed coverage",
            ));
        }
    }
    Ok(())
}

fn reviewed_primary_library_paths(identity: &str) -> Option<(&'static str, &'static str)> {
    match identity {
        "avcodec" => Some(("lib/libavcodec.so.61", "libavcodec.so.61")),
        "avfilter" => Some(("lib/libavfilter.so.10", "libavfilter.so.10")),
        "avformat" => Some(("lib/libavformat.so.61", "libavformat.so.61")),
        "avutil" => Some(("lib/libavutil.so.59", "libavutil.so.59")),
        "swresample" => Some(("lib/libswresample.so.5", "libswresample.so.5")),
        "swscale" => Some(("lib/libswscale.so.8", "libswscale.so.8")),
        _ => None,
    }
}

fn validate_license_manifest(
    manifest: &GeneralVideoLicenseManifest,
    coverage: &BTreeMap<String, CoverageEntry>,
) -> Result<(), GeneralVideoCodecPackageError> {
    if manifest.schema_version != 1
        || manifest.entries.is_empty()
        || manifest
            .entries
            .windows(2)
            .any(|pair| pair[0].path >= pair[1].path)
        || !manifest
            .entries
            .iter()
            .any(|entry| entry.role == GeneralVideoLicenseRole::License)
        || !manifest
            .entries
            .iter()
            .any(|entry| entry.role == GeneralVideoLicenseRole::Notice)
    {
        return Err(invalid_manifest(
            "license manifest must contain sorted license and notice dispositions",
        ));
    }
    validate_disposition_entries(
        manifest
            .entries
            .iter()
            .map(|entry| (&entry.path, &entry.sha256)),
        coverage,
        "license",
    )
}

fn validate_source_build_manifest(
    manifest: &GeneralVideoSourceBuildManifest,
    coverage: &BTreeMap<String, CoverageEntry>,
) -> Result<(), GeneralVideoCodecPackageError> {
    let roles = manifest
        .entries
        .iter()
        .map(|entry| entry.role)
        .collect::<BTreeSet<_>>();
    if manifest.schema_version != 1
        || manifest.source_archive_sha256 != FFMPEG_7_1_ARCHIVE_SHA256
        || manifest.source_signature_sha256 != FFMPEG_7_1_SIGNATURE_SHA256
        || manifest.source_signing_key_fingerprint != FFMPEG_7_1_SIGNING_KEY_FINGERPRINT
        || !manifest.runtime_compilation_forbidden
        || manifest.entries.is_empty()
        || manifest
            .entries
            .windows(2)
            .any(|pair| pair[0].path >= pair[1].path)
        || roles
            != BTreeSet::from([
                GeneralVideoSourceBuildRole::Source,
                GeneralVideoSourceBuildRole::BuildRecipe,
                GeneralVideoSourceBuildRole::BuildProvenance,
            ])
    {
        return Err(invalid_manifest(
            "source/build manifest differs from reviewed source and build dispositions",
        ));
    }
    validate_disposition_entries(
        manifest
            .entries
            .iter()
            .map(|entry| (&entry.path, &entry.sha256)),
        coverage,
        "source/build",
    )
}

fn validate_disposition_entries<'a>(
    entries: impl Iterator<Item = (&'a String, &'a String)>,
    coverage: &BTreeMap<String, CoverageEntry>,
    role: &str,
) -> Result<(), GeneralVideoCodecPackageError> {
    for (path, digest) in entries {
        if !valid_package_path(path) {
            return Err(invalid_manifest(format!(
                "{role} disposition path is invalid"
            )));
        }
        require_sha256(digest)?;
        if required_coverage(coverage, path)?.digest_sha256 != *digest {
            return Err(invalid_manifest(format!(
                "{role} disposition digest differs from signed coverage"
            )));
        }
    }
    Ok(())
}

fn validate_dependency_manifest(
    dependency: &GeneralVideoDependencyManifest,
    manifest: &GeneralVideoPackageManifest,
    coverage: &BTreeMap<String, CoverageEntry>,
) -> Result<(), GeneralVideoCodecPackageError> {
    if dependency
        .dependencies
        .windows(2)
        .any(|pair| pair[0].identity >= pair[1].identity)
        || dependency.edges.windows(2).any(|pair| pair[0] >= pair[1])
        || dependency
            .reviewed_system_sonames
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid_manifest(
            "dependency contract is not strictly sorted",
        ));
    }
    let primary = manifest
        .libraries
        .iter()
        .map(|library| library.identity.as_str())
        .collect::<BTreeSet<_>>();
    let dependencies = dependency
        .dependencies
        .iter()
        .map(|item| item.identity.as_str())
        .collect::<BTreeSet<_>>();
    if primary
        .iter()
        .any(|identity| dependencies.contains(identity))
    {
        return Err(invalid_manifest(
            "primary and dependency identities overlap",
        ));
    }
    for item in &dependency.dependencies {
        require_sha256(&item.sha256)?;
        if !valid_package_path(&item.filename)
            || !valid_filename(&item.soname)
            || item.abi_version.is_empty()
            || !coverage.contains_key(&item.filename)
            || item.needed.windows(2).any(|pair| pair[0] >= pair[1])
            || required_coverage(coverage, &item.filename)?.digest_sha256 != item.sha256
        {
            return Err(invalid_manifest("dependency identity is invalid"));
        }
    }
    let dependency_filenames = dependency
        .dependencies
        .iter()
        .map(|item| item.filename.as_str())
        .collect::<BTreeSet<_>>();
    let dependency_sonames = dependency
        .dependencies
        .iter()
        .map(|item| item.soname.as_str())
        .collect::<BTreeSet<_>>();
    if dependency_filenames.len() != dependency.dependencies.len()
        || dependency_sonames.len() != dependency.dependencies.len()
        || manifest
            .libraries
            .iter()
            .any(|library| dependency_filenames.contains(library.filename.as_str()))
        || manifest
            .libraries
            .iter()
            .any(|library| dependency_sonames.contains(library.soname.as_str()))
    {
        return Err(invalid_manifest(
            "primary and dependency file or SONAME identities overlap",
        ));
    }
    let mut consumers = BTreeMap::<String, BTreeSet<String>>::new();
    for edge in &dependency.edges {
        if (!primary.contains(edge.consumer.as_str())
            && !dependencies.contains(edge.consumer.as_str()))
            || !dependencies.contains(edge.dependency.as_str())
            || edge.consumer == edge.dependency
        {
            return Err(invalid_manifest(
                "dependency graph has an invalid or duplicate edge",
            ));
        }
        consumers
            .entry(edge.dependency.clone())
            .or_default()
            .insert(edge.consumer.clone());
    }
    if consumers.len() != dependencies.len() || dependency_graph_has_cycle(&dependency.edges) {
        return Err(invalid_manifest("dependency graph is incomplete or cyclic"));
    }
    if dependency.reviewed_system_sonames != REVIEWED_SYSTEM_SONAMES.map(str::to_owned).to_vec() {
        return Err(invalid_manifest(
            "reviewed system SONAME allowlist differs from the fixed package policy",
        ));
    }
    let expected_encoder_providers = BTreeMap::from([
        ("aac".to_owned(), "avcodec".to_owned()),
        ("libsvtav1".to_owned(), "svtav1".to_owned()),
        ("libvpx-vp9".to_owned(), "vpx".to_owned()),
        ("libx264".to_owned(), "x264".to_owned()),
    ]);
    if dependency.encoder_providers != expected_encoder_providers
        || ["svtav1", "vpx", "x264"]
            .iter()
            .any(|identity| !dependencies.contains(identity))
    {
        return Err(invalid_manifest(
            "codec provider mapping differs from the reviewed service contract",
        ));
    }
    let accounted_sonames = manifest
        .libraries
        .iter()
        .map(|library| library.soname.as_str())
        .chain(
            dependency
                .dependencies
                .iter()
                .map(|item| item.soname.as_str()),
        )
        .chain(
            dependency
                .reviewed_system_sonames
                .iter()
                .map(String::as_str),
        )
        .collect::<BTreeSet<_>>();
    if manifest
        .libraries
        .iter()
        .flat_map(|library| &library.needed)
        .chain(dependency.dependencies.iter().flat_map(|item| &item.needed))
        .any(|needed| !accounted_sonames.contains(needed.as_str()))
    {
        return Err(invalid_manifest(
            "DT_NEEDED graph contains an unaccounted SONAME",
        ));
    }
    let dependency_by_soname = dependency
        .dependencies
        .iter()
        .map(|item| (item.soname.as_str(), item.identity.as_str()))
        .collect::<BTreeMap<_, _>>();
    let expected_edges = manifest
        .libraries
        .iter()
        .map(|library| (library.identity.as_str(), library.needed.as_slice()))
        .chain(
            dependency
                .dependencies
                .iter()
                .map(|item| (item.identity.as_str(), item.needed.as_slice())),
        )
        .flat_map(|(consumer, needed)| {
            needed.iter().filter_map(|soname| {
                dependency_by_soname.get(soname.as_str()).map(|dependency| {
                    GeneralVideoDependencyEdge {
                        consumer: consumer.to_owned(),
                        dependency: (*dependency).to_owned(),
                    }
                })
            })
        })
        .collect::<BTreeSet<_>>();
    if expected_edges.iter().ne(dependency.edges.iter()) {
        return Err(invalid_manifest(
            "signed dependency edges differ from the exact DT_NEEDED package projection",
        ));
    }
    Ok(())
}

fn validate_primary_elf(
    manifest: &GeneralVideoLibraryManifest,
    contract: &NativeElfDynamicContract,
) -> Result<(), GeneralVideoCodecPackageError> {
    if contract.soname() != Some(manifest.soname.as_str())
        || contract.needed().iter().ne(manifest.needed.iter())
    {
        return Err(GeneralVideoCodecPackageError::InvalidElf(format!(
            "{} SONAME or DT_NEEDED closure differs",
            manifest.identity
        )));
    }
    for symbol in &manifest.symbols {
        let identities = contract.symbol_identities().get(symbol).ok_or_else(|| {
            GeneralVideoCodecPackageError::InvalidElf(format!(
                "{} symbol {symbol} is absent",
                manifest.identity
            ))
        })?;
        let matches = identities
            .iter()
            .filter(|identity| {
                identity.binding == 1
                    && identity.kind == 2
                    && identity.visibility == 0
                    && identity.section_index != 0
                    && identity.executable
                    && identity.version.as_ref().is_some_and(|version| {
                        version.is_default && version.name == manifest.symbol_version_namespace
                    })
            })
            .count();
        if matches != 1 {
            return Err(GeneralVideoCodecPackageError::InvalidElf(format!(
                "{} symbol {symbol} has no unique callable default-version provider",
                manifest.identity
            )));
        }
    }
    Ok(())
}

fn validate_dependency_elf(
    manifest: &GeneralVideoDependency,
    contract: &NativeElfDynamicContract,
) -> Result<(), GeneralVideoCodecPackageError> {
    if contract.soname() != Some(manifest.soname.as_str())
        || contract.needed().iter().ne(manifest.needed.iter())
    {
        return Err(GeneralVideoCodecPackageError::InvalidElf(format!(
            "{} dependency ELF contract differs",
            manifest.identity
        )));
    }
    Ok(())
}

fn dependency_sponsors(
    manifest: &GeneralVideoDependencyManifest,
) -> Result<BTreeMap<String, String>, GeneralVideoCodecPackageError> {
    let mut sponsors = BTreeMap::new();
    for edge in &manifest.edges {
        sponsors
            .entry(edge.dependency.clone())
            .and_modify(|consumer: &mut String| {
                if edge.consumer < *consumer {
                    *consumer = edge.consumer.clone();
                }
            })
            .or_insert_with(|| edge.consumer.clone());
    }
    Ok(sponsors)
}

fn dependency_graph_has_cycle(edges: &[GeneralVideoDependencyEdge]) -> bool {
    let graph = edges
        .iter()
        .fold(BTreeMap::<&str, Vec<&str>>::new(), |mut graph, edge| {
            graph
                .entry(edge.consumer.as_str())
                .or_default()
                .push(edge.dependency.as_str());
            graph
        });
    fn visit<'a>(
        node: &'a str,
        graph: &BTreeMap<&'a str, Vec<&'a str>>,
        visiting: &mut BTreeSet<&'a str>,
        visited: &mut BTreeSet<&'a str>,
    ) -> bool {
        if visited.contains(node) {
            return false;
        }
        if !visiting.insert(node) {
            return true;
        }
        if graph.get(node).is_some_and(|next| {
            next.iter()
                .copied()
                .any(|dependency| visit(dependency, graph, visiting, visited))
        }) {
            return true;
        }
        visiting.remove(node);
        visited.insert(node);
        false
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    graph
        .keys()
        .copied()
        .any(|node| visit(node, &graph, &mut visiting, &mut visited))
}

fn parse_coverage(
    bytes: &[u8],
) -> Result<BTreeMap<String, CoverageEntry>, GeneralVideoCodecPackageError> {
    let entries = parse_native_package_coverage(
        bytes,
        &[PACKAGE_COVERAGE_PATH, PACKAGE_RECEIPT_PATH],
        MAXIMUM_COVERAGE_BYTES,
    )
    .map_err(map_native_package_admission_error)?;
    if entries.values().any(|entry| entry.size == 0) || entries.len() + 2 > MAXIMUM_PACKAGE_ENTRIES
    {
        return Err(GeneralVideoCodecPackageError::UnsafePackage(
            "coverage entry count is outside the package bound".to_owned(),
        ));
    }
    Ok(entries)
}

fn map_native_package_admission_error(
    error: NativePackageAdmissionError,
) -> GeneralVideoCodecPackageError {
    match error {
        NativePackageAdmissionError::Cancelled => GeneralVideoCodecPackageError::Cancelled,
        NativePackageAdmissionError::UnsafePackage(message)
        | NativePackageAdmissionError::InvalidCoverage(message) => {
            GeneralVideoCodecPackageError::UnsafePackage(message)
        }
    }
}

fn parse_canonical_json<T>(bytes: &[u8], path: &str) -> Result<T, GeneralVideoCodecPackageError>
where
    T: for<'de> Deserialize<'de> + Serialize,
{
    if bytes.is_empty() || bytes.len() > MAXIMUM_METADATA_BYTES || bytes.last() != Some(&b'\n') {
        return Err(invalid_manifest(format!(
            "{path} is not bounded newline-terminated JSON"
        )));
    }
    let value = serde_json::from_slice::<T>(bytes)
        .map_err(|error| invalid_manifest(format!("{path} is malformed: {error}")))?;
    let mut canonical = serde_json::to_vec(&value)
        .map_err(|error| invalid_manifest(format!("{path} cannot be serialized: {error}")))?;
    canonical.push(b'\n');
    if canonical != bytes {
        return Err(invalid_manifest(format!("{path} is not canonical JSON")));
    }
    Ok(value)
}

fn semantic_identity(
    manifest: &GeneralVideoPackageManifest,
    dependency: &GeneralVideoDependencyManifest,
    coverage_digest: &str,
    verification_key: &GeneralVideoCodecPackageVerificationKey,
) -> Result<String, GeneralVideoCodecPackageError> {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-general-video-codec-semantic-identity-v1\0");
    digest.update(GENERAL_VIDEO_CODEC_PACKAGE_SIGNATURE_DOMAIN);
    digest.update(GENERAL_VIDEO_CODEC_DEPENDENCY_CONTRACT_SIGNATURE_DOMAIN);
    digest.update(
        serde_json::to_vec(manifest)
            .map_err(|error| invalid_manifest(format!("manifest identity failed: {error}")))?,
    );
    digest.update(
        serde_json::to_vec(dependency)
            .map_err(|error| invalid_manifest(format!("dependency identity failed: {error}")))?,
    );
    digest.update(coverage_digest.as_bytes());
    digest.update(verification_key.signer().as_bytes());
    digest.update(verification_key.public_key_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

fn capture_covered_file(
    root: &ArtifactRoot,
    path: &str,
    maximum_bytes: usize,
    coverage: &BTreeMap<String, CoverageEntry>,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, GeneralVideoCodecPackageError> {
    let expected = required_coverage(coverage, path)?;
    let captured = required_capture(root, path, maximum_bytes, cancellation)?;
    require_coverage_match(path, &captured, expected)?;
    Ok(captured.into_bytes())
}

fn required_capture(
    root: &ArtifactRoot,
    path: &str,
    maximum_bytes: usize,
    cancellation: &CancellationToken,
) -> Result<sim_model::artifact_index::CapturedPrivateArtifact, GeneralVideoCodecPackageError> {
    root.capture_private_file(Path::new(path), maximum_bytes, cancellation)
        .map_err(|error| map_artifact_error(error, path))?
        .ok_or_else(|| {
            GeneralVideoCodecPackageError::UnsafePackage(format!(
                "required package file is absent: {path}"
            ))
        })
}

fn required_coverage<'a>(
    coverage: &'a BTreeMap<String, CoverageEntry>,
    path: &str,
) -> Result<&'a CoverageEntry, GeneralVideoCodecPackageError> {
    coverage.get(path).ok_or_else(|| {
        GeneralVideoCodecPackageError::UnsafePackage(format!(
            "required package file is not signed: {path}"
        ))
    })
}

fn require_coverage_match(
    path: &str,
    captured: &sim_model::artifact_index::CapturedPrivateArtifact,
    expected: &CoverageEntry,
) -> Result<(), GeneralVideoCodecPackageError> {
    if captured.digest_sha256() != expected.digest_sha256
        || u64::try_from(captured.len()).unwrap_or(u64::MAX) != expected.size
    {
        return Err(GeneralVideoCodecPackageError::UnsafePackage(format!(
            "captured package file differs from signed coverage: {path}"
        )));
    }
    Ok(())
}

fn require_exact_tree(
    root: &ArtifactRoot,
    expected: &BTreeSet<String>,
    cancellation: &CancellationToken,
) -> Result<(), GeneralVideoCodecPackageError> {
    let observed = root
        .list_contained_regular_files_recursive(MAXIMUM_PACKAGE_ENTRIES, cancellation)
        .map_err(|error| map_artifact_error(error, "<package-tree>"))?
        .into_iter()
        .map(|path| {
            path.to_str().map(str::to_owned).ok_or_else(|| {
                GeneralVideoCodecPackageError::UnsafePackage(
                    "package tree contains a non-UTF-8 path".to_owned(),
                )
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if &observed != expected {
        return Err(GeneralVideoCodecPackageError::UnsafePackage(
            "recursive package tree differs from signed coverage".to_owned(),
        ));
    }
    Ok(())
}

fn checked_image_total(current: u64, added: u64) -> Result<u64, GeneralVideoCodecPackageError> {
    let total = current.checked_add(added).ok_or_else(|| {
        GeneralVideoCodecPackageError::UnsafePackage(
            "native image aggregate byte accounting overflowed".to_owned(),
        )
    })?;
    if total > MAXIMUM_IMAGE_TOTAL_BYTES {
        return Err(GeneralVideoCodecPackageError::UnsafePackage(
            "native images exceed the full-closure byte bound".to_owned(),
        ));
    }
    Ok(total)
}

fn checked_image_limit(size: u64) -> Result<usize, GeneralVideoCodecPackageError> {
    let size = usize::try_from(size).map_err(|_| {
        GeneralVideoCodecPackageError::UnsafePackage(
            "native image size exceeds this platform".to_owned(),
        )
    })?;
    if size == 0 || size > MAXIMUM_IMAGE_BYTES {
        return Err(GeneralVideoCodecPackageError::UnsafePackage(
            "native image size is outside the per-image bound".to_owned(),
        ));
    }
    Ok(size)
}

fn require_sha256(value: &str) -> Result<(), GeneralVideoCodecPackageError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(invalid_manifest(
            "SHA-256 value is not lowercase hexadecimal",
        ));
    }
    Ok(())
}

fn valid_package_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 4_096
        && !path.starts_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|component| valid_filename(component) && component != "." && component != "..")
}

fn valid_filename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), GeneralVideoCodecPackageError> {
    cancellation
        .check()
        .map_err(|_| GeneralVideoCodecPackageError::Cancelled)
}

fn invalid_manifest(message: impl Into<String>) -> GeneralVideoCodecPackageError {
    GeneralVideoCodecPackageError::InvalidManifest(message.into())
}

fn map_artifact_error(
    error: ArtifactIndexError,
    relative_path: &str,
) -> GeneralVideoCodecPackageError {
    match error {
        ArtifactIndexError::Cancelled => GeneralVideoCodecPackageError::Cancelled,
        _ => GeneralVideoCodecPackageError::UnsafePackage(format!(
            "capability-relative package operation failed for {relative_path}"
        )),
    }
}

fn map_native_elf_error(error: NativeElfInspectionError) -> GeneralVideoCodecPackageError {
    match error {
        NativeElfInspectionError::Cancelled(_) => GeneralVideoCodecPackageError::Cancelled,
        NativeElfInspectionError::Invalid(message) => {
            GeneralVideoCodecPackageError::InvalidElf(message)
        }
    }
}

fn map_native_library_error(error: NativeLibraryImageError) -> GeneralVideoCodecPackageError {
    match error {
        NativeLibraryImageError::Cancelled => GeneralVideoCodecPackageError::Cancelled,
        NativeLibraryImageError::UnsupportedPlatform => {
            GeneralVideoCodecPackageError::UnsafePackage(
                "native-library sealing is unsupported on this target".to_owned(),
            )
        }
        NativeLibraryImageError::Invalid(message) => {
            GeneralVideoCodecPackageError::UnsafePackage(message)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_COVERAGE: &[u8] = include_bytes!(
        "../../sim_test_support/fixtures/video/codec-package-bootstrap/package-coverage.sha256"
    );
    const FIXTURE_MANIFEST: &[u8] = include_bytes!(
        "../../sim_test_support/fixtures/video/codec-package-bootstrap/package-manifest.json"
    );
    const FIXTURE_DEPENDENCY_MANIFEST: &[u8] = include_bytes!(
        "../../sim_test_support/fixtures/video/codec-package-bootstrap/dependency-contract-v1.json"
    );
    const DIGEST_A: &str = "0000000000000000000000000000000000000000000000000000000000000001";
    const DIGEST_B: &str = "0000000000000000000000000000000000000000000000000000000000000002";

    #[test]
    fn general_video_codec_package_bootstrap_accepts_only_canonical_coverage() {
        let coverage = format!("{DIGEST_A} 1  a\n{DIGEST_B} 2  b\n");
        let parsed = parse_coverage(coverage.as_bytes()).expect("canonical coverage must parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.get("a").map(|entry| entry.size), Some(1));
        assert_eq!(parsed.get("b").map(|entry| entry.size), Some(2));

        for invalid in [
            format!("{DIGEST_B} 2  b\n{DIGEST_A} 1  a\n"),
            format!("{DIGEST_A} 01  a\n"),
            format!("{DIGEST_A} 1  a\n{DIGEST_B} 2  a\n"),
            format!("{DIGEST_A} 1  {PACKAGE_COVERAGE_PATH}\n"),
            format!("{DIGEST_A} 1  ../a\n"),
        ] {
            assert!(parse_coverage(invalid.as_bytes()).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn general_video_codec_package_bootstrap_delegates_elf_inspection_to_the_shared_owner() {
        let source = include_str!("native_video_codec_package.rs");
        let shared_inspector_call =
            ["inspect_elf64_dynamic_contract", "(captured.as_bytes()"].concat();
        assert_eq!(source.matches(&shared_inspector_call).count(), 2);
        for forbidden_definition in [
            ["const PT", "_DYNAMIC"].concat(),
            ["const DT", "_NEEDED"].concat(),
            ["const DT", "_RPATH"].concat(),
            ["const DT", "_RUNPATH"].concat(),
            ["const SHT", "_DYNSYM"].concat(),
            ["const STT", "_FUNC"].concat(),
            ["fn parse", "_elf"].concat(),
        ] {
            assert!(
                !source.contains(&forbidden_definition),
                "{forbidden_definition}"
            );
        }
        assert!(source.contains("capture_native_library_bytes(captured"));
    }

    #[test]
    fn general_video_codec_package_bootstrap_rejects_dependency_graph_and_provider_drift() {
        let coverage = parse_coverage(FIXTURE_COVERAGE).expect("fixture coverage must parse");
        let manifest: GeneralVideoPackageManifest =
            parse_canonical_json(FIXTURE_MANIFEST, PACKAGE_MANIFEST_PATH)
                .expect("fixture manifest must parse");
        let dependency: GeneralVideoDependencyManifest =
            parse_canonical_json(FIXTURE_DEPENDENCY_MANIFEST, DEPENDENCY_MANIFEST_PATH)
                .expect("fixture dependency manifest must parse");
        validate_dependency_manifest(&dependency, &manifest, &coverage)
            .expect("fixture dependency closure must be exact");

        let mut missing_edge = dependency.clone();
        assert!(missing_edge.edges.pop().is_some());
        assert!(validate_dependency_manifest(&missing_edge, &manifest, &coverage).is_err());

        let mut changed_provider = dependency.clone();
        changed_provider
            .encoder_providers
            .insert("aac".to_owned(), "x264".to_owned());
        assert!(validate_dependency_manifest(&changed_provider, &manifest, &coverage).is_err());

        let mut unreviewed_system_library = dependency;
        unreviewed_system_library
            .reviewed_system_sonames
            .push("libunreviewed.so.1".to_owned());
        assert!(
            validate_dependency_manifest(&unreviewed_system_library, &manifest, &coverage).is_err()
        );
    }
}
