use crate::{
    MetalPackageVerificationKey, NativeFfiContract, NativeFfiRegistry, TrustError,
    trust::{
        NativePackageAdmissionError, NativePackagePayloadLimit, capture_native_package,
        validate_native_package_coverage,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_backend_metal::{
    ABI_FLOOR, EXECUTION_CONTRACT, EXECUTION_UNSAFE_OWNER, METAL_3_FAMILY_VALUE,
    METAL_ADD_F16_FUNCTION, METAL_ADD_F32_FUNCTION, MetalRuntime, READINESS_FUNCTION, probe_device,
};
use sim_model::ArtifactRoot;
use sim_types::{BackendUnavailable, CancellationError, CancellationToken, DeviceKind};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};
use thiserror::Error;

#[cfg(test)]
use std::fs;

const PACKAGE_POLICY: &str = include_str!("../../../nix/sim-backends/metal/package-policy.json");
const ABI_MANIFEST: &str = sim_backend_metal::ABI_MANIFEST_JSON;
const EXECUTION_ABI: &str = sim_backend_metal::EXECUTION_ABI_JSON;
const MAX_PACKAGE_FILE_BYTES: usize = 16 * 1024 * 1024;
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_COVERAGE_BYTES: usize = 64 * 1024;
const MAX_PACKAGE_PAYLOADS: usize = 32;
const MAX_PACKAGE_BYTES: usize = 32 * 1024 * 1024;
const LOADER_UNSAFE_OWNER: &str = "sim_backend_metal::loader";
const FRAMEWORK_PROVENANCE_DOMAIN: &[u8] = b"zed-sim-metal-framework-contract-v1\0";
const PACKAGE_PAYLOAD_LIMITS: [NativePackagePayloadLimit; 15] = [
    NativePackagePayloadLimit::new("LICENSES", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("LICENSES.execution", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("abi/execution-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new(
        "abi/reviewed-execution-bindings-v1.txt",
        MAX_PACKAGE_FILE_BYTES,
    ),
    NativePackagePayloadLimit::new("abi/symbols-v1.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("adapter-manifest.sig", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("execution-policy.json", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("ffi-contracts-v1.json", MAX_CATALOG_BYTES),
    NativePackagePayloadLimit::new("kernels/readiness.metal", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("kernels/readiness.metallib", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("kernels/tensor_ops.metal", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("kernels/tensor_ops.metallib", MAX_PACKAGE_FILE_BYTES),
    NativePackagePayloadLimit::new("package-coverage.sha256", MAX_COVERAGE_BYTES),
    NativePackagePayloadLimit::new("package-policy.json", MAX_PACKAGE_FILE_BYTES),
];
const COVERAGE_EXCLUDES: [&str; 2] = ["adapter-manifest.sig", "package-coverage.sha256"];
const FRAMEWORK_IDENTITIES: [&str; 3] = [
    "metal-framework",
    "metal-performance-shaders-framework",
    "metal-performance-shaders-graph-framework",
];
const METALLIB_IDENTITIES: [&str; 2] = ["metal-readiness-metallib", "metal-tensor-ops-metallib"];

#[cfg(test)]
thread_local! {
    static REVIEWED_POLICY_OVERRIDE: std::cell::RefCell<Option<Vec<u8>>> = const {
        std::cell::RefCell::new(None)
    };
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalFrameworkContract {
    identity: String,
    name: String,
    install_name: String,
    image_name: String,
    required_symbols: Vec<String>,
    unsafe_owner: String,
}

impl MetalFrameworkContract {
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn install_name(&self) -> &str {
        &self.install_name
    }

    pub fn image_name(&self) -> &str {
        &self.image_name
    }

    pub fn required_symbols(&self) -> &[String] {
        &self.required_symbols
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetalMetallibContract {
    identity: String,
    path: String,
    sha256: String,
    source_path: String,
    source_sha256: String,
    required_functions: Vec<String>,
    unsafe_owner: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetalFfiContractCatalogDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    minimum_macos_version: String,
    metal_language_version: String,
    minimum_metal_family: u64,
    abi_manifest_sha256: String,
    execution_abi_sha256: String,
    frameworks: Vec<MetalFrameworkContract>,
    classes: Vec<Value>,
    resource_selectors: Vec<Value>,
    layouts: Vec<Value>,
    metallibs: Vec<MetalMetallibContract>,
    runtime_compilation_forbidden: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MetalPackageManifestDto {
    schema_version: u16,
    backend: String,
    abi_floor: String,
    target: String,
    minimum_macos_version: String,
    metal_language_version: String,
    minimum_metal_family: u64,
    abi_manifest_sha256: String,
    execution_abi_sha256: String,
    ffi_contracts_sha256: String,
    readiness_source_sha256: String,
    readiness_metallib_sha256: String,
    readiness_metallib_size: u64,
    tensor_ops_source_sha256: String,
    tensor_ops_metallib_sha256: String,
    tensor_ops_metallib_size: u64,
    readiness_function: String,
    tensor_functions: Vec<String>,
    redistributes_apple_frameworks: bool,
    required_system_frameworks: Vec<String>,
    required_entitlements: Vec<String>,
    reviewed_sdk_version: String,
    reviewed_xcode_build: String,
    signer: String,
    signature_algorithm: String,
    signature_coverage: String,
    signature_domain: String,
    final_application_signing_required: bool,
    runtime_compilation_forbidden: bool,
}

#[derive(Clone, Debug)]
pub struct VerifiedMetalFfiContracts {
    package_root: ArtifactRoot,
    target: String,
    registry: NativeFfiRegistry,
    frameworks: Vec<MetalFrameworkContract>,
    framework_provenance_digests: BTreeMap<String, String>,
    readiness_metallib: Vec<u8>,
    tensor_ops_metallib: Vec<u8>,
}

impl VerifiedMetalFfiContracts {
    pub fn package_root(&self) -> &ArtifactRoot {
        &self.package_root
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn registry(&self) -> &NativeFfiRegistry {
        &self.registry
    }

    pub fn frameworks(&self) -> &[MetalFrameworkContract] {
        &self.frameworks
    }

    pub fn framework_provenance_digest(&self, identity: &str) -> Option<&str> {
        self.framework_provenance_digests
            .get(identity)
            .map(String::as_str)
    }

    pub fn readiness_metallib(&self) -> &[u8] {
        &self.readiness_metallib
    }

    pub fn tensor_ops_metallib(&self) -> &[u8] {
        &self.tensor_ops_metallib
    }
}

#[derive(Debug, Error)]
pub enum MetalPackageContractError {
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Trust(#[from] TrustError),
    #[error("Metal package tree is unsafe or incomplete: {0}")]
    UnsafePackage(String),
    #[error("signed Metal package metadata is invalid: {0}")]
    InvalidPackage(String),
    #[error("signed Metal FFI contract catalog is invalid: {0}")]
    InvalidCatalog(String),
}

struct MetalCertificationRetention {
    _verified: VerifiedMetalFfiContracts,
    _certificates: Vec<crate::CertifiedNativeFfi>,
}

struct CertifiedMetalInputs {
    certificates: Vec<crate::CertifiedNativeFfi>,
    readiness_metallib: Arc<[u8]>,
    tensor_ops_metallib: Arc<[u8]>,
}

pub struct CertifiedMetalRuntime {
    runtime: MetalRuntime,
    certificates: Vec<crate::CertifiedNativeFfi>,
    host_physical_memory_bytes: u64,
}

impl CertifiedMetalRuntime {
    pub fn runtime(&self) -> &MetalRuntime {
        &self.runtime
    }

    pub fn certificates(&self) -> &[crate::CertifiedNativeFfi] {
        &self.certificates
    }

    pub const fn host_physical_memory_bytes(&self) -> u64 {
        self.host_physical_memory_bytes
    }

    pub fn into_runtime(self) -> MetalRuntime {
        self.runtime
    }
}

pub fn initialize_certified_metal_runtime(
    settings: &crate::NativeMetalPackageSettings,
    cancellation: &CancellationToken,
) -> Result<CertifiedMetalRuntime, BackendUnavailable> {
    let unavailable = |reason: &'static str| BackendUnavailable::new(DeviceKind::Metal, reason);
    let verified = verify_metal_package_contracts(
        settings.package_root(),
        settings.verification_key(),
        cancellation,
    )
    .map_err(|_| unavailable("signed package or contract verification failed"))?;
    cancellation
        .check()
        .map_err(|_| unavailable("Metal initialization was cancelled"))?;
    let certified_inputs = certify_verified_metal_inputs(&verified)
        .map_err(|_| unavailable("framework or metallib certification failed"))?;
    require_supported_macos_version()
        .map_err(|_| unavailable("macOS 13 or newer is required for Metal execution"))?;
    let device_probe = probe_device()
        .map_err(|_| unavailable("fixed framework ABI or Metal/MPS device probe failed"))?;
    let host_physical_memory_bytes = observe_host_physical_memory_bytes()
        .map_err(|_| unavailable("host physical memory observation failed"))?;

    let certificates = certified_inputs.certificates;
    let readiness_metallib = certified_inputs.readiness_metallib;
    let tensor_ops_metallib = certified_inputs.tensor_ops_metallib;
    cancellation
        .check()
        .map_err(|_| unavailable("Metal initialization was cancelled"))?;
    let retained_certificates = certificates.clone();
    let retention = Arc::new(MetalCertificationRetention {
        _verified: verified,
        _certificates: retained_certificates,
    });
    let runtime = unsafe {
        MetalRuntime::from_certified_metallibs(readiness_metallib, tensor_ops_metallib, retention)
    }
    .map_err(|_| unavailable("certified metallib admission or readiness execution failed"))?;
    let properties = runtime.properties();
    if !metal_device_identity_matches(
        properties.name(),
        properties.registry_id(),
        properties.recommended_working_set_bytes(),
        properties.unified_memory(),
        &device_probe,
    ) {
        return Err(unavailable(
            "the default Metal device changed during certified initialization",
        ));
    }
    Ok(CertifiedMetalRuntime {
        runtime,
        certificates,
        host_physical_memory_bytes,
    })
}

fn certify_verified_metal_inputs(
    verified: &VerifiedMetalFfiContracts,
) -> Result<CertifiedMetalInputs, TrustError> {
    let mut certificates = Vec::with_capacity(5);
    for framework in verified.frameworks() {
        let digest = verified
            .framework_provenance_digest(framework.identity())
            .ok_or(TrustError::UncertifiedFfi)?;
        let available_symbols = framework.required_symbols().iter().cloned().collect();
        certificates.push(verified.registry().authorize(
            framework.identity(),
            digest,
            ABI_FLOOR,
            &available_symbols,
        )?);
    }
    let readiness_metallib = Arc::<[u8]>::from(verified.readiness_metallib());
    let tensor_ops_metallib = Arc::<[u8]>::from(verified.tensor_ops_metallib());
    certificates.push(verified.registry().authorize(
        "metal-readiness-metallib",
        &sha256_hex(&readiness_metallib),
        ABI_FLOOR,
        &BTreeSet::from([READINESS_FUNCTION.to_owned()]),
    )?);
    certificates.push(verified.registry().authorize(
        "metal-tensor-ops-metallib",
        &sha256_hex(&tensor_ops_metallib),
        EXECUTION_CONTRACT,
        &BTreeSet::from([
            METAL_ADD_F16_FUNCTION.to_owned(),
            METAL_ADD_F32_FUNCTION.to_owned(),
        ]),
    )?);
    Ok(CertifiedMetalInputs {
        certificates,
        readiness_metallib,
        tensor_ops_metallib,
    })
}

fn metal_device_identity_matches(
    name: &str,
    registry_id: u64,
    recommended_working_set_bytes: u64,
    unified_memory: bool,
    expected: &sim_backend_metal::MetalDeviceProbe,
) -> bool {
    name == expected.name
        && registry_id == expected.registry_id
        && recommended_working_set_bytes == expected.recommended_working_set_bytes
        && unified_memory == expected.unified_memory
}

#[cfg(any(test, target_os = "macos"))]
fn parse_macos_major_version(bytes: &[u8]) -> Result<u32, ()> {
    if bytes.is_empty() || bytes.len() > 32 || !bytes.is_ascii() {
        return Err(());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| ())?;
    let major = text.split('.').next().ok_or(())?;
    if major.is_empty()
        || major.len() > 3
        || major.starts_with('0')
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || text
            .split('.')
            .skip(1)
            .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(());
    }
    major.parse().map_err(|_| ())
}

#[cfg(any(test, target_os = "macos"))]
fn validate_macos_version_observation(status: i32, length: usize, bytes: &[u8]) -> Result<(), ()> {
    if status != 0 || length == 0 || length > bytes.len() {
        return Err(());
    }
    let version = bytes.get(..length).ok_or(())?;
    let version = version.strip_suffix(&[0]).unwrap_or(version);
    if parse_macos_major_version(version)? < 13 {
        return Err(());
    }
    Ok(())
}

#[cfg(any(test, target_os = "macos"))]
fn validate_host_memory_observation(status: i32, length: usize, value: u64) -> Result<u64, ()> {
    if status != 0 || length != std::mem::size_of::<u64>() || value == 0 {
        return Err(());
    }
    Ok(value)
}

#[cfg(target_os = "macos")]
fn require_supported_macos_version() -> Result<(), ()> {
    let mut bytes = [0_u8; 32];
    let mut length = bytes.len();
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.osproductversion".as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    validate_macos_version_observation(status, length, &bytes)
}

#[cfg(not(target_os = "macos"))]
fn require_supported_macos_version() -> Result<(), ()> {
    Err(())
}

#[cfg(target_os = "macos")]
fn observe_host_physical_memory_bytes() -> Result<u64, ()> {
    let mut value = 0_u64;
    let mut length = std::mem::size_of::<u64>();
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    validate_host_memory_observation(status, length, value)
}

#[cfg(not(target_os = "macos"))]
fn observe_host_physical_memory_bytes() -> Result<u64, ()> {
    Err(())
}

pub fn verify_metal_package_contracts(
    package_root: &Path,
    verification_key: &MetalPackageVerificationKey,
    cancellation: &CancellationToken,
) -> Result<VerifiedMetalFfiContracts, MetalPackageContractError> {
    cancellation.check()?;
    let root = ArtifactRoot::canonical(
        "sim-metal-package",
        "native-ffi-package",
        package_root,
        std::iter::empty::<String>(),
    )
    .map_err(|error| MetalPackageContractError::UnsafePackage(error.to_string()))?;
    let payloads = capture_native_package(
        &root,
        &PACKAGE_PAYLOAD_LIMITS,
        MAX_PACKAGE_PAYLOADS,
        MAX_PACKAGE_BYTES,
        cancellation,
    )
    .map_err(map_package_admission_error)?;

    let coverage_bytes = required_payload(&payloads, "package-coverage.sha256")?;
    validate_native_package_coverage(
        coverage_bytes,
        &payloads,
        &COVERAGE_EXCLUDES,
        MAX_COVERAGE_BYTES,
    )
    .map_err(map_package_admission_error)?;
    let packaged_policy: Value = parse_strict_json(
        required_payload(&payloads, "package-policy.json")?,
        "package policy",
    )?;
    let reviewed_policy_bytes = reviewed_package_policy_bytes();
    let reviewed_policy: Value = parse_strict_json(&reviewed_policy_bytes, "reviewed policy")?;
    if packaged_policy != reviewed_policy {
        return Err(MetalPackageContractError::InvalidPackage(
            "installed package policy differs from the compiled reviewed policy".to_owned(),
        ));
    }
    let manifest: MetalPackageManifestDto = parse_strict_json(
        required_payload(&payloads, "adapter-manifest.json")?,
        "adapter manifest",
    )?;
    validate_manifest(&manifest, &payloads, &reviewed_policy)?;
    verification_key.verify_package(
        &manifest.signer,
        coverage_bytes,
        required_payload(&payloads, "adapter-manifest.sig")?,
    )?;
    cancellation.check()?;

    let catalog_bytes = required_payload(&payloads, "ffi-contracts-v1.json")?;
    let catalog: MetalFfiContractCatalogDto = parse_strict_json(catalog_bytes, "FFI catalog")?;
    let (registry, framework_provenance_digests) =
        validate_and_map_catalog(&catalog, &manifest, &payloads)?;
    cancellation.check()?;

    Ok(VerifiedMetalFfiContracts {
        package_root: root,
        target: catalog.target,
        registry,
        frameworks: catalog.frameworks,
        framework_provenance_digests,
        readiness_metallib: required_payload(&payloads, "kernels/readiness.metallib")?.to_vec(),
        tensor_ops_metallib: required_payload(&payloads, "kernels/tensor_ops.metallib")?.to_vec(),
    })
}

fn map_package_admission_error(error: NativePackageAdmissionError) -> MetalPackageContractError {
    match error {
        NativePackageAdmissionError::Cancelled => CancellationError.into(),
        NativePackageAdmissionError::UnsafePackage(reason) => {
            MetalPackageContractError::UnsafePackage(reason)
        }
        NativePackageAdmissionError::InvalidCoverage(reason) => {
            MetalPackageContractError::InvalidPackage(reason)
        }
    }
}

fn validate_manifest(
    manifest: &MetalPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
    reviewed_policy: &Value,
) -> Result<(), MetalPackageContractError> {
    let expected_frameworks = vec![
        "/System/Library/Frameworks/Metal.framework/Metal".to_owned(),
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders"
            .to_owned(),
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph"
            .to_owned(),
    ];
    let expected_functions = vec![
        METAL_ADD_F16_FUNCTION.to_owned(),
        METAL_ADD_F32_FUNCTION.to_owned(),
    ];
    if manifest.schema_version != 2
        || manifest.backend != "metal"
        || manifest.abi_floor != ABI_FLOOR
        || !matches!(
            manifest.target.as_str(),
            "aarch64-apple-darwin" | "x86_64-apple-darwin"
        )
        || manifest.minimum_macos_version != "13.0"
        || manifest.metal_language_version != "metal3.0"
        || manifest.minimum_metal_family != METAL_3_FAMILY_VALUE
        || manifest.readiness_function != "zed_sim_metal_readiness_v1"
        || manifest.tensor_functions != expected_functions
        || manifest.redistributes_apple_frameworks
        || manifest.required_system_frameworks != expected_frameworks
        || !manifest.required_entitlements.is_empty()
        || manifest.reviewed_sdk_version != "26.2"
        || manifest.reviewed_xcode_build != "17C52"
        || manifest.signer.is_empty()
        || manifest.signature_algorithm != "ed25519"
        || manifest.signature_coverage != "package-coverage-v1"
        || manifest.signature_domain != "zed-sim-metal-package-v1"
        || !manifest.final_application_signing_required
        || !manifest.runtime_compilation_forbidden
    {
        return Err(MetalPackageContractError::InvalidPackage(
            "adapter manifest identity, target, ABI, or security policy is invalid".to_owned(),
        ));
    }
    let digest_fields = [
        (&manifest.abi_manifest_sha256, "abi/symbols-v1.json"),
        (&manifest.execution_abi_sha256, "abi/execution-v1.json"),
        (&manifest.ffi_contracts_sha256, "ffi-contracts-v1.json"),
        (&manifest.readiness_source_sha256, "kernels/readiness.metal"),
        (
            &manifest.readiness_metallib_sha256,
            "kernels/readiness.metallib",
        ),
        (
            &manifest.tensor_ops_source_sha256,
            "kernels/tensor_ops.metal",
        ),
        (
            &manifest.tensor_ops_metallib_sha256,
            "kernels/tensor_ops.metallib",
        ),
    ];
    for (digest, path) in digest_fields {
        if !valid_lower_hex_digest(digest)
            || digest != &sha256_hex(required_payload(payloads, path)?)
        {
            return Err(MetalPackageContractError::InvalidPackage(format!(
                "adapter manifest digest does not match {path}"
            )));
        }
    }
    let expected_policy_digests = [
        (
            &manifest.abi_manifest_sha256,
            required_string(reviewed_policy, "abi_manifest_sha256")?,
        ),
        (
            &manifest.execution_abi_sha256,
            required_string(reviewed_policy, "execution_abi_sha256")?,
        ),
        (
            &manifest.readiness_source_sha256,
            required_string(reviewed_policy, "readiness_source_sha256")?,
        ),
        (
            &manifest.tensor_ops_source_sha256,
            required_string(reviewed_policy, "tensor_ops_source_sha256")?,
        ),
        (
            &manifest.readiness_metallib_sha256,
            required_target_digest(
                reviewed_policy,
                "readiness_metallib_sha256_by_target",
                &manifest.target,
            )?,
        ),
        (
            &manifest.tensor_ops_metallib_sha256,
            required_target_digest(
                reviewed_policy,
                "tensor_ops_metallib_sha256_by_target",
                &manifest.target,
            )?,
        ),
    ];
    if expected_policy_digests
        .iter()
        .any(|(actual, expected)| actual.as_str() != *expected)
    {
        return Err(MetalPackageContractError::InvalidPackage(
            "adapter manifest digests differ from the reviewed target policy".to_owned(),
        ));
    }
    for (field, path) in [
        (
            "reviewed_execution_bindings_sha256",
            "abi/reviewed-execution-bindings-v1.txt",
        ),
        ("execution_policy_sha256", "execution-policy.json"),
        ("license_notice_sha256", "LICENSES"),
        ("execution_license_notice_sha256", "LICENSES.execution"),
    ] {
        let expected = required_string(reviewed_policy, field)?;
        if expected != sha256_hex(required_payload(payloads, path)?) {
            return Err(MetalPackageContractError::InvalidPackage(format!(
                "reviewed policy digest does not match {path}"
            )));
        }
    }
    for (size, path) in [
        (
            manifest.readiness_metallib_size,
            "kernels/readiness.metallib",
        ),
        (
            manifest.tensor_ops_metallib_size,
            "kernels/tensor_ops.metallib",
        ),
    ] {
        if usize::try_from(size).ok() != Some(required_payload(payloads, path)?.len()) {
            return Err(MetalPackageContractError::InvalidPackage(format!(
                "adapter manifest size does not match {path}"
            )));
        }
    }
    Ok(())
}

fn validate_and_map_catalog(
    catalog: &MetalFfiContractCatalogDto,
    manifest: &MetalPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(NativeFfiRegistry, BTreeMap<String, String>), MetalPackageContractError> {
    if catalog.schema_version != 1
        || catalog.backend != "metal"
        || catalog.abi_floor != ABI_FLOOR
        || catalog.target != manifest.target
        || catalog.minimum_macos_version != manifest.minimum_macos_version
        || catalog.metal_language_version != manifest.metal_language_version
        || catalog.minimum_metal_family != METAL_3_FAMILY_VALUE
        || !catalog.runtime_compilation_forbidden
        || catalog.abi_manifest_sha256 != manifest.abi_manifest_sha256
        || catalog.execution_abi_sha256 != manifest.execution_abi_sha256
    {
        return Err(MetalPackageContractError::InvalidCatalog(
            "catalog envelope differs from the signed package contract".to_owned(),
        ));
    }
    let embedded_abi: Value = parse_strict_json(ABI_MANIFEST.as_bytes(), "embedded ABI")?;
    let embedded_execution: Value =
        parse_strict_json(EXECUTION_ABI.as_bytes(), "embedded execution ABI")?;
    let expected_classes = normalized_embedded_classes(&embedded_abi)?;
    if catalog.abi_manifest_sha256 != sha256_hex(ABI_MANIFEST.as_bytes())
        || catalog.execution_abi_sha256 != sha256_hex(EXECUTION_ABI.as_bytes())
        || catalog.classes != expected_classes
        || catalog.layouts.as_slice() != required_array(&embedded_abi, "layouts")?.as_slice()
        || catalog.resource_selectors.as_slice()
            != required_array(&embedded_execution, "resource_selectors")?.as_slice()
    {
        return Err(MetalPackageContractError::InvalidCatalog(
            "class, selector, layout, or reviewed ABI digest differs".to_owned(),
        ));
    }
    validate_frameworks(&catalog.frameworks, &embedded_abi)?;
    validate_metallibs(&catalog.metallibs, manifest, payloads)?;

    let mut contracts = Vec::with_capacity(5);
    let mut provenance = BTreeMap::new();
    for framework in &catalog.frameworks {
        let digest = framework_provenance_digest(catalog, framework)?;
        contracts.push(NativeFfiContract::new(
            framework.identity.clone(),
            digest.clone(),
            ABI_FLOOR,
            framework.required_symbols.clone(),
            framework.unsafe_owner.clone(),
        )?);
        provenance.insert(framework.identity.clone(), digest);
    }
    for metallib in &catalog.metallibs {
        let abi = if metallib.identity == "metal-tensor-ops-metallib" {
            EXECUTION_CONTRACT
        } else {
            ABI_FLOOR
        };
        contracts.push(NativeFfiContract::new(
            metallib.identity.clone(),
            metallib.sha256.clone(),
            abi,
            metallib.required_functions.clone(),
            metallib.unsafe_owner.clone(),
        )?);
    }
    Ok((NativeFfiRegistry::new(contracts)?, provenance))
}

fn normalized_embedded_classes(
    embedded_abi: &Value,
) -> Result<Vec<Value>, MetalPackageContractError> {
    let mut classes = required_array(embedded_abi, "classes")?.clone();
    for class in &mut classes {
        let selectors = class
            .get_mut("selectors")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                MetalPackageContractError::InvalidCatalog(
                    "embedded class selectors must be an array".to_owned(),
                )
            })?;
        selectors.sort_by(|left, right| {
            let left_key = (
                left.get("kind").and_then(Value::as_str).unwrap_or_default(),
                left.get("name").and_then(Value::as_str).unwrap_or_default(),
            );
            let right_key = (
                right
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                right
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            left_key.cmp(&right_key)
        });
        if selectors.windows(2).any(|pair| {
            selector_identity(&pair[0])
                .zip(selector_identity(&pair[1]))
                .is_none_or(|(left, right)| left >= right)
        }) {
            return Err(MetalPackageContractError::InvalidCatalog(
                "embedded class selectors are duplicated".to_owned(),
            ));
        }
    }
    Ok(classes)
}

fn selector_identity(value: &Value) -> Option<(&str, &str)> {
    Some((value.get("kind")?.as_str()?, value.get("name")?.as_str()?))
}

fn validate_frameworks(
    frameworks: &[MetalFrameworkContract],
    embedded_abi: &Value,
) -> Result<(), MetalPackageContractError> {
    if frameworks.len() != FRAMEWORK_IDENTITIES.len()
        || frameworks
            .iter()
            .map(|row| row.identity.as_str())
            .ne(FRAMEWORK_IDENTITIES)
    {
        return Err(MetalPackageContractError::InvalidCatalog(
            "framework identities must be exact, sorted, and unique".to_owned(),
        ));
    }
    let expected = required_array(embedded_abi, "frameworks")?;
    for (index, framework) in frameworks.iter().enumerate() {
        let source = expected.get(index).ok_or_else(|| {
            MetalPackageContractError::InvalidCatalog("framework coverage is incomplete".to_owned())
        })?;
        let expected_name = required_string(source, "name")?;
        let expected_install_name = required_string(source, "install_name")?;
        let expected_image_name = required_string(source, "image_name")?;
        let source_symbols = required_array(source, "symbols")?
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    MetalPackageContractError::InvalidCatalog(
                        "embedded framework symbol is invalid".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected_symbols = if source_symbols.is_empty() {
            vec![
                "objc.MPSGraph".to_owned(),
                "objc.MPSGraphDevice".to_owned(),
                "objc.MPSGraphTensorData".to_owned(),
            ]
        } else {
            source_symbols
        };
        if framework.name != expected_name
            || framework.install_name != expected_install_name
            || framework.image_name != expected_image_name
            || framework.required_symbols != expected_symbols
            || framework.unsafe_owner != LOADER_UNSAFE_OWNER
            || framework
                .required_symbols
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(MetalPackageContractError::InvalidCatalog(format!(
                "framework contract {} differs from the reviewed ABI",
                framework.identity
            )));
        }
    }
    Ok(())
}

fn validate_metallibs(
    metallibs: &[MetalMetallibContract],
    manifest: &MetalPackageManifestDto,
    payloads: &BTreeMap<String, Vec<u8>>,
) -> Result<(), MetalPackageContractError> {
    if metallibs.len() != METALLIB_IDENTITIES.len()
        || metallibs
            .iter()
            .map(|row| row.identity.as_str())
            .ne(METALLIB_IDENTITIES)
    {
        return Err(MetalPackageContractError::InvalidCatalog(
            "metallib identities must be exact, sorted, and unique".to_owned(),
        ));
    }
    let expected = [
        (
            "metal-readiness-metallib",
            "kernels/readiness.metallib",
            &manifest.readiness_metallib_sha256,
            "kernels/readiness.metal",
            &manifest.readiness_source_sha256,
            vec!["zed_sim_metal_readiness_v1".to_owned()],
            LOADER_UNSAFE_OWNER,
        ),
        (
            "metal-tensor-ops-metallib",
            "kernels/tensor_ops.metallib",
            &manifest.tensor_ops_metallib_sha256,
            "kernels/tensor_ops.metal",
            &manifest.tensor_ops_source_sha256,
            vec![
                METAL_ADD_F16_FUNCTION.to_owned(),
                METAL_ADD_F32_FUNCTION.to_owned(),
            ],
            EXECUTION_UNSAFE_OWNER,
        ),
    ];
    for (row, expected) in metallibs.iter().zip(expected) {
        if row.identity != expected.0
            || row.path != expected.1
            || &row.sha256 != expected.2
            || row.source_path != expected.3
            || &row.source_sha256 != expected.4
            || row.required_functions != expected.5
            || row.unsafe_owner != expected.6
            || row.sha256 != sha256_hex(required_payload(payloads, &row.path)?)
            || row.source_sha256 != sha256_hex(required_payload(payloads, &row.source_path)?)
        {
            return Err(MetalPackageContractError::InvalidCatalog(format!(
                "metallib contract {} differs from signed payloads",
                row.identity
            )));
        }
    }
    Ok(())
}

fn framework_provenance_digest(
    catalog: &MetalFfiContractCatalogDto,
    framework: &MetalFrameworkContract,
) -> Result<String, MetalPackageContractError> {
    let canonical = serde_json::to_vec(&(
        &catalog.target,
        &catalog.abi_manifest_sha256,
        &catalog.execution_abi_sha256,
        &catalog.minimum_macos_version,
        &catalog.metal_language_version,
        catalog.minimum_metal_family,
        framework,
        &catalog.classes,
        &catalog.resource_selectors,
        &catalog.layouts,
    ))
    .map_err(|error| MetalPackageContractError::InvalidCatalog(error.to_string()))?;
    let mut digest = Sha256::new();
    digest.update(FRAMEWORK_PROVENANCE_DOMAIN);
    digest.update(canonical);
    Ok(format!("{:x}", digest.finalize()))
}

fn required_payload<'a>(
    payloads: &'a BTreeMap<String, Vec<u8>>,
    path: &str,
) -> Result<&'a [u8], MetalPackageContractError> {
    payloads.get(path).map(Vec::as_slice).ok_or_else(|| {
        MetalPackageContractError::UnsafePackage(format!("required payload is missing: {path}"))
    })
}

fn required_array<'a>(
    value: &'a Value,
    field: &str,
) -> Result<&'a Vec<Value>, MetalPackageContractError> {
    value.get(field).and_then(Value::as_array).ok_or_else(|| {
        MetalPackageContractError::InvalidCatalog(format!("{field} must be an array"))
    })
}

fn required_string<'a>(
    value: &'a Value,
    field: &str,
) -> Result<&'a str, MetalPackageContractError> {
    value.get(field).and_then(Value::as_str).ok_or_else(|| {
        MetalPackageContractError::InvalidCatalog(format!("{field} must be a string"))
    })
}

fn required_target_digest<'a>(
    value: &'a Value,
    field: &str,
    target: &str,
) -> Result<&'a str, MetalPackageContractError> {
    value
        .get(field)
        .and_then(Value::as_object)
        .and_then(|values| values.get(target))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            MetalPackageContractError::InvalidPackage(format!(
                "reviewed policy has no {field} digest for {target}"
            ))
        })
}

fn reviewed_package_policy_bytes() -> Cow<'static, [u8]> {
    #[cfg(test)]
    if let Some(bytes) = REVIEWED_POLICY_OVERRIDE.with(|value| value.borrow().clone()) {
        return Cow::Owned(bytes);
    }
    Cow::Borrowed(PACKAGE_POLICY.as_bytes())
}

fn valid_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_strict_json<T: DeserializeOwned>(
    bytes: &[u8],
    label: &str,
) -> Result<T, MetalPackageContractError> {
    let strict = crate::trust::parse_strict_json_value(bytes).map_err(|error| {
        MetalPackageContractError::InvalidPackage(format!("{label} is not strict JSON: {error}"))
    })?;
    serde_json::from_value(strict).map_err(|error| {
        MetalPackageContractError::InvalidPackage(format!("{label} is invalid: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use std::io;

    struct ReviewedPolicyOverrideGuard;

    impl ReviewedPolicyOverrideGuard {
        fn install(bytes: Vec<u8>) -> Self {
            REVIEWED_POLICY_OVERRIDE.with(|value| {
                let previous = value.borrow_mut().replace(bytes);
                assert!(
                    previous.is_none(),
                    "reviewed policy override is already active"
                );
            });
            Self
        }
    }

    impl Drop for ReviewedPolicyOverrideGuard {
        fn drop(&mut self) {
            REVIEWED_POLICY_OVERRIDE.with(|value| *value.borrow_mut() = None);
        }
    }

    fn fixture_contracts() -> Result<
        (
            MetalFfiContractCatalogDto,
            MetalPackageManifestDto,
            BTreeMap<String, Vec<u8>>,
        ),
        Box<dyn std::error::Error>,
    > {
        let embedded_abi: Value = serde_json::from_str(ABI_MANIFEST)?;
        let embedded_execution: Value = serde_json::from_str(EXECUTION_ABI)?;
        let mut frameworks = Vec::new();
        for (identity, row) in FRAMEWORK_IDENTITIES
            .iter()
            .zip(required_array(&embedded_abi, "frameworks")?)
        {
            let mut required_symbols = required_array(row, "symbols")?
                .iter()
                .map(|symbol| {
                    symbol
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("invalid fixture symbol")
                })
                .collect::<Result<Vec<_>, _>>()?;
            if required_symbols.is_empty() {
                required_symbols = vec![
                    "objc.MPSGraph".to_owned(),
                    "objc.MPSGraphDevice".to_owned(),
                    "objc.MPSGraphTensorData".to_owned(),
                ];
            }
            frameworks.push(MetalFrameworkContract {
                identity: (*identity).to_owned(),
                name: required_string(row, "name")?.to_owned(),
                install_name: required_string(row, "install_name")?.to_owned(),
                image_name: required_string(row, "image_name")?.to_owned(),
                required_symbols,
                unsafe_owner: LOADER_UNSAFE_OWNER.to_owned(),
            });
        }
        let readiness_source = b"readiness source".to_vec();
        let readiness_metallib = b"readiness metallib".to_vec();
        let tensor_source = b"tensor source".to_vec();
        let tensor_metallib = b"tensor metallib".to_vec();
        let mut payloads = BTreeMap::new();
        payloads.insert(
            "kernels/readiness.metal".to_owned(),
            readiness_source.clone(),
        );
        payloads.insert(
            "kernels/readiness.metallib".to_owned(),
            readiness_metallib.clone(),
        );
        payloads.insert("kernels/tensor_ops.metal".to_owned(), tensor_source.clone());
        payloads.insert(
            "kernels/tensor_ops.metallib".to_owned(),
            tensor_metallib.clone(),
        );
        let manifest = MetalPackageManifestDto {
            schema_version: 2,
            backend: "metal".to_owned(),
            abi_floor: ABI_FLOOR.to_owned(),
            target: "aarch64-apple-darwin".to_owned(),
            minimum_macos_version: "13.0".to_owned(),
            metal_language_version: "metal3.0".to_owned(),
            minimum_metal_family: METAL_3_FAMILY_VALUE,
            abi_manifest_sha256: sha256_hex(ABI_MANIFEST.as_bytes()),
            execution_abi_sha256: sha256_hex(EXECUTION_ABI.as_bytes()),
            ffi_contracts_sha256: "0".repeat(64),
            readiness_source_sha256: sha256_hex(&readiness_source),
            readiness_metallib_sha256: sha256_hex(&readiness_metallib),
            readiness_metallib_size: readiness_metallib.len().try_into()?,
            tensor_ops_source_sha256: sha256_hex(&tensor_source),
            tensor_ops_metallib_sha256: sha256_hex(&tensor_metallib),
            tensor_ops_metallib_size: tensor_metallib.len().try_into()?,
            readiness_function: "zed_sim_metal_readiness_v1".to_owned(),
            tensor_functions: vec![
                METAL_ADD_F16_FUNCTION.to_owned(),
                METAL_ADD_F32_FUNCTION.to_owned(),
            ],
            redistributes_apple_frameworks: false,
            required_system_frameworks: frameworks
                .iter()
                .map(|framework| framework.install_name.clone())
                .collect(),
            required_entitlements: Vec::new(),
            reviewed_sdk_version: "26.2".to_owned(),
            reviewed_xcode_build: "17C52".to_owned(),
            signer: "metal.release".to_owned(),
            signature_algorithm: "ed25519".to_owned(),
            signature_coverage: "package-coverage-v1".to_owned(),
            signature_domain: "zed-sim-metal-package-v1".to_owned(),
            final_application_signing_required: true,
            runtime_compilation_forbidden: true,
        };
        let catalog = MetalFfiContractCatalogDto {
            schema_version: 1,
            backend: "metal".to_owned(),
            abi_floor: ABI_FLOOR.to_owned(),
            target: manifest.target.clone(),
            minimum_macos_version: manifest.minimum_macos_version.clone(),
            metal_language_version: manifest.metal_language_version.clone(),
            minimum_metal_family: METAL_3_FAMILY_VALUE,
            abi_manifest_sha256: manifest.abi_manifest_sha256.clone(),
            execution_abi_sha256: manifest.execution_abi_sha256.clone(),
            frameworks,
            classes: normalized_embedded_classes(&embedded_abi)?,
            resource_selectors: required_array(&embedded_execution, "resource_selectors")?.clone(),
            layouts: required_array(&embedded_abi, "layouts")?.clone(),
            metallibs: vec![
                MetalMetallibContract {
                    identity: "metal-readiness-metallib".to_owned(),
                    path: "kernels/readiness.metallib".to_owned(),
                    sha256: manifest.readiness_metallib_sha256.clone(),
                    source_path: "kernels/readiness.metal".to_owned(),
                    source_sha256: manifest.readiness_source_sha256.clone(),
                    required_functions: vec!["zed_sim_metal_readiness_v1".to_owned()],
                    unsafe_owner: LOADER_UNSAFE_OWNER.to_owned(),
                },
                MetalMetallibContract {
                    identity: "metal-tensor-ops-metallib".to_owned(),
                    path: "kernels/tensor_ops.metallib".to_owned(),
                    sha256: manifest.tensor_ops_metallib_sha256.clone(),
                    source_path: "kernels/tensor_ops.metal".to_owned(),
                    source_sha256: manifest.tensor_ops_source_sha256.clone(),
                    required_functions: manifest.tensor_functions.clone(),
                    unsafe_owner: EXECUTION_UNSAFE_OWNER.to_owned(),
                },
            ],
            runtime_compilation_forbidden: true,
        };
        Ok((catalog, manifest, payloads))
    }

    fn write_fixture_coverage(root: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut rows = Vec::new();
        for relative in PACKAGE_PAYLOAD_LIMITS
            .iter()
            .map(NativePackagePayloadLimit::path)
        {
            if COVERAGE_EXCLUDES.contains(&relative) {
                continue;
            }
            let bytes = fs::read(root.join(relative))?;
            rows.push(format!(
                "{} {}  {relative}\n",
                sha256_hex(&bytes),
                bytes.len()
            ));
        }
        let coverage = rows.concat().into_bytes();
        fs::write(root.join("package-coverage.sha256"), &coverage)?;
        Ok(coverage)
    }

    fn encode_fixture_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn strict_json_rejects_duplicate_and_unknown_fields() {
        let duplicate = br#"{"schema_version":1,"schema_version":1}"#;
        assert!(parse_strict_json::<MetalFfiContractCatalogDto>(duplicate, "fixture").is_err());
        let unknown = br#"{"schema_version":1,"backend":"metal","abi_floor":"macos-13-metal-3","target":"aarch64-apple-darwin","minimum_macos_version":"13.0","metal_language_version":"metal3.0","minimum_metal_family":5001,"abi_manifest_sha256":"00","execution_abi_sha256":"00","frameworks":[],"classes":[],"resource_selectors":[],"layouts":[],"metallibs":[],"runtime_compilation_forbidden":true,"unknown":true}"#;
        assert!(parse_strict_json::<MetalFfiContractCatalogDto>(unknown, "fixture").is_err());
    }

    #[test]
    fn coverage_rejects_unsorted_duplicate_uncovered_and_tampered_rows() {
        let mut payloads = BTreeMap::new();
        for path in PACKAGE_PAYLOAD_LIMITS
            .iter()
            .map(NativePackagePayloadLimit::path)
        {
            payloads.insert(path.to_owned(), path.as_bytes().to_vec());
        }
        let valid = payloads
            .iter()
            .filter(|(path, _)| !COVERAGE_EXCLUDES.contains(&path.as_str()))
            .map(|(path, bytes)| format!("{} {}  {path}\n", sha256_hex(bytes), bytes.len()))
            .collect::<String>();
        assert!(
            validate_native_package_coverage(
                valid.as_bytes(),
                &payloads,
                &COVERAGE_EXCLUDES,
                MAX_COVERAGE_BYTES,
            )
            .is_ok()
        );
        let mut rows = valid.lines().collect::<Vec<_>>();
        rows.swap(0, 1);
        let unsorted = format!("{}\n", rows.join("\n"));
        assert!(
            validate_native_package_coverage(
                unsorted.as_bytes(),
                &payloads,
                &COVERAGE_EXCLUDES,
                MAX_COVERAGE_BYTES,
            )
            .is_err()
        );
        let duplicate = format!(
            "{}\n{}\n",
            valid.trim_end(),
            valid.lines().next().unwrap_or("")
        );
        assert!(
            validate_native_package_coverage(
                duplicate.as_bytes(),
                &payloads,
                &COVERAGE_EXCLUDES,
                MAX_COVERAGE_BYTES,
            )
            .is_err()
        );
        let uncovered = valid.replace("LICENSES\n", "unknown.bin\n");
        assert!(
            validate_native_package_coverage(
                uncovered.as_bytes(),
                &payloads,
                &COVERAGE_EXCLUDES,
                MAX_COVERAGE_BYTES,
            )
            .is_err()
        );
        let tampered = valid.replacen(&sha256_hex(b"LICENSES"), &"0".repeat(64), 1);
        assert!(
            validate_native_package_coverage(
                tampered.as_bytes(),
                &payloads,
                &COVERAGE_EXCLUDES,
                MAX_COVERAGE_BYTES,
            )
            .is_err()
        );
    }

    #[test]
    fn exact_catalog_maps_once_to_canonical_registry_and_rejects_every_contract_mismatch()
    -> Result<(), Box<dyn std::error::Error>> {
        let (catalog, manifest, payloads) = fixture_contracts()?;
        let (registry, provenance) = validate_and_map_catalog(&catalog, &manifest, &payloads)?;
        assert_eq!(provenance.len(), 3);
        for framework in &catalog.frameworks {
            let digest = provenance
                .get(&framework.identity)
                .ok_or("missing framework provenance digest")?;
            let available = framework.required_symbols.iter().cloned().collect();
            registry.authorize(&framework.identity, digest, ABI_FLOOR, &available)?;
        }
        for metallib in &catalog.metallibs {
            let abi = if metallib.identity == "metal-tensor-ops-metallib" {
                EXECUTION_CONTRACT
            } else {
                ABI_FLOOR
            };
            registry.authorize(
                &metallib.identity,
                &metallib.sha256,
                abi,
                &metallib.required_functions.iter().cloned().collect(),
            )?;
        }

        let mut failures = Vec::new();
        let mut unsorted = catalog.clone();
        unsorted.frameworks.swap(0, 1);
        failures.push(unsorted);
        let mut duplicate = catalog.clone();
        duplicate.frameworks[1] = duplicate.frameworks[0].clone();
        failures.push(duplicate);
        let mut wrong_target = catalog.clone();
        wrong_target.target = "x86_64-apple-darwin".to_owned();
        failures.push(wrong_target);
        let mut wrong_abi = catalog.clone();
        wrong_abi.execution_abi_sha256 = "0".repeat(64);
        failures.push(wrong_abi);
        let mut wrong_selector = catalog.clone();
        wrong_selector.resource_selectors[0]["encoding"] = Value::String("v0".to_owned());
        failures.push(wrong_selector);
        let mut wrong_owner = catalog.clone();
        wrong_owner.metallibs[1].unsafe_owner = LOADER_UNSAFE_OWNER.to_owned();
        failures.push(wrong_owner);
        let mut wrong_digest = catalog.clone();
        wrong_digest.metallibs[0].sha256 = "0".repeat(64);
        failures.push(wrong_digest);
        let mut wrong_functions = catalog;
        wrong_functions.metallibs[1].required_functions.reverse();
        failures.push(wrong_functions);
        for invalid in failures {
            assert!(validate_and_map_catalog(&invalid, &manifest, &payloads).is_err());
        }
        Ok(())
    }

    #[test]
    fn public_verifier_accepts_a_complete_signed_deterministic_fixture_and_checks_signature_before_catalog()
    -> Result<(), Box<dyn std::error::Error>> {
        let (catalog, mut manifest, payloads) = fixture_contracts()?;
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        fs::create_dir(root.join("abi"))?;
        fs::create_dir(root.join("kernels"))?;
        fs::write(root.join("abi/symbols-v1.json"), ABI_MANIFEST)?;
        fs::write(root.join("abi/execution-v1.json"), EXECUTION_ABI)?;
        fs::write(
            root.join("abi/reviewed-execution-bindings-v1.txt"),
            b"reviewed fixture bindings\n",
        )?;
        fs::write(root.join("LICENSES"), b"fixture license\n")?;
        fs::write(
            root.join("LICENSES.execution"),
            b"fixture execution license\n",
        )?;
        fs::write(root.join("execution-policy.json"), b"{}\n")?;
        for (path, bytes) in &payloads {
            fs::write(root.join(path), bytes)?;
        }
        let catalog_bytes = serde_json::to_vec(&catalog)?;
        manifest.ffi_contracts_sha256 = sha256_hex(&catalog_bytes);
        fs::write(root.join("ffi-contracts-v1.json"), &catalog_bytes)?;
        fs::write(
            root.join("adapter-manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        let policy = serde_json::to_vec(&serde_json::json!({
            "abi_manifest_sha256": &manifest.abi_manifest_sha256,
            "execution_abi_sha256": &manifest.execution_abi_sha256,
            "readiness_source_sha256": &manifest.readiness_source_sha256,
            "readiness_metallib_sha256_by_target": {
                "aarch64-apple-darwin": &manifest.readiness_metallib_sha256,
            },
            "tensor_ops_source_sha256": &manifest.tensor_ops_source_sha256,
            "tensor_ops_metallib_sha256_by_target": {
                "aarch64-apple-darwin": &manifest.tensor_ops_metallib_sha256,
            },
            "reviewed_execution_bindings_sha256": sha256_hex(b"reviewed fixture bindings\n"),
            "execution_policy_sha256": sha256_hex(b"{}\n"),
            "license_notice_sha256": sha256_hex(b"fixture license\n"),
            "execution_license_notice_sha256": sha256_hex(b"fixture execution license\n"),
        }))?;
        fs::write(root.join("package-policy.json"), &policy)?;
        fs::write(root.join("adapter-manifest.sig"), b"pending\n")?;
        let coverage = write_fixture_coverage(root)?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(b"0123456789abcdef0123456789abcdef")
            .map_err(|error| io::Error::other(format!("fixture key rejected: {error:?}")))?;
        let key =
            MetalPackageVerificationKey::new("metal.release", key_pair.public_key().as_ref())?;
        let signing_payload =
            crate::trust::metal_package_signing_payload("metal.release", &coverage)?;
        let signature = encode_fixture_hex(key_pair.sign(&signing_payload).as_ref());
        fs::write(
            root.join("adapter-manifest.sig"),
            format!(
                "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{signature}\"}}\n"
            ),
        )?;

        let _policy_override = ReviewedPolicyOverrideGuard::install(policy);
        let verified = verify_metal_package_contracts(root, &key, &CancellationToken::default())?;
        assert_eq!(verified.target(), "aarch64-apple-darwin");
        assert_eq!(verified.frameworks().len(), 3);
        assert_eq!(verified.readiness_metallib(), b"readiness metallib");
        assert_eq!(verified.tensor_ops_metallib(), b"tensor metallib");
        let certified = certify_verified_metal_inputs(&verified)?;
        assert_eq!(certified.certificates.len(), 5);
        assert_eq!(
            certified
                .certificates
                .iter()
                .map(crate::CertifiedNativeFfi::library_id)
                .collect::<Vec<_>>(),
            [
                "metal-framework",
                "metal-performance-shaders-framework",
                "metal-performance-shaders-graph-framework",
                "metal-readiness-metallib",
                "metal-tensor-ops-metallib",
            ]
        );
        assert_eq!(
            certified
                .certificates
                .last()
                .ok_or("missing tensor-operations certificate")?
                .unsafe_owner(),
            EXECUTION_UNSAFE_OWNER
        );
        let second = verify_metal_package_contracts(root, &key, &CancellationToken::default())?;
        assert_eq!(
            verified.framework_provenance_digest("metal-framework"),
            second.framework_provenance_digest("metal-framework")
        );

        fs::write(root.join("LICENSES"), b"changed fixture license\n")?;
        let changed_coverage = write_fixture_coverage(root)?;
        let changed_payload =
            crate::trust::metal_package_signing_payload("metal.release", &changed_coverage)?;
        let changed_signature = encode_fixture_hex(key_pair.sign(&changed_payload).as_ref());
        fs::write(
            root.join("adapter-manifest.sig"),
            format!(
                "{{\"schema_version\":1,\"algorithm\":\"ed25519\",\"signature\":\"{changed_signature}\"}}\n"
            ),
        )?;
        assert!(matches!(
            verify_metal_package_contracts(root, &key, &CancellationToken::default()),
            Err(MetalPackageContractError::InvalidPackage(reason))
                if reason.contains("reviewed policy digest does not match LICENSES")
        ));
        fs::write(root.join("LICENSES"), b"fixture license\n")?;

        fs::write(root.join("ffi-contracts-v1.json"), b"{malformed")?;
        manifest.ffi_contracts_sha256 = sha256_hex(b"{malformed");
        fs::write(
            root.join("adapter-manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        write_fixture_coverage(root)?;
        assert!(matches!(
            verify_metal_package_contracts(root, &key, &CancellationToken::default()),
            Err(MetalPackageContractError::Trust(
                TrustError::InvalidMetalPackageSignature
            ))
        ));
        Ok(())
    }

    #[test]
    fn macos_floor_parser_is_bounded_and_fail_closed() {
        assert_eq!(parse_macos_major_version(b"13.0"), Ok(13));
        assert_eq!(parse_macos_major_version(b"26.1.2"), Ok(26));
        for invalid in [
            b"".as_slice(),
            b"0".as_slice(),
            b"013.0".as_slice(),
            b"13.".as_slice(),
            b"13.beta".as_slice(),
            b"1234.0".as_slice(),
            &[b'1'; 33],
        ] {
            assert_eq!(parse_macos_major_version(invalid), Err(()));
        }
        assert!(parse_macos_major_version(b"12.6").is_ok());
        assert_eq!(validate_macos_version_observation(0, 5, b"13.0\0"), Ok(()));
        assert_eq!(validate_macos_version_observation(0, 4, b"12.6"), Err(()));
        assert_eq!(validate_macos_version_observation(-1, 4, b"13.0"), Err(()));
        assert_eq!(validate_macos_version_observation(0, 9, b"13.0"), Err(()));
        assert_eq!(validate_host_memory_observation(0, 8, 16), Ok(16));
        assert_eq!(validate_host_memory_observation(-1, 8, 16), Err(()));
        assert_eq!(validate_host_memory_observation(0, 4, 16), Err(()));
        assert_eq!(validate_host_memory_observation(0, 8, 0), Err(()));
    }

    #[test]
    fn device_identity_drift_is_rejected_exactly() {
        let expected = sim_backend_metal::MetalDeviceProbe {
            name: "fixture Metal".to_owned(),
            registry_id: 7,
            recommended_working_set_bytes: 20,
            unified_memory: true,
            metal_3: true,
            mps_supported: true,
        };
        assert!(metal_device_identity_matches(
            "fixture Metal",
            7,
            20,
            true,
            &expected
        ));
        assert!(!metal_device_identity_matches(
            "other Metal",
            7,
            20,
            true,
            &expected
        ));
        assert!(!metal_device_identity_matches(
            "fixture Metal",
            8,
            20,
            true,
            &expected
        ));
        assert!(!metal_device_identity_matches(
            "fixture Metal",
            7,
            19,
            true,
            &expected
        ));
        assert!(!metal_device_identity_matches(
            "fixture Metal",
            7,
            20,
            false,
            &expected
        ));

        let source = include_str!("native_ffi_metal.rs");
        let certification = source
            .find("certify_verified_metal_inputs(&verified)")
            .expect("registry certification call");
        let os_floor = source
            .find("require_supported_macos_version()")
            .expect("macOS floor call");
        let device_probe = source
            .find("let device_probe = probe_device()")
            .expect("device probe");
        let runtime = source
            .find("MetalRuntime::from_certified_metallibs")
            .expect("certified runtime construction");
        assert!(certification < os_floor);
        assert!(os_floor < device_probe);
        assert!(device_probe < runtime);
    }
}
