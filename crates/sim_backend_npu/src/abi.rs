use std::ffi::{c_char, c_void};

use serde::Deserialize;
use thiserror::Error;

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "CANN-8.0.RC3-AscendCL";
pub const UNSAFE_OWNER: &str = "sim_backend_npu::loader";
pub(crate) const ACL_SUCCESS: AclError = 0;

pub type AclError = i32;
pub type AclrtContext = *mut c_void;
pub type AclrtStream = *mut c_void;
pub type AclrtEvent = *mut c_void;
pub type AclTensorDesc = *mut c_void;
pub type AclDataBuffer = *mut c_void;
pub type AclOpAttr = *mut c_void;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AclDataType {
    Float = 0,
    Float16 = 1,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AclFormat {
    Nd = 2,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AclrtMemAttr {
    Ddr = 0,
    Hbm = 1,
    DdrHuge = 2,
    DdrNormal = 3,
    HbmHuge = 4,
    HbmNormal = 5,
    DdrP2pHuge = 6,
    DdrP2pNormal = 7,
    HbmP2pHuge = 8,
    HbmP2pNormal = 9,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AclrtMemMallocPolicy {
    HugeFirst = 0,
    HugeOnly = 1,
    NormalOnly = 2,
    HugeFirstP2p = 3,
    HugeOnlyP2p = 4,
    NormalOnlyP2p = 5,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AclrtMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

pub type AclInit = unsafe extern "C" fn(*const c_char) -> AclError;
pub type AclFinalize = unsafe extern "C" fn() -> AclError;
pub type AclGetRecentErrMsg = unsafe extern "C" fn() -> *const c_char;
pub type AclrtGetVersion = unsafe extern "C" fn(*mut i32, *mut i32, *mut i32) -> AclError;
pub type AclrtGetDeviceCount = unsafe extern "C" fn(*mut u32) -> AclError;
pub type AclrtSetDevice = unsafe extern "C" fn(i32) -> AclError;
pub type AclrtResetDevice = unsafe extern "C" fn(i32) -> AclError;
pub type AclrtCreateContext = unsafe extern "C" fn(*mut AclrtContext, i32) -> AclError;
pub type AclrtDestroyContext = unsafe extern "C" fn(AclrtContext) -> AclError;
pub type AclrtSetCurrentContext = unsafe extern "C" fn(AclrtContext) -> AclError;
pub type AclrtCreateStream = unsafe extern "C" fn(*mut AclrtStream) -> AclError;
pub type AclrtDestroyStream = unsafe extern "C" fn(AclrtStream) -> AclError;
pub type AclrtSynchronizeStream = unsafe extern "C" fn(AclrtStream) -> AclError;
pub type AclrtCreateEvent = unsafe extern "C" fn(*mut AclrtEvent) -> AclError;
pub type AclrtDestroyEvent = unsafe extern "C" fn(AclrtEvent) -> AclError;
pub type AclrtRecordEvent = unsafe extern "C" fn(AclrtEvent, AclrtStream) -> AclError;
pub type AclrtSynchronizeEvent = unsafe extern "C" fn(AclrtEvent) -> AclError;
pub type AclrtMalloc =
    unsafe extern "C" fn(*mut *mut c_void, usize, AclrtMemMallocPolicy) -> AclError;
pub type AclrtFree = unsafe extern "C" fn(*mut c_void) -> AclError;
pub type AclrtMemcpy =
    unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, AclrtMemcpyKind) -> AclError;
pub type AclrtMemcpyAsync = unsafe extern "C" fn(
    *mut c_void,
    usize,
    *const c_void,
    usize,
    AclrtMemcpyKind,
    AclrtStream,
) -> AclError;
pub type AclrtGetMemInfo = unsafe extern "C" fn(AclrtMemAttr, *mut usize, *mut usize) -> AclError;
pub type AclrtGetSocName = unsafe extern "C" fn() -> *const c_char;
pub type AclCreateTensorDesc =
    unsafe extern "C" fn(AclDataType, i32, *const i64, AclFormat) -> AclTensorDesc;
pub type AclDestroyTensorDesc = unsafe extern "C" fn(AclTensorDesc);
pub type AclCreateDataBuffer = unsafe extern "C" fn(*mut c_void, usize) -> AclDataBuffer;
pub type AclDestroyDataBuffer = unsafe extern "C" fn(AclDataBuffer) -> AclError;
pub type AclopExecuteV2 = unsafe extern "C" fn(
    *const c_char,
    i32,
    *mut AclTensorDesc,
    *mut AclDataBuffer,
    i32,
    *mut AclTensorDesc,
    *mut AclDataBuffer,
    AclOpAttr,
    AclrtStream,
) -> AclError;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CannVersion {
    pub major: u32,
    pub minor: u32,
    pub release_candidate: u32,
}

impl CannVersion {
    pub const FLOOR: Self = Self {
        major: 8,
        minor: 0,
        release_candidate: 3,
    };

    pub fn parse(value: &str) -> Result<Self, AbiManifestError> {
        let value = value.strip_prefix("CANN-").unwrap_or(value);
        let (version, release_candidate) = value
            .split_once(".RC")
            .ok_or_else(|| AbiManifestError::Version(value.to_owned()))?;
        let (major, minor) = version
            .split_once('.')
            .ok_or_else(|| AbiManifestError::Version(value.to_owned()))?;
        let release_candidate = release_candidate
            .strip_suffix("-AscendCL")
            .unwrap_or(release_candidate);
        Ok(Self {
            major: major
                .parse()
                .map_err(|_| AbiManifestError::Version(value.to_owned()))?,
            minor: minor
                .parse()
                .map_err(|_| AbiManifestError::Version(value.to_owned()))?,
            release_candidate: release_candidate
                .parse()
                .map_err(|_| AbiManifestError::Version(value.to_owned()))?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbiManifest {
    pub schema_version: u32,
    pub backend: String,
    pub abi_floor: String,
    pub targets: Vec<String>,
    pub discovery_order: Vec<String>,
    pub libraries: Vec<LibraryContract>,
    pub layouts: Vec<LayoutContract>,
    pub headers: Vec<HeaderContract>,
    pub review_artifact: ReviewArtifactContract,
    pub package: PackageContract,
    pub unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LibraryContract {
    pub id: String,
    pub filename: String,
    pub relative_path: String,
    pub dependency_only: bool,
    pub symbols: Vec<SymbolContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SymbolContract {
    pub name: String,
    pub signature: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayoutContract {
    pub name: String,
    pub size: usize,
    pub align: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HeaderContract {
    pub path: String,
    pub digest_kind: String,
    pub sha256: String,
    pub source_section_sha256: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReviewArtifactContract {
    pub path: String,
    pub sha256: String,
    pub source_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PackageContract {
    pub redistributes_cann: bool,
    pub license_approval_required_for_redistribution: bool,
    pub signature_algorithm: String,
    pub signature_domain: String,
    pub coverage: String,
    pub final_application_signing_required: bool,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AbiManifestError {
    #[error("NPU ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("NPU ABI manifest violates the reviewed contract: {0}")]
    Contract(String),
    #[error("invalid CANN version: {0}")]
    Version(String),
}

impl AbiManifest {
    pub fn embedded() -> Result<Self, AbiManifestError> {
        Self::parse(ABI_MANIFEST_JSON)
    }

    pub fn parse(json: &str) -> Result<Self, AbiManifestError> {
        let manifest: Self = serde_json::from_str(json)
            .map_err(|error| AbiManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn symbol_count(&self) -> usize {
        self.libraries
            .iter()
            .map(|library| library.symbols.len())
            .sum()
    }

    pub fn validate(&self) -> Result<(), AbiManifestError> {
        if self.schema_version != 1
            || self.backend != "npu"
            || self.abi_floor != ABI_FLOOR
            || self.unsafe_owner != UNSAFE_OWNER
            || CannVersion::parse(&self.abi_floor)? != CannVersion::FLOOR
        {
            return Err(AbiManifestError::Contract(
                "identity, ABI floor, or unsafe owner differs".to_owned(),
            ));
        }
        require_exact(
            &self.targets.iter().map(String::as_str).collect::<Vec<_>>(),
            &["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"],
            "targets",
        )?;
        require_exact(
            &self
                .discovery_order
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            &[
                "SIM_ASCEND_ROOT",
                "ASCEND_HOME_PATH",
                "signed_package_roots",
            ],
            "discovery order",
        )?;
        if self.libraries.len() != 2
            || self.libraries[0].id != "ascendcl"
            || self.libraries[0].filename != "libascendcl.so"
            || self.libraries[0].relative_path != "lib64/libascendcl.so"
            || self.libraries[0].dependency_only
            || self.libraries[1].id != "runtime"
            || self.libraries[1].filename != "libruntime.so"
            || self.libraries[1].relative_path != "lib64/libruntime.so"
            || !self.libraries[1].dependency_only
            || !self.libraries[1].symbols.is_empty()
        {
            return Err(AbiManifestError::Contract(
                "library set or fixed paths differ".to_owned(),
            ));
        }
        let names = self.libraries[0]
            .symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>();
        require_exact(&names, &REQUIRED_ASCENDCL_SYMBOLS, "AscendCL symbols")?;
        let signatures = self.libraries[0]
            .symbols
            .iter()
            .map(|symbol| symbol.signature.as_str())
            .collect::<Vec<_>>();
        require_exact(
            &signatures,
            &REQUIRED_ASCENDCL_SIGNATURES,
            "AscendCL signatures",
        )?;
        let expected_layouts = [
            ("aclError", 4, 4),
            ("aclDataBuffer", 8, 8),
            ("aclDataType", 4, 4),
            ("aclFormat", 4, 4),
            ("aclTensorDesc", 8, 8),
            ("aclrtContext", 8, 8),
            ("aclrtEvent", 8, 8),
            ("aclrtMemMallocPolicy", 4, 4),
            ("aclrtMemAttr", 4, 4),
            ("aclrtMemcpyKind", 4, 4),
            ("aclrtStream", 8, 8),
        ];
        if self.layouts.len() != expected_layouts.len()
            || self
                .layouts
                .iter()
                .zip(expected_layouts)
                .any(|(actual, expected)| {
                    actual.name != expected.0
                        || actual.size != expected.1
                        || actual.align != expected.2
                })
        {
            return Err(AbiManifestError::Contract(
                "64-bit C layout manifest differs".to_owned(),
            ));
        }
        let allocation_policy_values = [
            AclrtMemMallocPolicy::HugeFirst as i32,
            AclrtMemMallocPolicy::HugeOnly as i32,
            AclrtMemMallocPolicy::NormalOnly as i32,
            AclrtMemMallocPolicy::HugeFirstP2p as i32,
            AclrtMemMallocPolicy::HugeOnlyP2p as i32,
            AclrtMemMallocPolicy::NormalOnlyP2p as i32,
        ];
        let memcpy_kind_values = [
            AclrtMemcpyKind::HostToHost as i32,
            AclrtMemcpyKind::HostToDevice as i32,
            AclrtMemcpyKind::DeviceToHost as i32,
            AclrtMemcpyKind::DeviceToDevice as i32,
        ];
        let memory_attribute_values = [
            AclrtMemAttr::Ddr as i32,
            AclrtMemAttr::Hbm as i32,
            AclrtMemAttr::DdrHuge as i32,
            AclrtMemAttr::DdrNormal as i32,
            AclrtMemAttr::HbmHuge as i32,
            AclrtMemAttr::HbmNormal as i32,
            AclrtMemAttr::DdrP2pHuge as i32,
            AclrtMemAttr::DdrP2pNormal as i32,
            AclrtMemAttr::HbmP2pHuge as i32,
            AclrtMemAttr::HbmP2pNormal as i32,
        ];
        if allocation_policy_values != [0, 1, 2, 3, 4, 5] || memcpy_kind_values != [0, 1, 2, 3] {
            return Err(AbiManifestError::Contract(
                "reviewed AscendCL enum values differ".to_owned(),
            ));
        }
        if memory_attribute_values != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            || AclDataType::Float as i32 != 0
            || AclDataType::Float16 as i32 != 1
            || AclFormat::Nd as i32 != 2
        {
            return Err(AbiManifestError::Contract(
                "reviewed AscendCL execution enum values differ".to_owned(),
            ));
        }
        let expected_headers = [
            (
                "runtime/include/acl/acl.h",
                "1cab4286a330cfb10337e6fde6ffcc7390a4855ea143db63e84d497d185391bc",
                "8d745b66e690ac8d10aabb952e10ffdbc212de835e014b0ada0afbf17718c898",
            ),
            (
                "runtime/include/acl/acl_rt.h",
                "be4f6d8ec73bf30cdf1ba42ceea8aeb3f1aebd8900192537ce6a5f80f317e765",
                "2ad93a44e0a829abab940bd471b078414edab9cd3b1789fd1ecc595d376408ba",
            ),
            (
                "runtime/include/acl/acl_base.h",
                "2b2eeaf361f9e26fee8b0d30c4cede7c586e08090487b60d7e97fd42445e9c3c",
                "4dafca0edb13d12d5cdf8b2a61e852ec189b3fd34ed705851b8d0839ee9ca91f",
            ),
            (
                "runtime/include/acl/acl_op_compiler.h",
                "68d31a009dc580773ce7f568ec1082a0120864e15401f17ed3a9207bef3eabb1",
                "ac404bc76e96a7b091350c51fe40a8f75ee7032d0d0c29e5a472dc62969ea8f0",
            ),
        ];
        if self.headers.len() != expected_headers.len()
            || self.headers.iter().zip(expected_headers).any(
                |(header, (expected_path, expected_digest, expected_source_section_digest))| {
                    header.path != expected_path
                        || header.digest_kind != "sorted-reviewed-signatures-v1"
                        || header.sha256 != expected_digest
                        || header.source_section_sha256 != expected_source_section_digest
                        || header.source
                            != "https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/apiref/appdevgapi/CANN%208.0.RC3%20%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3%2001.pdf"
                },
            )
        {
            return Err(AbiManifestError::Contract(
                "reviewed header evidence is incomplete".to_owned(),
            ));
        }
        if self.review_artifact.path != "abi/reviewed-bindings-v1.txt"
            || self.review_artifact.sha256
                != "0b4481f131bfa8b311ee6e1f7a926eb3fdcfffc0e0165fb64ed4fd8e4036cb81"
            || self.review_artifact.source_sha256
                != "91d8bd8a346bda371c8175066ac5155fb27ccfe4ba63091730ec29dcd96dd091"
        {
            return Err(AbiManifestError::Contract(
                "reviewed execution declaration artifact differs".to_owned(),
            ));
        }
        if self.package.redistributes_cann
            || !self.package.license_approval_required_for_redistribution
            || self.package.signature_algorithm != "ed25519"
            || self.package.signature_domain != "zed-sim-npu-package-v1"
            || self.package.coverage != "package-coverage-v1"
            || !self.package.final_application_signing_required
        {
            return Err(AbiManifestError::Contract(
                "package license/signature policy differs".to_owned(),
            ));
        }
        Ok(())
    }
}

pub const REQUIRED_ASCENDCL_SYMBOLS: [&str; 28] = [
    "aclCreateDataBuffer",
    "aclCreateTensorDesc",
    "aclDestroyDataBuffer",
    "aclDestroyTensorDesc",
    "aclFinalize",
    "aclGetRecentErrMsg",
    "aclInit",
    "aclopExecuteV2",
    "aclrtCreateContext",
    "aclrtCreateEvent",
    "aclrtCreateStream",
    "aclrtDestroyContext",
    "aclrtDestroyEvent",
    "aclrtDestroyStream",
    "aclrtFree",
    "aclrtGetDeviceCount",
    "aclrtGetMemInfo",
    "aclrtGetSocName",
    "aclrtGetVersion",
    "aclrtMalloc",
    "aclrtMemcpy",
    "aclrtMemcpyAsync",
    "aclrtRecordEvent",
    "aclrtResetDevice",
    "aclrtSetCurrentContext",
    "aclrtSetDevice",
    "aclrtSynchronizeEvent",
    "aclrtSynchronizeStream",
];

const REQUIRED_ASCENDCL_SIGNATURES: [&str; 28] = [
    "aclDataBuffer *aclCreateDataBuffer(void *data, size_t size)",
    "aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format)",
    "aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)",
    "void aclDestroyTensorDesc(const aclTensorDesc *desc)",
    "aclError aclFinalize(void)",
    "const char *aclGetRecentErrMsg(void)",
    "aclError aclInit(const char *configPath)",
    "aclError aclopExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[], int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[], aclopAttr *attr, aclrtStream stream)",
    "aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId)",
    "aclError aclrtCreateEvent(aclrtEvent *event)",
    "aclError aclrtCreateStream(aclrtStream *stream)",
    "aclError aclrtDestroyContext(aclrtContext context)",
    "aclError aclrtDestroyEvent(aclrtEvent event)",
    "aclError aclrtDestroyStream(aclrtStream stream)",
    "aclError aclrtFree(void *devPtr)",
    "aclError aclrtGetDeviceCount(uint32_t *count)",
    "aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total)",
    "const char *aclrtGetSocName(void)",
    "aclError aclrtGetVersion(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion)",
    "aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)",
    "aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)",
    "aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)",
    "aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)",
    "aclError aclrtResetDevice(int32_t deviceId)",
    "aclError aclrtSetCurrentContext(aclrtContext context)",
    "aclError aclrtSetDevice(int32_t deviceId)",
    "aclError aclrtSynchronizeEvent(aclrtEvent event)",
    "aclError aclrtSynchronizeStream(aclrtStream stream)",
];

fn require_exact<T: PartialEq + std::fmt::Debug>(
    actual: &[T],
    expected: &[T],
    label: &str,
) -> Result<(), AbiManifestError> {
    if actual != expected {
        return Err(AbiManifestError::Contract(format!(
            "{label} must equal the reviewed ordered set"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewed_c_layouts_match_rust_on_supported_width() {
        assert_eq!(size_of::<AclError>(), 4);
        assert_eq!(align_of::<AclError>(), 4);
        assert_eq!(size_of::<AclrtContext>(), 8);
        assert_eq!(align_of::<AclrtContext>(), 8);
        assert_eq!(size_of::<AclrtEvent>(), 8);
        assert_eq!(align_of::<AclrtEvent>(), 8);
        assert_eq!(size_of::<AclrtStream>(), 8);
        assert_eq!(align_of::<AclrtStream>(), 8);
        assert_eq!(size_of::<AclrtMemMallocPolicy>(), 4);
        assert_eq!(align_of::<AclrtMemMallocPolicy>(), 4);
        assert_eq!(size_of::<AclrtMemcpyKind>(), 4);
        assert_eq!(align_of::<AclrtMemcpyKind>(), 4);
        assert_eq!(size_of::<AclrtMemAttr>(), 4);
        assert_eq!(align_of::<AclrtMemAttr>(), 4);
        assert_eq!(size_of::<AclDataType>(), 4);
        assert_eq!(align_of::<AclDataType>(), 4);
        assert_eq!(size_of::<AclFormat>(), 4);
        assert_eq!(align_of::<AclFormat>(), 4);
    }

    #[test]
    fn cann_floor_parser_is_ordered_and_strict() -> Result<(), AbiManifestError> {
        assert_eq!(CannVersion::parse(ABI_FLOOR)?, CannVersion::FLOOR);
        assert!(CannVersion::parse("CANN-8.0").is_err());
        assert!(CannVersion::parse("CANN-8.0.RCx").is_err());
        assert!(CannVersion::parse("CANN-7.0.RC9")? < CannVersion::FLOOR);
        Ok(())
    }
}
