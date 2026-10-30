use serde::Deserialize;
use std::ffi::{c_int, c_uint, c_void};
use thiserror::Error;

pub const ABI_MANIFEST: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "neuware-1.20-cnnl-1.20.4-cnrt-6.6.0";
pub const UNSAFE_OWNER: &str = "sim_backend_mlu::loader";
pub const CERTIFICATE_OWNER: &str = "sim_runtime::NativeFfiRegistry";

const TARGETS: [&str; 2] = ["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"];
const DISCOVERY_ORDER: [&str; 3] = ["SIM_MLU_ROOT", "NEUWARE_HOME", "signed_package_roots"];
const CNNL_SYMBOLS: [(&str, &str); 11] = [
    ("cnnlCreate", "cnnlStatus_t(cnnlHandle_t*)"),
    (
        "cnnlCreateOpTensorDescriptor",
        "cnnlStatus_t(cnnlOpTensorDescriptor_t*)",
    ),
    (
        "cnnlCreateTensorDescriptor",
        "cnnlStatus_t(cnnlTensorDescriptor_t*)",
    ),
    ("cnnlDestroy", "cnnlStatus_t(cnnlHandle_t)"),
    (
        "cnnlDestroyOpTensorDescriptor",
        "cnnlStatus_t(cnnlOpTensorDescriptor_t)",
    ),
    (
        "cnnlDestroyTensorDescriptor",
        "cnnlStatus_t(cnnlTensorDescriptor_t)",
    ),
    ("cnnlGetLibVersion", "void(int*,int*,int*)"),
    (
        "cnnlOpTensor",
        "cnnlStatus_t(cnnlHandle_t,const cnnlOpTensorDescriptor_t,const void*,const cnnlTensorDescriptor_t,const void*,const void*,const cnnlTensorDescriptor_t,const void*,void*,size_t,const void*,const cnnlTensorDescriptor_t,void*)",
    ),
    (
        "cnnlSetOpTensorDescriptor",
        "cnnlStatus_t(cnnlOpTensorDescriptor_t,cnnlOpTensorDesc_t,cnnlDataType_t,cnnlNanPropagation_t)",
    ),
    ("cnnlSetQueue", "cnnlStatus_t(cnnlHandle_t,cnrtQueue_t)"),
    (
        "cnnlSetTensorDescriptor",
        "cnnlStatus_t(cnnlTensorDescriptor_t,cnnlTensorLayout_t,cnnlDataType_t,int,const int[])",
    ),
];
const CNRT_SYMBOLS: [(&str, &str); 9] = [
    ("cnrtFree", "cnrtRet_t(void*)"),
    ("cnrtGetDeviceCount", "cnrtRet_t(unsigned int*)"),
    ("cnrtGetLibVersion", "cnrtRet_t(int*,int*,int*)"),
    ("cnrtMalloc", "cnrtRet_t(void**,size_t)"),
    (
        "cnrtMemcpy",
        "cnrtRet_t(void*,void*,size_t,cnrtMemTransDir_t)",
    ),
    ("cnrtQueueCreate", "cnrtRet_t(cnrtQueue_t*)"),
    ("cnrtQueueDestroy", "cnrtRet_t(cnrtQueue_t)"),
    ("cnrtQueueSync", "cnrtRet_t(cnrtQueue_t)"),
    ("cnrtSetDevice", "cnrtRet_t(int)"),
];
const LAYOUTS: [(&str, usize, usize); 8] = [
    ("cnnlHandle_t", 8, 8),
    ("cnnlOpTensorDescriptor_t", 8, 8),
    ("cnnlStatus_t", 4, 4),
    ("cnnlTensorDescriptor_t", 8, 8),
    ("cnrtMemTransDir_t", 4, 4),
    ("cnrtNotifier_t", 8, 8),
    ("cnrtQueue_t", 8, 8),
    ("cnrtRet_t", 4, 4),
];
const REVIEWED_ENUM_VALUES: [(&str, &str, i32); 10] = [
    ("cnrtRet_t", "cnrtSuccess", 0),
    ("cnrtRet_t", "cnrtErrorNoDevice", 100_004),
    ("cnrtRet_t", "cnrtErrorNoMem", 100_100),
    ("cnnlStatus_t", "CNNL_STATUS_SUCCESS", 0),
    ("cnnlStatus_t", "CNNL_STATUS_ALLOC_FAILED", 2),
    ("cnnlTensorLayout_t", "CNNL_LAYOUT_ARRAY", 4),
    ("cnnlDataType_t", "CNNL_DTYPE_HALF", 1),
    ("cnnlDataType_t", "CNNL_DTYPE_FLOAT", 2),
    ("cnnlOpTensorDesc_t", "CNNL_OP_TENSOR_ADD", 0),
    ("cnnlNanPropagation_t", "CNNL_PROPAGATE_NAN", 1),
];
const MEMORY_TRANSFER_DECLARATION: &str = "typedef enum { cnrtMemcpyHostToDev = 0, cnrtMemcpyDevToDev, cnrtMemcpyDevToHost, cnrtMemcpyHostToHost, cnrtMemcpyPeerToPeer, cnrtMemcpyNoDirection } cnrtMemTransDir_t;";
const MEMORY_TRANSFER_VARIANTS: [(&str, i32); 6] = [
    ("cnrtMemcpyHostToDev", 0),
    ("cnrtMemcpyDevToDev", 1),
    ("cnrtMemcpyDevToHost", 2),
    ("cnrtMemcpyHostToHost", 3),
    ("cnrtMemcpyPeerToPeer", 4),
    ("cnrtMemcpyNoDirection", 5),
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbiManifest {
    pub schema_version: u32,
    pub backend: String,
    pub abi_floor: String,
    pub targets: Vec<String>,
    pub discovery_order: Vec<String>,
    pub libraries: Vec<LibraryContract>,
    pub reviewed_enums: Vec<EnumContract>,
    pub reviewed_enum_values: Vec<EnumValueContract>,
    pub layouts: Vec<LayoutContract>,
    pub unsafe_owner: String,
    pub certificate_owner: String,
    pub package_policy: PackagePolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LibraryContract {
    pub id: String,
    pub filename: String,
    pub header: String,
    pub header_sha256: String,
    pub package: String,
    pub package_sha256: String,
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
pub struct EnumContract {
    pub name: String,
    pub header: String,
    pub normalized_declaration: String,
    pub variants: Vec<EnumVariantContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnumVariantContract {
    pub name: String,
    pub value: i32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnumValueContract {
    pub type_name: String,
    pub name: String,
    pub value: i32,
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
pub struct PackagePolicy {
    pub redistribute_vendor_runtime: bool,
    pub license_approval_required: bool,
    pub signature_algorithm: String,
    pub signature_domain: String,
    pub runtime_compilation_forbidden: bool,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AbiManifestError {
    #[error("MLU ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("MLU ABI manifest violates the reviewed contract: {0}")]
    Contract(String),
}

impl AbiManifest {
    pub fn embedded() -> Result<Self, AbiManifestError> {
        let manifest: Self = serde_json::from_str(ABI_MANIFEST)
            .map_err(|error| AbiManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), AbiManifestError> {
        if self.schema_version != 1
            || self.backend != "mlu"
            || self.abi_floor != ABI_FLOOR
            || self.unsafe_owner != UNSAFE_OWNER
            || self.certificate_owner != CERTIFICATE_OWNER
        {
            return Err(contract(
                "identity, ABI floor, or authoritative owner differs",
            ));
        }
        require_exact_strings(&self.targets, &TARGETS, "targets")?;
        require_exact_strings(&self.discovery_order, &DISCOVERY_ORDER, "discovery order")?;
        if self.libraries.len() != 2
            || self.libraries[0].id != "cnnl"
            || self.libraries[1].id != "cnrt"
        {
            return Err(contract("libraries must be exactly sorted cnnl then cnrt"));
        }
        validate_library(
            &self.libraries[0],
            "libcnnl.so",
            "include/cnnl.h",
            "69eaa4fa560fa02e1e636a0d19917244a2badd94a960721e57fa36059f6fa031",
            "cnnl_1.20.4-1.ubuntu20.04_amd64.deb",
            "8874376ecdb81fb555d523197c922baace34d5a6abe567b68e1e74254f2b11e8",
            &CNNL_SYMBOLS,
        )?;
        validate_library(
            &self.libraries[1],
            "libcnrt.so",
            "include/cnrt.h",
            "0c8ca727c2db85c60f69cf8cd96157ee1d547b175ddeaf45c0e94108f794a8f2",
            "cnrt_6.6.0-1.ubuntu20.04_amd64.deb",
            "70e23ae6197f68f9c0440db985f7a20dc174837b4190ba6c70d70117e3d8fb51",
            &CNRT_SYMBOLS,
        )?;
        if self.reviewed_enums.len() != 1 {
            return Err(contract("reviewed enum coverage is incomplete"));
        }
        let memory_transfer = &self.reviewed_enums[0];
        if memory_transfer.name != "cnrtMemTransDir_t"
            || memory_transfer.header != "include/cnrt.h"
            || memory_transfer.normalized_declaration != MEMORY_TRANSFER_DECLARATION
            || memory_transfer.variants.len() != MEMORY_TRANSFER_VARIANTS.len()
        {
            return Err(contract("cnrtMemTransDir_t declaration differs"));
        }
        for (actual, expected) in memory_transfer
            .variants
            .iter()
            .zip(MEMORY_TRANSFER_VARIANTS)
        {
            if actual.name != expected.0 || actual.value != expected.1 {
                return Err(contract(format!(
                    "cnrtMemTransDir_t variant {} differs",
                    actual.name
                )));
            }
        }
        if self.reviewed_enum_values.len() != REVIEWED_ENUM_VALUES.len()
            || self
                .reviewed_enum_values
                .iter()
                .zip(REVIEWED_ENUM_VALUES)
                .any(|(actual, expected)| {
                    actual.type_name != expected.0
                        || actual.name != expected.1
                        || actual.value != expected.2
                })
        {
            return Err(contract("reviewed CNRT/CNNL enum values differ"));
        }
        if self.layouts.len() != LAYOUTS.len() {
            return Err(contract("layout coverage is incomplete"));
        }
        for (actual, expected) in self.layouts.iter().zip(LAYOUTS) {
            if actual.name != expected.0 || actual.size != expected.1 || actual.align != expected.2
            {
                return Err(contract(format!(
                    "layout {} differs from the reviewed 64-bit C ABI",
                    actual.name
                )));
            }
        }
        if self.package_policy.redistribute_vendor_runtime
            || !self.package_policy.license_approval_required
            || self.package_policy.signature_algorithm != "ed25519"
            || self.package_policy.signature_domain != "zed-sim-mlu-package-v1"
            || !self.package_policy.runtime_compilation_forbidden
        {
            return Err(contract(
                "package, license, signature, or compilation policy differs",
            ));
        }
        Ok(())
    }
}

fn validate_library(
    library: &LibraryContract,
    filename: &str,
    header: &str,
    header_sha256: &str,
    package: &str,
    package_sha256: &str,
    symbols: &[(&str, &str)],
) -> Result<(), AbiManifestError> {
    if library.filename != filename
        || library.header != header
        || library.header_sha256 != header_sha256
        || library.package != package
        || library.package_sha256 != package_sha256
    {
        return Err(contract(format!(
            "library provenance for {} differs",
            library.id
        )));
    }
    if !is_sha256(&library.header_sha256) || !is_sha256(&library.package_sha256) {
        return Err(contract(format!(
            "library {} contains a malformed digest",
            library.id
        )));
    }
    if library.symbols.len() != symbols.len() {
        return Err(contract(format!(
            "symbol coverage for {} is incomplete",
            library.id
        )));
    }
    for (actual, expected) in library.symbols.iter().zip(symbols) {
        if actual.name != expected.0 || actual.signature != expected.1 {
            return Err(contract(format!(
                "symbol {} differs from the reviewed declaration",
                actual.name
            )));
        }
    }
    Ok(())
}

fn require_exact_strings(
    actual: &[String],
    expected: &[&str],
    label: &str,
) -> Result<(), AbiManifestError> {
    if actual
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
    {
        Ok(())
    } else {
        Err(contract(format!("{label} differs from the reviewed order")))
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn contract(message: impl Into<String>) -> AbiManifestError {
    AbiManifestError::Contract(message.into())
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CnrtStatus(pub c_int);

impl CnrtStatus {
    pub const SUCCESS: Self = Self(0);
    pub const NO_DEVICE: Self = Self(100_004);
    pub const NO_MEMORY: Self = Self(100_100);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CnnlStatus(pub c_int);

impl CnnlStatus {
    pub const SUCCESS: Self = Self(0);
    pub const ALLOCATION_FAILED: Self = Self(2);
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CnrtMemTransferDirection {
    HostToDevice = 0,
    DeviceToDevice = 1,
    DeviceToHost = 2,
    HostToHost = 3,
    PeerToPeer = 4,
    NoDirection = 5,
}

pub type CnrtQueue = *mut c_void;
pub type CnrtNotifier = *mut c_void;
pub type CnnlHandle = *mut c_void;
pub type CnnlOpTensorDescriptor = *mut c_void;
pub type CnnlTensorDescriptor = *mut c_void;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CnnlTensorLayout {
    Array = 4,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CnnlDataType {
    Half = 1,
    Float = 2,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CnnlOpTensorDescription {
    Add = 0,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CnnlNanPropagation {
    Propagate = 1,
}

pub type CnrtGetLibVersion = unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int) -> CnrtStatus;
pub type CnrtGetDeviceCount = unsafe extern "C" fn(*mut c_uint) -> CnrtStatus;
pub type CnrtSetDevice = unsafe extern "C" fn(c_int) -> CnrtStatus;
pub type CnrtMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> CnrtStatus;
pub type CnrtFree = unsafe extern "C" fn(*mut c_void) -> CnrtStatus;
pub type CnrtMemcpy =
    unsafe extern "C" fn(*mut c_void, *mut c_void, usize, CnrtMemTransferDirection) -> CnrtStatus;
pub type CnrtQueueCreate = unsafe extern "C" fn(*mut CnrtQueue) -> CnrtStatus;
pub type CnrtQueueDestroy = unsafe extern "C" fn(CnrtQueue) -> CnrtStatus;
pub type CnrtQueueSync = unsafe extern "C" fn(CnrtQueue) -> CnrtStatus;
pub type CnnlGetLibVersion = unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int);
pub type CnnlCreate = unsafe extern "C" fn(*mut CnnlHandle) -> CnnlStatus;
pub type CnnlCreateOpTensorDescriptor =
    unsafe extern "C" fn(*mut CnnlOpTensorDescriptor) -> CnnlStatus;
pub type CnnlDestroy = unsafe extern "C" fn(CnnlHandle) -> CnnlStatus;
pub type CnnlDestroyOpTensorDescriptor = unsafe extern "C" fn(CnnlOpTensorDescriptor) -> CnnlStatus;
pub type CnnlSetQueue = unsafe extern "C" fn(CnnlHandle, CnrtQueue) -> CnnlStatus;
pub type CnnlCreateTensorDescriptor = unsafe extern "C" fn(*mut CnnlTensorDescriptor) -> CnnlStatus;
pub type CnnlDestroyTensorDescriptor = unsafe extern "C" fn(CnnlTensorDescriptor) -> CnnlStatus;
pub type CnnlSetTensorDescriptor = unsafe extern "C" fn(
    CnnlTensorDescriptor,
    CnnlTensorLayout,
    CnnlDataType,
    c_int,
    *const c_int,
) -> CnnlStatus;
pub type CnnlSetOpTensorDescriptor = unsafe extern "C" fn(
    CnnlOpTensorDescriptor,
    CnnlOpTensorDescription,
    CnnlDataType,
    CnnlNanPropagation,
) -> CnnlStatus;
pub type CnnlOpTensor = unsafe extern "C" fn(
    CnnlHandle,
    CnnlOpTensorDescriptor,
    *const c_void,
    CnnlTensorDescriptor,
    *const c_void,
    *const c_void,
    CnnlTensorDescriptor,
    *const c_void,
    *mut c_void,
    usize,
    *const c_void,
    CnnlTensorDescriptor,
    *mut c_void,
) -> CnnlStatus;

const _: [(); 4] = [(); std::mem::size_of::<CnrtStatus>()];
const _: [(); 4] = [(); std::mem::align_of::<CnrtStatus>()];
const _: [(); 4] = [(); std::mem::size_of::<CnnlStatus>()];
const _: [(); 4] = [(); std::mem::align_of::<CnnlStatus>()];
const _: [(); 4] = [(); std::mem::size_of::<CnrtMemTransferDirection>()];
const _: [(); 4] = [(); std::mem::align_of::<CnrtMemTransferDirection>()];
const _: [(); 8] = [(); std::mem::size_of::<CnrtQueue>()];
const _: [(); 8] = [(); std::mem::align_of::<CnrtQueue>()];
const _: [(); 8] = [(); std::mem::size_of::<CnnlOpTensorDescriptor>()];
const _: [(); 4] = [(); std::mem::size_of::<CnnlTensorLayout>()];
const _: [(); 4] = [(); std::mem::size_of::<CnnlDataType>()];
const _: [(); 4] = [(); std::mem::size_of::<CnnlOpTensorDescription>()];
const _: [(); 4] = [(); std::mem::size_of::<CnnlNanPropagation>()];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_declarations_and_layouts_match_pinned_headers()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(std::mem::size_of::<CnrtStatus>(), 4);
        assert_eq!(std::mem::size_of::<CnnlStatus>(), 4);
        assert_eq!(CnrtStatus::SUCCESS.0, 0);
        assert_eq!(CnrtStatus::NO_DEVICE.0, 100_004);
        assert_eq!(CnrtStatus::NO_MEMORY.0, 100_100);
        assert_eq!(CnnlStatus::SUCCESS.0, 0);
        assert_eq!(CnnlStatus::ALLOCATION_FAILED.0, 2);
        assert_eq!(std::mem::size_of::<CnrtQueue>(), 8);
        assert_eq!(CnrtMemTransferDirection::HostToDevice as i32, 0);
        assert_eq!(CnrtMemTransferDirection::DeviceToDevice as i32, 1);
        assert_eq!(CnrtMemTransferDirection::DeviceToHost as i32, 2);
        assert_eq!(CnrtMemTransferDirection::HostToHost as i32, 3);
        assert_eq!(CnrtMemTransferDirection::PeerToPeer as i32, 4);
        assert_eq!(CnrtMemTransferDirection::NoDirection as i32, 5);
        assert_eq!(CnnlTensorLayout::Array as i32, 4);
        assert_eq!(CnnlDataType::Half as i32, 1);
        assert_eq!(CnnlDataType::Float as i32, 2);
        assert_eq!(CnnlOpTensorDescription::Add as i32, 0);
        assert_eq!(CnnlNanPropagation::Propagate as i32, 1);
        assert_eq!(
            manifest.libraries[0].header_sha256,
            "69eaa4fa560fa02e1e636a0d19917244a2badd94a960721e57fa36059f6fa031"
        );
        assert_eq!(
            manifest.libraries[1].header_sha256,
            "0c8ca727c2db85c60f69cf8cd96157ee1d547b175ddeaf45c0e94108f794a8f2"
        );
        Ok(())
    }

    #[test]
    fn rejects_reordered_symbols() -> Result<(), Box<dyn std::error::Error>> {
        let mut manifest: AbiManifest = serde_json::from_str(ABI_MANIFEST)?;
        manifest.libraries[0].symbols.swap(0, 1);
        assert!(matches!(
            manifest.validate(),
            Err(AbiManifestError::Contract(message)) if message.contains("symbol")
        ));
        Ok(())
    }

    #[test]
    fn rejects_missing_or_changed_execution_status_values() -> Result<(), Box<dyn std::error::Error>>
    {
        let manifest: AbiManifest = serde_json::from_str(ABI_MANIFEST)?;
        for index in 0..5 {
            let mut missing = manifest.clone();
            missing.reviewed_enum_values.remove(index);
            assert!(matches!(
                missing.validate(),
                Err(AbiManifestError::Contract(message)) if message.contains("enum values")
            ));

            let mut changed = manifest.clone();
            changed.reviewed_enum_values[index].value += 1;
            assert!(matches!(
                changed.validate(),
                Err(AbiManifestError::Contract(message)) if message.contains("enum values")
            ));
        }
        Ok(())
    }
}
