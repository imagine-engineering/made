use serde::Deserialize;
use std::ffi::{c_char, c_int, c_uint, c_void};
use thiserror::Error;

pub const ABI_MANIFEST: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "level-zero-loader-1.11.0-api-1.6.3+onednn-3.5.0";
pub const UNSAFE_OWNER: &str = "sim_backend_xpu::loader";
pub const CERTIFICATE_OWNER: &str = "sim_runtime::NativeFfiRegistry";
pub const BINDING_STATUS_OWNER: &str = "sim_types::NativeBackendBindingStatus";
pub const SEMANTIC_CAPABILITY_OWNER: &str = "sim_tensor::BackendCapabilityMatrix";
pub const LEVEL_ZERO_MINIMUM_API_VERSION: ZeApiVersion = ZeApiVersion::new(1, 6);
pub const ONEDNN_MINIMUM_MAJOR: c_int = 3;
pub const ONEDNN_MINIMUM_MINOR: c_int = 5;

const TARGETS: [&str; 2] = ["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"];
const DISCOVERY_ORDER: [&str; 4] = [
    "SIM_XPU_ROOT",
    "ONEAPI_ROOT",
    "signed_package_roots",
    "system_level_zero_loader",
];
const LEVEL_ZERO_SYMBOLS: [(&str, &str); 12] = [
    (
        "zeCommandQueueCreate",
        "ze_result_t(ze_context_handle_t,ze_device_handle_t,const ze_command_queue_desc_t*,ze_command_queue_handle_t*)",
    ),
    (
        "zeCommandQueueDestroy",
        "ze_result_t(ze_command_queue_handle_t)",
    ),
    (
        "zeCommandQueueSynchronize",
        "ze_result_t(ze_command_queue_handle_t,uint64_t)",
    ),
    (
        "zeContextCreate",
        "ze_result_t(ze_driver_handle_t,const ze_context_desc_t*,ze_context_handle_t*)",
    ),
    ("zeContextDestroy", "ze_result_t(ze_context_handle_t)"),
    (
        "zeDeviceGet",
        "ze_result_t(ze_driver_handle_t,uint32_t*,ze_device_handle_t*)",
    ),
    (
        "zeDeviceGetCommandQueueGroupProperties",
        "ze_result_t(ze_device_handle_t,uint32_t*,ze_command_queue_group_properties_t*)",
    ),
    (
        "zeDeviceGetMemoryProperties",
        "ze_result_t(ze_device_handle_t,uint32_t*,ze_device_memory_properties_t*)",
    ),
    (
        "zeDeviceGetProperties",
        "ze_result_t(ze_device_handle_t,ze_device_properties_t*)",
    ),
    ("zeDriverGet", "ze_result_t(uint32_t*,ze_driver_handle_t*)"),
    (
        "zeDriverGetApiVersion",
        "ze_result_t(ze_driver_handle_t,ze_api_version_t*)",
    ),
    ("zeInit", "ze_result_t(ze_init_flags_t)"),
];
const ONEDNN_SYMBOLS: [(&str, &str); 18] = [
    (
        "dnnl_binary_primitive_desc_create",
        "dnnl_status_t(dnnl_primitive_desc_t*,dnnl_engine_t,dnnl_alg_kind_t,const_dnnl_memory_desc_t,const_dnnl_memory_desc_t,const_dnnl_memory_desc_t,const_dnnl_primitive_attr_t)",
    ),
    (
        "dnnl_engine_create",
        "dnnl_status_t(dnnl_engine_t*,dnnl_engine_kind_t,size_t)",
    ),
    ("dnnl_engine_destroy", "dnnl_status_t(dnnl_engine_t)"),
    ("dnnl_engine_get_count", "size_t(dnnl_engine_kind_t)"),
    (
        "dnnl_memory_create",
        "dnnl_status_t(dnnl_memory_t*,const_dnnl_memory_desc_t,dnnl_engine_t,void*)",
    ),
    ("dnnl_memory_destroy", "dnnl_status_t(dnnl_memory_t)"),
    (
        "dnnl_memory_desc_create_with_strides",
        "dnnl_status_t(dnnl_memory_desc_t*,int,const dnnl_dim_t*,dnnl_data_type_t,const dnnl_dim_t*)",
    ),
    (
        "dnnl_memory_desc_destroy",
        "dnnl_status_t(dnnl_memory_desc_t)",
    ),
    (
        "dnnl_memory_map_data",
        "dnnl_status_t(const_dnnl_memory_t,void**)",
    ),
    (
        "dnnl_memory_unmap_data",
        "dnnl_status_t(const_dnnl_memory_t,void*)",
    ),
    (
        "dnnl_primitive_create",
        "dnnl_status_t(dnnl_primitive_t*,const_dnnl_primitive_desc_t)",
    ),
    (
        "dnnl_primitive_desc_destroy",
        "dnnl_status_t(dnnl_primitive_desc_t)",
    ),
    ("dnnl_primitive_destroy", "dnnl_status_t(dnnl_primitive_t)"),
    (
        "dnnl_primitive_execute",
        "dnnl_status_t(const_dnnl_primitive_t,dnnl_stream_t,int,const dnnl_exec_arg_t*)",
    ),
    (
        "dnnl_stream_create",
        "dnnl_status_t(dnnl_stream_t*,dnnl_engine_t,unsigned)",
    ),
    ("dnnl_stream_destroy", "dnnl_status_t(dnnl_stream_t)"),
    ("dnnl_stream_wait", "dnnl_status_t(dnnl_stream_t)"),
    ("dnnl_version", "const dnnl_version_t*(void)"),
];
const LEVEL_ZERO_HEADERS: [(&str, &str); 1] = [(
    "include/ze_api.h",
    "72108c7826744ed58e1c86e6abfd407d89fdf8e1b09e96ac0439f40c3b6d0175",
)];
const ONEDNN_HEADERS: [(&str, &str); 4] = [
    (
        "include/oneapi/dnnl/dnnl.h",
        "23dc8d9f3f5bcfc5c5500a1c86528fcc67439e8d8fb512e66cb905e4d30eb373",
    ),
    (
        "include/oneapi/dnnl/dnnl_common.h",
        "c8af25f31fecbd810a6b8851a2d764eb29c69efeb4e729c8ad030e973f2b731f",
    ),
    (
        "include/oneapi/dnnl/dnnl_common_types.h",
        "6cc4ae6d144f0f8000e3e9b01875ff06d064ede6f64cd46ead2395844bc33349",
    ),
    (
        "include/oneapi/dnnl/dnnl_types.h",
        "09f99a6b1736e93e0b73c4c75675ccab1bcb4562763f04d3dd879c3a0819a078",
    ),
];
const VALUES: [(&str, &str, i64); 20] = [
    ("ze_api_version_t", "ZE_API_VERSION_1_6", 65_542),
    (
        "ze_structure_type_t",
        "ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES",
        6,
    ),
    (
        "ze_structure_type_t",
        "ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES",
        3,
    ),
    (
        "ze_structure_type_t",
        "ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES",
        7,
    ),
    ("ze_structure_type_t", "ZE_STRUCTURE_TYPE_CONTEXT_DESC", 13),
    (
        "ze_structure_type_t",
        "ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC",
        14,
    ),
    (
        "ze_command_queue_group_property_flag_t",
        "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE",
        1,
    ),
    (
        "ze_command_queue_mode_t",
        "ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS",
        2,
    ),
    (
        "ze_command_queue_priority_t",
        "ZE_COMMAND_QUEUE_PRIORITY_NORMAL",
        0,
    ),
    ("ze_device_type_t", "ZE_DEVICE_TYPE_GPU", 1),
    ("dnnl_status_t", "dnnl_success", 0),
    ("dnnl_engine_kind_t", "dnnl_gpu", 2),
    ("dnnl_stream_flags_t", "dnnl_stream_default_flags", 1),
    ("dnnl_data_type_t", "dnnl_f16", 1),
    ("dnnl_data_type_t", "dnnl_f32", 3),
    ("dnnl_alg_kind_t", "dnnl_binary_add", 131_056),
    ("dnnl_exec_arg", "DNNL_ARG_SRC_0", 1),
    ("dnnl_exec_arg", "DNNL_ARG_SRC_1", 2),
    ("dnnl_exec_arg", "DNNL_ARG_DST", 17),
    ("void*", "DNNL_MEMORY_ALLOCATE", -1),
];
const LAYOUTS: [(&str, usize, usize); 20] = [
    ("ze_result_t", 4, 4),
    ("ze_driver_handle_t", 8, 8),
    ("ze_device_handle_t", 8, 8),
    ("ze_context_handle_t", 8, 8),
    ("ze_command_queue_handle_t", 8, 8),
    ("ze_context_desc_t", 24, 8),
    ("ze_command_queue_desc_t", 40, 8),
    ("ze_command_queue_group_properties_t", 40, 8),
    ("ze_device_uuid_t", 16, 1),
    ("ze_device_properties_t", 368, 8),
    ("ze_device_memory_properties_t", 296, 8),
    ("dnnl_status_t", 4, 4),
    ("dnnl_engine_t", 8, 8),
    ("dnnl_stream_t", 8, 8),
    ("dnnl_memory_desc_t", 8, 8),
    ("dnnl_memory_t", 8, 8),
    ("dnnl_primitive_desc_t", 8, 8),
    ("dnnl_primitive_t", 8, 8),
    ("dnnl_exec_arg_t", 16, 8),
    ("dnnl_version_t", 32, 8),
];
const FIELD_OFFSETS: [(&str, &str, usize); 51] = [
    ("ze_context_desc_t", "stype", 0),
    ("ze_context_desc_t", "pNext", 8),
    ("ze_context_desc_t", "flags", 16),
    ("ze_command_queue_desc_t", "stype", 0),
    ("ze_command_queue_desc_t", "pNext", 8),
    ("ze_command_queue_desc_t", "ordinal", 16),
    ("ze_command_queue_desc_t", "index", 20),
    ("ze_command_queue_desc_t", "flags", 24),
    ("ze_command_queue_desc_t", "mode", 28),
    ("ze_command_queue_desc_t", "priority", 32),
    ("ze_command_queue_group_properties_t", "stype", 0),
    ("ze_command_queue_group_properties_t", "pNext", 8),
    ("ze_command_queue_group_properties_t", "flags", 16),
    (
        "ze_command_queue_group_properties_t",
        "maxMemoryFillPatternSize",
        24,
    ),
    ("ze_command_queue_group_properties_t", "numQueues", 32),
    ("ze_device_properties_t", "stype", 0),
    ("ze_device_properties_t", "pNext", 8),
    ("ze_device_properties_t", "type", 16),
    ("ze_device_properties_t", "vendorId", 20),
    ("ze_device_properties_t", "deviceId", 24),
    ("ze_device_properties_t", "flags", 28),
    ("ze_device_properties_t", "subdeviceId", 32),
    ("ze_device_properties_t", "coreClockRate", 36),
    ("ze_device_properties_t", "maxMemAllocSize", 40),
    ("ze_device_properties_t", "maxHardwareContexts", 48),
    ("ze_device_properties_t", "maxCommandQueuePriority", 52),
    ("ze_device_properties_t", "numThreadsPerEU", 56),
    ("ze_device_properties_t", "physicalEUSimdWidth", 60),
    ("ze_device_properties_t", "numEUsPerSubslice", 64),
    ("ze_device_properties_t", "numSubslicesPerSlice", 68),
    ("ze_device_properties_t", "numSlices", 72),
    ("ze_device_properties_t", "timerResolution", 80),
    ("ze_device_properties_t", "timestampValidBits", 88),
    ("ze_device_properties_t", "kernelTimestampValidBits", 92),
    ("ze_device_properties_t", "uuid", 96),
    ("ze_device_properties_t", "name", 112),
    ("ze_device_memory_properties_t", "stype", 0),
    ("ze_device_memory_properties_t", "pNext", 8),
    ("ze_device_memory_properties_t", "flags", 16),
    ("ze_device_memory_properties_t", "maxClockRate", 20),
    ("ze_device_memory_properties_t", "maxBusWidth", 24),
    ("ze_device_memory_properties_t", "totalSize", 32),
    ("ze_device_memory_properties_t", "name", 40),
    ("dnnl_exec_arg_t", "arg", 0),
    ("dnnl_exec_arg_t", "memory", 8),
    ("dnnl_version_t", "major", 0),
    ("dnnl_version_t", "minor", 4),
    ("dnnl_version_t", "patch", 8),
    ("dnnl_version_t", "hash", 16),
    ("dnnl_version_t", "cpu_runtime", 24),
    ("dnnl_version_t", "gpu_runtime", 28),
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
    pub reviewed_values: Vec<ValueContract>,
    pub layouts: Vec<LayoutContract>,
    pub field_offsets: Vec<FieldOffsetContract>,
    pub unsafe_owner: String,
    pub certificate_owner: String,
    pub binding_status_owner: String,
    pub semantic_capability_owner: String,
    pub package_policy: PackagePolicy,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LibraryContract {
    pub id: String,
    pub filenames: FilenameContract,
    pub source_repository: String,
    pub source_tag: String,
    pub source_commit: String,
    pub headers: Vec<HeaderContract>,
    pub symbols: Vec<SymbolContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FilenameContract {
    pub linux: String,
    pub windows: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HeaderContract {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SymbolContract {
    pub name: String,
    pub signature: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ValueContract {
    pub type_name: String,
    pub name: String,
    pub value: i64,
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
pub struct FieldOffsetContract {
    pub type_name: String,
    pub field: String,
    pub offset: usize,
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
    #[error("XPU ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("XPU ABI manifest violates the reviewed contract: {0}")]
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
            || self.backend != "xpu"
            || self.abi_floor != ABI_FLOOR
            || self.unsafe_owner != UNSAFE_OWNER
            || self.certificate_owner != CERTIFICATE_OWNER
            || self.binding_status_owner != BINDING_STATUS_OWNER
            || self.semantic_capability_owner != SEMANTIC_CAPABILITY_OWNER
        {
            return Err(contract(
                "identity, ABI floor, or authoritative owner differs",
            ));
        }
        exact_strings(&self.targets, &TARGETS, "targets")?;
        exact_strings(&self.discovery_order, &DISCOVERY_ORDER, "discovery order")?;
        if self.libraries.len() != 2
            || self.libraries[0].id != "level_zero"
            || self.libraries[1].id != "onednn"
        {
            return Err(contract(
                "libraries must be exactly sorted level_zero then onednn",
            ));
        }
        validate_library(
            &self.libraries[0],
            "libze_loader.so.1",
            "ze_loader.dll",
            "https://github.com/oneapi-src/level-zero",
            "v1.11.0",
            "f35123bead54a471a7e5f3bf8d439a4a44527d8e",
            &LEVEL_ZERO_HEADERS,
            &LEVEL_ZERO_SYMBOLS,
        )?;
        validate_library(
            &self.libraries[1],
            "libdnnl.so.3",
            "dnnl.dll",
            "https://github.com/oneapi-src/oneDNN",
            "v3.5",
            "6860e98e71c748f956150f72cdbe14efe6fc2ac2",
            &ONEDNN_HEADERS,
            &ONEDNN_SYMBOLS,
        )?;
        if self.reviewed_values.len() != VALUES.len()
            || self
                .reviewed_values
                .iter()
                .zip(VALUES)
                .any(|(actual, expected)| {
                    actual.type_name != expected.0
                        || actual.name != expected.1
                        || actual.value != expected.2
                })
        {
            return Err(contract("reviewed enum values differ"));
        }
        if self.layouts.len() != LAYOUTS.len()
            || self.layouts.iter().zip(LAYOUTS).any(|(actual, expected)| {
                actual.name != expected.0 || actual.size != expected.1 || actual.align != expected.2
            })
        {
            return Err(contract("reviewed 64-bit C layouts differ"));
        }
        if self.field_offsets.len() != FIELD_OFFSETS.len()
            || self
                .field_offsets
                .iter()
                .zip(FIELD_OFFSETS)
                .any(|(actual, expected)| {
                    actual.type_name != expected.0
                        || actual.field != expected.1
                        || actual.offset != expected.2
                })
        {
            return Err(contract("reviewed 64-bit C field offsets differ"));
        }
        if self.package_policy.redistribute_vendor_runtime
            || !self.package_policy.license_approval_required
            || self.package_policy.signature_algorithm != "ed25519"
            || self.package_policy.signature_domain != "zed-sim-xpu-package-v1"
            || !self.package_policy.runtime_compilation_forbidden
        {
            return Err(contract(
                "package, license, signature, or compilation policy differs",
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_library(
    library: &LibraryContract,
    linux_filename: &str,
    windows_filename: &str,
    source_repository: &str,
    source_tag: &str,
    source_commit: &str,
    headers: &[(&str, &str)],
    symbols: &[(&str, &str)],
) -> Result<(), AbiManifestError> {
    if library.filenames.linux != linux_filename
        || library.filenames.windows != windows_filename
        || library.source_repository != source_repository
        || library.source_tag != source_tag
        || library.source_commit != source_commit
        || !is_git_commit(&library.source_commit)
    {
        return Err(contract(format!(
            "library provenance for {} differs",
            library.id
        )));
    }
    if library.headers.len() != headers.len()
        || library
            .headers
            .iter()
            .zip(headers)
            .any(|(actual, expected)| {
                actual.path != expected.0
                    || actual.sha256 != expected.1
                    || !is_sha256(&actual.sha256)
            })
    {
        return Err(contract(format!(
            "header coverage for {} differs",
            library.id
        )));
    }
    if library.symbols.len() != symbols.len()
        || library
            .symbols
            .iter()
            .zip(symbols)
            .any(|(actual, expected)| actual.name != expected.0 || actual.signature != expected.1)
    {
        return Err(contract(format!(
            "symbol coverage for {} differs",
            library.id
        )));
    }
    Ok(())
}

fn exact_strings(
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

fn is_git_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn contract(message: impl Into<String>) -> AbiManifestError {
    AbiManifestError::Contract(message.into())
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ZeApiVersion(pub c_uint);

impl ZeApiVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self(((major as c_uint) << 16) | minor as c_uint)
    }

    pub const fn major(self) -> u16 {
        (self.0 >> 16) as u16
    }

    pub const fn minor(self) -> u16 {
        (self.0 & 0xffff) as u16
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeResult(pub c_int);

impl ZeResult {
    pub const SUCCESS: Self = Self(0);
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZeContextDesc {
    pub stype: c_int,
    pub next: *const c_void,
    pub flags: c_uint,
}

impl Default for ZeContextDesc {
    fn default() -> Self {
        Self {
            stype: 13,
            next: std::ptr::null(),
            flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZeCommandQueueDesc {
    pub stype: c_int,
    pub next: *const c_void,
    pub ordinal: c_uint,
    pub index: c_uint,
    pub flags: c_uint,
    pub mode: c_int,
    pub priority: c_int,
}

impl ZeCommandQueueDesc {
    pub const fn asynchronous(ordinal: c_uint, index: c_uint) -> Self {
        Self {
            stype: 14,
            next: std::ptr::null(),
            ordinal,
            index,
            flags: 0,
            mode: 2,
            priority: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZeCommandQueueGroupProperties {
    pub stype: c_int,
    pub next: *mut c_void,
    pub flags: c_uint,
    pub maximum_memory_fill_pattern_size: usize,
    pub queue_count: c_uint,
}

impl Default for ZeCommandQueueGroupProperties {
    fn default() -> Self {
        Self {
            stype: 6,
            next: std::ptr::null_mut(),
            flags: 0,
            maximum_memory_fill_pattern_size: 0,
            queue_count: 0,
        }
    }
}

pub const ZE_MAX_DEVICE_NAME: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ZeDeviceUuid {
    pub id: [u8; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZeDeviceProperties {
    pub stype: c_int,
    pub next: *mut c_void,
    pub device_type: c_int,
    pub vendor_id: c_uint,
    pub device_id: c_uint,
    pub flags: c_uint,
    pub subdevice_id: c_uint,
    pub core_clock_rate: c_uint,
    pub maximum_memory_allocation_size: u64,
    pub maximum_hardware_contexts: c_uint,
    pub maximum_command_queue_priority: c_uint,
    pub threads_per_execution_unit: c_uint,
    pub physical_execution_unit_simd_width: c_uint,
    pub execution_units_per_subslice: c_uint,
    pub subslices_per_slice: c_uint,
    pub slices: c_uint,
    pub timer_resolution: u64,
    pub timestamp_valid_bits: c_uint,
    pub kernel_timestamp_valid_bits: c_uint,
    pub uuid: ZeDeviceUuid,
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

impl Default for ZeDeviceProperties {
    fn default() -> Self {
        Self {
            stype: 3,
            next: std::ptr::null_mut(),
            device_type: 0,
            vendor_id: 0,
            device_id: 0,
            flags: 0,
            subdevice_id: 0,
            core_clock_rate: 0,
            maximum_memory_allocation_size: 0,
            maximum_hardware_contexts: 0,
            maximum_command_queue_priority: 0,
            threads_per_execution_unit: 0,
            physical_execution_unit_simd_width: 0,
            execution_units_per_subslice: 0,
            subslices_per_slice: 0,
            slices: 0,
            timer_resolution: 0,
            timestamp_valid_bits: 0,
            kernel_timestamp_valid_bits: 0,
            uuid: ZeDeviceUuid::default(),
            name: [0; ZE_MAX_DEVICE_NAME],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ZeDeviceMemoryProperties {
    pub stype: c_int,
    pub next: *mut c_void,
    pub flags: c_uint,
    pub maximum_clock_rate: c_uint,
    pub maximum_bus_width: c_uint,
    pub total_size: u64,
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

impl Default for ZeDeviceMemoryProperties {
    fn default() -> Self {
        Self {
            stype: 7,
            next: std::ptr::null_mut(),
            flags: 0,
            maximum_clock_rate: 0,
            maximum_bus_width: 0,
            total_size: 0,
            name: [0; ZE_MAX_DEVICE_NAME],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DnnlVersion {
    pub major: c_int,
    pub minor: c_int,
    pub patch: c_int,
    pub hash: *const c_char,
    pub cpu_runtime: c_uint,
    pub gpu_runtime: c_uint,
}

pub type ZeDriverHandle = *mut c_void;
pub type ZeDeviceHandle = *mut c_void;
pub type ZeContextHandle = *mut c_void;
pub type ZeCommandQueueHandle = *mut c_void;
pub type DnnlEngineHandle = *mut c_void;
pub type DnnlStreamHandle = *mut c_void;
pub type DnnlMemoryDescHandle = *mut c_void;
pub type DnnlMemoryHandle = *mut c_void;
pub type DnnlPrimitiveDescHandle = *mut c_void;
pub type DnnlPrimitiveHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DnnlExecArg {
    pub argument: c_int,
    pub memory: DnnlMemoryHandle,
}

pub type ZeInit = unsafe extern "C" fn(c_uint) -> ZeResult;
pub type ZeDriverGet = unsafe extern "C" fn(*mut c_uint, *mut ZeDriverHandle) -> ZeResult;
pub type ZeDriverGetApiVersion =
    unsafe extern "C" fn(ZeDriverHandle, *mut ZeApiVersion) -> ZeResult;
pub type ZeDeviceGet =
    unsafe extern "C" fn(ZeDriverHandle, *mut c_uint, *mut ZeDeviceHandle) -> ZeResult;
pub type ZeDeviceGetCommandQueueGroupProperties = unsafe extern "C" fn(
    ZeDeviceHandle,
    *mut c_uint,
    *mut ZeCommandQueueGroupProperties,
) -> ZeResult;
pub type ZeDeviceGetMemoryProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut c_uint, *mut ZeDeviceMemoryProperties) -> ZeResult;
pub type ZeDeviceGetProperties =
    unsafe extern "C" fn(ZeDeviceHandle, *mut ZeDeviceProperties) -> ZeResult;
pub type ZeContextCreate =
    unsafe extern "C" fn(ZeDriverHandle, *const ZeContextDesc, *mut ZeContextHandle) -> ZeResult;
pub type ZeContextDestroy = unsafe extern "C" fn(ZeContextHandle) -> ZeResult;
pub type ZeCommandQueueCreate = unsafe extern "C" fn(
    ZeContextHandle,
    ZeDeviceHandle,
    *const ZeCommandQueueDesc,
    *mut ZeCommandQueueHandle,
) -> ZeResult;
pub type ZeCommandQueueDestroy = unsafe extern "C" fn(ZeCommandQueueHandle) -> ZeResult;
pub type ZeCommandQueueSynchronize = unsafe extern "C" fn(ZeCommandQueueHandle, u64) -> ZeResult;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DnnlStatus(pub c_int);

impl DnnlStatus {
    pub const SUCCESS: Self = Self(0);
    pub const OUT_OF_MEMORY: Self = Self(1);
    pub const RUNTIME_ERROR: Self = Self(5);
}

pub type DnnlBinaryPrimitiveDescCreate = unsafe extern "C" fn(
    *mut DnnlPrimitiveDescHandle,
    DnnlEngineHandle,
    c_int,
    DnnlMemoryDescHandle,
    DnnlMemoryDescHandle,
    DnnlMemoryDescHandle,
    *const c_void,
) -> DnnlStatus;
pub type DnnlVersionFn = unsafe extern "C" fn() -> *const DnnlVersion;
pub type DnnlEngineGetCount = unsafe extern "C" fn(c_int) -> usize;
pub type DnnlEngineCreate = unsafe extern "C" fn(*mut DnnlEngineHandle, c_int, usize) -> DnnlStatus;
pub type DnnlEngineDestroy = unsafe extern "C" fn(DnnlEngineHandle) -> DnnlStatus;
pub type DnnlStreamCreate =
    unsafe extern "C" fn(*mut DnnlStreamHandle, DnnlEngineHandle, c_uint) -> DnnlStatus;
pub type DnnlStreamWait = unsafe extern "C" fn(DnnlStreamHandle) -> DnnlStatus;
pub type DnnlStreamDestroy = unsafe extern "C" fn(DnnlStreamHandle) -> DnnlStatus;
pub type DnnlMemoryDescCreateWithStrides = unsafe extern "C" fn(
    *mut DnnlMemoryDescHandle,
    c_int,
    *const i64,
    c_int,
    *const i64,
) -> DnnlStatus;
pub type DnnlMemoryDescDestroy = unsafe extern "C" fn(DnnlMemoryDescHandle) -> DnnlStatus;
pub type DnnlMemoryCreate = unsafe extern "C" fn(
    *mut DnnlMemoryHandle,
    DnnlMemoryDescHandle,
    DnnlEngineHandle,
    *mut c_void,
) -> DnnlStatus;
pub type DnnlMemoryMapData = unsafe extern "C" fn(DnnlMemoryHandle, *mut *mut c_void) -> DnnlStatus;
pub type DnnlMemoryUnmapData = unsafe extern "C" fn(DnnlMemoryHandle, *mut c_void) -> DnnlStatus;
pub type DnnlMemoryDestroy = unsafe extern "C" fn(DnnlMemoryHandle) -> DnnlStatus;
pub type DnnlPrimitiveDescDestroy = unsafe extern "C" fn(DnnlPrimitiveDescHandle) -> DnnlStatus;
pub type DnnlPrimitiveCreate =
    unsafe extern "C" fn(*mut DnnlPrimitiveHandle, DnnlPrimitiveDescHandle) -> DnnlStatus;
pub type DnnlPrimitiveExecute = unsafe extern "C" fn(
    DnnlPrimitiveHandle,
    DnnlStreamHandle,
    c_int,
    *const DnnlExecArg,
) -> DnnlStatus;
pub type DnnlPrimitiveDestroy = unsafe extern "C" fn(DnnlPrimitiveHandle) -> DnnlStatus;

const _: [(); 24] = [(); std::mem::size_of::<ZeContextDesc>()];
const _: [(); 8] = [(); std::mem::align_of::<ZeContextDesc>()];
const _: [(); 40] = [(); std::mem::size_of::<ZeCommandQueueDesc>()];
const _: [(); 8] = [(); std::mem::align_of::<ZeCommandQueueDesc>()];
const _: [(); 40] = [(); std::mem::size_of::<ZeCommandQueueGroupProperties>()];
const _: [(); 8] = [(); std::mem::align_of::<ZeCommandQueueGroupProperties>()];
const _: [(); 16] = [(); std::mem::size_of::<ZeDeviceUuid>()];
const _: [(); 1] = [(); std::mem::align_of::<ZeDeviceUuid>()];
const _: [(); 368] = [(); std::mem::size_of::<ZeDeviceProperties>()];
const _: [(); 8] = [(); std::mem::align_of::<ZeDeviceProperties>()];
const _: [(); 296] = [(); std::mem::size_of::<ZeDeviceMemoryProperties>()];
const _: [(); 8] = [(); std::mem::align_of::<ZeDeviceMemoryProperties>()];
const _: [(); 16] = [(); std::mem::size_of::<DnnlExecArg>()];
const _: [(); 8] = [(); std::mem::align_of::<DnnlExecArg>()];
const _: [(); 32] = [(); std::mem::size_of::<DnnlVersion>()];
const _: [(); 8] = [(); std::mem::align_of::<DnnlVersion>()];
const _: [(); 0] = [(); std::mem::offset_of!(ZeContextDesc, stype)];
const _: [(); 8] = [(); std::mem::offset_of!(ZeContextDesc, next)];
const _: [(); 16] = [(); std::mem::offset_of!(ZeContextDesc, flags)];
const _: [(); 0] = [(); std::mem::offset_of!(ZeCommandQueueDesc, stype)];
const _: [(); 8] = [(); std::mem::offset_of!(ZeCommandQueueDesc, next)];
const _: [(); 16] = [(); std::mem::offset_of!(ZeCommandQueueDesc, ordinal)];
const _: [(); 20] = [(); std::mem::offset_of!(ZeCommandQueueDesc, index)];
const _: [(); 24] = [(); std::mem::offset_of!(ZeCommandQueueDesc, flags)];
const _: [(); 28] = [(); std::mem::offset_of!(ZeCommandQueueDesc, mode)];
const _: [(); 32] = [(); std::mem::offset_of!(ZeCommandQueueDesc, priority)];
const _: [(); 0] = [(); std::mem::offset_of!(ZeCommandQueueGroupProperties, stype)];
const _: [(); 8] = [(); std::mem::offset_of!(ZeCommandQueueGroupProperties, next)];
const _: [(); 16] = [(); std::mem::offset_of!(ZeCommandQueueGroupProperties, flags)];
const _: [(); 24] = [(); std::mem::offset_of!(
    ZeCommandQueueGroupProperties,
    maximum_memory_fill_pattern_size
)];
const _: [(); 32] = [(); std::mem::offset_of!(ZeCommandQueueGroupProperties, queue_count)];
const _: [(); 0] = [(); std::mem::offset_of!(ZeDeviceProperties, stype)];
const _: [(); 8] = [(); std::mem::offset_of!(ZeDeviceProperties, next)];
const _: [(); 16] = [(); std::mem::offset_of!(ZeDeviceProperties, device_type)];
const _: [(); 20] = [(); std::mem::offset_of!(ZeDeviceProperties, vendor_id)];
const _: [(); 24] = [(); std::mem::offset_of!(ZeDeviceProperties, device_id)];
const _: [(); 28] = [(); std::mem::offset_of!(ZeDeviceProperties, flags)];
const _: [(); 32] = [(); std::mem::offset_of!(ZeDeviceProperties, subdevice_id)];
const _: [(); 36] = [(); std::mem::offset_of!(ZeDeviceProperties, core_clock_rate)];
const _: [(); 40] = [(); std::mem::offset_of!(ZeDeviceProperties, maximum_memory_allocation_size)];
const _: [(); 48] = [(); std::mem::offset_of!(ZeDeviceProperties, maximum_hardware_contexts)];
const _: [(); 52] = [(); std::mem::offset_of!(ZeDeviceProperties, maximum_command_queue_priority)];
const _: [(); 56] = [(); std::mem::offset_of!(ZeDeviceProperties, threads_per_execution_unit)];
const _: [(); 60] =
    [(); std::mem::offset_of!(ZeDeviceProperties, physical_execution_unit_simd_width)];
const _: [(); 64] = [(); std::mem::offset_of!(ZeDeviceProperties, execution_units_per_subslice)];
const _: [(); 68] = [(); std::mem::offset_of!(ZeDeviceProperties, subslices_per_slice)];
const _: [(); 72] = [(); std::mem::offset_of!(ZeDeviceProperties, slices)];
const _: [(); 80] = [(); std::mem::offset_of!(ZeDeviceProperties, timer_resolution)];
const _: [(); 88] = [(); std::mem::offset_of!(ZeDeviceProperties, timestamp_valid_bits)];
const _: [(); 92] = [(); std::mem::offset_of!(ZeDeviceProperties, kernel_timestamp_valid_bits)];
const _: [(); 96] = [(); std::mem::offset_of!(ZeDeviceProperties, uuid)];
const _: [(); 112] = [(); std::mem::offset_of!(ZeDeviceProperties, name)];
const _: [(); 0] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, stype)];
const _: [(); 8] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, next)];
const _: [(); 16] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, flags)];
const _: [(); 20] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, maximum_clock_rate)];
const _: [(); 24] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, maximum_bus_width)];
const _: [(); 32] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, total_size)];
const _: [(); 40] = [(); std::mem::offset_of!(ZeDeviceMemoryProperties, name)];
const _: [(); 0] = [(); std::mem::offset_of!(DnnlExecArg, argument)];
const _: [(); 8] = [(); std::mem::offset_of!(DnnlExecArg, memory)];
const _: [(); 0] = [(); std::mem::offset_of!(DnnlVersion, major)];
const _: [(); 4] = [(); std::mem::offset_of!(DnnlVersion, minor)];
const _: [(); 8] = [(); std::mem::offset_of!(DnnlVersion, patch)];
const _: [(); 16] = [(); std::mem::offset_of!(DnnlVersion, hash)];
const _: [(); 24] = [(); std::mem::offset_of!(DnnlVersion, cpu_runtime)];
const _: [(); 28] = [(); std::mem::offset_of!(DnnlVersion, gpu_runtime)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_matches_the_reviewed_contract() -> Result<(), AbiManifestError> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.libraries[0].symbols.len(), 12);
        assert_eq!(manifest.libraries[1].symbols.len(), 18);
        Ok(())
    }

    #[test]
    fn manifest_rejects_tampered_symbol_signature() -> Result<(), serde_json::Error> {
        let mut manifest: AbiManifest = serde_json::from_str(ABI_MANIFEST)?;
        manifest.libraries[0].symbols[0].signature.push_str("void");
        assert!(matches!(
            manifest.validate(),
            Err(AbiManifestError::Contract(_))
        ));
        Ok(())
    }

    #[test]
    fn manifest_rejects_unknown_fields() {
        let tampered = ABI_MANIFEST.replacen(
            "\"schema_version\": 1,",
            "\"schema_version\": 1, \"trusted\": true,",
            1,
        );
        assert!(matches!(
            serde_json::from_str::<AbiManifest>(&tampered),
            Err(_)
        ));
    }

    #[test]
    fn reviewed_descriptors_set_exact_stype_and_null_extension_chain() {
        let context = ZeContextDesc::default();
        assert_eq!(context.stype, 13);
        assert!(context.next.is_null());
        let queue = ZeCommandQueueDesc::asynchronous(4, 2);
        assert_eq!(queue.stype, 14);
        assert!(queue.next.is_null());
        assert_eq!(queue.ordinal, 4);
        assert_eq!(queue.index, 2);
    }
}
