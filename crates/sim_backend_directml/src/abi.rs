#![cfg_attr(
    not(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )),
    allow(dead_code)
)]

use serde::Deserialize;
use std::ffi::c_void;
use thiserror::Error;

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "1.13.1";
pub const FILE_VERSION: FileVersion = FileVersion::new(1, 13, 1, 0);
pub const TARGET_VERSION: i32 = 0x6200;
pub const MINIMUM_FEATURE_LEVEL: DmlFeatureLevel = DmlFeatureLevel(TARGET_VERSION);
pub const MINIMUM_WINDOWS_BUILD: u32 = 19_041;
pub const UNSAFE_OWNER: &str = "sim_backend_directml::loader";

pub const DML_CREATE_DEVICE_FLAG_NONE: DmlCreateDeviceFlags = DmlCreateDeviceFlags(0);
pub const DML_CREATE_DEVICE_FLAG_DEBUG: DmlCreateDeviceFlags = DmlCreateDeviceFlags(1);
pub const DML_EXECUTION_FLAG_NONE: DmlExecutionFlags = DmlExecutionFlags(0);
pub const DML_TENSOR_FLAG_NONE: DmlTensorFlags = DmlTensorFlags(0);
pub const DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT: DmlFeature = DmlFeature(0);
pub const DML_FEATURE_FEATURE_LEVELS: DmlFeature = DmlFeature(1);
pub const DML_TENSOR_TYPE_BUFFER: DmlTensorType = DmlTensorType(1);
pub const DML_TENSOR_DATA_TYPE_FLOAT32: DmlTensorDataType = DmlTensorDataType(1);
pub const DML_TENSOR_DATA_TYPE_FLOAT16: DmlTensorDataType = DmlTensorDataType(2);
pub const DML_OPERATOR_ELEMENT_WISE_ADD: DmlOperatorType = DmlOperatorType(4);
pub const DML_BINDING_TYPE_NONE: DmlBindingType = DmlBindingType(0);
pub const DML_BINDING_TYPE_BUFFER: DmlBindingType = DmlBindingType(1);
pub const DML_BINDING_TYPE_BUFFER_ARRAY: DmlBindingType = DmlBindingType(2);
pub const DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT: u32 = 16;
pub const DML_TEMPORARY_BUFFER_ALIGNMENT: u32 = 256;
pub const DML_PERSISTENT_BUFFER_ALIGNMENT: u32 = 256;
pub const D3D_FEATURE_LEVEL_11_0: D3dFeatureLevel = D3dFeatureLevel(0xb000);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct D3dFeatureLevel(pub i32);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FileVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub build: u16,
}

impl FileVersion {
    pub const fn new(major: u16, minor: u16, patch: u16, build: u16) -> Self {
        Self {
            major,
            minor,
            patch,
            build,
        }
    }
}

impl std::fmt::Display for FileVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}.{}.{}.{}",
            self.major, self.minor, self.patch, self.build
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

pub type D3d12CreateDeviceFn = unsafe extern "system" fn(
    adapter: *mut c_void,
    minimum_feature_level: D3dFeatureLevel,
    interface_id: *const Guid,
    device: *mut *mut c_void,
) -> HResult;

pub type CreateDxgiFactory2Fn = unsafe extern "system" fn(
    flags: u32,
    interface_id: *const Guid,
    factory: *mut *mut c_void,
) -> HResult;

impl Guid {
    pub const fn from_values(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }
}

#[allow(dead_code)]
pub const IID_IDML_OBJECT: Guid = Guid::from_values(
    0xc8263aac,
    0x9e0c,
    0x4a2d,
    [0x9b, 0x8e, 0x00, 0x75, 0x21, 0xa3, 0x31, 0x7c],
);
pub const IID_IDML_DEVICE: Guid = Guid::from_values(
    0x6dbd6437,
    0x96fd,
    0x423f,
    [0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f],
);
#[allow(dead_code)]
pub const IID_IDML_DEVICE_CHILD: Guid = Guid::from_values(
    0x27e83142,
    0x8165,
    0x49e3,
    [0x97, 0x4e, 0x2f, 0xd6, 0x6e, 0x4c, 0xb6, 0x9d],
);
#[allow(dead_code)]
pub const IID_IDML_PAGEABLE: Guid = Guid::from_values(
    0xb1ab0825,
    0x4542,
    0x4a4b,
    [0x86, 0x17, 0x6d, 0xde, 0x6e, 0x8f, 0x62, 0x01],
);
#[allow(dead_code)]
pub const IID_IDML_DISPATCHABLE: Guid = Guid::from_values(
    0xdcb821a8,
    0x1039,
    0x441e,
    [0x9f, 0x1c, 0xb1, 0x75, 0x9c, 0x2f, 0x3c, 0xec],
);
pub const IID_IDML_OPERATOR: Guid = Guid::from_values(
    0x26caae7a,
    0x3081,
    0x4633,
    [0x95, 0x81, 0x22, 0x6f, 0xbe, 0x57, 0x69, 0x5d],
);
pub const IID_IDML_COMPILED_OPERATOR: Guid = Guid::from_values(
    0x6b15e56a,
    0xbf5c,
    0x4902,
    [0x92, 0xd8, 0xda, 0x3a, 0x65, 0x0a, 0xfe, 0xa4],
);
pub const IID_IDML_OPERATOR_INITIALIZER: Guid = Guid::from_values(
    0x427c1113,
    0x435c,
    0x469c,
    [0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13],
);
pub const IID_IDML_BINDING_TABLE: Guid = Guid::from_values(
    0x29c687dc,
    0xde74,
    0x4e3b,
    [0xab, 0x00, 0x11, 0x68, 0xf2, 0xfc, 0x3c, 0xfc],
);
pub const IID_IDML_COMMAND_RECORDER: Guid = Guid::from_values(
    0xe6857a76,
    0x2e3e,
    0x4fdd,
    [0xbf, 0xf4, 0x5d, 0x2b, 0xa1, 0x0f, 0xb4, 0x53],
);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlTensorDataType(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlTensorType(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlTensorFlags(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlOperatorType(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlFeature(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlFeatureLevel(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlExecutionFlags(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlCreateDeviceFlags(pub i32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DmlBindingType(pub i32);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlBufferTensorDesc {
    pub data_type: DmlTensorDataType,
    pub flags: DmlTensorFlags,
    pub dimension_count: u32,
    pub sizes: *const u32,
    pub strides: *const u32,
    pub total_tensor_size_in_bytes: u64,
    pub guaranteed_base_offset_alignment: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlTensorDesc {
    pub tensor_type: DmlTensorType,
    pub desc: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlOperatorDesc {
    pub operator_type: DmlOperatorType,
    pub desc: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlElementWiseAddOperatorDesc {
    pub a_tensor: *const DmlTensorDesc,
    pub b_tensor: *const DmlTensorDesc,
    pub output_tensor: *const DmlTensorDesc,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlFeatureQueryTensorDataTypeSupport {
    pub data_type: DmlTensorDataType,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlFeatureDataTensorDataTypeSupport {
    pub is_supported: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlFeatureQueryFeatureLevels {
    pub requested_feature_level_count: u32,
    pub requested_feature_levels: *const DmlFeatureLevel,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlFeatureDataFeatureLevels {
    pub max_supported_feature_level: DmlFeatureLevel,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlBindingProperties {
    pub required_descriptor_count: u32,
    pub temporary_resource_size: u64,
    pub persistent_resource_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlBindingDesc {
    pub binding_type: DmlBindingType,
    pub desc: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlBufferBinding {
    pub buffer: *mut c_void,
    pub offset: u64,
    pub size_in_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlBufferArrayBinding {
    pub binding_count: u32,
    pub bindings: *const DmlBufferBinding,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DmlBindingTableDesc {
    pub dispatchable: *mut c_void,
    pub cpu_descriptor_handle: usize,
    pub gpu_descriptor_handle: u64,
    pub size_in_descriptors: u32,
}

pub type HResult = i32;
pub type DmlCreateDeviceFn = unsafe extern "system" fn(
    *mut c_void,
    DmlCreateDeviceFlags,
    *const Guid,
    *mut *mut c_void,
) -> HResult;
pub type DmlCreateDevice1Fn = unsafe extern "system" fn(
    *mut c_void,
    DmlCreateDeviceFlags,
    DmlFeatureLevel,
    *const Guid,
    *mut *mut c_void,
) -> HResult;

#[repr(C)]
pub struct IUnknownVTable {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> HResult,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
pub struct DmlObjectVTable {
    pub base: IUnknownVTable,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut u32, *mut c_void) -> HResult,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, u32, *const c_void) -> HResult,
    pub set_private_data_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void) -> HResult,
    pub set_name: unsafe extern "system" fn(*mut c_void, *const u16) -> HResult,
}

#[repr(C)]
pub struct DmlDeviceVTable {
    pub base: DmlObjectVTable,
    pub check_feature_support: unsafe extern "system" fn(
        *mut c_void,
        DmlFeature,
        u32,
        *const c_void,
        u32,
        *mut c_void,
    ) -> HResult,
    pub create_operator: unsafe extern "system" fn(
        *mut c_void,
        *const DmlOperatorDesc,
        *const Guid,
        *mut *mut c_void,
    ) -> HResult,
    pub compile_operator: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        DmlExecutionFlags,
        *const Guid,
        *mut *mut c_void,
    ) -> HResult,
    pub create_operator_initializer: unsafe extern "system" fn(
        *mut c_void,
        u32,
        *const *mut c_void,
        *const Guid,
        *mut *mut c_void,
    ) -> HResult,
    pub create_command_recorder:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> HResult,
    pub create_binding_table: unsafe extern "system" fn(
        *mut c_void,
        *const DmlBindingTableDesc,
        *const Guid,
        *mut *mut c_void,
    ) -> HResult,
    pub evict: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) -> HResult,
    pub make_resident: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) -> HResult,
    pub get_device_removed_reason: unsafe extern "system" fn(*mut c_void) -> HResult,
    pub get_parent_device:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> HResult,
}

#[repr(C)]
pub struct DmlDeviceChildVTable {
    pub base: DmlObjectVTable,
    pub get_device:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> HResult,
}

#[repr(C)]
pub struct DmlDispatchableVTable {
    pub base: DmlDeviceChildVTable,
    pub get_binding_properties: unsafe extern "system" fn(*mut c_void) -> DmlBindingProperties,
}

#[repr(C)]
pub struct DmlOperatorInitializerVTable {
    pub base: DmlDispatchableVTable,
    pub reset: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) -> HResult,
}

#[repr(C)]
pub struct DmlBindingTableVTable {
    pub base: DmlDeviceChildVTable,
    pub bind_inputs: unsafe extern "system" fn(*mut c_void, u32, *const DmlBindingDesc),
    pub bind_outputs: unsafe extern "system" fn(*mut c_void, u32, *const DmlBindingDesc),
    pub bind_temporary_resource: unsafe extern "system" fn(*mut c_void, *const DmlBindingDesc),
    pub bind_persistent_resource: unsafe extern "system" fn(*mut c_void, *const DmlBindingDesc),
    pub reset: unsafe extern "system" fn(*mut c_void, *const DmlBindingTableDesc) -> HResult,
}

#[repr(C)]
pub struct DmlCommandRecorderVTable {
    pub base: DmlDeviceChildVTable,
    pub record_dispatch:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbiManifest {
    pub schema_version: u32,
    pub backend: String,
    pub abi_floor: String,
    pub target_version: String,
    pub minimum_feature_level: String,
    pub minimum_windows_build: u32,
    pub targets: Vec<String>,
    pub reviewed_package: ReviewedPackage,
    pub headers: Vec<HeaderContract>,
    pub libraries: Vec<LibraryContract>,
    pub interfaces: Vec<InterfaceContract>,
    pub layouts: Vec<LayoutContract>,
    pub redistributable: RedistributableContract,
    pub unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReviewedPackage {
    pub id: String,
    pub version: String,
    pub nupkg_sha256: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HeaderContract {
    pub path: String,
    pub byte_length: u64,
    pub sha256: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LibraryContract {
    pub name: String,
    pub abi_version: String,
    pub discovery: Vec<String>,
    pub symbols: Vec<SymbolContract>,
    pub binding_owner: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SymbolContract {
    pub name: String,
    pub signature: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterfaceContract {
    pub name: String,
    pub iid: String,
    pub vtable_slots: usize,
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
pub struct RedistributableContract {
    pub file_version: String,
    pub authenticode_required: bool,
    pub final_application_signing_required: bool,
    pub architectures: Vec<ArchitectureContract>,
    pub license_files: Vec<LicenseContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureContract {
    pub target: String,
    pub source_path: String,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LicenseContract {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AbiManifestError {
    #[error("DirectML ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("DirectML ABI manifest violates the reviewed contract: {0}")]
    Contract(String),
}

impl AbiManifest {
    pub fn embedded() -> Result<Self, AbiManifestError> {
        let manifest = serde_json::from_str::<Self>(ABI_MANIFEST_JSON)
            .map_err(|error| AbiManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), AbiManifestError> {
        if self.schema_version != 1
            || self.backend != "directml"
            || self.abi_floor != ABI_FLOOR
            || self.target_version != "0x6200"
            || self.minimum_feature_level != "DML_FEATURE_LEVEL_6_2"
            || self.minimum_windows_build != MINIMUM_WINDOWS_BUILD
            || self.unsafe_owner != UNSAFE_OWNER
        {
            return Err(AbiManifestError::Contract(
                "identity, ABI floor, target version, Windows floor, or unsafe owner differs"
                    .to_owned(),
            ));
        }
        if self.targets != ["aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc"] {
            return Err(AbiManifestError::Contract(
                "targets must be the two reviewed Windows MSVC targets".to_owned(),
            ));
        }
        validate_reviewed_package(&self.reviewed_package)?;
        validate_headers(&self.headers)?;
        validate_libraries(&self.libraries)?;
        validate_interfaces(&self.interfaces)?;
        validate_layouts(&self.layouts)?;
        validate_redistributable(&self.redistributable)?;
        Ok(())
    }

    pub fn directml_library(&self) -> Result<&LibraryContract, AbiManifestError> {
        self.libraries
            .iter()
            .find(|library| library.name == "DirectML.dll")
            .ok_or_else(|| AbiManifestError::Contract("DirectML.dll row is missing".to_owned()))
    }
}

fn validate_reviewed_package(package: &ReviewedPackage) -> Result<(), AbiManifestError> {
    if package.id != "Microsoft.AI.DirectML"
        || package.version != ABI_FLOOR
        || package.nupkg_sha256
            != "a38cef0d59f314fbcc0cd6551c5a762db7cfdaf8a977f85df32a0b1e279d3ba7"
        || package.source != "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.13.1"
    {
        return Err(AbiManifestError::Contract(
            "reviewed Microsoft package identity or digest differs".to_owned(),
        ));
    }
    Ok(())
}

fn validate_headers(headers: &[HeaderContract]) -> Result<(), AbiManifestError> {
    let expected = [
        (
            "include/DirectML.h",
            74_869_u64,
            "87971f944896fb983d7655a17a913f0417f6b3260eaa78853b3450124dc72239",
        ),
        (
            "windows-0.61.3/Win32/AI/MachineLearning/DirectML/mod.rs",
            140_416,
            "dc72b21a574679ade5b02a08003ea58cb3d766cb7a1b8d75256aaa9e5dda1e9c",
        ),
        (
            "windows-0.61.3/Win32/Graphics/Direct3D12/mod.rs",
            1_073_792,
            "a496524f82273e4f9efec04a360799d19b8b30362c75d0c56a1d2ab9bbf069a9",
        ),
        (
            "windows-0.61.3/Win32/Graphics/Dxgi/mod.rs",
            387_278,
            "9e61e4880c444ad90db534b173daa36cabfb5d26e240299306e923e7fcf30f43",
        ),
    ];
    if headers.len() != expected.len()
        || !expected.iter().all(|(path, byte_length, sha256)| {
            headers.iter().any(|header| {
                header.path == *path
                    && header.byte_length == *byte_length
                    && header.sha256 == *sha256
                    && !header.source.is_empty()
            })
        })
    {
        return Err(AbiManifestError::Contract(
            "reviewed header and generated-binding evidence is incomplete".to_owned(),
        ));
    }
    Ok(())
}

fn validate_libraries(libraries: &[LibraryContract]) -> Result<(), AbiManifestError> {
    let expected = [
        (
            "D3D12.dll",
            "windows-10.0.19041",
            ["D3D12CreateDevice"].as_slice(),
        ),
        (
            "DirectML.dll",
            ABI_FLOOR,
            ["DMLCreateDevice", "DMLCreateDevice1"].as_slice(),
        ),
        (
            "DXGI.dll",
            "windows-10.0.19041",
            ["CreateDXGIFactory2"].as_slice(),
        ),
    ];
    if libraries.len() != expected.len() {
        return Err(AbiManifestError::Contract(
            "library coverage is incomplete".to_owned(),
        ));
    }
    for (library_name, abi_version, symbols) in expected {
        let library = libraries
            .iter()
            .find(|library| library.name == library_name)
            .ok_or_else(|| {
                AbiManifestError::Contract(format!("required library {library_name} is missing"))
            })?;
        let actual = library
            .symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>();
        if library.abi_version != abi_version
            || actual != symbols
            || library
                .symbols
                .iter()
                .any(|symbol| symbol.signature.is_empty())
            || library.discovery.is_empty()
            || library.binding_owner.is_empty()
        {
            return Err(AbiManifestError::Contract(format!(
                "library {library_name} ABI, symbol, signature, discovery, or owner contract differs"
            )));
        }
    }
    Ok(())
}

fn validate_interfaces(interfaces: &[InterfaceContract]) -> Result<(), AbiManifestError> {
    const EXPECTED: [(&str, &str, usize); 10] = [
        ("IDMLObject", "c8263aac-9e0c-4a2d-9b8e-007521a3317c", 7),
        ("IDMLDevice", "6dbd6437-96fd-423f-a98c-ae5e7c2a573f", 17),
        ("IDMLDeviceChild", "27e83142-8165-49e3-974e-2fd66e4cb69d", 8),
        ("IDMLPageable", "b1ab0825-4542-4a4b-8617-6dde6e8f6201", 8),
        (
            "IDMLDispatchable",
            "dcb821a8-1039-441e-9f1c-b1759c2f3cec",
            9,
        ),
        ("IDMLOperator", "26caae7a-3081-4633-9581-226fbe57695d", 8),
        (
            "IDMLCompiledOperator",
            "6b15e56a-bf5c-4902-92d8-da3a650afea4",
            9,
        ),
        (
            "IDMLOperatorInitializer",
            "427c1113-435c-469c-8676-4d5dd072f813",
            10,
        ),
        (
            "IDMLBindingTable",
            "29c687dc-de74-4e3b-ab00-1168f2fc3cfc",
            13,
        ),
        (
            "IDMLCommandRecorder",
            "e6857a76-2e3e-4fdd-bff4-5d2ba10fb453",
            9,
        ),
    ];
    if interfaces.len() != EXPECTED.len()
        || !EXPECTED.iter().all(|(name, iid, slots)| {
            interfaces.iter().any(|interface| {
                interface.name == *name && interface.iid == *iid && interface.vtable_slots == *slots
            })
        })
    {
        return Err(AbiManifestError::Contract(
            "COM interface IID or vtable coverage differs".to_owned(),
        ));
    }
    Ok(())
}

fn validate_layouts(layouts: &[LayoutContract]) -> Result<(), AbiManifestError> {
    let expected = [
        ("DML_BINDING_DESC", 16, 8),
        ("DML_BINDING_PROPERTIES", 24, 8),
        ("DML_BINDING_TABLE_DESC", 32, 8),
        ("DML_BUFFER_ARRAY_BINDING", 16, 8),
        ("DML_BUFFER_BINDING", 24, 8),
        ("DML_BUFFER_TENSOR_DESC", 48, 8),
        ("DML_FEATURE_DATA_FEATURE_LEVELS", 4, 4),
        ("DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT", 4, 4),
        ("DML_FEATURE_QUERY_FEATURE_LEVELS", 16, 8),
        ("DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT", 4, 4),
        ("DML_ELEMENT_WISE_ADD_OPERATOR_DESC", 24, 8),
        ("DML_OPERATOR_DESC", 16, 8),
        ("DML_TENSOR_DESC", 16, 8),
        ("GUID", 16, 4),
    ];
    if layouts.len() != expected.len()
        || !expected.iter().all(|(name, size, align)| {
            layouts.iter().any(|layout| {
                layout.name == *name && layout.size == *size && layout.align == *align
            })
        })
    {
        return Err(AbiManifestError::Contract(
            "64-bit structure layout contract differs".to_owned(),
        ));
    }
    Ok(())
}

fn validate_redistributable(
    redistributable: &RedistributableContract,
) -> Result<(), AbiManifestError> {
    if redistributable.file_version != FILE_VERSION.to_string()
        || !redistributable.authenticode_required
        || !redistributable.final_application_signing_required
        || redistributable.architectures.len() != 2
        || redistributable.license_files.len() != 3
        || redistributable.architectures.iter().any(|architecture| {
            !self_consistent_architecture(architecture)
                || architecture.sha256.len() != 64
                || architecture.byte_length == 0
        })
        || redistributable
            .license_files
            .iter()
            .any(|license| license.path.is_empty() || license.sha256.len() != 64)
    {
        return Err(AbiManifestError::Contract(
            "redistributable version, architecture, signature, or license contract differs"
                .to_owned(),
        ));
    }
    Ok(())
}

fn self_consistent_architecture(architecture: &ArchitectureContract) -> bool {
    match architecture.target.as_str() {
        "aarch64-pc-windows-msvc" => {
            architecture.source_path == "bin/arm64-win/DirectML.dll"
                && architecture.byte_length == 13_947_936
                && architecture.sha256
                    == "96a5b8b75c4cd5e47fe9a4a87d0146276930fa58083b0538fd1b503e98b842f4"
        }
        "x86_64-pc-windows-msvc" => {
            architecture.source_path == "bin/x64-win/DirectML.dll"
                && architecture.byte_length == 14_021_152
                && architecture.sha256
                    == "5ab77cc5db8e1544d386fd28586598317da8dcbef098fb86d8d8a60e739e0e5d"
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn embedded_manifest_matches_the_reviewed_directml_1_13_contract() {
        let manifest = AbiManifest::embedded().expect("embedded DirectML ABI must be valid");
        assert_eq!(
            manifest
                .directml_library()
                .expect("DirectML row")
                .symbols
                .len(),
            2
        );
    }

    #[test]
    fn checked_declarations_match_reviewed_64_bit_layouts_and_vtables() {
        assert_eq!(D3D_FEATURE_LEVEL_11_0, D3dFeatureLevel(0xb000));
        assert_eq!(size_of::<D3dFeatureLevel>(), size_of::<i32>());
        assert_eq!((size_of::<Guid>(), align_of::<Guid>()), (16, 4));
        assert_eq!(
            (
                size_of::<DmlBufferTensorDesc>(),
                align_of::<DmlBufferTensorDesc>()
            ),
            (48, 8)
        );
        assert_eq!(
            (size_of::<DmlTensorDesc>(), align_of::<DmlTensorDesc>()),
            (16, 8)
        );
        assert_eq!(
            (size_of::<DmlOperatorDesc>(), align_of::<DmlOperatorDesc>()),
            (16, 8)
        );
        assert_eq!(
            (
                size_of::<DmlElementWiseAddOperatorDesc>(),
                align_of::<DmlElementWiseAddOperatorDesc>(),
            ),
            (24, 8)
        );
        assert_eq!(
            (
                size_of::<DmlFeatureQueryFeatureLevels>(),
                align_of::<DmlFeatureQueryFeatureLevels>()
            ),
            (16, 8)
        );
        assert_eq!(
            (
                size_of::<DmlBindingProperties>(),
                align_of::<DmlBindingProperties>()
            ),
            (24, 8)
        );
        assert_eq!(
            (size_of::<DmlBindingDesc>(), align_of::<DmlBindingDesc>()),
            (16, 8)
        );
        assert_eq!(
            (
                size_of::<DmlBufferBinding>(),
                align_of::<DmlBufferBinding>()
            ),
            (24, 8)
        );
        assert_eq!(
            (
                size_of::<DmlBufferArrayBinding>(),
                align_of::<DmlBufferArrayBinding>()
            ),
            (16, 8)
        );
        assert_eq!(
            (
                size_of::<DmlBindingTableDesc>(),
                align_of::<DmlBindingTableDesc>()
            ),
            (32, 8)
        );
        assert_eq!(size_of::<IUnknownVTable>(), 3 * size_of::<usize>());
        assert_eq!(size_of::<DmlObjectVTable>(), 7 * size_of::<usize>());
        assert_eq!(size_of::<DmlDeviceVTable>(), 17 * size_of::<usize>());
    }

    #[test]
    fn malformed_or_tampered_manifest_fails_closed() {
        let tampered = ABI_MANIFEST_JSON.replace("1.13.1", "1.13.0");
        let manifest = serde_json::from_str::<AbiManifest>(&tampered).expect("shape remains JSON");
        assert!(matches!(
            manifest.validate(),
            Err(AbiManifestError::Contract(_))
        ));
    }
}
