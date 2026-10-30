#![cfg_attr(
    not(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )),
    allow(dead_code)
)]

use crate::abi::{
    ABI_FLOOR, AbiManifest, AbiManifestError, CreateDxgiFactory2Fn, D3d12CreateDeviceFn,
    DmlCreateDevice1Fn, DmlCreateDeviceFn, FILE_VERSION, FileVersion, MINIMUM_WINDOWS_BUILD,
    UNSAFE_OWNER,
};
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};
use thiserror::Error;

#[allow(dead_code)]
const D3D12_LIBRARY_ID: &str = "D3D12.dll";
const DIRECTML_LIBRARY_ID: &str = "DirectML.dll";
#[allow(dead_code)]
const DXGI_LIBRARY_ID: &str = "DXGI.dll";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscoverySource {
    SignedApplicationPackage,
    CompatibleSystemComponent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlCandidate {
    source: DiscoverySource,
    path: PathBuf,
}

impl DirectMlCandidate {
    pub const fn source(&self) -> DiscoverySource {
        self.source
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlDiscoveryPlan {
    target: String,
    candidates: [DirectMlCandidate; 2],
}

impl DirectMlDiscoveryPlan {
    pub fn for_current_system(
        target: impl Into<String>,
        application_directory: impl Into<PathBuf>,
    ) -> Result<Self, DirectMlLoadError> {
        let target = target.into();
        let system_directory = canonical_system_directory(&target)?;
        Self::for_target(target, application_directory, system_directory)
    }

    pub fn for_target(
        target: impl Into<String>,
        application_directory: impl Into<PathBuf>,
        system_directory: impl Into<PathBuf>,
    ) -> Result<Self, DirectMlLoadError> {
        let target = target.into();
        ensure_target(&target)?;
        Ok(Self {
            target,
            candidates: [
                DirectMlCandidate {
                    source: DiscoverySource::SignedApplicationPackage,
                    path: application_directory.into().join(DIRECTML_LIBRARY_ID),
                },
                DirectMlCandidate {
                    source: DiscoverySource::CompatibleSystemComponent,
                    path: system_directory.into().join(DIRECTML_LIBRARY_ID),
                },
            ],
        })
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn candidates(&self) -> &[DirectMlCandidate; 2] {
        &self.candidates
    }

    pub fn validate_system_directory(&self) -> Result<(), DirectMlLoadError> {
        let candidate = self
            .candidates
            .get(1)
            .ok_or(DirectMlLoadError::SystemDirectoryMismatch)?;
        if candidate.source != DiscoverySource::CompatibleSystemComponent {
            return Err(DirectMlLoadError::SystemDirectoryMismatch);
        }
        let candidate_directory = candidate
            .path
            .parent()
            .ok_or(DirectMlLoadError::SystemDirectoryMismatch)?;
        let system_directory = canonical_system_directory(&self.target)?;
        if !windows_paths_equal(candidate_directory, &system_directory) {
            return Err(DirectMlLoadError::SystemDirectoryMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlCandidateObservation {
    target: String,
    source: DiscoverySource,
    path: PathBuf,
    windows_build: u32,
    file_version: FileVersion,
    digest_sha256: String,
}

impl DirectMlCandidateObservation {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub const fn source(&self) -> DiscoverySource {
        self.source
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn windows_build(&self) -> u32 {
        self.windows_build
    }

    pub const fn file_version(&self) -> FileVersion {
        self.file_version
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }
}

pub fn observe_directml_candidate(
    plan: &DirectMlDiscoveryPlan,
    certified_digest_sha256: impl Into<String>,
) -> Result<DirectMlCandidateObservation, DirectMlLoadError> {
    let certified_digest_sha256 = certified_digest_sha256.into();
    if !is_sha256(&certified_digest_sha256) {
        return Err(DirectMlLoadError::InvalidObservationDigest);
    }
    let candidate = plan
        .candidates
        .first()
        .ok_or(DirectMlLoadError::ObservationOutsidePlan)?;
    if candidate.source != DiscoverySource::SignedApplicationPackage {
        return Err(DirectMlLoadError::ObservationOutsidePlan);
    }
    let windows_build = observe_windows_build(&plan.target)?;
    let file_version = observe_file_version(&plan.target, &candidate.path)?;
    verify_authenticode_offline(&plan.target, &candidate.path)?;
    Ok(DirectMlCandidateObservation {
        target: plan.target.clone(),
        source: candidate.source,
        path: candidate.path.clone(),
        windows_build,
        file_version,
        digest_sha256: certified_digest_sha256,
    })
}

#[derive(Debug)]
pub struct RegistryCertifiedDirectMlImage {
    library_id: String,
    digest_sha256: String,
    abi_version: String,
    required_symbols: BTreeSet<String>,
    unsafe_owner: String,
    module: OwnedModule,
}

impl RegistryCertifiedDirectMlImage {
    /// Loads one immutable image whose fields were copied from a live registry certificate.
    ///
    /// # Safety
    ///
    /// Every field must be copied from the exact certificate issued by
    /// `sim_runtime::NativeFfiRegistry`, and `image_path` must name the exact sealed image whose
    /// digest the certificate covers. The returned value owns the module until drop.
    pub unsafe fn load_from_registry_certificate(
        library_id: impl Into<String>,
        digest_sha256: impl Into<String>,
        abi_version: impl Into<String>,
        required_symbols: BTreeSet<String>,
        unsafe_owner: impl Into<String>,
        image_path: &Path,
    ) -> Result<Self, DirectMlLoadError> {
        let library_id = library_id.into();
        Ok(Self {
            digest_sha256: digest_sha256.into(),
            abi_version: abi_version.into(),
            required_symbols,
            unsafe_owner: unsafe_owner.into(),
            module: unsafe { OwnedModule::load_exact(&library_id, image_path)? },
            library_id,
        })
    }

    #[cfg(test)]
    unsafe fn from_test_registry_certificate(
        library_id: impl Into<String>,
        digest_sha256: impl Into<String>,
        abi_version: impl Into<String>,
        required_symbols: BTreeSet<String>,
        unsafe_owner: impl Into<String>,
        retained_module_handle: *mut c_void,
    ) -> Result<Self, DirectMlLoadError> {
        let library_id = library_id.into();
        Ok(Self {
            digest_sha256: digest_sha256.into(),
            abi_version: abi_version.into(),
            required_symbols,
            unsafe_owner: unsafe_owner.into(),
            module: OwnedModule::from_test_handle(&library_id, retained_module_handle)?,
            library_id,
        })
    }
}

#[derive(Debug)]
pub struct RetainedDirectMlLibraryHandles {
    images: BTreeMap<String, CertifiedImage>,
    _retention: Arc<dyn Any + Send + Sync>,
}

#[derive(Debug)]
struct CertifiedImage {
    digest_sha256: String,
    module: OwnedModule,
}

#[derive(Debug)]
struct OwnedModule {
    handle: NonNull<c_void>,
    release_on_drop: bool,
    #[cfg(test)]
    drop_observer: Option<Arc<std::sync::atomic::AtomicUsize>>,
}

unsafe impl Send for OwnedModule {}
unsafe impl Sync for OwnedModule {}

impl OwnedModule {
    #[cfg(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    unsafe fn load_exact(library: &str, path: &Path) -> Result<Self, DirectMlLoadError> {
        use std::os::windows::ffi::OsStrExt;
        use windows::{
            Win32::System::LibraryLoader::{
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
            },
            core::PCWSTR,
        };

        let mut wide_path = path.as_os_str().encode_wide().collect::<Vec<_>>();
        wide_path.push(0);
        let handle = unsafe {
            LoadLibraryExW(
                PCWSTR::from_raw(wide_path.as_ptr()),
                None,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        }
        .map_err(|error| DirectMlLoadError::ModuleLoad {
            library: library.to_owned(),
            reason: error.to_string(),
        })?;
        let handle = NonNull::new(handle.0).ok_or_else(|| DirectMlLoadError::ModuleLoad {
            library: library.to_owned(),
            reason: "LoadLibraryExW returned a null module".to_owned(),
        })?;
        Ok(Self {
            handle,
            release_on_drop: true,
            #[cfg(test)]
            drop_observer: None,
        })
    }

    #[cfg(not(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    unsafe fn load_exact(library: &str, _path: &Path) -> Result<Self, DirectMlLoadError> {
        Err(DirectMlLoadError::UnsupportedImageLoad {
            library: library.to_owned(),
        })
    }

    #[cfg(test)]
    fn from_test_handle(library: &str, handle: *mut c_void) -> Result<Self, DirectMlLoadError> {
        Ok(Self {
            handle: NonNull::new(handle).ok_or_else(|| DirectMlLoadError::UncertifiedHandle {
                library: library.to_owned(),
            })?,
            release_on_drop: false,
            drop_observer: None,
        })
    }
}

impl Drop for OwnedModule {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(observer) = &self.drop_observer {
            observer.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        if !self.release_on_drop {
            return;
        }
        #[cfg(all(
            target_os = "windows",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        unsafe {
            use windows::Win32::Foundation::{FreeLibrary, HMODULE};
            if let Err(error) = FreeLibrary(HMODULE(self.handle.as_ptr())) {
                eprintln!("failed to release registry-certified DirectML module: {error}");
            }
        }
    }
}

impl RetainedDirectMlLibraryHandles {
    /// Projects exact registry-certified images into one owned lifetime bundle.
    ///
    /// # Safety
    ///
    /// Every row must be copied directly from a live `sim_runtime::NativeFfiRegistry`
    /// certificate, and `retention` must own those certificates and the immutable source images.
    /// Discovery paths, Authenticode observations, package manifests, and feature compilation
    /// cannot satisfy this contract.
    pub unsafe fn from_registry_certificates(
        retention: Arc<dyn Any + Send + Sync>,
        images: impl IntoIterator<Item = RegistryCertifiedDirectMlImage>,
    ) -> Result<Self, DirectMlLoadError> {
        let manifest = AbiManifest::embedded()?;
        let mut checked = BTreeMap::new();
        for image in images {
            let library = manifest
                .libraries
                .iter()
                .find(|library| library.name == image.library_id)
                .ok_or_else(|| DirectMlLoadError::UnexpectedCertifiedLibrary {
                    library: image.library_id.clone(),
                })?;
            validate_certificate_projection(&image, library)?;
            let library_id = image.library_id;
            let certified = CertifiedImage {
                digest_sha256: image.digest_sha256,
                module: image.module,
            };
            if checked.insert(library_id.clone(), certified).is_some() {
                return Err(DirectMlLoadError::DuplicateCertifiedLibrary {
                    library: library_id,
                });
            }
        }
        for library in &manifest.libraries {
            if !checked.contains_key(&library.name) {
                return Err(DirectMlLoadError::MissingCertifiedLibrary {
                    library: library.name.clone(),
                });
            }
        }
        Ok(Self {
            images: checked,
            _retention: retention,
        })
    }

    pub fn retained_library_count(&self) -> usize {
        self.images.len()
    }

    pub(crate) fn certification_retention(&self) -> Arc<dyn Any + Send + Sync> {
        self._retention.clone()
    }

    fn image(&self, library: &str) -> Result<&CertifiedImage, DirectMlLoadError> {
        self.images
            .get(library)
            .ok_or_else(|| DirectMlLoadError::MissingCertifiedLibrary {
                library: library.to_owned(),
            })
    }
}

unsafe impl Send for RetainedDirectMlLibraryHandles {}
unsafe impl Sync for RetainedDirectMlLibraryHandles {}

struct DirectMlSymbols {
    d3d12_create_device: D3d12CreateDeviceFn,
    create_dxgi_factory2: CreateDxgiFactory2Fn,
    _dml_create_device: DmlCreateDeviceFn,
    dml_create_device1: DmlCreateDevice1Fn,
}

unsafe impl Send for DirectMlSymbols {}
unsafe impl Sync for DirectMlSymbols {}

impl DirectMlSymbols {
    /// Resolves the reviewed exports from caller-retained registry-certified module handles.
    ///
    /// # Safety
    ///
    /// `handles` must satisfy `RetainedDirectMlLibraryHandles`' certification, immutability, and
    /// lifetime contract. The returned function pointers must not outlive those handles.
    unsafe fn load(handles: &RetainedDirectMlLibraryHandles) -> Result<Self, DirectMlLoadError> {
        #[cfg(all(
            target_os = "windows",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        {
            let manifest = AbiManifest::embedded()?;
            let mut resolver = WindowsRetainedResolver { handles };
            let resolved = resolve_with(&mut resolver, &manifest)?;
            let directml = handles.image(DIRECTML_LIBRARY_ID)?;
            let d3d12 = handles.image(D3D12_LIBRARY_ID)?;
            let dxgi = handles.image(DXGI_LIBRARY_ID)?;
            let d3d12_create_device =
                required_address(d3d12, D3D12_LIBRARY_ID, "D3D12CreateDevice")?;
            let create_dxgi_factory2 =
                required_address(dxgi, DXGI_LIBRARY_ID, "CreateDXGIFactory2")?;
            let create = required_address(directml, DIRECTML_LIBRARY_ID, "DMLCreateDevice")?;
            let create1 = required_address(directml, DIRECTML_LIBRARY_ID, "DMLCreateDevice1")?;
            debug_assert_eq!(resolved.symbol_count, 4);
            Ok(Self {
                d3d12_create_device: unsafe {
                    std::mem::transmute::<*mut c_void, D3d12CreateDeviceFn>(
                        d3d12_create_device.as_ptr(),
                    )
                },
                create_dxgi_factory2: unsafe {
                    std::mem::transmute::<*mut c_void, CreateDxgiFactory2Fn>(
                        create_dxgi_factory2.as_ptr(),
                    )
                },
                _dml_create_device: unsafe {
                    std::mem::transmute::<*mut c_void, DmlCreateDeviceFn>(create.as_ptr())
                },
                dml_create_device1: unsafe {
                    std::mem::transmute::<*mut c_void, DmlCreateDevice1Fn>(create1.as_ptr())
                },
            })
        }
        #[cfg(not(all(
            target_os = "windows",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )))]
        {
            let _ = handles;
            Err(DirectMlLoadError::UnsupportedTarget {
                target: env!("SIM_DIRECTML_TARGET").to_owned(),
            })
        }
    }
}

pub(crate) struct CertifiedDirectMlExecutionInputs {
    symbols: DirectMlSymbols,
    _handles: RetainedDirectMlLibraryHandles,
}

unsafe impl Send for CertifiedDirectMlExecutionInputs {}
unsafe impl Sync for CertifiedDirectMlExecutionInputs {}

impl RetainedDirectMlLibraryHandles {
    pub(crate) fn into_execution_inputs(
        self,
    ) -> Result<CertifiedDirectMlExecutionInputs, DirectMlLoadError> {
        let symbols = unsafe { DirectMlSymbols::load(&self)? };
        Ok(CertifiedDirectMlExecutionInputs {
            symbols,
            _handles: self,
        })
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod com_wrappers {
    use super::*;
    use crate::abi::{
        D3D_FEATURE_LEVEL_11_0, DML_BINDING_TYPE_BUFFER, DML_BINDING_TYPE_BUFFER_ARRAY,
        DML_BINDING_TYPE_NONE, DML_CREATE_DEVICE_FLAG_DEBUG, DML_CREATE_DEVICE_FLAG_NONE,
        DML_FEATURE_FEATURE_LEVELS, DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, DmlBindingDesc,
        DmlBindingProperties, DmlBindingTableDesc, DmlBindingTableVTable, DmlBufferArrayBinding,
        DmlBufferBinding, DmlCommandRecorderVTable, DmlDeviceVTable, DmlDispatchableVTable,
        DmlExecutionFlags, DmlFeatureDataFeatureLevels, DmlFeatureDataTensorDataTypeSupport,
        DmlFeatureQueryFeatureLevels, DmlFeatureQueryTensorDataTypeSupport, DmlOperatorDesc,
        DmlOperatorInitializerVTable, DmlTensorDataType, Guid, HResult, IID_IDML_BINDING_TABLE,
        IID_IDML_COMMAND_RECORDER, IID_IDML_COMPILED_OPERATOR, IID_IDML_DEVICE, IID_IDML_OPERATOR,
        IID_IDML_OPERATOR_INITIALIZER, IUnknownVTable, MINIMUM_FEATURE_LEVEL,
    };
    use std::{marker::PhantomData, ptr};
    use windows::{
        Win32::Graphics::Direct3D12::{
            D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE, ID3D12Device,
            ID3D12GraphicsCommandList, ID3D12Resource,
        },
        Win32::Graphics::Dxgi::{IDXGIAdapter4, IDXGIFactory6},
        core::Interface,
    };

    impl CertifiedDirectMlExecutionInputs {
        pub(crate) fn create_dxgi_factory6(&self) -> Result<IDXGIFactory6, DirectMlLoadError> {
            let mut factory = ptr::null_mut();
            let interface_id = (&IDXGIFactory6::IID as *const windows::core::GUID).cast::<Guid>();
            let result =
                unsafe { (self.symbols.create_dxgi_factory2)(0, interface_id, &mut factory) };
            check_hresult("CreateDXGIFactory2", result)?;
            if factory.is_null() {
                return Err(DirectMlLoadError::NullComResult {
                    operation: "CreateDXGIFactory2",
                });
            }
            Ok(unsafe { IDXGIFactory6::from_raw(factory) })
        }

        pub(crate) fn create_d3d12_device(
            &self,
            adapter: &IDXGIAdapter4,
        ) -> Result<ID3D12Device, DirectMlLoadError> {
            let mut device = ptr::null_mut();
            let interface_id = (&ID3D12Device::IID as *const windows::core::GUID).cast::<Guid>();
            let result = unsafe {
                (self.symbols.d3d12_create_device)(
                    adapter.as_raw(),
                    D3D_FEATURE_LEVEL_11_0,
                    interface_id,
                    &mut device,
                )
            };
            check_hresult("D3D12CreateDevice", result)?;
            if device.is_null() {
                return Err(DirectMlLoadError::NullComResult {
                    operation: "D3D12CreateDevice",
                });
            }
            Ok(unsafe { ID3D12Device::from_raw(device) })
        }

        pub(crate) fn create_directml_device(
            &self,
            d3d12_device: &ID3D12Device,
            debug: bool,
        ) -> Result<DirectMlDevice, DirectMlLoadError> {
            let flags = if debug {
                DML_CREATE_DEVICE_FLAG_DEBUG
            } else {
                DML_CREATE_DEVICE_FLAG_NONE
            };
            let mut device = ptr::null_mut();
            let result = unsafe {
                (self.symbols.dml_create_device1)(
                    d3d12_device.as_raw(),
                    flags,
                    MINIMUM_FEATURE_LEVEL,
                    &IID_IDML_DEVICE,
                    &mut device,
                )
            };
            check_hresult("DMLCreateDevice1", result)?;
            Ok(DirectMlDevice {
                object: ComObject::from_raw("DMLCreateDevice1", device)?,
            })
        }
    }

    struct ComObject {
        raw: NonNull<c_void>,
    }

    impl ComObject {
        fn from_raw(operation: &'static str, raw: *mut c_void) -> Result<Self, DirectMlLoadError> {
            Ok(Self {
                raw: NonNull::new(raw).ok_or(DirectMlLoadError::NullComResult { operation })?,
            })
        }

        fn raw(&self) -> *mut c_void {
            self.raw.as_ptr()
        }

        fn vtable<T>(&self) -> &T {
            unsafe { &**self.raw.as_ptr().cast::<*const T>() }
        }
    }

    impl Drop for ComObject {
        fn drop(&mut self) {
            let vtable = self.vtable::<IUnknownVTable>();
            unsafe { (vtable.release)(self.raw()) };
        }
    }

    unsafe impl Send for ComObject {}
    unsafe impl Sync for ComObject {}

    pub(crate) struct DirectMlDevice {
        object: ComObject,
    }

    impl DirectMlDevice {
        pub fn maximum_supported_feature_level(
            &self,
            requested: &[i32],
        ) -> Result<i32, DirectMlLoadError> {
            if requested.is_empty() || requested.len() > u32::MAX as usize {
                return Err(DirectMlLoadError::InvalidArgument {
                    operation: "IDMLDevice::CheckFeatureSupport(feature levels)",
                    reason: "requested feature levels must be nonempty and fit u32".to_owned(),
                });
            }
            let requested = requested
                .iter()
                .copied()
                .map(crate::abi::DmlFeatureLevel)
                .collect::<Vec<_>>();
            let query = DmlFeatureQueryFeatureLevels {
                requested_feature_level_count: requested.len() as u32,
                requested_feature_levels: requested.as_ptr(),
            };
            let mut result = DmlFeatureDataFeatureLevels {
                max_supported_feature_level: crate::abi::DmlFeatureLevel(0),
            };
            let status = unsafe {
                (self
                    .object
                    .vtable::<DmlDeviceVTable>()
                    .check_feature_support)(
                    self.object.raw(),
                    DML_FEATURE_FEATURE_LEVELS,
                    std::mem::size_of_val(&query) as u32,
                    (&query as *const DmlFeatureQueryFeatureLevels).cast(),
                    std::mem::size_of_val(&result) as u32,
                    (&mut result as *mut DmlFeatureDataFeatureLevels).cast(),
                )
            };
            check_hresult("IDMLDevice::CheckFeatureSupport(feature levels)", status)?;
            Ok(result.max_supported_feature_level.0)
        }

        pub fn tensor_data_type_supported(
            &self,
            data_type: DmlTensorDataType,
        ) -> Result<bool, DirectMlLoadError> {
            let query = DmlFeatureQueryTensorDataTypeSupport { data_type };
            let mut result = DmlFeatureDataTensorDataTypeSupport { is_supported: 0 };
            let status = unsafe {
                (self
                    .object
                    .vtable::<DmlDeviceVTable>()
                    .check_feature_support)(
                    self.object.raw(),
                    DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,
                    std::mem::size_of_val(&query) as u32,
                    (&query as *const DmlFeatureQueryTensorDataTypeSupport).cast(),
                    std::mem::size_of_val(&result) as u32,
                    (&mut result as *mut DmlFeatureDataTensorDataTypeSupport).cast(),
                )
            };
            check_hresult("IDMLDevice::CheckFeatureSupport(tensor data type)", status)?;
            Ok(result.is_supported != 0)
        }

        pub(crate) fn create_operator(
            &self,
            descriptor: &DmlOperatorDesc,
        ) -> Result<DirectMlOperator, DirectMlLoadError> {
            if descriptor.desc.is_null() {
                return Err(DirectMlLoadError::InvalidArgument {
                    operation: "IDMLDevice::CreateOperator",
                    reason: "operator descriptor pointer is null".to_owned(),
                });
            }
            let mut operator = ptr::null_mut();
            let status = unsafe {
                (self.object.vtable::<DmlDeviceVTable>().create_operator)(
                    self.object.raw(),
                    descriptor,
                    &IID_IDML_OPERATOR,
                    &mut operator,
                )
            };
            check_hresult("IDMLDevice::CreateOperator", status)?;
            Ok(DirectMlOperator {
                object: ComObject::from_raw("IDMLDevice::CreateOperator", operator)?,
            })
        }

        pub(crate) fn compile_operator(
            &self,
            operator: &DirectMlOperator,
            flags: DmlExecutionFlags,
        ) -> Result<DirectMlCompiledOperator, DirectMlLoadError> {
            let mut compiled = ptr::null_mut();
            let status = unsafe {
                (self.object.vtable::<DmlDeviceVTable>().compile_operator)(
                    self.object.raw(),
                    operator.object.raw(),
                    flags,
                    &IID_IDML_COMPILED_OPERATOR,
                    &mut compiled,
                )
            };
            check_hresult("IDMLDevice::CompileOperator", status)?;
            Ok(DirectMlCompiledOperator {
                object: ComObject::from_raw("IDMLDevice::CompileOperator", compiled)?,
            })
        }

        pub(crate) fn create_operator_initializer(
            &self,
            operators: &[&DirectMlCompiledOperator],
        ) -> Result<DirectMlOperatorInitializer, DirectMlLoadError> {
            let count =
                u32::try_from(operators.len()).map_err(|_| DirectMlLoadError::InvalidArgument {
                    operation: "IDMLDevice::CreateOperatorInitializer",
                    reason: "operator count exceeds u32".to_owned(),
                })?;
            let pointers = operators
                .iter()
                .map(|operator| operator.object.raw())
                .collect::<Vec<_>>();
            let mut initializer = ptr::null_mut();
            let status = unsafe {
                (self
                    .object
                    .vtable::<DmlDeviceVTable>()
                    .create_operator_initializer)(
                    self.object.raw(),
                    count,
                    pointers.as_ptr(),
                    &IID_IDML_OPERATOR_INITIALIZER,
                    &mut initializer,
                )
            };
            check_hresult("IDMLDevice::CreateOperatorInitializer", status)?;
            Ok(DirectMlOperatorInitializer {
                object: ComObject::from_raw("IDMLDevice::CreateOperatorInitializer", initializer)?,
            })
        }

        pub(crate) fn create_command_recorder(
            &self,
        ) -> Result<DirectMlCommandRecorder, DirectMlLoadError> {
            let mut recorder = ptr::null_mut();
            let status = unsafe {
                (self
                    .object
                    .vtable::<DmlDeviceVTable>()
                    .create_command_recorder)(
                    self.object.raw(),
                    &IID_IDML_COMMAND_RECORDER,
                    &mut recorder,
                )
            };
            check_hresult("IDMLDevice::CreateCommandRecorder", status)?;
            Ok(DirectMlCommandRecorder {
                object: ComObject::from_raw("IDMLDevice::CreateCommandRecorder", recorder)?,
            })
        }

        pub(crate) fn create_binding_table<'dispatchable>(
            &self,
            dispatchable: DirectMlDispatchable<'dispatchable>,
            cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
            gpu_descriptor_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
            size_in_descriptors: u32,
        ) -> Result<DirectMlBindingTable<'dispatchable>, DirectMlLoadError> {
            let descriptor = binding_table_descriptor(
                dispatchable,
                cpu_descriptor_handle,
                gpu_descriptor_handle,
                size_in_descriptors,
            )?;
            let mut table = ptr::null_mut();
            let status = unsafe {
                (self.object.vtable::<DmlDeviceVTable>().create_binding_table)(
                    self.object.raw(),
                    &descriptor,
                    &IID_IDML_BINDING_TABLE,
                    &mut table,
                )
            };
            check_hresult("IDMLDevice::CreateBindingTable", status)?;
            Ok(DirectMlBindingTable {
                object: ComObject::from_raw("IDMLDevice::CreateBindingTable", table)?,
                _dispatchable: PhantomData,
            })
        }

        pub fn removed_reason(&self) -> Result<(), DirectMlLoadError> {
            let status = unsafe {
                (self
                    .object
                    .vtable::<DmlDeviceVTable>()
                    .get_device_removed_reason)(self.object.raw())
            };
            check_hresult("IDMLDevice::GetDeviceRemovedReason", status)
        }
    }

    pub(crate) struct DirectMlOperator {
        object: ComObject,
    }

    pub(crate) struct DirectMlCompiledOperator {
        object: ComObject,
    }

    impl DirectMlCompiledOperator {
        pub(crate) fn binding_properties(&self) -> DmlBindingProperties {
            unsafe {
                (self
                    .object
                    .vtable::<DmlDispatchableVTable>()
                    .get_binding_properties)(self.object.raw())
            }
        }
    }

    pub(crate) struct DirectMlOperatorInitializer {
        object: ComObject,
    }

    impl DirectMlOperatorInitializer {
        pub(crate) fn reset(
            &self,
            operators: &[&DirectMlCompiledOperator],
        ) -> Result<(), DirectMlLoadError> {
            let count =
                u32::try_from(operators.len()).map_err(|_| DirectMlLoadError::InvalidArgument {
                    operation: "IDMLOperatorInitializer::Reset",
                    reason: "operator count exceeds u32".to_owned(),
                })?;
            let pointers = operators
                .iter()
                .map(|operator| operator.object.raw())
                .collect::<Vec<_>>();
            let status = unsafe {
                (self.object.vtable::<DmlOperatorInitializerVTable>().reset)(
                    self.object.raw(),
                    count,
                    pointers.as_ptr(),
                )
            };
            check_hresult("IDMLOperatorInitializer::Reset", status)
        }

        pub(crate) fn binding_properties(&self) -> DmlBindingProperties {
            unsafe {
                (self
                    .object
                    .vtable::<DmlDispatchableVTable>()
                    .get_binding_properties)(self.object.raw())
            }
        }
    }

    pub(crate) enum DirectMlDispatchable<'object> {
        Compiled(&'object DirectMlCompiledOperator),
        Initializer(&'object DirectMlOperatorInitializer),
    }

    impl DirectMlDispatchable<'_> {
        fn raw(&self) -> *mut c_void {
            match self {
                Self::Compiled(value) => value.object.raw(),
                Self::Initializer(value) => value.object.raw(),
            }
        }
    }

    pub(crate) enum DirectMlBinding<'resource> {
        None,
        Buffer {
            resource: &'resource ID3D12Resource,
            offset: u64,
            size_in_bytes: u64,
        },
        #[allow(dead_code, reason = "the reviewed binding ABI includes buffer arrays")]
        BufferArray(Vec<DirectMlBufferRange<'resource>>),
    }

    pub(crate) struct DirectMlBufferRange<'resource> {
        pub resource: &'resource ID3D12Resource,
        pub offset: u64,
        pub size_in_bytes: u64,
    }

    pub(crate) struct DirectMlBindingTable<'dispatchable> {
        object: ComObject,
        _dispatchable: PhantomData<DirectMlDispatchable<'dispatchable>>,
    }

    impl DirectMlBindingTable<'_> {
        pub(crate) fn bind_inputs(
            &self,
            bindings: &[DirectMlBinding<'_>],
        ) -> Result<(), DirectMlLoadError> {
            let checked = CheckedBindings::new(bindings)?;
            unsafe {
                (self.object.vtable::<DmlBindingTableVTable>().bind_inputs)(
                    self.object.raw(),
                    checked.descriptors.len() as u32,
                    checked.descriptors.as_ptr(),
                )
            };
            Ok(())
        }

        pub(crate) fn bind_outputs(
            &self,
            bindings: &[DirectMlBinding<'_>],
        ) -> Result<(), DirectMlLoadError> {
            let checked = CheckedBindings::new(bindings)?;
            unsafe {
                (self.object.vtable::<DmlBindingTableVTable>().bind_outputs)(
                    self.object.raw(),
                    checked.descriptors.len() as u32,
                    checked.descriptors.as_ptr(),
                )
            };
            Ok(())
        }

        pub(crate) fn bind_temporary_resource(
            &self,
            binding: &DirectMlBinding<'_>,
        ) -> Result<(), DirectMlLoadError> {
            let checked = CheckedBindings::new(std::slice::from_ref(binding))?;
            let descriptor =
                checked
                    .descriptors
                    .first()
                    .ok_or_else(|| DirectMlLoadError::InvalidArgument {
                        operation: "IDMLBindingTable::BindTemporaryResource",
                        reason: "checked binding descriptor is missing".to_owned(),
                    })?;
            unsafe {
                (self
                    .object
                    .vtable::<DmlBindingTableVTable>()
                    .bind_temporary_resource)(self.object.raw(), descriptor)
            };
            Ok(())
        }

        pub(crate) fn bind_persistent_resource(
            &self,
            binding: &DirectMlBinding<'_>,
        ) -> Result<(), DirectMlLoadError> {
            let checked = CheckedBindings::new(std::slice::from_ref(binding))?;
            let descriptor =
                checked
                    .descriptors
                    .first()
                    .ok_or_else(|| DirectMlLoadError::InvalidArgument {
                        operation: "IDMLBindingTable::BindPersistentResource",
                        reason: "checked binding descriptor is missing".to_owned(),
                    })?;
            unsafe {
                (self
                    .object
                    .vtable::<DmlBindingTableVTable>()
                    .bind_persistent_resource)(self.object.raw(), descriptor)
            };
            Ok(())
        }
    }

    pub(crate) struct DirectMlCommandRecorder {
        object: ComObject,
    }

    impl DirectMlCommandRecorder {
        pub(crate) fn record_dispatch(
            &self,
            command_list: &ID3D12GraphicsCommandList,
            dispatchable: DirectMlDispatchable<'_>,
            bindings: &DirectMlBindingTable<'_>,
        ) {
            unsafe {
                (self
                    .object
                    .vtable::<DmlCommandRecorderVTable>()
                    .record_dispatch)(
                    self.object.raw(),
                    command_list.as_raw(),
                    dispatchable.raw(),
                    bindings.object.raw(),
                )
            }
        }
    }

    struct CheckedBindings {
        descriptors: Vec<DmlBindingDesc>,
        buffers: Vec<DmlBufferBinding>,
        arrays: Vec<Vec<DmlBufferBinding>>,
        array_descriptors: Vec<DmlBufferArrayBinding>,
    }

    impl CheckedBindings {
        fn new(bindings: &[DirectMlBinding<'_>]) -> Result<Self, DirectMlLoadError> {
            if bindings.len() > u32::MAX as usize {
                return Err(DirectMlLoadError::InvalidArgument {
                    operation: "IDMLBindingTable binding",
                    reason: "binding count exceeds u32".to_owned(),
                });
            }
            let mut checked = Self {
                descriptors: Vec::with_capacity(bindings.len()),
                buffers: Vec::with_capacity(bindings.len()),
                arrays: Vec::with_capacity(bindings.len()),
                array_descriptors: Vec::with_capacity(bindings.len()),
            };
            for binding in bindings {
                match binding {
                    DirectMlBinding::None => checked.descriptors.push(DmlBindingDesc {
                        binding_type: DML_BINDING_TYPE_NONE,
                        desc: ptr::null(),
                    }),
                    DirectMlBinding::Buffer {
                        resource,
                        offset,
                        size_in_bytes,
                    } => {
                        if *size_in_bytes == 0 {
                            return Err(DirectMlLoadError::InvalidArgument {
                                operation: "IDMLBindingTable buffer binding",
                                reason: "buffer size must be nonzero".to_owned(),
                            });
                        }
                        checked.buffers.push(DmlBufferBinding {
                            buffer: resource.as_raw(),
                            offset: *offset,
                            size_in_bytes: *size_in_bytes,
                        });
                        let buffer = checked.buffers.last().ok_or_else(|| {
                            DirectMlLoadError::InvalidArgument {
                                operation: "IDMLBindingTable buffer binding",
                                reason: "buffer staging failed".to_owned(),
                            }
                        })?;
                        checked.descriptors.push(DmlBindingDesc {
                            binding_type: DML_BINDING_TYPE_BUFFER,
                            desc: (buffer as *const DmlBufferBinding).cast(),
                        });
                    }
                    DirectMlBinding::BufferArray(ranges) => {
                        let count = u32::try_from(ranges.len()).map_err(|_| {
                            DirectMlLoadError::InvalidArgument {
                                operation: "IDMLBindingTable buffer-array binding",
                                reason: "buffer-array count exceeds u32".to_owned(),
                            }
                        })?;
                        if count == 0 || ranges.iter().any(|range| range.size_in_bytes == 0) {
                            return Err(DirectMlLoadError::InvalidArgument {
                                operation: "IDMLBindingTable buffer-array binding",
                                reason: "buffer arrays and ranges must be nonempty".to_owned(),
                            });
                        }
                        checked.arrays.push(
                            ranges
                                .iter()
                                .map(|range| DmlBufferBinding {
                                    buffer: range.resource.as_raw(),
                                    offset: range.offset,
                                    size_in_bytes: range.size_in_bytes,
                                })
                                .collect(),
                        );
                        let values = checked.arrays.last().ok_or_else(|| {
                            DirectMlLoadError::InvalidArgument {
                                operation: "IDMLBindingTable buffer-array binding",
                                reason: "buffer-array staging failed".to_owned(),
                            }
                        })?;
                        checked.array_descriptors.push(DmlBufferArrayBinding {
                            binding_count: count,
                            bindings: values.as_ptr(),
                        });
                        let array = checked.array_descriptors.last().ok_or_else(|| {
                            DirectMlLoadError::InvalidArgument {
                                operation: "IDMLBindingTable buffer-array binding",
                                reason: "buffer-array descriptor staging failed".to_owned(),
                            }
                        })?;
                        checked.descriptors.push(DmlBindingDesc {
                            binding_type: DML_BINDING_TYPE_BUFFER_ARRAY,
                            desc: (array as *const DmlBufferArrayBinding).cast(),
                        });
                    }
                }
            }
            Ok(checked)
        }
    }

    fn binding_table_descriptor(
        dispatchable: DirectMlDispatchable<'_>,
        cpu_descriptor_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
        gpu_descriptor_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
        size_in_descriptors: u32,
    ) -> Result<DmlBindingTableDesc, DirectMlLoadError> {
        if size_in_descriptors == 0 {
            return Err(DirectMlLoadError::InvalidArgument {
                operation: "IDMLDevice::CreateBindingTable",
                reason: "descriptor-table size must be nonzero".to_owned(),
            });
        }
        Ok(DmlBindingTableDesc {
            dispatchable: dispatchable.raw(),
            cpu_descriptor_handle: cpu_descriptor_handle.ptr,
            gpu_descriptor_handle: gpu_descriptor_handle.ptr,
            size_in_descriptors,
        })
    }

    fn check_hresult(operation: &'static str, status: HResult) -> Result<(), DirectMlLoadError> {
        if status < 0 {
            Err(DirectMlLoadError::ComCall { operation, status })
        } else {
            Ok(())
        }
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
pub(crate) use com_wrappers::{
    DirectMlBinding, DirectMlBindingTable, DirectMlCommandRecorder, DirectMlDevice,
    DirectMlDispatchable,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectMlAbiProbe {
    pub target: String,
    pub source: DiscoverySource,
    pub windows_build: u32,
    pub file_version: FileVersion,
    pub digest_sha256: String,
    pub symbol_count: usize,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum DirectMlLoadError {
    #[error(
        "DirectML target is unsupported: {target}; expected aarch64-pc-windows-msvc or x86_64-pc-windows-msvc"
    )]
    UnsupportedTarget { target: String },
    #[error("DirectML ABI manifest is invalid: {0}")]
    Manifest(String),
    #[error("registry-certified DirectML library {library} is missing")]
    MissingCertifiedLibrary { library: String },
    #[error("registry-certified DirectML library {library} is duplicated")]
    DuplicateCertifiedLibrary { library: String },
    #[error("certified library {library} is outside the reviewed DirectML dependency set")]
    UnexpectedCertifiedLibrary { library: String },
    #[error("certificate identity, ABI, unsafe owner, or required symbols differ for {library}")]
    CertificateMismatch { library: String },
    #[error("certificate digest is not lowercase SHA-256 for {library}")]
    InvalidCertificateDigest { library: String },
    #[error("registry-certified retained module handle is null for {library}")]
    UncertifiedHandle { library: String },
    #[error("registry-certified DirectML image {library} cannot be loaded: {reason}")]
    ModuleLoad { library: String, reason: String },
    #[error(
        "registry-certified DirectML image loading is unsupported for {library} on this target"
    )]
    UnsupportedImageLoad { library: String },
    #[error("DirectML COM operation {operation} returned HRESULT {status:#x}")]
    ComCall {
        operation: &'static str,
        status: i32,
    },
    #[error("DirectML COM operation {operation} succeeded without returning an interface")]
    NullComResult { operation: &'static str },
    #[error("invalid DirectML argument for {operation}: {reason}")]
    InvalidArgument {
        operation: &'static str,
        reason: String,
    },
    #[error("required symbol {symbol} is missing from registry-certified {library}")]
    MissingSymbol { library: String, symbol: String },
    #[error("DirectML observation does not match a candidate in the deterministic discovery plan")]
    ObservationOutsidePlan,
    #[error("DirectML system component root differs from the operating-system system directory")]
    SystemDirectoryMismatch,
    #[error("DirectML operating-system observation failed while reading {fact}")]
    OperatingSystemObservation { fact: &'static str },
    #[error("DirectML candidate digest observation is not lowercase SHA-256")]
    InvalidObservationDigest,
    #[error("DirectML observation requires Windows build {minimum} or newer, got {actual}")]
    WindowsBuild { minimum: u32, actual: u32 },
    #[error("DirectML Authenticode observation is not trusted")]
    InvalidSignatureObservation,
    #[error("DirectML file version mismatch: expected {expected}, got {actual}")]
    VersionMismatch {
        expected: FileVersion,
        actual: FileVersion,
    },
    #[error("DirectML observation digest differs from the registry certificate")]
    ObservationDigestMismatch,
}

impl From<AbiManifestError> for DirectMlLoadError {
    fn from(error: AbiManifestError) -> Self {
        Self::Manifest(error.to_string())
    }
}

pub fn probe_certified(
    handles: &RetainedDirectMlLibraryHandles,
    plan: &DirectMlDiscoveryPlan,
    observation: &DirectMlCandidateObservation,
) -> Result<DirectMlAbiProbe, DirectMlLoadError> {
    let directml = handles.image(DIRECTML_LIBRARY_ID)?;
    validate_candidate_observation(plan, observation, &directml.digest_sha256)?;

    let manifest = AbiManifest::embedded()?;
    let symbol_count = probe_symbols(handles, &manifest)?;
    Ok(DirectMlAbiProbe {
        target: observation.target.clone(),
        source: observation.source,
        windows_build: observation.windows_build,
        file_version: observation.file_version,
        digest_sha256: observation.digest_sha256.clone(),
        symbol_count,
    })
}

pub fn validate_candidate_observation(
    plan: &DirectMlDiscoveryPlan,
    observation: &DirectMlCandidateObservation,
    certified_digest_sha256: &str,
) -> Result<(), DirectMlLoadError> {
    ensure_target(&observation.target)?;
    if observation.target != plan.target
        || !plan.candidates.iter().any(|candidate| {
            candidate.source == observation.source && candidate.path == observation.path
        })
    {
        return Err(DirectMlLoadError::ObservationOutsidePlan);
    }
    if observation.windows_build < MINIMUM_WINDOWS_BUILD {
        return Err(DirectMlLoadError::WindowsBuild {
            minimum: MINIMUM_WINDOWS_BUILD,
            actual: observation.windows_build,
        });
    }
    if observation.file_version != FILE_VERSION {
        return Err(DirectMlLoadError::VersionMismatch {
            expected: FILE_VERSION,
            actual: observation.file_version,
        });
    }
    if certified_digest_sha256 != observation.digest_sha256 {
        return Err(DirectMlLoadError::ObservationDigestMismatch);
    }
    Ok(())
}

fn validate_certificate_projection(
    image: &RegistryCertifiedDirectMlImage,
    library: &crate::abi::LibraryContract,
) -> Result<(), DirectMlLoadError> {
    let expected_symbols = library
        .symbols
        .iter()
        .map(|symbol| symbol.name.clone())
        .collect::<BTreeSet<_>>();
    if image.abi_version != library.abi_version
        || image.unsafe_owner != UNSAFE_OWNER
        || image.required_symbols != expected_symbols
    {
        return Err(DirectMlLoadError::CertificateMismatch {
            library: image.library_id.clone(),
        });
    }
    if !is_sha256(&image.digest_sha256) {
        return Err(DirectMlLoadError::InvalidCertificateDigest {
            library: image.library_id.clone(),
        });
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn ensure_target(target: &str) -> Result<(), DirectMlLoadError> {
    if matches!(target, "aarch64-pc-windows-msvc" | "x86_64-pc-windows-msvc") {
        Ok(())
    } else {
        Err(DirectMlLoadError::UnsupportedTarget {
            target: target.to_owned(),
        })
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn ensure_current_target(target: &str) -> Result<(), DirectMlLoadError> {
    let current = if cfg!(target_arch = "aarch64") {
        "aarch64-pc-windows-msvc"
    } else {
        "x86_64-pc-windows-msvc"
    };
    if target == current {
        Ok(())
    } else {
        Err(DirectMlLoadError::UnsupportedTarget {
            target: target.to_owned(),
        })
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn canonical_system_directory(target: &str) -> Result<PathBuf, DirectMlLoadError> {
    use std::os::windows::ffi::OsStringExt;

    ensure_current_target(target)?;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
    }

    let mut buffer = [0_u16; 32_768];
    let size =
        u32::try_from(buffer.len()).map_err(|_| DirectMlLoadError::OperatingSystemObservation {
            fact: "system directory",
        })?;
    let length = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), size) };
    let length =
        usize::try_from(length).map_err(|_| DirectMlLoadError::OperatingSystemObservation {
            fact: "system directory",
        })?;
    if length == 0 || length >= buffer.len() {
        return Err(DirectMlLoadError::OperatingSystemObservation {
            fact: "system directory",
        });
    }
    Ok(PathBuf::from(std::ffi::OsString::from_wide(
        &buffer[..length],
    )))
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn canonical_system_directory(target: &str) -> Result<PathBuf, DirectMlLoadError> {
    Err(DirectMlLoadError::UnsupportedTarget {
        target: target.to_owned(),
    })
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn observe_windows_build(target: &str) -> Result<u32, DirectMlLoadError> {
    #[repr(C)]
    struct RtlOsVersionInfo {
        _size: u32,
        _major: u32,
        _minor: u32,
        build: u32,
        _platform: u32,
        _service_pack: [u16; 128],
    }

    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn RtlGetVersion(version: *mut RtlOsVersionInfo) -> i32;
    }

    ensure_current_target(target)?;
    let mut version = RtlOsVersionInfo {
        _size: u32::try_from(std::mem::size_of::<RtlOsVersionInfo>()).map_err(|_| {
            DirectMlLoadError::OperatingSystemObservation {
                fact: "Windows build",
            }
        })?,
        _major: 0,
        _minor: 0,
        build: 0,
        _platform: 0,
        _service_pack: [0; 128],
    };
    if unsafe { RtlGetVersion(&mut version) } != 0 || version.build == 0 {
        return Err(DirectMlLoadError::OperatingSystemObservation {
            fact: "Windows build",
        });
    }
    Ok(version.build)
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn observe_windows_build(target: &str) -> Result<u32, DirectMlLoadError> {
    Err(DirectMlLoadError::UnsupportedTarget {
        target: target.to_owned(),
    })
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn observe_file_version(target: &str, path: &Path) -> Result<FileVersion, DirectMlLoadError> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{
        Win32::Storage::FileSystem::{
            GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FFI_SIGNATURE, VS_FIXEDFILEINFO,
            VerQueryValueW,
        },
        core::PCWSTR,
    };

    ensure_current_target(target)?;
    let wide_path = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let path = PCWSTR(wide_path.as_ptr());
    let byte_length = unsafe { GetFileVersionInfoSizeW(path, None) };
    if byte_length == 0 {
        return Err(DirectMlLoadError::OperatingSystemObservation {
            fact: "DirectML file version",
        });
    }
    let mut version_bytes = vec![
        0_u8;
        usize::try_from(byte_length).map_err(|_| {
            DirectMlLoadError::OperatingSystemObservation {
                fact: "DirectML file version",
            }
        })?
    ];
    unsafe { GetFileVersionInfoW(path, None, byte_length, version_bytes.as_mut_ptr().cast()) }
        .map_err(|_| DirectMlLoadError::OperatingSystemObservation {
            fact: "DirectML file version",
        })?;
    let mut fixed = std::ptr::null_mut::<c_void>();
    let mut fixed_length = 0_u32;
    if !unsafe {
        VerQueryValueW(
            version_bytes.as_ptr().cast(),
            windows::core::w!("\\"),
            &mut fixed,
            &mut fixed_length,
        )
    }
    .as_bool()
        || fixed.is_null()
        || !usize::try_from(fixed_length)
            .is_ok_and(|length| length >= std::mem::size_of::<VS_FIXEDFILEINFO>())
    {
        return Err(DirectMlLoadError::OperatingSystemObservation {
            fact: "DirectML file version",
        });
    }
    let fixed = unsafe { &*fixed.cast::<VS_FIXEDFILEINFO>() };
    if fixed.dwSignature != VS_FFI_SIGNATURE as u32 {
        return Err(DirectMlLoadError::OperatingSystemObservation {
            fact: "DirectML file version",
        });
    }
    Ok(FileVersion::new(
        (fixed.dwFileVersionMS >> 16) as u16,
        fixed.dwFileVersionMS as u16,
        (fixed.dwFileVersionLS >> 16) as u16,
        fixed.dwFileVersionLS as u16,
    ))
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn observe_file_version(target: &str, _path: &Path) -> Result<FileVersion, DirectMlLoadError> {
    Err(DirectMlLoadError::UnsupportedTarget {
        target: target.to_owned(),
    })
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn verify_authenticode_offline(target: &str, path: &Path) -> Result<(), DirectMlLoadError> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{
        Win32::{
            Foundation::HWND,
            Security::WinTrust::{
                WINTRUST_ACTION_GENERIC_VERIFY_V2, WINTRUST_DATA, WINTRUST_DATA_0,
                WINTRUST_FILE_INFO, WTD_CACHE_ONLY_URL_RETRIEVAL, WTD_CHOICE_FILE,
                WTD_REVOCATION_CHECK_NONE, WTD_REVOKE_NONE, WTD_STATEACTION_CLOSE,
                WTD_STATEACTION_VERIFY, WTD_UI_NONE, WinVerifyTrust,
            },
        },
        core::PCWSTR,
    };

    ensure_current_target(target)?;
    let wide_path = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let mut file = WINTRUST_FILE_INFO {
        cbStruct: u32::try_from(std::mem::size_of::<WINTRUST_FILE_INFO>()).map_err(|_| {
            DirectMlLoadError::OperatingSystemObservation {
                fact: "DirectML Authenticode",
            }
        })?,
        pcwszFilePath: PCWSTR(wide_path.as_ptr()),
        ..Default::default()
    };
    let mut trust = WINTRUST_DATA {
        cbStruct: u32::try_from(std::mem::size_of::<WINTRUST_DATA>()).map_err(|_| {
            DirectMlLoadError::OperatingSystemObservation {
                fact: "DirectML Authenticode",
            }
        })?,
        dwUIChoice: WTD_UI_NONE,
        fdwRevocationChecks: WTD_REVOKE_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 { pFile: &mut file },
        dwStateAction: WTD_STATEACTION_VERIFY,
        dwProvFlags: WTD_CACHE_ONLY_URL_RETRIEVAL | WTD_REVOCATION_CHECK_NONE,
        ..Default::default()
    };
    let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    let status = unsafe {
        WinVerifyTrust(
            HWND::default(),
            &mut action,
            (&mut trust as *mut WINTRUST_DATA).cast(),
        )
    };
    trust.dwStateAction = WTD_STATEACTION_CLOSE;
    let close_status = unsafe {
        WinVerifyTrust(
            HWND::default(),
            &mut action,
            (&mut trust as *mut WINTRUST_DATA).cast(),
        )
    };
    if close_status != 0 {
        return Err(DirectMlLoadError::OperatingSystemObservation {
            fact: "DirectML Authenticode state close",
        });
    }
    if status != 0 {
        return Err(DirectMlLoadError::InvalidSignatureObservation);
    }
    Ok(())
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn verify_authenticode_offline(target: &str, _path: &Path) -> Result<(), DirectMlLoadError> {
    Err(DirectMlLoadError::UnsupportedTarget {
        target: target.to_owned(),
    })
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn windows_paths_equal(left: &Path, right: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;

    fn normalized(path: &Path) -> Vec<u16> {
        const ASCII_SLASH: u16 = b'/' as u16;
        const ASCII_BACKSLASH: u16 = b'\\' as u16;
        const ASCII_UPPER_A: u16 = b'A' as u16;
        const ASCII_UPPER_Z: u16 = b'Z' as u16;
        let mut units = path
            .as_os_str()
            .encode_wide()
            .map(|unit| match unit {
                ASCII_SLASH => ASCII_BACKSLASH,
                ASCII_UPPER_A..=ASCII_UPPER_Z => unit + u16::from(b'a' - b'A'),
                _ => unit,
            })
            .collect::<Vec<_>>();
        while units.last() == Some(&ASCII_BACKSLASH) {
            units.pop();
        }
        units
    }

    normalized(left) == normalized(right)
}

#[cfg(not(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
fn windows_paths_equal(left: &Path, right: &Path) -> bool {
    left == right
}

trait RetainedSymbolResolver {
    fn require_symbol(&mut self, library: &str, symbol: &str) -> Result<(), DirectMlLoadError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResolvedContract {
    symbol_count: usize,
}

fn resolve_with<R: RetainedSymbolResolver>(
    resolver: &mut R,
    manifest: &AbiManifest,
) -> Result<ResolvedContract, DirectMlLoadError> {
    let mut symbol_count = 0;
    for library in &manifest.libraries {
        for symbol in &library.symbols {
            resolver.require_symbol(&library.name, &symbol.name)?;
            symbol_count += 1;
        }
    }
    Ok(ResolvedContract { symbol_count })
}

fn probe_symbols(
    handles: &RetainedDirectMlLibraryHandles,
    manifest: &AbiManifest,
) -> Result<usize, DirectMlLoadError> {
    #[cfg(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    {
        let mut resolver = WindowsRetainedResolver { handles };
        Ok(resolve_with(&mut resolver, manifest)?.symbol_count)
    }
    #[cfg(not(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    {
        let _ = (handles, manifest);
        Err(DirectMlLoadError::UnsupportedTarget {
            target: env!("SIM_DIRECTML_TARGET").to_owned(),
        })
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
struct WindowsRetainedResolver<'a> {
    handles: &'a RetainedDirectMlLibraryHandles,
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
impl RetainedSymbolResolver for WindowsRetainedResolver<'_> {
    fn require_symbol(&mut self, library: &str, symbol: &str) -> Result<(), DirectMlLoadError> {
        required_address(self.handles.image(library)?, library, symbol).map(|_| ())
    }
}

#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
fn required_address(
    image: &CertifiedImage,
    library: &str,
    symbol: &str,
) -> Result<NonNull<c_void>, DirectMlLoadError> {
    use windows::{
        Win32::{Foundation::HMODULE, System::LibraryLoader::GetProcAddress},
        core::PCSTR,
    };

    let mut name = symbol.as_bytes().to_vec();
    name.push(0);
    let address = unsafe {
        GetProcAddress(
            HMODULE(image.module.handle.as_ptr()),
            PCSTR::from_raw(name.as_ptr()),
        )
    };
    address
        .and_then(|function| NonNull::new(function as *mut c_void))
        .ok_or_else(|| DirectMlLoadError::MissingSymbol {
            library: library.to_owned(),
            symbol: symbol.to_owned(),
        })
}

pub fn unavailable_reason() -> String {
    if matches!(
        env!("SIM_DIRECTML_TARGET"),
        "aarch64-pc-windows-msvc" | "x86_64-pc-windows-msvc"
    ) {
        format!(
            "DirectML {ABI_FLOOR} ABI foundation is present, but sim_runtime::NativeFfiRegistry has not supplied certified retained D3D12.dll, DXGI.dll, and DirectML.dll module handles"
        )
    } else {
        format!(
            "DirectML unsupported target {}; expected aarch64-pc-windows-msvc or x86_64-pc-windows-msvc and sim_runtime::NativeFfiRegistry-certified retained handles",
            env!("SIM_DIRECTML_TARGET")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct FakeResolver {
        missing: Option<(&'static str, &'static str)>,
        observed: Vec<(String, String)>,
    }

    impl RetainedSymbolResolver for FakeResolver {
        fn require_symbol(&mut self, library: &str, symbol: &str) -> Result<(), DirectMlLoadError> {
            self.observed.push((library.to_owned(), symbol.to_owned()));
            if self.missing == Some((library, symbol)) {
                Err(DirectMlLoadError::MissingSymbol {
                    library: library.to_owned(),
                    symbol: symbol.to_owned(),
                })
            } else {
                Ok(())
            }
        }
    }

    fn manifest() -> AbiManifest {
        AbiManifest::embedded().expect("embedded manifest")
    }

    fn certified_image(library: &str, handle: usize) -> RegistryCertifiedDirectMlImage {
        let manifest = manifest();
        let library_contract = manifest
            .libraries
            .iter()
            .find(|candidate| candidate.name == library)
            .expect("library row");
        unsafe {
            RegistryCertifiedDirectMlImage::from_test_registry_certificate(
                library,
                if library == DIRECTML_LIBRARY_ID {
                    "5ab77cc5db8e1544d386fd28586598317da8dcbef098fb86d8d8a60e739e0e5d".to_owned()
                } else {
                    "1111111111111111111111111111111111111111111111111111111111111111".to_owned()
                },
                library_contract.abi_version.clone(),
                library_contract
                    .symbols
                    .iter()
                    .map(|symbol| symbol.name.clone())
                    .collect(),
                UNSAFE_OWNER,
                handle as *mut c_void,
            )
        }
        .expect("registry image fixture")
    }

    fn observed_certified_image(
        library: &str,
        handle: usize,
        drops: &Arc<AtomicUsize>,
    ) -> RegistryCertifiedDirectMlImage {
        let mut image = certified_image(library, handle);
        image.module.drop_observer = Some(drops.clone());
        image
    }

    fn retained_handles() -> RetainedDirectMlLibraryHandles {
        unsafe {
            RetainedDirectMlLibraryHandles::from_registry_certificates(
                Arc::new(()),
                [
                    certified_image(D3D12_LIBRARY_ID, 1),
                    certified_image(DIRECTML_LIBRARY_ID, 2),
                    certified_image(DXGI_LIBRARY_ID, 3),
                ],
            )
        }
        .expect("fixture certificates")
    }

    #[test]
    fn discovery_order_is_application_package_then_system_component() {
        let plan = DirectMlDiscoveryPlan::for_target(
            "x86_64-pc-windows-msvc",
            "C:/Zed",
            "C:/Windows/System32",
        )
        .expect("supported target");
        assert_eq!(
            plan.candidates()[0].source(),
            DiscoverySource::SignedApplicationPackage
        );
        assert_eq!(
            plan.candidates()[1].source(),
            DiscoverySource::CompatibleSystemComponent
        );
        assert_eq!(
            plan.candidates()[0].path(),
            Path::new("C:/Zed/DirectML.dll")
        );
    }

    #[test]
    fn certificate_projection_requires_exact_complete_non_null_handles() {
        let handles = retained_handles();
        assert_eq!(handles.retained_library_count(), 3);

        let missing = unsafe {
            RetainedDirectMlLibraryHandles::from_registry_certificates(
                Arc::new(()),
                [
                    certified_image(D3D12_LIBRARY_ID, 1),
                    certified_image(DIRECTML_LIBRARY_ID, 2),
                ],
            )
        };
        assert!(matches!(
            missing,
            Err(DirectMlLoadError::MissingCertifiedLibrary { library })
                if library == DXGI_LIBRARY_ID
        ));

        let null = unsafe {
            RegistryCertifiedDirectMlImage::from_test_registry_certificate(
                DIRECTML_LIBRARY_ID,
                "5ab77cc5db8e1544d386fd28586598317da8dcbef098fb86d8d8a60e739e0e5d",
                ABI_FLOOR,
                BTreeSet::from(["DMLCreateDevice".to_owned(), "DMLCreateDevice1".to_owned()]),
                UNSAFE_OWNER,
                std::ptr::null_mut(),
            )
        };
        assert!(matches!(
            null,
            Err(DirectMlLoadError::UncertifiedHandle { library })
                if library == DIRECTML_LIBRARY_ID
        ));
    }

    #[test]
    fn owned_modules_and_certification_are_released_on_success_and_partial_failure() {
        struct DropMarker(Arc<AtomicBool>);

        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let successful_module_drops = Arc::new(AtomicUsize::new(0));
        let successful_retention_dropped = Arc::new(AtomicBool::new(false));
        let handles = unsafe {
            RetainedDirectMlLibraryHandles::from_registry_certificates(
                Arc::new(DropMarker(successful_retention_dropped.clone())),
                [
                    observed_certified_image(D3D12_LIBRARY_ID, 1, &successful_module_drops),
                    observed_certified_image(DIRECTML_LIBRARY_ID, 2, &successful_module_drops),
                    observed_certified_image(DXGI_LIBRARY_ID, 3, &successful_module_drops),
                ],
            )
        }
        .expect("complete fixture certificates");
        assert_eq!(successful_module_drops.load(Ordering::Acquire), 0);
        assert!(!successful_retention_dropped.load(Ordering::Acquire));
        drop(handles);
        assert_eq!(successful_module_drops.load(Ordering::Acquire), 3);
        assert!(successful_retention_dropped.load(Ordering::Acquire));

        let failed_module_drops = Arc::new(AtomicUsize::new(0));
        let failed_retention_dropped = Arc::new(AtomicBool::new(false));
        let mut mismatched = observed_certified_image(DIRECTML_LIBRARY_ID, 2, &failed_module_drops);
        mismatched.unsafe_owner = "sim_backend_directml::manifest".to_owned();
        let result = unsafe {
            RetainedDirectMlLibraryHandles::from_registry_certificates(
                Arc::new(DropMarker(failed_retention_dropped.clone())),
                [
                    observed_certified_image(D3D12_LIBRARY_ID, 1, &failed_module_drops),
                    mismatched,
                    observed_certified_image(DXGI_LIBRARY_ID, 3, &failed_module_drops),
                ],
            )
        };
        assert!(matches!(
            result,
            Err(DirectMlLoadError::CertificateMismatch { library })
                if library == DIRECTML_LIBRARY_ID
        ));
        assert_eq!(failed_module_drops.load(Ordering::Acquire), 3);
        assert!(failed_retention_dropped.load(Ordering::Acquire));
    }

    #[test]
    fn certificate_cannot_be_minted_from_manifest_path_or_feature_state() {
        let mut wrong = certified_image(DIRECTML_LIBRARY_ID, 2);
        wrong.unsafe_owner = "sim_backend_directml::manifest".to_owned();
        let result = unsafe {
            RetainedDirectMlLibraryHandles::from_registry_certificates(
                Arc::new(()),
                [
                    certified_image(D3D12_LIBRARY_ID, 1),
                    wrong,
                    certified_image(DXGI_LIBRARY_ID, 3),
                ],
            )
        };
        assert!(matches!(
            result,
            Err(DirectMlLoadError::CertificateMismatch { library })
                if library == DIRECTML_LIBRARY_ID
        ));

        let mut extra_symbol = certified_image(DIRECTML_LIBRARY_ID, 2);
        extra_symbol
            .required_symbols
            .insert("ManifestOnlySymbol".to_owned());
        let result = unsafe {
            RetainedDirectMlLibraryHandles::from_registry_certificates(
                Arc::new(()),
                [
                    certified_image(D3D12_LIBRARY_ID, 1),
                    extra_symbol,
                    certified_image(DXGI_LIBRARY_ID, 3),
                ],
            )
        };
        assert!(matches!(
            result,
            Err(DirectMlLoadError::CertificateMismatch { library })
                if library == DIRECTML_LIBRARY_ID
        ));
    }

    #[test]
    fn exact_symbol_set_is_resolved_only_from_retained_handles() {
        let mut resolver = FakeResolver {
            missing: None,
            observed: Vec::new(),
        };
        let resolved = resolve_with(&mut resolver, &manifest()).expect("symbols resolve");
        assert_eq!(resolved.symbol_count, 4);
        assert_eq!(
            resolver.observed,
            [
                (D3D12_LIBRARY_ID.to_owned(), "D3D12CreateDevice".to_owned()),
                (DIRECTML_LIBRARY_ID.to_owned(), "DMLCreateDevice".to_owned()),
                (
                    DIRECTML_LIBRARY_ID.to_owned(),
                    "DMLCreateDevice1".to_owned()
                ),
                (DXGI_LIBRARY_ID.to_owned(), "CreateDXGIFactory2".to_owned()),
            ]
        );
    }

    #[test]
    fn missing_symbol_is_typed() {
        let mut resolver = FakeResolver {
            missing: Some((DIRECTML_LIBRARY_ID, "DMLCreateDevice1")),
            observed: Vec::new(),
        };
        assert!(matches!(
            resolve_with(&mut resolver, &manifest()),
            Err(DirectMlLoadError::MissingSymbol { library, symbol })
                if library == DIRECTML_LIBRARY_ID && symbol == "DMLCreateDevice1"
        ));
    }

    #[test]
    fn observation_is_only_accepted_beside_matching_registry_certificate() {
        let handles = retained_handles();
        let plan = DirectMlDiscoveryPlan::for_target(
            "x86_64-pc-windows-msvc",
            "C:/Zed",
            "C:/Windows/System32",
        )
        .expect("plan");
        let mut observation = DirectMlCandidateObservation {
            target: "x86_64-pc-windows-msvc".to_owned(),
            source: DiscoverySource::SignedApplicationPackage,
            path: PathBuf::from("C:/Zed/DirectML.dll"),
            windows_build: MINIMUM_WINDOWS_BUILD,
            file_version: FILE_VERSION,
            digest_sha256: "5ab77cc5db8e1544d386fd28586598317da8dcbef098fb86d8d8a60e739e0e5d"
                .to_owned(),
        };

        #[cfg(not(all(
            target_os = "windows",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )))]
        assert!(matches!(
            probe_certified(&handles, &plan, &observation),
            Err(DirectMlLoadError::UnsupportedTarget { .. })
        ));

        observation.file_version = FileVersion::new(1, 12, 0, 0);
        assert!(matches!(
            probe_certified(&handles, &plan, &observation),
            Err(DirectMlLoadError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn unsupported_target_is_explicit() {
        assert!(matches!(
            DirectMlDiscoveryPlan::for_target(
                "aarch64-apple-darwin",
                "/Applications/Zed.app",
                "/System32"
            ),
            Err(DirectMlLoadError::UnsupportedTarget { .. })
        ));
    }

    #[cfg(not(all(
        target_os = "windows",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    #[test]
    fn production_observation_has_no_synthetic_non_windows_success() {
        let plan = DirectMlDiscoveryPlan::for_target(
            "x86_64-pc-windows-msvc",
            "C:/Zed",
            "C:/Windows/System32",
        )
        .expect("supported catalog target");
        assert!(matches!(
            observe_directml_candidate(
                &plan,
                "5ab77cc5db8e1544d386fd28586598317da8dcbef098fb86d8d8a60e739e0e5d"
            ),
            Err(DirectMlLoadError::UnsupportedTarget { .. })
        ));
    }
}
