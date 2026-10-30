use crate::abi::AbiManifest;
#[cfg(any(
    test,
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
use crate::abi::{ClassContract, FrameworkContract, LayoutContract, SelectorContract};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalAbiProbe {
    pub target: String,
    pub framework_count: usize,
    pub symbol_count: usize,
    pub class_count: usize,
    pub selector_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceProbe {
    pub name: String,
    pub registry_id: u64,
    pub recommended_working_set_bytes: u64,
    pub unified_memory: bool,
    pub metal_3: bool,
    pub mps_supported: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MetalLoadError {
    #[error("Metal target is unsupported: {target}")]
    UnsupportedTarget { target: String },
    #[error("failed to open fixed Apple framework {framework}: {reason}")]
    FrameworkOpen { framework: String, reason: String },
    #[error("required symbol {symbol} is missing from {framework}")]
    MissingSymbol { framework: String, symbol: String },
    #[error("symbol {symbol} resolved from unexpected image {actual}; expected {expected}")]
    WrongSymbolImage {
        symbol: String,
        expected: String,
        actual: String,
    },
    #[error("required Objective-C class {class} is missing")]
    MissingClass { class: String },
    #[error(
        "Objective-C class {class} resolved from unexpected image {actual}; expected {expected}"
    )]
    WrongClassImage {
        class: String,
        expected: String,
        actual: String,
    },
    #[error("required selector {class} {selector} is missing")]
    MissingSelector { class: String, selector: String },
    #[error("selector encoding mismatch for {class} {selector}: expected {expected}, got {actual}")]
    SelectorEncoding {
        class: String,
        selector: String,
        expected: String,
        actual: String,
    },
    #[error(
        "reviewed layout mismatch for {name}: expected size {expected_size}/align {expected_align}, got {actual_size}/{actual_align}"
    )]
    Layout {
        name: String,
        expected_size: usize,
        expected_align: usize,
        actual_size: usize,
        actual_align: usize,
    },
    #[error("MTLCreateSystemDefaultDevice returned no Metal device")]
    NoSystemDevice,
    #[error("the selected Metal device does not support the Metal 3 family")]
    MissingMetal3,
    #[error("MPSSupportsMTLDevice rejected the selected Metal device")]
    MpsUnsupported,
    #[error("Metal ABI manifest is invalid: {0}")]
    Manifest(String),
}

#[cfg(any(
    test,
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
trait MetalProbeSystem {
    type FrameworkHandle;
    type ClassHandle;
    type Device;

    fn target(&mut self) -> String;
    fn open_framework(
        &mut self,
        framework: &FrameworkContract,
    ) -> Result<Self::FrameworkHandle, MetalLoadError>;
    fn check_symbol(
        &mut self,
        handle: &Self::FrameworkHandle,
        framework: &FrameworkContract,
        symbol: &str,
    ) -> Result<(), MetalLoadError>;
    fn check_class(
        &mut self,
        contract: &ClassContract,
    ) -> Result<Self::ClassHandle, MetalLoadError>;
    fn check_selector(
        &mut self,
        class: &Self::ClassHandle,
        contract: &ClassContract,
        selector: &SelectorContract,
    ) -> Result<(), MetalLoadError>;
    fn check_layout(&mut self, layout: &LayoutContract) -> Result<(), MetalLoadError>;
    fn system_device(&mut self) -> Result<Self::Device, MetalLoadError>;
    fn require_metal_3(&mut self, device: &Self::Device) -> Result<(), MetalLoadError>;
    fn require_mps(
        &mut self,
        manifest: &AbiManifest,
        device: &Self::Device,
    ) -> Result<(), MetalLoadError>;
    fn project_device(&self, device: &Self::Device) -> MetalDeviceProbe;
}

#[cfg(any(
    test,
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn probe_abi_with_system<S: MetalProbeSystem>(
    system: &mut S,
    manifest: &AbiManifest,
) -> Result<MetalAbiProbe, MetalLoadError> {
    let target = system.target();
    if !manifest
        .targets
        .iter()
        .any(|candidate| candidate == &target)
    {
        return Err(MetalLoadError::UnsupportedTarget { target });
    }

    let mut handles = Vec::with_capacity(manifest.frameworks.len());
    for framework in &manifest.frameworks {
        handles.push(system.open_framework(framework)?);
    }

    let mut symbol_count = 0;
    for (framework, handle) in manifest.frameworks.iter().zip(handles.iter()) {
        for symbol in &framework.symbols {
            system.check_symbol(handle, framework, symbol)?;
            symbol_count += 1;
        }
    }

    let mut selector_count = 0;
    for contract in &manifest.classes {
        let class = system.check_class(contract)?;
        for selector in &contract.selectors {
            system.check_selector(&class, contract, selector)?;
            selector_count += 1;
        }
    }

    for layout in &manifest.layouts {
        system.check_layout(layout)?;
    }

    Ok(MetalAbiProbe {
        target,
        framework_count: handles.len(),
        symbol_count,
        class_count: manifest.classes.len(),
        selector_count,
    })
}

#[cfg(any(
    test,
    all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )
))]
fn probe_device_with_system<S: MetalProbeSystem>(
    system: &mut S,
    manifest: &AbiManifest,
) -> Result<MetalDeviceProbe, MetalLoadError> {
    probe_abi_with_system(system, manifest)?;
    let device = system.system_device()?;
    system.require_metal_3(&device)?;
    system.require_mps(manifest, &device)?;
    Ok(system.project_device(&device))
}

pub fn probe_abi() -> Result<MetalAbiProbe, MetalLoadError> {
    let manifest =
        AbiManifest::embedded().map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
    platform::probe_abi(&manifest)
}

pub fn probe_device() -> Result<MetalDeviceProbe, MetalLoadError> {
    let manifest =
        AbiManifest::embedded().map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
    platform::probe_device(&manifest)
}

#[cfg(not(all(
    target_os = "macos",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod platform {
    use super::*;

    pub fn probe_abi(_manifest: &AbiManifest) -> Result<MetalAbiProbe, MetalLoadError> {
        Err(MetalLoadError::UnsupportedTarget {
            target: env!("SIM_METAL_TARGET").to_owned(),
        })
    }

    pub fn probe_device(_manifest: &AbiManifest) -> Result<MetalDeviceProbe, MetalLoadError> {
        Err(MetalLoadError::UnsupportedTarget {
            target: env!("SIM_METAL_TARGET").to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ProbePoint {
        Target,
        Framework,
        Symbol,
        Class,
        Selector,
        Layout,
        Device,
        Metal3,
        Mps,
    }

    struct FakeProbeSystem {
        target: String,
        failure: Option<(ProbePoint, MetalLoadError)>,
        trace: Vec<ProbePoint>,
    }

    impl FakeProbeSystem {
        fn passing() -> Self {
            Self {
                target: "aarch64-apple-darwin".to_owned(),
                failure: None,
                trace: Vec::new(),
            }
        }

        fn failing(point: ProbePoint, error: MetalLoadError) -> Self {
            Self {
                failure: Some((point, error)),
                ..Self::passing()
            }
        }

        fn visit(&mut self, point: ProbePoint) -> Result<(), MetalLoadError> {
            self.trace.push(point);
            if self
                .failure
                .as_ref()
                .is_some_and(|(failure_point, _)| *failure_point == point)
                && let Some((_, error)) = self.failure.take()
            {
                return Err(error);
            }
            Ok(())
        }
    }

    impl MetalProbeSystem for FakeProbeSystem {
        type FrameworkHandle = String;
        type ClassHandle = String;
        type Device = ();

        fn target(&mut self) -> String {
            self.trace.push(ProbePoint::Target);
            self.target.clone()
        }

        fn open_framework(
            &mut self,
            framework: &FrameworkContract,
        ) -> Result<Self::FrameworkHandle, MetalLoadError> {
            self.visit(ProbePoint::Framework)?;
            Ok(framework.name.clone())
        }

        fn check_symbol(
            &mut self,
            _handle: &Self::FrameworkHandle,
            _framework: &FrameworkContract,
            _symbol: &str,
        ) -> Result<(), MetalLoadError> {
            self.visit(ProbePoint::Symbol)
        }

        fn check_class(
            &mut self,
            contract: &ClassContract,
        ) -> Result<Self::ClassHandle, MetalLoadError> {
            self.visit(ProbePoint::Class)?;
            Ok(contract.name.clone())
        }

        fn check_selector(
            &mut self,
            _class: &Self::ClassHandle,
            _contract: &ClassContract,
            _selector: &SelectorContract,
        ) -> Result<(), MetalLoadError> {
            self.visit(ProbePoint::Selector)
        }

        fn check_layout(&mut self, _layout: &LayoutContract) -> Result<(), MetalLoadError> {
            self.visit(ProbePoint::Layout)
        }

        fn system_device(&mut self) -> Result<Self::Device, MetalLoadError> {
            self.visit(ProbePoint::Device)
        }

        fn require_metal_3(&mut self, _device: &Self::Device) -> Result<(), MetalLoadError> {
            self.visit(ProbePoint::Metal3)
        }

        fn require_mps(
            &mut self,
            _manifest: &AbiManifest,
            _device: &Self::Device,
        ) -> Result<(), MetalLoadError> {
            self.visit(ProbePoint::Mps)
        }

        fn project_device(&self, _device: &Self::Device) -> MetalDeviceProbe {
            MetalDeviceProbe {
                name: "fixture".to_owned(),
                registry_id: 1,
                recommended_working_set_bytes: 2,
                unified_memory: true,
                metal_3: true,
                mps_supported: true,
            }
        }
    }

    fn embedded_manifest() -> AbiManifest {
        match AbiManifest::embedded() {
            Ok(manifest) => manifest,
            Err(error) => panic!("embedded Metal ABI manifest must remain valid: {error}"),
        }
    }

    #[test]
    fn probe_order_is_target_framework_abi_layout_then_device_capabilities() {
        let manifest = embedded_manifest();
        let mut system = FakeProbeSystem::passing();
        let result = probe_device_with_system(&mut system, &manifest);
        assert!(result.is_ok());

        let mut expected = vec![ProbePoint::Target];
        expected.extend([ProbePoint::Framework; 3]);
        expected.extend([ProbePoint::Symbol; 2]);
        expected.push(ProbePoint::Class);
        expected.extend([ProbePoint::Selector; 8]);
        expected.push(ProbePoint::Class);
        expected.extend([ProbePoint::Selector; 2]);
        expected.push(ProbePoint::Class);
        expected.extend([ProbePoint::Selector; 2]);
        expected.extend([ProbePoint::Layout; 5]);
        expected.extend([ProbePoint::Device, ProbePoint::Metal3, ProbePoint::Mps]);
        assert_eq!(system.trace, expected);
    }

    #[test]
    fn unsupported_target_fails_before_framework_access() {
        let manifest = embedded_manifest();
        let mut system = FakeProbeSystem {
            target: "x86_64-unknown-linux-gnu".to_owned(),
            ..FakeProbeSystem::passing()
        };
        assert_eq!(
            probe_abi_with_system(&mut system, &manifest),
            Err(MetalLoadError::UnsupportedTarget {
                target: "x86_64-unknown-linux-gnu".to_owned()
            })
        );
        assert_eq!(system.trace, [ProbePoint::Target]);
    }

    #[test]
    fn abi_probe_propagates_exact_fail_closed_errors_at_first_failed_stage() {
        let manifest = embedded_manifest();
        let cases = [
            (
                ProbePoint::Framework,
                MetalLoadError::FrameworkOpen {
                    framework: "Metal".to_owned(),
                    reason: "fixture open denial".to_owned(),
                },
            ),
            (
                ProbePoint::Symbol,
                MetalLoadError::MissingSymbol {
                    framework: "Metal".to_owned(),
                    symbol: "MTLCreateSystemDefaultDevice".to_owned(),
                },
            ),
            (
                ProbePoint::Symbol,
                MetalLoadError::WrongSymbolImage {
                    symbol: "MTLCreateSystemDefaultDevice".to_owned(),
                    expected: "/System/Library/Frameworks/Metal.framework/Versions/A/Metal"
                        .to_owned(),
                    actual: "/tmp/injected/Metal".to_owned(),
                },
            ),
            (
                ProbePoint::Class,
                MetalLoadError::MissingClass {
                    class: "MPSGraph".to_owned(),
                },
            ),
            (
                ProbePoint::Class,
                MetalLoadError::WrongClassImage {
                    class: "MPSGraph".to_owned(),
                    expected: "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/MetalPerformanceShadersGraph".to_owned(),
                    actual: "/tmp/injected/MPSGraph".to_owned(),
                },
            ),
            (
                ProbePoint::Selector,
                MetalLoadError::MissingSelector {
                    class: "MPSGraph".to_owned(),
                    selector: "new".to_owned(),
                },
            ),
            (
                ProbePoint::Selector,
                MetalLoadError::SelectorEncoding {
                    class: "MPSGraph".to_owned(),
                    selector: "new".to_owned(),
                    expected: "@16@0:8".to_owned(),
                    actual: "v16@0:8".to_owned(),
                },
            ),
            (
                ProbePoint::Layout,
                MetalLoadError::Layout {
                    name: "MPSDataType".to_owned(),
                    expected_size: 4,
                    expected_align: 4,
                    actual_size: 8,
                    actual_align: 8,
                },
            ),
        ];

        for (point, expected) in cases {
            let mut system = FakeProbeSystem::failing(point, expected.clone());
            let actual = probe_abi_with_system(&mut system, &manifest);
            let injected = system.failure.take();
            assert!(injected.is_none(), "injected failure was not reached");
            assert_eq!(actual, Err(expected));
            assert_eq!(system.trace.last(), Some(&point));
        }
    }

    #[test]
    fn device_probe_propagates_exact_capability_failures_without_running_later_stages() {
        let manifest = embedded_manifest();
        let cases = [
            (ProbePoint::Device, MetalLoadError::NoSystemDevice),
            (ProbePoint::Metal3, MetalLoadError::MissingMetal3),
            (ProbePoint::Mps, MetalLoadError::MpsUnsupported),
        ];

        for (point, expected) in cases {
            let mut system = FakeProbeSystem::failing(point, expected.clone());
            assert_eq!(
                probe_device_with_system(&mut system, &manifest),
                Err(expected)
            );
            assert_eq!(system.trace.last(), Some(&point));
            assert!(system.failure.is_none(), "injected failure was not reached");
        }
    }
}

#[cfg(all(
    target_os = "macos",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod platform {
    use super::*;
    use crate::abi::SelectorKind;
    use metal::{Device, MTLGPUFamily, foreign_types::ForeignType};
    use std::{
        ffi::{CStr, CString, c_char, c_int, c_void},
        mem::{align_of, size_of},
        ptr::{self, NonNull},
        rc::Rc,
    };

    const RTLD_LOCAL: c_int = 0x4;
    const RTLD_NOW: c_int = 0x2;
    const RTLD_FIRST: c_int = 0x100;

    #[repr(C)]
    struct DlInfo {
        image_name: *const c_char,
        image_base: *mut c_void,
        symbol_name: *const c_char,
        symbol_address: *mut c_void,
    }

    enum ObjcClass {}
    enum ObjcMethod {}
    enum ObjcSelector {}

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
        fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
        fn objc_getClass(name: *const c_char) -> *mut ObjcClass;
        fn sel_registerName(name: *const c_char) -> *mut ObjcSelector;
        fn class_getClassMethod(
            class: *const ObjcClass,
            selector: *const ObjcSelector,
        ) -> *mut ObjcMethod;
        fn class_getInstanceMethod(
            class: *const ObjcClass,
            selector: *const ObjcSelector,
        ) -> *mut ObjcMethod;
        fn class_getImageName(class: *const ObjcClass) -> *const c_char;
        fn method_getTypeEncoding(method: *const ObjcMethod) -> *const c_char;
    }

    struct FrameworkHandle {
        name: String,
        handle: NonNull<c_void>,
    }

    impl Drop for FrameworkHandle {
        fn drop(&mut self) {
            let result = unsafe { dlclose(self.handle.as_ptr()) };
            if result != 0 {
                eprintln!("failed to release Metal framework handle {}", self.name);
            }
        }
    }

    type CreateSystemDevice = unsafe extern "C" fn() -> *mut metal::MTLDevice;

    #[derive(Default)]
    struct AppleProbeSystem {
        device_factory: Option<(CreateSystemDevice, Rc<FrameworkHandle>)>,
    }

    impl MetalProbeSystem for AppleProbeSystem {
        type FrameworkHandle = Rc<FrameworkHandle>;
        type ClassHandle = NonNull<ObjcClass>;
        type Device = Device;

        fn target(&mut self) -> String {
            env!("SIM_METAL_TARGET").to_owned()
        }

        fn open_framework(
            &mut self,
            framework: &FrameworkContract,
        ) -> Result<Self::FrameworkHandle, MetalLoadError> {
            let path = CString::new(framework.install_name.as_str())
                .map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
            let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL | RTLD_FIRST) };
            NonNull::new(handle)
                .map(|handle| {
                    Rc::new(FrameworkHandle {
                        name: framework.name.clone(),
                        handle,
                    })
                })
                .ok_or_else(|| MetalLoadError::FrameworkOpen {
                    framework: framework.name.clone(),
                    reason: dl_error(),
                })
        }

        fn check_symbol(
            &mut self,
            handle: &Self::FrameworkHandle,
            framework: &FrameworkContract,
            symbol: &str,
        ) -> Result<(), MetalLoadError> {
            let address = checked_symbol(handle, framework.image_name.as_str(), symbol)?;
            if framework.name == "Metal" && symbol == "MTLCreateSystemDefaultDevice" {
                let function: CreateSystemDevice = unsafe { std::mem::transmute(address.as_ptr()) };
                self.device_factory = Some((function, handle.clone()));
            }
            Ok(())
        }

        fn check_class(
            &mut self,
            contract: &ClassContract,
        ) -> Result<Self::ClassHandle, MetalLoadError> {
            checked_class(contract)
        }

        fn check_selector(
            &mut self,
            class: &Self::ClassHandle,
            contract: &ClassContract,
            selector: &SelectorContract,
        ) -> Result<(), MetalLoadError> {
            checked_selector(*class, contract, selector)
        }

        fn check_layout(&mut self, layout: &LayoutContract) -> Result<(), MetalLoadError> {
            checked_layout(layout)
        }

        fn system_device(&mut self) -> Result<Self::Device, MetalLoadError> {
            let (create_device, _framework_handle) =
                self.device_factory
                    .as_ref()
                    .ok_or_else(|| MetalLoadError::MissingSymbol {
                        framework: "Metal".to_owned(),
                        symbol: "MTLCreateSystemDefaultDevice".to_owned(),
                    })?;
            let device = unsafe { create_device() };
            let device = NonNull::new(device).ok_or(MetalLoadError::NoSystemDevice)?;
            Ok(unsafe { Device::from_ptr(device.as_ptr()) })
        }

        fn require_metal_3(&mut self, device: &Self::Device) -> Result<(), MetalLoadError> {
            if device.supports_family(MTLGPUFamily::Metal3) {
                Ok(())
            } else {
                Err(MetalLoadError::MissingMetal3)
            }
        }

        fn require_mps(
            &mut self,
            manifest: &AbiManifest,
            device: &Self::Device,
        ) -> Result<(), MetalLoadError> {
            let framework = manifest
                .frameworks
                .iter()
                .find(|framework| framework.name == "MetalPerformanceShaders")
                .ok_or_else(|| MetalLoadError::FrameworkOpen {
                    framework: "MetalPerformanceShaders".to_owned(),
                    reason: "reviewed framework contract is absent".to_owned(),
                })?;
            let handle = self.open_framework(framework)?;
            let symbol = CString::new("MPSSupportsMTLDevice")
                .map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
            let function = unsafe { dlsym(handle.handle.as_ptr(), symbol.as_ptr()) };
            let function = NonNull::new(function).ok_or_else(|| MetalLoadError::MissingSymbol {
                framework: handle.name.clone(),
                symbol: "MPSSupportsMTLDevice".to_owned(),
            })?;
            type MpsSupportsDevice = unsafe extern "C" fn(*mut c_void) -> i8;
            let supports: MpsSupportsDevice = unsafe { std::mem::transmute(function.as_ptr()) };
            // metal's foreign type is a transparent retained Objective-C object; the C shim consumes that exact object pointer without taking ownership.
            let device_pointer = (&**device as *const metal::DeviceRef)
                .cast_mut()
                .cast::<c_void>();
            if unsafe { supports(device_pointer) } == 0 {
                return Err(MetalLoadError::MpsUnsupported);
            }
            Ok(())
        }

        fn project_device(&self, device: &Self::Device) -> MetalDeviceProbe {
            MetalDeviceProbe {
                name: device.name().to_owned(),
                registry_id: device.registry_id(),
                recommended_working_set_bytes: device.recommended_max_working_set_size(),
                unified_memory: device.has_unified_memory(),
                metal_3: true,
                mps_supported: true,
            }
        }
    }

    pub fn probe_abi(manifest: &AbiManifest) -> Result<MetalAbiProbe, MetalLoadError> {
        probe_abi_with_system(&mut AppleProbeSystem::default(), manifest)
    }

    pub fn probe_device(manifest: &AbiManifest) -> Result<MetalDeviceProbe, MetalLoadError> {
        probe_device_with_system(&mut AppleProbeSystem::default(), manifest)
    }

    fn checked_symbol(
        handle: &FrameworkHandle,
        expected_image: &str,
        symbol: &str,
    ) -> Result<NonNull<c_void>, MetalLoadError> {
        let symbol_name =
            CString::new(symbol).map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
        let address = unsafe { dlsym(handle.handle.as_ptr(), symbol_name.as_ptr()) };
        let address = NonNull::new(address).ok_or_else(|| MetalLoadError::MissingSymbol {
            framework: handle.name.clone(),
            symbol: symbol.to_owned(),
        })?;
        let actual = image_for_address(address.as_ptr())?;
        if actual != expected_image {
            return Err(MetalLoadError::WrongSymbolImage {
                symbol: symbol.to_owned(),
                expected: expected_image.to_owned(),
                actual,
            });
        }
        Ok(address)
    }

    fn checked_class(contract: &ClassContract) -> Result<NonNull<ObjcClass>, MetalLoadError> {
        let name = CString::new(contract.name.as_str())
            .map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
        let class = unsafe { objc_getClass(name.as_ptr()) };
        let class = NonNull::new(class).ok_or_else(|| MetalLoadError::MissingClass {
            class: contract.name.clone(),
        })?;
        let image = unsafe { class_getImageName(class.as_ptr()) };
        let actual = checked_c_string(image, "Objective-C class image")?;
        if actual != contract.image_name {
            return Err(MetalLoadError::WrongClassImage {
                class: contract.name.clone(),
                expected: contract.image_name.clone(),
                actual,
            });
        }
        Ok(class)
    }

    fn checked_selector(
        class: NonNull<ObjcClass>,
        contract: &ClassContract,
        selector: &SelectorContract,
    ) -> Result<(), MetalLoadError> {
        let selector_name = CString::new(selector.name.as_str())
            .map_err(|error| MetalLoadError::Manifest(error.to_string()))?;
        let registered = unsafe { sel_registerName(selector_name.as_ptr()) };
        let method = match selector.kind {
            SelectorKind::Class => unsafe { class_getClassMethod(class.as_ptr(), registered) },
            SelectorKind::Instance => unsafe {
                class_getInstanceMethod(class.as_ptr(), registered)
            },
        };
        let method = NonNull::new(method).ok_or_else(|| MetalLoadError::MissingSelector {
            class: contract.name.clone(),
            selector: selector.name.clone(),
        })?;
        let encoding = unsafe { method_getTypeEncoding(method.as_ptr()) };
        let actual = checked_c_string(encoding, "Objective-C method encoding")?;
        if actual != selector.encoding {
            return Err(MetalLoadError::SelectorEncoding {
                class: contract.name.clone(),
                selector: selector.name.clone(),
                expected: selector.encoding.clone(),
                actual,
            });
        }
        Ok(())
    }

    fn checked_layout(layout: &LayoutContract) -> Result<(), MetalLoadError> {
        let (actual_size, actual_align) = match layout.name.as_str() {
            "MTLOrigin" => (
                size_of::<metal::MTLOrigin>(),
                align_of::<metal::MTLOrigin>(),
            ),
            "MTLSize" => (size_of::<metal::MTLSize>(), align_of::<metal::MTLSize>()),
            "MTLRegion" => (
                size_of::<metal::MTLRegion>(),
                align_of::<metal::MTLRegion>(),
            ),
            "MPSDataType" => (size_of::<u32>(), align_of::<u32>()),
            "Objective-C BOOL" => (size_of::<i8>(), align_of::<i8>()),
            other => {
                return Err(MetalLoadError::Manifest(format!(
                    "unknown reviewed layout {other}"
                )));
            }
        };
        if actual_size != layout.size || actual_align != layout.align {
            return Err(MetalLoadError::Layout {
                name: layout.name.clone(),
                expected_size: layout.size,
                expected_align: layout.align,
                actual_size,
                actual_align,
            });
        }
        Ok(())
    }

    fn image_for_address(address: *mut c_void) -> Result<String, MetalLoadError> {
        let mut info = DlInfo {
            image_name: ptr::null(),
            image_base: ptr::null_mut(),
            symbol_name: ptr::null(),
            symbol_address: ptr::null_mut(),
        };
        if unsafe { dladdr(address, &mut info) } == 0 {
            return Err(MetalLoadError::Manifest(
                "dladdr did not return symbol provenance".to_owned(),
            ));
        }
        checked_c_string(info.image_name, "dladdr image")
    }

    fn checked_c_string(value: *const c_char, field: &str) -> Result<String, MetalLoadError> {
        if value.is_null() {
            return Err(MetalLoadError::Manifest(format!("{field} is null")));
        }
        Ok(unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned())
    }

    fn dl_error() -> String {
        let error = unsafe { dlerror() };
        if error.is_null() {
            "dyld returned no error detail".to_owned()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        }
    }

    #[test]
    fn reviewed_constants_match_locked_bindings() {
        assert_eq!(
            MTLGPUFamily::Metal3 as u64,
            crate::abi::METAL_3_FAMILY_VALUE
        );
        assert_eq!(size_of::<u32>(), 4);
        assert_eq!(crate::abi::MPS_DATA_TYPE_FLOAT16, 0x1000_0010);
        assert_eq!(crate::abi::MPS_DATA_TYPE_FLOAT32, 0x1000_0020);
    }
}
