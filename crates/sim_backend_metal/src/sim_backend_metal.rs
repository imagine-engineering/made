mod abi;
mod execution;
mod execution_abi;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, AbiManifest, AbiManifestError, METAL_3_FAMILY_VALUE, MPS_DATA_TYPE_FLOAT16,
    MPS_DATA_TYPE_FLOAT32, READINESS_FUNCTION,
};
pub use execution::{
    MetalAllocation, MetalDeviceProperties, MetalDiagnostic, MetalElementType, MetalEvent,
    MetalExecutionError, MetalRuntime, MetalStorageMode, MetalStream,
};
pub use execution_abi::{
    EXECUTION_ABI_JSON, EXECUTION_CONTRACT, EXECUTION_UNSAFE_OWNER,
    MAXIMUM_COMMAND_BUFFERS_PER_STREAM, METAL_ADD_F16_FUNCTION, METAL_ADD_F32_FUNCTION,
    MetalExecutionAbi, MetalExecutionAbiError, MetalKernelContract, ResourceSelectorContract,
    ReturnNullability, StorageModeContract,
};
pub use loader::{MetalAbiProbe, MetalDeviceProbe, MetalLoadError, probe_abi, probe_device};

pub struct MetalBackend;

impl NativeBackendBinding for MetalBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(
            DeviceKind::Metal,
            "Metal remains unavailable until sim_runtime::NativeFfiRegistry certifies the fixed Apple framework provenance and signed readiness metallib",
        )
    }
}

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const REVIEWED_BINDINGS: &str = include_str!("../abi/reviewed-bindings-v1.txt");
pub const REVIEWED_EXECUTION_BINDINGS: &str =
    include_str!("../abi/reviewed-execution-bindings-v1.txt");
pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");
pub const EXECUTION_PACKAGE_LICENSES: &str = include_str!("../LICENSES.execution");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_inputs_are_complete_and_binding_remains_unbound()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.frameworks.len(), 3);
        assert!(REVIEWED_BINDINGS.contains("Xcode 26.2 build 17C52"));
        assert!(PACKAGE_LICENSES.contains("Apple system frameworks are not redistributed"));
        let status = NativeBackendBinding::binding_status(&MetalBackend);
        assert!(
            matches!(status, NativeBackendBindingStatus::Unbound { reason, .. } if reason.contains("NativeFfiRegistry"))
        );
        Ok(())
    }

    #[test]
    fn manifest_rejects_unknown_fields() {
        let input = ABI_MANIFEST_JSON.replacen("{", "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<AbiManifest>(&input).is_err());
    }

    #[cfg(all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    #[test]
    fn installed_frameworks_match_the_reviewed_abi_without_claiming_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let probe = probe_abi()?;
        assert_eq!(probe.framework_count, 3);
        assert_eq!(probe.symbol_count, 2);
        assert_eq!(probe.class_count, 3);
        assert_eq!(probe.selector_count, 12);
        match probe_device() {
            Ok(device) => {
                assert!(device.metal_3);
                assert!(device.mps_supported);
            }
            Err(MetalLoadError::NoSystemDevice) => {}
            Err(error) => return Err(error.into()),
        }
        assert!(matches!(
            NativeBackendBinding::binding_status(&MetalBackend),
            NativeBackendBindingStatus::Unbound { .. }
        ));
        Ok(())
    }
}
