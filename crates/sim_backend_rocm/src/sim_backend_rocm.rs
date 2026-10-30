mod abi;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use loader::{
    ComponentVersions, DiscoveryRoot, DiscoverySource, PackageSignatureContract,
    PlatformPackageVerifier, RocmAllocation, RocmDependencyCandidate, RocmDependencyEdge,
    RocmDeviceProperties, RocmEvent, RocmExecutionError, RocmLibraryCandidate, RocmLibraryRole,
    RocmLibrarySet, RocmLoadError, RocmRuntime, RocmStream, VerifiedRocmPackageRoot,
    admit_signed_package_root, discover_from_environment, discover_library_set,
    discover_with_verified_package_roots, verify_signed_package_root,
};

pub struct RocmBackend;

impl NativeBackendBinding for RocmBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(
            DeviceKind::Rocm,
            "ROCm remains unavailable until sim_runtime::NativeFfiRegistry certifies the exact library digests, ABI, symbol sets, and unsafe owner",
        )
    }
}

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::NativeBackendBindingStatus;

    #[test]
    fn binding_remains_unbound_until_runtime_certification_is_integrated() {
        let status = NativeBackendBinding::binding_status(&RocmBackend);
        assert!(
            matches!(status, NativeBackendBindingStatus::Unbound { reason, .. } if reason.contains("NativeFfiRegistry"))
        );
    }

    #[test]
    fn checked_package_inputs_are_embedded() {
        assert!(ABI_MANIFEST_JSON.contains("\"abi_floor\": \"6.1.0\""));
        assert!(PACKAGE_LICENSES.contains("AMD runtime libraries are not redistributed"));
    }
}
