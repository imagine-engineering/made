mod abi;
mod execution;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, ABI_MANIFEST_JSON, AbiManifest, AbiManifestError, CannVersion, UNSAFE_OWNER,
};
pub use execution::{
    NpuAllocation, NpuDeviceProperties, NpuElementType, NpuEvent, NpuExecutionError,
    NpuExecutionSession, NpuStream,
};
pub use loader::{
    DiscoveredNpuLibraries, DiscoveryEnvironment, DiscoveryRoot, DiscoverySource, LibraryFailure,
    NpuLibraryCandidates, NpuLoadError, RegistryCertifiedNpuImages, SignedPackageRoot,
    discover_installed_libraries, discover_installed_libraries_for_target,
    discover_library_candidates, discover_library_candidates_for_target, supported_target,
    validate_package_version,
};

pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

pub struct NpuBackend;

impl NativeBackendBinding for NpuBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        let reason = loader::unavailable_reason();
        NativeBackendBindingStatus::unbound(DeviceKind::Npu, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_contract_is_complete_and_strict() -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.libraries.len(), 2);
        assert_eq!(manifest.symbol_count(), 28);
        assert_eq!(manifest.targets.len(), 2);
        assert_eq!(manifest.abi_floor, ABI_FLOOR);
        assert!(PACKAGE_LICENSES.contains("are not redistributed by this package"));
        Ok(())
    }

    #[test]
    fn binding_never_self_certifies_from_feature_compilation() {
        let status = NativeBackendBinding::binding_status(&NpuBackend);
        assert!(matches!(
            status,
            NativeBackendBindingStatus::Unbound { device: DeviceKind::Npu, reason }
                if reason.contains("NativeFfiRegistry")
                    && (reason.contains("unsupported target")
                        || reason.contains("registry-certified retained handles"))
        ));
    }

    #[test]
    fn execution_resource_owner_is_unique_owned_and_opaque() {
        let loader = include_str!("loader.rs");
        let execution = include_str!("execution.rs");
        for superseded in [
            "RetainedNpuLibraryHandles",
            "AscendClSymbols",
            "AscendClSession",
            "NpuContext<'",
            "NpuPendingCopy",
        ] {
            assert!(!loader.contains(superseded));
        }
        assert_eq!(
            loader
                .matches("pub unsafe fn from_registry_certified_handles(")
                .count(),
            1
        );
        assert!(loader.contains("pub struct RegistryCertifiedNpuImages"));
        assert!(execution.contains("pub struct NpuExecutionSession"));
        assert!(execution.contains("pub fn from_registry_certified_images("));
        assert!(!execution.contains("*mut c_void"));
        assert!(!execution.contains("NativeBackendBindingStatus::bound"));
    }

    #[test]
    fn manifest_rejects_tampering_and_unknown_fields() {
        let wrong_floor = ABI_MANIFEST_JSON.replace(ABI_FLOOR, "CANN-7.0");
        let error = AbiManifest::parse(&wrong_floor).expect_err("floor tamper must fail");
        assert!(matches!(error, AbiManifestError::Contract(_)));

        let unknown = ABI_MANIFEST_JSON.replacen('{', "{\"unknown\":true,", 1);
        let error = AbiManifest::parse(&unknown).expect_err("unknown fields must fail");
        assert!(matches!(error, AbiManifestError::Json(_)));

        let signature = ABI_MANIFEST_JSON.replace(
            "aclError aclrtSynchronizeEvent(aclrtEvent event)",
            "aclError aclrtSynchronizeEvent(void *event)",
        );
        let error = AbiManifest::parse(&signature).expect_err("signature tamper must fail");
        assert!(matches!(error, AbiManifestError::Contract(_)));
    }

    #[test]
    fn catalog_maps_the_manifest_to_canonical_runtime_owners() {
        let catalog = include_str!(
            "../../../crates/sim_test_support/parity/catalogs/native-backend-abi/npu.json"
        );
        assert!(catalog.contains(
            "\"canonical_abi_manifest_sha256\": \"2df75a090079b923cdeea2f5464b29a9c78ef35223b6a16884ed07a778466b2d\""
        ));
        assert!(catalog.contains("\"execution_owner\": \"sim_backend_npu::NpuExecutionSession\""));
        assert!(catalog.contains("\"certification_owner\": \"sim_runtime::NativeFfiRegistry\""));
        assert!(
            catalog.contains("\"binding_status_owner\": \"sim_types::NativeBackendBindingStatus\"")
        );
        assert!(catalog.contains("\"discovery_is_authorization\": false"));
        assert!(catalog.contains("\"package_receipt_is_authorization\": false"));
    }
}
