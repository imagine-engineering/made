mod abi;
mod execution;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, ABI_MANIFEST, AbiManifest, AbiManifestError, BINDING_STATUS_OWNER,
    CERTIFICATE_OWNER, DnnlStatus, DnnlVersion, LEVEL_ZERO_MINIMUM_API_VERSION,
    ONEDNN_MINIMUM_MAJOR, ONEDNN_MINIMUM_MINOR, SEMANTIC_CAPABILITY_OWNER, UNSAFE_OWNER,
    ZeApiVersion, ZeCommandQueueDesc, ZeCommandQueueGroupProperties, ZeContextDesc, ZeResult,
};
pub use execution::{
    XpuAllocation, XpuDeviceProperties, XpuElementType, XpuEvent, XpuExecutionError,
    XpuExecutionSession,
};
pub use loader::{
    DiscoveryCandidate, DiscoveryPlan, DiscoverySource, LibraryLocation,
    RegistryCertifiedXpuImages, XpuAbiProbe, XpuLoadError,
};

pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

pub struct XpuBackend;

impl NativeBackendBinding for XpuBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(DeviceKind::Xpu, loader::unavailable_reason())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_never_self_certifies_binding() {
        assert!(matches!(
            NativeBackendBinding::binding_status(&XpuBackend),
            NativeBackendBindingStatus::Unbound {
                device: DeviceKind::Xpu,
                ..
            }
        ));
    }

    #[test]
    fn reviewed_manifest_and_notices_are_embedded() {
        assert!(ABI_MANIFEST.contains("\"source_tag\": \"v1.11.0\""));
        assert!(ABI_MANIFEST.contains("\"source_tag\": \"v3.5\""));
        assert!(PACKAGE_LICENSES.contains("sim_runtime::NativeFfiRegistry"));
    }

    #[test]
    fn canonical_owners_are_explicit_and_separate() {
        assert_eq!(UNSAFE_OWNER, "sim_backend_xpu::loader");
        assert_eq!(CERTIFICATE_OWNER, "sim_runtime::NativeFfiRegistry");
        assert_eq!(
            BINDING_STATUS_OWNER,
            "sim_types::NativeBackendBindingStatus"
        );
        assert_eq!(
            SEMANTIC_CAPABILITY_OWNER,
            "sim_tensor::BackendCapabilityMatrix"
        );
    }

    #[test]
    fn generated_catalog_maps_to_the_canonical_owners() {
        let catalog = include_str!(
            "../../../.agents/specs/comfy-parity/catalogs/native-backend-abi/xpu.json"
        );
        assert!(catalog.contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\""));
        assert!(
            catalog.contains("\"binding_status_owner\": \"sim_types::NativeBackendBindingStatus\"")
        );
        assert!(catalog.contains("\"discovery_is_authorization\": false"));
        assert!(catalog.contains("\"package_receipt_is_authorization\": false"));
    }
}
