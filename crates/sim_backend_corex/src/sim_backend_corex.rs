mod abi;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, ABI_MANIFEST_JSON, AbiManifest, AbiManifestError, EvidenceContract, LibraryContract,
    MissingEvidenceContract, PackageContract, ReviewState, UNSAFE_OWNER,
};
pub use loader::{
    CertifiedCoreXImages, CoreXLoadError, DiscoveryEnvironment, DiscoveryPlan, DiscoveryRoot,
    DiscoverySource, RegistryCertifiedImage, SignedPackageRoot, supported_target,
    supported_target_name,
};

pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

pub struct CoreXBackend;

impl NativeBackendBinding for CoreXBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(DeviceKind::CoreX, loader::unavailable_reason())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_strict_and_records_the_provenance_blocker()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.libraries.len(), 2);
        assert_eq!(manifest.symbol_count(), 0);
        assert!(manifest.layouts.is_empty());
        assert_eq!(
            manifest.review_state,
            ReviewState::BlockedMissingVendorHeaders
        );
        assert!(manifest.missing_evidence.len() >= 3);
        assert!(PACKAGE_LICENSES.contains("redistributed by this package"));
        Ok(())
    }

    #[test]
    fn manifest_rejects_tampering_and_unknown_fields() {
        let wrong_floor = ABI_MANIFEST_JSON.replace(ABI_FLOOR, "CoreX-IXRT-0.9-ABI-profile");
        assert!(matches!(
            AbiManifest::parse(&wrong_floor),
            Err(AbiManifestError::Contract(_))
        ));

        let invented_symbol = ABI_MANIFEST_JSON.replacen(
            "\"symbols\": []",
            "\"symbols\": [{\"name\":\"cudaMalloc\",\"signature\":\"invented\"}]",
            1,
        );
        assert!(matches!(
            AbiManifest::parse(&invented_symbol),
            Err(AbiManifestError::Contract(_))
        ));

        let unknown = ABI_MANIFEST_JSON.replacen('{', "{\"unknown\":true,", 1);
        assert!(matches!(
            AbiManifest::parse(&unknown),
            Err(AbiManifestError::Json(_))
        ));
    }

    #[test]
    fn binding_is_unbound_even_when_the_feature_is_compiled() {
        let status = NativeBackendBinding::binding_status(&CoreXBackend);
        assert!(matches!(
            status,
            NativeBackendBindingStatus::Unbound {
                device: DeviceKind::CoreX,
                reason
            } if reason.contains(ABI_FLOOR)
                && reason.contains("reviewed IXRT/IXBLAS headers")
                && reason.contains("NativeFfiRegistry")
        ));
    }

    #[test]
    fn catalog_maps_the_manifest_to_canonical_runtime_owners() {
        let catalog = include_str!(
            "../../../.agents/specs/comfy-parity/catalogs/native-backend-abi/corex.json"
        );
        assert!(catalog.contains(
            "\"canonical_abi_manifest_sha256\": \"315597fb5ed4e1d0ae322ecda5437cb2583cb861e325a34c5259e534c58e166d\""
        ));
        assert!(catalog.contains("\"certification_owner\": \"sim_runtime::NativeFfiRegistry\""));
        assert!(
            catalog.contains("\"binding_status_owner\": \"sim_types::NativeBackendBindingStatus\"")
        );
        assert!(catalog.contains("\"discovery_is_authorization\": false"));
        assert!(catalog.contains("\"package_receipt_is_authorization\": false"));
        assert!(catalog.contains("\"runtime_loading_enabled\": false"));
    }
}
