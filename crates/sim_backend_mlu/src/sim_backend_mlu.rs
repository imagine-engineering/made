mod abi;
mod execution;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, AbiManifest, AbiManifestError, CnnlCreate, CnnlCreateOpTensorDescriptor,
    CnnlCreateTensorDescriptor, CnnlDataType, CnnlDestroy, CnnlDestroyOpTensorDescriptor,
    CnnlDestroyTensorDescriptor, CnnlGetLibVersion, CnnlHandle, CnnlNanPropagation, CnnlOpTensor,
    CnnlOpTensorDescription, CnnlOpTensorDescriptor, CnnlSetOpTensorDescriptor, CnnlSetQueue,
    CnnlSetTensorDescriptor, CnnlStatus, CnnlTensorDescriptor, CnnlTensorLayout, CnrtFree,
    CnrtGetDeviceCount, CnrtGetLibVersion, CnrtMalloc, CnrtMemTransferDirection, CnrtMemcpy,
    CnrtNotifier, CnrtQueue, CnrtQueueCreate, CnrtQueueDestroy, CnrtQueueSync, CnrtSetDevice,
    CnrtStatus, EnumContract, EnumValueContract, EnumVariantContract, LibraryContract,
    SymbolContract,
};
pub use execution::{
    MluElementType, MluExecutionAllocation, MluExecutionError, MluExecutionEvent,
    MluExecutionRuntime, MluExecutionStream,
};
pub use loader::{
    DiscoveryPlan, LibraryVersion, MluAbiProbe, MluLoadError, RegistryCertifiedImage,
};

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

pub struct MluBackend;

impl NativeBackendBinding for MluBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(
            DeviceKind::Mlu,
            format!(
                "Cambricon MLU unavailable on {}: the Neuware 1.20 ABI foundation is present, but sim_runtime::NativeFfiRegistry has not supplied certified libcnrt.so and libcnnl.so images",
                env!("SIM_MLU_TARGET")
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_is_strict_and_does_not_self_certify() -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.libraries.len(), 2);
        assert_eq!(manifest.libraries[0].id, "cnnl");
        assert_eq!(manifest.libraries[1].id, "cnrt");
        assert!(PACKAGE_LICENSES.contains("must not be redistributed"));
        assert!(matches!(
            NativeBackendBinding::binding_status(&MluBackend),
            NativeBackendBindingStatus::Unbound { reason, .. }
                if reason.contains("NativeFfiRegistry") && reason.contains(env!("SIM_MLU_TARGET"))
        ));
        Ok(())
    }

    #[test]
    fn manifest_rejects_unknown_fields_and_tampering() -> Result<(), Box<dyn std::error::Error>> {
        let unknown = ABI_MANIFEST_JSON.replacen('{', "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<AbiManifest>(&unknown).is_err());

        let tampered = ABI_MANIFEST_JSON.replace("cnrtQueueSync", "cnrtQueueWait");
        let parsed = serde_json::from_str::<AbiManifest>(&tampered)?;
        assert!(parsed.validate().is_err());
        Ok(())
    }

    #[test]
    fn catalog_maps_the_manifest_to_canonical_runtime_owners() {
        let catalog = include_str!(
            "../../../crates/sim_test_support/parity/catalogs/native-backend-abi/mlu.json"
        );
        assert!(catalog.contains(
            "\"canonical_abi_manifest_sha256\": \"40dbc7cbdda33dd1f0cd59b43057a8c00b2574393f3bd56838a57aa565068660\""
        ));
        assert!(catalog.contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\""));
        assert!(
            catalog.contains("\"binding_status_owner\": \"sim_types::NativeBackendBindingStatus\"")
        );
        assert!(catalog.contains("\"discovery_is_authorization\": false"));
        assert!(catalog.contains("\"package_receipt_is_authorization\": false"));
    }
}
