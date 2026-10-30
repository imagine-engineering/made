mod abi;
mod execution;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, AbiManifest, AbiManifestError, CERTIFICATE_OWNER, CUBLASLT_VERSION_MINIMUM,
    CUDA_DRIVER_VERSION_MINIMUM, CUDA_ERROR_CONTEXT_IS_DESTROYED, CUDA_ERROR_DEVICE_UNAVAILABLE,
    CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_OUT_OF_MEMORY,
    CUDNN_VERSION_MINIMUM, HeaderContract, LayoutContract, LibraryContract, PackagePolicy,
    SymbolContract, VersionContract,
};
pub use execution::{
    CudaAllocation, CudaDeviceProperties, CudaElementType, CudaEvent, CudaExecutionError,
    CudaExecutionSession,
};
pub use loader::{
    CORE_PTX_SHA256, CudaLibraryCandidates, CudaLoadError, DiscoveryEnvironment, DiscoverySource,
    RegistryCertifiedCudaImages, RuntimeVersions, SignedPackageRoot, discovery_candidates,
    unavailable_reason, validate_discovered_library,
};

pub const ABI_MANIFEST_JSON: &str = include_str!("../abi/symbols-v1.json");
pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

pub struct CudaBackend;

impl NativeBackendBinding for CudaBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(DeviceKind::Cuda, unavailable_reason())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foundation_is_strict_and_never_self_certifies() -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        assert_eq!(manifest.libraries.len(), 4);
        assert_eq!(manifest.libraries[2].id, "driver");
        assert!(PACKAGE_LICENSES.contains("redistribute the NVIDIA display"));
        assert!(PACKAGE_LICENSES.contains("driver"));
        assert!(matches!(
            NativeBackendBinding::binding_status(&CudaBackend),
            NativeBackendBindingStatus::Unbound { reason, .. }
                if reason.contains("NativeFfiRegistry")
                    && reason.contains(env!("SIM_CUDA_TARGET"))
                    && reason.contains(ABI_FLOOR)
        ));
        Ok(())
    }

    #[test]
    fn manifest_rejects_unknown_fields_symbol_and_version_tampering()
    -> Result<(), Box<dyn std::error::Error>> {
        let unknown = ABI_MANIFEST_JSON.replacen('{', "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<AbiManifest>(&unknown).is_err());

        let renamed = ABI_MANIFEST_JSON.replace("cuInit", "cuInitialize");
        let parsed = serde_json::from_str::<AbiManifest>(&renamed)?;
        assert!(parsed.validate().is_err());

        let weakened = ABI_MANIFEST_JSON.replace("12020", "12010");
        let parsed = serde_json::from_str::<AbiManifest>(&weakened)?;
        assert!(parsed.validate().is_err());

        let changed_header = ABI_MANIFEST_JSON.replace(
            "e752b21d073b4fdaf19957cd8a63fd3babe46bc26a05d79b8d928258a65a92de",
            &"0".repeat(64),
        );
        let parsed = serde_json::from_str::<AbiManifest>(&changed_header)?;
        assert!(parsed.validate().is_err());
        Ok(())
    }

    #[test]
    fn layouts_match_the_reviewed_64_bit_c_abi() -> Result<(), Box<dyn std::error::Error>> {
        let manifest = AbiManifest::embedded()?;
        let uuid = manifest
            .layouts
            .iter()
            .find(|layout| layout.name == "CUuuid")
            .ok_or("CUuuid layout missing")?;
        assert_eq!((uuid.size, uuid.align), (16, 1));
        for pointer in [
            "CUcontext",
            "CUevent",
            "CUfunction",
            "CUmodule",
            "CUstream",
            "cublasLtHandle_t",
            "cudnnHandle_t",
            "nvrtcProgram",
        ] {
            let layout = manifest
                .layouts
                .iter()
                .find(|layout| layout.name == pointer)
                .ok_or("pointer layout missing")?;
            assert_eq!((layout.size, layout.align), (8, 8));
        }
        Ok(())
    }

    #[test]
    fn catalog_maps_boundary_dtos_to_canonical_domain_owners() {
        let catalog = include_str!(
            "../../../.agents/specs/comfy-parity/catalogs/native-backend-abi/cuda.json"
        );
        assert!(catalog.contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\""));
        assert!(
            catalog.contains("\"binding_status_owner\": \"sim_types::NativeBackendBindingStatus\"")
        );
        assert!(
            catalog
                .contains("\"semantic_capability_owner\": \"sim_tensor::BackendCapabilityMatrix\"")
        );
        assert!(catalog.contains("\"discovery_is_authorization\": false"));
        assert!(catalog.contains("\"package_receipt_is_authorization\": false"));
        assert!(catalog.contains(
            "\"canonical_abi_manifest_sha256\": \"6bd8b7e8657a60a7e203c1c4bc2cea55f5b07781a2c9e9ba89ffad5b5dde0440\""
        ));
    }
}
