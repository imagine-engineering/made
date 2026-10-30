mod abi;
mod execution;
mod loader;

use sim_types::{DeviceKind, NativeBackendBinding, NativeBackendBindingStatus};

pub use abi::{
    ABI_FLOOR, ABI_MANIFEST_JSON, AbiManifest, AbiManifestError, ArchitectureContract,
    DML_BINDING_TYPE_BUFFER, DML_BINDING_TYPE_BUFFER_ARRAY, DML_BINDING_TYPE_NONE,
    DML_CREATE_DEVICE_FLAG_DEBUG, DML_CREATE_DEVICE_FLAG_NONE, DML_EXECUTION_FLAG_NONE,
    DML_FEATURE_FEATURE_LEVELS, DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,
    DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT, DML_OPERATOR_ELEMENT_WISE_ADD,
    DML_PERSISTENT_BUFFER_ALIGNMENT, DML_TEMPORARY_BUFFER_ALIGNMENT, DML_TENSOR_DATA_TYPE_FLOAT16,
    DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_FLAG_NONE, DML_TENSOR_TYPE_BUFFER, DmlBindingDesc,
    DmlBindingProperties, DmlBindingTableDesc, DmlBindingType, DmlBufferArrayBinding,
    DmlBufferBinding, DmlBufferTensorDesc, DmlCreateDeviceFlags, DmlElementWiseAddOperatorDesc,
    DmlExecutionFlags, DmlFeature, DmlFeatureDataFeatureLevels,
    DmlFeatureDataTensorDataTypeSupport, DmlFeatureLevel, DmlFeatureQueryFeatureLevels,
    DmlFeatureQueryTensorDataTypeSupport, DmlOperatorDesc, DmlOperatorType, DmlTensorDataType,
    DmlTensorDesc, DmlTensorFlags, DmlTensorType, FILE_VERSION, FileVersion, MINIMUM_FEATURE_LEVEL,
    MINIMUM_WINDOWS_BUILD, RedistributableContract, ReviewedPackage, TARGET_VERSION, UNSAFE_OWNER,
};
#[cfg(feature = "test-support")]
pub use execution::DirectMlTestControl;
pub use execution::{
    DirectMlAllocation, DirectMlDeviceProperties, DirectMlElementType, DirectMlEvent,
    DirectMlExecutionError, DirectMlExecutionSession, DirectMlStream,
};
pub use loader::{
    DirectMlAbiProbe, DirectMlCandidate, DirectMlCandidateObservation, DirectMlDiscoveryPlan,
    DirectMlLoadError, DiscoverySource, RegistryCertifiedDirectMlImage,
    RetainedDirectMlLibraryHandles, observe_directml_candidate, probe_certified,
    validate_candidate_observation,
};

pub const PACKAGE_LICENSES: &str = include_str!("../LICENSES");

pub struct DirectMlBackend;

impl NativeBackendBinding for DirectMlBackend {
    fn binding_status(&self) -> NativeBackendBindingStatus {
        NativeBackendBindingStatus::unbound(DeviceKind::DirectMl, loader::unavailable_reason())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_probe_never_self_certifies_the_backend() {
        let status = NativeBackendBinding::binding_status(&DirectMlBackend);
        assert!(matches!(
            status,
            NativeBackendBindingStatus::Unbound {
                device: DeviceKind::DirectMl,
                ..
            }
        ));
    }

    #[test]
    fn checked_package_inputs_are_embedded() {
        assert!(ABI_MANIFEST_JSON.contains("\"abi_floor\": \"1.13.1\""));
        assert!(PACKAGE_LICENSES.contains("Microsoft.AI.DirectML 1.13.1"));
    }

    #[test]
    fn vendor_loader_owns_exact_module_lifetime_and_focused_local_observation() {
        let source = include_str!("loader.rs");
        assert_eq!(source.matches("LoadLibraryExW(").count(), 1);
        assert_eq!(source.matches("FreeLibrary(HMODULE(").count(), 1);
        assert!(source.contains("pub unsafe fn load_from_registry_certificate("));
        assert!(source.contains("pub unsafe fn from_registry_certificates("));
        assert_eq!(source.matches("fn verify_authenticode_offline(").count(), 2);
        assert_eq!(source.matches("WinVerifyTrust(").count(), 2);
        assert!(source.contains("WTD_CACHE_ONLY_URL_RETRIEVAL"));
        assert!(source.contains("RtlGetVersion"));
        assert!(source.contains("GetFileVersionInfoW"));
        assert!(source.contains("pub fn observe_directml_candidate("));
        assert!(!source.contains("pub authenticode_trusted: bool"));
        assert!(!source.contains("authenticode_trusted: bool"));
        assert!(!source.contains("NativeBackendBindingStatus::bound"));
        assert!(source.contains("_retention: Arc<dyn Any + Send + Sync>"));
        assert!(
            !source.contains("derive(Clone, Debug)]\npub struct RetainedDirectMlLibraryHandles")
        );
        assert!(!source.contains("pub struct DirectMlRuntime"));
        assert!(!source.contains("RegistryCertificationSession"));
    }

    #[test]
    fn execution_uses_loader_owned_ffi_and_reviewed_binding_rules() {
        let loader = include_str!("loader.rs");
        let execution = include_str!("execution.rs");
        assert!(loader.contains("self.symbols.create_dxgi_factory2"));
        assert!(loader.contains("self.symbols.d3d12_create_device"));
        assert!(loader.contains("D3D_FEATURE_LEVEL_11_0"));
        assert!(!execution.contains("transmute"));
        assert!(!execution.contains("0xc000"));
        assert!(!execution.contains("d3d12_create_device_address"));
        assert!(execution.contains("initialize_table.bind_outputs"));
        assert!(execution.contains("uav_barrier(&initialize_list, resource)"));
        assert!(execution.contains("D3D12_RESOURCE_BARRIER_TYPE_UAV"));
        assert!(execution.contains("IDXGIAdapter3::QueryVideoMemoryInfo(local)"));
        assert!(execution.contains("ID3D12Fence::SetEventOnCompletion"));
        assert!(execution.contains("DirectMlBinding::None"));
        assert_eq!(execution.matches("bind_persistent_resource").count(), 1);
        assert!(execution.contains("binding_byte_length"));
        assert!(matches!(
            (
                execution.find("DirectMlDispatchable::Initializer(&initializer)"),
                execution.find("uav_barrier(&initialize_list, resource)"),
                execution.find("execute_table.bind_persistent_resource"),
                execution.rfind("DirectMlDispatchable::Compiled(&compiled)"),
            ),
            (Some(initialize), Some(barrier), Some(bind), Some(execute))
                if initialize < barrier && barrier < bind && bind < execute
        ));
    }

    #[test]
    fn generated_catalog_maps_to_the_canonical_abi_and_registry_owner() {
        let catalog = include_str!(
            "../../../crates/sim_test_support/parity/catalogs/native-backend-abi/directml.json"
        );
        assert!(catalog.contains(
            "\"canonical_abi_manifest_sha256\": \"54065e17bc9d69a4e377315caa6bcdef1b4898c660e2e0c1660ed4dcf4146e35\""
        ));
        assert!(catalog.contains("\"status\": \"execution_supported\""));
        assert!(
            catalog.contains("\"unsafe_ffi_owner\": \"sim_backend_directml::{loader,execution}\"")
        );
        assert!(catalog.contains(
            "\"execution_task_id\": \"sim-parity-directml-execution-resource-ownership-consolidation\""
        ));
        assert!(!catalog.contains("DirectMlRuntime"));
        assert!(catalog.contains("\"certificate_issuer\": \"sim_runtime::NativeFfiRegistry\""));
        assert!(catalog.contains("\"discovery_is_authorization\": false"));
        assert!(catalog.contains("\"package_receipt_is_authorization\": false"));

        let package_policy = include_str!("../../../nix/sim-backends/directml/package-policy.json");
        assert!(package_policy.contains("\"runtime_authorization_from_structure\": false"));
        assert!(
            package_policy.contains(
                "\"signature_authority\": \"sim_runtime::DirectMlPackageVerificationKey\""
            )
        );
        assert!(
            package_policy.contains("\"certificate_owner\": \"sim_runtime::NativeFfiRegistry\"")
        );
        assert!(package_policy.contains("\"ffi-contracts-v1.json\""));
    }
}
