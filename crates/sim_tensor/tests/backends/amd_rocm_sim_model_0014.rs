use sim_tensor::{
    BackendCapabilityMatrix, DeviceId, GENERATED_MODULES, native_backend_binding_status,
};
use sim_types::{DeviceKind, NativeBackendBindingStatus};

#[test]
fn rocm_adapter_has_a_feature_gated_generated_root_edge() {
    assert!(GENERATED_MODULES.contains(&"ops/backend_amd_rocm_comfy_model_0014"));
    assert_ne!(
        std::mem::size_of::<sim_tensor::generated_backend_amd_rocm_sim_model_0014::RocmTensorBackend>(
        ),
        0
    );
}

#[test]
fn compilation_alone_does_not_publish_a_rocm_backend() {
    let status = native_backend_binding_status(DeviceKind::Rocm);
    assert!(matches!(
        status,
        NativeBackendBindingStatus::Unbound { reason, .. }
            if reason.contains("NativeFfiRegistry")
    ));
    let unavailable =
        BackendCapabilityMatrix::for_native_device(DeviceId::new(DeviceKind::Rocm, 0));
    assert!(unavailable.is_err());
}
