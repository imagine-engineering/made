#![cfg(feature = "mlu")]

use sim_tensor::{
    BackendCapabilityMatrix, DeviceId, GENERATED_MODULES, native_backend_binding_status,
};
use sim_types::{DeviceKind, NativeBackendBindingStatus};

#[test]
fn mlu_adapter_has_a_feature_gated_generated_root_edge() {
    assert!(GENERATED_MODULES.contains(&"ops/backend_cambricon_mlu_comfy_model_0017"));
    assert_ne!(
        std::mem::size_of::<
            sim_tensor::generated_backend_cambricon_mlu_sim_model_0017::MluTensorBackend,
        >(),
        0,
    );
}

#[test]
fn compilation_alone_does_not_publish_an_mlu_backend() {
    let status = native_backend_binding_status(DeviceKind::Mlu);
    assert!(matches!(
        status,
        NativeBackendBindingStatus::Unbound { reason, .. }
            if reason.contains("NativeFfiRegistry")
    ));
    let unavailable = BackendCapabilityMatrix::for_native_device(DeviceId::new(DeviceKind::Mlu, 0));
    assert!(unavailable.is_err());
}
