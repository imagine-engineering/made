#![cfg(feature = "xpu")]

use sim_tensor::generated_backend_intel_xpu_sim_model_0021::XpuTensorBackend;
use sim_types::{NativeBackendBinding, NativeBackendBindingStatus};

#[test]
fn generated_xpu_semantic_adapter_is_feature_registered() {
    assert!(sim_tensor::GENERATED_MODULES.contains(&"ops/backend_intel_xpu_comfy_model_0021"));
    assert_eq!(
        std::any::type_name::<XpuTensorBackend>(),
        "sim_tensor::generated_backend_intel_xpu_comfy_model_0021::XpuTensorBackend"
    );
}

#[test]
fn production_adapter_accepts_only_registry_certified_execution_sessions() {
    let source = include_str!("../src/backends/intel_xpu_sim_model_0021.rs");
    assert!(source.contains("XpuExecutionSession"));
    assert!(source.contains("pub fn from_certified_session("));
    assert!(source.contains("struct RuntimeAdapter(XpuExecutionSession);"));
    assert!(source.contains("RuntimeAdapter(session)"));
    for forbidden in [
        "TestRuntime",
        "TestAllocation",
        "RuntimeAdapter::Test",
        "AllocationAdapter::Test",
        "StreamAdapter::Test",
        "EventAdapter::Test",
    ] {
        assert!(
            !source.contains(forbidden),
            "tensor XPU adapter must not own parallel test runtime token {forbidden}"
        );
    }
    assert!(!source.contains("pub fn from_test"));
    assert!(!source.contains("pub fn fake_session"));
    assert!(!source.contains("XpuBackend::binding_status"));
    assert!(!source.contains("NativeBackendBindingStatus::Bound"));
    assert!(!source.contains("std::process"));
    assert!(!source.contains("Command::new"));
}

#[test]
fn test_support_is_dev_only_and_cannot_grant_backend_availability() {
    let tensor_manifest = include_str!("../Cargo.toml");
    assert!(
        tensor_manifest
            .lines()
            .any(|line| line == "xpu = [\"cpu\", \"dep:sim_backend_xpu\"]")
    );
    assert!(
        tensor_manifest
            .contains("sim_backend_xpu = { workspace = true, features = [\"test-support\"] }")
    );

    let backend_manifest = include_str!("../../sim_backend_xpu/Cargo.toml");
    assert!(backend_manifest.contains("test-support = []"));

    let status = NativeBackendBinding::binding_status(&sim_backend_xpu::XpuBackend);
    assert!(matches!(
        status,
        NativeBackendBindingStatus::Unbound {
            device: sim_types::DeviceKind::Xpu,
            ..
        }
    ));
}
