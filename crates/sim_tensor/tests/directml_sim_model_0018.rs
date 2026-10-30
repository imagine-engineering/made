#![cfg(feature = "directml")]

use sim_tensor::generated_backend_directml_sim_model_0018::DirectMlTensorBackend;
use sim_types::{NativeBackendBinding, NativeBackendBindingStatus};

#[test]
fn generated_directml_semantic_adapter_is_feature_registered() {
    assert!(sim_tensor::GENERATED_MODULES.contains(&"ops/backend_directml_comfy_model_0018"));
    assert_eq!(
        std::any::type_name::<DirectMlTensorBackend>(),
        "sim_tensor::generated_backend_directml_comfy_model_0018::DirectMlTensorBackend"
    );
}

#[test]
fn production_adapter_accepts_only_registry_certified_execution_sessions() {
    let source = include_str!("../src/backends/directml_sim_model_0018.rs");
    assert!(source.contains("DirectMlExecutionSession"));
    assert!(source.contains("pub fn from_certified_session("));
    assert!(source.contains("struct RuntimeAdapter(DirectMlExecutionSession);"));
    assert!(source.contains("RuntimeAdapter(session)"));
    assert!(!source.contains("NativeSessionAdapter"));
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
            "tensor DirectML adapter must not own parallel test runtime token {forbidden}"
        );
    }
    assert!(!source.contains("pub fn from_test"));
    assert!(!source.contains("pub fn fake_session"));
    assert!(!source.contains("DirectMlBackend::binding_status"));
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
            .any(|line| line == "directml = [\"cpu\", \"dep:sim_backend_directml\"]")
    );
    assert!(
        tensor_manifest
            .contains("sim_backend_directml = { workspace = true, features = [\"test-support\"] }")
    );

    let backend_manifest = include_str!("../../sim_backend_directml/Cargo.toml");
    assert!(backend_manifest.contains("test-support = []"));

    let status = NativeBackendBinding::binding_status(&sim_backend_directml::DirectMlBackend);
    assert!(matches!(
        status,
        NativeBackendBindingStatus::Unbound {
            device: sim_types::DeviceKind::DirectMl,
            ..
        }
    ));
}
