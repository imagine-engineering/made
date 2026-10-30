#![cfg(feature = "cuda")]

use sim_tensor::generated_backend_nvidia_cuda_sim_model_0022::CudaTensorBackend;
use sim_types::{NativeBackendBinding, NativeBackendBindingStatus};

#[test]
fn generated_cuda_semantic_adapter_is_feature_registered() {
    assert!(sim_tensor::GENERATED_MODULES.contains(&"ops/backend_nvidia_cuda_comfy_model_0022"));
    assert_eq!(
        std::any::type_name::<CudaTensorBackend>(),
        "sim_tensor::generated_backend_nvidia_cuda_comfy_model_0022::CudaTensorBackend"
    );
}

#[test]
fn production_adapter_accepts_only_registry_certified_execution_sessions() {
    let source = include_str!("../src/backends/nvidia_cuda_sim_model_0022.rs");
    assert!(source.contains("CudaExecutionSession"));
    assert!(source.contains("pub fn from_certified_session("));
    assert!(source.contains("struct RuntimeAdapter(CudaExecutionSession);"));
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
            "tensor CUDA adapter must not own parallel test runtime token {forbidden}"
        );
    }
    assert!(!source.contains("pub fn from_test"));
    assert!(!source.contains("pub fn fake_session"));
    assert!(!source.contains("CudaBackend::binding_status"));
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
            .any(|line| line == "cuda = [\"cpu\", \"dep:sim_backend_cuda\"]")
    );
    assert!(
        tensor_manifest
            .contains("sim_backend_cuda = { workspace = true, features = [\"test-support\"] }")
    );

    let backend_manifest = include_str!("../../sim_backend_cuda/Cargo.toml");
    assert!(backend_manifest.contains("test-support = []"));

    let status = NativeBackendBinding::binding_status(&sim_backend_cuda::CudaBackend);
    assert!(matches!(
        status,
        NativeBackendBindingStatus::Unbound {
            device: sim_types::DeviceKind::Cuda,
            ..
        }
    ));
}
