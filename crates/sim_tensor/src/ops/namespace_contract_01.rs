use crate::{
    CanonicalReference, NamespaceReference, OPERATION_CONTRACTS, TypedReferenceContract,
    VersionValueReference,
};

pub const SIM_OPS_OPERATION_ID: &str = "COMFY-TENSOR-OP-7D7398921719";
pub const TORCH_NEURAL_NETWORK_OPERATION_ID: &str = "COMFY-TENSOR-OP-764A8E60B071";
pub const TORCH_CUDA_VERSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-7A62A3A11490";

pub const SIM_OPS_REFERENCE: NamespaceReference = NamespaceReference::ComfyOps;
pub const TORCH_NEURAL_NETWORK_REFERENCE: NamespaceReference =
    NamespaceReference::TorchNeuralNetwork;
pub const TORCH_CUDA_VERSION_REFERENCE: VersionValueReference = VersionValueReference::Cuda;

fn assigned_reference(operation_id: &str) -> Option<TypedReferenceContract> {
    OPERATION_CONTRACTS.iter().find_map(|record| {
        let reference = record.typed_reference()?;
        (reference.operation_id() == operation_id).then_some(reference)
    })
}

pub fn sim_ops_contract() -> Option<TypedReferenceContract> {
    assigned_reference(SIM_OPS_OPERATION_ID).filter(|contract| {
        contract.semantic() == CanonicalReference::Namespace(SIM_OPS_REFERENCE)
    })
}

pub fn torch_neural_network_contract() -> Option<TypedReferenceContract> {
    assigned_reference(TORCH_NEURAL_NETWORK_OPERATION_ID).filter(|contract| {
        contract.semantic() == CanonicalReference::Namespace(TORCH_NEURAL_NETWORK_REFERENCE)
    })
}

pub fn torch_cuda_version_contract() -> Option<TypedReferenceContract> {
    assigned_reference(TORCH_CUDA_VERSION_OPERATION_ID).filter(|contract| {
        contract.semantic() == CanonicalReference::VersionValue(TORCH_CUDA_VERSION_REFERENCE)
    })
}
