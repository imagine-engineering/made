use crate::{
    CanonicalReference, ContractInventoryKind, OPERATION_CONTRACTS, TypeMarkerReference,
    TypedReferenceContract,
};

pub const SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID: &str = "COMFY-TENSOR-OP-394EC82DC4A9";
pub const SIM_DISABLE_WEIGHT_INIT_OPERATION_ID: &str = "COMFY-TENSOR-OP-6BEA6C96D82D";
pub const TORCH_LONG_TENSOR_OPERATION_ID: &str = "COMFY-TENSOR-OP-846A2E54A5E1";
pub const TORCH_AUTOGRAD_FUNCTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-0AA720652F2F";
pub const TORCH_DTYPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-741CFECE3BBA";
pub const TORCH_EMPTY_DEVICE_OPERATION_ID: &str = "COMFY-TENSOR-OP-3B7C406ED382";
pub const TORCH_JIT_FINAL_OPERATION_ID: &str = "COMFY-TENSOR-OP-629C78356D4F";
pub const TORCH_CONV_TRANSPOSE_1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-6CC96FA32F3C";
pub const TORCH_CONV_TRANSPOSE_2D_OPERATION_ID: &str = "COMFY-TENSOR-OP-9657182ECBB6";
pub const TORCH_OPTIMIZER_OPERATION_ID: &str = "COMFY-TENSOR-OP-62083D582404";
pub const TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID: &str = "COMFY-TENSOR-OP-7A771C76A245";
pub const TORCH_DATASET_OPERATION_ID: &str = "COMFY-TENSOR-OP-57854A0E567E";

pub const SIM_CAST_WEIGHT_BIAS_OP_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::ComfyCastWeightBiasOp;
pub const SIM_DISABLE_WEIGHT_INIT_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::ComfyDisableWeightInit;
pub const TORCH_LONG_TENSOR_REFERENCE: TypeMarkerReference = TypeMarkerReference::LongTensor;
pub const TORCH_AUTOGRAD_FUNCTION_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::AutogradFunction;
pub const TORCH_DTYPE_REFERENCE: TypeMarkerReference = TypeMarkerReference::DType;
pub const TORCH_EMPTY_DEVICE_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::EmptyTensorDevice;
pub const TORCH_JIT_FINAL_REFERENCE: TypeMarkerReference = TypeMarkerReference::JitFinal;
pub const TORCH_CONV_TRANSPOSE_1D_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::ConvTranspose1d;
pub const TORCH_CONV_TRANSPOSE_2D_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::ConvTranspose2d;
pub const TORCH_OPTIMIZER_REFERENCE: TypeMarkerReference = TypeMarkerReference::Optimizer;
pub const TORCH_LEARNING_RATE_SCHEDULER_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::LearningRateScheduler;
pub const TORCH_DATASET_REFERENCE: TypeMarkerReference = TypeMarkerReference::Dataset;

pub const ASSIGNED_TYPE_REFERENCES: &[(&str, TypeMarkerReference)] = &[
    (
        SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID,
        SIM_CAST_WEIGHT_BIAS_OP_REFERENCE,
    ),
    (
        SIM_DISABLE_WEIGHT_INIT_OPERATION_ID,
        SIM_DISABLE_WEIGHT_INIT_REFERENCE,
    ),
    (TORCH_LONG_TENSOR_OPERATION_ID, TORCH_LONG_TENSOR_REFERENCE),
    (
        TORCH_AUTOGRAD_FUNCTION_OPERATION_ID,
        TORCH_AUTOGRAD_FUNCTION_REFERENCE,
    ),
    (TORCH_DTYPE_OPERATION_ID, TORCH_DTYPE_REFERENCE),
    (
        TORCH_EMPTY_DEVICE_OPERATION_ID,
        TORCH_EMPTY_DEVICE_REFERENCE,
    ),
    (TORCH_JIT_FINAL_OPERATION_ID, TORCH_JIT_FINAL_REFERENCE),
    (
        TORCH_CONV_TRANSPOSE_1D_OPERATION_ID,
        TORCH_CONV_TRANSPOSE_1D_REFERENCE,
    ),
    (
        TORCH_CONV_TRANSPOSE_2D_OPERATION_ID,
        TORCH_CONV_TRANSPOSE_2D_REFERENCE,
    ),
    (TORCH_OPTIMIZER_OPERATION_ID, TORCH_OPTIMIZER_REFERENCE),
    (
        TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID,
        TORCH_LEARNING_RATE_SCHEDULER_REFERENCE,
    ),
    (TORCH_DATASET_OPERATION_ID, TORCH_DATASET_REFERENCE),
];

pub fn assigned_type_contract(operation_id: &str) -> Option<TypedReferenceContract> {
    let (_, marker) = ASSIGNED_TYPE_REFERENCES
        .iter()
        .find(|(assigned_id, _)| *assigned_id == operation_id)?;
    OPERATION_CONTRACTS.iter().find_map(|record| {
        let reference = record.typed_reference()?;
        (reference.operation_id() == operation_id
            && reference.inventory_kind() == ContractInventoryKind::TypeReference
            && reference.semantic() == CanonicalReference::TypeMarker(*marker))
        .then_some(reference)
    })
}

pub fn sim_cast_weight_bias_op_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(SIM_CAST_WEIGHT_BIAS_OP_OPERATION_ID)
}

pub fn sim_disable_weight_init_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(SIM_DISABLE_WEIGHT_INIT_OPERATION_ID)
}

pub fn torch_long_tensor_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_LONG_TENSOR_OPERATION_ID)
}

pub fn torch_autograd_function_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_AUTOGRAD_FUNCTION_OPERATION_ID)
}

pub fn torch_dtype_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_DTYPE_OPERATION_ID)
}

pub fn torch_empty_device_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_EMPTY_DEVICE_OPERATION_ID)
}

pub fn torch_jit_final_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_JIT_FINAL_OPERATION_ID)
}

pub fn torch_conv_transpose_1d_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_CONV_TRANSPOSE_1D_OPERATION_ID)
}

pub fn torch_conv_transpose_2d_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_CONV_TRANSPOSE_2D_OPERATION_ID)
}

pub fn torch_optimizer_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_OPTIMIZER_OPERATION_ID)
}

pub fn torch_learning_rate_scheduler_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_LEARNING_RATE_SCHEDULER_OPERATION_ID)
}

pub fn torch_dataset_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_DATASET_OPERATION_ID)
}
