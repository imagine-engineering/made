use crate::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    FunctionReference, NumericConstantReference, OPERATION_CONTRACTS, TypedReferenceContract,
    VersionValueReference,
};

pub const TORCH_BOOL_OPERATION_ID: &str = "COMFY-TENSOR-OP-4E4A9B623E55";
pub const TORCH_FINFO_EPS_OPERATION_ID: &str = "COMFY-TENSOR-OP-6473DB33DEF5";
pub const TORCH_FLOAT64_OPERATION_ID: &str = "COMFY-TENSOR-OP-54D5667F5955";
pub const TORCH_FLOAT8_E8M0FNU_OPERATION_ID: &str = "COMFY-TENSOR-OP-5A86177BA7E5";
pub const TORCH_INT32_OPERATION_ID: &str = "COMFY-TENSOR-OP-46930C6A85E0";
pub const TORCH_INT8_OPERATION_ID: &str = "COMFY-TENSOR-OP-4C14248E1A42";
pub const TORCH_NN_SOFTSIGN_OPERATION_ID: &str = "COMFY-TENSOR-OP-488843497A27";
pub const TORCH_UINT8_OPERATION_ID: &str = "COMFY-TENSOR-OP-4373837EB7FF";
pub const TORCH_VERSION_HIP_OPERATION_ID: &str = "COMFY-TENSOR-OP-49932406A466";
pub const XPU_HAS_FP16_OPERATION_ID: &str = "COMFY-TENSOR-OP-4C8967E2E390";
pub const XFORMERS_VERSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-51B5E5EF8766";
pub const XFORMERS_HAS_CPP_LIBRARY_OPERATION_ID: &str = "COMFY-TENSOR-OP-53979B2B90A6";

pub const TORCH_BOOL_REFERENCE: DType = DType::Bool;
pub const TORCH_FINFO_EPS_REFERENCE: NumericConstantReference =
    NumericConstantReference::FloatInfoEpsilon;
pub const TORCH_FLOAT64_REFERENCE: DType = DType::F64;
pub const TORCH_FLOAT8_E8M0FNU_REFERENCE: DType = DType::Float8E8m0Fnu;
pub const TORCH_INT32_REFERENCE: DType = DType::I32;
pub const TORCH_INT8_REFERENCE: DType = DType::I8;
pub const TORCH_NN_SOFTSIGN_REFERENCE: FunctionReference = FunctionReference::Softsign;
pub const TORCH_UINT8_REFERENCE: DType = DType::U8;
pub const TORCH_VERSION_HIP_REFERENCE: VersionValueReference = VersionValueReference::Hip;
pub const XPU_HAS_FP16_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::XpuHasFp16;
pub const XFORMERS_VERSION_REFERENCE: VersionValueReference = VersionValueReference::Xformers;
pub const XFORMERS_HAS_CPP_LIBRARY_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::XformersHasCppLibrary;

pub const ASSIGNED_VALUE_OR_CONSTANT_REFERENCES: &[(&str, CanonicalReference)] = &[
    (
        TORCH_BOOL_OPERATION_ID,
        CanonicalReference::DType(TORCH_BOOL_REFERENCE),
    ),
    (
        TORCH_FINFO_EPS_OPERATION_ID,
        CanonicalReference::NumericConstant(TORCH_FINFO_EPS_REFERENCE),
    ),
    (
        TORCH_FLOAT64_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT64_REFERENCE),
    ),
    (
        TORCH_FLOAT8_E8M0FNU_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT8_E8M0FNU_REFERENCE),
    ),
    (
        TORCH_INT32_OPERATION_ID,
        CanonicalReference::DType(TORCH_INT32_REFERENCE),
    ),
    (
        TORCH_INT8_OPERATION_ID,
        CanonicalReference::DType(TORCH_INT8_REFERENCE),
    ),
    (
        TORCH_NN_SOFTSIGN_OPERATION_ID,
        CanonicalReference::Function(TORCH_NN_SOFTSIGN_REFERENCE),
    ),
    (
        TORCH_UINT8_OPERATION_ID,
        CanonicalReference::DType(TORCH_UINT8_REFERENCE),
    ),
    (
        TORCH_VERSION_HIP_OPERATION_ID,
        CanonicalReference::VersionValue(TORCH_VERSION_HIP_REFERENCE),
    ),
    (
        XPU_HAS_FP16_OPERATION_ID,
        CanonicalReference::BooleanCapability(XPU_HAS_FP16_REFERENCE),
    ),
    (
        XFORMERS_VERSION_OPERATION_ID,
        CanonicalReference::VersionValue(XFORMERS_VERSION_REFERENCE),
    ),
    (
        XFORMERS_HAS_CPP_LIBRARY_OPERATION_ID,
        CanonicalReference::BooleanCapability(XFORMERS_HAS_CPP_LIBRARY_REFERENCE),
    ),
];

pub fn assigned_value_or_constant_contract(operation_id: &str) -> Option<TypedReferenceContract> {
    let (_, semantic) = ASSIGNED_VALUE_OR_CONSTANT_REFERENCES
        .iter()
        .find(|(assigned_id, _)| *assigned_id == operation_id)?;
    OPERATION_CONTRACTS.iter().find_map(|record| {
        let reference = record.typed_reference()?;
        (reference.operation_id() == operation_id
            && reference.inventory_kind() == ContractInventoryKind::NamespaceValueReference
            && reference.semantic() == *semantic)
            .then_some(reference)
    })
}

pub fn torch_bool_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_BOOL_OPERATION_ID)
}

pub fn torch_finfo_eps_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FINFO_EPS_OPERATION_ID)
}

pub fn torch_float64_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT64_OPERATION_ID)
}

pub fn torch_float8_e8m0fnu_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT8_E8M0FNU_OPERATION_ID)
}

pub fn torch_int32_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_INT32_OPERATION_ID)
}

pub fn torch_int8_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_INT8_OPERATION_ID)
}

pub fn torch_nn_softsign_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_NN_SOFTSIGN_OPERATION_ID)
}

pub fn torch_uint8_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_UINT8_OPERATION_ID)
}

pub fn torch_version_hip_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_VERSION_HIP_OPERATION_ID)
}

pub fn xpu_has_fp16_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(XPU_HAS_FP16_OPERATION_ID)
}

pub fn xformers_version_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(XFORMERS_VERSION_OPERATION_ID)
}

pub fn xformers_has_cpp_library_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(XFORMERS_HAS_CPP_LIBRARY_OPERATION_ID)
}
