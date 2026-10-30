use crate::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    DevicePropertyReference, EnumVariantReference, FunctionReference, OPERATION_CONTRACTS,
    TypedReferenceContract, VersionValueReference,
};

pub const CUDNN_BENCHMARK_OPERATION_ID: &str = "COMFY-TENSOR-OP-B92DC7E2F35F";
pub const TORCH_COMPLEX64_OPERATION_ID: &str = "COMFY-TENSOR-OP-D905B4531CBB";
pub const TORCH_FLOAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-C86C8A53B4E8";
pub const TORCH_FLOAT8_E4M3FN_OPERATION_ID: &str = "COMFY-TENSOR-OP-DC1A47F73314";
pub const TORCH_FLOAT8_E5M2FNUZ_OPERATION_ID: &str = "COMFY-TENSOR-OP-C25E7D705E0B";
pub const TORCH_INT16_OPERATION_ID: &str = "COMFY-TENSOR-OP-C993484611F1";
pub const TORCH_NN_HARDSWISH_OPERATION_ID: &str = "COMFY-TENSOR-OP-B37B2E52BEFE";
pub const TORCH_NN_SELU_OPERATION_ID: &str = "COMFY-TENSOR-OP-C4AC5E5E45C7";
pub const XPU_TOTAL_MEMORY_OPERATION_ID: &str = "COMFY-TENSOR-OP-D31F4FB613FB";
pub const INTERPOLATION_NEAREST_OPERATION_ID: &str = "COMFY-TENSOR-OP-CB3E6D0F9373";
pub const FUNCTIONAL_INTERPOLATION_BICUBIC_OPERATION_ID: &str = "COMFY-TENSOR-OP-C1E5061C3330";
pub const XFORMERS_MODULE_VERSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-DC28FD314A01";

pub const CUDNN_BENCHMARK_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::CudnnBenchmark;
pub const TORCH_COMPLEX64_REFERENCE: DType = DType::Complex64;
pub const TORCH_FLOAT_REFERENCE: DType = DType::F32;
pub const TORCH_FLOAT8_E4M3FN_REFERENCE: DType = DType::Float8E4m3Fn;
pub const TORCH_FLOAT8_E5M2FNUZ_REFERENCE: DType = DType::Float8E5m2Fnuz;
pub const TORCH_INT16_REFERENCE: DType = DType::I16;
pub const TORCH_NN_HARDSWISH_REFERENCE: FunctionReference = FunctionReference::Hardswish;
pub const TORCH_NN_SELU_REFERENCE: FunctionReference = FunctionReference::Selu;
pub const XPU_TOTAL_MEMORY_REFERENCE: DevicePropertyReference =
    DevicePropertyReference::XpuTotalMemory;
pub const INTERPOLATION_NEAREST_REFERENCE: EnumVariantReference =
    EnumVariantReference::InterpolationNearest;
pub const FUNCTIONAL_INTERPOLATION_BICUBIC_REFERENCE: EnumVariantReference =
    EnumVariantReference::FunctionalInterpolationBicubic;
pub const XFORMERS_MODULE_VERSION_REFERENCE: VersionValueReference =
    VersionValueReference::XformersModule;

pub const ASSIGNED_VALUE_OR_CONSTANT_REFERENCES: &[(&str, CanonicalReference)] = &[
    (
        CUDNN_BENCHMARK_OPERATION_ID,
        CanonicalReference::BooleanCapability(CUDNN_BENCHMARK_REFERENCE),
    ),
    (
        TORCH_COMPLEX64_OPERATION_ID,
        CanonicalReference::DType(TORCH_COMPLEX64_REFERENCE),
    ),
    (
        TORCH_FLOAT_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT_REFERENCE),
    ),
    (
        TORCH_FLOAT8_E4M3FN_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT8_E4M3FN_REFERENCE),
    ),
    (
        TORCH_FLOAT8_E5M2FNUZ_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT8_E5M2FNUZ_REFERENCE),
    ),
    (
        TORCH_INT16_OPERATION_ID,
        CanonicalReference::DType(TORCH_INT16_REFERENCE),
    ),
    (
        TORCH_NN_HARDSWISH_OPERATION_ID,
        CanonicalReference::Function(TORCH_NN_HARDSWISH_REFERENCE),
    ),
    (
        TORCH_NN_SELU_OPERATION_ID,
        CanonicalReference::Function(TORCH_NN_SELU_REFERENCE),
    ),
    (
        XPU_TOTAL_MEMORY_OPERATION_ID,
        CanonicalReference::DeviceProperty(XPU_TOTAL_MEMORY_REFERENCE),
    ),
    (
        INTERPOLATION_NEAREST_OPERATION_ID,
        CanonicalReference::EnumVariant(INTERPOLATION_NEAREST_REFERENCE),
    ),
    (
        FUNCTIONAL_INTERPOLATION_BICUBIC_OPERATION_ID,
        CanonicalReference::EnumVariant(FUNCTIONAL_INTERPOLATION_BICUBIC_REFERENCE),
    ),
    (
        XFORMERS_MODULE_VERSION_OPERATION_ID,
        CanonicalReference::VersionValue(XFORMERS_MODULE_VERSION_REFERENCE),
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

pub fn cudnn_benchmark_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDNN_BENCHMARK_OPERATION_ID)
}
pub fn torch_complex64_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_COMPLEX64_OPERATION_ID)
}
pub fn torch_float_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT_OPERATION_ID)
}
pub fn torch_float8_e4m3fn_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT8_E4M3FN_OPERATION_ID)
}
pub fn torch_float8_e5m2fnuz_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT8_E5M2FNUZ_OPERATION_ID)
}
pub fn torch_int16_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_INT16_OPERATION_ID)
}
pub fn torch_nn_hardswish_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_NN_HARDSWISH_OPERATION_ID)
}
pub fn torch_nn_selu_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_NN_SELU_OPERATION_ID)
}
pub fn xpu_total_memory_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(XPU_TOTAL_MEMORY_OPERATION_ID)
}
pub fn interpolation_nearest_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(INTERPOLATION_NEAREST_OPERATION_ID)
}
pub fn functional_interpolation_bicubic_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(FUNCTIONAL_INTERPOLATION_BICUBIC_OPERATION_ID)
}
pub fn xformers_module_version_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(XFORMERS_MODULE_VERSION_OPERATION_ID)
}
