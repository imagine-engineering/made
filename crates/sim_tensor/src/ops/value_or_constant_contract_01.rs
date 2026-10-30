use crate::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    FunctionReference, NumericConstantReference, OPERATION_CONTRACTS, TensorPropertyReference,
    TypedReferenceContract,
};

pub const CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_OPERATION_ID: &str = "COMFY-TENSOR-OP-346EEAEC4F8E";
pub const INVERSE_FFT_REAL_OPERATION_ID: &str = "COMFY-TENSOR-OP-2737FC6A68CF";
pub const TORCH_INT_OPERATION_ID: &str = "COMFY-TENSOR-OP-20F701F33DAC";
pub const TORCH_INT64_OPERATION_ID: &str = "COMFY-TENSOR-OP-3B92F5DD9D3B";
pub const TORCH_LOG10_OPERATION_ID: &str = "COMFY-TENSOR-OP-201D1047CF9B";
pub const TORCH_LONG_OPERATION_ID: &str = "COMFY-TENSOR-OP-1C1307AB38E7";
pub const TORCH_NN_MISH_OPERATION_ID: &str = "COMFY-TENSOR-OP-1E672554B9EE";
pub const TORCH_PI_OPERATION_ID: &str = "COMFY-TENSOR-OP-1429830307D9";
pub const TORCH_UINT32_OPERATION_ID: &str = "COMFY-TENSOR-OP-211859D17671";
pub const UNIQUE_SHAPE_OPERATION_ID: &str = "COMFY-TENSOR-OP-137DED7F8918";
pub const VANDERMONDE_TRANSPOSE_OPERATION_ID: &str = "COMFY-TENSOR-OP-323AE28F5D91";
pub const XPU_STREAM_OPERATION_ID: &str = "COMFY-TENSOR-OP-2982F632EBD3";

pub const CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::CudaMatmulAllowFp16Accumulation;
pub const INVERSE_FFT_REAL_REFERENCE: TensorPropertyReference =
    TensorPropertyReference::InverseFftReal;
pub const TORCH_INT_REFERENCE: DType = DType::I32;
pub const TORCH_INT64_REFERENCE: DType = DType::I64;
pub const TORCH_LOG10_REFERENCE: FunctionReference = FunctionReference::Log10;
pub const TORCH_LONG_REFERENCE: DType = DType::I64;
pub const TORCH_NN_MISH_REFERENCE: FunctionReference = FunctionReference::Mish;
pub const TORCH_PI_REFERENCE: NumericConstantReference = NumericConstantReference::Pi;
pub const TORCH_UINT32_REFERENCE: DType = DType::U32;
pub const UNIQUE_SHAPE_REFERENCE: TensorPropertyReference = TensorPropertyReference::UniqueShape;
pub const VANDERMONDE_TRANSPOSE_REFERENCE: TensorPropertyReference =
    TensorPropertyReference::VandermondeTranspose;
pub const XPU_STREAM_REFERENCE: FunctionReference = FunctionReference::XpuStream;

pub const ASSIGNED_VALUE_OR_CONSTANT_REFERENCES: &[(&str, CanonicalReference)] = &[
    (
        CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_OPERATION_ID,
        CanonicalReference::BooleanCapability(CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_REFERENCE),
    ),
    (
        INVERSE_FFT_REAL_OPERATION_ID,
        CanonicalReference::TensorProperty(INVERSE_FFT_REAL_REFERENCE),
    ),
    (
        TORCH_INT_OPERATION_ID,
        CanonicalReference::DType(TORCH_INT_REFERENCE),
    ),
    (
        TORCH_INT64_OPERATION_ID,
        CanonicalReference::DType(TORCH_INT64_REFERENCE),
    ),
    (
        TORCH_LOG10_OPERATION_ID,
        CanonicalReference::Function(TORCH_LOG10_REFERENCE),
    ),
    (
        TORCH_LONG_OPERATION_ID,
        CanonicalReference::DType(TORCH_LONG_REFERENCE),
    ),
    (
        TORCH_NN_MISH_OPERATION_ID,
        CanonicalReference::Function(TORCH_NN_MISH_REFERENCE),
    ),
    (
        TORCH_PI_OPERATION_ID,
        CanonicalReference::NumericConstant(TORCH_PI_REFERENCE),
    ),
    (
        TORCH_UINT32_OPERATION_ID,
        CanonicalReference::DType(TORCH_UINT32_REFERENCE),
    ),
    (
        UNIQUE_SHAPE_OPERATION_ID,
        CanonicalReference::TensorProperty(UNIQUE_SHAPE_REFERENCE),
    ),
    (
        VANDERMONDE_TRANSPOSE_OPERATION_ID,
        CanonicalReference::TensorProperty(VANDERMONDE_TRANSPOSE_REFERENCE),
    ),
    (
        XPU_STREAM_OPERATION_ID,
        CanonicalReference::Function(XPU_STREAM_REFERENCE),
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

pub fn cuda_matmul_allow_fp16_accumulation_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDA_MATMUL_ALLOW_FP16_ACCUMULATION_OPERATION_ID)
}

pub fn inverse_fft_real_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(INVERSE_FFT_REAL_OPERATION_ID)
}

pub fn torch_int_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_INT_OPERATION_ID)
}

pub fn torch_int64_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_INT64_OPERATION_ID)
}

pub fn torch_log10_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_LOG10_OPERATION_ID)
}

pub fn torch_long_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_LONG_OPERATION_ID)
}

pub fn torch_nn_mish_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_NN_MISH_OPERATION_ID)
}

pub fn torch_pi_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_PI_OPERATION_ID)
}

pub fn torch_uint32_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_UINT32_OPERATION_ID)
}

pub fn unique_shape_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(UNIQUE_SHAPE_OPERATION_ID)
}

pub fn vandermonde_transpose_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(VANDERMONDE_TRANSPOSE_OPERATION_ID)
}

pub fn xpu_stream_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(XPU_STREAM_OPERATION_ID)
}
