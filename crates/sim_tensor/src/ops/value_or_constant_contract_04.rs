use crate::{
    BooleanCapabilityReference, CanonicalReference, ContractInventoryKind, DType,
    EnumVariantReference, FunctionReference, Layout, MemoryFormatReference,
    NumericConstantReference, OPERATION_CONTRACTS, TypeMarkerReference, TypedReferenceContract,
};

pub const CUDA_MATMUL_ALLOW_TF32_OPERATION_ID: &str = "COMFY-TENSOR-OP-AF01B777BDC3";
pub const CUDNN_ALLOW_TF32_OPERATION_ID: &str = "COMFY-TENSOR-OP-95755CF02E17";
pub const CUDNN_ENABLED_OPERATION_ID: &str = "COMFY-TENSOR-OP-8A525D4E1849";
pub const CHANNELS_LAST_OPERATION_ID: &str = "COMFY-TENSOR-OP-A19697FC37D3";
pub const CUDA_OUT_OF_MEMORY_ERROR_OPERATION_ID: &str = "COMFY-TENSOR-OP-AC05E270104F";
pub const FLOAT_INFO_BITS_OPERATION_ID: &str = "COMFY-TENSOR-OP-ADEAE4C0A95C";
pub const FLOAT_INFO_MAXIMUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-998DEB98A475";
pub const TORCH_FLOAT32_OPERATION_ID: &str = "COMFY-TENSOR-OP-AD88DF0B8B9D";
pub const TORCH_NN_HARDTANH_OPERATION_ID: &str = "COMFY-TENSOR-OP-A385533B7CE3";
pub const SDP_CUDNN_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-95733C16308E";
pub const SDP_EFFICIENT_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-92B804E99EF9";
pub const SDP_MATH_OPERATION_ID: &str = "COMFY-TENSOR-OP-A23E9664DC07";

pub const CUDA_MATMUL_ALLOW_TF32_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::CudaMatmulAllowTf32;
pub const CUDNN_ALLOW_TF32_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::CudnnAllowTf32;
pub const CUDNN_ENABLED_REFERENCE: BooleanCapabilityReference =
    BooleanCapabilityReference::CudnnEnabled;
pub const CHANNELS_LAST_REFERENCE: MemoryFormatReference =
    MemoryFormatReference::Layout(Layout::ChannelsLast);
pub const CUDA_OUT_OF_MEMORY_ERROR_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::CudaOutOfMemoryError;
pub const FLOAT_INFO_BITS_REFERENCE: NumericConstantReference =
    NumericConstantReference::FloatInfoBits;
pub const FLOAT_INFO_MAXIMUM_REFERENCE: NumericConstantReference =
    NumericConstantReference::FloatInfoMaximum;
pub const TORCH_FLOAT32_REFERENCE: DType = DType::F32;
pub const TORCH_NN_HARDTANH_REFERENCE: FunctionReference = FunctionReference::Hardtanh;
pub const SDP_CUDNN_ATTENTION_REFERENCE: EnumVariantReference =
    EnumVariantReference::SdpCudnnAttention;
pub const SDP_EFFICIENT_ATTENTION_REFERENCE: EnumVariantReference =
    EnumVariantReference::SdpEfficientAttention;
pub const SDP_MATH_REFERENCE: EnumVariantReference = EnumVariantReference::SdpMath;

pub const ASSIGNED_VALUE_OR_CONSTANT_REFERENCES: &[(&str, CanonicalReference)] = &[
    (
        CUDA_MATMUL_ALLOW_TF32_OPERATION_ID,
        CanonicalReference::BooleanCapability(CUDA_MATMUL_ALLOW_TF32_REFERENCE),
    ),
    (
        CUDNN_ALLOW_TF32_OPERATION_ID,
        CanonicalReference::BooleanCapability(CUDNN_ALLOW_TF32_REFERENCE),
    ),
    (
        CUDNN_ENABLED_OPERATION_ID,
        CanonicalReference::BooleanCapability(CUDNN_ENABLED_REFERENCE),
    ),
    (
        CHANNELS_LAST_OPERATION_ID,
        CanonicalReference::MemoryFormat(CHANNELS_LAST_REFERENCE),
    ),
    (
        CUDA_OUT_OF_MEMORY_ERROR_OPERATION_ID,
        CanonicalReference::TypeMarker(CUDA_OUT_OF_MEMORY_ERROR_REFERENCE),
    ),
    (
        FLOAT_INFO_BITS_OPERATION_ID,
        CanonicalReference::NumericConstant(FLOAT_INFO_BITS_REFERENCE),
    ),
    (
        FLOAT_INFO_MAXIMUM_OPERATION_ID,
        CanonicalReference::NumericConstant(FLOAT_INFO_MAXIMUM_REFERENCE),
    ),
    (
        TORCH_FLOAT32_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT32_REFERENCE),
    ),
    (
        TORCH_NN_HARDTANH_OPERATION_ID,
        CanonicalReference::Function(TORCH_NN_HARDTANH_REFERENCE),
    ),
    (
        SDP_CUDNN_ATTENTION_OPERATION_ID,
        CanonicalReference::EnumVariant(SDP_CUDNN_ATTENTION_REFERENCE),
    ),
    (
        SDP_EFFICIENT_ATTENTION_OPERATION_ID,
        CanonicalReference::EnumVariant(SDP_EFFICIENT_ATTENTION_REFERENCE),
    ),
    (
        SDP_MATH_OPERATION_ID,
        CanonicalReference::EnumVariant(SDP_MATH_REFERENCE),
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

pub fn cuda_matmul_allow_tf32_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDA_MATMUL_ALLOW_TF32_OPERATION_ID)
}

pub fn cudnn_allow_tf32_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDNN_ALLOW_TF32_OPERATION_ID)
}

pub fn cudnn_enabled_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDNN_ENABLED_OPERATION_ID)
}

pub fn channels_last_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CHANNELS_LAST_OPERATION_ID)
}

pub fn cuda_out_of_memory_error_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDA_OUT_OF_MEMORY_ERROR_OPERATION_ID)
}

pub fn float_info_bits_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(FLOAT_INFO_BITS_OPERATION_ID)
}

pub fn float_info_maximum_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(FLOAT_INFO_MAXIMUM_OPERATION_ID)
}

pub fn torch_float32_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT32_OPERATION_ID)
}

pub fn torch_nn_hardtanh_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_NN_HARDTANH_OPERATION_ID)
}

pub fn sdp_cudnn_attention_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(SDP_CUDNN_ATTENTION_OPERATION_ID)
}

pub fn sdp_efficient_attention_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(SDP_EFFICIENT_ATTENTION_OPERATION_ID)
}

pub fn sdp_math_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(SDP_MATH_OPERATION_ID)
}
