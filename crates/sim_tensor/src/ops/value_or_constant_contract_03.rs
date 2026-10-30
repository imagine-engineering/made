use crate::{
    CanonicalReference, ContractInventoryKind, DType, DevicePropertyReference,
    EnumVariantReference, FunctionReference, NumericConstantReference, OPERATION_CONTRACTS,
    TensorPropertyReference, TypeMarkerReference, TypedReferenceContract, VersionValueReference,
};

pub const ACCELERATOR_ERROR_OPERATION_ID: &str = "COMFY-TENSOR-OP-69B5DAB42F01";
pub const AUTOGRAD_ONCE_DIFFERENTIABLE_OPERATION_ID: &str = "COMFY-TENSOR-OP-884EF2E5681D";
pub const CUDA_GCN_ARCHITECTURE_NAME_OPERATION_ID: &str = "COMFY-TENSOR-OP-7D161437B5F7";
pub const FLOAT_INFO_MINIMUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-77DB8879A02F";
pub const TORCH_FLOAT16_OPERATION_ID: &str = "COMFY-TENSOR-OP-6542124FE760";
pub const TORCH_FLOAT8_E4M3FNUZ_OPERATION_ID: &str = "COMFY-TENSOR-OP-75D5287B779B";
pub const TORCH_FLOAT8_E5M2_OPERATION_ID: &str = "COMFY-TENSOR-OP-71F49361E719";
pub const TORCH_INFINITY_OPERATION_ID: &str = "COMFY-TENSOR-OP-73718661D305";
pub const MEDIAN_VALUES_OPERATION_ID: &str = "COMFY-TENSOR-OP-7775E33D5750";
pub const SDP_FLASH_ATTENTION_OPERATION_ID: &str = "COMFY-TENSOR-OP-881B897E511D";
pub const TORCH_UINT64_OPERATION_ID: &str = "COMFY-TENSOR-OP-67AAEEB293AD";
pub const TORCH_VERSION_OPERATION_ID: &str = "COMFY-TENSOR-OP-83FC32D08FD1";

pub const ACCELERATOR_ERROR_REFERENCE: TypeMarkerReference = TypeMarkerReference::AcceleratorError;
pub const AUTOGRAD_ONCE_DIFFERENTIABLE_REFERENCE: FunctionReference =
    FunctionReference::AutogradOnceDifferentiable;
pub const CUDA_GCN_ARCHITECTURE_NAME_REFERENCE: DevicePropertyReference =
    DevicePropertyReference::CudaGcnArchitectureName;
pub const FLOAT_INFO_MINIMUM_REFERENCE: NumericConstantReference =
    NumericConstantReference::FloatInfoMinimum;
pub const TORCH_FLOAT16_REFERENCE: DType = DType::F16;
pub const TORCH_FLOAT8_E4M3FNUZ_REFERENCE: DType = DType::Float8E4m3Fnuz;
pub const TORCH_FLOAT8_E5M2_REFERENCE: DType = DType::Float8E5m2;
pub const TORCH_INFINITY_REFERENCE: NumericConstantReference = NumericConstantReference::Infinity;
pub const MEDIAN_VALUES_REFERENCE: TensorPropertyReference = TensorPropertyReference::MedianValues;
pub const SDP_FLASH_ATTENTION_REFERENCE: EnumVariantReference =
    EnumVariantReference::SdpFlashAttention;
pub const TORCH_UINT64_REFERENCE: DType = DType::U64;
pub const TORCH_VERSION_REFERENCE: VersionValueReference = VersionValueReference::Torch;

pub const ASSIGNED_VALUE_OR_CONSTANT_REFERENCES: &[(&str, CanonicalReference)] = &[
    (
        ACCELERATOR_ERROR_OPERATION_ID,
        CanonicalReference::TypeMarker(ACCELERATOR_ERROR_REFERENCE),
    ),
    (
        AUTOGRAD_ONCE_DIFFERENTIABLE_OPERATION_ID,
        CanonicalReference::Function(AUTOGRAD_ONCE_DIFFERENTIABLE_REFERENCE),
    ),
    (
        CUDA_GCN_ARCHITECTURE_NAME_OPERATION_ID,
        CanonicalReference::DeviceProperty(CUDA_GCN_ARCHITECTURE_NAME_REFERENCE),
    ),
    (
        FLOAT_INFO_MINIMUM_OPERATION_ID,
        CanonicalReference::NumericConstant(FLOAT_INFO_MINIMUM_REFERENCE),
    ),
    (
        TORCH_FLOAT16_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT16_REFERENCE),
    ),
    (
        TORCH_FLOAT8_E4M3FNUZ_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT8_E4M3FNUZ_REFERENCE),
    ),
    (
        TORCH_FLOAT8_E5M2_OPERATION_ID,
        CanonicalReference::DType(TORCH_FLOAT8_E5M2_REFERENCE),
    ),
    (
        TORCH_INFINITY_OPERATION_ID,
        CanonicalReference::NumericConstant(TORCH_INFINITY_REFERENCE),
    ),
    (
        MEDIAN_VALUES_OPERATION_ID,
        CanonicalReference::TensorProperty(MEDIAN_VALUES_REFERENCE),
    ),
    (
        SDP_FLASH_ATTENTION_OPERATION_ID,
        CanonicalReference::EnumVariant(SDP_FLASH_ATTENTION_REFERENCE),
    ),
    (
        TORCH_UINT64_OPERATION_ID,
        CanonicalReference::DType(TORCH_UINT64_REFERENCE),
    ),
    (
        TORCH_VERSION_OPERATION_ID,
        CanonicalReference::VersionValue(TORCH_VERSION_REFERENCE),
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

pub fn accelerator_error_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(ACCELERATOR_ERROR_OPERATION_ID)
}

pub fn autograd_once_differentiable_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(AUTOGRAD_ONCE_DIFFERENTIABLE_OPERATION_ID)
}

pub fn cuda_gcn_architecture_name_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(CUDA_GCN_ARCHITECTURE_NAME_OPERATION_ID)
}

pub fn float_info_minimum_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(FLOAT_INFO_MINIMUM_OPERATION_ID)
}

pub fn torch_float16_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT16_OPERATION_ID)
}

pub fn torch_float8_e4m3fnuz_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT8_E4M3FNUZ_OPERATION_ID)
}

pub fn torch_float8_e5m2_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_FLOAT8_E5M2_OPERATION_ID)
}

pub fn torch_infinity_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_INFINITY_OPERATION_ID)
}

pub fn median_values_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(MEDIAN_VALUES_OPERATION_ID)
}

pub fn sdp_flash_attention_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(SDP_FLASH_ATTENTION_OPERATION_ID)
}

pub fn torch_uint64_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_UINT64_OPERATION_ID)
}

pub fn torch_version_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_VERSION_OPERATION_ID)
}
