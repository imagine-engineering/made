use crate::{
    CanonicalReference, ContractInventoryKind, DType, MemoryFormatReference, NamespaceReference,
    OPERATION_CONTRACTS, TypedReferenceContract,
};

pub const TORCH_PACKAGE_PATH_OPERATION_ID: &str = "COMFY-TENSOR-OP-EC92FAA68139";
pub const TORCH_BFLOAT16_OPERATION_ID: &str = "COMFY-TENSOR-OP-EA594D0A5B7F";
pub const TORCH_PRESERVE_FORMAT_OPERATION_ID: &str = "COMFY-TENSOR-OP-E7D1824E03F0";
pub const TORCH_UINT16_OPERATION_ID: &str = "COMFY-TENSOR-OP-E43E68AB67D6";

pub const TORCH_PACKAGE_PATH_REFERENCE: NamespaceReference = NamespaceReference::TorchPackagePath;
pub const TORCH_BFLOAT16_REFERENCE: DType = DType::Bf16;
pub const TORCH_PRESERVE_FORMAT_REFERENCE: MemoryFormatReference =
    MemoryFormatReference::PreserveFormat;
pub const TORCH_UINT16_REFERENCE: DType = DType::U16;

pub const ASSIGNED_VALUE_OR_CONSTANT_REFERENCES: &[(&str, CanonicalReference)] = &[
    (
        TORCH_PACKAGE_PATH_OPERATION_ID,
        CanonicalReference::Namespace(TORCH_PACKAGE_PATH_REFERENCE),
    ),
    (
        TORCH_BFLOAT16_OPERATION_ID,
        CanonicalReference::DType(TORCH_BFLOAT16_REFERENCE),
    ),
    (
        TORCH_PRESERVE_FORMAT_OPERATION_ID,
        CanonicalReference::MemoryFormat(TORCH_PRESERVE_FORMAT_REFERENCE),
    ),
    (
        TORCH_UINT16_OPERATION_ID,
        CanonicalReference::DType(TORCH_UINT16_REFERENCE),
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

pub fn torch_package_path_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_PACKAGE_PATH_OPERATION_ID)
}

pub fn torch_bfloat16_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_BFLOAT16_OPERATION_ID)
}

pub fn torch_preserve_format_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_PRESERVE_FORMAT_OPERATION_ID)
}

pub fn torch_uint16_contract() -> Option<TypedReferenceContract> {
    assigned_value_or_constant_contract(TORCH_UINT16_OPERATION_ID)
}
