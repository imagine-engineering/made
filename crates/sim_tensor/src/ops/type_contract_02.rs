use crate::{
    CanonicalReference, ContractInventoryKind, OPERATION_CONTRACTS, TypeMarkerReference,
    TypedReferenceContract,
};

pub const SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-CA96BCF6B334";
pub const SIM_MANUAL_CAST_OPERATION_ID: &str = "COMFY-TENSOR-OP-BA6AE52D4258";
pub const TORCH_RMS_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-FD4EE56E61FC";

pub const SIM_DISABLE_WEIGHT_INIT_RMS_NORM_REFERENCE: TypeMarkerReference =
    TypeMarkerReference::ComfyDisableWeightInitRmsNorm;
pub const SIM_MANUAL_CAST_REFERENCE: TypeMarkerReference = TypeMarkerReference::ComfyManualCast;
pub const TORCH_RMS_NORM_REFERENCE: TypeMarkerReference = TypeMarkerReference::RmsNorm;

pub const ASSIGNED_TYPE_REFERENCES: &[(&str, TypeMarkerReference)] = &[
    (
        SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID,
        SIM_DISABLE_WEIGHT_INIT_RMS_NORM_REFERENCE,
    ),
    (SIM_MANUAL_CAST_OPERATION_ID, SIM_MANUAL_CAST_REFERENCE),
    (TORCH_RMS_NORM_OPERATION_ID, TORCH_RMS_NORM_REFERENCE),
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

pub fn sim_disable_weight_init_rms_norm_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(SIM_DISABLE_WEIGHT_INIT_RMS_NORM_OPERATION_ID)
}

pub fn sim_manual_cast_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(SIM_MANUAL_CAST_OPERATION_ID)
}

pub fn torch_rms_norm_contract() -> Option<TypedReferenceContract> {
    assigned_type_contract(TORCH_RMS_NORM_OPERATION_ID)
}
