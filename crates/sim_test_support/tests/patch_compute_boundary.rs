use std::{collections::BTreeMap, error::Error};

use sim_model::{
    MappedModelWeights, PatchComputeBoundary, PatchGraph, PatchPayload, PatchTensor,
    PatchValueTransform, SemanticPatchOperation,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, StreamId,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_backend_exact_native, tensor_to_f32_with_backend_exact_native,
    },
};
use sim_worker::{
    EffectiveMemoryMode, MemoryModeCapabilities, MemoryModeRequest, ModelResidencyMode,
};

const MIB: u64 = 1024 * 1024;

fn patch_compute_boundary(
    mode: EffectiveMemoryMode,
    lora_compute_dtype: DType,
) -> Result<PatchComputeBoundary, sim_model::PatchGraphError> {
    if mode.patch_uses_weight_dtype() {
        Ok(PatchComputeBoundary::weight_dtype())
    } else {
        PatchComputeBoundary::configured(lora_compute_dtype)
    }
}

#[test]
fn production_memory_mode_adapter_drives_ordered_bf16_patch_compute() -> Result<(), Box<dyn Error>>
{
    const BASE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * MIB)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(MIB)?,
        &cancellation,
    );
    let source_tensor = tensor_from_f32_with_backend_exact_native(
        &backend,
        &[2, 2],
        &[0.0; 4],
        DType::Bf16,
        DeviceId::CPU,
        &context,
    )?;
    let source = MappedModelWeights::from_test_parts(
        BASE_DIGEST.to_owned(),
        BTreeMap::from([("weight".to_owned(), source_tensor)]),
        Vec::new(),
    )?;
    let patch_tensor =
        |shape: &[u64], values: &[f32]| PatchTensor::checked(shape.to_vec(), values.to_vec());
    let operation = |identifier: &str, payload: PatchPayload| SemanticPatchOperation {
        identifier: identifier.to_owned(),
        target_key: "weight".to_owned(),
        expected_shape: vec![2, 2],
        strength: 1.0,
        strength_model: 1.0,
        slices: Vec::new(),
        transform: PatchValueTransform::default(),
        payload,
    };
    let nonlinear = operation(
        "nonlinear-bf16",
        PatchPayload::Loha {
            first_up: patch_tensor(&[2, 2], &[1.00390625, 0.0, 0.0, 1.01171875])?,
            first_down: patch_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?,
            second_up: patch_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?,
            second_down: patch_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?,
            first_tucker: None,
            second_tucker: None,
            alpha: Some(2.0),
            dora_scale: None,
        },
    );
    let repeated_target = operation(
        "ordered-bf16-followup",
        PatchPayload::DenseDiff {
            tensor: patch_tensor(&[2, 2], &[0.00390625, 0.0, 0.0, -0.00390625])?,
            pad_weight: false,
        },
    );
    let graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![nonlinear, repeated_target])?;
    let normal_mode = EffectiveMemoryMode::resolve(
        MemoryModeRequest::default(),
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    let low_vram_mode = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::LowVram,
            ..MemoryModeRequest::default()
        },
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    let normal_boundary = patch_compute_boundary(normal_mode, DType::F32)?;
    let normal_f16_boundary = patch_compute_boundary(normal_mode, DType::F16)?;
    let low_vram_boundary = patch_compute_boundary(low_vram_mode, DType::F32)?;
    let normal = graph.apply_with_compute_boundary(&backend, &source, normal_boundary, &context)?;
    let normal_f16 =
        graph.apply_with_compute_boundary(&backend, &source, normal_f16_boundary, &context)?;
    let low_vram =
        graph.apply_with_compute_boundary(&backend, &source, low_vram_boundary, &context)?;
    let weight_bytes = |weights: &MappedModelWeights| -> Result<Vec<u8>, std::io::Error> {
        weights
            .tensors()
            .get("weight")
            .ok_or_else(|| std::io::Error::other("fixture weight missing"))?
            .contiguous_bytes()
            .map(<[u8]>::to_vec)
            .map_err(std::io::Error::other)
    };
    assert_ne!(weight_bytes(&normal)?, weight_bytes(&low_vram)?);
    assert_ne!(normal.cache_identity(), low_vram.cache_identity());
    assert_ne!(normal.cache_identity(), normal_f16.cache_identity());
    assert_ne!(normal_f16.cache_identity(), low_vram.cache_identity());
    assert_eq!(
        weight_bytes(&normal_f16)?,
        vec![129, 63, 0, 0, 0, 0, 129, 63]
    );
    assert_eq!(
        normal_f16.cache_identity(),
        "014be02a973f058088cbbeb121d0536c01c29fda44217bd252a771bff7f6a2f9"
    );
    assert_eq!(
        normal_f16
            .tensors()
            .get("weight")
            .ok_or_else(|| std::io::Error::other("fixture weight missing"))?
            .descriptor()
            .dtype(),
        DType::Bf16
    );

    let repeated_normal =
        graph.apply_with_compute_boundary(&backend, &source, normal_boundary, &context)?;
    let repeated_normal_f16 =
        graph.apply_with_compute_boundary(&backend, &source, normal_f16_boundary, &context)?;
    let repeated_low_vram =
        graph.apply_with_compute_boundary(&backend, &source, low_vram_boundary, &context)?;
    assert_eq!(normal.cache_identity(), repeated_normal.cache_identity());
    assert_eq!(
        normal_f16.cache_identity(),
        repeated_normal_f16.cache_identity()
    );
    assert_eq!(
        low_vram.cache_identity(),
        repeated_low_vram.cache_identity()
    );
    assert_eq!(weight_bytes(&normal)?, weight_bytes(&repeated_normal)?);
    assert_eq!(
        weight_bytes(&normal_f16)?,
        weight_bytes(&repeated_normal_f16)?
    );
    assert_eq!(weight_bytes(&low_vram)?, weight_bytes(&repeated_low_vram)?);

    let fabricated_source_tensor = tensor_from_f32_with_backend_exact_native(
        &backend,
        &[2, 2],
        &[0.0; 4],
        DType::F32,
        DeviceId::CPU,
        &context,
    )?;
    let fabricated_source = MappedModelWeights::from_test_parts(
        BASE_DIGEST.to_owned(),
        BTreeMap::from([("weight".to_owned(), fabricated_source_tensor)]),
        Vec::new(),
    )?;
    let fabricated_f32 = graph.apply_with_compute_boundary(
        &backend,
        &fabricated_source,
        normal_boundary,
        &context,
    )?;
    let fabricated_values = tensor_to_f32_with_backend_exact_native(
        &backend,
        fabricated_f32
            .tensors()
            .get("weight")
            .ok_or_else(|| std::io::Error::other("fabricated fixture weight missing"))?,
        &context,
    )?;
    let fabricated_first = fabricated_values
        .first()
        .copied()
        .ok_or_else(|| std::io::Error::other("fabricated fixture output is empty"))?;
    let fabricated_scalar = tensor_from_f32_with_backend_exact_native(
        &backend,
        &[1],
        &[fabricated_first],
        DType::Bf16,
        DeviceId::CPU,
        &context,
    )?;
    let fabricated_scalar =
        tensor_to_f32_with_backend_exact_native(&backend, &fabricated_scalar, &context)?;
    let low_vram_values = tensor_to_f32_with_backend_exact_native(
        &backend,
        low_vram
            .tensors()
            .get("weight")
            .ok_or_else(|| std::io::Error::other("LowVram fixture weight missing"))?,
        &context,
    )?;
    assert_ne!(fabricated_scalar.first(), low_vram_values.first());
    Ok(())
}

#[test]
fn canonical_model_boundary_alone_rejects_invalid_configured_dtypes() -> Result<(), Box<dyn Error>>
{
    let normal = EffectiveMemoryMode::resolve(
        MemoryModeRequest::default(),
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    let low_vram = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::LowVram,
            ..MemoryModeRequest::default()
        },
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    assert_eq!(
        patch_compute_boundary(normal, DType::F32)?,
        PatchComputeBoundary::Configured(DType::F32)
    );
    assert!(patch_compute_boundary(normal, DType::Bf16).is_err());
    assert!(patch_compute_boundary(normal, DType::U8).is_err());
    assert_eq!(
        patch_compute_boundary(low_vram, DType::U8)?,
        PatchComputeBoundary::WeightDType
    );
    Ok(())
}
