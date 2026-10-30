use sha2::{Digest, Sha256};
use sim_model::{
    AdapterTensor, NativeModelBackingKind, NativeModelPayload, NativeModelPayloadError,
    NativeModelResourceRole, NativeVae, PatchGraph, PatchPayload,
    clip::NativeClipResource,
    conditioning::{
        ConditioningError, ConditioningIdentity, ConditioningMetadataKey, ConditioningMetadataMap,
        ConditioningMetadataValue, ConditioningWindow, ResolvedConditioningEntry,
    },
    controlnet::{
        ControlChain, ControlConditioning, ControlIsolation, ControlModelExecutor, ControlNetError,
        ControlNetRebindRequest, ControlResult, ControlRuntime, ControlTensorBinding,
    },
    generated_native_diffusion::{
        Sd15TinyModel, sd15_latent_format_identity, sd15_model_family_identity,
    },
    patches::{
        ModelStateExportKind, PatchKeyMap, PatchMergeRule, StateKeySelector,
        export_safetensors_exact, export_safetensors_exact_with_optional_metadata,
        load_lora_patch_graph,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, ExecutionContext, NativeLatentBundle, NativeLatentBundleError,
    NativeLatentMetadataRetention, NativeLatentNoiseMask, NativeLatentSamples, Tensor,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    sync::Arc,
};
use thiserror::Error;

use crate::GUIDANCE_ADAPTER_ID;
use crate::{
    ExecutionHookGraph, NativeGuidanceModifierChain, NativeSamplingPolicy, profile_for_model,
};
use crate::{NativeNoiseGeneration, NativeNoisePayload, NativeSamplerPayloadError, NoiseRequest};

#[derive(Clone, Debug)]
pub enum NativeLatentNoiseGeneration {
    Tensor(NativeNoiseGeneration),
    AudioVideo {
        video: NativeNoiseGeneration,
        audio: NativeNoiseGeneration,
    },
}

#[derive(Clone, Debug)]
pub struct NativeLatentSamplingInput {
    bundle: NativeLatentBundle,
}

impl NativeLatentSamplingInput {
    pub fn checked_for_model(
        bundle: &NativeLatentBundle,
        model: &NativeModelPayload,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        bundle.validate(context)?;
        let NativeLatentSamples::Tensor(samples) = bundle.samples() else {
            return Self::checked(bundle, context);
        };
        let identity = match model.native_family_model_resource() {
            Some(resource) => resource.conditioning_identity().latent_format().clone(),
            None => sd15_latent_format_identity()
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
        };
        let registry =
            sim_model::LatentFormatRegistry::checked(sim_model::GENERATED_LATENT_FORMATS)?;
        let definition = registry.get(&identity).ok_or_else(|| {
            NativeDiffusionPayloadError::Invalid("model latent format is unavailable".to_owned())
        })?;
        let samples = sim_model::latent_format::normalize_sampling_latent(
            definition,
            backend,
            samples,
            bundle.metadata().spatial_downscale_ratio(),
            bundle.metadata().temporal_downscale_ratio(),
            context,
        )?;
        let bundle = bundle.replaced_samples(
            NativeLatentSamples::Tensor(samples),
            NativeLatentMetadataRetention::Preserve,
            context,
        )?;
        Ok(Self { bundle })
    }

    pub fn checked(
        bundle: &NativeLatentBundle,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        bundle.validate(context)?;
        Ok(Self {
            bundle: bundle.clone(),
        })
    }

    pub fn samples(&self) -> &NativeLatentSamples {
        self.bundle.samples()
    }

    pub fn noise_mask(&self) -> Option<&NativeLatentNoiseMask> {
        self.bundle.noise_mask()
    }

    pub fn batch_indices(&self) -> Option<&[u64]> {
        self.bundle.batch_indices()
    }

    pub fn generate_noise(
        &self,
        noise: &NativeNoisePayload,
        backend: &CpuBackend,
        request: &NoiseRequest,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentNoiseGeneration, NativeDiffusionPayloadError> {
        context.check()?;
        Ok(match self.bundle.samples() {
            NativeLatentSamples::Tensor(samples) => {
                NativeLatentNoiseGeneration::Tensor(noise.generate(
                    backend,
                    samples,
                    self.bundle.batch_indices(),
                    request,
                    context,
                )?)
            }
            NativeLatentSamples::AudioVideo { video, audio } => {
                NativeLatentNoiseGeneration::AudioVideo {
                    video: noise.generate(
                        backend,
                        video,
                        self.bundle.batch_indices(),
                        request,
                        context,
                    )?,
                    audio: noise.generate(
                        backend,
                        audio,
                        self.bundle.batch_indices(),
                        request,
                        context,
                    )?,
                }
            }
        })
    }

    pub fn masked_model_input(
        &self,
        current: &NativeLatentSamples,
        noise: &NativeLatentNoiseGeneration,
        sigma: f32,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentSamples, NativeDiffusionPayloadError> {
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(NativeDiffusionPayloadError::Invalid(
                "native latent sampling sigma must be finite and nonnegative".to_owned(),
            ));
        }
        match (
            self.bundle.samples(),
            current,
            noise,
            self.bundle.noise_mask(),
        ) {
            (
                NativeLatentSamples::Tensor(original),
                NativeLatentSamples::Tensor(current),
                NativeLatentNoiseGeneration::Tensor(noise),
                mask,
            ) => Ok(NativeLatentSamples::Tensor(masked_model_component(
                original,
                current,
                &noise.noise,
                mask.and_then(NativeLatentNoiseMask::tensor),
                sigma,
                backend,
                context,
            )?)),
            (
                NativeLatentSamples::AudioVideo {
                    video: original_video,
                    audio: original_audio,
                },
                NativeLatentSamples::AudioVideo {
                    video: current_video,
                    audio: current_audio,
                },
                NativeLatentNoiseGeneration::AudioVideo {
                    video: video_noise,
                    audio: audio_noise,
                },
                mask,
            ) => {
                let (video_mask, audio_mask) = match mask {
                    Some(NativeLatentNoiseMask::AudioVideo { video, audio }) => {
                        (Some(video), Some(audio))
                    }
                    None => (None, None),
                    Some(NativeLatentNoiseMask::Tensor(_)) => {
                        return Err(NativeDiffusionPayloadError::LatentStructureMismatch);
                    }
                };
                Ok(NativeLatentSamples::AudioVideo {
                    video: masked_model_component(
                        original_video,
                        current_video,
                        &video_noise.noise,
                        video_mask,
                        sigma,
                        backend,
                        context,
                    )?,
                    audio: masked_model_component(
                        original_audio,
                        current_audio,
                        &audio_noise.noise,
                        audio_mask,
                        sigma,
                        backend,
                        context,
                    )?,
                })
            }
            _ => Err(NativeDiffusionPayloadError::LatentStructureMismatch),
        }
    }

    pub fn masked_denoised(
        &self,
        denoised: &NativeLatentSamples,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentSamples, NativeDiffusionPayloadError> {
        match (self.bundle.samples(), denoised, self.bundle.noise_mask()) {
            (
                NativeLatentSamples::Tensor(original),
                NativeLatentSamples::Tensor(denoised),
                mask,
            ) => Ok(NativeLatentSamples::Tensor(masked_denoised_component(
                original,
                denoised,
                mask.and_then(NativeLatentNoiseMask::tensor),
                backend,
                context,
            )?)),
            (
                NativeLatentSamples::AudioVideo {
                    video: original_video,
                    audio: original_audio,
                },
                NativeLatentSamples::AudioVideo {
                    video: denoised_video,
                    audio: denoised_audio,
                },
                mask,
            ) => {
                let (video_mask, audio_mask) = match mask {
                    Some(NativeLatentNoiseMask::AudioVideo { video, audio }) => {
                        (Some(video), Some(audio))
                    }
                    None => (None, None),
                    Some(NativeLatentNoiseMask::Tensor(_)) => {
                        return Err(NativeDiffusionPayloadError::LatentStructureMismatch);
                    }
                };
                Ok(NativeLatentSamples::AudioVideo {
                    video: masked_denoised_component(
                        original_video,
                        denoised_video,
                        video_mask,
                        backend,
                        context,
                    )?,
                    audio: masked_denoised_component(
                        original_audio,
                        denoised_audio,
                        audio_mask,
                        backend,
                        context,
                    )?,
                })
            }
            _ => Err(NativeDiffusionPayloadError::LatentStructureMismatch),
        }
    }

    pub fn finish(
        &self,
        samples: NativeLatentSamples,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeLatentBundle, NativeDiffusionPayloadError> {
        Ok(self.bundle.replaced_samples(
            samples,
            NativeLatentMetadataRetention::DropDownscaleRatios,
            context,
        )?)
    }
}

pub fn prepare_native_latent_mask(
    mask: &Tensor,
    samples: &Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionPayloadError> {
    use sim_tensor::{
        MemoryFormatReference, ViewAccess,
        generated_shape_layout_transform_02::tensor_repeat_with_context_exact_native,
        generated_spatial_functional_kernel_01::{
            InterpolateConfiguration, InterpolateMode, interpolate_tensor_with_context_exact_native,
        },
        generated_storage_dtype_device_01::contiguous_with_context_exact_native,
    };
    context.check()?;
    let sample_shape = samples.descriptor().shape();
    let mask_shape = mask.descriptor().shape();
    if !(3..=5).contains(&sample_shape.len())
        || !(3..=5).contains(&mask_shape.len())
        || sample_shape.contains(&0)
        || mask_shape.contains(&0)
    {
        return Err(NativeDiffusionPayloadError::LatentStructureMismatch);
    }
    let dimensions = sample_shape.len() - 2;
    let mut shape = mask_shape.to_vec();
    let mode = match dimensions {
        1 => InterpolateMode::Linear,
        2 => {
            let rank = shape.len();
            let leading = shape[..rank - 2].iter().try_fold(1_u64, |count, size| {
                count
                    .checked_mul(*size)
                    .ok_or(sim_tensor::TensorError::ShapeOverflow)
            })?;
            shape = vec![leading, 1, shape[rank - 2], shape[rank - 1]];
            InterpolateMode::Bilinear
        }
        3 => {
            if shape.len() < 5 {
                let rank = shape.len();
                let frames = shape[..rank - 2].iter().try_fold(1_u64, |count, size| {
                    count
                        .checked_mul(*size)
                        .ok_or(sim_tensor::TensorError::ShapeOverflow)
                })?;
                shape = vec![1, 1, frames, shape[rank - 2], shape[rank - 1]];
            }
            InterpolateMode::Trilinear
        }
        _ => return Err(NativeDiffusionPayloadError::LatentStructureMismatch),
    };
    let mask = mask.view(
        mask.descriptor().reshaped_view(shape)?,
        ViewAccess::ReadOnly,
    )?;
    let mut mask = interpolate_tensor_with_context_exact_native(
        backend,
        &mask,
        &InterpolateConfiguration {
            output_size: Some(
                sample_shape[2..]
                    .iter()
                    .map(|size| {
                        usize::try_from(*size).map_err(|_| sim_tensor::TensorError::ShapeOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            scale_factor: None,
            mode,
            align_corners: Some(false),
            recompute_scale_factor: None,
            antialias: false,
        },
        context,
    )
    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
    for dimension in [1, 0] {
        let current = *mask
            .descriptor()
            .shape()
            .get(dimension)
            .ok_or(NativeDiffusionPayloadError::LatentStructureMismatch)?;
        let target = sample_shape[dimension];
        if current < target {
            let mut repeats = vec![1_i64; sample_shape.len()];
            repeats[dimension] = i64::try_from(target.div_ceil(current))
                .map_err(|_| sim_tensor::TensorError::ShapeOverflow)?;
            mask = tensor_repeat_with_context_exact_native(backend, &mask, &repeats, context)
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
            mask = mask.narrow_read_only(dimension, 0, target)?;
        } else if dimension == 0 && current > target {
            mask = mask.narrow_read_only(dimension, 0, target)?;
        }
        mask = contiguous_with_context_exact_native(
            backend,
            &mask,
            MemoryFormatReference::Layout(sim_tensor::Layout::Contiguous),
            context,
        )
        .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
    }
    context.check()?;
    Ok(mask)
}

fn masked_model_component(
    original: &Tensor,
    current: &Tensor,
    noise: &Tensor,
    mask: Option<&Tensor>,
    sigma: f32,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionPayloadError> {
    let Some(mask) = mask else {
        context.check()?;
        return Ok(current.clone());
    };
    require_same_latent_descriptor(original, current)?;
    require_same_latent_descriptor(original, noise)?;
    let original_values = tensor_to_f32(backend, original, context)?;
    let current_values = tensor_to_f32(backend, current, context)?;
    let noise_values = tensor_to_f32(backend, noise, context)?;
    let mask = prepare_native_latent_mask(mask, original, backend, context)?;
    let mask_values = tensor_to_f32(backend, &mask, context)?;
    let mut output = backend.workspace_vec(context, current_values.len())?;
    for index in 0..current_values.len() {
        if index % 4096 == 0 {
            context.check()?;
        }
        let mask_value = broadcast_mask_value(
            index,
            original.descriptor().shape(),
            mask.descriptor().shape(),
            &mask_values,
        )?;
        let preserved = original_values[index] + noise_values[index] * sigma;
        output.try_push(current_values[index] * mask_value + preserved * (1.0 - mask_value))?;
    }
    Ok(tensor_from_f32(
        backend,
        original.descriptor().shape(),
        &output,
        context,
    )?)
}

fn masked_denoised_component(
    original: &Tensor,
    denoised: &Tensor,
    mask: Option<&Tensor>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDiffusionPayloadError> {
    let Some(mask) = mask else {
        context.check()?;
        return Ok(denoised.clone());
    };
    require_same_latent_descriptor(original, denoised)?;
    let original_values = tensor_to_f32(backend, original, context)?;
    let denoised_values = tensor_to_f32(backend, denoised, context)?;
    let mask = prepare_native_latent_mask(mask, original, backend, context)?;
    let mask_values = tensor_to_f32(backend, &mask, context)?;
    let mut output = backend.workspace_vec(context, denoised_values.len())?;
    for index in 0..denoised_values.len() {
        if index % 4096 == 0 {
            context.check()?;
        }
        let mask_value = broadcast_mask_value(
            index,
            original.descriptor().shape(),
            mask.descriptor().shape(),
            &mask_values,
        )?;
        output.try_push(
            denoised_values[index] * mask_value + original_values[index] * (1.0 - mask_value),
        )?;
    }
    Ok(tensor_from_f32(
        backend,
        original.descriptor().shape(),
        &output,
        context,
    )?)
}

fn require_same_latent_descriptor(
    expected: &Tensor,
    actual: &Tensor,
) -> Result<(), NativeDiffusionPayloadError> {
    if expected.descriptor() != actual.descriptor() {
        return Err(NativeDiffusionPayloadError::LatentStructureMismatch);
    }
    Ok(())
}

fn broadcast_mask_value(
    mut output_index: usize,
    sample_shape: &[u64],
    mask_shape: &[u64],
    mask_values: &[f32],
) -> Result<f32, NativeDiffusionPayloadError> {
    if sample_shape.len() != mask_shape.len() {
        return Err(NativeDiffusionPayloadError::LatentStructureMismatch);
    }
    let mut mask_index = 0_usize;
    let mut mask_stride = 1_usize;
    for (sample_dimension, mask_dimension) in sample_shape.iter().zip(mask_shape).rev() {
        let sample_dimension = usize::try_from(*sample_dimension)
            .map_err(|_| NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        let mask_dimension = usize::try_from(*mask_dimension)
            .map_err(|_| NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        if sample_dimension == 0 || (mask_dimension != 1 && mask_dimension != sample_dimension) {
            return Err(NativeDiffusionPayloadError::LatentStructureMismatch);
        }
        let coordinate = output_index % sample_dimension;
        output_index /= sample_dimension;
        let mask_coordinate = if mask_dimension == 1 { 0 } else { coordinate };
        mask_index = mask_index
            .checked_add(
                mask_coordinate
                    .checked_mul(mask_stride)
                    .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?,
            )
            .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        mask_stride = mask_stride
            .checked_mul(mask_dimension)
            .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
    }
    mask_values
        .get(mask_index)
        .copied()
        .ok_or(NativeDiffusionPayloadError::LatentStructureMismatch)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeDiffusionResidentAllocationId {
    DiffusionPayloadArc {
        address: usize,
    },
    ConditioningSetArc {
        address: usize,
    },
    ModelPayloadArc {
        address: usize,
    },
    ModelBacking {
        kind: NativeModelBackingKind,
        address: usize,
    },
    ConditioningPayloadArc {
        address: usize,
    },
    PatchGraphArc {
        address: usize,
    },
    ExecutionHookGraphArc {
        address: usize,
    },
    SamplingPolicyArc {
        address: usize,
    },
    GuidanceModifierChainArc {
        address: usize,
    },
    ControlExecutionArc {
        address: usize,
    },
    ControlChainArc {
        address: usize,
    },
    ControlExecutorArc {
        address: usize,
    },
    TensorStorage {
        storage_id: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDiffusionResidentAllocation {
    id: NativeDiffusionResidentAllocationId,
    resident_bytes: u64,
}

impl NativeDiffusionResidentAllocation {
    pub fn id(&self) -> &NativeDiffusionResidentAllocationId {
        &self.id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDiffusionResidentParts {
    owned_bytes: u64,
    shared_allocations: Vec<NativeDiffusionResidentAllocation>,
}

impl NativeDiffusionResidentParts {
    pub(crate) fn for_guider(
        guider: &crate::NativeGuiderPayload,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let mut allocations = Vec::new();
        append_model_allocations(&mut allocations, guider.model())?;
        let conditionings = match guider.conditioning_sets() {
            crate::NativeGuiderConditioningSets::Basic { conditioning } => vec![conditioning],
            crate::NativeGuiderConditioningSets::Cfg { positive, negative } => {
                vec![positive, negative]
            }
            crate::NativeGuiderConditioningSets::DualCfg {
                positive,
                middle,
                negative,
            } => vec![positive, middle, negative],
            crate::NativeGuiderConditioningSets::DualModel {
                positive,
                negative,
                negative_model,
            } => {
                append_model_allocations(&mut allocations, negative_model)?;
                vec![positive, negative]
            }
        };
        for conditioning in conditionings {
            let parts = conditioning.resident_parts()?;
            allocations.push(NativeDiffusionResidentAllocation {
                id: NativeDiffusionResidentAllocationId::ConditioningSetArc {
                    address: arc_address(conditioning),
                },
                resident_bytes: parts.owned_bytes(),
            });
            allocations.extend(parts.tensor_allocations().iter().map(|allocation| {
                NativeDiffusionResidentAllocation {
                    id: NativeDiffusionResidentAllocationId::TensorStorage {
                        storage_id: allocation.storage_id().get(),
                    },
                    resident_bytes: allocation.resident_bytes(),
                }
            }));
        }
        for diffusion in [
            guider.execution_payload(),
            guider.negative_execution_payload(),
        ]
        .into_iter()
        .flatten()
        {
            let parts = diffusion.resident_parts()?;
            allocations.push(NativeDiffusionResidentAllocation {
                id: NativeDiffusionResidentAllocationId::DiffusionPayloadArc {
                    address: arc_address(diffusion),
                },
                resident_bytes: parts.owned_bytes(),
            });
            allocations.extend_from_slice(parts.shared_allocations());
        }
        Self::checked(guider.owned_resident_bytes()?, allocations)
    }

    fn checked(
        owned_bytes: u64,
        allocations: impl IntoIterator<Item = NativeDiffusionResidentAllocation>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let mut unique = BTreeMap::new();
        for allocation in allocations {
            if let Some(existing) = unique.insert(allocation.id, allocation.resident_bytes)
                && existing != allocation.resident_bytes
            {
                return Err(NativeDiffusionPayloadError::ResidentAllocationChanged);
            }
        }
        let shared_allocations = unique
            .into_iter()
            .map(|(id, resident_bytes)| NativeDiffusionResidentAllocation { id, resident_bytes })
            .collect::<Vec<_>>();
        let parts = Self {
            owned_bytes,
            shared_allocations,
        };
        parts.resident_bytes()?;
        Ok(parts)
    }

    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn shared_allocations(&self) -> &[NativeDiffusionResidentAllocation] {
        &self.shared_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeDiffusionPayloadError> {
        self.shared_allocations
            .iter()
            .try_fold(self.owned_bytes, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes)
                    .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)
            })
    }
}

#[derive(Clone)]
pub struct NativeControlExecution {
    chain: Arc<ControlChain>,
    executor: Arc<dyn ControlModelExecutor>,
    vae: Option<Arc<NativeVae>>,
    vae_execution_digest: Option<String>,
    execution_digest: String,
}

#[derive(Clone)]
pub struct NativeControlPayload {
    role: NativeModelResourceRole,
    execution: Arc<NativeControlExecution>,
    digest_sha256: String,
}

impl NativeControlPayload {
    pub fn checked(
        role: NativeModelResourceRole,
        execution: Arc<NativeControlExecution>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        if !matches!(role, NativeModelResourceRole::ControlNet) {
            return Err(NativeDiffusionPayloadError::RoleMismatch);
        }
        execution.validate()?;
        let digest_sha256 = sha256_tagged(
            "zed.sim.native-control-payload.v1",
            [
                role.source_type_id().as_bytes(),
                execution.execution_digest().as_bytes(),
            ],
        );
        Ok(Self {
            role,
            execution,
            digest_sha256,
        })
    }

    pub const fn role(&self) -> NativeModelResourceRole {
        self.role
    }

    pub fn execution(&self) -> &Arc<NativeControlExecution> {
        &self.execution
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn resident_bytes(&self) -> Result<usize, NativeDiffusionPayloadError> {
        usize::try_from(self.resident_parts()?.resident_bytes()?)
            .map_err(|_| NativeDiffusionPayloadError::ResidentBytesOverflow)
    }

    pub fn resident_parts(
        &self,
    ) -> Result<NativeDiffusionResidentParts, NativeDiffusionPayloadError> {
        let owned_bytes = resident_size::<Self>()?
            .checked_add(resident_capacity(&self.digest_sha256)?)
            .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        let mut allocations = vec![NativeDiffusionResidentAllocation {
            id: NativeDiffusionResidentAllocationId::ControlExecutionArc {
                address: arc_address(&self.execution),
            },
            resident_bytes: self.execution.owned_resident_bytes()?,
        }];
        allocations.extend(self.execution.shared_resident_allocations()?);
        NativeDiffusionResidentParts::checked(owned_bytes, allocations)
    }

    pub fn validate(&self) -> Result<(), NativeDiffusionPayloadError> {
        let expected = Self::checked(self.role, self.execution.clone())?;
        if expected.digest_sha256 != self.digest_sha256 {
            return Err(NativeDiffusionPayloadError::Invalid(
                "native ControlNet payload identity changed".to_owned(),
            ));
        }
        Ok(())
    }
}

impl NativeControlExecution {
    pub fn checked(
        chain: Arc<ControlChain>,
        executor: Arc<dyn ControlModelExecutor>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        Self::checked_inner(chain, executor, None)
    }

    pub fn checked_with_vae(
        chain: Arc<ControlChain>,
        executor: Arc<dyn ControlModelExecutor>,
        vae: Arc<NativeVae>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        Self::checked_inner(chain, executor, Some(vae))
    }

    fn checked_inner(
        chain: Arc<ControlChain>,
        executor: Arc<dyn ControlModelExecutor>,
        vae: Option<Arc<NativeVae>>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let executor_digest = executor.execution_digest();
        validate_sha256("ControlNet executor identity", executor_digest)?;
        chain
            .require_executor_digest(executor_digest)
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        let vae_execution_digest = vae.as_ref().map(|vae| vae.execution_digest());
        if let Some(digest) = &vae_execution_digest {
            validate_sha256("ControlNet VAE execution identity", digest)?;
        }
        chain
            .require_vae_execution_digest(vae_execution_digest.as_deref())
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        let vae_binding_digest = vae_execution_digest.as_deref().map_or_else(
            || sha256_tagged("zed.sim.controlnet.vae-binding.absent.v1", []),
            |digest| {
                sha256_tagged(
                    "zed.sim.controlnet.vae-binding.exact.v1",
                    [digest.as_bytes()],
                )
            },
        );
        let execution_digest = sha256_tagged(
            "zed.sim.controlnet.prebound-execution.v1",
            [
                chain.identity().digest().as_bytes(),
                executor_digest.as_bytes(),
                vae_binding_digest.as_bytes(),
            ],
        );
        Ok(Self {
            chain,
            executor,
            vae,
            vae_execution_digest,
            execution_digest,
        })
    }

    pub fn chain(&self) -> &Arc<ControlChain> {
        &self.chain
    }

    pub fn execution_digest(&self) -> &str {
        &self.execution_digest
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeDiffusionPayloadError> {
        NativeDiffusionResidentParts::checked(
            self.owned_resident_bytes()?,
            self.shared_resident_allocations()?,
        )?
        .resident_bytes()
    }

    fn owned_resident_bytes(&self) -> Result<u64, NativeDiffusionPayloadError> {
        let mut bytes = resident_size::<Self>()?
            .checked_add(resident_capacity(&self.execution_digest)?)
            .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        if let Some(digest) = &self.vae_execution_digest {
            bytes = bytes
                .checked_add(resident_capacity(digest)?)
                .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        }
        Ok(bytes)
    }

    fn embedded_owned_resident_bytes(&self) -> Result<u64, NativeDiffusionPayloadError> {
        self.owned_resident_bytes()?
            .checked_sub(resident_size::<Self>()?)
            .ok_or(NativeDiffusionPayloadError::ResidentAllocationChanged)
    }

    fn shared_resident_allocations(
        &self,
    ) -> Result<Vec<NativeDiffusionResidentAllocation>, NativeDiffusionPayloadError> {
        let chain_parts = self
            .chain
            .resident_parts()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        let mut allocations = vec![
            NativeDiffusionResidentAllocation {
                id: NativeDiffusionResidentAllocationId::ControlChainArc {
                    address: arc_address(&self.chain),
                },
                resident_bytes: chain_parts.owned_bytes(),
            },
            NativeDiffusionResidentAllocation {
                id: NativeDiffusionResidentAllocationId::ControlExecutorArc {
                    address: trait_arc_address(&self.executor),
                },
                resident_bytes: self
                    .executor
                    .resident_bytes()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            },
        ];
        allocations.extend(chain_parts.tensor_allocations().iter().map(|allocation| {
            NativeDiffusionResidentAllocation {
                id: NativeDiffusionResidentAllocationId::TensorStorage {
                    storage_id: allocation.storage_id().get(),
                },
                resident_bytes: allocation.resident_bytes(),
            }
        }));
        if let Some(vae) = &self.vae {
            allocations.push(NativeDiffusionResidentAllocation {
                id: NativeDiffusionResidentAllocationId::ModelBacking {
                    kind: NativeModelBackingKind::NativeVae,
                    address: arc_address(vae),
                },
                resident_bytes: vae
                    .resident_bytes()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            });
        }
        Ok(allocations)
    }

    pub fn vae_execution_digest(&self) -> Option<&str> {
        self.vae_execution_digest.as_deref()
    }

    pub fn shares_executor_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.executor, &other.executor)
    }

    pub fn rebound(
        &self,
        request: &ControlNetRebindRequest,
        predecessor: Option<&NativeControlExecution>,
        vae: Option<Arc<NativeVae>>,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        cancellation
            .check()
            .map_err(|_| NativeDiffusionPayloadError::Cancelled)?;
        if let Some(predecessor) = predecessor
            && predecessor.executor.execution_digest() != self.executor.execution_digest()
        {
            return Err(NativeDiffusionPayloadError::Invalid(
                "ControlNet predecessor uses a different executor identity".to_owned(),
            ));
        }
        let chain = Arc::new(
            self.chain
                .rebind_current(
                    request,
                    predecessor.map(|execution| execution.chain.as_ref()),
                    cancellation,
                )
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
        );
        cancellation
            .check()
            .map_err(|_| NativeDiffusionPayloadError::Cancelled)?;
        Self::checked_inner(chain, self.executor.clone(), vae)
    }

    pub fn validate(&self) -> Result<(), NativeDiffusionPayloadError> {
        let executor_digest = self.executor.execution_digest();
        validate_sha256("ControlNet executor identity", executor_digest)?;
        if self.execution_digest
            != Self::checked_inner(self.chain.clone(), self.executor.clone(), self.vae.clone())?
                .execution_digest
        {
            return Err(NativeDiffusionPayloadError::Invalid(
                "ControlNet execution identity changed".to_owned(),
            ));
        }
        Ok(())
    }
}

pub struct NativeConditioningMetadata<'a> {
    entry_identifier: &'a str,
    metadata: &'a ConditioningMetadataMap,
    window: Option<ConditioningWindow>,
}

impl<'a> NativeConditioningMetadata<'a> {
    pub fn checked(
        entry: &'a ResolvedConditioningEntry,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        entry.metadata().validate()?;
        let window = entry.metadata().sampler_window()?;
        Ok(Self {
            entry_identifier: entry.identifier(),
            metadata: entry.metadata(),
            window,
        })
    }

    pub fn entry_identifier(&self) -> &str {
        self.entry_identifier
    }

    pub fn window(&self) -> Option<ConditioningWindow> {
        self.window
    }

    pub fn get(&self, key: ConditioningMetadataKey) -> Option<&ConditioningMetadataValue> {
        self.metadata.get(key)
    }

    pub fn control_applies_to_unconditioned(&self) -> bool {
        match self.metadata.get(ConditioningMetadataKey::Control) {
            Some(ConditioningMetadataValue::Control(binding)) => binding.apply_to_unconditioned(),
            _ => true,
        }
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (&ConditioningMetadataKey, &ConditioningMetadataValue)> {
        self.metadata.iter()
    }
}

#[derive(Clone)]
pub struct NativeConditioningPayload {
    identity: ConditioningIdentity,
    identity_digest: String,
    patch_graph: Arc<PatchGraph>,
    model_execution_digest: String,
    control: Option<NativeControlExecution>,
    execution_digest: String,
}

impl NativeConditioningPayload {
    pub fn checked_sd15(
        model_digest: &str,
        model: &Sd15TinyModel,
        patch_graph: Arc<PatchGraph>,
        control: Option<NativeControlExecution>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        validate_sha256("model artifact identity", model_digest)?;
        patch_graph
            .identity()
            .validate_for_base(model_digest)
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        if &patch_graph.identity() != model.patch_identity() {
            return Err(NativeDiffusionPayloadError::Invalid(
                "model patch identity does not match its prebound graph".to_owned(),
            ));
        }
        validate_sha256("model execution identity", model.patch_execution_digest())?;
        let identity = ConditioningIdentity::new(
            "sd15-native-diffusion",
            sd15_model_family_identity()
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            sd15_latent_format_identity()
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
        )?;
        let identity_digest = identity.digest()?;
        let control_digest = control.as_ref().map_or_else(
            || sha256_tagged("zed.sim.controlnet.absent.v1", []),
            |control| control.execution_digest().to_owned(),
        );
        let execution_digest = sha256_tagged(
            "zed.sim.conditioning.execution.v1",
            [
                identity_digest.as_bytes(),
                GUIDANCE_ADAPTER_ID.as_bytes(),
                patch_graph.identity().ordered_digest.as_bytes(),
                model.patch_execution_digest().as_bytes(),
                control_digest.as_bytes(),
            ],
        );
        Ok(Self {
            identity,
            identity_digest,
            patch_graph,
            model_execution_digest: model.patch_execution_digest().to_owned(),
            control,
            execution_digest,
        })
    }

    pub fn checked_family(
        model: &NativeModelPayload,
        patch_graph: Arc<PatchGraph>,
        control: Option<NativeControlExecution>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        if control.is_some() {
            return Err(NativeDiffusionPayloadError::FamilyControlUnsupported);
        }
        let resource = model
            .native_family_model_resource()
            .ok_or(NativeDiffusionPayloadError::RoleMismatch)?;
        let cancellation = CancellationToken::default();
        resource
            .validate(&cancellation)
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        if !Arc::ptr_eq(resource.patch_graph(), &patch_graph)
            || resource.patch_graph().identity() != patch_graph.identity()
        {
            return Err(NativeDiffusionPayloadError::Invalid(
                "family model and conditioning must share one checked patch graph".to_owned(),
            ));
        }
        let identity = resource.conditioning_identity().clone();
        let identity_digest = identity.digest()?;
        let model_execution_digest = resource.semantic_digest_sha256().to_owned();
        validate_sha256("model execution identity", &model_execution_digest)?;
        let control_digest = sha256_tagged("zed.sim.controlnet.absent.v1", []);
        let execution_digest = sha256_tagged(
            "zed.sim.conditioning.execution.v1",
            [
                identity_digest.as_bytes(),
                GUIDANCE_ADAPTER_ID.as_bytes(),
                patch_graph.identity().ordered_digest.as_bytes(),
                model_execution_digest.as_bytes(),
                control_digest.as_bytes(),
            ],
        );
        Ok(Self {
            identity,
            identity_digest,
            patch_graph,
            model_execution_digest,
            control: None,
            execution_digest,
        })
    }

    pub fn identity(&self) -> &ConditioningIdentity {
        &self.identity
    }

    pub fn identity_digest(&self) -> &str {
        &self.identity_digest
    }

    pub fn patch_graph(&self) -> &Arc<PatchGraph> {
        &self.patch_graph
    }

    pub fn model_execution_digest(&self) -> &str {
        &self.model_execution_digest
    }

    pub fn control(&self) -> Option<&NativeControlExecution> {
        self.control.as_ref()
    }

    pub fn execution_digest(&self) -> &str {
        &self.execution_digest
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeDiffusionPayloadError> {
        self.resident_parts()?.resident_bytes()
    }

    pub fn resident_parts(
        &self,
    ) -> Result<NativeDiffusionResidentParts, NativeDiffusionPayloadError> {
        let mut owned_bytes = resident_size::<Self>()?;
        let identity_owned_bytes = self
            .identity
            .resident_bytes()?
            .checked_sub(resident_size::<ConditioningIdentity>()?)
            .ok_or(NativeDiffusionPayloadError::ResidentAllocationChanged)?;
        for bytes in [
            identity_owned_bytes,
            resident_capacity(&self.identity_digest)?,
            resident_capacity(&self.model_execution_digest)?,
            resident_capacity(&self.execution_digest)?,
        ] {
            owned_bytes = owned_bytes
                .checked_add(bytes)
                .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        }
        let mut allocations = vec![NativeDiffusionResidentAllocation {
            id: NativeDiffusionResidentAllocationId::PatchGraphArc {
                address: arc_address(&self.patch_graph),
            },
            resident_bytes: self
                .patch_graph
                .resident_bytes()
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
        }];
        if let Some(control) = &self.control {
            owned_bytes = owned_bytes
                .checked_add(control.embedded_owned_resident_bytes()?)
                .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
            allocations.extend(control.shared_resident_allocations()?);
        }
        NativeDiffusionResidentParts::checked(owned_bytes, allocations)
    }

    pub fn validate(&self) -> Result<(), NativeDiffusionPayloadError> {
        validate_sha256("conditioning identity", &self.identity_digest)?;
        validate_sha256("conditioning execution identity", &self.execution_digest)?;
        validate_sha256("model execution identity", &self.model_execution_digest)?;
        if let Some(control) = &self.control {
            control.validate()?;
        }
        self.resident_parts()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_control(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        model_time: f32,
        cross_attention: &Tensor,
        sampling_percent: f64,
        entry: &ResolvedConditioningEntry,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<ControlResult>, ControlNetError> {
        let Some(control) = &self.control else {
            context.check().map_err(|_| ControlNetError::Cancelled)?;
            return Ok(None);
        };
        control
            .validate()
            .map_err(|error| ControlNetError::Invalid(error.to_string()))?;
        let timestep =
            tensor_from_f32(backend, &[1], &[model_time], context).map_err(
                |error| match error {
                    NativeDiffusionTensorError::Tensor(error) => ControlNetError::from(error),
                    error => ControlNetError::CanonicalTensor(Box::new(error)),
                },
            )?;
        let binding = |tensor: Tensor| -> Result<ControlTensorBinding, ControlNetError> {
            let digest = format!("{:x}", Sha256::digest(tensor.contiguous_bytes()?));
            ControlTensorBinding::checked(tensor, digest)
        };
        let batch = latent
            .descriptor()
            .shape()
            .first()
            .copied()
            .ok_or_else(|| {
                ControlNetError::Invalid("canonical ControlNet latent has no batch axis".to_owned())
            })?;
        let mut concat = Vec::new();
        let mut mask = None;
        let mut control_cross_attention = None;
        let mut extra = BTreeMap::new();
        let metadata = NativeConditioningMetadata::checked(entry)
            .map_err(|error| ControlNetError::Invalid(error.to_string()))?;
        if let Some(window) = metadata.window()
            && !window.contains(sampling_percent)
        {
            return Ok(None);
        }
        for (key, value) in metadata.iter() {
            match (key, value) {
                (
                    ConditioningMetadataKey::CrossAttentionControlnet,
                    ConditioningMetadataValue::Tensor(tensor),
                ) => control_cross_attention = Some(binding(tensor.clone())?),
                (ConditioningMetadataKey::Control, ConditioningMetadataValue::Control(binding))
                    if binding.execution_digest() != control.execution_digest() =>
                {
                    return Err(ControlNetError::Invalid(
                        "conditioning metadata references a different ControlNet execution"
                            .to_owned(),
                    ));
                }
                (
                    ConditioningMetadataKey::ConcatLatentImage,
                    ConditioningMetadataValue::Tensor(tensor),
                ) => concat.push(binding(tensor.clone())?),
                (
                    ConditioningMetadataKey::ConcatLatentImage,
                    ConditioningMetadataValue::LatentBundle(bundle),
                ) => {
                    let tensor = bundle.samples().tensor().ok_or_else(|| {
                        ControlNetError::Invalid(
                            "ControlNet concat metadata requires a single latent tensor".to_owned(),
                        )
                    })?;
                    concat.push(binding(tensor.clone())?);
                }
                (
                    ConditioningMetadataKey::ConcatMask,
                    ConditioningMetadataValue::Tensor(tensor),
                ) => mask = Some(binding(tensor.clone())?),
                (key, ConditioningMetadataValue::Tensor(tensor)) => {
                    let name = match key {
                        ConditioningMetadataKey::CameraConditions => "camera_conditions",
                        ConditioningMetadataKey::TimeDimensionConcat => "time_dim_concat",
                        ConditioningMetadataKey::AudioEmbedding => "audio_embed",
                        ConditioningMetadataKey::ReferenceMotion => "reference_motion",
                        ConditioningMetadataKey::ControlVideo => "control_video",
                        ConditioningMetadataKey::PoseVideoLatent => "pose_video_latent",
                        ConditioningMetadataKey::FaceVideoPixels => "face_video_pixels",
                        ConditioningMetadataKey::DrivingMask28Channels => "driving_mask_28ch",
                        ConditioningMetadataKey::ReferenceMask28Channels => "ref_mask_28ch",
                        _ => continue,
                    };
                    extra.insert(name.to_owned(), binding(tensor.clone())?);
                }
                _ => {}
            }
        }
        let conditioning = ControlConditioning::checked(
            binding(latent.clone())?,
            binding(timestep)?,
            binding(cross_attention.clone())?,
            control_cross_attention,
            concat,
            mask,
            extra,
            sampling_percent as f32,
            batch,
        )?;
        ControlRuntime::new(backend, control.executor.as_ref()).execute(
            &control.chain,
            ControlIsolation::CompleteChain,
            &conditioning,
            control.vae.as_deref(),
            context,
        )
    }
}

#[derive(Clone)]
pub struct NativeDiffusionPayload {
    resource: NativeDiffusionResource,
    hook_graph: Arc<ExecutionHookGraph>,
    sampling_policy: Arc<NativeSamplingPolicy>,
    guidance_modifiers: Arc<NativeGuidanceModifierChain>,
    digest_sha256: String,
}

#[derive(Clone)]
enum NativeDiffusionResource {
    Model {
        model: Arc<NativeModelPayload>,
        conditioning: Arc<NativeConditioningPayload>,
    },
    Clip {
        clip: Arc<NativeModelPayload>,
    },
    Vae {
        vae: Arc<NativeModelPayload>,
    },
}

impl NativeDiffusionPayload {
    pub fn model(
        model: Arc<NativeModelPayload>,
        conditioning: Arc<NativeConditioningPayload>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        require_role(&model, NativeModelResourceRole::Model)?;
        model.validate()?;
        conditioning.validate()?;
        let model_resource = model
            .model()
            .ok_or(NativeDiffusionPayloadError::RoleMismatch)?;
        if model_resource.patch_identity() != conditioning.patch_graph().identity()
            || model_resource.execution_digest() != conditioning.model_execution_digest()
        {
            return Err(NativeDiffusionPayloadError::Invalid(
                "conditioning does not belong to the model payload".to_owned(),
            ));
        }
        let digest_sha256 = sha256_tagged(
            "zed.sim.native-diffusion-model-payload.v1",
            [
                model.identity().digest_sha256().as_bytes(),
                conditioning.execution_digest().as_bytes(),
            ],
        );
        let mut payload = Self {
            sampling_policy: Arc::new(
                profile_for_model(&model)
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            guidance_modifiers: Arc::new(
                NativeGuidanceModifierChain::empty()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            resource: NativeDiffusionResource::Model {
                model,
                conditioning,
            },
            hook_graph: Arc::new(
                ExecutionHookGraph::checked(0, Vec::new())
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            digest_sha256,
        };
        payload.digest_sha256 = payload.project_digest()?;
        Ok(payload)
    }

    pub fn clip(clip: Arc<NativeModelPayload>) -> Result<Self, NativeDiffusionPayloadError> {
        require_role(&clip, NativeModelResourceRole::Clip)?;
        clip.validate()?;
        let digest_sha256 = sha256_tagged(
            "zed.sim.native-diffusion-clip-payload.v1",
            [clip.identity().digest_sha256().as_bytes()],
        );
        let mut payload = Self {
            sampling_policy: Arc::new(
                NativeSamplingPolicy::sd15()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            guidance_modifiers: Arc::new(
                NativeGuidanceModifierChain::empty()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            resource: NativeDiffusionResource::Clip { clip },
            hook_graph: Arc::new(
                ExecutionHookGraph::checked(0, Vec::new())
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            digest_sha256,
        };
        payload.digest_sha256 = payload.project_digest()?;
        Ok(payload)
    }

    pub fn vae(vae: Arc<NativeModelPayload>) -> Result<Self, NativeDiffusionPayloadError> {
        require_role(&vae, NativeModelResourceRole::Vae)?;
        vae.validate()?;
        let digest_sha256 = sha256_tagged(
            "zed.sim.native-diffusion-vae-payload.v1",
            [vae.identity().digest_sha256().as_bytes()],
        );
        let mut payload = Self {
            sampling_policy: Arc::new(
                NativeSamplingPolicy::sd15()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            guidance_modifiers: Arc::new(
                NativeGuidanceModifierChain::empty()
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            resource: NativeDiffusionResource::Vae { vae },
            hook_graph: Arc::new(
                ExecutionHookGraph::checked(0, Vec::new())
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
            ),
            digest_sha256,
        };
        payload.digest_sha256 = payload.project_digest()?;
        Ok(payload)
    }

    pub const fn role(&self) -> NativeModelResourceRole {
        match &self.resource {
            NativeDiffusionResource::Model { .. } => NativeModelResourceRole::Model,
            NativeDiffusionResource::Clip { .. } => NativeModelResourceRole::Clip,
            NativeDiffusionResource::Vae { .. } => NativeModelResourceRole::Vae,
        }
    }

    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }

    pub fn model_payload(&self) -> &Arc<NativeModelPayload> {
        match &self.resource {
            NativeDiffusionResource::Model { model, .. } => model,
            NativeDiffusionResource::Clip { clip } => clip,
            NativeDiffusionResource::Vae { vae } => vae,
        }
    }

    pub fn model_resources(
        &self,
    ) -> Option<(&Arc<NativeModelPayload>, &Arc<NativeConditioningPayload>)> {
        match &self.resource {
            NativeDiffusionResource::Model {
                model,
                conditioning,
            } => Some((model, conditioning)),
            NativeDiffusionResource::Clip { .. } | NativeDiffusionResource::Vae { .. } => None,
        }
    }

    pub fn vae_payload(&self) -> Option<&Arc<NativeModelPayload>> {
        match &self.resource {
            NativeDiffusionResource::Vae { vae } => Some(vae),
            NativeDiffusionResource::Model { .. } | NativeDiffusionResource::Clip { .. } => None,
        }
    }

    pub fn clip_payload(&self) -> Option<&Arc<NativeModelPayload>> {
        match &self.resource {
            NativeDiffusionResource::Clip { clip } => Some(clip),
            NativeDiffusionResource::Model { .. } | NativeDiffusionResource::Vae { .. } => None,
        }
    }

    pub fn native_clip_resource(&self) -> Option<&Arc<NativeClipResource>> {
        self.clip_payload()
            .and_then(|payload| payload.native_clip_resource())
    }

    pub fn hook_graph(&self) -> &Arc<ExecutionHookGraph> {
        &self.hook_graph
    }

    pub fn sampling_policy(&self) -> &Arc<NativeSamplingPolicy> {
        &self.sampling_policy
    }

    pub fn guidance_modifiers(&self) -> &Arc<NativeGuidanceModifierChain> {
        &self.guidance_modifiers
    }

    pub fn with_sampling_policy(
        mut self,
        policy: Arc<NativeSamplingPolicy>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        if self.role() != NativeModelResourceRole::Model {
            return Err(NativeDiffusionPayloadError::RoleMismatch);
        }
        self.sampling_policy = policy;
        self.digest_sha256 = self.project_digest()?;
        self.validate()?;
        Ok(self)
    }

    pub fn with_guidance_modifiers(
        mut self,
        modifiers: Arc<NativeGuidanceModifierChain>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        if self.role() != NativeModelResourceRole::Model {
            return Err(NativeDiffusionPayloadError::RoleMismatch);
        }
        modifiers
            .validate()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        self.guidance_modifiers = modifiers;
        self.digest_sha256 = self.project_digest()?;
        self.validate()?;
        Ok(self)
    }

    pub fn export_safetensors(
        &self,
        metadata: BTreeMap<String, String>,
        maximum_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, NativeDiffusionPayloadError> {
        let kind = match self.role() {
            NativeModelResourceRole::Model => ModelStateExportKind::Model,
            NativeModelResourceRole::Clip => ModelStateExportKind::Clip,
            NativeModelResourceRole::Vae => ModelStateExportKind::Vae,
            _ => return Err(NativeDiffusionPayloadError::RoleMismatch),
        };
        let state = self.model_payload().keyed_transform_state(cancellation)?;
        export_safetensors_exact(kind, &state, metadata, maximum_bytes, cancellation)
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))
    }

    pub fn export_lora_safetensors(
        state: &BTreeMap<String, Tensor>,
        maximum_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, NativeDiffusionPayloadError> {
        export_safetensors_exact_with_optional_metadata(
            ModelStateExportKind::Lora,
            state,
            None,
            maximum_bytes,
            cancellation,
        )
        .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_lora(
        &self,
        key_map: &PatchKeyMap,
        tensors: &BTreeMap<String, AdapterTensor>,
        strength_patch: f32,
        strength_model: f32,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        context.check()?;
        let mut transformed = self.clone();
        let mut loaded_keys = BTreeSet::new();
        let mut applied_operations = 0_usize;
        if let Some(resource) = self.model_payload().native_family_model_resource() {
            let report = load_lora_patch_graph(
                resource.artifact_sha256(),
                key_map,
                tensors,
                &state_shapes(resource.keyed_state()),
                strength_patch,
                strength_model,
                backend,
                context,
            )
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
            loaded_keys.extend(report.loaded_keys.iter().cloned());
            applied_operations = report.graph.semantic_operations().len();
            let graph =
                lora_graph_with_bypass(resource.artifact_sha256(), &report.graph, strength_patch)?;
            transformed = self.apply_patch_graph(&graph, backend, context)?;
        } else if let Some(resource) = self.model_payload().native_clip_resource() {
            let mut mapped_targets = BTreeMap::<String, usize>::new();
            for component in resource.components() {
                let component_mappings = key_map
                    .mappings()
                    .iter()
                    .filter(|mapping| component.keyed_state().contains_key(&mapping.target))
                    .cloned()
                    .collect::<Vec<_>>();
                for mapping in &component_mappings {
                    if mapped_targets.insert(mapping.target.clone(), 1).is_some() {
                        return Err(NativeDiffusionPayloadError::Invalid(
                            "LoRA CLIP target is ambiguous across retained components".to_owned(),
                        ));
                    }
                }
                if component_mappings.is_empty() {
                    continue;
                }
                let component_key_map = PatchKeyMap::checked(component_mappings)
                    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
                let report = load_lora_patch_graph(
                    component.artifact_sha256(),
                    &component_key_map,
                    tensors,
                    &state_shapes(component.keyed_state()),
                    strength_patch,
                    strength_model,
                    backend,
                    context,
                )
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
                loaded_keys.extend(report.loaded_keys.iter().cloned());
                applied_operations = applied_operations
                    .checked_add(report.graph.semantic_operations().len())
                    .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
                let graph = lora_graph_with_bypass(
                    component.artifact_sha256(),
                    &report.graph,
                    strength_patch,
                )?;
                transformed = transformed.apply_patch_graph(&graph, backend, context)?;
            }
        } else {
            return Err(NativeDiffusionPayloadError::RoleMismatch);
        }
        if applied_operations == 0 || tensors.keys().any(|key| !loaded_keys.contains(key)) {
            return Err(NativeDiffusionPayloadError::Invalid(
                "LoRA adapter has no exact retained MODEL or CLIP mapping".to_owned(),
            ));
        }
        Ok(transformed)
    }

    pub fn with_hook_graph(
        mut self,
        hook_graph: Arc<ExecutionHookGraph>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        if self.role() != NativeModelResourceRole::Model {
            return Err(NativeDiffusionPayloadError::IncompatibleHookGraph);
        }
        hook_graph
            .validate()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        self.hook_graph = hook_graph;
        self.digest_sha256 = self.project_digest()?;
        self.validate()?;
        Ok(self)
    }

    pub fn scale_selected_state(
        &self,
        selector: &StateKeySelector,
        scale: f32,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let transformed = Arc::new(
            self.model_payload()
                .scale_selected_state(selector, scale, backend, context)?,
        );
        self.rebind_model_payload(transformed)
    }

    pub fn scale_matching_state(
        &self,
        rules: &[(StateKeySelector, f32)],
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let transformed = Arc::new(
            self.model_payload()
                .scale_matching_state(rules, backend, context)?,
        );
        self.rebind_model_payload(transformed)
    }

    pub fn merge_from(
        &self,
        source: &Self,
        rule: &PatchMergeRule,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        if self.role() != source.role() {
            return Err(NativeDiffusionPayloadError::RoleMismatch);
        }
        let transformed = Arc::new(self.model_payload().merge_from(
            source.model_payload(),
            rule,
            backend,
            context,
        )?);
        self.rebind_model_payload(transformed)
    }

    pub fn apply_patch_graph(
        &self,
        patch: &PatchGraph,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let transformed = Arc::new(
            self.model_payload()
                .apply_patch_graph(patch, backend, context)?,
        );
        self.rebind_model_payload(transformed)
    }

    fn rebind_model_payload(
        &self,
        payload: Arc<NativeModelPayload>,
    ) -> Result<Self, NativeDiffusionPayloadError> {
        let mut rebound = match &self.resource {
            NativeDiffusionResource::Model { conditioning, .. } => {
                let resource = payload
                    .native_family_model_resource()
                    .ok_or(NativeDiffusionPayloadError::RoleMismatch)?;
                if resource.conditioning_identity() != conditioning.identity()
                    || conditioning.control().is_some()
                {
                    return Err(NativeDiffusionPayloadError::IncompatibleConditioning);
                }
                let conditioning = Arc::new(NativeConditioningPayload::checked_family(
                    &payload,
                    resource.patch_graph().clone(),
                    None,
                )?);
                Self::model(payload, conditioning)?
            }
            NativeDiffusionResource::Clip { .. } => Self::clip(payload)?,
            NativeDiffusionResource::Vae { .. } => Self::vae(payload)?,
        };
        rebound.hook_graph = self.hook_graph.clone();
        rebound.sampling_policy = self.sampling_policy.clone();
        rebound.guidance_modifiers = self.guidance_modifiers.clone();
        rebound.digest_sha256 = rebound.project_digest()?;
        rebound.validate()?;
        Ok(rebound)
    }

    fn project_digest(&self) -> Result<String, NativeDiffusionPayloadError> {
        let hook_identity = self
            .hook_graph
            .identity()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        let sampling_identity = self.sampling_policy.semantic_digest_sha256();
        let modifier_identity = self.guidance_modifiers.semantic_digest_sha256();
        Ok(match &self.resource {
            NativeDiffusionResource::Model {
                model,
                conditioning,
            } => sha256_tagged(
                "zed.sim.native-diffusion-model-transform.v1",
                [
                    model.identity().digest_sha256().as_bytes(),
                    conditioning.execution_digest().as_bytes(),
                    hook_identity.as_bytes(),
                    sampling_identity.as_bytes(),
                    modifier_identity.as_bytes(),
                ],
            ),
            NativeDiffusionResource::Clip { clip } => sha256_tagged(
                "zed.sim.native-diffusion-clip-transform.v1",
                [
                    clip.identity().digest_sha256().as_bytes(),
                    hook_identity.as_bytes(),
                    sampling_identity.as_bytes(),
                    modifier_identity.as_bytes(),
                ],
            ),
            NativeDiffusionResource::Vae { vae } => sha256_tagged(
                "zed.sim.native-diffusion-vae-transform.v1",
                [
                    vae.identity().digest_sha256().as_bytes(),
                    hook_identity.as_bytes(),
                    sampling_identity.as_bytes(),
                    modifier_identity.as_bytes(),
                ],
            ),
        })
    }

    pub fn resident_bytes(&self) -> Result<usize, NativeDiffusionPayloadError> {
        usize::try_from(self.resident_parts()?.resident_bytes()?)
            .map_err(|_| NativeDiffusionPayloadError::ResidentBytesOverflow)
    }

    pub fn resident_parts(
        &self,
    ) -> Result<NativeDiffusionResidentParts, NativeDiffusionPayloadError> {
        let owned_bytes = resident_size::<Self>()?
            .checked_add(resident_capacity(&self.digest_sha256)?)
            .ok_or(NativeDiffusionPayloadError::ResidentBytesOverflow)?;
        let mut allocations = Vec::new();
        allocations.push(NativeDiffusionResidentAllocation {
            id: NativeDiffusionResidentAllocationId::ExecutionHookGraphArc {
                address: arc_address(&self.hook_graph),
            },
            resident_bytes: self
                .hook_graph
                .resident_bytes()
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
        });
        allocations.push(NativeDiffusionResidentAllocation {
            id: NativeDiffusionResidentAllocationId::SamplingPolicyArc {
                address: arc_address(&self.sampling_policy),
            },
            resident_bytes: self.sampling_policy.resident_bytes(),
        });
        allocations.push(NativeDiffusionResidentAllocation {
            id: NativeDiffusionResidentAllocationId::GuidanceModifierChainArc {
                address: arc_address(&self.guidance_modifiers),
            },
            resident_bytes: self
                .guidance_modifiers
                .resident_bytes()
                .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?,
        });
        match &self.resource {
            NativeDiffusionResource::Model {
                model,
                conditioning,
            } => {
                append_model_allocations(&mut allocations, model)?;
                append_conditioning_allocations(&mut allocations, conditioning)?;
            }
            NativeDiffusionResource::Clip { clip } => {
                append_model_allocations(&mut allocations, clip)?;
            }
            NativeDiffusionResource::Vae { vae } => {
                append_model_allocations(&mut allocations, vae)?;
            }
        }
        NativeDiffusionResidentParts::checked(owned_bytes, allocations)
    }

    pub fn validate(&self) -> Result<(), NativeDiffusionPayloadError> {
        self.hook_graph
            .validate()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        self.guidance_modifiers
            .validate()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        self.sampling_policy
            .validate()
            .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))?;
        let mut expected = match &self.resource {
            NativeDiffusionResource::Model {
                model,
                conditioning,
                ..
            } => Self::model(model.clone(), conditioning.clone())?,
            NativeDiffusionResource::Clip { clip } => Self::clip(clip.clone())?,
            NativeDiffusionResource::Vae { vae } => Self::vae(vae.clone())?,
        };
        expected.hook_graph = self.hook_graph.clone();
        expected.sampling_policy = self.sampling_policy.clone();
        expected.guidance_modifiers = self.guidance_modifiers.clone();
        expected.digest_sha256 = expected.project_digest()?;
        if expected.digest_sha256 != self.digest_sha256 {
            return Err(NativeDiffusionPayloadError::Invalid(
                "native diffusion payload identity changed".to_owned(),
            ));
        }
        Ok(())
    }
}

fn state_shapes(state: &BTreeMap<String, Tensor>) -> BTreeMap<String, Vec<u64>> {
    state
        .iter()
        .map(|(key, tensor)| (key.clone(), tensor.descriptor().shape().to_vec()))
        .collect()
}

fn lora_graph_with_bypass(
    base_artifact_digest: &str,
    graph: &PatchGraph,
    strength_patch: f32,
) -> Result<PatchGraph, NativeDiffusionPayloadError> {
    if strength_patch != 0.0 {
        return Ok(graph.clone());
    }
    PatchGraph::checked_semantic(
        base_artifact_digest,
        graph
            .semantic_operations()
            .iter()
            .cloned()
            .map(|mut operation| {
                operation.payload = PatchPayload::ScaleOnly;
                operation
            })
            .collect(),
    )
    .map_err(|error| NativeDiffusionPayloadError::Invalid(error.to_string()))
}

#[derive(Debug, Error)]
pub enum NativeDiffusionPayloadError {
    #[error(transparent)]
    LatentFormat(#[from] sim_model::LatentFormatError),
    #[error(transparent)]
    Tensor(#[from] sim_tensor::TensorError),
    #[error(transparent)]
    TensorKernel(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Latent(#[from] NativeLatentBundleError),
    #[error(transparent)]
    SamplerPayload(#[from] Box<NativeSamplerPayloadError>),
    #[error(transparent)]
    Model(#[from] NativeModelPayloadError),
    #[error(transparent)]
    Conditioning(#[from] ConditioningError),
    #[error("native diffusion payload role does not match its concrete resource")]
    RoleMismatch,
    #[error("native latent samples, masks, or generated noise have incompatible structure")]
    LatentStructureMismatch,
    #[error("native family MODEL conditioning does not support ControlNet execution yet")]
    FamilyControlUnsupported,
    #[error("native diffusion model transform cannot preserve incompatible conditioning")]
    IncompatibleConditioning,
    #[error("the execution-hook graph is incompatible with this diffusion payload role")]
    IncompatibleHookGraph,
    #[error("native diffusion payload operation was cancelled")]
    Cancelled,
    #[error("native diffusion payload is invalid: {0}")]
    Invalid(String),
    #[error("native diffusion payload resident byte accounting overflowed")]
    ResidentBytesOverflow,
    #[error("native diffusion payload resident allocation identity changed its byte count")]
    ResidentAllocationChanged,
}

impl From<NativeSamplerPayloadError> for NativeDiffusionPayloadError {
    fn from(error: NativeSamplerPayloadError) -> Self {
        Self::SamplerPayload(Box::new(error))
    }
}

fn resident_size<Value>() -> Result<u64, NativeDiffusionPayloadError> {
    u64::try_from(mem::size_of::<Value>())
        .map_err(|_| NativeDiffusionPayloadError::ResidentBytesOverflow)
}

fn resident_capacity(value: &String) -> Result<u64, NativeDiffusionPayloadError> {
    u64::try_from(value.capacity()).map_err(|_| NativeDiffusionPayloadError::ResidentBytesOverflow)
}

fn arc_address<Value>(value: &Arc<Value>) -> usize {
    Arc::as_ptr(value) as usize
}

fn trait_arc_address(value: &Arc<dyn ControlModelExecutor>) -> usize {
    Arc::as_ptr(value) as *const () as usize
}

fn append_model_allocations(
    allocations: &mut Vec<NativeDiffusionResidentAllocation>,
    model: &Arc<NativeModelPayload>,
) -> Result<(), NativeDiffusionPayloadError> {
    let parts = model.resident_parts()?;
    allocations.push(NativeDiffusionResidentAllocation {
        id: NativeDiffusionResidentAllocationId::ModelPayloadArc {
            address: arc_address(model),
        },
        resident_bytes: parts.owned_bytes(),
    });
    allocations.extend(parts.backing_allocations().iter().map(|allocation| {
        let id = if allocation.kind() == NativeModelBackingKind::NativeFamilyModelPatchGraph {
            NativeDiffusionResidentAllocationId::PatchGraphArc {
                address: allocation.address(),
            }
        } else {
            NativeDiffusionResidentAllocationId::ModelBacking {
                kind: allocation.kind(),
                address: allocation.address(),
            }
        };
        NativeDiffusionResidentAllocation {
            id,
            resident_bytes: allocation.resident_bytes(),
        }
    }));
    allocations.extend(parts.tensor_allocations().iter().map(|allocation| {
        NativeDiffusionResidentAllocation {
            id: NativeDiffusionResidentAllocationId::TensorStorage {
                storage_id: allocation.storage_id().get(),
            },
            resident_bytes: allocation.resident_bytes(),
        }
    }));
    Ok(())
}

fn append_conditioning_allocations(
    allocations: &mut Vec<NativeDiffusionResidentAllocation>,
    conditioning: &Arc<NativeConditioningPayload>,
) -> Result<(), NativeDiffusionPayloadError> {
    let parts = conditioning.resident_parts()?;
    allocations.push(NativeDiffusionResidentAllocation {
        id: NativeDiffusionResidentAllocationId::ConditioningPayloadArc {
            address: arc_address(conditioning),
        },
        resident_bytes: parts.owned_bytes(),
    });
    allocations.extend_from_slice(parts.shared_allocations());
    Ok(())
}

fn require_role(
    payload: &NativeModelPayload,
    expected: NativeModelResourceRole,
) -> Result<(), NativeDiffusionPayloadError> {
    if payload.identity().role() == expected {
        Ok(())
    } else {
        Err(NativeDiffusionPayloadError::RoleMismatch)
    }
}

fn validate_sha256(subject: &str, digest: &str) -> Result<(), NativeDiffusionPayloadError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(NativeDiffusionPayloadError::Invalid(format!(
            "{subject} is not a SHA-256 digest"
        )))
    }
}

fn sha256_tagged<const N: usize>(tag: &str, fields: [&[u8]; N]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tag.as_bytes());
    hasher.update([0]);
    for field in fields {
        hasher.update(field);
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_model::{
        ModelFamilyIdentity, PatchGraphIdentity,
        controlnet::{
            ControlBase, ControlHintLayout, ControlHintPreprocess, ControlModelBinding,
            ControlModelInput, ControlNet, ControlNetRebindRequest, ControlNode,
            ControlPercentWindow, StrengthType,
        },
    };
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, DType, DeviceId, NativeLatentMetadata,
        NativeLatentType, StreamId,
        generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    };
    use std::{
        error::Error,
        sync::atomic::{AtomicBool, Ordering},
    };

    const EXECUTOR_DIGEST: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CHANGED_EXECUTOR_DIGEST: &str =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    struct MutableIdentityExecutor {
        changed: AtomicBool,
    }

    #[test]
    fn latent_sampling_uses_ordered_batch_indices_and_preserves_repetitions()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let samples = tensor_from_f32(&backend, &[3, 1, 2], &[0.0; 6], &context)?;
        let bundle = NativeLatentBundle::single(
            samples,
            None,
            Some(vec![2, 0, 2]),
            NativeLatentMetadata::default(),
            &context,
        )?;
        let sampling = NativeLatentSamplingInput::checked(&bundle, &context)?;
        let request = NoiseRequest::native_diffusion("prompt", "node")?;
        let generation = sampling.generate_noise(
            &NativeNoisePayload::random(17)?,
            &backend,
            &request,
            &context,
        )?;
        let NativeLatentNoiseGeneration::Tensor(generation) = generation else {
            return Err("single latent generated nested noise".into());
        };
        let values = tensor_to_f32(&backend, &generation.noise, &context)?;
        assert_eq!(&values[0..2], &values[4..6]);
        assert_ne!(&values[0..2], &values[2..4]);
        Ok(())
    }

    #[test]
    fn latent_sampling_masks_model_input_and_denoised_output_at_canonical_owner()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let original = tensor_from_f32(&backend, &[1, 1, 2], &[10.0, 20.0], &context)?;
        let mask = tensor_from_f32(&backend, &[1, 1, 2], &[0.0, 1.0], &context)?;
        let bundle = NativeLatentBundle::single(
            original,
            Some(mask),
            Some(vec![5]),
            NativeLatentMetadata::checked(
                Some(NativeLatentType::Audio),
                Some(48_000),
                Some(8),
                Some(4),
            )?,
            &context,
        )?;
        let sampling = NativeLatentSamplingInput::checked(&bundle, &context)?;
        let current = tensor_from_f32(&backend, &[1, 1, 2], &[100.0, 200.0], &context)?;
        let noise = tensor_from_f32(&backend, &[1, 1, 2], &[1.0, 2.0], &context)?;
        let masked = sampling.masked_model_input(
            &NativeLatentSamples::Tensor(current),
            &NativeLatentNoiseGeneration::Tensor(NativeNoiseGeneration {
                noise,
                before: None,
                after: None,
            }),
            2.0,
            &backend,
            &context,
        )?;
        let masked = masked.tensor().ok_or("masked samples are nested")?;
        assert_eq!(&*tensor_to_f32(&backend, masked, &context)?, &[12.0, 200.0]);

        let denoised = tensor_from_f32(&backend, &[1, 1, 2], &[300.0, 400.0], &context)?;
        let denoised =
            sampling.masked_denoised(&NativeLatentSamples::Tensor(denoised), &backend, &context)?;
        let denoised = denoised.tensor().ok_or("denoised samples are nested")?;
        assert_eq!(
            &*tensor_to_f32(&backend, denoised, &context)?,
            &[10.0, 400.0]
        );

        let finished = sampling.finish(NativeLatentSamples::Tensor(denoised.clone()), &context)?;
        assert_eq!(finished.batch_indices(), Some([5].as_slice()));
        assert!(finished.noise_mask().is_some());
        assert_eq!(
            finished.metadata().latent_type(),
            Some(NativeLatentType::Audio)
        );
        assert_eq!(finished.metadata().sample_rate(), Some(48_000));
        assert_eq!(finished.metadata().spatial_downscale_ratio(), None);
        assert_eq!(finished.metadata().temporal_downscale_ratio(), None);
        Ok(())
    }

    #[test]
    fn maskless_sampling_allows_shape_transitions_but_masked_sampling_rejects_them()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let original = tensor_from_f32(&backend, &[1, 1, 2], &[1.0, 2.0], &context)?;
        let resized = tensor_from_f32(&backend, &[1, 1, 4], &[3.0, 4.0, 5.0, 6.0], &context)?;
        let resized_noise = tensor_from_f32(&backend, &[1, 1, 4], &[0.0, 0.0, 0.0, 0.0], &context)?;
        let maskless = NativeLatentSamplingInput::checked(
            &NativeLatentBundle::single(
                original.clone(),
                None,
                None,
                NativeLatentMetadata::default(),
                &context,
            )?,
            &context,
        )?;
        assert_eq!(
            maskless
                .masked_model_input(
                    &NativeLatentSamples::Tensor(resized.clone()),
                    &NativeLatentNoiseGeneration::Tensor(NativeNoiseGeneration {
                        noise: resized_noise,
                        before: None,
                        after: None,
                    }),
                    1.0,
                    &backend,
                    &context,
                )?
                .tensor()
                .ok_or("maskless model input became nested")?
                .descriptor(),
            resized.descriptor()
        );
        assert_eq!(
            maskless
                .masked_denoised(
                    &NativeLatentSamples::Tensor(resized.clone()),
                    &backend,
                    &context,
                )?
                .tensor()
                .ok_or("maskless denoised output became nested")?
                .descriptor(),
            resized.descriptor()
        );

        let mask = tensor_from_f32(&backend, &[1, 1, 2], &[1.0, 1.0], &context)?;
        let masked = NativeLatentSamplingInput::checked(
            &NativeLatentBundle::single(
                original,
                Some(mask),
                None,
                NativeLatentMetadata::default(),
                &context,
            )?,
            &context,
        )?;
        assert!(matches!(
            masked.masked_denoised(&NativeLatentSamples::Tensor(resized), &backend, &context,),
            Err(NativeDiffusionPayloadError::LatentStructureMismatch)
        ));
        Ok(())
    }

    impl ControlModelExecutor for MutableIdentityExecutor {
        fn execution_digest(&self) -> &str {
            if self.changed.load(Ordering::SeqCst) {
                CHANGED_EXECUTOR_DIGEST
            } else {
                EXECUTOR_DIGEST
            }
        }

        fn resident_bytes(&self) -> Result<u64, ControlNetError> {
            u64::try_from(mem::size_of::<Self>())
                .map_err(|_| ControlNetError::ResidentBytesOverflow)
        }

        fn execute_controlnet(
            &self,
            _binding: &ControlModelBinding,
            _input: &ControlModelInput,
            _context: &ExecutionContext<'_>,
        ) -> Result<ControlResult, ControlNetError> {
            Err(ControlNetError::Invalid("not exercised".to_owned()))
        }

        fn execute_t2i_adapter(
            &self,
            _binding: &ControlModelBinding,
            _hint: &Tensor,
            _context: &ExecutionContext<'_>,
        ) -> Result<ControlResult, ControlNetError> {
            Err(ControlNetError::Invalid("not exercised".to_owned()))
        }
    }

    #[test]
    fn control_payload_binds_executor_identity_and_accounts_retained_resources()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let hint = tensor_from_f32(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let model_digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let model = ControlModelBinding::checked(
            ModelFamilyIdentity::new("COMFY-MODEL-0001", "control", "v1")?,
            PatchGraphIdentity {
                schema_version: sim_model::PATCH_GRAPH_SCHEMA_VERSION,
                base_artifact_digest: model_digest.to_owned(),
                ordered_digest: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    .to_owned(),
            },
            model_digest,
            EXECUTOR_DIGEST,
            DType::F32,
            DeviceId::CPU,
        )?;
        let control = ControlNet::checked(
            ControlBase::checked(
                1.0,
                StrengthType::Constant,
                ControlPercentWindow::checked(0.0, 1.0)?,
                false,
                None,
            )?,
            model,
            ControlTensorBinding::checked(
                hint,
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            )?,
            1,
            sim_tensor::ResizeMode::NearestExact,
            ControlHintPreprocess::Identity,
            None,
            Vec::new(),
            false,
            Vec::new(),
        )?;
        let chain = Arc::new(ControlChain::checked(vec![ControlNode::ControlNet(
            control,
        )])?);
        let executor = Arc::new(MutableIdentityExecutor {
            changed: AtomicBool::new(false),
        });
        let execution = Arc::new(NativeControlExecution::checked(
            chain.clone(),
            executor.clone(),
        )?);
        let payload =
            NativeControlPayload::checked(NativeModelResourceRole::ControlNet, execution.clone())?;
        assert!(payload.resident_bytes()? > usize::try_from(chain.resident_bytes()?)?);
        let aliased =
            NativeControlPayload::checked(NativeModelResourceRole::ControlNet, execution.clone())?;
        assert_eq!(payload.resident_parts()?, aliased.resident_parts()?);

        let distinct_chain = Arc::new(ControlChain::checked(chain.nodes().to_vec())?);
        let distinct_execution = Arc::new(NativeControlExecution::checked(
            distinct_chain,
            executor.clone(),
        )?);
        let distinct =
            NativeControlPayload::checked(NativeModelResourceRole::ControlNet, distinct_execution)?;
        let shared_parts = payload.resident_parts()?;
        let distinct_parts = distinct.resident_parts()?;
        assert_ne!(shared_parts, distinct_parts);
        let shared_ids = shared_parts
            .shared_allocations()
            .iter()
            .map(NativeDiffusionResidentAllocation::id)
            .collect::<Vec<_>>();
        let distinct_ids = distinct_parts
            .shared_allocations()
            .iter()
            .map(NativeDiffusionResidentAllocation::id)
            .collect::<Vec<_>>();
        assert!(shared_ids.iter().any(|identity| {
            matches!(
                identity,
                NativeDiffusionResidentAllocationId::ControlChainArc { .. }
            ) && !distinct_ids.contains(identity)
        }));
        assert!(shared_ids.iter().any(|identity| {
            matches!(
                identity,
                NativeDiffusionResidentAllocationId::ControlExecutorArc { .. }
            ) && distinct_ids.contains(identity)
        }));
        assert!(shared_ids.iter().any(|identity| {
            matches!(
                identity,
                NativeDiffusionResidentAllocationId::TensorStorage { .. }
            ) && distinct_ids.contains(identity)
        }));
        assert!(shared_ids.iter().any(|identity| {
            matches!(
                identity,
                NativeDiffusionResidentAllocationId::ControlExecutionArc { .. }
            ) && !distinct_ids.contains(identity)
        }));
        assert_eq!(
            u64::try_from(payload.resident_bytes()?)?,
            shared_parts.resident_bytes()?
        );
        payload.validate()?;

        executor.changed.store(true, Ordering::SeqCst);
        assert!(matches!(
            execution.validate(),
            Err(NativeDiffusionPayloadError::Invalid(_))
        ));
        Ok(())
    }

    #[test]
    fn diffusion_resident_parts_sort_deduplicate_and_reject_changed_bytes()
    -> Result<(), Box<dyn Error>> {
        let chain = NativeDiffusionResidentAllocationId::ControlChainArc { address: 7 };
        let model = NativeDiffusionResidentAllocationId::ModelPayloadArc { address: 3 };
        let parts = NativeDiffusionResidentParts::checked(
            11,
            [
                NativeDiffusionResidentAllocation {
                    id: chain.clone(),
                    resident_bytes: 13,
                },
                NativeDiffusionResidentAllocation {
                    id: model.clone(),
                    resident_bytes: 17,
                },
                NativeDiffusionResidentAllocation {
                    id: chain.clone(),
                    resident_bytes: 13,
                },
            ],
        )?;
        assert_eq!(parts.shared_allocations().len(), 2);
        assert_eq!(parts.shared_allocations()[0].id(), &model);
        assert_eq!(parts.shared_allocations()[1].id(), &chain);
        assert_eq!(parts.resident_bytes()?, 41);

        assert!(matches!(
            NativeDiffusionResidentParts::checked(
                0,
                [
                    NativeDiffusionResidentAllocation {
                        id: chain.clone(),
                        resident_bytes: 13,
                    },
                    NativeDiffusionResidentAllocation {
                        id: chain,
                        resident_bytes: 14,
                    },
                ],
            ),
            Err(NativeDiffusionPayloadError::ResidentAllocationChanged)
        ));
        Ok(())
    }

    #[test]
    fn task403_rebind_preserves_executor_and_rejects_cancelled_or_invalid_vae_binding()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let model_digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let model = ControlModelBinding::checked(
            ModelFamilyIdentity::new("COMFY-MODEL-0001", "control", "v1")?,
            PatchGraphIdentity {
                schema_version: sim_model::PATCH_GRAPH_SCHEMA_VERSION,
                base_artifact_digest: model_digest.to_owned(),
                ordered_digest: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    .to_owned(),
            },
            model_digest,
            EXECUTOR_DIGEST,
            DType::F32,
            DeviceId::CPU,
        )?;
        let hint = tensor_from_f32(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let control = ControlNet::checked(
            ControlBase::checked(
                1.0,
                StrengthType::Constant,
                ControlPercentWindow::checked(0.0, 1.0)?,
                false,
                None,
            )?,
            model,
            ControlTensorBinding::checked(
                hint,
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            )?,
            1,
            sim_tensor::ResizeMode::NearestExact,
            ControlHintPreprocess::Identity,
            None,
            Vec::new(),
            false,
            Vec::new(),
        )?;
        let executor: Arc<dyn ControlModelExecutor> = Arc::new(MutableIdentityExecutor {
            changed: AtomicBool::new(false),
        });
        let execution = NativeControlExecution::checked(
            Arc::new(ControlChain::checked(vec![ControlNode::ControlNet(
                control,
            )])?),
            executor,
        )?;
        let request = ControlNetRebindRequest::checked(
            ControlTensorBinding::checked(
                tensor_from_f32(&backend, &[1, 1, 1, 1], &[2.0], &context)?,
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            )?,
            ControlHintLayout::Nhwc,
            0.5,
            ControlPercentWindow::checked(0.1, 0.9)?,
            None,
            Vec::new(),
            false,
            Some(3),
        )?;
        let rebound = execution.rebound(&request, Some(&execution), None, &cancellation)?;
        assert!(execution.shares_executor_with(&rebound));
        assert_eq!(rebound.chain().nodes().len(), 2);
        assert_ne!(execution.execution_digest(), rebound.execution_digest());

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            execution.rebound(&request, None, None, &cancelled),
            Err(NativeDiffusionPayloadError::Cancelled)
        ));
        let wrong_vae = ControlNetRebindRequest::checked(
            ControlTensorBinding::checked(
                tensor_from_f32(&backend, &[1, 1, 1, 1], &[3.0], &context)?,
                "9999999999999999999999999999999999999999999999999999999999999999",
            )?,
            ControlHintLayout::Nchw,
            1.0,
            ControlPercentWindow::checked(0.0, 1.0)?,
            Some("8888888888888888888888888888888888888888888888888888888888888888".to_owned()),
            Vec::new(),
            false,
            None,
        )?;
        assert!(matches!(
            execution.rebound(&wrong_vae, None, None, &cancellation),
            Err(NativeDiffusionPayloadError::Invalid(_))
        ));
        Ok(())
    }
}
