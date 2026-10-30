use crate::{
    NativeLatentTransform, NativePredictionKind, NativeSamplingPolicy, NativeSamplingPolicyKind,
    SamplingPlan, SamplingProfile, SamplingProfileError,
    generated_native_diffusion::{
        NativeDiffusionSamplerError, interpret_prediction_for_profile, model_time_for_profile,
        scale_model_input_for_profile,
    },
    profile_for_model,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::conditioning::{
    ConditioningError, ConditioningIdentity, ConditioningMetadataKey, ConditioningMetadataValue,
    ConditioningSet, ConditioningValue, ResolvedConditioningEntry,
};
use sim_model::{
    ModelFamilyError, NativeLatentOperation, NativeLatentOperationKind, NativeModelPayload,
    model_family::{
        NativeFamilyDenoiserContext, NativeFamilyDenoiserHooks, NativeFamilyDenoiserInvocation,
    },
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceVec, DType, DeviceId, ExecutionContext, Tensor, TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, add_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_18::sub_method_with_context_exact_native,
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const GUIDANCE_ADAPTER_ID: &str = "zed.sim.guidance.v1";
const PYTHON_MATH_ISCLOSE_RELATIVE_TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuidanceBranch {
    Conditional,
    Unconditional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuidanceOptions {
    pub disable_unconditional_skip: bool,
    pub maximum_batch_bytes: u64,
}

impl Default for GuidanceOptions {
    fn default() -> Self {
        Self {
            disable_unconditional_skip: false,
            maximum_batch_bytes: u64::MAX,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuidanceEvaluation {
    branch: GuidanceBranch,
    conditioning_identity: ConditioningIdentity,
    conditioning_digest: String,
    entry: ResolvedConditioningEntry,
    latent: Tensor,
    sigma: f32,
    sampling_percent: f64,
    sampling_profile: Option<SamplingProfileReplacement>,
}

impl GuidanceEvaluation {
    pub fn branch(&self) -> GuidanceBranch {
        self.branch
    }

    pub fn conditioning_identity(&self) -> &ConditioningIdentity {
        &self.conditioning_identity
    }

    pub fn conditioning_digest(&self) -> &str {
        &self.conditioning_digest
    }

    pub fn entry(&self) -> &ResolvedConditioningEntry {
        &self.entry
    }

    pub fn latent(&self) -> &Tensor {
        &self.latent
    }

    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    pub fn sampling_percent(&self) -> f64 {
        self.sampling_percent
    }
}

pub trait GuidanceDenoiser {
    fn evaluate_batch(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError>;

    fn evaluate_batch_with_model_hooks(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        hooks: &dyn NativeFamilyDenoiserHooks,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError> {
        let _ = (evaluations, hooks, context);
        Err(GuidanceError::Invalid(
            "the selected denoiser does not expose native attention hook sites".to_owned(),
        ))
    }
}

pub struct NativeFamilyGuidanceDenoiser<'a> {
    model: &'a NativeModelPayload,
    profile: NativeSamplingPolicy,
    backend: &'a CpuBackend,
}

impl<'a> NativeFamilyGuidanceDenoiser<'a> {
    pub fn checked(
        model: &'a NativeModelPayload,
        backend: &'a CpuBackend,
    ) -> Result<Self, GuidanceError> {
        Self::checked_with_profile(model, profile_for_model(model)?, backend)
    }

    pub fn checked_with_profile(
        model: &'a NativeModelPayload,
        profile: NativeSamplingPolicy,
        backend: &'a CpuBackend,
    ) -> Result<Self, GuidanceError> {
        model
            .validate()
            .map_err(|error| GuidanceError::Invalid(error.to_string()))?;
        profile.validate()?;
        if model.native_family_model_resource().is_none() {
            return Err(GuidanceError::Invalid(
                "native family guidance requires a retained family MODEL".to_owned(),
            ));
        }
        Ok(Self {
            profile,
            model,
            backend,
        })
    }

    pub fn profile(&self) -> &NativeSamplingPolicy {
        &self.profile
    }
}

impl GuidanceDenoiser for NativeFamilyGuidanceDenoiser<'_> {
    fn evaluate_batch(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError> {
        self.evaluate_batch_impl(evaluations, None, context)
    }

    fn evaluate_batch_with_model_hooks(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        hooks: &dyn NativeFamilyDenoiserHooks,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError> {
        self.evaluate_batch_impl(evaluations, Some(hooks), context)
    }
}

impl NativeFamilyGuidanceDenoiser<'_> {
    fn evaluate_batch_impl(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        hooks: Option<&dyn NativeFamilyDenoiserHooks>,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError> {
        let resource = self.model.native_family_model_resource().ok_or_else(|| {
            GuidanceError::Invalid(
                "native family guidance lost its retained family MODEL".to_owned(),
            )
        })?;
        let expected = resource.conditioning_identity();
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(evaluations.len())
            .map_err(|_| GuidanceError::ShapeOverflow("family guidance outputs"))?;
        for evaluation in evaluations {
            context
                .cancellation
                .check()
                .map_err(|_| GuidanceError::Cancelled)?;
            if evaluation.conditioning_identity().model_family() != expected.model_family()
                || evaluation.conditioning_identity().latent_format() != expected.latent_format()
            {
                return Err(GuidanceError::Invalid(
                    "family guidance conditioning identity does not match the MODEL".to_owned(),
                ));
            }
            if !matches!(
                evaluation.entry().value(),
                ConditioningValue::CrossAttention(_)
            ) {
                return Err(GuidanceError::Invalid(
                    "family guidance requires cross-attention conditioning".to_owned(),
                ));
            }
            let replacement_profile = evaluation
                .sampling_profile
                .map(SamplingProfileReplacement::resolve)
                .transpose()?;
            let profile = replacement_profile.as_ref().unwrap_or(&self.profile);
            let scaled_latent = scale_model_input_for_profile(
                profile,
                self.backend,
                evaluation.latent(),
                evaluation.sigma(),
                context,
            )?;
            let model_time = model_time_for_profile(profile, evaluation.sigma())?;
            let batch = scaled_latent
                .descriptor()
                .shape()
                .first()
                .copied()
                .ok_or(GuidanceError::ShapeOverflow("family guidance batch"))?;
            let batch = usize::try_from(batch)
                .map_err(|_| GuidanceError::ShapeOverflow("family guidance batch"))?;
            let mut model_times = Vec::new();
            model_times
                .try_reserve_exact(batch)
                .map_err(|_| GuidanceError::ShapeOverflow("family guidance model time"))?;
            model_times.resize(batch, model_time);
            let batch = u64::try_from(batch)
                .map_err(|_| GuidanceError::ShapeOverflow("family guidance batch"))?;
            let model_time = tensor_from_f32(self.backend, &[batch], &model_times, context)?;
            let family_context = match hooks {
                Some(hooks) => NativeFamilyDenoiserContext::checked_with_hooks(
                    evaluation.conditioning_identity(),
                    context,
                    hooks,
                )?,
                None => NativeFamilyDenoiserContext::checked(
                    evaluation.conditioning_identity(),
                    context,
                )?,
            };
            let reference_latents = match evaluation
                .entry()
                .metadata()
                .get(ConditioningMetadataKey::ReferenceLatents)
            {
                Some(ConditioningMetadataValue::TensorList(tensors)) => tensors.as_slice(),
                Some(_) => {
                    return Err(GuidanceError::Invalid(
                        "reference-latent metadata has an incompatible value".to_owned(),
                    ));
                }
                None => &[],
            };
            let additional_timestep_condition = match evaluation
                .entry()
                .metadata()
                .get(ConditioningMetadataKey::TimeDimensionConcat)
            {
                Some(ConditioningMetadataValue::Tensor(tensor)) => Some(tensor),
                Some(_) => {
                    return Err(GuidanceError::Invalid(
                        "time-dimension metadata has an incompatible value".to_owned(),
                    ));
                }
                None => None,
            };
            let raw = resource.invoke_denoiser(
                self.backend,
                NativeFamilyDenoiserInvocation {
                    scaled_latent: &scaled_latent,
                    model_time: &model_time,
                    conditioning: evaluation.entry(),
                    attention_mask: None,
                    reference_latents,
                    additional_timestep_condition,
                },
                &family_context,
            )?;
            outputs.push(interpret_prediction_for_profile(
                profile,
                self.backend,
                &raw,
                evaluation.latent(),
                evaluation.sigma(),
                context,
            )?);
        }
        Ok(outputs)
    }
}

#[derive(Clone, Debug)]
pub struct GuidancePredictions {
    conditional: Tensor,
    unconditional: Option<Tensor>,
}

impl GuidancePredictions {
    pub fn conditional(&self) -> &Tensor {
        &self.conditional
    }

    pub fn unconditional(&self) -> Option<&Tensor> {
        self.unconditional.as_ref()
    }

    pub fn replace_conditional(&mut self, conditional: Tensor) {
        self.conditional = conditional;
    }

    pub fn replace_unconditional(&mut self, unconditional: Option<Tensor>) {
        self.unconditional = unconditional;
    }
}

pub struct GuidanceHookContext<'a> {
    pub plan: &'a SamplingPlan,
    pub guidance_scale: f32,
    pub sigma: f32,
    pub sampling_percent: f64,
    pub conditional_identity: &'a ConditioningIdentity,
    pub unconditional_identity: &'a ConditioningIdentity,
}

pub trait GuidanceHook {
    fn pre_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &mut GuidancePredictions,
        _context: &ExecutionContext<'_>,
    ) -> Result<(), GuidanceError> {
        Ok(())
    }

    fn override_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &GuidancePredictions,
        _current: &Tensor,
        _context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        Ok(None)
    }

    fn post_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &GuidancePredictions,
        _current: &Tensor,
        _context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        Ok(None)
    }
}

pub const MAX_GUIDANCE_MODIFIERS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeGuidanceModifier {
    LatentOperation {
        operation: NativeLatentOperationKind,
    },
    AdaptiveProjectedGuidance {
        eta: f32,
        norm_threshold: f32,
        momentum: f32,
    },
    CfgOverride {
        scale: f32,
    },
    VideoLinearCfg {
        minimum_scale: f32,
    },
    VideoTriangleCfg {
        minimum_scale: f32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NativeGuidanceModifierChainWire")]
pub struct NativeGuidanceModifierChain {
    modifiers: Vec<NativeGuidanceModifier>,
    semantic_digest_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeGuidanceModifierChainWire {
    modifiers: Vec<NativeGuidanceModifier>,
    semantic_digest_sha256: String,
}

impl TryFrom<NativeGuidanceModifierChainWire> for NativeGuidanceModifierChain {
    type Error = GuidanceError;

    fn try_from(wire: NativeGuidanceModifierChainWire) -> Result<Self, Self::Error> {
        let expected = Self::checked(wire.modifiers)?;
        if expected.semantic_digest_sha256 != wire.semantic_digest_sha256 {
            return Err(GuidanceError::Invalid(
                "guidance modifier identity changed".to_owned(),
            ));
        }
        Ok(expected)
    }
}

impl NativeGuidanceModifierChain {
    pub fn checked(modifiers: Vec<NativeGuidanceModifier>) -> Result<Self, GuidanceError> {
        if modifiers.len() > MAX_GUIDANCE_MODIFIERS {
            return Err(GuidanceError::ModifierLimit {
                actual: modifiers.len(),
                maximum: MAX_GUIDANCE_MODIFIERS,
            });
        }
        for modifier in &modifiers {
            validate_guidance_modifier(*modifier)?;
        }
        let semantic_digest_sha256 = guidance_modifier_digest(&modifiers);
        Ok(Self {
            modifiers,
            semantic_digest_sha256,
        })
    }

    pub fn empty() -> Result<Self, GuidanceError> {
        Self::checked(Vec::new())
    }

    pub fn modifiers(&self) -> &[NativeGuidanceModifier] {
        &self.modifiers
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub fn resident_bytes(&self) -> Result<u64, GuidanceError> {
        let entries = self
            .modifiers
            .capacity()
            .checked_mul(std::mem::size_of::<NativeGuidanceModifier>())
            .ok_or(GuidanceError::ShapeOverflow("guidance modifier residency"))?;
        u64::try_from(
            std::mem::size_of::<Self>()
                .checked_add(entries)
                .and_then(|bytes| bytes.checked_add(self.semantic_digest_sha256.capacity()))
                .ok_or(GuidanceError::ShapeOverflow("guidance modifier residency"))?,
        )
        .map_err(|_| GuidanceError::ShapeOverflow("guidance modifier residency"))
    }

    pub fn validate(&self) -> Result<(), GuidanceError> {
        let expected = Self::checked(self.modifiers.clone())?;
        if expected.semantic_digest_sha256 != self.semantic_digest_sha256 {
            return Err(GuidanceError::Invalid(
                "guidance modifier identity changed".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn begin_attempt<'a>(
        &self,
        backend: &'a CpuBackend,
        attempt_seed: u64,
    ) -> Result<NativeGuidanceModifierSession<'a>, GuidanceError> {
        self.validate()?;
        let states = (0..self.modifiers.len()).map(|_| None).collect();
        Ok(NativeGuidanceModifierSession {
            modifiers: self.modifiers.clone(),
            states,
            attempt_seed,
            backend,
        })
    }
}

#[derive(Debug)]
struct AdaptiveProjectedGuidanceState {
    previous_sigma: Option<f32>,
    running_average: CpuWorkspaceVec<f32>,
}

pub struct NativeGuidanceModifierSession<'a> {
    modifiers: Vec<NativeGuidanceModifier>,
    states: Vec<Option<AdaptiveProjectedGuidanceState>>,
    attempt_seed: u64,
    backend: &'a CpuBackend,
}

impl NativeGuidanceModifierSession<'_> {
    pub fn attempt_seed(&self) -> u64 {
        self.attempt_seed
    }
}

impl GuidanceHook for NativeGuidanceModifierSession<'_> {
    fn pre_cfg(
        &mut self,
        hook: &GuidanceHookContext<'_>,
        predictions: &mut GuidancePredictions,
        context: &ExecutionContext<'_>,
    ) -> Result<(), GuidanceError> {
        for (index, modifier) in self.modifiers.iter().copied().enumerate() {
            if let NativeGuidanceModifier::LatentOperation { operation } = modifier {
                let operation = NativeLatentOperation::checked(operation)
                    .map_err(|error| GuidanceError::Invalid(error.to_string()))?;
                let conditional = if let Some(unconditional) = predictions.unconditional() {
                    let difference = sub_method_with_context_exact_native(
                        self.backend,
                        predictions.conditional(),
                        ElementwiseOperand::Tensor(unconditional),
                        1.0,
                        context,
                    )
                    .map_err(|error| GuidanceError::Invalid(error.to_string()))?;
                    let transformed = operation
                        .apply(&difference, self.backend, context)
                        .map_err(|error| GuidanceError::Invalid(error.to_string()))?;
                    add_method_with_context_exact_native(
                        self.backend,
                        &transformed,
                        ElementwiseOperand::Tensor(unconditional),
                        1.0,
                        context,
                    )
                    .map_err(|error| GuidanceError::Invalid(error.to_string()))?
                } else {
                    operation
                        .apply(predictions.conditional(), self.backend, context)
                        .map_err(|error| GuidanceError::Invalid(error.to_string()))?
                };
                predictions.replace_conditional(conditional);
            }
            if let NativeGuidanceModifier::AdaptiveProjectedGuidance {
                eta,
                norm_threshold,
                momentum,
            } = modifier
            {
                let Some(unconditional) = predictions.unconditional() else {
                    continue;
                };
                let original_conditional_values =
                    tensor_to_f32(self.backend, predictions.conditional(), context)?;
                let mut conditional_values = self
                    .backend
                    .workspace_vec::<f32>(context, original_conditional_values.len())?;
                for value in original_conditional_values.iter().copied() {
                    conditional_values.try_push(value)?;
                }
                let unconditional_values = tensor_to_f32(self.backend, unconditional, context)?;
                let shape = predictions.conditional().descriptor().shape();
                if shape.len() < 4 || conditional_values.len() != unconditional_values.len() {
                    return Err(GuidanceError::Invalid(
                        "APG requires matching rank-four-or-higher predictions".to_owned(),
                    ));
                }
                let group_count = shape
                    .get(..shape.len() - 3)
                    .ok_or(GuidanceError::ShapeOverflow("APG leading dimensions"))?
                    .iter()
                    .try_fold(1_usize, |count, dimension| {
                        let dimension = usize::try_from(*dimension)
                            .map_err(|_| GuidanceError::ShapeOverflow("APG group count"))?;
                        count
                            .checked_mul(dimension)
                            .ok_or(GuidanceError::ShapeOverflow("APG group count"))
                    })?;
                if group_count == 0 || conditional_values.len() % group_count != 0 {
                    return Err(GuidanceError::Invalid(
                        "APG batch geometry is invalid".to_owned(),
                    ));
                }
                let state = self
                    .states
                    .get_mut(index)
                    .ok_or(GuidanceError::ShapeOverflow("APG state"))?;
                let reset = state
                    .as_ref()
                    .and_then(|state| state.previous_sigma)
                    .is_some_and(|previous| hook.sigma > previous);
                if reset {
                    *state = None;
                }
                if state.is_none() {
                    *state = Some(AdaptiveProjectedGuidanceState {
                        previous_sigma: None,
                        running_average: self
                            .backend
                            .workspace_vec::<f32>(context, conditional_values.len())?,
                    });
                }
                let state = state
                    .as_mut()
                    .ok_or(GuidanceError::ShapeOverflow("APG state initialization"))?;
                let mut guidance = self
                    .backend
                    .workspace_vec::<f32>(context, conditional_values.len())?;
                for (conditional, unconditional) in
                    conditional_values.iter().zip(unconditional_values.iter())
                {
                    guidance.try_push(conditional - unconditional)?;
                }
                if momentum != 0.0 {
                    if state.running_average.len() != guidance.len() {
                        state.running_average =
                            self.backend.workspace_vec::<f32>(context, guidance.len())?;
                        for value in guidance.iter().copied() {
                            state.running_average.try_push(value)?;
                        }
                    } else {
                        for (average, current) in
                            state.running_average.iter_mut().zip(guidance.iter())
                        {
                            *average = momentum.mul_add(*average, *current);
                        }
                    }
                    guidance.copy_from_slice(&state.running_average);
                }
                state.previous_sigma = Some(hook.sigma);
                let elements_per_group = guidance.len() / group_count;
                for group_index in 0..group_count {
                    let start = group_index
                        .checked_mul(elements_per_group)
                        .ok_or(GuidanceError::ShapeOverflow("APG batch offset"))?;
                    let end = start
                        .checked_add(elements_per_group)
                        .ok_or(GuidanceError::ShapeOverflow("APG batch end"))?;
                    let row = guidance
                        .get_mut(start..end)
                        .ok_or(GuidanceError::ShapeOverflow("APG batch slice"))?;
                    if norm_threshold > 0.0 {
                        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
                        let scale = if norm > 0.0 {
                            1.0_f32.min(norm_threshold / norm)
                        } else {
                            1.0
                        };
                        for value in row.iter_mut() {
                            *value *= scale;
                        }
                    }
                    let conditional_row = conditional_values
                        .get(start..end)
                        .ok_or(GuidanceError::ShapeOverflow("APG conditional slice"))?;
                    let norm = conditional_row
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        .sqrt();
                    let dot = if norm > 0.0 {
                        row.iter()
                            .zip(conditional_row)
                            .map(|(left, right)| left * (right / norm))
                            .sum::<f32>()
                    } else {
                        0.0
                    };
                    for (offset, value) in row.iter_mut().enumerate() {
                        let unit = if norm > 0.0 {
                            conditional_row[offset] / norm
                        } else {
                            0.0
                        };
                        let parallel = dot * unit;
                        *value = (*value - parallel) + eta * parallel;
                    }
                }
                let scale = hook.guidance_scale;
                if scale == 0.0 {
                    return Err(GuidanceError::Invalid(
                        "APG guidance scale cannot be zero".to_owned(),
                    ));
                }
                for (((conditional, unconditional), modified), original_conditional) in
                    conditional_values
                        .iter_mut()
                        .zip(unconditional_values.iter())
                        .zip(guidance.iter())
                        .zip(original_conditional_values.iter().copied())
                {
                    *conditional =
                        unconditional + modified + (original_conditional - unconditional) / scale;
                }
                predictions.replace_conditional(tensor_from_f32(
                    self.backend,
                    shape,
                    &conditional_values,
                    context,
                )?);
            }
            check_cancel(context)?;
        }
        Ok(())
    }

    fn override_cfg(
        &mut self,
        hook: &GuidanceHookContext<'_>,
        predictions: &GuidancePredictions,
        _current: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        let Some(unconditional) = predictions.unconditional() else {
            return Ok(None);
        };
        let mut result = None;
        for modifier in self.modifiers.iter().copied() {
            let scales = match modifier {
                NativeGuidanceModifier::CfgOverride { scale } => Some(vec![scale]),
                NativeGuidanceModifier::VideoLinearCfg { minimum_scale } => Some(video_scales(
                    predictions.conditional(),
                    minimum_scale,
                    hook.guidance_scale,
                    false,
                )?),
                NativeGuidanceModifier::VideoTriangleCfg { minimum_scale } => Some(video_scales(
                    predictions.conditional(),
                    minimum_scale,
                    hook.guidance_scale,
                    true,
                )?),
                NativeGuidanceModifier::AdaptiveProjectedGuidance { .. } => None,
                NativeGuidanceModifier::LatentOperation { .. } => None,
            };
            if let Some(scales) = scales {
                result = Some(cfg_with_batch_scales(
                    self.backend,
                    predictions.conditional(),
                    unconditional,
                    &scales,
                    context,
                )?);
            }
            check_cancel(context)?;
        }
        Ok(result)
    }
}

fn validate_guidance_modifier(modifier: NativeGuidanceModifier) -> Result<(), GuidanceError> {
    let valid = match modifier {
        NativeGuidanceModifier::LatentOperation { operation } => {
            NativeLatentOperation::checked(operation).is_ok()
        }
        NativeGuidanceModifier::AdaptiveProjectedGuidance {
            eta,
            norm_threshold,
            momentum,
        } => {
            eta.is_finite()
                && (-10.0..=10.0).contains(&eta)
                && norm_threshold.is_finite()
                && (0.0..=50.0).contains(&norm_threshold)
                && momentum.is_finite()
                && (-5.0..=1.0).contains(&momentum)
        }
        NativeGuidanceModifier::CfgOverride { scale } => {
            scale.is_finite() && (0.0..=100.0).contains(&scale)
        }
        NativeGuidanceModifier::VideoLinearCfg { minimum_scale }
        | NativeGuidanceModifier::VideoTriangleCfg { minimum_scale } => {
            minimum_scale.is_finite() && (0.0..=100.0).contains(&minimum_scale)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(GuidanceError::InvalidModifier)
    }
}

fn guidance_modifier_digest(modifiers: &[NativeGuidanceModifier]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.guidance-modifier-chain.v1");
    hasher.update((modifiers.len() as u64).to_le_bytes());
    for modifier in modifiers {
        match *modifier {
            NativeGuidanceModifier::LatentOperation { operation } => {
                hasher.update([4]);
                if let Ok(operation) = NativeLatentOperation::checked(operation) {
                    hasher.update(operation.digest_sha256().as_bytes());
                }
            }
            NativeGuidanceModifier::AdaptiveProjectedGuidance {
                eta,
                norm_threshold,
                momentum,
            } => {
                hasher.update([0]);
                hasher.update(eta.to_bits().to_le_bytes());
                hasher.update(norm_threshold.to_bits().to_le_bytes());
                hasher.update(momentum.to_bits().to_le_bytes());
            }
            NativeGuidanceModifier::CfgOverride { scale } => {
                hasher.update([1]);
                hasher.update(scale.to_bits().to_le_bytes());
            }
            NativeGuidanceModifier::VideoLinearCfg { minimum_scale } => {
                hasher.update([2]);
                hasher.update(minimum_scale.to_bits().to_le_bytes());
            }
            NativeGuidanceModifier::VideoTriangleCfg { minimum_scale } => {
                hasher.update([3]);
                hasher.update(minimum_scale.to_bits().to_le_bytes());
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

fn video_scales(
    conditional: &Tensor,
    minimum_scale: f32,
    maximum_scale: f32,
    triangle: bool,
) -> Result<Vec<f32>, GuidanceError> {
    let batch = conditional
        .descriptor()
        .shape()
        .first()
        .copied()
        .ok_or_else(|| GuidanceError::Invalid("video CFG requires a batch axis".to_owned()))?;
    let batch =
        usize::try_from(batch).map_err(|_| GuidanceError::ShapeOverflow("video CFG batch"))?;
    let mut scales = Vec::new();
    scales
        .try_reserve_exact(batch)
        .map_err(|_| GuidanceError::ShapeOverflow("video CFG scales"))?;
    for index in 0..batch {
        let fraction = if batch <= 1 {
            0.0
        } else {
            index as f32 / (batch - 1) as f32
        };
        let position = if triangle {
            2.0 * (fraction - (fraction + 0.5).floor()).abs()
        } else {
            fraction
        };
        scales.push(position.mul_add(maximum_scale - minimum_scale, minimum_scale));
    }
    Ok(scales)
}

fn cfg_with_batch_scales(
    backend: &CpuBackend,
    conditional: &Tensor,
    unconditional: &Tensor,
    scales: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let shape = conditional.descriptor().shape();
    let batch = shape
        .first()
        .copied()
        .ok_or_else(|| GuidanceError::Invalid("CFG requires a batch axis".to_owned()))?;
    let batch = usize::try_from(batch).map_err(|_| GuidanceError::ShapeOverflow("CFG batch"))?;
    if scales.len() != 1 && scales.len() != batch {
        return Err(GuidanceError::Invalid(
            "CFG scale count does not match batch".to_owned(),
        ));
    }
    let conditional_values = tensor_to_f32(backend, conditional, context)?;
    let unconditional_values = tensor_to_f32(backend, unconditional, context)?;
    if batch == 0
        || conditional_values.len() != unconditional_values.len()
        || conditional_values.len() % batch != 0
    {
        return Err(GuidanceError::Invalid(
            "CFG tensor geometry is invalid".to_owned(),
        ));
    }
    let per_batch = conditional_values.len() / batch;
    let values = conditional_values
        .iter()
        .zip(unconditional_values.iter())
        .enumerate()
        .map(|(index, (conditional, unconditional))| {
            let scale_index = if scales.len() == 1 {
                0
            } else {
                index / per_batch
            };
            scales
                .get(scale_index)
                .copied()
                .map(|scale| (conditional - unconditional).mul_add(scale, *unconditional))
                .ok_or(GuidanceError::ShapeOverflow("CFG scale index"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(tensor_from_f32(backend, shape, &values, context)?)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionHookKind {
    Cfg,
    PostCfg,
    Attention,
    ContextWindow,
    OutputBlock,
    SamplingProfile,
    Rope,
    ModelFunction,
    DiffusionWrapper,
    Hypernetwork,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionHookOperation {
    pub identifier: String,
    pub order: u32,
    pub kind: ExecutionHookKind,
    pub scale: f32,
    pub bias: f32,
    pub start_percent: f64,
    pub end_percent: f64,
    pub rng_address: u64,
    #[serde(default)]
    pub rng_amplitude: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypernetwork: Option<HypernetworkPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<ContextWindowPatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_profile: Option<SamplingProfileReplacement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "profile", rename_all = "snake_case")]
pub enum SamplingProfileReplacement {
    DiscreteEpsilon,
    DiscreteVPrediction,
    DiscreteX0,
    DiscreteLcm,
    DiscreteImg2Img,
    DiscreteImg2ImgFlow,
    StableCascade {
        shift: f32,
    },
    Sd3 {
        shift: f32,
        multiplier: f32,
    },
    AuraFlow {
        shift: f32,
    },
    Flux {
        shift: f32,
    },
    ContinuousEdm {
        sampling: ContinuousSamplingReplacement,
        sigma_min: f32,
        sigma_max: f32,
        sigma_data: f32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuousSamplingReplacement {
    VPrediction,
    Edm,
    EdmPlaygroundV25,
    Epsilon,
    CosmosRFlow,
    ContinuousV,
}

impl SamplingProfileReplacement {
    pub fn resolve(self) -> Result<NativeSamplingPolicy, GuidanceError> {
        match self {
            Self::DiscreteEpsilon => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                    prediction: NativePredictionKind::Epsilon,
                    zero_terminal_snr: false,
                })
            }
            Self::DiscreteVPrediction => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                    prediction: NativePredictionKind::VPrediction,
                    zero_terminal_snr: false,
                })
            }
            Self::DiscreteX0 => NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                prediction: NativePredictionKind::Denoised,
                zero_terminal_snr: false,
            }),
            Self::DiscreteLcm => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                    prediction: NativePredictionKind::Lcm,
                    zero_terminal_snr: false,
                })
            }
            Self::DiscreteImg2Img => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                    prediction: NativePredictionKind::ImageToImage,
                    zero_terminal_snr: false,
                })
            }
            Self::DiscreteImg2ImgFlow => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Discrete {
                    prediction: NativePredictionKind::ImageToImageFlow,
                    zero_terminal_snr: false,
                })
            }
            Self::StableCascade { shift } => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::StableCascade { shift })
            }
            Self::Sd3 { shift, multiplier } => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Sd3 {
                    shift,
                    multiplier,
                    noise_scale: 1.0,
                })
            }
            Self::AuraFlow { shift } => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Flow {
                    shift,
                    multiplier: 1.0,
                    noise_scale: 1.0,
                })
            }
            Self::Flux { shift } => {
                NativeSamplingPolicy::checked(NativeSamplingPolicyKind::Flux { shift })
            }
            Self::ContinuousEdm {
                sampling,
                sigma_min,
                sigma_max,
                sigma_data,
            } => {
                let (prediction, latent) = match sampling {
                    ContinuousSamplingReplacement::VPrediction => (
                        NativePredictionKind::VPrediction,
                        NativeLatentTransform::Identity,
                    ),
                    ContinuousSamplingReplacement::Edm => {
                        (NativePredictionKind::Edm, NativeLatentTransform::Identity)
                    }
                    ContinuousSamplingReplacement::EdmPlaygroundV25 => (
                        NativePredictionKind::Edm,
                        NativeLatentTransform::PlaygroundV25,
                    ),
                    ContinuousSamplingReplacement::Epsilon => (
                        NativePredictionKind::Epsilon,
                        NativeLatentTransform::Identity,
                    ),
                    ContinuousSamplingReplacement::CosmosRFlow => {
                        return NativeSamplingPolicy::checked(
                            NativeSamplingPolicyKind::CosmosRFlow {
                                sigma_min,
                                sigma_max,
                                sigma_data,
                            },
                        )
                        .map_err(GuidanceError::from);
                    }
                    ContinuousSamplingReplacement::ContinuousV => {
                        return NativeSamplingPolicy::checked(
                            NativeSamplingPolicyKind::ContinuousV {
                                sigma_min,
                                sigma_max,
                                sigma_data,
                            },
                        )
                        .map_err(GuidanceError::from);
                    }
                };
                NativeSamplingPolicy::checked_with_latent(
                    NativeSamplingPolicyKind::ContinuousEdm {
                        prediction,
                        sigma_min,
                        sigma_max,
                        sigma_data,
                    },
                    latent,
                )
            }
        }
        .map_err(GuidanceError::from)
    }

    fn validate(self) -> Result<(), GuidanceError> {
        self.resolve().map(|_| ())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypernetworkActivation {
    Linear,
    Relu,
    LeakyRelu,
    Elu,
    HardSwish,
    Tanh,
    Sigmoid,
    SoftSign,
    Mish,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypernetworkLinearLayer {
    pub input_dimension: u64,
    pub output_dimension: u64,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: HypernetworkActivation,
    pub layer_norm_scale: Option<Vec<f32>>,
    pub layer_norm_bias: Option<Vec<f32>>,
    pub dropout_probability: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypernetworkDimensionPatch {
    pub key_layers: Vec<HypernetworkLinearLayer>,
    pub value_layers: Vec<HypernetworkLinearLayer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypernetworkPatch {
    pub strength: f32,
    pub dimensions: BTreeMap<u64, HypernetworkDimensionPatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextWindowFuse {
    Relative,
    Pyramid,
    Flat,
    OverlapLinear,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextWindowSchedule {
    #[default]
    StaticStandard,
    UniformStandard,
    UniformLooped,
    Batched,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextWindowPatch {
    pub context_length: u64,
    pub overlap: u64,
    pub stride: u64,
    pub closed_loop: bool,
    pub fuse: ContextWindowFuse,
    pub dimension: usize,
    #[serde(default)]
    pub schedule: ContextWindowSchedule,
    #[serde(default)]
    pub schedule_step: u64,
    #[serde(default)]
    pub freenoise: bool,
    #[serde(default)]
    pub conditioning_retain_indices: Vec<u64>,
    #[serde(default)]
    pub latent_retain_indices: Vec<u64>,
    #[serde(default)]
    pub split_conditioning: bool,
    #[serde(default)]
    pub causal_window_fix: bool,
    #[serde(default)]
    pub slice_conditioning: bool,
    #[serde(default)]
    pub slice_timestep: bool,
}

impl ContextWindowPatch {
    fn validate(&self) -> Result<(), GuidanceError> {
        if self.context_length == 0
            || self.overlap >= self.context_length
            || self.stride == 0
            || self.stride > 16
        {
            return Err(GuidanceError::Invalid(
                "context-window geometry is invalid".to_owned(),
            ));
        }
        if self.freenoise {
            return Err(GuidanceError::UnsupportedContextWindowFeature("freenoise"));
        }
        if !self.conditioning_retain_indices.is_empty() {
            return Err(GuidanceError::UnsupportedContextWindowFeature(
                "conditioning_retain_indices",
            ));
        }
        if !self.latent_retain_indices.is_empty() {
            return Err(GuidanceError::UnsupportedContextWindowFeature(
                "latent_retain_indices",
            ));
        }
        if self.split_conditioning {
            return Err(GuidanceError::UnsupportedContextWindowFeature(
                "split_conditioning",
            ));
        }
        if self.causal_window_fix {
            return Err(GuidanceError::UnsupportedContextWindowFeature(
                "causal_window_fix",
            ));
        }
        if self.slice_conditioning {
            return Err(GuidanceError::UnsupportedContextWindowFeature(
                "temporal_conditioning_slice",
            ));
        }
        if self.slice_timestep {
            return Err(GuidanceError::UnsupportedContextWindowFeature(
                "timestep_slice",
            ));
        }
        match self.schedule {
            ContextWindowSchedule::UniformStandard => Err(
                GuidanceError::UnsupportedContextWindowFeature("uniform_standard"),
            ),
            ContextWindowSchedule::UniformLooped => Err(
                GuidanceError::UnsupportedContextWindowFeature("uniform_looped"),
            ),
            ContextWindowSchedule::StaticStandard
                if self.closed_loop || self.schedule_step != 0 =>
            {
                Err(GuidanceError::Invalid(
                    "static context windows cannot carry loop or step state".to_owned(),
                ))
            }
            ContextWindowSchedule::Batched
                if self.closed_loop || self.schedule_step != 0 || self.overlap != 0 =>
            {
                Err(GuidanceError::Invalid(
                    "batched context windows require zero overlap, loop, and step".to_owned(),
                ))
            }
            ContextWindowSchedule::StaticStandard | ContextWindowSchedule::Batched => Ok(()),
        }
    }
}

impl HypernetworkPatch {
    fn validate(&self) -> Result<(), GuidanceError> {
        if !self.strength.is_finite() || self.dimensions.is_empty() {
            return Err(GuidanceError::Invalid(
                "hypernetwork strength and dimension map are invalid".to_owned(),
            ));
        }
        if self.dimensions.len() > 64 {
            return Err(GuidanceError::Invalid(
                "hypernetwork has too many dimension entries".to_owned(),
            ));
        }
        let mut parameter_count = 0_usize;
        for (dimension, patch) in &self.dimensions {
            if *dimension == 0 || patch.key_layers.is_empty() || patch.value_layers.is_empty() {
                return Err(GuidanceError::Invalid(
                    "hypernetwork dimension patch is invalid".to_owned(),
                ));
            }
            for layers in [&patch.key_layers, &patch.value_layers] {
                if layers.len() > 32 {
                    return Err(GuidanceError::Invalid(
                        "hypernetwork branch has too many layers".to_owned(),
                    ));
                }
                let mut expected_input = *dimension;
                for layer in layers {
                    let weights = usize::try_from(layer.input_dimension)
                        .ok()
                        .and_then(|input| {
                            usize::try_from(layer.output_dimension)
                                .ok()
                                .and_then(|output| input.checked_mul(output))
                        })
                        .ok_or_else(|| {
                            GuidanceError::Invalid("hypernetwork layer shape overflowed".to_owned())
                        })?;
                    if layer.input_dimension != expected_input
                        || layer.weights.len() != weights
                        || layer.bias.len()
                            != usize::try_from(layer.output_dimension).unwrap_or(usize::MAX)
                        || !layer
                            .weights
                            .iter()
                            .chain(&layer.bias)
                            .all(|value| value.is_finite())
                        || !layer.dropout_probability.is_finite()
                        || !(0.0..1.0).contains(&layer.dropout_probability)
                    {
                        return Err(GuidanceError::Invalid(
                            "hypernetwork linear layer is invalid".to_owned(),
                        ));
                    }
                    match (&layer.layer_norm_scale, &layer.layer_norm_bias) {
                        (Some(scale), Some(bias))
                            if scale.len() == layer.bias.len()
                                && bias.len() == layer.bias.len()
                                && scale.iter().chain(bias).all(|value| value.is_finite()) => {}
                        (None, None) => {}
                        _ => {
                            return Err(GuidanceError::Invalid(
                                "hypernetwork layer norm is invalid".to_owned(),
                            ));
                        }
                    }
                    parameter_count = parameter_count
                        .checked_add(layer.weights.len())
                        .and_then(|count| count.checked_add(layer.bias.len()))
                        .ok_or_else(|| GuidanceError::ShapeOverflow("hypernetwork parameters"))?;
                    expected_input = layer.output_dimension;
                }
                if expected_input != *dimension {
                    return Err(GuidanceError::Invalid(
                        "hypernetwork branch must return its keyed attention dimension".to_owned(),
                    ));
                }
            }
        }
        if parameter_count > 1_048_576 {
            return Err(GuidanceError::Invalid(
                "hypernetwork parameter count exceeds the bounded hook graph".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionHookGraph {
    pub schema_version: u16,
    pub attempt_seed: u64,
    pub operations: Vec<ExecutionHookOperation>,
}

impl ExecutionHookGraph {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn checked(
        attempt_seed: u64,
        mut operations: Vec<ExecutionHookOperation>,
    ) -> Result<Self, GuidanceError> {
        operations.sort_by_key(|operation| operation.order);
        let graph = Self {
            schema_version: Self::SCHEMA_VERSION,
            attempt_seed,
            operations,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<(), GuidanceError> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(GuidanceError::Invalid(
                "execution-hook graph schema is unsupported".to_owned(),
            ));
        }
        if self.operations.len() > 256 {
            return Err(GuidanceError::Invalid(
                "execution-hook graph exceeds the operation bound".to_owned(),
            ));
        }
        let mut identifiers = BTreeSet::new();
        let mut addresses = BTreeSet::new();
        let mut previous_order = None;
        let mut identifier_bytes = 0_usize;
        let mut sampling_profile_count = 0_usize;
        for operation in &self.operations {
            identifier_bytes = identifier_bytes
                .checked_add(operation.identifier.len())
                .ok_or(GuidanceError::ShapeOverflow("execution-hook identifiers"))?;
            if operation.identifier.trim().is_empty()
                || operation.identifier.len() > 4096
                || !operation.scale.is_finite()
                || !operation.bias.is_finite()
                || !operation.start_percent.is_finite()
                || !operation.end_percent.is_finite()
                || !operation.rng_amplitude.is_finite()
                || !(0.0..=1.0).contains(&operation.start_percent)
                || !(0.0..=1.0).contains(&operation.end_percent)
                || operation.start_percent >= operation.end_percent
                || previous_order.is_some_and(|previous| previous >= operation.order)
                || !identifiers.insert(operation.identifier.clone())
                || !addresses.insert(operation.rng_address)
            {
                return Err(GuidanceError::Invalid(
                    "execution-hook graph operation is invalid or unordered".to_owned(),
                ));
            }
            match (&operation.kind, &operation.hypernetwork) {
                (ExecutionHookKind::Hypernetwork, Some(patch)) => patch.validate()?,
                (ExecutionHookKind::Hypernetwork, None) => {
                    return Err(GuidanceError::Invalid(
                        "hypernetwork hook has no dimension-keyed patch".to_owned(),
                    ));
                }
                (_, Some(_)) => {
                    return Err(GuidanceError::Invalid(
                        "only a hypernetwork hook may carry a hypernetwork patch".to_owned(),
                    ));
                }
                (_, None) => {}
            }
            match (&operation.kind, &operation.context_window) {
                (ExecutionHookKind::ContextWindow, Some(window))
                    if operation.scale.to_bits() == 1.0_f32.to_bits()
                        && operation.bias.to_bits() == 0.0_f32.to_bits()
                        && operation.rng_amplitude.to_bits() == 0.0_f32.to_bits() =>
                {
                    window.validate()?;
                }
                (ExecutionHookKind::ContextWindow, _) => {
                    return Err(GuidanceError::Invalid(
                        "context-window hook has no valid window definition".to_owned(),
                    ));
                }
                (_, None) => {}
                (_, Some(_)) => {
                    return Err(GuidanceError::Invalid(
                        "only a context-window hook may carry window state".to_owned(),
                    ));
                }
            }
            match (operation.kind, operation.sampling_profile) {
                (ExecutionHookKind::SamplingProfile, Some(replacement))
                    if operation.scale.to_bits() == 1.0_f32.to_bits()
                        && operation.bias.to_bits() == 0.0_f32.to_bits()
                        && operation.rng_amplitude.to_bits() == 0.0_f32.to_bits()
                        && operation.start_percent.to_bits() == 0.0_f64.to_bits()
                        && operation.end_percent.to_bits() == 1.0_f64.to_bits() =>
                {
                    sampling_profile_count = sampling_profile_count
                        .checked_add(1)
                        .ok_or(GuidanceError::ShapeOverflow("sampling-profile hooks"))?;
                    replacement.validate()?;
                }
                (ExecutionHookKind::SamplingProfile, _) => {
                    return Err(GuidanceError::Invalid(
                        "sampling-profile hook has no valid typed replacement".to_owned(),
                    ));
                }
                (_, None) => {}
                (_, Some(_)) => {
                    return Err(GuidanceError::Invalid(
                        "only a sampling-profile hook may carry profile state".to_owned(),
                    ));
                }
            }
            previous_order = Some(operation.order);
        }
        if identifier_bytes > 64 * 1024 {
            return Err(GuidanceError::Invalid(
                "execution-hook identifiers exceed the aggregate byte bound".to_owned(),
            ));
        }
        if sampling_profile_count > 1 {
            return Err(GuidanceError::Invalid(
                "execution-hook graph has more than one sampling-profile replacement".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn resolved_sampling_profile(&self) -> Result<Option<NativeSamplingPolicy>, GuidanceError> {
        self.validate()?;
        self.operations
            .iter()
            .find_map(|operation| operation.sampling_profile)
            .map(SamplingProfileReplacement::resolve)
            .transpose()
    }

    pub fn identity(&self) -> Result<String, GuidanceError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.execution-hook-graph.v1");
        hasher.update(self.schema_version.to_le_bytes());
        hasher.update(self.attempt_seed.to_le_bytes());
        for operation in &self.operations {
            hasher.update((operation.identifier.len() as u64).to_le_bytes());
            hasher.update(operation.identifier.as_bytes());
            hasher.update(operation.order.to_le_bytes());
            hasher.update([operation.kind as u8]);
            hasher.update(operation.scale.to_bits().to_le_bytes());
            hasher.update(operation.bias.to_bits().to_le_bytes());
            hasher.update(operation.start_percent.to_bits().to_le_bytes());
            hasher.update(operation.end_percent.to_bits().to_le_bytes());
            hasher.update(operation.rng_address.to_le_bytes());
            hasher.update(operation.rng_amplitude.to_bits().to_le_bytes());
            match &operation.hypernetwork {
                Some(patch) => {
                    hasher.update([1]);
                    hasher.update(patch.strength.to_bits().to_le_bytes());
                    for (dimension, dimension_patch) in &patch.dimensions {
                        hasher.update(dimension.to_le_bytes());
                        for layers in [&dimension_patch.key_layers, &dimension_patch.value_layers] {
                            hasher.update(
                                u64::try_from(layers.len())
                                    .map_err(|_| {
                                        GuidanceError::ShapeOverflow("hypernetwork identity")
                                    })?
                                    .to_le_bytes(),
                            );
                            for layer in layers {
                                hasher.update(layer.input_dimension.to_le_bytes());
                                hasher.update(layer.output_dimension.to_le_bytes());
                                hasher.update([layer.activation as u8]);
                                hasher.update(layer.dropout_probability.to_bits().to_le_bytes());
                                for values in [&layer.weights, &layer.bias] {
                                    hasher.update(
                                        u64::try_from(values.len())
                                            .map_err(|_| {
                                                GuidanceError::ShapeOverflow(
                                                    "hypernetwork identity",
                                                )
                                            })?
                                            .to_le_bytes(),
                                    );
                                    for value in values {
                                        hasher.update(value.to_bits().to_le_bytes());
                                    }
                                }
                                for values in [&layer.layer_norm_scale, &layer.layer_norm_bias] {
                                    match values {
                                        Some(values) => {
                                            hasher.update([1]);
                                            for value in values {
                                                hasher.update(value.to_bits().to_le_bytes());
                                            }
                                        }
                                        None => hasher.update([0]),
                                    }
                                }
                            }
                        }
                    }
                }
                None => hasher.update([0]),
            }
            match &operation.context_window {
                Some(window) => {
                    hasher.update([1]);
                    hasher.update(window.context_length.to_le_bytes());
                    hasher.update(window.overlap.to_le_bytes());
                    hasher.update(window.stride.to_le_bytes());
                    hasher.update([
                        u8::from(window.closed_loop),
                        window.fuse as u8,
                        window.schedule as u8,
                        u8::from(window.freenoise),
                        u8::from(window.split_conditioning),
                        u8::from(window.causal_window_fix),
                        u8::from(window.slice_conditioning),
                        u8::from(window.slice_timestep),
                    ]);
                    hasher.update(window.schedule_step.to_le_bytes());
                    hasher.update(
                        u64::try_from(window.dimension)
                            .map_err(|_| GuidanceError::ShapeOverflow("context-window identity"))?
                            .to_le_bytes(),
                    );
                    for indices in [
                        &window.conditioning_retain_indices,
                        &window.latent_retain_indices,
                    ] {
                        hasher.update(
                            u64::try_from(indices.len())
                                .map_err(|_| {
                                    GuidanceError::ShapeOverflow("context-window identity")
                                })?
                                .to_le_bytes(),
                        );
                        for index in indices {
                            hasher.update(index.to_le_bytes());
                        }
                    }
                }
                None => hasher.update([0]),
            }
            match operation.sampling_profile {
                Some(replacement) => {
                    hasher.update([1]);
                    match replacement {
                        SamplingProfileReplacement::DiscreteEpsilon => hasher.update([0]),
                        SamplingProfileReplacement::DiscreteVPrediction => hasher.update([1]),
                        SamplingProfileReplacement::DiscreteX0 => hasher.update([2]),
                        SamplingProfileReplacement::DiscreteLcm => hasher.update([3]),
                        SamplingProfileReplacement::DiscreteImg2Img => hasher.update([4]),
                        SamplingProfileReplacement::DiscreteImg2ImgFlow => hasher.update([5]),
                        SamplingProfileReplacement::StableCascade { shift } => {
                            hasher.update([6]);
                            hasher.update(shift.to_bits().to_le_bytes());
                        }
                        SamplingProfileReplacement::Sd3 { shift, multiplier } => {
                            hasher.update([7]);
                            hasher.update(shift.to_bits().to_le_bytes());
                            hasher.update(multiplier.to_bits().to_le_bytes());
                        }
                        SamplingProfileReplacement::AuraFlow { shift } => {
                            hasher.update([8]);
                            hasher.update(shift.to_bits().to_le_bytes());
                        }
                        SamplingProfileReplacement::Flux { shift } => {
                            hasher.update([9]);
                            hasher.update(shift.to_bits().to_le_bytes());
                        }
                        SamplingProfileReplacement::ContinuousEdm {
                            sampling,
                            sigma_min,
                            sigma_max,
                            sigma_data,
                        } => {
                            hasher.update([10, sampling as u8]);
                            hasher.update(sigma_min.to_bits().to_le_bytes());
                            hasher.update(sigma_max.to_bits().to_le_bytes());
                            hasher.update(sigma_data.to_bits().to_le_bytes());
                        }
                    }
                }
                None => hasher.update([0]),
            }
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_bytes(&self) -> Result<u64, GuidanceError> {
        let operations = self
            .operations
            .capacity()
            .checked_mul(std::mem::size_of::<ExecutionHookOperation>())
            .ok_or(GuidanceError::ShapeOverflow(
                "execution-hook graph residency",
            ))?;
        self.operations.iter().try_fold(
            u64::try_from(std::mem::size_of::<Self>() + operations)
                .map_err(|_| GuidanceError::ShapeOverflow("execution-hook graph residency"))?,
            |total, operation| {
                let total = total
                    .checked_add(u64::try_from(operation.identifier.capacity()).map_err(|_| {
                        GuidanceError::ShapeOverflow("execution-hook graph residency")
                    })?)
                    .ok_or(GuidanceError::ShapeOverflow(
                        "execution-hook graph residency",
                    ))?;
                let total = if let Some(window) = &operation.context_window {
                    let indices = window
                        .conditioning_retain_indices
                        .capacity()
                        .checked_add(window.latent_retain_indices.capacity())
                        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
                        .ok_or(GuidanceError::ShapeOverflow(
                            "execution-hook graph residency",
                        ))?;
                    total
                        .checked_add(u64::try_from(indices).map_err(|_| {
                            GuidanceError::ShapeOverflow("execution-hook graph residency")
                        })?)
                        .ok_or(GuidanceError::ShapeOverflow(
                            "execution-hook graph residency",
                        ))?
                } else {
                    total
                };
                let Some(patch) = &operation.hypernetwork else {
                    return Ok(total);
                };
                patch
                    .dimensions
                    .values()
                    .try_fold(total, |total, dimension| {
                        dimension
                            .key_layers
                            .iter()
                            .chain(&dimension.value_layers)
                            .try_fold(total, |total, layer| {
                                let heap_values = layer
                                    .weights
                                    .capacity()
                                    .checked_add(layer.bias.capacity())
                                    .and_then(|count| {
                                        count.checked_add(
                                            layer
                                                .layer_norm_scale
                                                .as_ref()
                                                .map_or(0, Vec::capacity),
                                        )
                                    })
                                    .and_then(|count| {
                                        count.checked_add(
                                            layer.layer_norm_bias.as_ref().map_or(0, Vec::capacity),
                                        )
                                    })
                                    .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
                                    .ok_or(GuidanceError::ShapeOverflow(
                                        "execution-hook graph residency",
                                    ))?;
                                total
                                    .checked_add(u64::try_from(heap_values).map_err(|_| {
                                        GuidanceError::ShapeOverflow(
                                            "execution-hook graph residency",
                                        )
                                    })?)
                                    .and_then(|total| {
                                        total.checked_add(
                                            u64::try_from(
                                                std::mem::size_of::<(
                                                    u64,
                                                    HypernetworkDimensionPatch,
                                                )>(
                                                ) + 4 * std::mem::size_of::<usize>(),
                                            )
                                            .ok()?,
                                        )
                                    })
                                    .ok_or(GuidanceError::ShapeOverflow(
                                        "execution-hook graph residency",
                                    ))
                            })
                    })
            },
        )
    }
}

struct SerializableGuidanceHook<'a> {
    backend: &'a CpuBackend,
    attempt_seed: u64,
    operation: &'a ExecutionHookOperation,
}

impl GuidanceHook for SerializableGuidanceHook<'_> {
    fn pre_cfg(
        &mut self,
        _hook: &GuidanceHookContext<'_>,
        _predictions: &mut GuidancePredictions,
        _context: &ExecutionContext<'_>,
    ) -> Result<(), GuidanceError> {
        Ok(())
    }

    fn override_cfg(
        &mut self,
        hook: &GuidanceHookContext<'_>,
        _predictions: &GuidancePredictions,
        current: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        if self.operation.kind == ExecutionHookKind::Cfg
            && hook_operation_active(self.operation, hook.sampling_percent)
        {
            return Ok(Some(affine_guidance_tensor(
                self.backend,
                current,
                self.operation.scale,
                self.operation.bias + self.rng_bias(),
                context,
            )?));
        }
        Ok(None)
    }

    fn post_cfg(
        &mut self,
        hook: &GuidanceHookContext<'_>,
        _predictions: &GuidancePredictions,
        current: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<Tensor>, GuidanceError> {
        if self.operation.kind == ExecutionHookKind::PostCfg
            && hook_operation_active(self.operation, hook.sampling_percent)
        {
            return Ok(Some(affine_guidance_tensor(
                self.backend,
                current,
                self.operation.scale,
                self.operation.bias + self.rng_bias(),
                context,
            )?));
        }
        Ok(None)
    }
}

struct SerializedGraphDenoiser<'a> {
    backend: &'a CpuBackend,
    hook_graph: &'a ExecutionHookGraph,
    inner: &'a mut dyn GuidanceDenoiser,
}

impl GuidanceDenoiser for SerializedGraphDenoiser<'_> {
    fn evaluate_batch(
        &mut self,
        evaluations: &[GuidanceEvaluation],
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, GuidanceError> {
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(evaluations.len())
            .map_err(|_| GuidanceError::ShapeOverflow("serialized hook evaluations"))?;
        for evaluation in evaluations {
            outputs.push(evaluate_serialized_operation(
                self.backend,
                self.hook_graph,
                self.inner,
                evaluation.clone(),
                0,
                context,
            )?);
        }
        Ok(outputs)
    }
}

fn evaluate_serialized_operation(
    backend: &CpuBackend,
    hook_graph: &ExecutionHookGraph,
    inner: &mut dyn GuidanceDenoiser,
    mut evaluation: GuidanceEvaluation,
    operation_index: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let Some(operation) = hook_graph.operations.get(operation_index) else {
        let model_hooks = NativeFamilySerializedHooks {
            hook_graph,
            sampling_percent: evaluation.sampling_percent,
        };
        let has_model_hooks = hook_graph.operations.iter().any(|operation| {
            hook_operation_active(operation, evaluation.sampling_percent)
                && matches!(
                    operation.kind,
                    ExecutionHookKind::Attention
                        | ExecutionHookKind::OutputBlock
                        | ExecutionHookKind::Rope
                        | ExecutionHookKind::Hypernetwork
                )
        });
        let outputs = if has_model_hooks {
            inner.evaluate_batch_with_model_hooks(&[evaluation], &model_hooks, context)?
        } else {
            inner.evaluate_batch(&[evaluation], context)?
        };
        return take_single_denoiser_output(outputs);
    };
    let next = operation_index
        .checked_add(1)
        .ok_or(GuidanceError::ShapeOverflow("serialized hook operation"))?;
    if !hook_operation_active(operation, evaluation.sampling_percent) {
        return evaluate_serialized_operation(
            backend, hook_graph, inner, evaluation, next, context,
        );
    }
    let random_bias = operation_rng_bias(hook_graph.attempt_seed, operation);
    match operation.kind {
        ExecutionHookKind::ContextWindow => evaluate_context_windows(
            backend,
            hook_graph,
            inner,
            evaluation,
            next,
            operation.context_window.as_ref().ok_or_else(|| {
                GuidanceError::Invalid("context-window hook lost its definition".to_owned())
            })?,
            context,
        ),
        ExecutionHookKind::SamplingProfile => {
            let replacement = operation.sampling_profile.ok_or_else(|| {
                GuidanceError::Invalid("sampling-profile hook lost its replacement".to_owned())
            })?;
            let profile = replacement.resolve()?;
            evaluation.sampling_percent =
                f64::from(profile.sampling_percent_for_sigma(evaluation.sigma)?);
            evaluation.sampling_profile = Some(replacement);
            evaluate_serialized_operation(backend, hook_graph, inner, evaluation, next, context)
        }
        ExecutionHookKind::ModelFunction => {
            evaluation.latent = affine_guidance_tensor(
                backend,
                &evaluation.latent,
                operation.scale,
                operation.bias + random_bias,
                context,
            )?;
            evaluate_serialized_operation(backend, hook_graph, inner, evaluation, next, context)
        }
        ExecutionHookKind::DiffusionWrapper => {
            let output = evaluate_serialized_operation(
                backend, hook_graph, inner, evaluation, next, context,
            )?;
            affine_guidance_tensor(
                backend,
                &output,
                operation.scale,
                operation.bias + random_bias,
                context,
            )
        }
        ExecutionHookKind::Cfg
        | ExecutionHookKind::PostCfg
        | ExecutionHookKind::Attention
        | ExecutionHookKind::OutputBlock
        | ExecutionHookKind::Rope
        | ExecutionHookKind::Hypernetwork => {
            evaluate_serialized_operation(backend, hook_graph, inner, evaluation, next, context)
        }
    }
}

fn take_single_denoiser_output(outputs: Vec<Tensor>) -> Result<Tensor, GuidanceError> {
    if outputs.len() != 1 {
        return Err(GuidanceError::DenoiserOutputCount {
            expected: 1,
            actual: outputs.len(),
        });
    }
    outputs
        .into_iter()
        .next()
        .ok_or(GuidanceError::DenoiserOutputCount {
            expected: 1,
            actual: 0,
        })
}

struct NativeFamilySerializedHooks<'a> {
    hook_graph: &'a ExecutionHookGraph,
    sampling_percent: f64,
}

impl NativeFamilyDenoiserHooks for NativeFamilySerializedHooks<'_> {
    fn requires_rotary_site(&self) -> bool {
        self.active(ExecutionHookKind::Rope).next().is_some()
    }

    fn after_rotary(
        &self,
        query: &mut [f32],
        key: &mut [f32],
        _channel_dimension: usize,
        execution: &ExecutionContext<'_>,
    ) -> Result<(), ModelFamilyError> {
        for operation in self.active(ExecutionHookKind::Rope) {
            apply_affine_values(
                query,
                operation.scale,
                operation.bias + operation_rng_bias(self.hook_graph.attempt_seed, operation),
                execution,
            )?;
            apply_affine_values(
                key,
                operation.scale,
                operation.bias + operation_rng_bias(self.hook_graph.attempt_seed, operation),
                execution,
            )?;
        }
        Ok(())
    }

    fn before_attention(
        &self,
        query: &mut [f32],
        key: &mut [f32],
        value: &mut [f32],
        channel_dimension: usize,
        execution: &ExecutionContext<'_>,
    ) -> Result<(), ModelFamilyError> {
        for operation in self.hook_graph.operations.iter().filter(|operation| {
            hook_operation_active(operation, self.sampling_percent)
                && matches!(
                    operation.kind,
                    ExecutionHookKind::Attention | ExecutionHookKind::Hypernetwork
                )
        }) {
            match operation.kind {
                ExecutionHookKind::Attention => {
                    let bias = operation.bias
                        + operation_rng_bias(self.hook_graph.attempt_seed, operation);
                    apply_affine_values(query, operation.scale, bias, execution)?;
                    apply_affine_values(key, operation.scale, bias, execution)?;
                    apply_affine_values(value, operation.scale, bias, execution)?;
                }
                ExecutionHookKind::Hypernetwork => apply_hypernetwork_values(
                    key,
                    value,
                    channel_dimension,
                    operation.hypernetwork.as_ref().ok_or_else(|| {
                        ModelFamilyError::DenoiserTensorContract(
                            "hypernetwork hook lost its patch".to_owned(),
                        )
                    })?,
                    self.hook_graph.attempt_seed,
                    operation.rng_address,
                    execution,
                )?,
                _ => {}
            }
        }
        Ok(())
    }

    fn after_attention(
        &self,
        output: &mut [f32],
        _channel_dimension: usize,
        execution: &ExecutionContext<'_>,
    ) -> Result<(), ModelFamilyError> {
        for operation in self.active(ExecutionHookKind::OutputBlock) {
            apply_affine_values(
                output,
                operation.scale,
                operation.bias + operation_rng_bias(self.hook_graph.attempt_seed, operation),
                execution,
            )?;
        }
        Ok(())
    }
}

impl<'a> NativeFamilySerializedHooks<'a> {
    fn active(
        &'a self,
        kind: ExecutionHookKind,
    ) -> impl Iterator<Item = &'a ExecutionHookOperation> + 'a {
        self.hook_graph.operations.iter().filter(move |operation| {
            operation.kind == kind && hook_operation_active(operation, self.sampling_percent)
        })
    }
}

fn apply_affine_values(
    values: &mut [f32],
    scale: f32,
    bias: f32,
    context: &ExecutionContext<'_>,
) -> Result<(), ModelFamilyError> {
    for (index, value) in values.iter_mut().enumerate() {
        if index.is_multiple_of(4_096) {
            context.cancellation.check()?;
        }
        *value = value.mul_add(scale, bias);
    }
    Ok(())
}

fn apply_hypernetwork_values(
    key: &mut [f32],
    value: &mut [f32],
    channel_dimension: usize,
    patch: &HypernetworkPatch,
    attempt_seed: u64,
    rng_address: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), ModelFamilyError> {
    let dimension =
        u64::try_from(channel_dimension).map_err(|_| ModelFamilyError::ForwardShapeOverflow)?;
    let Some(dimension_patch) = patch.dimensions.get(&dimension) else {
        return Err(ModelFamilyError::DenoiserTensorContract(format!(
            "hypernetwork has no branch for attention dimension {channel_dimension}"
        )));
    };
    for (branch_index, (branch, layers)) in [
        (key, &dimension_patch.key_layers),
        (value, &dimension_patch.value_layers),
    ]
    .into_iter()
    .enumerate()
    {
        if !branch.len().is_multiple_of(channel_dimension) {
            return Err(ModelFamilyError::DenoiserTensorContract(
                "hypernetwork attention rows are incomplete".to_owned(),
            ));
        }
        for (row_index, row) in branch.chunks_exact_mut(channel_dimension).enumerate() {
            context.cancellation.check()?;
            let mut current = Vec::new();
            current
                .try_reserve_exact(row.len())
                .map_err(|_| ModelFamilyError::MemoryOverflow)?;
            current.extend_from_slice(row);
            for (layer_index, layer) in layers.iter().enumerate() {
                current = execute_hypernetwork_layer(
                    &current,
                    layer,
                    attempt_seed,
                    rng_address,
                    branch_index as u8,
                    row_index,
                    layer_index,
                )
                .map_err(|error| ModelFamilyError::DenoiserTensorContract(error.to_string()))?;
            }
            for (original, residual) in row.iter_mut().zip(current) {
                *original += residual * patch.strength;
            }
        }
    }
    Ok(())
}

impl SerializableGuidanceHook<'_> {
    fn rng_bias(&self) -> f32 {
        operation_rng_bias(self.attempt_seed, self.operation)
    }
}

fn operation_rng_bias(attempt_seed: u64, operation: &ExecutionHookOperation) -> f32 {
    if operation.rng_amplitude == 0.0 {
        return 0.0;
    }
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.execution-hook-rng.v1");
    hasher.update(attempt_seed.to_le_bytes());
    hasher.update(operation.rng_address.to_le_bytes());
    let digest = hasher.finalize();
    let value = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    (value as f32 / u32::MAX as f32) * operation.rng_amplitude
}

pub fn apply_hypernetwork_attention(
    backend: &CpuBackend,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    patch: &HypernetworkPatch,
    attempt_seed: u64,
    rng_address: u64,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor, Tensor), GuidanceError> {
    patch.validate()?;
    if query.descriptor().shape().is_empty()
        || key.descriptor().shape().is_empty()
        || value.descriptor().shape().is_empty()
        || key.descriptor().shape().last() != value.descriptor().shape().last()
    {
        return Err(GuidanceError::Invalid(
            "hypernetwork attention tensors have incompatible dimensions".to_owned(),
        ));
    }
    Ok((
        query.clone(),
        apply_hypernetwork_tensor(
            backend,
            key,
            patch,
            true,
            attempt_seed,
            rng_address,
            context,
        )?,
        apply_hypernetwork_tensor(
            backend,
            value,
            patch,
            false,
            attempt_seed,
            rng_address,
            context,
        )?,
    ))
}

fn apply_hypernetwork_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    patch: &HypernetworkPatch,
    key: bool,
    attempt_seed: u64,
    rng_address: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let dimension = tensor.descriptor().shape().last().copied().ok_or_else(|| {
        GuidanceError::Invalid("hypernetwork tensor has no channel dimension".to_owned())
    })?;
    let Some(dimension_patch) = patch.dimensions.get(&dimension) else {
        return Ok(tensor.clone());
    };
    let layers = if key {
        &dimension_patch.key_layers
    } else {
        &dimension_patch.value_layers
    };
    let input = tensor_to_f32(backend, tensor, context)?;
    let dimension = usize::try_from(dimension)
        .map_err(|_| GuidanceError::ShapeOverflow("hypernetwork dimension"))?;
    if !input.len().is_multiple_of(dimension) {
        return Err(GuidanceError::Invalid(
            "hypernetwork tensor does not contain complete attention rows".to_owned(),
        ));
    }
    let mut output = backend.workspace_vec(context, input.len())?;
    for (row_index, row) in input.chunks_exact(dimension).enumerate() {
        let mut current = Vec::new();
        current
            .try_reserve_exact(row.len())
            .map_err(|_| GuidanceError::ShapeOverflow("hypernetwork row"))?;
        current.extend_from_slice(row);
        for (layer_index, layer) in layers.iter().enumerate() {
            current = execute_hypernetwork_layer(
                &current,
                layer,
                attempt_seed,
                rng_address,
                u8::from(!key),
                row_index,
                layer_index,
            )?;
        }
        for (original, residual) in row.iter().zip(current) {
            output.try_push(original + residual * patch.strength)?;
        }
    }
    Ok(tensor_from_f32(
        backend,
        tensor.descriptor().shape(),
        &output,
        context,
    )?)
}

fn execute_hypernetwork_layer(
    input: &[f32],
    layer: &HypernetworkLinearLayer,
    attempt_seed: u64,
    rng_address: u64,
    branch_index: u8,
    row_index: usize,
    layer_index: usize,
) -> Result<Vec<f32>, GuidanceError> {
    let input_dimension = usize::try_from(layer.input_dimension)
        .map_err(|_| GuidanceError::ShapeOverflow("hypernetwork input dimension"))?;
    let output_dimension = usize::try_from(layer.output_dimension)
        .map_err(|_| GuidanceError::ShapeOverflow("hypernetwork output dimension"))?;
    if input.len() != input_dimension {
        return Err(GuidanceError::Invalid(
            "hypernetwork layer input dimension changed".to_owned(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_dimension)
        .map_err(|_| GuidanceError::ShapeOverflow("hypernetwork layer output"))?;
    for output_index in 0..output_dimension {
        let start = output_index
            .checked_mul(input_dimension)
            .ok_or(GuidanceError::ShapeOverflow("hypernetwork matrix row"))?;
        let end = start
            .checked_add(input_dimension)
            .ok_or(GuidanceError::ShapeOverflow("hypernetwork matrix row"))?;
        let weights = layer.weights.get(start..end).ok_or_else(|| {
            GuidanceError::Invalid("hypernetwork matrix row is missing".to_owned())
        })?;
        let bias = layer
            .bias
            .get(output_index)
            .copied()
            .ok_or_else(|| GuidanceError::Invalid("hypernetwork bias is missing".to_owned()))?;
        let value = input
            .iter()
            .zip(weights)
            .fold(bias, |sum, (input, weight)| input.mul_add(*weight, sum));
        output.push(apply_hypernetwork_activation(value, layer.activation));
    }
    if let (Some(scale), Some(bias)) = (&layer.layer_norm_scale, &layer.layer_norm_bias) {
        let mean = output.iter().copied().sum::<f32>() / output.len() as f32;
        let variance = output
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / output.len() as f32;
        let inverse = (variance + 1.0e-5).sqrt().recip();
        for ((value, scale), bias) in output.iter_mut().zip(scale).zip(bias) {
            *value = (*value - mean) * inverse * scale + bias;
        }
    }
    if layer.dropout_probability > 0.0 {
        let keep_probability = 1.0 - layer.dropout_probability;
        for (output_index, value) in output.iter_mut().enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(b"zed.sim.hypernetwork-dropout.v1");
            hasher.update(attempt_seed.to_le_bytes());
            hasher.update(rng_address.to_le_bytes());
            hasher.update([branch_index]);
            hasher.update(row_index.to_le_bytes());
            hasher.update(layer_index.to_le_bytes());
            hasher.update(output_index.to_le_bytes());
            let digest = hasher.finalize();
            let random = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as f32
                / u32::MAX as f32;
            *value = if random < keep_probability {
                *value / keep_probability
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

fn apply_hypernetwork_activation(value: f32, activation: HypernetworkActivation) -> f32 {
    match activation {
        HypernetworkActivation::Linear => value,
        HypernetworkActivation::Relu => value.max(0.0),
        HypernetworkActivation::LeakyRelu => value.max(value * 0.01),
        HypernetworkActivation::Elu => {
            if value >= 0.0 {
                value
            } else {
                value.exp() - 1.0
            }
        }
        HypernetworkActivation::HardSwish => value * (value + 3.0).clamp(0.0, 6.0) / 6.0,
        HypernetworkActivation::Tanh => value.tanh(),
        HypernetworkActivation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        HypernetworkActivation::SoftSign => value / (1.0 + value.abs()),
        HypernetworkActivation::Mish => value * value.exp().ln_1p().tanh(),
    }
}

fn evaluate_context_windows(
    backend: &CpuBackend,
    hook_graph: &ExecutionHookGraph,
    inner: &mut dyn GuidanceDenoiser,
    evaluation: GuidanceEvaluation,
    next_operation: usize,
    window: &ContextWindowPatch,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let shape = evaluation.latent.descriptor().shape();
    let axis = shape.get(window.dimension).copied().ok_or_else(|| {
        GuidanceError::Invalid("context-window dimension is unavailable".to_owned())
    })?;
    if axis == 0 || window.context_length > axis || window.stride > 16 {
        return Err(GuidanceError::Invalid(
            "context-window shape is incompatible with the latent".to_owned(),
        ));
    }
    let windows = context_window_indices(axis, window)?;
    if windows.len() > 256 {
        return Err(GuidanceError::Invalid(
            "context-window schedule exceeds the execution bound".to_owned(),
        ));
    }
    let full_values = tensor_to_f32(backend, &evaluation.latent, context)?;
    let axis_length =
        usize::try_from(axis).map_err(|_| GuidanceError::ShapeOverflow("context-window axis"))?;
    let inner_width = shape
        .iter()
        .skip(window.dimension + 1)
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(usize::try_from(*dimension).ok()?)
        })
        .ok_or(GuidanceError::ShapeOverflow("context-window inner width"))?;
    let outer_count = shape
        .iter()
        .take(window.dimension)
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(usize::try_from(*dimension).ok()?)
        })
        .ok_or(GuidanceError::ShapeOverflow("context-window outer count"))?;
    let mut accumulated = backend.workspace_vec(context, full_values.len())?;
    for _ in 0..full_values.len() {
        accumulated.try_push(0.0)?;
    }
    let mut weights = backend.workspace_vec(context, axis_length)?;
    for _ in 0..axis_length {
        weights.try_push(0.0)?;
    }
    for indices in &windows {
        check_cancel(context)?;
        let window_values = slice_context_values(
            &full_values,
            outer_count,
            axis_length,
            inner_width,
            indices,
            backend,
            context,
        )?;
        let mut window_shape = shape.to_vec();
        window_shape[window.dimension] = u64::try_from(indices.len())
            .map_err(|_| GuidanceError::ShapeOverflow("context-window length"))?;
        let mut window_evaluation = evaluation.clone();
        window_evaluation.latent =
            tensor_from_f32(backend, &window_shape, &window_values, context)?;
        let output = evaluate_serialized_operation(
            backend,
            hook_graph,
            inner,
            window_evaluation,
            next_operation,
            context,
        )?;
        if output.descriptor().shape() != window_shape.as_slice() {
            return Err(GuidanceError::Invalid(
                "context-window denoiser changed the window shape".to_owned(),
            ));
        }
        let output_values = tensor_to_f32(backend, &output, context)?;
        fuse_context_values(
            &mut accumulated,
            &mut weights,
            &output_values,
            outer_count,
            axis_length,
            inner_width,
            indices,
            window.fuse,
            usize::try_from(window.overlap)
                .map_err(|_| GuidanceError::ShapeOverflow("context-window overlap"))?,
            context,
        )?;
    }
    if weights.contains(&0.0) {
        return Err(GuidanceError::Invalid(
            "context-window schedule leaves latent positions uncovered".to_owned(),
        ));
    }
    for outer in 0..outer_count {
        for (axis_index, weight) in weights.iter().copied().enumerate() {
            let start = (outer * axis_length + axis_index)
                .checked_mul(inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?;
            for value in accumulated
                .get_mut(start..start + inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?
            {
                *value /= weight;
            }
        }
    }
    Ok(tensor_from_f32(backend, shape, &accumulated, context)?)
}

fn context_window_indices(
    axis: u64,
    window: &ContextWindowPatch,
) -> Result<Vec<Vec<usize>>, GuidanceError> {
    let mut windows = Vec::new();
    if axis <= window.context_length {
        windows.push((0..axis).map(|index| index as usize).collect());
        return Ok(windows);
    }
    if window.schedule == ContextWindowSchedule::Batched {
        let mut start = 0_u64;
        while start < axis {
            let end = start
                .checked_add(window.context_length)
                .map_or(axis, |end| end.min(axis));
            windows.push(
                (start..end)
                    .map(|index| {
                        usize::try_from(index)
                            .map_err(|_| GuidanceError::ShapeOverflow("context-window position"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            start = end;
        }
        return Ok(windows);
    }
    if window.schedule == ContextWindowSchedule::StaticStandard {
        let advance = window.context_length - window.overlap;
        let mut start = 0_u64;
        loop {
            if start + window.context_length >= axis {
                start = axis - window.context_length;
            }
            let indices = (start..start + window.context_length)
                .map(|index| {
                    usize::try_from(index)
                        .map_err(|_| GuidanceError::ShapeOverflow("context-window position"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if windows.last() != Some(&indices) {
                windows.push(indices);
            }
            if start + window.context_length >= axis {
                break;
            }
            start = start
                .checked_add(advance)
                .ok_or(GuidanceError::ShapeOverflow("context-window schedule"))?;
        }
        return Ok(windows);
    }
    let levels = usize::try_from(window.stride)
        .map_err(|_| GuidanceError::ShapeOverflow("context-window stride"))?;
    for level in 0..levels {
        let context_step = 1_u64
            .checked_shl(
                u32::try_from(level)
                    .map_err(|_| GuidanceError::ShapeOverflow("context-window stride"))?,
            )
            .ok_or(GuidanceError::ShapeOverflow("context-window stride"))?;
        let span = window
            .context_length
            .checked_sub(1)
            .and_then(|length| length.checked_mul(context_step))
            .and_then(|length| length.checked_add(1))
            .ok_or(GuidanceError::ShapeOverflow("context-window span"))?;
        if !window.closed_loop && span > axis {
            continue;
        }
        let advance = window
            .context_length
            .checked_mul(context_step)
            .and_then(|length| length.checked_sub(window.overlap.min(length - 1)))
            .ok_or(GuidanceError::ShapeOverflow("context-window advance"))?;
        let mut starts = Vec::new();
        let mut start = 0_u64;
        loop {
            if !window.closed_loop && start + span > axis {
                break;
            }
            starts.push(start);
            start = start
                .checked_add(advance)
                .ok_or(GuidanceError::ShapeOverflow("context-window schedule"))?;
            if start >= axis {
                break;
            }
        }
        if !window.closed_loop {
            let final_start = axis - span;
            if starts.last().copied() != Some(final_start) {
                starts.push(final_start);
            }
        }
        for start in starts {
            let indices = (0..window.context_length)
                .map(|offset| {
                    let position = start + offset * context_step;
                    usize::try_from(if window.closed_loop {
                        position % axis
                    } else {
                        position
                    })
                    .map_err(|_| GuidanceError::ShapeOverflow("context-window position"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !windows.contains(&indices) {
                windows.push(indices);
            }
        }
    }
    Ok(windows)
}

fn slice_context_values(
    full: &[f32],
    outer_count: usize,
    axis_length: usize,
    inner_width: usize,
    indices: &[usize],
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, GuidanceError> {
    let length = outer_count
        .checked_mul(indices.len())
        .and_then(|count| count.checked_mul(inner_width))
        .ok_or(GuidanceError::ShapeOverflow("context-window slice"))?;
    let mut output = backend.workspace_vec(context, length)?;
    for outer in 0..outer_count {
        for index in indices {
            check_cancel(context)?;
            let start = (outer * axis_length + *index)
                .checked_mul(inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window slice"))?;
            for value in full
                .get(start..start + inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window slice"))?
            {
                output.try_push(*value)?;
            }
        }
    }
    Ok(output)
}

fn fuse_context_values(
    accumulated: &mut [f32],
    weights: &mut [f32],
    window_values: &[f32],
    outer_count: usize,
    axis_length: usize,
    inner_width: usize,
    indices: &[usize],
    fuse: ContextWindowFuse,
    context_overlap: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), GuidanceError> {
    let expected = outer_count
        .checked_mul(indices.len())
        .and_then(|count| count.checked_mul(inner_width))
        .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?;
    if window_values.len() != expected {
        return Err(GuidanceError::Invalid(
            "context-window output storage is incompatible".to_owned(),
        ));
    }
    for (position, index) in indices.iter().copied().enumerate() {
        let weight = match fuse {
            ContextWindowFuse::Pyramid => (position + 1).min(indices.len() - position) as f32,
            ContextWindowFuse::Relative => {
                if indices.len() == 1 {
                    1.0
                } else {
                    let center = (indices.len() - 1) as f32 / 2.0;
                    (1.0 - (position as f32 - center).abs() / (center + 0.005)).max(0.01)
                }
            }
            ContextWindowFuse::Flat => 1.0,
            ContextWindowFuse::OverlapLinear => {
                let overlap = context_overlap.min(indices.len().saturating_sub(1));
                let mut weight = 1.0;
                if indices.first().copied().unwrap_or(0) > 0 && position < overlap {
                    weight = if overlap == 1 {
                        1.0e-37
                    } else {
                        1.0e-37 + (1.0 - 1.0e-37) * position as f32 / (overlap - 1) as f32
                    };
                }
                if indices.last().copied().unwrap_or(0) < axis_length.saturating_sub(1)
                    && position >= indices.len().saturating_sub(overlap)
                {
                    let offset = position - (indices.len() - overlap);
                    weight = if overlap == 1 {
                        1.0e-37
                    } else {
                        1.0 - (1.0 - 1.0e-37) * offset as f32 / (overlap - 1) as f32
                    };
                }
                weight
            }
        };
        weights[index] += weight;
        for outer in 0..outer_count {
            check_cancel(context)?;
            let target_start = (outer * axis_length + index)
                .checked_mul(inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?;
            let source_start = (outer * indices.len() + position)
                .checked_mul(inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?;
            let target = accumulated
                .get_mut(target_start..target_start + inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?;
            let source = window_values
                .get(source_start..source_start + inner_width)
                .ok_or(GuidanceError::ShapeOverflow("context-window fusion"))?;
            for (target, source) in target.iter_mut().zip(source) {
                *target += source * weight;
            }
        }
    }
    Ok(())
}

fn hook_operation_active(operation: &ExecutionHookOperation, percent: f64) -> bool {
    operation.start_percent <= percent && percent < operation.end_percent
}

fn affine_guidance_tensor(
    backend: &CpuBackend,
    tensor: &Tensor,
    scale: f32,
    bias: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let values = tensor_to_f32(backend, tensor, context)?;
    let mut output = backend.workspace_vec(context, values.len())?;
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(4096) {
            check_cancel(context)?;
        }
        output.try_push(value.mul_add(scale, bias))?;
    }
    Ok(tensor_from_f32(
        backend,
        tensor.descriptor().shape(),
        &output,
        context,
    )?)
}

#[derive(Clone, Debug)]
pub struct GuidanceResult {
    guided: Tensor,
    unconditional: Option<Tensor>,
    identity: String,
    denoiser_evaluations: usize,
    denoiser_batches: usize,
    unconditional_skipped: bool,
    sampling_percent: f64,
}

impl GuidanceResult {
    pub fn guided(&self) -> &Tensor {
        &self.guided
    }

    pub fn unconditional(&self) -> Option<&Tensor> {
        self.unconditional.as_ref()
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn denoiser_evaluations(&self) -> usize {
        self.denoiser_evaluations
    }

    pub fn denoiser_batches(&self) -> usize {
        self.denoiser_batches
    }

    pub fn unconditional_skipped(&self) -> bool {
        self.unconditional_skipped
    }

    pub fn sampling_percent(&self) -> f64 {
        self.sampling_percent
    }
}

#[derive(Debug, Error)]
pub enum GuidanceError {
    #[error("guidance modifier chain contains {actual} entries; maximum is {maximum}")]
    ModifierLimit { actual: usize, maximum: usize },
    #[error("guidance modifier parameters are invalid")]
    InvalidModifier,
    #[error(transparent)]
    Conditioning(#[from] ConditioningError),
    #[error(transparent)]
    Profile(#[from] SamplingProfileError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Indexing(#[from] IndexingMaskingPartOneError),
    #[error(transparent)]
    Elementwise(#[from] ElementwiseRuntimePartSixteenError),
    #[error(transparent)]
    NativeDiffusion(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Sampler(#[from] NativeDiffusionSamplerError),
    #[error(transparent)]
    FamilyModel(#[from] ModelFamilyError),
    #[error("guidance execution was cancelled")]
    Cancelled,
    #[error("guidance contract is invalid: {0}")]
    Invalid(String),
    #[error("guidance denoiser returned {actual} outputs for a batch of {expected}")]
    DenoiserOutputCount { expected: usize, actual: usize },
    #[error("guidance batch requires {required} bytes, above its {limit}-byte limit")]
    BatchMemoryLimit { required: u64, limit: u64 },
    #[error("guidance shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("guidance has no active conditioning at sampling percent {0}")]
    NoActiveConditioning(f64),
    #[error("sampling-profile replacement {0} is represented but not executable natively")]
    UnsupportedSamplingProfileReplacement(&'static str),
    #[error("context-window source feature {0} is represented but not executable natively")]
    UnsupportedContextWindowFeature(&'static str),
    #[error("guidance regional output leaves an uncovered latent element")]
    UncoveredRegion,
}

pub fn sampling_percent_for_sigma(
    profile: &dyn SamplingProfile,
    sigma: f32,
) -> Result<f64, GuidanceError> {
    Ok(f64::from(profile.sampling_percent_for_sigma(sigma)?))
}

pub fn execute_guidance(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    conditional: &ConditioningSet,
    unconditional: &ConditioningSet,
    options: GuidanceOptions,
    denoiser: &mut dyn GuidanceDenoiser,
    hooks: &mut [&mut dyn GuidanceHook],
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    check_cancel(context)?;
    validate_boundary(latent, profile, plan, conditional, unconditional, options)?;
    let sampling_percent = sampling_percent_for_sigma(profile, sigma)?;
    let unconditional_skipped =
        !options.disable_unconditional_skip && source_math_isclose_to_one(plan.guidance());

    let mut evaluations = build_evaluations(
        backend,
        latent,
        sigma,
        sampling_percent,
        GuidanceBranch::Conditional,
        conditional,
        context,
    )?;
    if evaluations.is_empty() {
        return Err(GuidanceError::NoActiveConditioning(sampling_percent));
    }
    if !unconditional_skipped {
        let unconditional_evaluations = build_evaluations(
            backend,
            latent,
            sigma,
            sampling_percent,
            GuidanceBranch::Unconditional,
            unconditional,
            context,
        )?;
        if unconditional_evaluations.is_empty() {
            return Err(GuidanceError::NoActiveConditioning(sampling_percent));
        }
        evaluations
            .try_reserve_exact(unconditional_evaluations.len())
            .map_err(|_| GuidanceError::ShapeOverflow("guidance evaluations"))?;
        evaluations.extend(unconditional_evaluations);
    }

    let (outputs, denoiser_batches) = evaluate_in_compatible_batches(
        &evaluations,
        options.maximum_batch_bytes,
        denoiser,
        context,
    )?;
    check_cancel(context)?;

    let conditional_prediction = accumulate_branch(
        backend,
        latent,
        GuidanceBranch::Conditional,
        &evaluations,
        &outputs,
        context,
    )?;
    let unconditional_prediction = if unconditional_skipped {
        None
    } else {
        Some(accumulate_branch(
            backend,
            latent,
            GuidanceBranch::Unconditional,
            &evaluations,
            &outputs,
            context,
        )?)
    };
    let mut predictions = GuidancePredictions {
        conditional: conditional_prediction,
        unconditional: unconditional_prediction,
    };
    let hook_context = GuidanceHookContext {
        plan,
        guidance_scale: plan.guidance(),
        sigma,
        sampling_percent,
        conditional_identity: conditional.identity(),
        unconditional_identity: unconditional.identity(),
    };

    for hook in hooks.iter_mut() {
        check_cancel(context)?;
        hook.pre_cfg(&hook_context, &mut predictions, context)?;
        check_cancel(context)?;
        validate_prediction_descriptors(latent, &predictions)?;
    }
    let unconditional_output = predictions.unconditional.clone();
    check_cancel(context)?;
    let mut guided = default_cfg(backend, &predictions, plan.guidance(), context)?;
    for hook in hooks.iter_mut() {
        check_cancel(context)?;
        let overridden = hook.override_cfg(&hook_context, &predictions, &guided, context)?;
        check_cancel(context)?;
        if let Some(overridden) = overridden {
            require_same_descriptor(latent, &overridden, "CFG override")?;
            guided = overridden;
        }
    }
    for hook in hooks.iter_mut() {
        check_cancel(context)?;
        let post_processed = hook.post_cfg(&hook_context, &predictions, &guided, context)?;
        check_cancel(context)?;
        if let Some(post_processed) = post_processed {
            require_same_descriptor(latent, &post_processed, "post-CFG hook")?;
            guided = post_processed;
        }
    }
    check_cancel(context)?;
    let identity = guidance_identity(
        plan,
        sigma,
        conditional,
        unconditional,
        &guided,
        evaluations.len(),
        denoiser_batches,
        unconditional_skipped,
        backend,
        context,
    )?;
    check_cancel(context)?;
    Ok(GuidanceResult {
        guided,
        unconditional: unconditional_output,
        identity,
        denoiser_evaluations: evaluations.len(),
        denoiser_batches,
        unconditional_skipped,
        sampling_percent,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dual_cfg_guidance(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    positive: &ConditioningSet,
    middle: &ConditioningSet,
    negative: &ConditioningSet,
    first_scale: f32,
    second_scale: f32,
    nested: bool,
    options: GuidanceOptions,
    denoiser: &mut dyn GuidanceDenoiser,
    hooks: &mut [&mut dyn GuidanceHook],
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    check_cancel(context)?;
    if !first_scale.is_finite()
        || !second_scale.is_finite()
        || first_scale < 0.0
        || second_scale < 0.0
    {
        return Err(GuidanceError::Invalid(
            "dual CFG scales are invalid".to_owned(),
        ));
    }
    validate_boundary(latent, profile, plan, positive, negative, options)?;
    middle.validate()?;
    let sampling_percent = sampling_percent_for_sigma(profile, sigma)?;
    let mut branch_evaluations = Vec::new();
    for conditioning in [negative, middle, positive] {
        let evaluations = build_evaluations(
            backend,
            latent,
            sigma,
            sampling_percent,
            GuidanceBranch::Conditional,
            conditioning,
            context,
        )?;
        if evaluations.is_empty() {
            return Err(GuidanceError::NoActiveConditioning(sampling_percent));
        }
        branch_evaluations.push(evaluations);
    }
    let mut evaluations = Vec::new();
    let mut ranges = Vec::new();
    for branch in &branch_evaluations {
        let start = evaluations.len();
        evaluations
            .try_reserve_exact(branch.len())
            .map_err(|_| GuidanceError::ShapeOverflow("dual CFG evaluations"))?;
        evaluations.extend(branch.iter().cloned());
        ranges.push(start..evaluations.len());
    }
    let (outputs, denoiser_batches) = evaluate_in_compatible_batches(
        &evaluations,
        options.maximum_batch_bytes,
        denoiser,
        context,
    )?;
    let mut accumulated = Vec::new();
    for range in &ranges {
        accumulated.push(accumulate_branch(
            backend,
            latent,
            GuidanceBranch::Conditional,
            evaluations
                .get(range.clone())
                .ok_or(GuidanceError::ShapeOverflow("dual CFG evaluation range"))?,
            outputs
                .get(range.clone())
                .ok_or(GuidanceError::ShapeOverflow("dual CFG output range"))?,
            context,
        )?);
    }
    let negative_prediction = accumulated
        .first()
        .ok_or(GuidanceError::ShapeOverflow("dual CFG negative"))?;
    let middle_prediction = accumulated
        .get(1)
        .ok_or(GuidanceError::ShapeOverflow("dual CFG middle"))?;
    let positive_prediction = accumulated
        .get(2)
        .ok_or(GuidanceError::ShapeOverflow("dual CFG positive"))?;
    let guided = if nested {
        let text_guided = cfg_pair_with_hooks(
            backend,
            latent,
            positive_prediction,
            middle_prediction,
            first_scale,
            plan,
            sigma,
            sampling_percent,
            positive.identity(),
            middle.identity(),
            hooks,
            context,
        )?;
        cfg_pair(
            backend,
            negative_prediction,
            &text_guided,
            second_scale,
            context,
        )?
    } else {
        let second_guided = cfg_pair_with_hooks(
            backend,
            latent,
            middle_prediction,
            negative_prediction,
            second_scale,
            plan,
            sigma,
            sampling_percent,
            middle.identity(),
            negative.identity(),
            hooks,
            context,
        )?;
        let text_delta = add_method_with_context_exact_native(
            backend,
            positive_prediction,
            ElementwiseOperand::Tensor(middle_prediction),
            -1.0,
            context,
        )?;
        add_method_with_context_exact_native(
            backend,
            &second_guided,
            ElementwiseOperand::Tensor(&text_delta),
            first_scale,
            context,
        )?
    };
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.dual-cfg-result.v1");
    hasher.update(first_scale.to_bits().to_le_bytes());
    hasher.update(second_scale.to_bits().to_le_bytes());
    hasher.update([u8::from(nested)]);
    hasher.update(guided.contiguous_bytes()?);
    Ok(GuidanceResult {
        guided,
        unconditional: Some(negative_prediction.clone()),
        identity: format!("{:x}", hasher.finalize()),
        denoiser_evaluations: evaluations.len(),
        denoiser_batches,
        unconditional_skipped: false,
        sampling_percent,
    })
}

#[allow(clippy::too_many_arguments)]
fn cfg_pair_with_hooks(
    backend: &CpuBackend,
    latent: &Tensor,
    conditional: &Tensor,
    unconditional: &Tensor,
    scale: f32,
    plan: &SamplingPlan,
    sigma: f32,
    sampling_percent: f64,
    conditional_identity: &ConditioningIdentity,
    unconditional_identity: &ConditioningIdentity,
    hooks: &mut [&mut dyn GuidanceHook],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let hook_context = GuidanceHookContext {
        plan,
        guidance_scale: scale,
        sigma,
        sampling_percent,
        conditional_identity,
        unconditional_identity,
    };
    let mut predictions = GuidancePredictions {
        conditional: conditional.clone(),
        unconditional: Some(unconditional.clone()),
    };
    for hook in hooks.iter_mut() {
        hook.pre_cfg(&hook_context, &mut predictions, context)?;
        check_cancel(context)?;
        validate_prediction_descriptors(latent, &predictions)?;
    }
    let mut guided = default_cfg(backend, &predictions, scale, context)?;
    for hook in hooks.iter_mut() {
        if let Some(replacement) =
            hook.override_cfg(&hook_context, &predictions, &guided, context)?
        {
            require_same_descriptor(latent, &replacement, "dual CFG override")?;
            guided = replacement;
        }
        check_cancel(context)?;
    }
    for hook in hooks.iter_mut() {
        if let Some(replacement) = hook.post_cfg(&hook_context, &predictions, &guided, context)? {
            require_same_descriptor(latent, &replacement, "dual CFG post-CFG hook")?;
            guided = replacement;
        }
        check_cancel(context)?;
    }
    Ok(guided)
}

fn cfg_pair(
    backend: &CpuBackend,
    unconditional: &Tensor,
    conditional: &Tensor,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let predictions = GuidancePredictions {
        conditional: conditional.clone(),
        unconditional: Some(unconditional.clone()),
    };
    default_cfg(backend, &predictions, scale, context)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dual_model_guidance(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    positive: &ConditioningSet,
    negative: &ConditioningSet,
    options: GuidanceOptions,
    positive_denoiser: &mut dyn GuidanceDenoiser,
    negative_denoiser: &mut dyn GuidanceDenoiser,
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    execute_dual_model_guidance_with_hooks(
        backend,
        latent,
        sigma,
        profile,
        plan,
        positive,
        negative,
        options,
        positive_denoiser,
        negative_denoiser,
        &mut [],
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_dual_model_guidance_with_hooks(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    positive: &ConditioningSet,
    negative: &ConditioningSet,
    options: GuidanceOptions,
    positive_denoiser: &mut dyn GuidanceDenoiser,
    negative_denoiser: &mut dyn GuidanceDenoiser,
    hooks: &mut [&mut dyn GuidanceHook],
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    validate_boundary(latent, profile, plan, positive, negative, options)?;
    let sampling_percent = sampling_percent_for_sigma(profile, sigma)?;
    let (conditional, positive_evaluations, positive_batches) =
        execute_single_conditioning_prediction(
            backend,
            latent,
            sigma,
            sampling_percent,
            positive,
            options.maximum_batch_bytes,
            positive_denoiser,
            context,
        )?;
    let unconditional_skipped =
        source_math_isclose_to_one(plan.guidance()) && !options.disable_unconditional_skip;
    let (unconditional, negative_evaluations, negative_batches) = if unconditional_skipped {
        (None, 0, 0)
    } else {
        let (unconditional, evaluations, batches) = execute_single_conditioning_prediction(
            backend,
            latent,
            sigma,
            sampling_percent,
            negative,
            options.maximum_batch_bytes,
            negative_denoiser,
            context,
        )?;
        (Some(unconditional), evaluations, batches)
    };
    let hook_context = GuidanceHookContext {
        plan,
        guidance_scale: plan.guidance(),
        sigma,
        sampling_percent,
        conditional_identity: positive.identity(),
        unconditional_identity: negative.identity(),
    };
    let mut predictions = GuidancePredictions {
        conditional,
        unconditional,
    };
    for hook in hooks.iter_mut() {
        check_cancel(context)?;
        hook.pre_cfg(&hook_context, &mut predictions, context)?;
        validate_prediction_descriptors(latent, &predictions)?;
    }
    let unconditional_output = predictions.unconditional.clone();
    let mut guided = default_cfg(backend, &predictions, plan.guidance(), context)?;
    for hook in hooks.iter_mut() {
        check_cancel(context)?;
        if let Some(replacement) =
            hook.override_cfg(&hook_context, &predictions, &guided, context)?
        {
            require_same_descriptor(latent, &replacement, "dual-model CFG override")?;
            guided = replacement;
        }
    }
    for hook in hooks.iter_mut() {
        check_cancel(context)?;
        if let Some(replacement) = hook.post_cfg(&hook_context, &predictions, &guided, context)? {
            require_same_descriptor(latent, &replacement, "dual-model post-CFG hook")?;
            guided = replacement;
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.dual-model-guidance.v1");
    hasher.update(plan.guidance().to_bits().to_le_bytes());
    hasher.update(guided.contiguous_bytes()?);
    Ok(GuidanceResult {
        guided,
        unconditional: unconditional_output,
        identity: format!("{:x}", hasher.finalize()),
        denoiser_evaluations: positive_evaluations + negative_evaluations,
        denoiser_batches: positive_batches + negative_batches,
        unconditional_skipped,
        sampling_percent,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_single_conditioning_prediction(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    sampling_percent: f64,
    conditioning: &ConditioningSet,
    maximum_batch_bytes: u64,
    denoiser: &mut dyn GuidanceDenoiser,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, usize, usize), GuidanceError> {
    let evaluations = build_evaluations(
        backend,
        latent,
        sigma,
        sampling_percent,
        GuidanceBranch::Conditional,
        conditioning,
        context,
    )?;
    if evaluations.is_empty() {
        return Err(GuidanceError::NoActiveConditioning(sampling_percent));
    }
    let (outputs, batches) =
        evaluate_in_compatible_batches(&evaluations, maximum_batch_bytes, denoiser, context)?;
    let prediction = accumulate_branch(
        backend,
        latent,
        GuidanceBranch::Conditional,
        &evaluations,
        &outputs,
        context,
    )?;
    Ok((prediction, evaluations.len(), batches))
}

#[allow(clippy::too_many_arguments)]
pub fn execute_guidance_with_hook_graph(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    conditional: &ConditioningSet,
    unconditional: &ConditioningSet,
    options: GuidanceOptions,
    denoiser: &mut dyn GuidanceDenoiser,
    hook_graph: &ExecutionHookGraph,
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    let modifiers = NativeGuidanceModifierChain::empty()?;
    execute_guidance_with_attached_modifiers(
        backend,
        latent,
        sigma,
        profile,
        plan,
        conditional,
        unconditional,
        options,
        denoiser,
        hook_graph,
        &modifiers,
        hook_graph.attempt_seed,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_guidance_with_attached_modifiers(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    conditional: &ConditioningSet,
    unconditional: &ConditioningSet,
    options: GuidanceOptions,
    denoiser: &mut dyn GuidanceDenoiser,
    hook_graph: &ExecutionHookGraph,
    modifiers: &NativeGuidanceModifierChain,
    attempt_seed: u64,
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    hook_graph.validate()?;
    modifiers.validate()?;
    let mut modifier_session = modifiers.begin_attempt(backend, attempt_seed)?;
    execute_guidance_with_modifier_session(
        backend,
        latent,
        sigma,
        profile,
        plan,
        conditional,
        unconditional,
        options,
        denoiser,
        hook_graph,
        &mut modifier_session,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_guidance_with_modifier_session(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    conditional: &ConditioningSet,
    unconditional: &ConditioningSet,
    options: GuidanceOptions,
    denoiser: &mut dyn GuidanceDenoiser,
    hook_graph: &ExecutionHookGraph,
    modifier_session: &mut NativeGuidanceModifierSession<'_>,
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    hook_graph.validate()?;
    let mut adapters = hook_graph
        .operations
        .iter()
        .filter(|operation| {
            matches!(
                operation.kind,
                ExecutionHookKind::Cfg | ExecutionHookKind::PostCfg
            )
        })
        .map(|operation| SerializableGuidanceHook {
            backend,
            attempt_seed: hook_graph.attempt_seed,
            operation,
        })
        .collect::<Vec<_>>();
    let mut hooks = adapters
        .iter_mut()
        .map(|hook| hook as &mut dyn GuidanceHook)
        .collect::<Vec<_>>();
    hooks.push(modifier_session as &mut dyn GuidanceHook);
    let mut denoiser = SerializedGraphDenoiser {
        backend,
        hook_graph,
        inner: denoiser,
    };
    execute_guidance(
        backend,
        latent,
        sigma,
        profile,
        plan,
        conditional,
        unconditional,
        options,
        &mut denoiser,
        &mut hooks,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dual_cfg_guidance_with_modifier_session(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    positive: &ConditioningSet,
    middle: &ConditioningSet,
    negative: &ConditioningSet,
    first_scale: f32,
    second_scale: f32,
    nested: bool,
    options: GuidanceOptions,
    denoiser: &mut dyn GuidanceDenoiser,
    hook_graph: &ExecutionHookGraph,
    modifier_session: &mut NativeGuidanceModifierSession<'_>,
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    hook_graph.validate()?;
    let mut adapters = hook_graph
        .operations
        .iter()
        .filter(|operation| {
            matches!(
                operation.kind,
                ExecutionHookKind::Cfg | ExecutionHookKind::PostCfg
            )
        })
        .map(|operation| SerializableGuidanceHook {
            backend,
            attempt_seed: hook_graph.attempt_seed,
            operation,
        })
        .collect::<Vec<_>>();
    let mut hooks = adapters
        .iter_mut()
        .map(|hook| hook as &mut dyn GuidanceHook)
        .collect::<Vec<_>>();
    hooks.push(modifier_session as &mut dyn GuidanceHook);
    let mut denoiser = SerializedGraphDenoiser {
        backend,
        hook_graph,
        inner: denoiser,
    };
    execute_dual_cfg_guidance(
        backend,
        latent,
        sigma,
        profile,
        plan,
        positive,
        middle,
        negative,
        first_scale,
        second_scale,
        nested,
        options,
        &mut denoiser,
        &mut hooks,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_dual_model_guidance_with_modifier_session(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    positive: &ConditioningSet,
    negative: &ConditioningSet,
    options: GuidanceOptions,
    positive_denoiser: &mut dyn GuidanceDenoiser,
    negative_denoiser: &mut dyn GuidanceDenoiser,
    positive_hook_graph: &ExecutionHookGraph,
    negative_hook_graph: &ExecutionHookGraph,
    modifier_session: &mut NativeGuidanceModifierSession<'_>,
    context: &ExecutionContext<'_>,
) -> Result<GuidanceResult, GuidanceError> {
    positive_hook_graph.validate()?;
    negative_hook_graph.validate()?;
    let mut adapters = positive_hook_graph
        .operations
        .iter()
        .filter(|operation| {
            matches!(
                operation.kind,
                ExecutionHookKind::Cfg | ExecutionHookKind::PostCfg
            )
        })
        .map(|operation| SerializableGuidanceHook {
            backend,
            attempt_seed: positive_hook_graph.attempt_seed,
            operation,
        })
        .collect::<Vec<_>>();
    let mut hooks = adapters
        .iter_mut()
        .map(|hook| hook as &mut dyn GuidanceHook)
        .collect::<Vec<_>>();
    hooks.push(modifier_session as &mut dyn GuidanceHook);
    let mut positive_denoiser = SerializedGraphDenoiser {
        backend,
        hook_graph: positive_hook_graph,
        inner: positive_denoiser,
    };
    let mut negative_denoiser = SerializedGraphDenoiser {
        backend,
        hook_graph: negative_hook_graph,
        inner: negative_denoiser,
    };
    execute_dual_model_guidance_with_hooks(
        backend,
        latent,
        sigma,
        profile,
        plan,
        positive,
        negative,
        options,
        &mut positive_denoiser,
        &mut negative_denoiser,
        &mut hooks,
        context,
    )
}

fn validate_boundary(
    latent: &Tensor,
    profile: &dyn SamplingProfile,
    plan: &SamplingPlan,
    conditional: &ConditioningSet,
    unconditional: &ConditioningSet,
    options: GuidanceOptions,
) -> Result<(), GuidanceError> {
    if latent.descriptor().rank() < 3
        || latent.descriptor().shape().contains(&0)
        || latent.descriptor().dtype() != DType::F32
        || latent.descriptor().device() != DeviceId::CPU
    {
        return Err(GuidanceError::Invalid(
            "guidance requires a nonempty rank-three-or-greater f32 CPU latent".to_owned(),
        ));
    }
    if latent.descriptor().stream() != sim_tensor::StreamId::DEFAULT {
        return Err(GuidanceError::Invalid(
            "guidance CPU execution currently requires the default stream".to_owned(),
        ));
    }
    if plan.profile() != profile.identity() {
        return Err(GuidanceError::Invalid(
            "sampling plan and sampling profile identities differ".to_owned(),
        ));
    }
    for (subject, identity) in [
        ("conditional", conditional.identity()),
        ("unconditional", unconditional.identity()),
    ] {
        if identity.model_family() != conditional.identity().model_family()
            || identity.latent_format() != conditional.identity().latent_format()
        {
            return Err(GuidanceError::Invalid(format!(
                "{subject} conditioning targets a different model or latent format"
            )));
        }
    }
    if options.maximum_batch_bytes == 0 {
        return Err(GuidanceError::Invalid(
            "guidance batch memory limit must be nonzero".to_owned(),
        ));
    }
    Ok(())
}

fn build_evaluations(
    backend: &CpuBackend,
    latent: &Tensor,
    sigma: f32,
    sampling_percent: f64,
    branch: GuidanceBranch,
    conditioning: &ConditioningSet,
    context: &ExecutionContext<'_>,
) -> Result<Vec<GuidanceEvaluation>, GuidanceError> {
    let resolved = conditioning.resolve(latent.descriptor(), backend, context)?;
    let mut evaluations = Vec::new();
    evaluations
        .try_reserve_exact(resolved.len())
        .map_err(|_| GuidanceError::ShapeOverflow("active conditioning evaluations"))?;
    for entry in resolved {
        check_cancel(context)?;
        if !entry.window().contains(sampling_percent) {
            continue;
        }
        if let Some(window) = entry.metadata().sampler_window()?
            && !window.contains(sampling_percent)
        {
            continue;
        }
        let latent = crop_to_region(latent, &entry, context)?;
        evaluations.push(GuidanceEvaluation {
            branch,
            conditioning_identity: conditioning.identity().clone(),
            conditioning_digest: conditioning.digest().to_owned(),
            entry,
            latent,
            sigma,
            sampling_percent,
            sampling_profile: None,
        });
    }
    check_cancel(context)?;
    Ok(evaluations)
}

fn crop_to_region(
    latent: &Tensor,
    entry: &ResolvedConditioningEntry,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    if entry.region().is_full() {
        return Ok(latent.clone());
    }
    let mut cropped = latent.clone();
    for (spatial_axis, (&offset, &size)) in entry
        .region()
        .offsets()
        .iter()
        .zip(entry.region().sizes())
        .enumerate()
    {
        check_cancel(context)?;
        let axis = i64::try_from(spatial_axis + 2)
            .map_err(|_| GuidanceError::ShapeOverflow("regional crop axis"))?;
        let offset = i64::try_from(offset)
            .map_err(|_| GuidanceError::ShapeOverflow("regional crop offset"))?;
        cropped = narrow_method_exact_native(&cropped, axis, offset, size, context.cancellation)?;
    }
    Ok(cropped)
}

fn evaluate_in_compatible_batches(
    evaluations: &[GuidanceEvaluation],
    maximum_batch_bytes: u64,
    denoiser: &mut dyn GuidanceDenoiser,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<Tensor>, usize), GuidanceError> {
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(evaluations.len())
        .map_err(|_| GuidanceError::ShapeOverflow("guidance denoiser outputs"))?;
    let mut batch_start = 0;
    let mut batch_count = 0_usize;
    while batch_start < evaluations.len() {
        check_cancel(context)?;
        let first = evaluations
            .get(batch_start)
            .ok_or(GuidanceError::ShapeOverflow("guidance batch start"))?;
        let first_bytes = evaluation_bytes(first)?;
        if first_bytes > maximum_batch_bytes {
            return Err(GuidanceError::BatchMemoryLimit {
                required: first_bytes,
                limit: maximum_batch_bytes,
            });
        }
        let mut batch_bytes = first_bytes;
        let mut batch_end = batch_start + 1;
        while let Some(candidate) = evaluations.get(batch_end) {
            let candidate_bytes = evaluation_bytes(candidate)?;
            let next_bytes = batch_bytes
                .checked_add(candidate_bytes)
                .ok_or(GuidanceError::ShapeOverflow("guidance batch bytes"))?;
            if next_bytes > maximum_batch_bytes || !evaluations_compatible(first, candidate) {
                break;
            }
            batch_bytes = next_bytes;
            batch_end += 1;
        }
        let batch = evaluations
            .get(batch_start..batch_end)
            .ok_or(GuidanceError::ShapeOverflow("guidance batch range"))?;
        let batch_outputs = denoiser.evaluate_batch(batch, context)?;
        check_cancel(context)?;
        if batch_outputs.len() != batch.len() {
            return Err(GuidanceError::DenoiserOutputCount {
                expected: batch.len(),
                actual: batch_outputs.len(),
            });
        }
        for (evaluation, output) in batch.iter().zip(&batch_outputs) {
            require_same_descriptor(evaluation.latent(), output, "denoiser output")?;
        }
        outputs.extend(batch_outputs);
        batch_count = batch_count
            .checked_add(1)
            .ok_or(GuidanceError::ShapeOverflow("guidance batch count"))?;
        batch_start = batch_end;
    }
    Ok((outputs, batch_count))
}

fn evaluation_bytes(evaluation: &GuidanceEvaluation) -> Result<u64, GuidanceError> {
    let mut bytes = evaluation.latent.descriptor().byte_len()?;
    bytes = bytes
        .checked_add(conditioning_value_bytes(evaluation.entry.value())?)
        .ok_or(GuidanceError::ShapeOverflow("guidance evaluation bytes"))?;
    if let Some(mask) = evaluation.entry.mask() {
        bytes = bytes
            .checked_add(mask.tensor().descriptor().byte_len()?)
            .ok_or(GuidanceError::ShapeOverflow(
                "guidance evaluation mask bytes",
            ))?;
    }
    bytes = bytes
        .checked_add(evaluation.entry.metadata().resident_tensor_bytes()?)
        .ok_or(GuidanceError::ShapeOverflow(
            "guidance evaluation metadata bytes",
        ))?;
    Ok(bytes)
}

fn conditioning_value_bytes(value: &ConditioningValue) -> Result<u64, GuidanceError> {
    match value {
        ConditioningValue::Regular(tensor)
        | ConditioningValue::NoiseShape(tensor)
        | ConditioningValue::CrossAttention(tensor) => Ok(tensor.descriptor().byte_len()?),
        ConditioningValue::Constant(_) => Ok(0),
        ConditioningValue::List(tensors) => tensors.iter().try_fold(0_u64, |bytes, tensor| {
            bytes
                .checked_add(
                    tensor
                        .descriptor()
                        .byte_len()
                        .map_err(GuidanceError::from)?,
                )
                .ok_or(GuidanceError::ShapeOverflow(
                    "guidance conditioning-list bytes",
                ))
        }),
    }
}

fn evaluations_compatible(left: &GuidanceEvaluation, right: &GuidanceEvaluation) -> bool {
    tensor_contract_matches(&left.latent, &right.latent)
        && left.entry.value().can_concat(right.entry.value())
}

fn accumulate_branch(
    backend: &CpuBackend,
    latent: &Tensor,
    branch: GuidanceBranch,
    evaluations: &[GuidanceEvaluation],
    outputs: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let element_count_u64 = latent.descriptor().element_count()?;
    let element_count = usize::try_from(element_count_u64)
        .map_err(|_| GuidanceError::ShapeOverflow("guidance accumulation size"))?;
    let mut sum = zero_workspace(backend, context, element_count)?;
    let mut weight = zero_workspace(backend, context, element_count)?;
    let mut default_sum = zero_workspace(backend, context, element_count)?;
    let mut default_weight = zero_workspace(backend, context, element_count)?;
    let mut saw_entry = false;
    for (evaluation, output) in evaluations.iter().zip(outputs) {
        if evaluation.branch != branch {
            continue;
        }
        saw_entry = true;
        let (target_sum, target_weight) = if evaluation.entry.is_default_region() {
            (&mut default_sum, &mut default_weight)
        } else {
            (&mut sum, &mut weight)
        };
        accumulate_entry(
            backend,
            latent,
            evaluation.entry(),
            output,
            target_sum,
            target_weight,
            context,
        )?;
    }
    if !saw_entry {
        return Err(GuidanceError::NoActiveConditioning(0.0));
    }
    let mut values = backend.workspace_vec(context, element_count)?;
    for index in 0..element_count {
        if index.is_multiple_of(1024) {
            check_cancel(context)?;
        }
        let (sum_value, weight_value) = if weight[index] != 0.0 {
            (sum[index], weight[index])
        } else if default_weight[index] != 0.0 {
            (default_sum[index], default_weight[index])
        } else {
            return Err(GuidanceError::UncoveredRegion);
        };
        let value = sum_value / weight_value;
        if !value.is_finite() {
            return Err(GuidanceError::Invalid(
                "regional guidance produced a non-finite value".to_owned(),
            ));
        }
        values.try_push(value)?;
    }
    check_cancel(context)?;
    Ok(tensor_from_f32(
        backend,
        latent.descriptor().shape(),
        &values,
        context,
    )?)
}

fn zero_workspace(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    length: usize,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, GuidanceError> {
    let mut values = backend.workspace_vec(context, length)?;
    for _ in 0..length {
        values.try_push(0.0)?;
    }
    Ok(values)
}

fn accumulate_entry(
    backend: &CpuBackend,
    full: &Tensor,
    entry: &ResolvedConditioningEntry,
    output: &Tensor,
    sum: &mut [f32],
    weight: &mut [f32],
    context: &ExecutionContext<'_>,
) -> Result<(), GuidanceError> {
    let output_values = tensor_to_f32(backend, output, context)?;
    let mask_values = entry
        .mask()
        .map(|mask| tensor_to_f32(backend, mask.tensor(), context))
        .transpose()?;
    if let Some(mask) = &mask_values {
        if mask.len() != output_values.len() {
            return Err(GuidanceError::Invalid(
                "resolved mask and denoiser output shapes differ".to_owned(),
            ));
        }
    }
    let local_shape = output.descriptor().shape();
    let full_shape = full.descriptor().shape();
    let mut local_coordinates = Vec::new();
    local_coordinates
        .try_reserve_exact(local_shape.len())
        .map_err(|_| GuidanceError::ShapeOverflow("regional coordinates"))?;
    local_coordinates.resize(local_shape.len(), 0_u64);
    for (local_index, output_value) in output_values.iter().copied().enumerate() {
        if local_index.is_multiple_of(1024) {
            check_cancel(context)?;
        }
        decode_row_major_index(local_index, local_shape, &mut local_coordinates)?;
        let spatial_coordinates = local_coordinates
            .get(2..)
            .ok_or(GuidanceError::ShapeOverflow("regional spatial coordinates"))?;
        let mask_weight = match (entry.mask(), mask_values.as_ref()) {
            (Some(mask), Some(mask_values)) => {
                mask.feather_weight(spatial_coordinates)? * mask_values[local_index]
            }
            (None, None) => 1.0,
            _ => {
                return Err(GuidanceError::Invalid(
                    "resolved mask state is inconsistent".to_owned(),
                ));
            }
        };
        let contribution_weight = entry.strength_for_cpu_f32()? * mask_weight;
        if !contribution_weight.is_finite() {
            return Err(GuidanceError::Invalid(
                "regional guidance weight must be finite".to_owned(),
            ));
        }
        if contribution_weight == 0.0 {
            continue;
        }
        let full_index =
            regional_full_index(&local_coordinates, full_shape, entry.region().offsets())?;
        let sum_slot = sum
            .get_mut(full_index)
            .ok_or(GuidanceError::ShapeOverflow("regional sum index"))?;
        *sum_slot = output_value.mul_add(contribution_weight, *sum_slot);
        let weight_slot = weight
            .get_mut(full_index)
            .ok_or(GuidanceError::ShapeOverflow("regional weight index"))?;
        *weight_slot += contribution_weight;
    }
    check_cancel(context)?;
    Ok(())
}

fn decode_row_major_index(
    index: usize,
    shape: &[u64],
    coordinates: &mut [u64],
) -> Result<(), GuidanceError> {
    if shape.len() != coordinates.len() {
        return Err(GuidanceError::ShapeOverflow("regional coordinate rank"));
    }
    let mut remainder =
        u64::try_from(index).map_err(|_| GuidanceError::ShapeOverflow("regional linear index"))?;
    for (coordinate, dimension) in coordinates.iter_mut().zip(shape).rev() {
        if *dimension == 0 {
            return Err(GuidanceError::ShapeOverflow("regional zero dimension"));
        }
        *coordinate = remainder % dimension;
        remainder /= dimension;
    }
    Ok(())
}

fn regional_full_index(
    local_coordinates: &[u64],
    full_shape: &[u64],
    offsets: &[u64],
) -> Result<usize, GuidanceError> {
    if local_coordinates.len() != full_shape.len() || offsets.len() + 2 != full_shape.len() {
        return Err(GuidanceError::ShapeOverflow("regional full index rank"));
    }
    let mut index = 0_u64;
    for (axis, (&coordinate, &dimension)) in local_coordinates.iter().zip(full_shape).enumerate() {
        let coordinate = if axis < 2 {
            coordinate
        } else {
            coordinate
                .checked_add(offsets[axis - 2])
                .ok_or(GuidanceError::ShapeOverflow("regional full coordinate"))?
        };
        if coordinate >= dimension {
            return Err(GuidanceError::ShapeOverflow(
                "regional full coordinate bounds",
            ));
        }
        index = index
            .checked_mul(dimension)
            .and_then(|index| index.checked_add(coordinate))
            .ok_or(GuidanceError::ShapeOverflow("regional full linear index"))?;
    }
    usize::try_from(index).map_err(|_| GuidanceError::ShapeOverflow("regional host index"))
}

fn default_cfg(
    backend: &CpuBackend,
    predictions: &GuidancePredictions,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, GuidanceError> {
    let Some(unconditional) = predictions.unconditional() else {
        return Ok(predictions.conditional().clone());
    };
    let delta = add_method_with_context_exact_native(
        backend,
        predictions.conditional(),
        ElementwiseOperand::Tensor(unconditional),
        -1.0,
        context,
    )?;
    Ok(add_method_with_context_exact_native(
        backend,
        unconditional,
        ElementwiseOperand::Tensor(&delta),
        scale,
        context,
    )?)
}

fn validate_prediction_descriptors(
    latent: &Tensor,
    predictions: &GuidancePredictions,
) -> Result<(), GuidanceError> {
    require_same_descriptor(latent, predictions.conditional(), "conditional prediction")?;
    if let Some(unconditional) = predictions.unconditional() {
        require_same_descriptor(latent, unconditional, "unconditional prediction")?;
    }
    Ok(())
}

fn require_same_descriptor(
    expected: &Tensor,
    actual: &Tensor,
    subject: &str,
) -> Result<(), GuidanceError> {
    if !tensor_contract_matches(expected, actual) {
        return Err(GuidanceError::Invalid(format!(
            "{subject} descriptor does not match the guidance latent"
        )));
    }
    Ok(())
}

fn tensor_contract_matches(left: &Tensor, right: &Tensor) -> bool {
    left.descriptor().shape() == right.descriptor().shape()
        && left.descriptor().dtype() == right.descriptor().dtype()
        && left.descriptor().device() == right.descriptor().device()
        && left.descriptor().stream() == right.descriptor().stream()
}

#[allow(clippy::too_many_arguments)]
fn guidance_identity(
    plan: &SamplingPlan,
    sigma: f32,
    conditional: &ConditioningSet,
    unconditional: &ConditioningSet,
    guided: &Tensor,
    evaluations: usize,
    batches: usize,
    unconditional_skipped: bool,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<String, GuidanceError> {
    let mut hasher = Sha256::new();
    hasher.update(GUIDANCE_ADAPTER_ID.as_bytes());
    hash_string(&mut hasher, plan.profile().as_str())?;
    hasher.update(plan.guidance().to_bits().to_le_bytes());
    hasher.update(sigma.to_bits().to_le_bytes());
    hash_string(&mut hasher, conditional.digest())?;
    hash_string(&mut hasher, unconditional.digest())?;
    hasher.update(
        u64::try_from(evaluations)
            .map_err(|_| GuidanceError::ShapeOverflow("guidance identity evaluations"))?
            .to_le_bytes(),
    );
    hasher.update(
        u64::try_from(batches)
            .map_err(|_| GuidanceError::ShapeOverflow("guidance identity batches"))?
            .to_le_bytes(),
    );
    hasher.update([u8::from(unconditional_skipped)]);
    for value in tensor_to_f32(backend, guided, context)?.iter() {
        hasher.update(value.to_bits().to_le_bytes());
    }
    check_cancel(context)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_string(hasher: &mut Sha256, value: &str) -> Result<(), GuidanceError> {
    hasher.update(
        u64::try_from(value.len())
            .map_err(|_| GuidanceError::ShapeOverflow("guidance identity string"))?
            .to_le_bytes(),
    );
    hasher.update(value.as_bytes());
    Ok(())
}

fn check_cancel(context: &ExecutionContext<'_>) -> Result<(), GuidanceError> {
    context
        .cancellation
        .check()
        .map_err(|_| GuidanceError::Cancelled)
}

fn source_math_isclose_to_one(value: f32) -> bool {
    let value = f64::from(value);
    (value - 1.0).abs() <= PYTHON_MATH_ISCLOSE_RELATIVE_TOLERANCE * value.abs().max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DiscreteSamplingProfile, PredictionInterpretation, SamplingProfileIdentity, SamplingSession,
    };
    use sim_model::conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningMask, ConditioningMetadataMap,
        ConditioningRegion, ConditioningTimestepRange, ConditioningValue, ConditioningWindow,
    };
    use sim_model::{LatentFormatIdentity, ModelFamilyIdentity};
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId, TensorDescriptor};
    use std::{cell::RefCell, error::Error, rc::Rc, sync::Arc};

    type TestResult = Result<(), Box<dyn Error>>;

    struct Harness {
        backend: CpuBackend,
        authority: CpuWorkspaceAuthority,
        cancellation: CancellationToken,
        scratch_bytes: u64,
    }

    impl Harness {
        fn new(memory_bytes: u64, scratch_bytes: u64) -> Result<Self, TensorError> {
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_bytes)?;
            Ok(Self {
                backend,
                authority,
                cancellation: CancellationToken::default(),
                scratch_bytes,
            })
        }

        fn context(&self) -> Result<ExecutionContext<'_>, TensorError> {
            Ok(self.backend.execution_context(
                StreamId::DEFAULT,
                self.authority.authorize_workspace(self.scratch_bytes)?,
                &self.cancellation,
            ))
        }

        fn tensor(
            &self,
            shape: &[u64],
            values: &[f32],
            context: &ExecutionContext<'_>,
        ) -> Result<Tensor, NativeDiffusionTensorError> {
            tensor_from_f32(&self.backend, shape, values, context)
        }
    }

    fn profile() -> Result<DiscreteSamplingProfile, SamplingProfileError> {
        DiscreteSamplingProfile::new(
            SamplingProfileIdentity::new("guidance-test-profile")?,
            PredictionInterpretation::Epsilon,
            Arc::from([1.0_f32, 2.0_f32]),
        )
    }

    fn plan(
        profile: &DiscreteSamplingProfile,
        guidance: f32,
    ) -> Result<SamplingPlan, crate::SamplingError> {
        SamplingPlan::new(
            "euler",
            "normal",
            profile.identity().clone(),
            7,
            1,
            guidance,
            1.0,
        )
    }

    fn identity(namespace: &str) -> Result<ConditioningIdentity, Box<dyn Error>> {
        Ok(ConditioningIdentity::new(
            namespace,
            ModelFamilyIdentity::new("COMFY-MODEL-0001", "guidance_test", "v1")?,
            LatentFormatIdentity::new("COMFY-MODEL-0002", "guidance_latent")?,
        )?)
    }

    fn conditioning(
        harness: &Harness,
        namespace: &str,
        entries: Vec<(&str, ConditioningEntryOptions)>,
        context: &ExecutionContext<'_>,
    ) -> Result<ConditioningSet, Box<dyn Error>> {
        let value = harness.tensor(&[1, 1], &[1.0], context)?;
        let mut checked = Vec::new();
        checked.try_reserve_exact(entries.len())?;
        for (identifier, options) in entries {
            checked.push(ConditioningEntry::checked(
                identifier,
                ConditioningValue::regular(value.clone())?,
                options,
            )?);
        }
        Ok(ConditioningSet::checked(
            identity(namespace)?,
            checked,
            &harness.cancellation,
        )?)
    }

    struct ValueDenoiser<'a> {
        backend: &'a CpuBackend,
        batches: Vec<Vec<GuidanceBranch>>,
        cancel_after_evaluation: bool,
    }

    impl GuidanceDenoiser for ValueDenoiser<'_> {
        fn evaluate_batch(
            &mut self,
            evaluations: &[GuidanceEvaluation],
            context: &ExecutionContext<'_>,
        ) -> Result<Vec<Tensor>, GuidanceError> {
            self.batches
                .push(evaluations.iter().map(GuidanceEvaluation::branch).collect());
            let mut outputs = Vec::new();
            outputs
                .try_reserve_exact(evaluations.len())
                .map_err(|_| GuidanceError::ShapeOverflow("test denoiser outputs"))?;
            for evaluation in evaluations {
                let value = match evaluation.entry().identifier() {
                    "left" => 2.0,
                    "right" => 4.0,
                    "strong" => 6.0,
                    "default" => 8.0,
                    _ => match evaluation.branch() {
                        GuidanceBranch::Conditional => 3.0,
                        GuidanceBranch::Unconditional => 1.0,
                    },
                };
                let count = usize::try_from(evaluation.latent().descriptor().element_count()?)
                    .map_err(|_| GuidanceError::ShapeOverflow("test output"))?;
                let values = vec![value; count];
                outputs.push(tensor_from_f32(
                    self.backend,
                    evaluation.latent().descriptor().shape(),
                    &values,
                    context,
                )?);
            }
            if self.cancel_after_evaluation {
                context.cancellation.cancel();
            }
            Ok(outputs)
        }

        fn evaluate_batch_with_model_hooks(
            &mut self,
            evaluations: &[GuidanceEvaluation],
            _hooks: &dyn NativeFamilyDenoiserHooks,
            context: &ExecutionContext<'_>,
        ) -> Result<Vec<Tensor>, GuidanceError> {
            self.evaluate_batch(evaluations, context)
        }
    }

    fn values(
        harness: &Harness,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, NativeDiffusionTensorError> {
        Ok(tensor_to_f32(&harness.backend, tensor, context)?.to_vec())
    }

    fn full_options() -> ConditioningEntryOptions {
        ConditioningEntryOptions::default()
    }

    #[test]
    fn cfg_scale_zero_one_and_above_one_follow_exact_formula() -> TestResult {
        for (scale, expected, skipped) in [(0.0, 1.0, false), (1.0, 3.0, true), (2.5, 6.0, false)] {
            let harness = Harness::new(1 << 20, 1 << 18)?;
            let context = harness.context()?;
            let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
            let conditional = conditioning(
                &harness,
                "positive",
                vec![("full", full_options())],
                &context,
            )?;
            let unconditional = conditioning(
                &harness,
                "negative",
                vec![("full", full_options())],
                &context,
            )?;
            let profile = profile()?;
            let plan = plan(&profile, scale)?;
            let mut denoiser = ValueDenoiser {
                backend: &harness.backend,
                batches: Vec::new(),
                cancel_after_evaluation: false,
            };
            let result = execute_guidance(
                &harness.backend,
                &latent,
                2.0,
                &profile,
                &plan,
                &conditional,
                &unconditional,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut [],
                &context,
            )?;
            assert_eq!(
                values(&harness, result.guided(), &context)?,
                vec![expected; 2]
            );
            assert_eq!(result.unconditional_skipped(), skipped);
        }
        Ok(())
    }

    #[test]
    fn serializable_execution_hook_graph_is_ordered_attempt_scoped_and_consumed() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("full", full_options())],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 2.0)?;
        let kinds = [
            ExecutionHookKind::Attention,
            ExecutionHookKind::ContextWindow,
            ExecutionHookKind::OutputBlock,
            ExecutionHookKind::SamplingProfile,
            ExecutionHookKind::Rope,
            ExecutionHookKind::ModelFunction,
            ExecutionHookKind::DiffusionWrapper,
            ExecutionHookKind::Hypernetwork,
            ExecutionHookKind::Cfg,
            ExecutionHookKind::PostCfg,
        ];
        let operations = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                let (scale, bias) = match kind {
                    ExecutionHookKind::Attention => (2.0, 1.0),
                    ExecutionHookKind::ModelFunction => (0.5, -1.0),
                    ExecutionHookKind::Cfg => (2.0, 0.0),
                    ExecutionHookKind::PostCfg => (1.0, 3.0),
                    _ => (1.0, 0.0),
                };
                ExecutionHookOperation {
                    identifier: format!("hook-{index}"),
                    order: u32::try_from(index + 1).expect("bounded hook order"),
                    kind,
                    scale,
                    bias,
                    start_percent: 0.0,
                    end_percent: 1.0,
                    rng_address: u64::try_from(index + 100).expect("bounded RNG address"),
                    rng_amplitude: 0.0,
                    hypernetwork: (kind == ExecutionHookKind::Hypernetwork).then(|| {
                        HypernetworkPatch {
                            strength: 1.0,
                            dimensions: BTreeMap::from([(
                                2,
                                HypernetworkDimensionPatch {
                                    key_layers: vec![HypernetworkLinearLayer {
                                        input_dimension: 2,
                                        output_dimension: 2,
                                        weights: vec![0.25, 0.0, 0.0, 0.25],
                                        bias: vec![0.5, 0.5],
                                        activation: HypernetworkActivation::Linear,
                                        layer_norm_scale: None,
                                        layer_norm_bias: None,
                                        dropout_probability: 0.0,
                                    }],
                                    value_layers: vec![HypernetworkLinearLayer {
                                        input_dimension: 2,
                                        output_dimension: 2,
                                        weights: vec![-0.25, 0.0, 0.0, -0.25],
                                        bias: vec![1.0, 1.0],
                                        activation: HypernetworkActivation::Linear,
                                        layer_norm_scale: None,
                                        layer_norm_bias: None,
                                        dropout_probability: 0.0,
                                    }],
                                },
                            )]),
                        }
                    }),
                    context_window: (kind == ExecutionHookKind::ContextWindow).then_some(
                        ContextWindowPatch {
                            context_length: 2,
                            overlap: 1,
                            stride: 1,
                            closed_loop: false,
                            fuse: ContextWindowFuse::Relative,
                            dimension: 2,
                            schedule: ContextWindowSchedule::StaticStandard,
                            schedule_step: 0,
                            freenoise: false,
                            conditioning_retain_indices: Vec::new(),
                            latent_retain_indices: Vec::new(),
                            split_conditioning: false,
                            causal_window_fix: false,
                            slice_conditioning: false,
                            slice_timestep: false,
                        },
                    ),
                    sampling_profile: (kind == ExecutionHookKind::SamplingProfile)
                        .then_some(SamplingProfileReplacement::DiscreteVPrediction),
                }
            })
            .collect();
        let graph = ExecutionHookGraph::checked(41, operations)?;
        let restarted = ExecutionHookGraph::checked(41, graph.operations.clone())?;
        assert_eq!(graph.identity()?, restarted.identity()?);
        assert_ne!(
            graph.identity()?,
            ExecutionHookGraph::checked(42, graph.operations.clone())?.identity()?
        );
        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let result = execute_guidance_with_hook_graph(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &graph,
            &context,
        )?;
        assert_eq!(values(&harness, result.guided(), &context)?, vec![13.0; 2]);

        struct SemanticDenoiser<'a> {
            backend: &'a CpuBackend,
            window_lengths: Vec<u64>,
        }
        impl GuidanceDenoiser for SemanticDenoiser<'_> {
            fn evaluate_batch(
                &mut self,
                evaluations: &[GuidanceEvaluation],
                context: &ExecutionContext<'_>,
            ) -> Result<Vec<Tensor>, GuidanceError> {
                evaluations
                    .iter()
                    .map(|evaluation| {
                        let axis = evaluation.latent().descriptor().shape()[1];
                        self.window_lengths.push(axis);
                        let sampling_offset = evaluation
                            .sampling_profile
                            .map(SamplingProfileReplacement::resolve)
                            .transpose()?
                            .map(|profile| profile.model_time_for_sigma(evaluation.sigma()))
                            .transpose()?
                            .unwrap_or(0.0);
                        let input = tensor_to_f32(self.backend, evaluation.latent(), context)?;
                        let values = input
                            .iter()
                            .enumerate()
                            .map(|(index, value)| {
                                value
                                    + evaluation.sigma()
                                    + sampling_offset
                                    + (index / 2 % axis as usize) as f32
                            })
                            .collect::<Vec<_>>();
                        Ok(tensor_from_f32(
                            self.backend,
                            evaluation.latent().descriptor().shape(),
                            &values,
                            context,
                        )?)
                    })
                    .collect()
            }

            fn evaluate_batch_with_model_hooks(
                &mut self,
                evaluations: &[GuidanceEvaluation],
                hooks: &dyn NativeFamilyDenoiserHooks,
                context: &ExecutionContext<'_>,
            ) -> Result<Vec<Tensor>, GuidanceError> {
                evaluations
                    .iter()
                    .map(|evaluation| {
                        let axis = evaluation.latent().descriptor().shape()[1];
                        self.window_lengths.push(axis);
                        let sampling_offset = evaluation
                            .sampling_profile
                            .map(SamplingProfileReplacement::resolve)
                            .transpose()?
                            .map(|profile| profile.model_time_for_sigma(evaluation.sigma()))
                            .transpose()?
                            .unwrap_or(0.0);
                        let input = tensor_to_f32(self.backend, evaluation.latent(), context)?;
                        let mut query = input.to_vec();
                        let mut key = input.to_vec();
                        let mut value = input.to_vec();
                        hooks.after_rotary(&mut query, &mut key, 2, context)?;
                        hooks.before_attention(&mut query, &mut key, &mut value, 2, context)?;
                        let mut output = query
                            .iter()
                            .zip(&key)
                            .zip(&value)
                            .enumerate()
                            .map(|(index, ((query, key), value))| {
                                (query + key + value) / 3.0
                                    + evaluation.sigma()
                                    + sampling_offset
                                    + (index / 2 % axis as usize) as f32
                            })
                            .collect::<Vec<_>>();
                        hooks.after_attention(&mut output, 2, context)?;
                        Ok(tensor_from_f32(
                            self.backend,
                            evaluation.latent().descriptor().shape(),
                            &output,
                            context,
                        )?)
                    })
                    .collect()
            }
        }
        let semantic_latent = harness.tensor(
            &[1, 5, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &context,
        )?;
        let empty_graph = ExecutionHookGraph::checked(41, Vec::new())?;
        let mut baseline_denoiser = SemanticDenoiser {
            backend: &harness.backend,
            window_lengths: Vec::new(),
        };
        let baseline = execute_guidance_with_hook_graph(
            &harness.backend,
            &semantic_latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut baseline_denoiser,
            &empty_graph,
            &context,
        )?;
        let baseline_values = values(&harness, baseline.guided(), &context)?;
        for (index, kind) in kinds.into_iter().enumerate() {
            let hypernetwork = (kind == ExecutionHookKind::Hypernetwork)
                .then(|| {
                    graph
                        .operations
                        .iter()
                        .find_map(|operation| operation.hypernetwork.clone())
                        .ok_or_else(|| {
                            GuidanceError::Invalid("hypernetwork fixture is missing".to_owned())
                        })
                })
                .transpose()?;
            let context_window =
                (kind == ExecutionHookKind::ContextWindow).then_some(ContextWindowPatch {
                    context_length: 3,
                    overlap: 2,
                    stride: 1,
                    closed_loop: false,
                    fuse: ContextWindowFuse::Pyramid,
                    dimension: 1,
                    schedule: ContextWindowSchedule::StaticStandard,
                    schedule_step: 0,
                    freenoise: false,
                    conditioning_retain_indices: Vec::new(),
                    latent_retain_indices: Vec::new(),
                    split_conditioning: false,
                    causal_window_fix: false,
                    slice_conditioning: false,
                    slice_timestep: false,
                });
            let (scale, bias) = if matches!(
                kind,
                ExecutionHookKind::ContextWindow | ExecutionHookKind::SamplingProfile
            ) {
                (1.0, 0.0)
            } else {
                (2.0, 0.25)
            };
            let category_graph = ExecutionHookGraph::checked(
                41,
                vec![ExecutionHookOperation {
                    identifier: format!("category-{index}"),
                    order: 1,
                    kind,
                    scale,
                    bias,
                    start_percent: 0.0,
                    end_percent: 1.0,
                    rng_address: u64::try_from(index + 500)?,
                    rng_amplitude: 0.0,
                    hypernetwork,
                    context_window,
                    sampling_profile: (kind == ExecutionHookKind::SamplingProfile)
                        .then_some(SamplingProfileReplacement::DiscreteVPrediction),
                }],
            )?;
            let mut category_denoiser = SemanticDenoiser {
                backend: &harness.backend,
                window_lengths: Vec::new(),
            };
            let category = execute_guidance_with_hook_graph(
                &harness.backend,
                &semantic_latent,
                2.0,
                &profile,
                &plan,
                &conditional,
                &unconditional,
                GuidanceOptions::default(),
                &mut category_denoiser,
                &category_graph,
                &context,
            )?;
            let category_values = values(&harness, category.guided(), &context)?;
            assert_ne!(
                category_values, baseline_values,
                "{kind:?} hook was not consumed at its execution phase"
            );
            if kind == ExecutionHookKind::ContextWindow {
                assert!(category_denoiser.window_lengths.len() >= 6);
                assert!(
                    category_denoiser
                        .window_lengths
                        .iter()
                        .all(|length| *length == 3)
                );
                let expected = [
                    3.0,
                    4.0,
                    5.0 + 2.0 / 3.0,
                    6.0 + 2.0 / 3.0,
                    8.0,
                    9.0,
                    10.0 + 1.0 / 3.0,
                    11.0 + 1.0 / 3.0,
                    13.0,
                    14.0,
                ];
                assert!(
                    category_values
                        .iter()
                        .zip(expected)
                        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-5)
                );

                let mut relative_operations = category_graph.operations.clone();
                relative_operations[0]
                    .context_window
                    .as_mut()
                    .ok_or_else(|| GuidanceError::Invalid("window fixture is missing".to_owned()))?
                    .fuse = ContextWindowFuse::Relative;
                let relative_graph = ExecutionHookGraph::checked(41, relative_operations)?;
                let mut relative_denoiser = SemanticDenoiser {
                    backend: &harness.backend,
                    window_lengths: Vec::new(),
                };
                let relative = execute_guidance_with_hook_graph(
                    &harness.backend,
                    &semantic_latent,
                    2.0,
                    &profile,
                    &plan,
                    &conditional,
                    &unconditional,
                    GuidanceOptions::default(),
                    &mut relative_denoiser,
                    &relative_graph,
                    &context,
                )?;
                assert_ne!(
                    category_values,
                    values(&harness, relative.guided(), &context)?,
                    "context-window fuse methods must retain distinct source weighting"
                );
                let mut flat_operations = category_graph.operations.clone();
                flat_operations[0]
                    .context_window
                    .as_mut()
                    .ok_or_else(|| GuidanceError::Invalid("window fixture is missing".to_owned()))?
                    .fuse = ContextWindowFuse::Flat;
                let flat_graph = ExecutionHookGraph::checked(41, flat_operations)?;
                let mut flat_denoiser = SemanticDenoiser {
                    backend: &harness.backend,
                    window_lengths: Vec::new(),
                };
                let flat = execute_guidance_with_hook_graph(
                    &harness.backend,
                    &semantic_latent,
                    2.0,
                    &profile,
                    &plan,
                    &conditional,
                    &unconditional,
                    GuidanceOptions::default(),
                    &mut flat_denoiser,
                    &flat_graph,
                    &context,
                )?;
                assert_ne!(values(&harness, flat.guided(), &context)?, category_values);

                let mut overlap_operations = category_graph.operations.clone();
                overlap_operations[0]
                    .context_window
                    .as_mut()
                    .ok_or_else(|| GuidanceError::Invalid("window fixture is missing".to_owned()))?
                    .fuse = ContextWindowFuse::OverlapLinear;
                let overlap_graph = ExecutionHookGraph::checked(41, overlap_operations)?;
                let mut overlap_denoiser = SemanticDenoiser {
                    backend: &harness.backend,
                    window_lengths: Vec::new(),
                };
                let overlap = execute_guidance_with_hook_graph(
                    &harness.backend,
                    &semantic_latent,
                    2.0,
                    &profile,
                    &plan,
                    &conditional,
                    &unconditional,
                    GuidanceOptions::default(),
                    &mut overlap_denoiser,
                    &overlap_graph,
                    &context,
                )?;
                assert_ne!(
                    values(&harness, overlap.guided(), &context)?,
                    category_values
                );

                let mut batched_operations = category_graph.operations.clone();
                let batched_window =
                    batched_operations[0]
                        .context_window
                        .as_mut()
                        .ok_or_else(|| {
                            GuidanceError::Invalid("window fixture is missing".to_owned())
                        })?;
                batched_window.schedule = ContextWindowSchedule::Batched;
                batched_window.context_length = 2;
                batched_window.overlap = 0;
                batched_window.fuse = ContextWindowFuse::Flat;
                let batched_graph = ExecutionHookGraph::checked(41, batched_operations)?;
                let mut batched_denoiser = SemanticDenoiser {
                    backend: &harness.backend,
                    window_lengths: Vec::new(),
                };
                let batched = execute_guidance_with_hook_graph(
                    &harness.backend,
                    &semantic_latent,
                    2.0,
                    &profile,
                    &plan,
                    &conditional,
                    &unconditional,
                    GuidanceOptions::default(),
                    &mut batched_denoiser,
                    &batched_graph,
                    &context,
                )?;
                assert_eq!(batched.guided().descriptor(), semantic_latent.descriptor());
                assert!(batched_denoiser.window_lengths.contains(&1));
                assert!(batched_denoiser.window_lengths.contains(&2));
            }
        }
        let hypernetwork = graph
            .operations
            .iter()
            .find_map(|operation| operation.hypernetwork.as_ref())
            .ok_or_else(|| GuidanceError::Invalid("hypernetwork fixture is missing".to_owned()))?;
        let query = harness.tensor(&[1, 1, 2], &[1.0, 2.0], &context)?;
        let key = harness.tensor(&[1, 1, 2], &[2.0, 4.0], &context)?;
        let value = harness.tensor(&[1, 1, 2], &[3.0, 5.0], &context)?;
        let (patched_query, patched_key, patched_value) = apply_hypernetwork_attention(
            &harness.backend,
            &query,
            &key,
            &value,
            hypernetwork,
            41,
            900,
            &context,
        )?;
        assert_eq!(values(&harness, &patched_query, &context)?, vec![1.0, 2.0]);
        assert_eq!(values(&harness, &patched_key, &context)?, vec![3.0, 5.5]);
        assert_eq!(
            values(&harness, &patched_value, &context)?,
            vec![3.25, 4.75]
        );
        let mut dropout_hypernetwork = hypernetwork.clone();
        let dropout_dimension = dropout_hypernetwork
            .dimensions
            .get_mut(&2)
            .ok_or_else(|| GuidanceError::Invalid("dropout fixture is missing".to_owned()))?;
        dropout_dimension.value_layers = dropout_dimension.key_layers.clone();
        for layer in dropout_dimension
            .key_layers
            .iter_mut()
            .chain(&mut dropout_dimension.value_layers)
        {
            layer.dropout_probability = 0.5;
        }
        let repeated_values = (0..64).map(|index| index as f32 / 16.0).collect::<Vec<_>>();
        let repeated = harness.tensor(&[32, 2], &repeated_values, &context)?;
        let (_, dropout_key, dropout_value) = apply_hypernetwork_attention(
            &harness.backend,
            &repeated,
            &repeated,
            &repeated,
            &dropout_hypernetwork,
            41,
            901,
            &context,
        )?;
        assert_ne!(
            values(&harness, &dropout_key, &context)?,
            values(&harness, &dropout_value, &context)?,
            "hypernetwork dropout RNG must distinguish sequential key and value branches"
        );
        assert!(
            graph.resident_bytes()? > u64::try_from(std::mem::size_of::<ExecutionHookGraph>())?
        );

        let mut randomized_operations = graph.operations.clone();
        randomized_operations
            .iter_mut()
            .find(|operation| operation.kind == ExecutionHookKind::DiffusionWrapper)
            .ok_or_else(|| GuidanceError::Invalid("hook fixture is empty".to_owned()))?
            .rng_amplitude = 1.0;
        let randomized = ExecutionHookGraph::checked(41, randomized_operations.clone())?;
        let changed_attempt = ExecutionHookGraph::checked(42, randomized_operations)?;
        let mut first_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let first = execute_guidance_with_hook_graph(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut first_denoiser,
            &randomized,
            &context,
        )?;
        let first_values = values(&harness, first.guided(), &context)?;
        let mut replay_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let replay = execute_guidance_with_hook_graph(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut replay_denoiser,
            &randomized,
            &context,
        )?;
        assert_eq!(first_values, values(&harness, replay.guided(), &context)?);
        let mut changed_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let changed = execute_guidance_with_hook_graph(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut changed_denoiser,
            &changed_attempt,
            &context,
        )?;
        assert_ne!(first_values, values(&harness, changed.guided(), &context)?);

        let mut invalid = graph.operations.clone();
        invalid.swap(0, 1);
        assert!(
            ExecutionHookGraph {
                schema_version: ExecutionHookGraph::SCHEMA_VERSION,
                attempt_seed: 41,
                operations: invalid,
            }
            .validate()
            .is_err()
        );
        let oversized = (0_u32..257)
            .map(|order| ExecutionHookOperation {
                identifier: format!("oversized-{order}"),
                order,
                kind: ExecutionHookKind::Attention,
                scale: 1.0,
                bias: 0.0,
                start_percent: 0.0,
                end_percent: 1.0,
                rng_address: u64::from(order),
                rng_amplitude: 0.0,
                hypernetwork: None,
                context_window: None,
                sampling_profile: None,
            })
            .collect();
        assert!(ExecutionHookGraph::checked(41, oversized).is_err());

        let mut unsupported_context = graph
            .operations
            .iter()
            .find(|operation| operation.kind == ExecutionHookKind::ContextWindow)
            .cloned()
            .ok_or_else(|| GuidanceError::Invalid("context fixture is missing".to_owned()))?;
        unsupported_context.order = 1;
        unsupported_context
            .context_window
            .as_mut()
            .ok_or_else(|| GuidanceError::Invalid("context fixture is missing".to_owned()))?
            .freenoise = true;
        assert!(matches!(
            ExecutionHookGraph::checked(41, vec![unsupported_context]),
            Err(GuidanceError::UnsupportedContextWindowFeature("freenoise"))
        ));
        let base_context = graph
            .operations
            .iter()
            .find_map(|operation| operation.context_window.clone())
            .ok_or_else(|| GuidanceError::Invalid("context fixture is missing".to_owned()))?;
        let mut unsupported_contexts = Vec::new();
        let mut context = base_context.clone();
        context.schedule = ContextWindowSchedule::UniformStandard;
        unsupported_contexts.push((context, "uniform_standard"));
        let mut context = base_context.clone();
        context.schedule = ContextWindowSchedule::UniformLooped;
        unsupported_contexts.push((context, "uniform_looped"));
        let mut context = base_context.clone();
        context.conditioning_retain_indices.push(0);
        unsupported_contexts.push((context, "conditioning_retain_indices"));
        let mut context = base_context.clone();
        context.latent_retain_indices.push(0);
        unsupported_contexts.push((context, "latent_retain_indices"));
        let mut context = base_context.clone();
        context.split_conditioning = true;
        unsupported_contexts.push((context, "split_conditioning"));
        let mut context = base_context.clone();
        context.causal_window_fix = true;
        unsupported_contexts.push((context, "causal_window_fix"));
        let mut context = base_context.clone();
        context.slice_conditioning = true;
        unsupported_contexts.push((context, "temporal_conditioning_slice"));
        let mut context = base_context;
        context.slice_timestep = true;
        unsupported_contexts.push((context, "timestep_slice"));
        for (context, feature) in unsupported_contexts {
            assert!(matches!(
                context.validate(),
                Err(GuidanceError::UnsupportedContextWindowFeature(actual)) if actual == feature
            ));
        }

        let mut sampling = graph
            .operations
            .iter()
            .find(|operation| operation.kind == ExecutionHookKind::SamplingProfile)
            .cloned()
            .ok_or_else(|| GuidanceError::Invalid("sampling fixture is missing".to_owned()))?;
        sampling.order = 1;
        sampling.sampling_profile = Some(SamplingProfileReplacement::DiscreteLcm);
        let lcm_graph = ExecutionHookGraph::checked(41, vec![sampling.clone()])?;
        assert_eq!(
            lcm_graph
                .operations
                .first()
                .and_then(|operation| operation.sampling_profile),
            Some(SamplingProfileReplacement::DiscreteLcm),
        );
        sampling.sampling_profile = Some(SamplingProfileReplacement::StableCascade { shift: 0.0 });
        assert!(ExecutionHookGraph::checked(41, vec![sampling]).is_err());
        Ok(())
    }

    #[test]
    fn task407_serialized_cfg_hooks_precede_attached_modifier_overrides() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("full", full_options())],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 2.0)?;
        let operation =
            |identifier: &str, order, kind, scale, bias, rng_address| ExecutionHookOperation {
                identifier: identifier.to_owned(),
                order,
                kind,
                scale,
                bias,
                start_percent: 0.0,
                end_percent: 1.0,
                rng_address,
                rng_amplitude: 0.0,
                hypernetwork: None,
                context_window: None,
                sampling_profile: None,
            };
        let graph = ExecutionHookGraph::checked(
            407,
            vec![
                operation("serialized-cfg", 1, ExecutionHookKind::Cfg, 2.0, 0.0, 1),
                operation(
                    "serialized-post-cfg",
                    2,
                    ExecutionHookKind::PostCfg,
                    1.0,
                    3.0,
                    2,
                ),
            ],
        )?;
        let modifiers =
            NativeGuidanceModifierChain::checked(vec![NativeGuidanceModifier::CfgOverride {
                scale: 3.0,
            }])?;
        let mut modifier_session = modifiers.begin_attempt(&harness.backend, 407)?;
        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let result = execute_guidance_with_modifier_session(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &graph,
            &mut modifier_session,
            &context,
        )?;
        assert_eq!(values(&harness, result.guided(), &context)?, vec![10.0; 2]);
        Ok(())
    }

    #[test]
    fn typed_sampling_profile_replacements_follow_source_equations_and_bounds() -> TestResult {
        let sigma = 2.0_f32;
        let input = [3.0_f32];
        for (replacement, expected) in [
            (
                SamplingProfileReplacement::DiscreteEpsilon,
                3.0 - 0.5 * sigma,
            ),
            (
                SamplingProfileReplacement::DiscreteVPrediction,
                3.0 / (sigma * sigma + 1.0) - 0.5 * sigma / (sigma * sigma + 1.0).sqrt(),
            ),
            (SamplingProfileReplacement::DiscreteX0, 0.5),
        ] {
            let profile = replacement.resolve()?;
            let mut scaled = input;
            profile.scale_model_input_in_place(&mut scaled, sigma)?;
            let expected_input = input[0] / (sigma * sigma + 1.0).sqrt();
            assert!((scaled[0] - expected_input).abs() < 1.0e-6);
            let mut output = [0.5];
            profile.interpret_prediction_in_place(&mut output, &input, sigma)?;
            assert!((output[0] - expected).abs() < 1.0e-6);
        }

        let aura_shift = 2.5_f32;
        let aura = SamplingProfileReplacement::AuraFlow { shift: aura_shift }.resolve()?;
        let aura_sigma = aura.sigma_for_model_time(0.25)?;
        assert!(
            (aura_sigma - aura_shift * 0.25 / (1.0 + (aura_shift - 1.0) * 0.25)).abs() < 1.0e-6
        );
        assert!((aura.model_time_for_sigma(aura_sigma)? - aura_sigma).abs() < 1.0e-6);
        let mut flow_input = input;
        aura.scale_model_input_in_place(&mut flow_input, aura_sigma)?;
        assert_eq!(flow_input, input);
        let mut flow_output = [0.5];
        aura.interpret_prediction_in_place(&mut flow_output, &input, aura_sigma)?;
        assert!((flow_output[0] - (input[0] - 0.5 * aura_sigma)).abs() < 1.0e-6);

        let flux_shift = 1.25_f32;
        let flux = SamplingProfileReplacement::Flux { shift: flux_shift }.resolve()?;
        let flux_sigma = flux.sigma_for_model_time(0.25)?;
        let exponential = flux_shift.exp();
        assert!(
            (flux_sigma - exponential * 0.25 / (exponential.mul_add(0.25, 0.75))).abs() < 1.0e-6
        );

        for replacement in [
            SamplingProfileReplacement::DiscreteLcm,
            SamplingProfileReplacement::DiscreteImg2Img,
            SamplingProfileReplacement::DiscreteImg2ImgFlow,
            SamplingProfileReplacement::StableCascade { shift: 1.0 },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::VPrediction,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::Edm,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::EdmPlaygroundV25,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::Epsilon,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::CosmosRFlow,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::ContinuousV,
                sigma_min: 0.002,
                sigma_max: 120.0,
                sigma_data: 1.0,
            },
        ] {
            let profile = replacement.resolve()?;
            let restored: NativeSamplingPolicy =
                serde_json::from_slice(&serde_json::to_vec(&profile)?)?;
            assert_eq!(restored.identity(), profile.identity());
        }
        for replacement in [
            SamplingProfileReplacement::StableCascade { shift: 0.0 },
            SamplingProfileReplacement::Sd3 {
                shift: 1.0,
                multiplier: 0.0,
            },
            SamplingProfileReplacement::ContinuousEdm {
                sampling: ContinuousSamplingReplacement::Edm,
                sigma_min: 1.0,
                sigma_max: 0.5,
                sigma_data: 1.0,
            },
        ] {
            assert!(replacement.resolve().is_err());
        }
        Ok(())
    }

    #[test]
    fn source_exact_skip_can_be_disabled_and_batching_respects_memory() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("full", full_options())],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let adjacent_to_one = f32::from_bits(1.0_f32.to_bits() + 1);
        let adjacent_plan = plan(&profile, adjacent_to_one)?;
        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let adjacent_result = execute_guidance(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &adjacent_plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut [],
            &context,
        )?;
        assert!(!adjacent_result.unconditional_skipped());

        let plan = plan(&profile, 1.0)?;
        denoiser.batches.clear();
        let options = GuidanceOptions {
            disable_unconditional_skip: true,
            maximum_batch_bytes: 16,
        };
        let result = execute_guidance(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            options,
            &mut denoiser,
            &mut [],
            &context,
        )?;
        assert!(!result.unconditional_skipped());
        assert_eq!(result.denoiser_batches(), 2);
        assert_eq!(
            denoiser.batches,
            vec![
                vec![GuidanceBranch::Conditional],
                vec![GuidanceBranch::Unconditional]
            ]
        );
        Ok(())
    }

    #[test]
    fn compatible_evaluations_share_one_batch() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("full", full_options())],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 2.0)?;
        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let result = execute_guidance(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut [],
            &context,
        )?;
        assert_eq!(result.denoiser_batches(), 1);
        assert_eq!(denoiser.batches[0].len(), 2);
        Ok(())
    }

    #[test]
    fn regional_accumulation_normalizes_overlap_and_uses_default_fill() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 4], &[0.0; 4], &context)?;
        let region = |sizes, offsets, strength, default_region| -> Result<_, ConditioningError> {
            Ok(ConditioningEntryOptions {
                strength,
                region: Some(ConditioningRegion::absolute(sizes, offsets)?),
                mask: None,
                window: ConditioningWindow::full(),
                default_region,
            })
        };
        let overlap = conditioning(
            &harness,
            "overlap",
            vec![
                ("left", region(vec![4], vec![0], 1.0, false)?),
                ("strong", region(vec![4], vec![0], 3.0, false)?),
            ],
            &context,
        )?;
        let fallback = conditioning(
            &harness,
            "fallback",
            vec![
                ("left", region(vec![2], vec![0], 1.0, false)?),
                (
                    "default",
                    ConditioningEntryOptions {
                        default_region: true,
                        ..full_options()
                    },
                ),
            ],
            &context,
        )?;
        let mask = harness.tensor(&[4], &[1.0, 0.0, 1.0, 0.0], &context)?;
        let masked = conditioning(
            &harness,
            "masked",
            vec![
                (
                    "left",
                    ConditioningEntryOptions {
                        mask: Some(ConditioningMask::new(mask, 1.0, vec![0], false)?),
                        ..full_options()
                    },
                ),
                (
                    "default",
                    ConditioningEntryOptions {
                        default_region: true,
                        ..full_options()
                    },
                ),
            ],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 1.0)?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        for (conditional, expected) in [
            (&overlap, vec![5.0; 4]),
            (&fallback, vec![2.0, 2.0, 8.0, 8.0]),
            (&masked, vec![2.0, 8.0, 2.0, 8.0]),
        ] {
            let mut denoiser = ValueDenoiser {
                backend: &harness.backend,
                batches: Vec::new(),
                cancel_after_evaluation: false,
            };
            let result = execute_guidance(
                &harness.backend,
                &latent,
                2.0,
                &profile,
                &plan,
                conditional,
                &unconditional,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut [],
                &context,
            )?;
            assert_eq!(values(&harness, result.guided(), &context)?, expected);
        }
        Ok(())
    }

    struct OrderedHook {
        name: &'static str,
        events: Rc<RefCell<Vec<String>>>,
        override_with_conditional: bool,
    }

    #[test]
    fn task403_metadata_timestep_range_changes_real_guidance_execution() -> TestResult {
        let harness = Harness::new(1_048_576, 1_048_576)?;
        let context = harness.context()?;
        let profile = profile()?;
        let sampling_percent = sampling_percent_for_sigma(&profile, 2.0)?;
        let latent = harness.tensor(&[1, 1, 2, 2], &[0.0; 4], &context)?;
        let base = conditioning(
            &harness,
            "positive",
            vec![("conditional", ConditioningEntryOptions::default())],
            &context,
        )?;
        let metadata = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::TimestepRange,
            ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(
                sampling_percent,
                sampling_percent,
            )?),
        )])?;
        let active = base.with_entry_metadata(vec![metadata], &harness.cancellation)?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("unconditional", ConditioningEntryOptions::default())],
            &context,
        )?;
        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let result = execute_guidance(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan(&profile, 1.0)?,
            &active,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut [],
            &context,
        )?;
        assert_eq!(values(&harness, result.guided(), &context)?, vec![3.0; 4]);

        let excluded_percent = if sampling_percent < 0.5 { 1.0 } else { 0.0 };
        let excluded = base.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::TimestepRange,
                ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(
                    excluded_percent,
                    excluded_percent,
                )?),
            )])?],
            &harness.cancellation,
        )?;
        assert!(matches!(
            execute_guidance(
                &harness.backend,
                &latent,
                2.0,
                &profile,
                &plan(&profile, 1.0)?,
                &excluded,
                &unconditional,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut [],
                &context,
            ),
            Err(GuidanceError::NoActiveConditioning(_))
        ));
        Ok(())
    }

    impl GuidanceHook for OrderedHook {
        fn pre_cfg(
            &mut self,
            _hook: &GuidanceHookContext<'_>,
            _predictions: &mut GuidancePredictions,
            _context: &ExecutionContext<'_>,
        ) -> Result<(), GuidanceError> {
            self.events.borrow_mut().push(format!("pre:{}", self.name));
            Ok(())
        }

        fn override_cfg(
            &mut self,
            _hook: &GuidanceHookContext<'_>,
            predictions: &GuidancePredictions,
            _current: &Tensor,
            _context: &ExecutionContext<'_>,
        ) -> Result<Option<Tensor>, GuidanceError> {
            self.events
                .borrow_mut()
                .push(format!("override:{}", self.name));
            Ok(self
                .override_with_conditional
                .then(|| predictions.conditional().clone()))
        }

        fn post_cfg(
            &mut self,
            _hook: &GuidanceHookContext<'_>,
            _predictions: &GuidancePredictions,
            current: &Tensor,
            _context: &ExecutionContext<'_>,
        ) -> Result<Option<Tensor>, GuidanceError> {
            self.events.borrow_mut().push(format!("post:{}", self.name));
            Ok(Some(current.clone()))
        }
    }

    #[test]
    fn hooks_run_pre_override_post_in_registration_order() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 1], &[0.0], &context)?;
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("full", full_options())],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 4.0)?;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut first = OrderedHook {
            name: "a",
            events: events.clone(),
            override_with_conditional: true,
        };
        let mut second = OrderedHook {
            name: "b",
            events: events.clone(),
            override_with_conditional: false,
        };
        let mut hooks: [&mut dyn GuidanceHook; 2] = [&mut first, &mut second];
        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let result = execute_guidance(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &conditional,
            &unconditional,
            GuidanceOptions::default(),
            &mut denoiser,
            &mut hooks,
            &context,
        )?;
        assert_eq!(values(&harness, result.guided(), &context)?, vec![3.0]);
        assert_eq!(
            &*events.borrow(),
            &[
                "pre:a",
                "pre:b",
                "override:a",
                "override:b",
                "post:a",
                "post:b"
            ]
        );
        Ok(())
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum CancelPhase {
        Pre,
        Override,
        Post,
    }

    struct CancellingHook {
        phase: CancelPhase,
        events: Rc<RefCell<Vec<&'static str>>>,
    }

    impl GuidanceHook for CancellingHook {
        fn pre_cfg(
            &mut self,
            _hook: &GuidanceHookContext<'_>,
            _predictions: &mut GuidancePredictions,
            context: &ExecutionContext<'_>,
        ) -> Result<(), GuidanceError> {
            self.events.borrow_mut().push("pre");
            if self.phase == CancelPhase::Pre {
                context.cancellation.cancel();
            }
            Ok(())
        }

        fn override_cfg(
            &mut self,
            _hook: &GuidanceHookContext<'_>,
            _predictions: &GuidancePredictions,
            _current: &Tensor,
            context: &ExecutionContext<'_>,
        ) -> Result<Option<Tensor>, GuidanceError> {
            self.events.borrow_mut().push("override");
            if self.phase == CancelPhase::Override {
                context.cancellation.cancel();
            }
            Ok(None)
        }

        fn post_cfg(
            &mut self,
            _hook: &GuidanceHookContext<'_>,
            _predictions: &GuidancePredictions,
            _current: &Tensor,
            context: &ExecutionContext<'_>,
        ) -> Result<Option<Tensor>, GuidanceError> {
            self.events.borrow_mut().push("post");
            if self.phase == CancelPhase::Post {
                context.cancellation.cancel();
            }
            Ok(None)
        }
    }

    #[test]
    fn cancellation_after_each_hook_phase_stops_before_the_next_phase() -> TestResult {
        for (phase, expected_events) in [
            (CancelPhase::Pre, vec!["pre"]),
            (CancelPhase::Override, vec!["pre", "override"]),
            (CancelPhase::Post, vec!["pre", "override", "post"]),
        ] {
            let harness = Harness::new(1 << 20, 1 << 18)?;
            let context = harness.context()?;
            let latent = harness.tensor(&[1, 1, 1], &[0.0], &context)?;
            let conditional = conditioning(
                &harness,
                "positive",
                vec![("full", full_options())],
                &context,
            )?;
            let unconditional = conditioning(
                &harness,
                "negative",
                vec![("full", full_options())],
                &context,
            )?;
            let profile = profile()?;
            let plan = plan(&profile, 2.0)?;
            let events = Rc::new(RefCell::new(Vec::new()));
            let mut hook = CancellingHook {
                phase,
                events: events.clone(),
            };
            let mut hooks: [&mut dyn GuidanceHook; 1] = [&mut hook];
            let mut denoiser = ValueDenoiser {
                backend: &harness.backend,
                batches: Vec::new(),
                cancel_after_evaluation: false,
            };
            assert!(matches!(
                execute_guidance(
                    &harness.backend,
                    &latent,
                    2.0,
                    &profile,
                    &plan,
                    &conditional,
                    &unconditional,
                    GuidanceOptions::default(),
                    &mut denoiser,
                    &mut hooks,
                    &context,
                ),
                Err(GuidanceError::Cancelled)
            ));
            assert_eq!(&*events.borrow(), &expected_events);
            assert_eq!(context.scratch.in_use_bytes(), 0);
        }
        Ok(())
    }

    #[test]
    fn percent_windows_and_result_identity_are_deterministic() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 1], &[0.0], &context)?;
        let early = ConditioningEntryOptions {
            window: ConditioningWindow::new(0.0, 0.4)?,
            ..full_options()
        };
        let late = ConditioningEntryOptions {
            window: ConditioningWindow::new(0.6, 1.0)?,
            ..full_options()
        };
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("left", early), ("right", late)],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 1.0)?;
        assert_eq!(sampling_percent_for_sigma(&profile, 2.0)?, 0.0);
        assert_eq!(sampling_percent_for_sigma(&profile, 1.0)?, 1.0);
        let mut identities = Vec::new();
        for _ in 0..2 {
            let mut denoiser = ValueDenoiser {
                backend: &harness.backend,
                batches: Vec::new(),
                cancel_after_evaluation: false,
            };
            let result = execute_guidance(
                &harness.backend,
                &latent,
                2.0,
                &profile,
                &plan,
                &conditional,
                &unconditional,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut [],
                &context,
            )?;
            assert_eq!(result.denoiser_evaluations(), 1);
            identities.push(result.identity().to_owned());
        }
        assert_eq!(identities[0], identities[1]);
        Ok(())
    }

    #[test]
    fn cancellation_batch_limit_dtype_and_workspace_fail_without_sampler_commit() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let conditional = conditioning(
            &harness,
            "positive",
            vec![("full", full_options())],
            &context,
        )?;
        let unconditional = conditioning(
            &harness,
            "negative",
            vec![("full", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 2.0)?;
        let session = SamplingSession::new(plan.clone(), vec![2.0, 0.0], latent.clone())?;

        let mut denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let limited = GuidanceOptions {
            maximum_batch_bytes: 1,
            ..GuidanceOptions::default()
        };
        assert!(matches!(
            execute_guidance(
                &harness.backend,
                &latent,
                2.0,
                &profile,
                &plan,
                &conditional,
                &unconditional,
                limited,
                &mut denoiser,
                &mut [],
                &context,
            ),
            Err(GuidanceError::BatchMemoryLimit { .. })
        ));
        assert_eq!(session.next_step(), 0);

        let f16_descriptor = TensorDescriptor::contiguous(
            vec![1, 1, 2],
            DType::F16,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (f16_latent, _) = harness
            .backend
            .upload_bytes(f16_descriptor, &[0; 4], &context)?;
        assert!(matches!(
            execute_guidance(
                &harness.backend,
                &f16_latent,
                2.0,
                &profile,
                &plan,
                &conditional,
                &unconditional,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut [],
                &context,
            ),
            Err(GuidanceError::Invalid(_))
        ));

        let cancelled_harness = Harness::new(1 << 20, 1 << 18)?;
        let cancelled_context = cancelled_harness.context()?;
        let cancelled_latent =
            cancelled_harness.tensor(&[1, 1, 2], &[0.0, 0.0], &cancelled_context)?;
        let cancelled_conditional = conditioning(
            &cancelled_harness,
            "positive",
            vec![("full", full_options())],
            &cancelled_context,
        )?;
        let cancelled_unconditional = conditioning(
            &cancelled_harness,
            "negative",
            vec![("full", full_options())],
            &cancelled_context,
        )?;
        let mut cancelling = ValueDenoiser {
            backend: &cancelled_harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: true,
        };
        assert!(matches!(
            execute_guidance(
                &cancelled_harness.backend,
                &cancelled_latent,
                2.0,
                &profile,
                &plan,
                &cancelled_conditional,
                &cancelled_unconditional,
                GuidanceOptions::default(),
                &mut cancelling,
                &mut [],
                &cancelled_context,
            ),
            Err(GuidanceError::Cancelled)
        ));
        assert_eq!(session.next_step(), 0);

        let low_scratch = Harness::new(1 << 20, 4)?;
        let build_context = low_scratch.backend.execution_context(
            StreamId::DEFAULT,
            low_scratch.authority.authorize_workspace(1 << 18)?,
            &low_scratch.cancellation,
        );
        let low_latent = low_scratch.tensor(&[1, 1, 2], &[0.0, 0.0], &build_context)?;
        let low_conditional = conditioning(
            &low_scratch,
            "positive",
            vec![("full", full_options())],
            &build_context,
        )?;
        let low_unconditional = conditioning(
            &low_scratch,
            "negative",
            vec![("full", full_options())],
            &build_context,
        )?;
        drop(build_context);
        let low_context = low_scratch.context()?;
        let mut low_denoiser = ValueDenoiser {
            backend: &low_scratch.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        assert!(
            execute_guidance(
                &low_scratch.backend,
                &low_latent,
                2.0,
                &profile,
                &plan,
                &low_conditional,
                &low_unconditional,
                GuidanceOptions::default(),
                &mut low_denoiser,
                &mut [],
                &low_context,
            )
            .is_err()
        );
        assert_eq!(session.next_step(), 0);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn family_model_guidance_adapter_delegates_without_owning_cfg() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let profile = DiscreteSamplingProfile::auraflow()?;
        let latent = harness.tensor(&[2], &[2.0, 3.0], &context)?;
        let conditional_raw = harness.tensor(&[2], &[0.5, -1.0], &context)?;
        let unconditional_raw = harness.tensor(&[2], &[-0.25, 0.25], &context)?;
        let conditional = interpret_prediction_for_profile(
            &profile,
            &harness.backend,
            &conditional_raw,
            &latent,
            0.25,
            &context,
        )?;
        let unconditional = interpret_prediction_for_profile(
            &profile,
            &harness.backend,
            &unconditional_raw,
            &latent,
            0.25,
            &context,
        )?;
        let predictions = GuidancePredictions {
            conditional,
            unconditional: Some(unconditional),
        };
        let guided = default_cfg(&harness.backend, &predictions, 2.0, &context)?;
        assert_eq!(
            &*tensor_to_f32(&harness.backend, &guided, &context)?,
            &[1.6875, 3.5625]
        );
        assert!(matches!(
            GuidanceError::from(ModelFamilyError::OutOfMemory {
                required: 2,
                budget: 1,
            }),
            GuidanceError::FamilyModel(ModelFamilyError::OutOfMemory {
                required: 2,
                budget: 1,
            })
        ));
        Ok(())
    }

    #[test]
    fn task629_latent_operation_pre_cfg_uses_conditional_delta_and_restarts_cleanly() -> TestResult
    {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let profile = profile()?;
        let plan = plan(&profile, 4.0)?;
        let conditional_identity = identity("task629-positive")?;
        let unconditional_identity = identity("task629-negative")?;
        let hook = GuidanceHookContext {
            plan: &plan,
            guidance_scale: plan.guidance(),
            sigma: 0.5,
            sampling_percent: 0.5,
            conditional_identity: &conditional_identity,
            unconditional_identity: &unconditional_identity,
        };
        let chain =
            NativeGuidanceModifierChain::checked(vec![NativeGuidanceModifier::LatentOperation {
                operation: NativeLatentOperationKind::TonemapReinhard { multiplier: 1.0 },
            }])?;
        let make_predictions = || -> Result<GuidancePredictions, Box<dyn Error>> {
            Ok(GuidancePredictions {
                conditional: harness.tensor(&[1, 1, 1, 3], &[2.0, 3.0, 4.0], &context)?,
                unconditional: Some(harness.tensor(&[1, 1, 1, 3], &[1.0, 1.0, 1.0], &context)?),
            })
        };
        let mut predictions = make_predictions()?;
        let mut session = chain.begin_attempt(&harness.backend, 629)?;
        session.pre_cfg(&hook, &mut predictions, &context)?;
        let values = tensor_to_f32(&harness.backend, predictions.conditional(), &context)?;
        for (actual, expected) in values.iter().zip([1.875, 1.0 + 14.0 / 9.0, 3.1]) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        let mut replay = make_predictions()?;
        chain
            .begin_attempt(&harness.backend, 629)?
            .pre_cfg(&hook, &mut replay, &context)?;
        assert_eq!(
            &*tensor_to_f32(&harness.backend, predictions.conditional(), &context)?,
            &*tensor_to_f32(&harness.backend, replay.conditional(), &context)?,
        );
        assert_eq!(
            chain,
            NativeGuidanceModifierChain::checked(chain.modifiers().to_vec())?
        );
        Ok(())
    }

    #[test]
    fn task407_modifier_chain_preserves_apg_attempt_state_and_video_cfg_order() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let profile = profile()?;
        let plan = plan(&profile, 4.0)?;
        let conditional_identity = identity("task407-positive")?;
        let unconditional_identity = identity("task407-negative")?;
        let hook = GuidanceHookContext {
            plan: &plan,
            guidance_scale: plan.guidance(),
            sigma: 0.5,
            sampling_percent: 0.5,
            conditional_identity: &conditional_identity,
            unconditional_identity: &unconditional_identity,
        };
        let chain = NativeGuidanceModifierChain::checked(vec![
            NativeGuidanceModifier::AdaptiveProjectedGuidance {
                eta: 0.0,
                norm_threshold: 0.0,
                momentum: 0.5,
            },
            NativeGuidanceModifier::CfgOverride { scale: 2.0 },
            NativeGuidanceModifier::VideoLinearCfg { minimum_scale: 1.0 },
            NativeGuidanceModifier::VideoTriangleCfg { minimum_scale: 0.5 },
        ])?;
        let mut first_session = chain.begin_attempt(&harness.backend, 17)?;
        let mut predictions = GuidancePredictions {
            conditional: harness.tensor(
                &[3, 1, 1, 2],
                &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                &context,
            )?,
            unconditional: Some(harness.tensor(
                &[3, 1, 1, 2],
                &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                &context,
            )?),
        };
        first_session.pre_cfg(&hook, &mut predictions, &context)?;
        let running_after_first = first_session
            .states
            .first()
            .and_then(Option::as_ref)
            .ok_or("APG did not retain first-step state")?
            .running_average
            .to_vec();
        let current = default_cfg(&harness.backend, &predictions, 4.0, &context)?;
        let guided = first_session
            .override_cfg(&hook, &predictions, &current, &context)?
            .ok_or("video CFG did not override")?;
        let guided_values = tensor_to_f32(&harness.backend, &guided, &context)?;
        assert_eq!(&*guided_values, &[0.125, 0.375, 1.0, -4.0, 0.125, 0.375]);

        first_session.pre_cfg(&hook, &mut predictions, &context)?;
        assert_ne!(
            &*first_session
                .states
                .first()
                .and_then(Option::as_ref)
                .ok_or("APG lost accumulated state")?
                .running_average,
            running_after_first.as_slice(),
        );
        let reset_hook = GuidanceHookContext {
            sigma: 0.75,
            ..hook
        };
        let mut replay_predictions = predictions.clone();
        first_session.pre_cfg(&reset_hook, &mut predictions, &context)?;
        let mut replay_session = chain.begin_attempt(&harness.backend, 17)?;
        replay_session.pre_cfg(&reset_hook, &mut replay_predictions, &context)?;
        let reset_values = tensor_to_f32(&harness.backend, predictions.conditional(), &context)?;
        let fresh_values =
            tensor_to_f32(&harness.backend, replay_predictions.conditional(), &context)?;
        assert_eq!(&*reset_values, &*fresh_values);
        assert_eq!(replay_session.attempt_seed(), 17);
        assert_eq!(
            chain,
            NativeGuidanceModifierChain::checked(chain.modifiers().to_vec())?
        );
        assert!(matches!(
            NativeGuidanceModifierChain::checked(vec![NativeGuidanceModifier::CfgOverride {
                scale: f32::NAN,
            }]),
            Err(GuidanceError::InvalidModifier)
        ));
        assert!(matches!(
            NativeGuidanceModifierChain::checked(vec![
                NativeGuidanceModifier::CfgOverride {
                    scale: 1.0
                };
                MAX_GUIDANCE_MODIFIERS + 1
            ]),
            Err(GuidanceError::ModifierLimit { .. })
        ));
        let mut encoded = serde_json::to_value(&chain)?;
        encoded["semantic_digest_sha256"] = serde_json::Value::String("0".repeat(64));
        assert!(serde_json::from_value::<NativeGuidanceModifierChain>(encoded).is_err());

        let constrained = Harness::new(1 << 20, 20)?;
        let constrained_context = constrained.context()?;
        let mut constrained_session = chain.begin_attempt(&constrained.backend, 17)?;
        let original_conditional = constrained.tensor(
            &[3, 1, 1, 2],
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            &constrained_context,
        )?;
        let mut constrained_predictions = GuidancePredictions {
            conditional: original_conditional.clone(),
            unconditional: Some(constrained.tensor(
                &[3, 1, 1, 2],
                &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                &constrained_context,
            )?),
        };
        let constrained_error = constrained_session
            .pre_cfg(&hook, &mut constrained_predictions, &constrained_context)
            .expect_err("APG must reject an insufficient caller workspace");
        assert!(
            matches!(
                constrained_error,
                GuidanceError::Tensor(TensorError::WorkspaceAuthorizationExceeded { .. })
                    | GuidanceError::Tensor(TensorError::AllocationFailed { .. })
                    | GuidanceError::NativeDiffusion(NativeDiffusionTensorError::Tensor(
                        TensorError::WorkspaceAuthorizationExceeded { .. }
                    ))
                    | GuidanceError::NativeDiffusion(NativeDiffusionTensorError::Tensor(
                        TensorError::AllocationFailed { .. }
                    ))
            ),
            "unexpected APG workspace failure: {constrained_error:?}",
        );
        assert_eq!(
            constrained_predictions.conditional().contiguous_bytes()?,
            original_conditional.contiguous_bytes()?,
        );
        assert!(constrained_session.states.iter().all(Option::is_none));
        Ok(())
    }

    #[test]
    fn task407_dual_cfg_and_dual_model_follow_source_branch_equations() -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let negative = conditioning(
            &harness,
            "negative",
            vec![("left", full_options())],
            &context,
        )?;
        let middle = conditioning(
            &harness,
            "middle",
            vec![("right", full_options())],
            &context,
        )?;
        let positive = conditioning(
            &harness,
            "positive",
            vec![("strong", full_options())],
            &context,
        )?;
        let profile = profile()?;
        let plan = plan(&profile, 5.0)?;
        for nested in [false, true] {
            let mut denoiser = ValueDenoiser {
                backend: &harness.backend,
                batches: Vec::new(),
                cancel_after_evaluation: false,
            };
            let result = execute_dual_cfg_guidance(
                &harness.backend,
                &latent,
                2.0,
                &profile,
                &plan,
                &positive,
                &middle,
                &negative,
                3.0,
                5.0,
                nested,
                GuidanceOptions::default(),
                &mut denoiser,
                &mut [],
                &context,
            )?;
            assert_eq!(
                values(&harness, result.guided(), &context)?,
                vec![if nested { 42.0 } else { 18.0 }; 2],
            );
            assert_eq!(result.denoiser_evaluations(), 3);
        }

        let operation =
            |identifier: &str, order, kind, scale, bias, rng_address| ExecutionHookOperation {
                identifier: identifier.to_owned(),
                order,
                kind,
                scale,
                bias,
                start_percent: 0.0,
                end_percent: 1.0,
                rng_address,
                rng_amplitude: 0.0,
                hypernetwork: None,
                context_window: None,
                sampling_profile: None,
            };
        let graph = ExecutionHookGraph::checked(
            407,
            vec![
                operation("dual-cfg-hook", 1, ExecutionHookKind::Cfg, 2.0, 0.0, 11),
                operation(
                    "dual-post-cfg-hook",
                    2,
                    ExecutionHookKind::PostCfg,
                    1.0,
                    3.0,
                    12,
                ),
            ],
        )?;
        let modifiers =
            NativeGuidanceModifierChain::checked(vec![NativeGuidanceModifier::CfgOverride {
                scale: 4.0,
            }])?;
        let mut modifier_session = modifiers.begin_attempt(&harness.backend, 407)?;
        let mut dual_cfg_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let dual_cfg = execute_dual_cfg_guidance_with_modifier_session(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &positive,
            &middle,
            &negative,
            3.0,
            5.0,
            false,
            GuidanceOptions::default(),
            &mut dual_cfg_denoiser,
            &graph,
            &mut modifier_session,
            &context,
        )?;
        assert_eq!(
            values(&harness, dual_cfg.guided(), &context)?,
            vec![19.0; 2]
        );

        let mut positive_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let mut negative_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let dual_model = execute_dual_model_guidance(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &positive,
            &negative,
            GuidanceOptions::default(),
            &mut positive_denoiser,
            &mut negative_denoiser,
            &context,
        )?;
        assert_eq!(
            values(&harness, dual_model.guided(), &context)?,
            vec![22.0; 2]
        );
        assert_eq!(positive_denoiser.batches.len(), 1);
        assert_eq!(negative_denoiser.batches.len(), 1);

        let mut positive_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let mut negative_denoiser = ValueDenoiser {
            backend: &harness.backend,
            batches: Vec::new(),
            cancel_after_evaluation: false,
        };
        let mut modifier_session = modifiers.begin_attempt(&harness.backend, 407)?;
        let dual_model = execute_dual_model_guidance_with_modifier_session(
            &harness.backend,
            &latent,
            2.0,
            &profile,
            &plan,
            &positive,
            &negative,
            GuidanceOptions::default(),
            &mut positive_denoiser,
            &mut negative_denoiser,
            &graph,
            &graph,
            &mut modifier_session,
            &context,
        )?;
        assert_eq!(
            values(&harness, dual_model.guided(), &context)?,
            vec![21.0; 2]
        );
        Ok(())
    }

    #[test]
    fn regional_accumulation_consumes_source_binary64_through_existing_cpu_f32_scalar_profile()
    -> TestResult {
        use sim_tensor::{BinaryOperation, Scalar, ScalarSide, TensorBackend};
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let output = harness.tensor(&[1, 1, 2], &[3.0, 7.0], &context)?;
        let ones = harness.tensor(&[1, 1, 2], &[1.0, 1.0], &context)?;
        for strength in [
            0.1,
            f64::from_bits(0.1_f64.to_bits() + 1),
            f64::from(0.1_f32),
            0.3,
            10.0,
        ] {
            let entry = ConditioningEntry::checked(
                "source",
                ConditioningValue::regular(harness.tensor(&[1, 1], &[1.0], &context)?)?,
                ConditioningEntryOptions {
                    strength,
                    ..Default::default()
                },
            )?;
            let resolved = entry.resolve(latent.descriptor(), &harness.backend, &context)?;
            assert_eq!(resolved.strength().to_bits(), strength.to_bits());
            let expected_weights = harness
                .backend
                .binary_scalar(
                    BinaryOperation::Multiply,
                    &ones,
                    Scalar::Float(strength),
                    ScalarSide::Right,
                    ones.descriptor().clone(),
                    &context,
                )?
                .0;
            let expected_weights = tensor_to_f32(&harness.backend, &expected_weights, &context)?;
            let mut sum = [0.0_f32; 2];
            let mut weights = [0.0_f32; 2];
            accumulate_entry(
                &harness.backend,
                &latent,
                &resolved,
                &output,
                &mut sum,
                &mut weights,
                &context,
            )?;
            assert_eq!(&weights, &*expected_weights);
            for ((actual, prediction), weight) in
                sum.iter().zip([3.0_f32, 7.0]).zip(expected_weights.iter())
            {
                assert_eq!(actual.to_bits(), (prediction * *weight).to_bits());
            }
        }
        Ok(())
    }

    #[test]
    fn regional_binary64_strength_preserves_two_source_mask_scalar_steps_and_input_aliases()
    -> TestResult {
        use sim_tensor::{BinaryOperation, Scalar, ScalarSide, TensorBackend};
        let harness = Harness::new(1 << 20, 1 << 18)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let output = harness.tensor(&[1, 1, 2], &[3.0, 7.0], &context)?;
        let mask = harness.tensor(&[2], &[0.1, 0.9], &context)?;
        let entry = ConditioningEntry::checked(
            "source",
            ConditioningValue::regular(harness.tensor(&[1, 1], &[1.0], &context)?)?,
            ConditioningEntryOptions {
                strength: 0.1,
                mask: Some(ConditioningMask::new(mask.clone(), 0.5, vec![0], false)?),
                ..Default::default()
            },
        )?;
        let resolved = entry.resolve(latent.descriptor(), &harness.backend, &context)?;
        let normalized = resolved.mask().ok_or("Resolved mask")?.tensor();
        let mask_scaled = harness
            .backend
            .binary_scalar(
                BinaryOperation::Multiply,
                normalized,
                Scalar::Float(0.5),
                ScalarSide::Right,
                normalized.descriptor().clone(),
                &context,
            )?
            .0;
        let expected = harness
            .backend
            .binary_scalar(
                BinaryOperation::Multiply,
                &mask_scaled,
                Scalar::Float(0.1),
                ScalarSide::Right,
                mask_scaled.descriptor().clone(),
                &context,
            )?
            .0;
        let expected = tensor_to_f32(&harness.backend, &expected, &context)?;
        let mut sum = [0.0_f32; 2];
        let mut weights = [0.0_f32; 2];
        accumulate_entry(
            &harness.backend,
            &latent,
            &resolved,
            &output,
            &mut sum,
            &mut weights,
            &context,
        )?;
        assert_eq!(&weights, &*expected);
        for ((actual, prediction), weight) in sum.iter().zip([3.0_f32, 7.0]).zip(expected.iter()) {
            assert_eq!(actual.to_bits(), (prediction * *weight).to_bits());
        }
        assert_eq!(values(&harness, &mask, &context)?, [0.1, 0.9]);
        assert_eq!(entry.options().strength.to_bits(), 0.1_f64.to_bits());
        Ok(())
    }

    #[test]
    fn regional_binary64_strength_precancel_and_read_workspace_oom_do_not_mutate_accumulators()
    -> TestResult {
        let harness = Harness::new(1 << 20, 1 << 20)?;
        let context = harness.context()?;
        let latent = harness.tensor(&[1, 1, 2], &[0.0, 0.0], &context)?;
        let output = harness.tensor(&[1, 1, 2], &[3.0, 7.0], &context)?;
        let entry = ConditioningEntry::checked(
            "source",
            ConditioningValue::regular(harness.tensor(&[1, 1], &[1.0], &context)?)?,
            ConditioningEntryOptions {
                strength: 0.1,
                ..Default::default()
            },
        )?;
        let resolved = entry.resolve(latent.descriptor(), &harness.backend, &context)?;
        let remaining = (1_u64 << 20)
            .checked_sub(harness.backend.memory_snapshot().current_bytes)
            .ok_or("Budget")?;
        let reservation = harness.backend.reserve_workspace(&context, remaining)?;
        let mut sum = [5.0_f32; 2];
        let mut weights = [9.0_f32; 2];
        assert!(
            accumulate_entry(
                &harness.backend,
                &latent,
                &resolved,
                &output,
                &mut sum,
                &mut weights,
                &context
            )
            .is_err()
        );
        assert_eq!(sum, [5.0; 2]);
        assert_eq!(weights, [9.0; 2]);
        drop(reservation);
        accumulate_entry(
            &harness.backend,
            &latent,
            &resolved,
            &output,
            &mut sum,
            &mut weights,
            &context,
        )?;
        assert_ne!(sum, [5.0; 2]);
        assert_ne!(weights, [9.0; 2]);
        harness.cancellation.cancel();
        let before = (sum, weights);
        assert!(
            accumulate_entry(
                &harness.backend,
                &latent,
                &resolved,
                &output,
                &mut sum,
                &mut weights,
                &context
            )
            .is_err()
        );
        assert_eq!((sum, weights), before);
        Ok(())
    }
}
