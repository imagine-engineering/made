use crate::{
    GENERATED_SCHEDULER_DEFINITIONS, SamplingError, SamplingPlan, SamplingProfile,
    SamplingProfileError, validate_identifier,
};
use serde::{Deserialize, Deserializer, Serialize};
use sim_tensor::{CpuBackend, CpuWorkspaceVec, ExecutionContext, TensorError};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const NORMAL_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0209";
pub const SIMPLE_SCHEDULER_FEATURE_ID: &str = "COMFY-MODEL-0211";
pub const NORMAL_SCHEDULER_ID: &str = "normal";
pub const SIMPLE_SCHEDULER_ID: &str = "simple";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct SchedulerIdentity(String);

impl SchedulerIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, SchedulerError> {
        let value = value.into();
        validate_identifier("scheduler", &value)
            .map_err(|_| SchedulerError::InvalidIdentity(value.clone()))?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SchedulerIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerDefinition {
    pub identity: &'static str,
    pub feature_id: &'static str,
    pub source_ordinal: u16,
    pub aliases: &'static [&'static str],
    pub implementation_module: &'static str,
}

pub const SIMPLE_FOUNDATION_DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: SIMPLE_SCHEDULER_ID,
    feature_id: SIMPLE_SCHEDULER_FEATURE_ID,
    source_ordinal: 0,
    aliases: &[],
    implementation_module: "catalog/simple",
};

pub const NORMAL_FOUNDATION_DEFINITION: SchedulerDefinition = SchedulerDefinition {
    identity: NORMAL_SCHEDULER_ID,
    feature_id: NORMAL_SCHEDULER_FEATURE_ID,
    source_ordinal: 6,
    aliases: &[],
    implementation_module: "schedulers/native_diffusion",
};

#[derive(Clone, Debug)]
pub struct SchedulerRegistry {
    definitions: Vec<SchedulerDefinition>,
    lookup: BTreeMap<String, usize>,
}

impl SchedulerRegistry {
    pub fn foundational() -> Result<Self, SchedulerError> {
        let mut definitions = GENERATED_SCHEDULER_DEFINITIONS.to_vec();
        for fallback in [SIMPLE_FOUNDATION_DEFINITION, NORMAL_FOUNDATION_DEFINITION] {
            if !definitions
                .iter()
                .any(|definition| definition.identity == fallback.identity)
            {
                definitions.push(fallback);
            }
        }
        Self::new(definitions)
    }

    pub fn new(mut definitions: Vec<SchedulerDefinition>) -> Result<Self, SchedulerError> {
        definitions.sort_by_key(|definition| definition.source_ordinal);
        let mut identities = BTreeSet::new();
        let mut features = BTreeSet::new();
        let mut ordinals = BTreeSet::new();
        let mut modules = BTreeSet::new();
        let mut lookup = BTreeMap::new();
        for (index, definition) in definitions.iter().enumerate() {
            SchedulerIdentity::new(definition.identity)?;
            validate_feature_id(definition.feature_id)?;
            if !identities.insert(definition.identity) {
                return Err(SchedulerError::DuplicateIdentity(
                    definition.identity.to_owned(),
                ));
            }
            if !features.insert(definition.feature_id) {
                return Err(SchedulerError::DuplicateFeatureId(
                    definition.feature_id.to_owned(),
                ));
            }
            if !ordinals.insert(definition.source_ordinal) {
                return Err(SchedulerError::DuplicateSourceOrdinal(
                    definition.source_ordinal,
                ));
            }
            if !modules.insert(definition.implementation_module) {
                return Err(SchedulerError::DuplicateImplementationModule(
                    definition.implementation_module.to_owned(),
                ));
            }
            insert_lookup(&mut lookup, definition.identity, index)?;
            for alias in definition.aliases {
                SchedulerIdentity::new(*alias)?;
                insert_lookup(&mut lookup, alias, index)?;
            }
        }
        if definitions.first().map(|definition| definition.identity) != Some(SIMPLE_SCHEDULER_ID) {
            return Err(SchedulerError::InvalidDefault);
        }
        Ok(Self {
            definitions,
            lookup,
        })
    }

    pub fn default_definition(&self) -> &SchedulerDefinition {
        &self.definitions[0]
    }

    pub fn resolve(
        &self,
        identity: &SchedulerIdentity,
    ) -> Result<&SchedulerDefinition, SchedulerError> {
        self.lookup
            .get(identity.as_str())
            .and_then(|index| self.definitions.get(*index))
            .ok_or_else(|| SchedulerError::Unknown(identity.as_str().to_owned()))
    }

    pub fn definitions(&self) -> &[SchedulerDefinition] {
        &self.definitions
    }
}

fn insert_lookup(
    lookup: &mut BTreeMap<String, usize>,
    value: &str,
    index: usize,
) -> Result<(), SchedulerError> {
    if lookup.insert(value.to_owned(), index).is_some() {
        return Err(SchedulerError::DuplicateAlias(value.to_owned()));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SchedulerRequestWire")]
pub struct SchedulerRequest {
    pub identity: SchedulerIdentity,
    pub steps: u32,
    pub denoise: f32,
    pub start_step: Option<u32>,
    pub end_step: Option<u32>,
    #[serde(default)]
    pub penultimate_sigma_policy: PenultimateSigmaPolicy,
}

#[derive(Deserialize)]
struct SchedulerRequestWire {
    identity: SchedulerIdentity,
    steps: u32,
    denoise: f32,
    start_step: Option<u32>,
    end_step: Option<u32>,
    #[serde(default)]
    penultimate_sigma_policy: PenultimateSigmaPolicy,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PenultimateSigmaPolicy {
    #[default]
    Keep,
    Discard,
}

impl SchedulerRequest {
    pub fn new(
        identity: impl Into<String>,
        steps: u32,
        denoise: f32,
    ) -> Result<Self, SchedulerError> {
        let request = Self {
            identity: SchedulerIdentity::new(identity)?,
            steps,
            denoise,
            start_step: None,
            end_step: None,
            penultimate_sigma_policy: PenultimateSigmaPolicy::Keep,
        };
        request.validate_bounds()?;
        Ok(request)
    }

    pub fn for_sampling_plan(plan: &SamplingPlan) -> Result<Self, SchedulerError> {
        let policy = match plan.sampler().as_str() {
            "dpm_2" | "dpm_2_ancestral" | "uni_pc" | "uni_pc_bh2" => {
                PenultimateSigmaPolicy::Discard
            }
            _ => PenultimateSigmaPolicy::Keep,
        };
        Ok(
            Self::new(plan.scheduler().as_str(), plan.steps(), plan.denoise())?
                .with_penultimate_sigma_policy(policy),
        )
    }

    pub fn with_window(
        mut self,
        start_step: Option<u32>,
        end_step: Option<u32>,
    ) -> Result<Self, SchedulerError> {
        self.start_step = start_step;
        self.end_step = end_step;
        self.validate_bounds()?;
        Ok(self)
    }

    pub fn with_penultimate_sigma_policy(mut self, policy: PenultimateSigmaPolicy) -> Self {
        self.penultimate_sigma_policy = policy;
        self
    }

    pub fn validate(
        &self,
        registry: &SchedulerRegistry,
        profile: &impl SamplingProfile,
    ) -> Result<(), SchedulerError> {
        registry.resolve(&self.identity)?;
        self.validate_bounds()?;
        if profile.sigma_count() < 2 {
            return Err(SchedulerError::InvalidProfile(
                "profile must contain at least two sigmas".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_bounds(&self) -> Result<(), SchedulerError> {
        if self.steps == 0 {
            return Err(SchedulerError::ZeroSteps);
        }
        if !self.denoise.is_finite() || self.denoise <= 0.0 || self.denoise > 1.0 {
            return Err(SchedulerError::InvalidDenoise(self.denoise));
        }
        let start = self.start_step.unwrap_or(0);
        let end = self.end_step.unwrap_or(self.steps);
        if start >= end || end > self.steps {
            return Err(SchedulerError::InvalidWindow {
                start,
                end,
                steps: self.steps,
            });
        }
        Ok(())
    }
}

impl TryFrom<SchedulerRequestWire> for SchedulerRequest {
    type Error = SchedulerError;

    fn try_from(wire: SchedulerRequestWire) -> Result<Self, Self::Error> {
        Ok(Self::new(wire.identity.as_str(), wire.steps, wire.denoise)?
            .with_window(wire.start_step, wire.end_step)?
            .with_penultimate_sigma_policy(wire.penultimate_sigma_policy))
    }
}

pub fn schedule_for_sampling_plan(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    profile: &impl SamplingProfile,
    plan: &SamplingPlan,
) -> Result<Vec<f32>, SchedulerError> {
    let registry = SchedulerRegistry::foundational()?;
    let request = SchedulerRequest::for_sampling_plan(plan)?;
    request.validate(&registry, profile)?;
    match plan.scheduler().as_str() {
        "beta" => crate::generated_beta_sim_model_0203::beta_schedule(
            backend, context, &registry, profile, &request,
        ),
        "ddim_uniform" => crate::generated_ddim_uniform_sim_model_0204::ddim_uniform_schedule(
            backend, context, &registry, profile, &request,
        ),
        "exponential" => crate::generated_exponential_sim_model_0205::exponential_schedule(
            backend, context, &registry, profile, &request,
        ),
        "karras" => crate::generated_karras_sim_model_0206::karras_schedule(
            backend, context, &registry, profile, &request,
        ),
        "kl_optimal" => crate::generated_kl_optimal_sim_model_0207::kl_optimal_schedule(
            backend, context, &registry, profile, &request,
        ),
        "linear_quadratic" => {
            crate::generated_linear_quadratic_sim_model_0208::linear_quadratic_schedule(
                backend, context, &registry, profile, &request,
            )
        }
        "sgm_uniform" => crate::generated_sgm_uniform_sim_model_0210::sgm_uniform_schedule(
            backend, context, &registry, profile, &request,
        ),
        "simple" => crate::generated_simple_sim_model_0211::simple_schedule(
            backend, context, &registry, profile, &request,
        ),
        "normal" if profile.prediction() == crate::PredictionInterpretation::Flow => {
            build_scheduler_schedule(
                backend,
                context,
                &registry,
                profile,
                &request,
                NORMAL_SCHEDULER_ID,
                |steps, profile, context, schedule| {
                    let sigmas = crate::generated_native_diffusion::normal_sigmas_for_profile(
                        profile, backend, context, steps, 1.0,
                    )
                    .map_err(|error| {
                        if context.cancellation.is_cancelled() {
                            SchedulerError::Cancelled
                        } else {
                            SchedulerError::InvalidProfile(error.to_string())
                        }
                    })?;
                    for sigma in sigmas {
                        schedule.try_push(sigma)?;
                    }
                    Ok(())
                },
            )
        }
        "normal" => normal_schedule(backend, context, &registry, profile, &request),
        identity => Err(SchedulerError::Unknown(identity.to_owned())),
    }
}

pub fn select_ksampler_step_window(
    mut sigmas: Vec<f32>,
    start_step: Option<usize>,
    last_step: Option<usize>,
    force_full_denoise: bool,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, SchedulerError> {
    context
        .cancellation
        .check()
        .map_err(|_| SchedulerError::Cancelled)?;
    if let Some(last_step) = last_step.filter(|step| *step < sigmas.len().saturating_sub(1)) {
        sigmas.truncate(last_step + 1);
        if force_full_denoise {
            if let Some(last) = sigmas.last_mut() {
                *last = 0.0;
            }
        }
    }
    if let Some(start_step) = start_step {
        if start_step < sigmas.len().saturating_sub(1) {
            sigmas.drain(..start_step);
        } else {
            sigmas.clear();
        }
    }
    Ok(sigmas)
}

pub fn normal_schedule(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &impl SamplingProfile,
    request: &SchedulerRequest,
) -> Result<Vec<f32>, SchedulerError> {
    normal_schedule_with_mode(
        backend,
        context,
        registry,
        profile,
        request,
        NORMAL_SCHEDULER_ID,
        false,
    )
}

pub fn normal_schedule_with_mode<P: SamplingProfile>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &P,
    request: &SchedulerRequest,
    expected_identity: &'static str,
    sgm_uniform: bool,
) -> Result<Vec<f32>, SchedulerError> {
    build_scheduler_schedule(
        backend,
        context,
        registry,
        profile,
        request,
        expected_identity,
        |effective_steps, profile, context, full| {
            let last_time = (profile.sigma_count() - 1) as f32;
            let end_is_zero = profile.sigma_at_index(0)?.abs() <= 0.00001;
            let schedule_steps = if !sgm_uniform && end_is_zero {
                effective_steps
                    .checked_add(1)
                    .ok_or(SchedulerError::StepOverflow)?
            } else {
                effective_steps
            };
            if schedule_steps == 1 {
                full.try_push(profile.sigma_at_model_time(last_time)?)?;
            } else {
                for index in 0..schedule_steps {
                    if index.is_multiple_of(256) {
                        context
                            .cancellation
                            .check()
                            .map_err(|_| SchedulerError::Cancelled)?;
                    }
                    let denominator = if sgm_uniform {
                        schedule_steps
                    } else {
                        schedule_steps - 1
                    };
                    let fraction = index as f32 / denominator as f32;
                    let model_time = (0.0 - last_time).mul_add(fraction, last_time);
                    full.try_push(profile.sigma_at_model_time(model_time)?)?;
                }
            }
            if sgm_uniform || !end_is_zero {
                full.try_push(0.0)?;
            }
            Ok(())
        },
    )
}

pub fn build_scheduler_schedule<P, F>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &P,
    request: &SchedulerRequest,
    expected_identity: &'static str,
    equation: F,
) -> Result<Vec<f32>, SchedulerError>
where
    P: SamplingProfile,
    F: FnOnce(
        usize,
        &P,
        &ExecutionContext<'_>,
        &mut CpuWorkspaceVec<f32>,
    ) -> Result<(), SchedulerError>,
{
    build_scheduler_schedule_with_capacity(
        backend,
        context,
        registry,
        profile,
        request,
        expected_identity,
        1,
        equation,
    )
}

pub(crate) fn build_scheduler_schedule_with_capacity<P, F>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &SchedulerRegistry,
    profile: &P,
    request: &SchedulerRequest,
    expected_identity: &'static str,
    equation_extra_capacity: usize,
    equation: F,
) -> Result<Vec<f32>, SchedulerError>
where
    P: SamplingProfile,
    F: FnOnce(
        usize,
        &P,
        &ExecutionContext<'_>,
        &mut CpuWorkspaceVec<f32>,
    ) -> Result<(), SchedulerError>,
{
    if !(1..=2).contains(&equation_extra_capacity) {
        return Err(SchedulerError::InvalidEquationCapacity(
            equation_extra_capacity,
        ));
    }
    request.validate(registry, profile)?;
    if request.identity.as_str() != expected_identity {
        return Err(SchedulerError::AlgorithmMismatch {
            expected: expected_identity,
            actual: request.identity.as_str().to_owned(),
        });
    }
    context
        .cancellation
        .check()
        .map_err(|_| SchedulerError::Cancelled)?;
    let mut effective_steps = if request.denoise > 0.9999 {
        usize::try_from(request.steps).map_err(|_| SchedulerError::StepOverflow)?
    } else {
        let expanded = ((f64::from(request.steps)) / f64::from(request.denoise)).floor();
        if !expanded.is_finite() || expanded < 1.0 || expanded > usize::MAX as f64 {
            return Err(SchedulerError::StepOverflow);
        }
        expanded as usize
    };
    if request.penultimate_sigma_policy == PenultimateSigmaPolicy::Discard {
        effective_steps = effective_steps
            .checked_add(1)
            .ok_or(SchedulerError::StepOverflow)?;
    }
    let capacity = effective_steps
        .checked_add(equation_extra_capacity)
        .ok_or(SchedulerError::StepOverflow)?;
    let mut full = backend.workspace_vec::<f32>(context, capacity)?;
    equation(effective_steps, profile, context, &mut full)?;
    context
        .cancellation
        .check()
        .map_err(|_| SchedulerError::Cancelled)?;
    if full.is_empty() {
        return Err(SchedulerError::EmptySchedule);
    }
    if let Some((index, value)) = full
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SchedulerError::NonFiniteSigma { index, value });
    }

    let discarded_index = if request.penultimate_sigma_policy == PenultimateSigmaPolicy::Discard {
        if full.len() < 3 {
            return Err(SchedulerError::PenultimateSigmaUnavailable);
        }
        Some(full.len() - 2)
    } else {
        None
    };
    let post_policy_length = full.len() - usize::from(discarded_index.is_some());
    let mut post_policy = Vec::new();
    post_policy
        .try_reserve_exact(post_policy_length)
        .map_err(|_| SchedulerError::OutOfMemory("penultimate sigma policy"))?;
    post_policy.extend(
        full.iter()
            .enumerate()
            .filter_map(|(index, value)| (Some(index) != discarded_index).then_some(*value)),
    );

    let requested_steps =
        usize::try_from(request.steps).map_err(|_| SchedulerError::StepOverflow)?;
    let requested_schedule_length = requested_steps
        .checked_add(1)
        .ok_or(SchedulerError::StepOverflow)?;
    let denoise_start = if request.denoise > 0.9999 {
        0
    } else {
        post_policy.len().saturating_sub(requested_schedule_length)
    };
    let denoised = post_policy
        .get(denoise_start..)
        .ok_or(SchedulerError::StepOverflow)?;
    let start = request
        .start_step
        .map(usize::try_from)
        .transpose()
        .map_err(|_| SchedulerError::StepOverflow)?;
    let end = request
        .end_step
        .map(usize::try_from)
        .transpose()
        .map_err(|_| SchedulerError::StepOverflow)?;
    let end_exclusive = match end {
        Some(end) if end < denoised.len().saturating_sub(1) => {
            end.checked_add(1).ok_or(SchedulerError::StepOverflow)?
        }
        _ => denoised.len(),
    };
    let ended = denoised
        .get(..end_exclusive)
        .ok_or(SchedulerError::StepOverflow)?;
    let selected = match start {
        Some(start) if start < ended.len().saturating_sub(1) => {
            ended.get(start..).ok_or(SchedulerError::StepOverflow)?
        }
        Some(_) => &[],
        None => ended,
    };
    let mut selected_schedule = Vec::new();
    selected_schedule
        .try_reserve_exact(selected.len())
        .map_err(|_| SchedulerError::OutOfMemory("selected sigma schedule"))?;
    selected_schedule.extend_from_slice(selected);
    context
        .cancellation
        .check()
        .map_err(|_| SchedulerError::Cancelled)?;
    Ok(selected_schedule)
}

fn validate_feature_id(value: &str) -> Result<(), SchedulerError> {
    if value.len() != "COMFY-MODEL-0000".len()
        || !value.starts_with("COMFY-MODEL-")
        || !value["COMFY-MODEL-".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_digit())
    {
        return Err(SchedulerError::InvalidFeatureId(value.to_owned()));
    }
    Ok(())
}

impl From<SchedulerError> for SamplingError {
    fn from(error: SchedulerError) -> Self {
        SamplingError::Scheduler(error.to_string())
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SchedulerError {
    #[error("invalid scheduler identity {0:?}")]
    InvalidIdentity(String),
    #[error("invalid scheduler feature ID {0:?}")]
    InvalidFeatureId(String),
    #[error("duplicate scheduler identity {0:?}")]
    DuplicateIdentity(String),
    #[error("duplicate scheduler feature ID {0:?}")]
    DuplicateFeatureId(String),
    #[error("duplicate scheduler source ordinal {0}")]
    DuplicateSourceOrdinal(u16),
    #[error("duplicate scheduler implementation module {0:?}")]
    DuplicateImplementationModule(String),
    #[error("duplicate scheduler identity or alias {0:?}")]
    DuplicateAlias(String),
    #[error("the source-order scheduler default must be simple")]
    InvalidDefault,
    #[error("unknown scheduler {0:?}")]
    Unknown(String),
    #[error("scheduler steps must be nonzero")]
    ZeroSteps,
    #[error("scheduler denoise must be finite and in (0, 1], got {0}")]
    InvalidDenoise(f32),
    #[error("scheduler window {start}..{end} is invalid for {steps} steps")]
    InvalidWindow { start: u32, end: u32, steps: u32 },
    #[error("scheduler algorithm mismatch: expected {expected}, got {actual}")]
    AlgorithmMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("invalid sampling profile: {0}")]
    InvalidProfile(String),
    #[error("invalid scheduler parameters: {0}")]
    InvalidParameters(&'static str),
    #[error("scheduler step arithmetic overflowed")]
    StepOverflow,
    #[error("scheduler cannot discard a penultimate sigma from fewer than three values")]
    PenultimateSigmaUnavailable,
    #[error("scheduler equation produced an empty sigma schedule")]
    EmptySchedule,
    #[error("scheduler equation produced non-finite sigma {value} at index {index}")]
    NonFiniteSigma { index: usize, value: f32 },
    #[error("scheduler equation requested invalid extra capacity {0}; expected one or two")]
    InvalidEquationCapacity(usize),
    #[error("scheduler allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("scheduler operation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Profile(#[from] SamplingProfileError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Creation(#[from] sim_tensor::generated_tensor_creation_01::TensorCreationPartOneError),
}

pub(crate) fn validate_parameterized_sigma_bounds(
    steps: u64,
    minimum: f64,
    maximum: f64,
) -> Result<usize, SchedulerError> {
    if !(1..=10000).contains(&steps)
        || !minimum.is_finite()
        || !(0.0..=5000.0).contains(&minimum)
        || !maximum.is_finite()
        || !(0.0..=5000.0).contains(&maximum)
    {
        return Err(SchedulerError::InvalidParameters(
            "steps and sigma bounds exceed the source node limits",
        ));
    }
    usize::try_from(steps).map_err(|_| SchedulerError::StepOverflow)
}

pub(crate) fn publish_parameterized_sigmas(
    backend: &CpuBackend,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    context.check()?;
    for (index, value) in values.iter().copied().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        if !value.is_finite() {
            return Err(SchedulerError::NonFiniteSigma { index, value });
        }
    }
    let length = u64::try_from(values.len()).map_err(|_| SchedulerError::StepOverflow)?;
    let descriptor = sim_tensor::TensorDescriptor::contiguous(
        vec![length],
        sim_tensor::DType::F32,
        sim_tensor::DeviceId::CPU,
        context.stream,
    )?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

pub(crate) fn build_parameterized_sigmas(
    backend: &CpuBackend,
    steps: usize,
    context: &ExecutionContext<'_>,
    equation: impl FnOnce(&mut CpuWorkspaceVec<f32>) -> Result<(), SchedulerError>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    context.check()?;
    let capacity = steps.checked_add(1).ok_or(SchedulerError::StepOverflow)?;
    let mut values = backend.workspace_vec(context, capacity)?;
    equation(&mut values)?;
    publish_parameterized_sigmas(backend, &values, context)
}

#[derive(Debug, thiserror::Error)]
pub enum VpScheduleError {
    #[error("VP schedule parameters exceed the source node bounds")]
    InvalidParameters,
    #[error(transparent)]
    Tensor(#[from] sim_tensor::TensorError),
    #[error(transparent)]
    Creation(#[from] sim_tensor::generated_tensor_creation_01::TensorCreationPartOneError),
    #[error(transparent)]
    Arithmetic(#[from] sim_tensor::generated_elementwise_or_runtime_operation_03::ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    Division(#[from] sim_tensor::generated_elementwise_or_runtime_operation_06::ElementwiseRuntimePartSixError),
    #[error(transparent)]
    Nonlinear(#[from] sim_tensor::generated_elementwise_or_runtime_operation_08::ElementwiseRuntimePartEightError),
    #[error(transparent)]
    Concatenation(#[from] sim_tensor::generated_shape_layout_transform_02::ShapeLayoutTransformPartTwoError),
}

pub fn vp_sigmas(
    backend: &sim_tensor::CpuBackend,
    steps: u64,
    beta_d: f64,
    beta_min: f64,
    eps_s: f64,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, VpScheduleError> {
    use sim_tensor::generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, real_add_with_context_exact_native,
        real_multiply_with_context_exact_native,
    };
    use sim_tensor::generated_elementwise_or_runtime_operation_06::{
        div_function_with_context_exact_native, sqrt_function_with_context_exact_native,
    };
    use sim_tensor::generated_elementwise_or_runtime_operation_08::{
        expm1_function_with_context_exact_native, square_with_context_exact_native,
    };
    use sim_tensor::generated_shape_layout_transform_02::torch_cat_with_context_exact_native;
    use sim_tensor::generated_tensor_creation_01::{
        linspace_with_context_exact_native, zeros_with_context_exact_native,
    };
    use sim_tensor::{DType, DeviceId, Layout, Scalar};
    context.check()?;
    if !(1..=10000).contains(&steps)
        || !beta_d.is_finite()
        || !(0.0..=5000.0).contains(&beta_d)
        || !beta_min.is_finite()
        || !(0.0..=5000.0).contains(&beta_min)
        || !eps_s.is_finite()
        || !(0.0..=1.0).contains(&eps_s)
    {
        return Err(VpScheduleError::InvalidParameters);
    }
    let time = linspace_with_context_exact_native(
        backend,
        1.0,
        eps_s,
        steps,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        context,
    )?;
    let squared = square_with_context_exact_native(backend, &time, context)?;
    let quadratic = real_multiply_with_context_exact_native(
        backend,
        &squared,
        ElementwiseOperand::Scalar(Scalar::Float(beta_d)),
        context,
    )?;
    let quadratic = div_function_with_context_exact_native(
        backend,
        &quadratic,
        ElementwiseOperand::Scalar(Scalar::Signed(2)),
        None,
        context,
    )?;
    let linear = real_multiply_with_context_exact_native(
        backend,
        &time,
        ElementwiseOperand::Scalar(Scalar::Float(beta_min)),
        context,
    )?;
    let exponent = real_add_with_context_exact_native(backend, &quadratic, &linear, context)?;
    let variance = expm1_function_with_context_exact_native(backend, &exponent, context)?;
    let sigmas = sqrt_function_with_context_exact_native(backend, &variance, context)?;
    let zero = zeros_with_context_exact_native(
        backend,
        &[1],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        context,
    )?;
    Ok(torch_cat_with_context_exact_native(
        backend,
        &[sigmas, zero],
        0,
        context,
    )?)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SigmaInterpolationSpacing {
    Linear,
    Cosine,
    Sine,
}

pub fn flip_sigmas(
    backend: &CpuBackend,
    sigmas: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    context.check()?;
    let mut values = backend.workspace_vec(context, sigmas.len())?;
    for (index, value) in sigmas.iter().rev().copied().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        values.try_push(if index == 0 && value == 0.0 {
            0.0001
        } else {
            value
        })?;
    }
    publish_parameterized_sigmas(backend, &values, context)
}

pub fn extend_intermediate_sigmas(
    backend: &CpuBackend,
    sigmas: &[f32],
    steps: u64,
    start: f64,
    end: f64,
    spacing: SigmaInterpolationSpacing,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    use sim_tensor::generated_tensor_creation_01::linspace_f32_value;
    context.check()?;
    if !(1..=100).contains(&steps)
        || !start.is_finite()
        || !(-1.0..=20000.0).contains(&start)
        || !end.is_finite()
        || !(0.0..=20000.0).contains(&end)
    {
        return Err(SchedulerError::InvalidParameters(
            "intermediate sigma parameters exceed source bounds",
        ));
    }
    let steps = usize::try_from(steps).map_err(|_| SchedulerError::StepOverflow)?;
    let start = if start < 0.0 {
        f32::INFINITY
    } else {
        start as f32
    };
    let end = end as f32;
    let mut capacity = sigmas.len();
    for (index, pair) in sigmas.windows(2).enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        if pair
            .first()
            .is_some_and(|value| *value >= end && *value <= start)
        {
            capacity = capacity
                .checked_add(steps - 1)
                .ok_or(SchedulerError::StepOverflow)?;
        }
    }
    let mut values = backend.workspace_vec(context, capacity)?;
    for (index, pair) in sigmas.windows(2).enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        let [current, next] = pair else {
            return Err(SchedulerError::InvalidParameters(
                "sigma interval is incomplete",
            ));
        };
        values.try_push(*current)?;
        if *current >= end && *current <= start {
            for position in 1..steps {
                let fraction = linspace_f32_value(0.0, 1.0, steps + 1, position)?;
                let fraction = match spacing {
                    SigmaInterpolationSpacing::Linear => fraction,
                    SigmaInterpolationSpacing::Cosine => {
                        (fraction * std::f32::consts::PI / 2.0).sin()
                    }
                    SigmaInterpolationSpacing::Sine => {
                        1.0 - (fraction * std::f32::consts::PI / 2.0).cos()
                    }
                };
                values.try_push(fraction * (next - current) + current)?;
            }
        }
    }
    if let Some(last) = sigmas.last() {
        values.try_push(*last)?;
    }
    publish_parameterized_sigmas(backend, &values, context)
}

pub fn flux2_sigmas(
    backend: &CpuBackend,
    steps: u64,
    width: u64,
    height: u64,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    use sim_tensor::generated_tensor_creation_01::linspace_f32_value;
    context.check()?;
    if !(1..=4096).contains(&steps)
        || !(16..=16384).contains(&width)
        || !(16..=16384).contains(&height)
    {
        return Err(SchedulerError::InvalidParameters(
            "Flux2 schedule dimensions or steps exceed source bounds",
        ));
    }
    let steps = usize::try_from(steps).map_err(|_| SchedulerError::StepOverflow)?;
    let sequence_length = ((width * height) as f64 / 256.0).round_ties_even();
    let at_ten_steps = 8.73809524e-05 * sequence_length + 1.89833333;
    let at_two_hundred_steps = 0.00016927 * sequence_length + 0.45666666;
    let mean = if sequence_length > 4300.0 {
        at_two_hundred_steps
    } else {
        let slope = (at_two_hundred_steps - at_ten_steps) / 190.0;
        slope * steps as f64 + (at_two_hundred_steps - 200.0 * slope)
    };
    let exponential = mean.exp() as f32;
    build_parameterized_sigmas(backend, steps, context, |values| {
        for index in 0..=steps {
            if index.is_multiple_of(256) {
                context.check()?;
            }
            let time = linspace_f32_value(1.0, 0.0, steps + 1, index)?;
            values.try_push(exponential / (exponential + (1.0 / time - 1.0)))?;
        }
        Ok(())
    })
}

pub fn ideogram4_sigmas(
    backend: &CpuBackend,
    steps: u64,
    width: u64,
    height: u64,
    mean: f64,
    standard_deviation: f64,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    use sim_tensor::generated_elementwise_or_runtime_operation_09::inverse_normal_cdf_f64;
    use sim_tensor::generated_tensor_creation_01::linspace_f64_value;
    context.check()?;
    if !(1..=200).contains(&steps)
        || !(256..=8192).contains(&width)
        || !(256..=8192).contains(&height)
        || !mean.is_finite()
        || !(-10.0..=10.0).contains(&mean)
        || !standard_deviation.is_finite()
        || !(0.1..=5.0).contains(&standard_deviation)
    {
        return Err(SchedulerError::InvalidParameters(
            "Ideogram4 schedule parameters exceed source bounds",
        ));
    }
    let steps = usize::try_from(steps).map_err(|_| SchedulerError::StepOverflow)?;
    let mean = mean + 0.5 * ((width * height) as f64 / (512.0 * 512.0)).ln();
    let minimum_time = 1.0 / (1.0 + 9.0_f64.exp());
    let maximum_time = 1.0 / (1.0 + (-7.5_f64).exp());
    build_parameterized_sigmas(backend, steps, context, |values| {
        for index in 0..steps {
            context.check()?;
            let probability = linspace_f64_value(0.0, 1.0, steps + 1, steps - index)?;
            let quantile = inverse_normal_cdf_f64(probability);
            let time = 1.0 - 1.0 / (1.0 + (-mean - standard_deviation * quantile).exp());
            values.try_push((1.0 - time.clamp(minimum_time, maximum_time)) as f32)?;
        }
        values.try_push(0.0)?;
        Ok(())
    })
}

pub fn gits_sigmas(
    backend: &CpuBackend,
    coefficient: f64,
    steps: u64,
    denoise: f64,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, SchedulerError> {
    context.check()?;
    if !(2..=1000).contains(&steps)
        || !coefficient.is_finite()
        || !(0.8..=1.5).contains(&coefficient)
        || !denoise.is_finite()
        || !(0.0..=1.0).contains(&denoise)
    {
        return Err(SchedulerError::InvalidParameters(
            "GITS schedule parameters exceed source bounds",
        ));
    }
    if denoise == 0.0 {
        return publish_parameterized_sigmas(backend, &[], context);
    }
    let steps = usize::try_from(steps).map_err(|_| SchedulerError::StepOverflow)?;
    // Decimal formatting rounds the original binary value like Python round(x, 2);
    // multiplying by 100 before rounding changes coefficient-table selection.
    let coefficient = format!("{coefficient:.2}")
        .parse::<f64>()
        .map_err(|_| SchedulerError::InvalidParameters("GITS coefficient cannot be rounded"))?;
    let levels = GITS_NOISE_LEVELS
        .iter()
        .find(|(key, _)| *key == coefficient)
        .map(|(_, levels)| *levels)
        .ok_or(SchedulerError::InvalidParameters(
            "GITS coefficient has no source schedule",
        ))?;
    let total_steps = if denoise < 1.0 {
        (steps as f64 * denoise).round_ties_even() as usize
    } else {
        steps
    };
    let source = if steps <= 20 {
        levels.get(steps - 2)
    } else {
        levels.last()
    }
    .copied()
    .ok_or(SchedulerError::InvalidParameters(
        "GITS source schedule is absent",
    ))?;
    build_parameterized_sigmas(backend, total_steps, context, |values| {
        for index in (steps - total_steps)..=steps {
            if index.is_multiple_of(256) {
                context.check()?;
            }
            let value = if index == steps {
                0.0
            } else if steps <= 20 {
                *source.get(index).ok_or(SchedulerError::InvalidParameters(
                    "GITS source step is absent",
                ))?
            } else {
                let maximum_index = source
                    .len()
                    .checked_sub(1)
                    .ok_or(SchedulerError::EmptySchedule)?;
                let fraction = (steps - index) as f64 / steps as f64;
                let position = fraction * maximum_index as f64;
                let left = (position.floor() as usize).min(maximum_index);
                let right = left.saturating_add(1).min(maximum_index);
                let left_value = source
                    .get(maximum_index - left)
                    .ok_or(SchedulerError::InvalidParameters(
                        "GITS interpolation bound is absent",
                    ))?
                    .ln();
                let right_value = source
                    .get(maximum_index - right)
                    .ok_or(SchedulerError::InvalidParameters(
                        "GITS interpolation bound is absent",
                    ))?
                    .ln();
                (left_value + (right_value - left_value) * (position - left as f64)).exp()
            };
            values.try_push(value as f32)?;
        }
        Ok(())
    })
}

#[rustfmt::skip]
const GITS_NOISE_LEVELS: &[(f64, &[&[f64]])] = &[
    (0.8, &[
        &[14.61464119, 7.49001646, 0.02916753],
        &[14.61464119, 11.54541874, 6.77309084, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 3.07277966, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 2.05039096, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 2.05039096, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 5.85520077, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.07277966, 1.56271636, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.1956799, 1.98035145, 0.86115354, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.1956799, 1.98035145, 0.86115354, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.07277966, 1.84880662, 0.83188516, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547, 9.24142551, 8.75849152, 8.30717278, 7.88507891, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.07277966, 1.84880662, 0.83188516, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547, 9.24142551, 8.75849152, 8.30717278, 7.88507891, 7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.75677586, 2.84484982, 1.78698075, 0.803307, 0.02916753],
    ]),
    (0.85, &[
        &[14.61464119, 7.49001646, 0.02916753],
        &[14.61464119, 7.49001646, 1.84880662, 0.02916753],
        &[14.61464119, 11.54541874, 6.77309084, 1.56271636, 0.02916753],
        &[14.61464119, 11.54541874, 7.11996698, 3.07277966, 1.24153244, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.09240818, 2.84484982, 0.95350921, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.09240818, 2.84484982, 0.95350921, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.58536053, 3.1956799, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 8.75849152, 7.49001646, 5.58536053, 3.1956799, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 8.75849152, 7.49001646, 6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 8.75849152, 7.49001646, 6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.65472794, 3.07277966, 1.84880662, 0.803307, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.60512662, 2.6383388, 1.56271636, 0.72133851, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.90732002, 10.31284904, 9.75859547, 9.24142551, 8.75849152, 8.30717278, 7.88507891, 7.49001646, 6.77309084, 5.85520077, 4.65472794, 3.46139455, 2.45070267, 1.56271636, 0.72133851, 0.02916753],
    ]),
    (0.9, &[
        &[14.61464119, 6.77309084, 0.02916753],
        &[14.61464119, 7.49001646, 1.56271636, 0.02916753],
        &[14.61464119, 7.49001646, 3.07277966, 0.95350921, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.54230714, 0.89115214, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 4.86714602, 2.54230714, 0.89115214, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.09240818, 3.07277966, 1.61558151, 0.69515091, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.11996698, 4.86714602, 3.07277966, 1.61558151, 0.69515091, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 2.95596409, 1.61558151, 0.69515091, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.19988537, 1.24153244, 0.57119018, 0.02916753],
        &[14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.19988537, 1.24153244, 0.57119018, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.19988537, 1.24153244, 0.57119018, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.75677586, 2.84484982, 1.84880662, 1.08895338, 0.52423614, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 4.86714602, 3.75677586, 2.84484982, 1.84880662, 1.08895338, 0.52423614, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.44769001, 5.58536053, 4.45427561, 3.32507086, 2.45070267, 1.61558151, 0.95350921, 0.45573691, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.44769001, 5.58536053, 4.45427561, 3.32507086, 2.45070267, 1.61558151, 0.95350921, 0.45573691, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.91689563, 3.07277966, 2.27973175, 1.56271636, 0.95350921, 0.45573691, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.91689563, 3.07277966, 2.27973175, 1.56271636, 0.95350921, 0.45573691, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 4.86714602, 3.91689563, 3.07277966, 2.27973175, 1.56271636, 0.95350921, 0.45573691, 0.02916753],
        &[14.61464119, 13.76078796, 12.96784878, 12.2308979, 11.54541874, 10.31284904, 9.24142551, 8.75849152, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818, 4.45427561, 3.60512662, 2.95596409, 2.19988537, 1.51179266, 0.89115214, 0.43325692, 0.02916753],
    ]),
    (0.95, &[
        &[14.61464119, 6.77309084, 0.02916753],
        &[14.61464119, 6.77309084, 1.56271636, 0.02916753],
        &[14.61464119, 7.49001646, 2.84484982, 0.89115214, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.36326075, 0.803307, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.95596409, 1.56271636, 0.64427125, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 4.86714602, 2.95596409, 1.56271636, 0.64427125, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 4.86714602, 3.07277966, 1.91321158, 1.08895338, 0.50118381, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.07277966, 1.91321158, 1.08895338, 0.50118381, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.07277966, 1.91321158, 1.08895338, 0.50118381, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.19988537, 1.41535246, 0.803307, 0.38853383, 0.02916753],
        &[14.61464119, 12.2308979, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.46139455, 2.6383388, 1.84880662, 1.24153244, 0.72133851, 0.34370604, 0.02916753],
        &[14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.46139455, 2.6383388, 1.84880662, 1.24153244, 0.72133851, 0.34370604, 0.02916753],
        &[14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 6.14220476, 4.86714602, 3.75677586, 2.95596409, 2.19988537, 1.56271636, 1.05362725, 0.64427125, 0.32104823, 0.02916753],
        &[14.61464119, 12.96784878, 10.90732002, 8.75849152, 7.49001646, 6.44769001, 5.58536053, 4.65472794, 3.60512662, 2.95596409, 2.19988537, 1.56271636, 1.05362725, 0.64427125, 0.32104823, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.44769001, 5.58536053, 4.65472794, 3.60512662, 2.95596409, 2.19988537, 1.56271636, 1.05362725, 0.64427125, 0.32104823, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.44769001, 5.58536053, 4.65472794, 3.75677586, 3.07277966, 2.45070267, 1.78698075, 1.24153244, 0.83188516, 0.50118381, 0.22545385, 0.02916753],
        &[14.61464119, 12.96784878, 11.54541874, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818, 4.45427561, 3.60512662, 2.95596409, 2.36326075, 1.72759056, 1.24153244, 0.83188516, 0.50118381, 0.22545385, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818, 4.45427561, 3.60512662, 2.95596409, 2.36326075, 1.72759056, 1.24153244, 0.83188516, 0.50118381, 0.22545385, 0.02916753],
        &[14.61464119, 13.76078796, 12.2308979, 10.90732002, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818, 4.45427561, 3.75677586, 3.07277966, 2.45070267, 1.91321158, 1.46270394, 1.05362725, 0.72133851, 0.43325692, 0.19894916, 0.02916753],
    ]),
    (1.0, &[
        &[14.61464119, 1.56271636, 0.02916753],
        &[14.61464119, 6.77309084, 0.95350921, 0.02916753],
        &[14.61464119, 6.77309084, 2.36326075, 0.803307, 0.02916753],
        &[14.61464119, 7.11996698, 3.07277966, 1.56271636, 0.59516323, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.41535246, 0.57119018, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.86115354, 0.38853383, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.86115354, 0.38853383, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 4.86714602, 3.07277966, 1.98035145, 1.24153244, 0.72133851, 0.34370604, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.07277966, 1.98035145, 1.24153244, 0.72133851, 0.34370604, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.27973175, 1.51179266, 0.95350921, 0.54755926, 0.25053367, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.36326075, 1.61558151, 1.08895338, 0.72133851, 0.41087446, 0.17026083, 0.02916753],
        &[14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.36326075, 1.61558151, 1.08895338, 0.72133851, 0.41087446, 0.17026083, 0.02916753],
        &[14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982, 2.12350607, 1.56271636, 1.08895338, 0.72133851, 0.41087446, 0.17026083, 0.02916753],
        &[14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982, 2.19988537, 1.61558151, 1.162866, 0.803307, 0.50118381, 0.27464288, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 8.75849152, 7.49001646, 5.85520077, 4.65472794, 3.75677586, 3.07277966, 2.45070267, 1.84880662, 1.36964464, 1.01931262, 0.72133851, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 8.75849152, 7.49001646, 6.14220476, 5.09240818, 4.26497746, 3.46139455, 2.84484982, 2.19988537, 1.67050016, 1.24153244, 0.92192322, 0.64427125, 0.43325692, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 8.75849152, 7.49001646, 6.14220476, 5.09240818, 4.26497746, 3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.12534678, 0.83188516, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 12.2308979, 9.24142551, 8.30717278, 7.49001646, 6.14220476, 5.09240818, 4.26497746, 3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.12534678, 0.83188516, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 12.2308979, 9.24142551, 8.30717278, 7.49001646, 6.77309084, 5.85520077, 5.09240818, 4.26497746, 3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.12534678, 0.83188516, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
    ]),
    (1.05, &[
        &[14.61464119, 0.95350921, 0.02916753],
        &[14.61464119, 6.77309084, 0.89115214, 0.02916753],
        &[14.61464119, 6.77309084, 2.05039096, 0.72133851, 0.02916753],
        &[14.61464119, 6.77309084, 2.84484982, 1.28281462, 0.52423614, 0.02916753],
        &[14.61464119, 6.77309084, 3.07277966, 1.61558151, 0.803307, 0.34370604, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.56271636, 0.803307, 0.34370604, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.95350921, 0.52423614, 0.22545385, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.07277966, 1.98035145, 1.24153244, 0.74807048, 0.41087446, 0.17026083, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.27973175, 1.51179266, 0.95350921, 0.59516323, 0.34370604, 0.13792117, 0.02916753],
        &[14.61464119, 7.49001646, 5.09240818, 3.46139455, 2.45070267, 1.61558151, 1.08895338, 0.72133851, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.09240818, 3.46139455, 2.45070267, 1.61558151, 1.08895338, 0.72133851, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.36326075, 1.61558151, 1.08895338, 0.72133851, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.45070267, 1.72759056, 1.24153244, 0.86115354, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982, 2.19988537, 1.61558151, 1.162866, 0.83188516, 0.59516323, 0.38853383, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.84484982, 2.19988537, 1.67050016, 1.28281462, 0.95350921, 0.72133851, 0.52423614, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.95596409, 2.36326075, 1.84880662, 1.41535246, 1.08895338, 0.83188516, 0.61951244, 0.45573691, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.65472794, 3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.20157266, 0.95350921, 0.74807048, 0.57119018, 0.43325692, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 8.30717278, 7.11996698, 5.85520077, 4.65472794, 3.60512662, 2.95596409, 2.45070267, 1.91321158, 1.51179266, 1.20157266, 0.95350921, 0.74807048, 0.57119018, 0.43325692, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 8.30717278, 7.11996698, 5.85520077, 4.65472794, 3.60512662, 2.95596409, 2.45070267, 1.98035145, 1.61558151, 1.32549286, 1.08895338, 0.86115354, 0.69515091, 0.54755926, 0.41087446, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
    ]),
    (1.1, &[
        &[14.61464119, 0.89115214, 0.02916753],
        &[14.61464119, 2.36326075, 0.72133851, 0.02916753],
        &[14.61464119, 5.85520077, 1.61558151, 0.57119018, 0.02916753],
        &[14.61464119, 6.77309084, 2.45070267, 1.08895338, 0.45573691, 0.02916753],
        &[14.61464119, 6.77309084, 2.95596409, 1.56271636, 0.803307, 0.34370604, 0.02916753],
        &[14.61464119, 6.77309084, 3.07277966, 1.61558151, 0.89115214, 0.4783645, 0.19894916, 0.02916753],
        &[14.61464119, 6.77309084, 3.07277966, 1.84880662, 1.08895338, 0.64427125, 0.34370604, 0.13792117, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.84484982, 1.61558151, 0.95350921, 0.54755926, 0.27464288, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.95596409, 1.91321158, 1.24153244, 0.803307, 0.4783645, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.05039096, 1.41535246, 0.95350921, 0.64427125, 0.41087446, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.27973175, 1.61558151, 1.12534678, 0.803307, 0.54755926, 0.36617002, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.32507086, 2.45070267, 1.72759056, 1.24153244, 0.89115214, 0.64427125, 0.45573691, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 5.09240818, 3.60512662, 2.84484982, 2.05039096, 1.51179266, 1.08895338, 0.803307, 0.59516323, 0.43325692, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 5.09240818, 3.60512662, 2.84484982, 2.12350607, 1.61558151, 1.24153244, 0.95350921, 0.72133851, 0.54755926, 0.41087446, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.08895338, 0.83188516, 0.64427125, 0.50118381, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 5.85520077, 4.45427561, 3.1956799, 2.45070267, 1.91321158, 1.51179266, 1.20157266, 0.95350921, 0.74807048, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 5.85520077, 4.45427561, 3.46139455, 2.84484982, 2.19988537, 1.72759056, 1.36964464, 1.08895338, 0.86115354, 0.69515091, 0.54755926, 0.43325692, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.46139455, 2.84484982, 2.19988537, 1.72759056, 1.36964464, 1.08895338, 0.86115354, 0.69515091, 0.54755926, 0.43325692, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 11.54541874, 7.49001646, 5.85520077, 4.45427561, 3.46139455, 2.84484982, 2.19988537, 1.72759056, 1.36964464, 1.08895338, 0.89115214, 0.72133851, 0.59516323, 0.4783645, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
    ]),
    (1.15, &[
        &[14.61464119, 0.83188516, 0.02916753],
        &[14.61464119, 1.84880662, 0.59516323, 0.02916753],
        &[14.61464119, 5.85520077, 1.56271636, 0.52423614, 0.02916753],
        &[14.61464119, 5.85520077, 1.91321158, 0.83188516, 0.34370604, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.24153244, 0.59516323, 0.25053367, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.51179266, 0.803307, 0.41087446, 0.17026083, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.56271636, 0.89115214, 0.50118381, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 6.77309084, 3.07277966, 1.84880662, 1.12534678, 0.72133851, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 6.77309084, 3.07277966, 1.91321158, 1.24153244, 0.803307, 0.52423614, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 2.95596409, 1.91321158, 1.24153244, 0.803307, 0.52423614, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.05039096, 1.36964464, 0.95350921, 0.69515091, 0.4783645, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.803307, 0.59516323, 0.43325692, 0.29807833, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.803307, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921, 0.74807048, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.78698075, 1.32549286, 1.01931262, 0.803307, 0.64427125, 0.50118381, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.78698075, 1.32549286, 1.01931262, 0.803307, 0.64427125, 0.52423614, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.12534678, 0.89115214, 0.72133851, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.12534678, 0.89115214, 0.72133851, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.86714602, 3.1956799, 2.45070267, 1.84880662, 1.41535246, 1.12534678, 0.89115214, 0.72133851, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.2, &[
        &[14.61464119, 0.803307, 0.02916753],
        &[14.61464119, 1.56271636, 0.52423614, 0.02916753],
        &[14.61464119, 2.36326075, 0.92192322, 0.36617002, 0.02916753],
        &[14.61464119, 2.84484982, 1.24153244, 0.59516323, 0.25053367, 0.02916753],
        &[14.61464119, 5.85520077, 2.05039096, 0.95350921, 0.45573691, 0.17026083, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.24153244, 0.64427125, 0.29807833, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.36964464, 0.803307, 0.45573691, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.61558151, 0.95350921, 0.59516323, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.67050016, 1.08895338, 0.74807048, 0.50118381, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.95596409, 1.84880662, 1.24153244, 0.83188516, 0.59516323, 0.41087446, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 3.07277966, 1.98035145, 1.36964464, 0.95350921, 0.69515091, 0.50118381, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 6.77309084, 3.46139455, 2.36326075, 1.56271636, 1.08895338, 0.803307, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 6.77309084, 3.46139455, 2.45070267, 1.61558151, 1.162866, 0.86115354, 0.64427125, 0.50118381, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516, 0.64427125, 0.50118381, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516, 0.64427125, 0.50118381, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.12350607, 1.51179266, 1.08895338, 0.83188516, 0.64427125, 0.50118381, 0.41087446, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.20157266, 0.92192322, 0.72133851, 0.57119018, 0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921, 0.74807048, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 7.49001646, 4.65472794, 3.07277966, 2.19988537, 1.61558151, 1.24153244, 0.95350921, 0.74807048, 0.59516323, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.25, &[
        &[14.61464119, 0.72133851, 0.02916753],
        &[14.61464119, 1.56271636, 0.50118381, 0.02916753],
        &[14.61464119, 2.05039096, 0.803307, 0.32104823, 0.02916753],
        &[14.61464119, 2.36326075, 0.95350921, 0.43325692, 0.17026083, 0.02916753],
        &[14.61464119, 2.84484982, 1.24153244, 0.59516323, 0.27464288, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.51179266, 0.803307, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.36326075, 1.24153244, 0.72133851, 0.41087446, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.36964464, 0.83188516, 0.52423614, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.61558151, 0.98595673, 0.64427125, 0.43325692, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.67050016, 1.08895338, 0.74807048, 0.52423614, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.72759056, 1.162866, 0.803307, 0.59516323, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.95596409, 1.84880662, 1.24153244, 0.86115354, 0.64427125, 0.4783645, 0.36617002, 0.27464288, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.95596409, 1.84880662, 1.28281462, 0.92192322, 0.69515091, 0.52423614, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.95596409, 1.91321158, 1.32549286, 0.95350921, 0.72133851, 0.54755926, 0.43325692, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.95596409, 1.91321158, 1.32549286, 0.95350921, 0.72133851, 0.57119018, 0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.95596409, 1.91321158, 1.32549286, 0.95350921, 0.74807048, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 3.07277966, 2.05039096, 1.41535246, 1.05362725, 0.803307, 0.61951244, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 3.07277966, 2.05039096, 1.41535246, 1.05362725, 0.803307, 0.64427125, 0.52423614, 0.43325692, 0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 3.07277966, 2.05039096, 1.46270394, 1.08895338, 0.83188516, 0.66947293, 0.54755926, 0.45573691, 0.38853383, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.3, &[
        &[14.61464119, 0.72133851, 0.02916753],
        &[14.61464119, 1.24153244, 0.43325692, 0.02916753],
        &[14.61464119, 1.56271636, 0.59516323, 0.22545385, 0.02916753],
        &[14.61464119, 1.84880662, 0.803307, 0.36617002, 0.13792117, 0.02916753],
        &[14.61464119, 2.36326075, 1.01931262, 0.52423614, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.36964464, 0.74807048, 0.41087446, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.56271636, 0.89115214, 0.54755926, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.61558151, 0.95350921, 0.61951244, 0.41087446, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.36964464, 0.83188516, 0.54755926, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.41535246, 0.92192322, 0.64427125, 0.45573691, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.6383388, 1.56271636, 1.01931262, 0.72133851, 0.50118381, 0.36617002, 0.27464288, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.05362725, 0.74807048, 0.54755926, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.08895338, 0.77538133, 0.57119018, 0.43325692, 0.34370604, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.59516323, 0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.72759056, 1.162866, 0.83188516, 0.64427125, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.72759056, 1.162866, 0.83188516, 0.64427125, 0.52423614, 0.43325692, 0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.78698075, 1.24153244, 0.92192322, 0.72133851, 0.57119018, 0.45573691, 0.38853383, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.84484982, 1.78698075, 1.24153244, 0.92192322, 0.72133851, 0.57119018, 0.4783645, 0.41087446, 0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.35, &[
        &[14.61464119, 0.69515091, 0.02916753],
        &[14.61464119, 0.95350921, 0.34370604, 0.02916753],
        &[14.61464119, 1.56271636, 0.57119018, 0.19894916, 0.02916753],
        &[14.61464119, 1.61558151, 0.69515091, 0.29807833, 0.09824532, 0.02916753],
        &[14.61464119, 1.84880662, 0.83188516, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.162866, 0.64427125, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.36964464, 0.803307, 0.50118381, 0.32104823, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.41535246, 0.83188516, 0.54755926, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 0.95350921, 0.64427125, 0.45573691, 0.32104823, 0.22545385, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 0.95350921, 0.64427125, 0.45573691, 0.34370604, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.61558151, 1.01931262, 0.72133851, 0.52423614, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.61558151, 1.01931262, 0.72133851, 0.52423614, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.61558151, 1.05362725, 0.74807048, 0.54755926, 0.43325692, 0.34370604, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.72759056, 1.12534678, 0.803307, 0.59516323, 0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 3.07277966, 1.72759056, 1.12534678, 0.803307, 0.59516323, 0.4783645, 0.38853383, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.45070267, 1.51179266, 1.01931262, 0.74807048, 0.57119018, 0.45573691, 0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.6383388, 1.61558151, 1.08895338, 0.803307, 0.61951244, 0.50118381, 0.41087446, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.6383388, 1.61558151, 1.08895338, 0.803307, 0.64427125, 0.52423614, 0.43325692, 0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 5.85520077, 2.6383388, 1.61558151, 1.08895338, 0.803307, 0.64427125, 0.52423614, 0.45573691, 0.38853383, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.4, &[
        &[14.61464119, 0.59516323, 0.02916753],
        &[14.61464119, 0.95350921, 0.34370604, 0.02916753],
        &[14.61464119, 1.08895338, 0.43325692, 0.13792117, 0.02916753],
        &[14.61464119, 1.56271636, 0.64427125, 0.27464288, 0.09824532, 0.02916753],
        &[14.61464119, 1.61558151, 0.803307, 0.43325692, 0.22545385, 0.09824532, 0.02916753],
        &[14.61464119, 2.05039096, 0.95350921, 0.54755926, 0.34370604, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.24153244, 0.72133851, 0.43325692, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.24153244, 0.74807048, 0.50118381, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.28281462, 0.803307, 0.52423614, 0.36617002, 0.27464288, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.28281462, 0.803307, 0.54755926, 0.38853383, 0.29807833, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.41535246, 0.86115354, 0.59516323, 0.43325692, 0.32104823, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.64427125, 0.45573691, 0.34370604, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.64427125, 0.4783645, 0.36617002, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 0.98595673, 0.69515091, 0.52423614, 0.41087446, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 1.01931262, 0.72133851, 0.54755926, 0.43325692, 0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.61558151, 1.05362725, 0.74807048, 0.57119018, 0.45573691, 0.38853383, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.61951244, 0.50118381, 0.41087446, 0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.61951244, 0.50118381, 0.43325692, 0.38853383, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.61558151, 1.08895338, 0.803307, 0.64427125, 0.52423614, 0.45573691, 0.41087446, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.45, &[
        &[14.61464119, 0.59516323, 0.02916753],
        &[14.61464119, 0.803307, 0.25053367, 0.02916753],
        &[14.61464119, 0.95350921, 0.34370604, 0.09824532, 0.02916753],
        &[14.61464119, 1.24153244, 0.54755926, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 1.56271636, 0.72133851, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 1.61558151, 0.803307, 0.45573691, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 1.91321158, 0.95350921, 0.57119018, 0.36617002, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 2.19988537, 1.08895338, 0.64427125, 0.41087446, 0.27464288, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.24153244, 0.74807048, 0.50118381, 0.34370604, 0.25053367, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.24153244, 0.74807048, 0.50118381, 0.36617002, 0.27464288, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.28281462, 0.803307, 0.54755926, 0.41087446, 0.32104823, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.28281462, 0.803307, 0.57119018, 0.43325692, 0.34370604, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.28281462, 0.83188516, 0.59516323, 0.45573691, 0.36617002, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.28281462, 0.83188516, 0.59516323, 0.45573691, 0.36617002, 0.32104823, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.69515091, 0.52423614, 0.41087446, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.51179266, 0.95350921, 0.69515091, 0.52423614, 0.43325692, 0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 0.98595673, 0.72133851, 0.54755926, 0.45573691, 0.38853383, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 1.01931262, 0.74807048, 0.57119018, 0.4783645, 0.41087446, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.84484982, 1.56271636, 1.01931262, 0.74807048, 0.59516323, 0.50118381, 0.43325692, 0.38853383, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
    (1.5, &[
        &[14.61464119, 0.54755926, 0.02916753],
        &[14.61464119, 0.803307, 0.25053367, 0.02916753],
        &[14.61464119, 0.86115354, 0.32104823, 0.09824532, 0.02916753],
        &[14.61464119, 1.24153244, 0.54755926, 0.25053367, 0.09824532, 0.02916753],
        &[14.61464119, 1.56271636, 0.72133851, 0.36617002, 0.19894916, 0.09824532, 0.02916753],
        &[14.61464119, 1.61558151, 0.803307, 0.45573691, 0.27464288, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 1.61558151, 0.83188516, 0.52423614, 0.34370604, 0.25053367, 0.17026083, 0.09824532, 0.02916753],
        &[14.61464119, 1.84880662, 0.95350921, 0.59516323, 0.38853383, 0.27464288, 0.19894916, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 1.84880662, 0.95350921, 0.59516323, 0.41087446, 0.29807833, 0.22545385, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 1.84880662, 0.95350921, 0.61951244, 0.43325692, 0.32104823, 0.25053367, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.19988537, 1.12534678, 0.72133851, 0.50118381, 0.36617002, 0.27464288, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.19988537, 1.12534678, 0.72133851, 0.50118381, 0.36617002, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.36326075, 1.24153244, 0.803307, 0.57119018, 0.43325692, 0.34370604, 0.29807833, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.36326075, 1.24153244, 0.803307, 0.57119018, 0.43325692, 0.34370604, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.36326075, 1.24153244, 0.803307, 0.59516323, 0.45573691, 0.36617002, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.36326075, 1.24153244, 0.803307, 0.59516323, 0.45573691, 0.38853383, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.32549286, 0.86115354, 0.64427125, 0.50118381, 0.41087446, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.36964464, 0.92192322, 0.69515091, 0.54755926, 0.45573691, 0.41087446, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
        &[14.61464119, 2.45070267, 1.41535246, 0.95350921, 0.72133851, 0.57119018, 0.4783645, 0.43325692, 0.38853383, 0.36617002, 0.34370604, 0.32104823, 0.29807833, 0.27464288, 0.25053367, 0.22545385, 0.19894916, 0.17026083, 0.13792117, 0.09824532, 0.02916753],
    ]),
];
