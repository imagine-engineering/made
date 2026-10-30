use crate::{
    GENERATED_SAMPLER_DEFINITIONS, SamplingProfileIdentity, SchedulerIdentity, SchedulerRegistry,
};
use serde::{Deserialize, Deserializer, Serialize};
use sim_tensor::{CancellationToken, Tensor, TensorError};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const SAMPLING_PLAN_SCHEMA_VERSION: u16 = 1;
pub const EULER_SAMPLER_ID: &str = "euler";
pub const EULER_SAMPLER_FEATURE_ID: &str = "COMFY-MODEL-0179";
pub const MAX_ADAPTIVE_SAMPLING_ATTEMPTS: u32 = 1_000_000;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct SamplerIdentity(String);

impl SamplerIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, SamplingError> {
        let value = value.into();
        validate_identifier("sampler", &value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SamplerIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplerDefinition {
    pub identity: &'static str,
    pub feature_id: &'static str,
    pub source_ordinal: u16,
    pub aliases: &'static [&'static str],
    pub implementation_module: &'static str,
    pub stochastic: bool,
}

#[derive(Clone, Debug)]
pub struct CfgPpDenoiserOutput {
    pub denoised: Tensor,
    pub unconditional_denoised: Tensor,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
#[error("CFG++ {output} descriptor does not match the denoiser input")]
pub struct CfgPpDenoiserContractError {
    pub output: &'static str,
}

pub fn validate_cfg_pp_denoiser_output(
    input: &Tensor,
    output: &CfgPpDenoiserOutput,
) -> Result<(), CfgPpDenoiserContractError> {
    for (name, tensor) in [
        ("guided denoiser output", &output.denoised),
        (
            "unconditional denoiser output",
            &output.unconditional_denoised,
        ),
    ] {
        if input.descriptor() != tensor.descriptor() {
            return Err(CfgPpDenoiserContractError { output: name });
        }
    }
    Ok(())
}

pub const EULER_FOUNDATION_DEFINITION: SamplerDefinition = SamplerDefinition {
    identity: EULER_SAMPLER_ID,
    feature_id: EULER_SAMPLER_FEATURE_ID,
    source_ordinal: 0,
    aliases: &[],
    implementation_module: "algorithms/native_diffusion",
    stochastic: false,
};

#[derive(Clone, Debug)]
pub struct SamplerRegistry {
    definitions: Vec<SamplerDefinition>,
    lookup: BTreeMap<String, usize>,
}

impl SamplerRegistry {
    pub fn foundational() -> Result<Self, SamplingError> {
        let mut definitions = GENERATED_SAMPLER_DEFINITIONS.to_vec();
        if !definitions
            .iter()
            .any(|definition| definition.identity == EULER_FOUNDATION_DEFINITION.identity)
        {
            definitions.push(EULER_FOUNDATION_DEFINITION);
        }
        for definition in [
            crate::generated_void_ddim::DEFINITION,
            crate::generated_lcm_upscale::DEFINITION,
        ] {
            if !definitions
                .iter()
                .any(|existing| existing.identity == definition.identity)
            {
                definitions.push(definition);
            }
        }
        Self::new(definitions)
    }

    pub fn new(mut definitions: Vec<SamplerDefinition>) -> Result<Self, SamplingError> {
        definitions.sort_by_key(|definition| definition.source_ordinal);
        let mut identities = BTreeSet::new();
        let mut feature_ids = BTreeSet::new();
        let mut ordinals = BTreeSet::new();
        let mut modules = BTreeSet::new();
        let mut lookup = BTreeMap::new();
        for (index, definition) in definitions.iter().enumerate() {
            validate_identifier("sampler", definition.identity)?;
            validate_feature_id(definition.feature_id)?;
            if !identities.insert(definition.identity) {
                return Err(SamplingError::DuplicateIdentity {
                    kind: "sampler",
                    value: definition.identity.to_owned(),
                });
            }
            if !feature_ids.insert(definition.feature_id) {
                return Err(SamplingError::DuplicateFeatureId(
                    definition.feature_id.to_owned(),
                ));
            }
            if !ordinals.insert(definition.source_ordinal) {
                return Err(SamplingError::DuplicateSourceOrdinal {
                    kind: "sampler",
                    value: definition.source_ordinal,
                });
            }
            if !modules.insert(definition.implementation_module) {
                return Err(SamplingError::DuplicateImplementationModule(
                    definition.implementation_module.to_owned(),
                ));
            }
            insert_lookup(&mut lookup, definition.identity, index)?;
            for alias in definition.aliases {
                validate_identifier("sampler alias", alias)?;
                insert_lookup(&mut lookup, alias, index)?;
            }
        }
        if definitions.first().map(|definition| definition.identity) != Some(EULER_SAMPLER_ID) {
            return Err(SamplingError::InvalidDefault {
                kind: "sampler",
                expected: "euler",
            });
        }
        Ok(Self {
            definitions,
            lookup,
        })
    }

    pub fn default_definition(&self) -> &SamplerDefinition {
        &self.definitions[0]
    }

    pub fn resolve(&self, identity: &SamplerIdentity) -> Result<&SamplerDefinition, SamplingError> {
        self.lookup
            .get(identity.as_str())
            .and_then(|index| self.definitions.get(*index))
            .ok_or_else(|| SamplingError::UnknownSampler(identity.as_str().to_owned()))
    }

    pub fn definitions(&self) -> &[SamplerDefinition] {
        &self.definitions
    }
}

fn insert_lookup(
    lookup: &mut BTreeMap<String, usize>,
    value: &str,
    index: usize,
) -> Result<(), SamplingError> {
    if lookup.insert(value.to_owned(), index).is_some() {
        return Err(SamplingError::DuplicateAlias(value.to_owned()));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(try_from = "SamplingPlanWire")]
pub struct SamplingPlan {
    schema_version: u16,
    sampler: SamplerIdentity,
    scheduler: SchedulerIdentity,
    profile: SamplingProfileIdentity,
    seed: u64,
    steps: u32,
    guidance: f32,
    denoise: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SamplingPlanWire {
    schema_version: u16,
    sampler: SamplerIdentity,
    scheduler: SchedulerIdentity,
    profile: SamplingProfileIdentity,
    seed: u64,
    steps: u32,
    guidance: f32,
    denoise: f32,
}

impl SamplingPlan {
    pub fn new(
        sampler: impl Into<String>,
        scheduler: impl Into<String>,
        profile: SamplingProfileIdentity,
        seed: u64,
        steps: u32,
        guidance: f32,
        denoise: f32,
    ) -> Result<Self, SamplingError> {
        Self::try_from(SamplingPlanWire {
            schema_version: SAMPLING_PLAN_SCHEMA_VERSION,
            sampler: SamplerIdentity::new(sampler)?,
            scheduler: SchedulerIdentity::new(scheduler)?,
            profile,
            seed,
            steps,
            guidance,
            denoise,
        })
    }

    pub fn validate(
        &self,
        samplers: &SamplerRegistry,
        schedulers: &SchedulerRegistry,
        expected_profile: &SamplingProfileIdentity,
    ) -> Result<(), SamplingError> {
        samplers.resolve(&self.sampler)?;
        schedulers
            .resolve(&self.scheduler)
            .map_err(|error| SamplingError::Scheduler(error.to_string()))?;
        if &self.profile != expected_profile {
            return Err(SamplingError::ProfileMismatch {
                expected: expected_profile.as_str().to_owned(),
                actual: self.profile.as_str().to_owned(),
            });
        }
        validate_plan_values(self.steps, self.guidance, self.denoise)
    }

    pub fn sampler(&self) -> &SamplerIdentity {
        &self.sampler
    }

    pub fn scheduler(&self) -> &SchedulerIdentity {
        &self.scheduler
    }

    pub fn profile(&self) -> &SamplingProfileIdentity {
        &self.profile
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    pub fn guidance(&self) -> f32 {
        self.guidance
    }

    pub fn denoise(&self) -> f32 {
        self.denoise
    }
}

impl TryFrom<SamplingPlanWire> for SamplingPlan {
    type Error = SamplingError;

    fn try_from(wire: SamplingPlanWire) -> Result<Self, Self::Error> {
        if wire.schema_version != SAMPLING_PLAN_SCHEMA_VERSION {
            return Err(SamplingError::SchemaVersion(wire.schema_version));
        }
        validate_plan_values(wire.steps, wire.guidance, wire.denoise)?;
        Ok(Self {
            schema_version: wire.schema_version,
            sampler: wire.sampler,
            scheduler: wire.scheduler,
            profile: wire.profile,
            seed: wire.seed,
            steps: wire.steps,
            guidance: wire.guidance,
            denoise: wire.denoise,
        })
    }
}

fn validate_plan_values(steps: u32, guidance: f32, denoise: f32) -> Result<(), SamplingError> {
    if steps == 0 {
        return Err(SamplingError::ZeroSteps);
    }
    if !guidance.is_finite() || guidance < 0.0 {
        return Err(SamplingError::InvalidGuidance(guidance));
    }
    if !denoise.is_finite() || denoise <= 0.0 || denoise > 1.0 {
        return Err(SamplingError::InvalidDenoise(denoise));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SamplingProgress {
    pub step: u32,
    pub total_steps: u32,
    pub sigma: f32,
    pub sigma_hat: f32,
    pub next_sigma: f32,
}

#[derive(Clone, Debug)]
pub struct SamplingTrace {
    pub sigmas: Vec<f32>,
    pub denoiser_evaluations: Vec<Tensor>,
    pub latents: Vec<Tensor>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct AdaptiveSamplingProgress {
    pub attempt: u32,
    pub steps: u32,
    pub sigma: f32,
    pub sigma_hat: f32,
    pub proposed_sigma: f32,
    pub error: f32,
    pub step_size: f32,
    pub nfe: u32,
    pub accepted: bool,
    pub n_accept: u32,
    pub n_reject: u32,
}

pub struct AdaptiveSamplingAttempt {
    pub proposed_sigma: f32,
    pub base_denoised: Tensor,
    pub evaluations: Vec<Tensor>,
    pub proposed_low: Tensor,
    pub proposed_high: Tensor,
    pub stochastic_noise: Option<Tensor>,
    pub accepted_next: Option<Tensor>,
    pub error: f32,
    pub next_step_size: f32,
}

#[derive(Clone, Debug)]
pub struct AdaptiveSamplingAttemptTrace {
    pub progress: AdaptiveSamplingProgress,
    pub base_denoised: Tensor,
    pub evaluations: Vec<Tensor>,
    pub proposed_low: Tensor,
    pub proposed_high: Tensor,
    pub stochastic_noise: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct AdaptiveSamplingTrace {
    pub plan: SamplingPlan,
    pub initial_sigma: f32,
    pub terminal_sigma: f32,
    pub attempts: Vec<AdaptiveSamplingAttemptTrace>,
    pub latents: Vec<Tensor>,
}

pub struct AdaptiveSamplingSession {
    plan: SamplingPlan,
    initial_sigma: f32,
    terminal_sigma: f32,
    current_sigma: f32,
    current: Tensor,
    attempt_limit: u32,
    evaluations_per_attempt: u32,
    nfe: u32,
    n_accept: u32,
    n_reject: u32,
    attempts: Vec<AdaptiveSamplingAttemptTrace>,
    latents: Vec<Tensor>,
}

impl AdaptiveSamplingSession {
    pub fn new(
        plan: SamplingPlan,
        initial_sigma: f32,
        terminal_sigma: f32,
        initial: Tensor,
        attempt_limit: u32,
        evaluations_per_attempt: u32,
    ) -> Result<Self, SamplingError> {
        if !initial_sigma.is_finite()
            || !terminal_sigma.is_finite()
            || initial_sigma <= 0.0
            || terminal_sigma <= 0.0
            || initial_sigma < terminal_sigma
        {
            return Err(SamplingError::InvalidAdaptiveEndpoints {
                initial_sigma,
                terminal_sigma,
            });
        }
        if attempt_limit == 0 || attempt_limit > MAX_ADAPTIVE_SAMPLING_ATTEMPTS {
            return Err(SamplingError::InvalidAdaptiveAttemptLimit(attempt_limit));
        }
        if evaluations_per_attempt == 0 || evaluations_per_attempt > 16 {
            return Err(SamplingError::InvalidAdaptiveEvaluationCount(
                evaluations_per_attempt,
            ));
        }
        let attempt_capacity = usize::try_from(attempt_limit)
            .map_err(|_| SamplingError::Overflow("adaptive attempt capacity"))?;
        let mut attempts = Vec::new();
        attempts
            .try_reserve_exact(attempt_capacity)
            .map_err(|_| SamplingError::OutOfMemory("adaptive attempt trace"))?;
        let latent_capacity = attempt_capacity
            .checked_add(1)
            .ok_or(SamplingError::Overflow("adaptive latent trace capacity"))?;
        let mut latents = Vec::new();
        latents
            .try_reserve_exact(latent_capacity)
            .map_err(|_| SamplingError::OutOfMemory("adaptive latent trace"))?;
        latents.push(initial.clone());
        Ok(Self {
            plan,
            initial_sigma,
            terminal_sigma,
            current_sigma: initial_sigma,
            current: initial,
            attempt_limit,
            evaluations_per_attempt,
            nfe: 0,
            n_accept: 0,
            n_reject: 0,
            attempts,
            latents,
        })
    }

    pub fn current(&self) -> &Tensor {
        &self.current
    }

    pub fn current_sigma(&self) -> f32 {
        self.current_sigma
    }

    pub fn is_complete(&self) -> bool {
        let current_time = -self.current_sigma.ln();
        let terminal_time = -self.terminal_sigma.ln();
        current_time >= terminal_time - 1.0e-5
    }

    pub fn next_attempt(&self, cancellation: &CancellationToken) -> Result<u32, SamplingError> {
        cancellation.check().map_err(|_| SamplingError::Cancelled)?;
        if self.is_complete() {
            return Err(SamplingError::SessionComplete);
        }
        let attempt = u32::try_from(self.attempts.len())
            .map_err(|_| SamplingError::Overflow("adaptive attempt index"))?;
        if attempt >= self.attempt_limit {
            return Err(SamplingError::AdaptiveAttemptLimitExceeded {
                limit: self.attempt_limit,
            });
        }
        Ok(attempt)
    }

    pub fn commit_attempt<E>(
        &mut self,
        attempt: AdaptiveSamplingAttempt,
        cancellation: &CancellationToken,
        callback: impl FnOnce(&AdaptiveSamplingProgress, &Tensor, &Tensor) -> Result<(), E>,
    ) -> Result<(), SamplingError>
    where
        E: std::fmt::Display,
    {
        let attempt_index = self.next_attempt(cancellation)?;
        if !attempt.proposed_sigma.is_finite()
            || attempt.proposed_sigma <= 0.0
            || attempt.proposed_sigma > self.current_sigma
            || attempt.proposed_sigma < self.terminal_sigma
        {
            return Err(SamplingError::InvalidAdaptiveProposedSigma {
                current_sigma: self.current_sigma,
                proposed_sigma: attempt.proposed_sigma,
                terminal_sigma: self.terminal_sigma,
            });
        }
        if !attempt.error.is_finite() || attempt.error < 0.0 {
            return Err(SamplingError::InvalidAdaptiveError(attempt.error));
        }
        if !attempt.next_step_size.is_finite() || attempt.next_step_size <= 0.0 {
            return Err(SamplingError::InvalidAdaptiveStepSize(
                attempt.next_step_size,
            ));
        }
        let expected_evaluations = usize::try_from(self.evaluations_per_attempt)
            .map_err(|_| SamplingError::Overflow("adaptive evaluation count"))?;
        if attempt.evaluations.len() != expected_evaluations {
            return Err(SamplingError::AdaptiveEvaluationCount {
                expected: self.evaluations_per_attempt,
                actual: attempt.evaluations.len(),
            });
        }
        validate_compatible(
            &self.current,
            &attempt.base_denoised,
            "adaptive base denoised",
        )?;
        validate_compatible(
            &self.current,
            &attempt.proposed_low,
            "adaptive low proposal",
        )?;
        validate_compatible(
            &self.current,
            &attempt.proposed_high,
            "adaptive high proposal",
        )?;
        for evaluation in &attempt.evaluations {
            validate_compatible(&self.current, evaluation, "adaptive denoiser evaluation")?;
        }
        if let Some(noise) = &attempt.stochastic_noise {
            validate_compatible(&self.current, noise, "adaptive stochastic noise")?;
        }
        let accepted = attempt.accepted_next.is_some();
        let post_latent = match &attempt.accepted_next {
            Some(next) => {
                validate_compatible(&self.current, next, "adaptive accepted latent")?;
                next
            }
            None => &self.current,
        };
        let nfe = self
            .nfe
            .checked_add(self.evaluations_per_attempt)
            .ok_or(SamplingError::Overflow("adaptive function evaluations"))?;
        let n_accept = self
            .n_accept
            .checked_add(u32::from(accepted))
            .ok_or(SamplingError::Overflow("adaptive accepted attempts"))?;
        let n_reject = self
            .n_reject
            .checked_add(u32::from(!accepted))
            .ok_or(SamplingError::Overflow("adaptive rejected attempts"))?;
        let steps = attempt_index
            .checked_add(1)
            .ok_or(SamplingError::Overflow("adaptive step count"))?;
        let sigma = if accepted {
            attempt.proposed_sigma
        } else {
            self.current_sigma
        };
        let progress = AdaptiveSamplingProgress {
            attempt: attempt_index,
            steps,
            sigma,
            sigma_hat: sigma,
            proposed_sigma: attempt.proposed_sigma,
            error: attempt.error,
            step_size: attempt.next_step_size,
            nfe,
            accepted,
            n_accept,
            n_reject,
        };
        callback(&progress, post_latent, &attempt.base_denoised)
            .map_err(|error| SamplingError::Callback(error.to_string()))?;
        cancellation.check().map_err(|_| SamplingError::Cancelled)?;

        let post_latent = attempt
            .accepted_next
            .unwrap_or_else(|| self.current.clone());
        self.attempts.push(AdaptiveSamplingAttemptTrace {
            progress,
            base_denoised: attempt.base_denoised,
            evaluations: attempt.evaluations,
            proposed_low: attempt.proposed_low,
            proposed_high: attempt.proposed_high,
            stochastic_noise: attempt.stochastic_noise,
        });
        self.latents.push(post_latent.clone());
        self.current = post_latent;
        self.current_sigma = sigma;
        self.nfe = nfe;
        self.n_accept = n_accept;
        self.n_reject = n_reject;
        Ok(())
    }

    pub fn finish(self) -> Result<AdaptiveSamplingTrace, SamplingError> {
        if !self.is_complete() {
            return Err(SamplingError::IncompleteAdaptiveSession {
                current_sigma: self.current_sigma,
                terminal_sigma: self.terminal_sigma,
            });
        }
        Ok(AdaptiveSamplingTrace {
            plan: self.plan,
            initial_sigma: self.initial_sigma,
            terminal_sigma: self.terminal_sigma,
            attempts: self.attempts,
            latents: self.latents,
        })
    }
}

pub struct SamplingSession {
    plan: SamplingPlan,
    sigmas: Vec<f32>,
    next_step: usize,
    current: Tensor,
    denoiser_evaluations: Vec<Tensor>,
    latents: Vec<Tensor>,
}

#[must_use = "an observed sampling step must be committed or deliberately dropped"]
pub struct ObservedSamplingStep<'a> {
    session: &'a mut SamplingSession,
    denoised: Tensor,
}

impl ObservedSamplingStep<'_> {
    pub fn commit(
        self,
        next: Tensor,
        cancellation: &CancellationToken,
    ) -> Result<(), SamplingError> {
        cancellation.check().map_err(|_| SamplingError::Cancelled)?;
        validate_compatible(&self.session.current, &next, "next latent")?;
        self.session.denoiser_evaluations.push(self.denoised);
        self.session.latents.push(next.clone());
        self.session.current = next;
        self.session.next_step += 1;
        Ok(())
    }

    pub fn commit_shape_transition(
        self,
        next: Tensor,
        cancellation: &CancellationToken,
    ) -> Result<(), SamplingError> {
        cancellation.check().map_err(|_| SamplingError::Cancelled)?;
        validate_storage_compatible(&self.session.current, &next, "next latent")?;
        self.session.denoiser_evaluations.push(self.denoised);
        self.session.latents.push(next.clone());
        self.session.current = next;
        self.session.next_step += 1;
        Ok(())
    }
}

impl SamplingSession {
    pub fn new(
        plan: SamplingPlan,
        sigmas: Vec<f32>,
        initial: Tensor,
    ) -> Result<Self, SamplingError> {
        let expected = usize::try_from(plan.steps)
            .ok()
            .and_then(|steps| steps.checked_add(1))
            .ok_or(SamplingError::Overflow("sampling schedule length"))?;
        validate_sigmas(&sigmas, expected)?;
        let mut denoiser_evaluations = Vec::new();
        denoiser_evaluations
            .try_reserve_exact(expected - 1)
            .map_err(|_| SamplingError::OutOfMemory("denoiser trace"))?;
        let mut latents = Vec::new();
        latents
            .try_reserve_exact(expected)
            .map_err(|_| SamplingError::OutOfMemory("latent trace"))?;
        latents.push(initial.clone());
        Ok(Self {
            plan,
            sigmas,
            next_step: 0,
            current: initial,
            denoiser_evaluations,
            latents,
        })
    }

    pub fn current(&self) -> &Tensor {
        &self.current
    }

    pub fn next_step(&self) -> usize {
        self.next_step
    }

    pub fn progress(&self) -> Result<SamplingProgress, SamplingError> {
        let sigma = self
            .sigmas
            .get(self.next_step)
            .copied()
            .ok_or(SamplingError::SessionComplete)?;
        let next_sigma = self
            .sigmas
            .get(self.next_step + 1)
            .copied()
            .ok_or(SamplingError::SessionComplete)?;
        Ok(SamplingProgress {
            step: u32::try_from(self.next_step)
                .map_err(|_| SamplingError::Overflow("sampling step"))?,
            total_steps: self.plan.steps,
            sigma,
            sigma_hat: sigma,
            next_sigma,
        })
    }

    pub fn observe_step<E>(
        &mut self,
        callback_latent: &Tensor,
        denoised: Tensor,
        cancellation: &CancellationToken,
        callback: impl FnOnce(&SamplingProgress, &Tensor, &Tensor) -> Result<(), E>,
    ) -> Result<ObservedSamplingStep<'_>, SamplingError>
    where
        E: std::fmt::Display,
    {
        cancellation.check().map_err(|_| SamplingError::Cancelled)?;
        let progress = self.progress()?;
        validate_compatible(&self.current, callback_latent, "callback latent")?;
        validate_compatible(&self.current, &denoised, "denoiser output")?;
        callback(&progress, callback_latent, &denoised)
            .map_err(|error| SamplingError::Callback(error.to_string()))?;
        cancellation.check().map_err(|_| SamplingError::Cancelled)?;
        Ok(ObservedSamplingStep {
            session: self,
            denoised,
        })
    }

    pub fn commit_step<E>(
        &mut self,
        denoised: Tensor,
        next: Tensor,
        cancellation: &CancellationToken,
        callback: impl FnOnce(&SamplingProgress, &Tensor, &Tensor) -> Result<(), E>,
    ) -> Result<(), SamplingError>
    where
        E: std::fmt::Display,
    {
        let callback_latent = self.current.clone();
        self.observe_step(
            &callback_latent,
            denoised,
            cancellation,
            |progress, _, denoised| callback(progress, denoised, &next),
        )?
        .commit(next, cancellation)
    }

    pub fn finish(self) -> Result<SamplingTrace, SamplingError> {
        if self.next_step != usize::try_from(self.plan.steps).unwrap_or(usize::MAX) {
            return Err(SamplingError::IncompleteSession {
                completed: self.next_step,
                expected: self.plan.steps,
            });
        }
        Ok(SamplingTrace {
            sigmas: self.sigmas,
            denoiser_evaluations: self.denoiser_evaluations,
            latents: self.latents,
        })
    }
}

fn validate_sigmas(sigmas: &[f32], expected: usize) -> Result<(), SamplingError> {
    if sigmas.len() != expected {
        return Err(SamplingError::ScheduleLength {
            expected,
            actual: sigmas.len(),
        });
    }
    for (step, pair) in sigmas.windows(2).enumerate() {
        if !pair[0].is_finite()
            || !pair[1].is_finite()
            || pair[0] <= 0.0
            || pair[1] < 0.0
            || pair[1] >= pair[0]
        {
            return Err(SamplingError::InvalidSigma {
                step,
                sigma: pair[0],
                next_sigma: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_compatible(
    expected: &Tensor,
    actual: &Tensor,
    role: &'static str,
) -> Result<(), SamplingError> {
    if expected.descriptor() != actual.descriptor() {
        return Err(SamplingError::TensorContract {
            role,
            expected: format!("{:?}", expected.descriptor()),
            actual: format!("{:?}", actual.descriptor()),
        });
    }
    Ok(())
}

fn validate_storage_compatible(
    expected: &Tensor,
    actual: &Tensor,
    role: &'static str,
) -> Result<(), SamplingError> {
    let expected = expected.descriptor();
    let actual = actual.descriptor();
    if expected.dtype() != actual.dtype()
        || expected.device() != actual.device()
        || expected.stream() != actual.stream()
        || expected.layout() != actual.layout()
        || expected.shape().len() != actual.shape().len()
    {
        return Err(SamplingError::TensorContract {
            role,
            expected: format!(
                "rank {}, dtype {:?}, layout {:?}, device {:?}, stream {:?}",
                expected.shape().len(),
                expected.dtype(),
                expected.layout(),
                expected.device(),
                expected.stream()
            ),
            actual: format!("{actual:?}"),
        });
    }
    Ok(())
}

pub(crate) fn validate_identifier(kind: &'static str, value: &str) -> Result<(), SamplingError> {
    if value.is_empty()
        || value.len() > 96
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
        || !value
            .as_bytes()
            .first()
            .is_some_and(|byte| byte.is_ascii_lowercase())
    {
        return Err(SamplingError::InvalidIdentity {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_feature_id(value: &str) -> Result<(), SamplingError> {
    if value.len() != "COMFY-MODEL-0000".len()
        || !value.starts_with("COMFY-MODEL-")
        || !value["COMFY-MODEL-".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_digit())
    {
        return Err(SamplingError::InvalidFeatureId(value.to_owned()));
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SamplingError {
    #[error("invalid {kind} identity {value:?}")]
    InvalidIdentity { kind: &'static str, value: String },
    #[error("invalid sampling feature ID {0:?}")]
    InvalidFeatureId(String),
    #[error("duplicate {kind} identity {value:?}")]
    DuplicateIdentity { kind: &'static str, value: String },
    #[error("duplicate sampling feature ID {0:?}")]
    DuplicateFeatureId(String),
    #[error("duplicate {kind} source ordinal {value}")]
    DuplicateSourceOrdinal { kind: &'static str, value: u16 },
    #[error("duplicate sampler implementation module {0:?}")]
    DuplicateImplementationModule(String),
    #[error("duplicate sampler identity or alias {0:?}")]
    DuplicateAlias(String),
    #[error("the source-order default {kind} must be {expected}")]
    InvalidDefault {
        kind: &'static str,
        expected: &'static str,
    },
    #[error("unknown sampler {0:?}")]
    UnknownSampler(String),
    #[error("sampling plan schema version {0} is unsupported")]
    SchemaVersion(u16),
    #[error("sampling steps must be nonzero")]
    ZeroSteps,
    #[error("sampling guidance must be finite and nonnegative, got {0}")]
    InvalidGuidance(f32),
    #[error("sampling denoise must be finite and in (0, 1], got {0}")]
    InvalidDenoise(f32),
    #[error("sampling profile mismatch: expected {expected}, got {actual}")]
    ProfileMismatch { expected: String, actual: String },
    #[error("scheduler validation failed: {0}")]
    Scheduler(String),
    #[error("sampling schedule length mismatch: expected {expected}, got {actual}")]
    ScheduleLength { expected: usize, actual: usize },
    #[error("invalid sigma transition at step {step}: {sigma} -> {next_sigma}")]
    InvalidSigma {
        step: usize,
        sigma: f32,
        next_sigma: f32,
    },
    #[error("sampling tensor contract failed for {role}: expected {expected}, got {actual}")]
    TensorContract {
        role: &'static str,
        expected: String,
        actual: String,
    },
    #[error("sampling callback failed: {0}")]
    Callback(String),
    #[error("sampling session is already complete")]
    SessionComplete,
    #[error("sampling session is incomplete: completed {completed}, expected {expected}")]
    IncompleteSession { completed: usize, expected: u32 },
    #[error(
        "adaptive sampling endpoints must be finite, positive, and descending, got {initial_sigma} -> {terminal_sigma}"
    )]
    InvalidAdaptiveEndpoints {
        initial_sigma: f32,
        terminal_sigma: f32,
    },
    #[error("adaptive sampling attempt limit {0} is invalid")]
    InvalidAdaptiveAttemptLimit(u32),
    #[error("adaptive sampling evaluation count {0} is invalid")]
    InvalidAdaptiveEvaluationCount(u32),
    #[error("adaptive sampling attempt limit {limit} was exceeded")]
    AdaptiveAttemptLimitExceeded { limit: u32 },
    #[error(
        "adaptive proposed sigma must remain within the descending interval {current_sigma} -> {terminal_sigma}, got {proposed_sigma}"
    )]
    InvalidAdaptiveProposedSigma {
        current_sigma: f32,
        proposed_sigma: f32,
        terminal_sigma: f32,
    },
    #[error("adaptive sampling error must be finite and nonnegative, got {0}")]
    InvalidAdaptiveError(f32),
    #[error("adaptive sampling step size must be finite and positive, got {0}")]
    InvalidAdaptiveStepSize(f32),
    #[error("adaptive sampling expected {expected} evaluations, got {actual}")]
    AdaptiveEvaluationCount { expected: u32, actual: usize },
    #[error(
        "adaptive sampling session is incomplete at sigma {current_sigma}; terminal sigma is {terminal_sigma}"
    )]
    IncompleteAdaptiveSession {
        current_sigma: f32,
        terminal_sigma: f32,
    },
    #[error("sampling operation was cancelled")]
    Cancelled,
    #[error("sampling allocation failed for {0}")]
    OutOfMemory(&'static str),
    #[error("sampling arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}
