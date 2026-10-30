use sim_sampler::{
    CompatibilityNoiseRequest, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfileIdentity,
    generated_dpm_adaptive_sim_model_0164::{
        DEFINITION, DPM_ADAPTIVE_NOISE_CONTRACT_ID, DPM_ADAPTIVE_SAMPLER_ID, DpmAdaptiveEvaluation,
        DpmAdaptiveEvaluationStage, DpmAdaptiveOptions, DpmAdaptiveSamplerError,
        sample_dpm_adaptive,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, RetryRngPolicy, RngCompatibilityError, RngCompatibilityPhase,
    RngGenerationPlacement, RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::PathBuf};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpm_adaptive_comfy_model_0164/oracle.json"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    tolerance: f32,
    seed: u64,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    options: OptionsFixture,
    attempts: Vec<AttemptFixture>,
    terminal: Vec<f32>,
    info: InfoFixture,
    noise: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    wrapper_path: String,
    wrapper_sha256: String,
    equation_path: String,
    equation_sha256: String,
    catalog_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct OptionsFixture {
    order: u8,
    relative_tolerance: f32,
    absolute_tolerance: f32,
    initial_step_size: f32,
    proportional_coefficient: f64,
    integral_coefficient: f64,
    derivative_coefficient: f64,
    acceptance_safety: f64,
    eta: f32,
    noise_scale: f32,
    attempt_limit: u32,
}

impl From<OptionsFixture> for DpmAdaptiveOptions {
    fn from(value: OptionsFixture) -> Self {
        Self {
            order: value.order,
            relative_tolerance: value.relative_tolerance,
            absolute_tolerance: value.absolute_tolerance,
            initial_step_size: value.initial_step_size,
            proportional_coefficient: value.proportional_coefficient,
            integral_coefficient: value.integral_coefficient,
            derivative_coefficient: value.derivative_coefficient,
            acceptance_safety: value.acceptance_safety,
            eta: value.eta,
            noise_scale: value.noise_scale,
            attempt_limit: value.attempt_limit,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AttemptFixture {
    attempt: u32,
    proposed_sigma: f32,
    reported_sigma: f32,
    step_size: f32,
    error: f32,
    accepted: bool,
    nfe: u32,
    n_accept: u32,
    n_reject: u32,
    latent_before: Vec<f32>,
    low: Vec<f32>,
    high: Vec<f32>,
    latent_after: Vec<f32>,
    base_denoised: Vec<f32>,
    evaluations: Vec<EvaluationFixture>,
}

#[derive(Debug, Deserialize)]
struct EvaluationFixture {
    stage: String,
    sigma: f32,
    input: Vec<f32>,
    denoised: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct InfoFixture {
    steps: u32,
    nfe: u32,
    n_accept: u32,
    n_reject: u32,
}

#[derive(Debug)]
struct ObservedEvaluation {
    address: DpmAdaptiveEvaluation,
    sigma: f32,
    input: Vec<f32>,
    denoised: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn digest(path: &str) -> Result<String, Box<dyn Error>> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(workspace()?.join(path))?)
    ))
}

fn profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("dpm-adaptive-row-v1")?)
}

fn plan(seed: u64, steps: u32) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        DPM_ADAPTIVE_SAMPLER_ID,
        "normal",
        profile()?,
        seed,
        steps,
        1.0,
        1.0,
    )?)
}

fn noise_request(retry: u32, retry_policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "dpm-adaptive-fixture-v1",
        "attempt-0164",
        "KSampler-12",
        12,
        164,
        73,
        retry,
        retry_policy,
    )
}

fn open_noise_transaction(
    request: CompatibilityNoiseRequest,
    seed: u64,
    cancellation: &CancellationToken,
) -> Result<CompatibilityRngTransaction, RngCompatibilityError> {
    request.open_transaction(
        DPM_ADAPTIVE_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        cancellation,
    )
}

fn execution_context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        cancellation,
    ))
}

fn values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor_to_f32(backend, tensor, context)?.to_vec())
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

fn assert_scalar(actual: f32, expected: f32, tolerance: f32, role: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{role}: expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

fn analytical_denoised(input: &[f32], sigma: f32) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| 0.72_f32 * *value + sigma * if index == 0 { 0.11 } else { -0.18 })
        .collect()
}

#[test]
fn val_sampler_001_dpm_adaptive_definition_and_provenance_are_exact() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPM_ADAPTIVE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DEFINITION.feature_id);
    assert_eq!(fixture.source_ordinal, DEFINITION.source_ordinal);
    assert_eq!(DEFINITION.identity, "dpm_adaptive");
    assert_eq!(DEFINITION.feature_id, "COMFY-MODEL-0164");
    assert_eq!(DEFINITION.source_ordinal, 12);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpm_adaptive_comfy_model_0164"
    );
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new("dpm_adaptive")?)?,
        &DEFINITION
    );
    assert!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new("dpm_fast")?)
            .map_or(true, |definition| definition != &DEFINITION)
    );
    assert_eq!(
        digest(&fixture.source.wrapper_path)?,
        fixture.source.wrapper_sha256
    );
    assert_eq!(
        digest(&fixture.source.equation_path)?,
        fixture.source.equation_sha256
    );
    assert_eq!(
        digest(".agents/specs/comfy-parity/catalogs/backend-models.csv")?,
        fixture.source.catalog_sha256
    );
    Ok(())
}

#[test]
fn val_sampler_001_dpm_adaptive_matches_every_attempt_and_intermediate()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &fixture.initial, &context)?;
    let observed_evaluations = RefCell::new(Vec::new());
    let observed_callbacks = RefCell::new(Vec::new());
    let trace = sample_dpm_adaptive(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        &profile()?,
        initial,
        &fixture.sigmas,
        fixture.options.into(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, sigma, address| {
            let input_values =
                values(&backend, input, &context).map_err(|error| error.to_string())?;
            let denoised_values = analytical_denoised(&input_values, sigma);
            observed_evaluations.borrow_mut().push(ObservedEvaluation {
                address,
                sigma,
                input: input_values,
                denoised: denoised_values.clone(),
            });
            tensor_from_f32(&backend, &[2], &denoised_values, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            observed_callbacks.borrow_mut().push((
                *progress,
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<(), String>(())
        },
    )?;
    let sampling = trace.sampling.as_ref().ok_or("missing adaptive trace")?;
    assert_eq!(
        sampling.plan.steps(),
        u32::try_from(fixture.sigmas.len() - 1)?
    );
    assert_eq!(sampling.attempts.len(), fixture.attempts.len());
    assert_eq!(sampling.latents.len(), fixture.attempts.len() + 1);
    assert_eq!(fixture.info.steps, u32::try_from(fixture.attempts.len())?);
    assert_eq!(
        fixture.info.nfe,
        fixture.info.steps * u32::from(fixture.options.order)
    );
    assert_eq!(
        fixture.info.n_accept + fixture.info.n_reject,
        fixture.info.steps
    );
    assert_eq!(fixture.noise.len(), usize::try_from(fixture.info.n_accept)?);
    let callbacks = observed_callbacks.into_inner();
    assert_eq!(callbacks.len(), fixture.attempts.len());
    let evaluations = observed_evaluations.into_inner();
    assert_eq!(evaluations.len(), usize::try_from(fixture.info.nfe)?);
    let mut evaluation_index = 0_usize;
    for (index, expected) in fixture.attempts.iter().enumerate() {
        let actual = sampling
            .attempts
            .get(index)
            .ok_or("missing attempt trace")?;
        let (progress, callback_latent, callback_denoised) =
            callbacks.get(index).ok_or("missing callback")?;
        assert_eq!(progress.attempt, expected.attempt);
        assert_eq!(progress.steps, expected.attempt + 1);
        assert_eq!(progress.accepted, expected.accepted);
        assert_eq!(progress.nfe, expected.nfe);
        assert_eq!(progress.n_accept, expected.n_accept);
        assert_eq!(progress.n_reject, expected.n_reject);
        assert_scalar(
            progress.proposed_sigma,
            expected.proposed_sigma,
            fixture.tolerance,
            "proposed sigma",
        );
        assert_scalar(
            progress.sigma,
            expected.reported_sigma,
            fixture.tolerance,
            "reported sigma",
        );
        assert_scalar(
            progress.sigma_hat,
            expected.reported_sigma,
            fixture.tolerance,
            "sigma hat",
        );
        assert_scalar(
            progress.step_size,
            expected.step_size,
            fixture.tolerance,
            "step size",
        );
        assert_scalar(progress.error, expected.error, fixture.tolerance, "error");
        assert_eq!(actual.progress, *progress);
        assert_close(callback_latent, &expected.latent_after, fixture.tolerance);
        assert_close(
            callback_denoised,
            &expected.base_denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &actual.proposed_low, &context)?,
            &expected.low,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &actual.proposed_high, &context)?,
            &expected.high,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &actual.base_denoised, &context)?,
            &expected.base_denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                sampling
                    .latents
                    .get(index)
                    .ok_or("missing pre-attempt latent")?,
                &context,
            )?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                sampling
                    .latents
                    .get(index + 1)
                    .ok_or("missing post-attempt latent")?,
                &context,
            )?,
            &expected.latent_after,
            fixture.tolerance,
        );
        assert_eq!(actual.evaluations.len(), expected.evaluations.len());
        for (stage_index, expected_evaluation) in expected.evaluations.iter().enumerate() {
            let observed = evaluations
                .get(evaluation_index)
                .ok_or("missing evaluation")?;
            evaluation_index += 1;
            assert_eq!(observed.address.attempt, expected.attempt);
            let expected_stage = match expected_evaluation.stage.as_str() {
                "base" => DpmAdaptiveEvaluationStage::Base,
                "first_intermediate" => DpmAdaptiveEvaluationStage::FirstIntermediate,
                "second_intermediate" => DpmAdaptiveEvaluationStage::SecondIntermediate,
                stage => return Err(format!("unknown fixture stage {stage:?}").into()),
            };
            assert_eq!(observed.address.stage, expected_stage);
            assert_scalar(
                observed.sigma,
                expected_evaluation.sigma,
                fixture.tolerance,
                "evaluation sigma",
            );
            assert_close(
                &observed.input,
                &expected_evaluation.input,
                fixture.tolerance,
            );
            assert_close(
                &observed.denoised,
                &expected_evaluation.denoised,
                fixture.tolerance,
            );
            assert_close(
                &values(
                    &backend,
                    actual
                        .evaluations
                        .get(stage_index)
                        .ok_or("missing traced evaluation")?,
                    &context,
                )?,
                &expected_evaluation.denoised,
                fixture.tolerance,
            );
        }
        assert!(actual.stochastic_noise.is_some() == expected.accepted);
    }
    assert_close(
        &values(&backend, &trace.output, &context)?,
        &fixture.terminal,
        fixture.tolerance,
    );

    let mut oracle = open_noise_transaction(
        noise_request(0, RetryRngPolicy::Replay),
        fixture.seed,
        &cancellation,
    )?;
    assert_eq!(oracle.contract().rng_id(), DPM_ADAPTIVE_NOISE_CONTRACT_ID);
    assert_eq!(
        oracle.contract().phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(trace.noise_before.as_ref(), Some(&oracle.checkpoint()));
    let mut observed_noise = Vec::new();
    for attempt in &sampling.attempts {
        if let Some(noise) = &attempt.stochastic_noise {
            let expected = oracle
                .draw_normal(fixture.initial.len(), &cancellation)?
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let actual = values(&backend, noise, &context)?;
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            observed_noise.push(actual);
        }
    }
    assert_eq!(trace.noise_after.as_ref(), Some(&oracle.commit()));
    assert_eq!(observed_noise.len(), fixture.noise.len());
    for (actual, expected) in observed_noise.iter().zip(&fixture.noise) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn dpm_adaptive_validates_profile_short_schedule_and_attempt_bound() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[0.25, -0.5], &context)?;
    let calls = RefCell::new(0_u32);
    let result = sample_dpm_adaptive(
        &backend,
        plan(164, 1)?,
        &SamplingProfileIdentity::new("wrong-profile-v1")?,
        initial.clone(),
        &[2.0, 0.0],
        DpmAdaptiveOptions::default(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |_, _, _| {
            *calls.borrow_mut() += 1;
            Ok(initial.clone())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        result,
        Err(DpmAdaptiveSamplerError::Sampling(
            SamplingError::ProfileMismatch { .. }
        ))
    ));
    assert_eq!(*calls.borrow(), 0);

    let short = sample_dpm_adaptive(
        &backend,
        plan(164, 1)?,
        &profile()?,
        initial.clone(),
        &[2.0],
        DpmAdaptiveOptions::default(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |_, _, _| Err("short schedule invoked denoiser".to_owned()),
        |_, _, _| Err("short schedule invoked callback".to_owned()),
    )?;
    assert!(short.sampling.is_none());
    assert!(short.noise_before.is_none());
    assert!(short.noise_after.is_none());
    assert_close(
        &values(&backend, &short.output, &context)?,
        &[0.25, -0.5],
        0.0,
    );

    let invalid_initial = tensor_from_f32(&backend, &[2], &[0.25, -0.5], &context)?;
    let invalid_schedule = sample_dpm_adaptive(
        &backend,
        plan(164, 2)?,
        &profile()?,
        invalid_initial.clone(),
        &[2.0, 0.0, 0.0],
        DpmAdaptiveOptions::default(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, _, _| Ok(input.clone()),
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        invalid_schedule,
        Err(DpmAdaptiveSamplerError::InvalidSigma(value)) if value == 0.0
    ));
    let invalid_order = sample_dpm_adaptive(
        &backend,
        plan(164, 2)?,
        &profile()?,
        invalid_initial,
        &[2.0, 0.5, 0.0],
        DpmAdaptiveOptions {
            order: 4,
            ..DpmAdaptiveOptions::default()
        },
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, _, _| Ok(input.clone()),
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        invalid_order,
        Err(DpmAdaptiveSamplerError::InvalidOrder(4))
    ));

    let options = DpmAdaptiveOptions {
        relative_tolerance: 0.0001,
        absolute_tolerance: 0.00003,
        initial_step_size: 0.35,
        attempt_limit: 1,
        ..DpmAdaptiveOptions::default()
    };
    let denoiser_calls = RefCell::new(0_u32);
    let result = sample_dpm_adaptive(
        &backend,
        plan(164, 3)?,
        &profile()?,
        initial,
        &[2.0, 1.0, 0.5, 0.0],
        options,
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, sigma, _| {
            *denoiser_calls.borrow_mut() += 1;
            let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
            tensor_from_f32(
                &backend,
                &[2],
                &analytical_denoised(&input, sigma),
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        result,
        Err(DpmAdaptiveSamplerError::Sampling(
            SamplingError::AdaptiveAttemptLimitExceeded { limit: 1 }
        ))
    ));
    assert_eq!(*denoiser_calls.borrow(), 3);
    Ok(())
}

#[test]
fn dpm_adaptive_is_cancellable_and_callback_failure_is_rng_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[0.25, -0.5], &context)?;
    assert!(cancellation.cancel());
    let calls = RefCell::new(0_u32);
    let cancelled = sample_dpm_adaptive(
        &backend,
        plan(164, 2)?,
        &profile()?,
        initial,
        &[2.0, 0.5, 0.0],
        DpmAdaptiveOptions::default(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, _, _| {
            *calls.borrow_mut() += 1;
            Ok(input.clone())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        cancelled,
        Err(DpmAdaptiveSamplerError::Tensor(TensorError::Cancelled))
    ));
    assert_eq!(*calls.borrow(), 0);

    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[0.25, -0.5], &context)?;
    let failed = sample_dpm_adaptive(
        &backend,
        plan(164, 2)?,
        &profile()?,
        initial.clone(),
        &[2.0, 0.5, 0.0],
        DpmAdaptiveOptions::default(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, _, _| Ok(input.clone()),
        |_, _, _| Err("injected adaptive callback failure"),
    );
    assert!(matches!(
        failed,
        Err(DpmAdaptiveSamplerError::Sampling(SamplingError::Callback(reason)))
            if reason == "injected adaptive callback failure"
    ));
    let mut replay =
        open_noise_transaction(noise_request(0, RetryRngPolicy::Replay), 164, &cancellation)?;
    let replayed = replay.draw_normal(2, &cancellation)?;
    assert_eq!(replayed.len(), 2);
    let mut replay_again =
        open_noise_transaction(noise_request(9, RetryRngPolicy::Replay), 164, &cancellation)?;
    assert_eq!(replay_again.draw_normal(2, &cancellation)?, replayed);

    let callback_cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &callback_cancellation)?;
    let cancelled_after_callback = sample_dpm_adaptive(
        &backend,
        plan(164, 2)?,
        &profile()?,
        initial,
        &[2.0, 0.5, 0.0],
        DpmAdaptiveOptions::default(),
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, _, _| Ok(input.clone()),
        |_, _, _| {
            assert!(callback_cancellation.cancel());
            Ok::<(), String>(())
        },
    );
    assert!(matches!(
        cancelled_after_callback,
        Err(DpmAdaptiveSamplerError::Sampling(SamplingError::Cancelled))
    ));
    Ok(())
}

#[test]
fn val_rng_001_dpm_adaptive_retry_policy_and_order_two_are_exact() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let mut replay_zero =
        open_noise_transaction(noise_request(0, RetryRngPolicy::Replay), 164, &cancellation)?;
    let mut replay_seven =
        open_noise_transaction(noise_request(7, RetryRngPolicy::Replay), 164, &cancellation)?;
    let replay_zero_values = replay_zero.draw_normal(4, &cancellation)?;
    let replay_seven_values = replay_seven.draw_normal(4, &cancellation)?;
    assert_eq!(replay_zero_values, replay_seven_values);
    let mut advance = open_noise_transaction(
        noise_request(7, RetryRngPolicy::Advance),
        164,
        &cancellation,
    )?;
    assert_ne!(advance.draw_normal(4, &cancellation)?, replay_zero_values);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial_values = [0.25, -0.5];
    let initial = tensor_from_f32(&backend, &[2], &initial_values, &context)?;
    let trace = sample_dpm_adaptive(
        &backend,
        plan(164, 2)?,
        &profile()?,
        initial,
        &[2.0, 0.5, 0.0],
        DpmAdaptiveOptions {
            order: 2,
            initial_step_size: 2.0,
            attempt_limit: 4,
            ..DpmAdaptiveOptions::default()
        },
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, _, _| Ok(input.clone()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    let sampling = trace.sampling.as_ref().ok_or("missing order-two trace")?;
    assert_eq!(sampling.attempts.len(), 1);
    let attempt = sampling
        .attempts
        .first()
        .ok_or("missing order-two attempt")?;
    assert!(attempt.progress.accepted);
    assert_eq!(attempt.progress.nfe, 2);
    assert_eq!(attempt.evaluations.len(), 2);
    assert_close(
        &values(&backend, &trace.output, &context)?,
        &initial_values,
        0.0,
    );
    Ok(())
}

#[test]
fn dpm_adaptive_uses_canonical_ownership_only() -> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(
        workspace()?.join("crates/sim_sampler/src/algorithms/dpm_adaptive_sim_model_0164.rs"),
    )?;
    for forbidden in [
        "struct SamplingPlan",
        "struct AdaptiveSamplingSession",
        "struct CompatibilityRngTransaction",
        "RngStream::new",
        "SamplingTrace {",
        "thread_rng",
        "rand::",
        ".exp_m1()",
        "fn solver_first_intermediate",
        "fn solver_one",
        "fn solver_two_from_intermediate",
        "fn solver_three_from_first",
    ] {
        assert!(
            !source.contains(forbidden),
            "row duplicates owner via {forbidden}"
        );
    }
    for required in [
        "AdaptiveSamplingSession::new",
        "session.next_attempt",
        "session.commit_attempt",
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction",
        "backend.workspace_vec",
        "expected_profile",
        "dpm_solver_first_order",
        "dpm_solver_first_intermediate",
        "dpm_solver_second_order",
        "dpm_solver_third_order",
    ] {
        assert!(
            source.contains(required),
            "missing canonical delegation {required}"
        );
    }
    assert!(!source.contains("CompatibilityRngTransaction::open"));
    assert!(!source.contains("RngCompatibilityRequest::new"));
    Ok(())
}
