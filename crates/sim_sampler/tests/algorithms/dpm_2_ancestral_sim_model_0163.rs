use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation,
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    SamplingProgress, SamplingTrace,
    generated_dpm_2_ancestral_sim_model_0163::{
        DEFINITION, DPM_2_ANCESTRAL_FEATURE_ID, DPM_2_ANCESTRAL_NOISE_CONTRACT_ID,
        DPM_2_ANCESTRAL_SAMPLER_ID, DPM_2_ANCESTRAL_SOURCE_ORDINAL, Dpm2AncestralDenoiserStage,
        Dpm2AncestralError, Dpm2AncestralMode, Dpm2AncestralOptions, sample_dpm_2_ancestral,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, NativeRngExecutionProfile, RetryRngPolicy, RngCheckpoint,
    RngCompatibilityPhase, RngCompatibilityRequest, RngExecutionScope, RngGenerationPlacement,
    RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::PathBuf, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpm_2_ancestral_comfy_model_0163/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/dpm_2_ancestral_sim_model_0163.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    source: SourceFixture,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    seed: u64,
    initial: Vec<f32>,
    primary_denoised: Vec<Vec<f32>>,
    midpoint_denoised: Vec<Vec<f32>>,
    standard: CaseFixture,
    flow: CaseFixture,
    tolerance: f32,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    sampling_path: String,
    sampling_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
    standard_lines: [usize; 2],
    flow_lines: [usize; 2],
    ancestral_lines: [usize; 2],
    noise_lines: [usize; 2],
    registry_line: usize,
    discard_penultimate_line: usize,
}

#[derive(Debug, Deserialize)]
struct CaseFixture {
    profile: String,
    prediction: String,
    eta: f32,
    noise_scale: f32,
    flow_noise_scale: f32,
    sigmas: Vec<f32>,
    latents: Vec<Vec<f32>>,
    steps: Vec<StepFixture>,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    sigma_down: f32,
    stochastic_sigma: f32,
    sigma_mid: Option<f32>,
    primary_derivative: Vec<f32>,
    midpoint_latent: Option<Vec<f32>>,
    midpoint_derivative: Option<Vec<f32>>,
    noise: Option<Vec<f32>>,
    next: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct CallbackObservation {
    step: u32,
    total_steps: u32,
    sigma: f32,
    next_sigma: f32,
    current: Vec<f32>,
    denoised: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn digest(path: &str) -> Result<String, Box<dyn Error>> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(workspace_root()?.join(path))?)
    ))
}

fn source_range(source: &str, range: [usize; 2]) -> String {
    let [start, end] = range;
    source
        .lines()
        .skip(start.saturating_sub(1))
        .take(end.saturating_sub(start) + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

fn execution_context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        cancellation,
    ))
}

fn noise_request() -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "dpm2-ancestral-fixture-workflow",
        "dpm2-ancestral-fixture-attempt",
        "KSampler",
        0,
        0,
        0,
        0,
        RetryRngPolicy::Replay,
    )
}

fn plan(case: &CaseFixture, seed: u64) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        DPM_2_ANCESTRAL_SAMPLER_ID,
        "normal",
        SamplingProfileIdentity::new(case.profile.clone())?,
        seed,
        u32::try_from(case.steps.len()).map_err(|_| "fixture step count overflowed")?,
        1.0,
        1.0,
    )?)
}

fn profile(case: &CaseFixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    let prediction = match case.prediction.as_str() {
        "epsilon" => PredictionInterpretation::Epsilon,
        "flow" => PredictionInterpretation::Flow,
        value => return Err(format!("unknown fixture prediction {value:?}").into()),
    };
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new(case.profile.clone())?,
        prediction,
        Arc::from([0.1_f32, 1.0_f32]),
    )?)
}

fn options(case: &CaseFixture) -> Dpm2AncestralOptions {
    Dpm2AncestralOptions {
        eta: case.eta,
        noise_scale: case.noise_scale,
        flow_noise_scale: case.flow_noise_scale,
    }
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

fn tensor_values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor_to_f32(backend, tensor, context)?.to_vec())
}

type AncestralRun = (
    SamplingTrace,
    Dpm2AncestralMode,
    RngCheckpoint,
    RngCheckpoint,
);

#[test]
fn val_sampler_001_dpm_2_ancestral_definition_and_source_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPM_2_ANCESTRAL_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPM_2_ANCESTRAL_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPM_2_ANCESTRAL_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, DPM_2_ANCESTRAL_SAMPLER_ID);
    assert_eq!(DEFINITION.feature_id, DPM_2_ANCESTRAL_FEATURE_ID);
    assert_eq!(DEFINITION.source_ordinal, 9);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpm_2_ancestral_comfy_model_0163"
    );
    assert!(DEFINITION.stochastic);
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPM_2_ANCESTRAL_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        registry
            .resolve(&SamplerIdentity::new("dpm_2_ancestral_alias")?)
            .is_err()
    );

    let contract = rng_compatibility_contract(DPM_2_ANCESTRAL_NOISE_CONTRACT_ID)
        .ok_or("canonical normal-noise contract is unavailable")?;
    assert_eq!(
        contract.phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(DPM_2_ANCESTRAL_NOISE_CONTRACT_ID, "COMFY-RNG-B35F0F617BFA");

    assert_eq!(
        digest(&fixture.source.sampling_path)?,
        fixture.source.sampling_sha256
    );
    assert_eq!(
        digest(&fixture.source.samplers_path)?,
        fixture.source.samplers_sha256
    );
    assert_eq!(
        digest(&fixture.source.catalog_path)?,
        fixture.source.catalog_sha256
    );
    let sampling = fs::read_to_string(workspace_root()?.join(&fixture.source.sampling_path))?;
    let samplers = fs::read_to_string(workspace_root()?.join(&fixture.source.samplers_path))?;
    let standard = source_range(&sampling, fixture.source.standard_lines);
    let flow = source_range(&sampling, fixture.source.flow_lines);
    let ancestral = source_range(&sampling, fixture.source.ancestral_lines);
    let noise = source_range(&sampling, fixture.source.noise_lines);
    for fragment in [
        "sample_dpm_2_ancestral",
        "get_ancestral_step",
        "callback",
        "sigma_mid",
        "denoised_2",
        "noise_sampler",
        "sigma_up",
    ] {
        assert!(
            standard.contains(fragment),
            "missing standard source {fragment}"
        );
    }
    for fragment in [
        "sample_dpm_2_ancestral_RF",
        "downstep_ratio",
        "renoise_coeff",
        "alpha_ip1/alpha_down",
    ] {
        assert!(flow.contains(fragment), "missing flow source {fragment}");
    }
    for fragment in ["sigma_up", "sigma_down", "eta"] {
        assert!(
            ancestral.contains(fragment),
            "missing ancestral source {fragment}"
        );
    }
    for fragment in ["seed += 1", "torch.Generator", "torch.randn"] {
        assert!(noise.contains(fragment), "missing noise source {fragment}");
    }
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("dpm_2_ancestral"))
    );
    assert!(
        samplers
            .lines()
            .nth(fixture.source.discard_penultimate_line - 1)
            .is_some_and(|line| line.contains("dpm_2_ancestral"))
    );
    Ok(())
}

fn oracle_noise(
    fixture: &Fixture,
    case: &CaseFixture,
    cancellation: &CancellationToken,
) -> Result<(RngCheckpoint, RngCheckpoint), Box<dyn Error>> {
    let mut transaction = CompatibilityRngTransaction::open(
        DPM_2_ANCESTRAL_NOISE_CONTRACT_ID,
        RngCompatibilityRequest::new(
            "dpm2-ancestral-fixture-workflow",
            "dpm2-ancestral-fixture-attempt",
            "KSampler",
            0,
            0,
            0,
            0,
            RetryRngPolicy::Replay,
            i128::from(fixture.seed),
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
            RngExecutionScope::Production,
        ),
        None,
        cancellation,
    )?;
    let before = transaction.checkpoint();
    for expected in case.steps.iter().filter_map(|step| step.noise.as_ref()) {
        let actual = transaction
            .draw_normal(expected.len(), cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_close(&actual, expected, 0.0);
    }
    Ok((before, transaction.commit()))
}

fn assert_case_equations(
    fixture: &Fixture,
    case: &CaseFixture,
    mode: Dpm2AncestralMode,
) -> Result<(), Box<dyn Error>> {
    for (step, expected) in case.steps.iter().enumerate() {
        let sigma = *case.sigmas.get(step).ok_or("missing analytical sigma")?;
        let next_sigma = *case
            .sigmas
            .get(step + 1)
            .ok_or("missing analytical next sigma")?;
        let current = case
            .latents
            .get(step)
            .ok_or("missing analytical current latent")?;
        let primary_denoised = fixture
            .primary_denoised
            .get(step)
            .ok_or("missing analytical primary denoised")?;
        let primary_derivative = current
            .iter()
            .zip(primary_denoised)
            .map(|(current, denoised)| (current - denoised) / sigma)
            .collect::<Vec<_>>();
        assert_close(
            &primary_derivative,
            &expected.primary_derivative,
            fixture.tolerance,
        );

        let (sigma_down, stochastic_sigma) = match mode {
            Dpm2AncestralMode::Standard if case.eta == 0.0 => (next_sigma, 0.0),
            Dpm2AncestralMode::Standard => {
                let sigma_squared = sigma * sigma;
                let next_squared = next_sigma * next_sigma;
                let radicand = next_squared * (sigma_squared - next_squared) / sigma_squared;
                let sigma_up = next_sigma.min(case.eta * radicand.sqrt());
                ((next_squared - sigma_up * sigma_up).sqrt(), sigma_up)
            }
            Dpm2AncestralMode::RectifiedFlow => {
                let downstep_ratio = 1.0 + (next_sigma / sigma - 1.0) * case.eta;
                let sigma_down = next_sigma * downstep_ratio;
                let alpha_next = 1.0 - next_sigma;
                let alpha_down = 1.0 - sigma_down;
                let radicand = next_sigma * next_sigma
                    - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down);
                (sigma_down, radicand.sqrt())
            }
        };
        assert!((sigma_down - expected.sigma_down).abs() <= fixture.tolerance);
        assert!((stochastic_sigma - expected.stochastic_sigma).abs() <= fixture.tolerance);

        if sigma_down == 0.0 {
            assert!(expected.sigma_mid.is_none());
            assert!(expected.midpoint_latent.is_none());
            assert!(expected.midpoint_derivative.is_none());
            assert!(expected.noise.is_none());
            let next = current
                .iter()
                .zip(&primary_derivative)
                .map(|(current, derivative)| current + derivative * (sigma_down - sigma))
                .collect::<Vec<_>>();
            assert_close(&next, &expected.next, fixture.tolerance);
            continue;
        }

        let sigma_mid = (sigma.ln() + (sigma_down.ln() - sigma.ln()) * 0.5).exp();
        assert!(
            expected
                .sigma_mid
                .is_some_and(|expected| (sigma_mid - expected).abs() <= fixture.tolerance)
        );
        let midpoint_latent = current
            .iter()
            .zip(&primary_derivative)
            .map(|(current, derivative)| current + derivative * (sigma_mid - sigma))
            .collect::<Vec<_>>();
        assert_close(
            &midpoint_latent,
            expected
                .midpoint_latent
                .as_deref()
                .ok_or("missing midpoint-latent fixture")?,
            fixture.tolerance,
        );
        let midpoint_denoised = fixture
            .midpoint_denoised
            .get(step)
            .ok_or("missing midpoint-denoised fixture")?;
        let midpoint_derivative = midpoint_latent
            .iter()
            .zip(midpoint_denoised)
            .map(|(latent, denoised)| (latent - denoised) / sigma_mid)
            .collect::<Vec<_>>();
        assert_close(
            &midpoint_derivative,
            expected
                .midpoint_derivative
                .as_deref()
                .ok_or("missing midpoint-derivative fixture")?,
            fixture.tolerance,
        );
        let deterministic = current
            .iter()
            .zip(&midpoint_derivative)
            .map(|(current, derivative)| current + derivative * (sigma_down - sigma))
            .collect::<Vec<_>>();
        let noise = expected.noise.as_ref().ok_or("missing noise fixture")?;
        let deterministic_scale = match mode {
            Dpm2AncestralMode::Standard => 1.0,
            Dpm2AncestralMode::RectifiedFlow => (1.0 - next_sigma) / (1.0 - sigma_down),
        };
        let source_noise_scale = match mode {
            Dpm2AncestralMode::Standard => case.noise_scale,
            Dpm2AncestralMode::RectifiedFlow => case.noise_scale * case.flow_noise_scale,
        };
        let next = deterministic
            .iter()
            .zip(noise)
            .map(|(deterministic, noise)| {
                deterministic_scale * deterministic
                    + (noise * source_noise_scale) * stochastic_sigma
            })
            .collect::<Vec<_>>();
        assert_close(&next, &expected.next, fixture.tolerance);
    }
    Ok(())
}

fn run_case(
    fixture: &Fixture,
    case: &CaseFixture,
    expected_mode: Dpm2AncestralMode,
) -> Result<AncestralRun, Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    let events = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let (trace, mode, noise_before, noise_after) = sample_dpm_2_ancestral(
        &backend,
        plan(case, fixture.seed)?,
        &profile(case)?,
        initial,
        &case.sigmas,
        noise_request(),
        options(case),
        &context,
        |input, sigma, step, stage| {
            events.borrow_mut().push(format!("{stage:?}:{step}"));
            let expected_sigma = match stage {
                Dpm2AncestralDenoiserStage::Primary => case
                    .sigmas
                    .get(step)
                    .copied()
                    .ok_or_else(|| format!("missing primary sigma at step {step}"))?,
                Dpm2AncestralDenoiserStage::Midpoint => case
                    .steps
                    .get(step)
                    .and_then(|step| step.sigma_mid)
                    .ok_or_else(|| format!("missing midpoint sigma at step {step}"))?,
            };
            if (sigma - expected_sigma).abs() > fixture.tolerance {
                return Err(format!("denoiser sigma diverged at step {step}"));
            }
            let expected_input = match stage {
                Dpm2AncestralDenoiserStage::Primary => case
                    .latents
                    .get(step)
                    .ok_or_else(|| format!("missing primary input at step {step}"))?,
                Dpm2AncestralDenoiserStage::Midpoint => case
                    .steps
                    .get(step)
                    .and_then(|step| step.midpoint_latent.as_ref())
                    .ok_or_else(|| format!("missing midpoint input at step {step}"))?,
            };
            let actual =
                tensor_to_f32(&backend, input, &context).map_err(|error| error.to_string())?;
            if actual
                .iter()
                .zip(expected_input.iter())
                .any(|(actual, expected)| (actual - expected).abs() > fixture.tolerance)
            {
                return Err(format!(
                    "denoiser input diverged at step {step}: actual={actual:?}, expected={expected_input:?}"
                ));
            }
            let values = match stage {
                Dpm2AncestralDenoiserStage::Primary => fixture
                    .primary_denoised
                    .get(step)
                    .ok_or_else(|| format!("missing primary denoised at step {step}"))?,
                Dpm2AncestralDenoiserStage::Midpoint => fixture
                    .midpoint_denoised
                    .get(step)
                    .ok_or_else(|| format!("missing midpoint denoised at step {step}"))?,
            };
            tensor_from_f32(&backend, &[4], values, &context).map_err(|error| error.to_string())
        },
        |progress: &SamplingProgress, current, denoised| {
            events
                .borrow_mut()
                .push(format!("Callback:{}", progress.step));
            callbacks.borrow_mut().push(CallbackObservation {
                step: progress.step,
                total_steps: progress.total_steps,
                sigma: progress.sigma,
                next_sigma: progress.next_sigma,
                current: tensor_to_f32(&backend, current, &context)?.to_vec(),
                denoised: tensor_to_f32(&backend, denoised, &context)?.to_vec(),
            });
            Ok::<(), sim_tensor::generated_native_diffusion::NativeDiffusionTensorError>(())
        },
    )?;

    assert_eq!(mode, expected_mode);
    assert_eq!(trace.sigmas, case.sigmas);
    assert_eq!(trace.latents.len(), case.latents.len());
    assert_eq!(
        trace.denoiser_evaluations.len(),
        fixture.primary_denoised.len()
    );
    assert_eq!(
        noise_before.profile,
        NativeRngExecutionProfile::CpuMt19937V1.stream_profile()
    );
    assert_ne!(noise_before, noise_after);
    let (oracle_before, oracle_after) = oracle_noise(fixture, case, &cancellation)?;
    assert_eq!(noise_before, oracle_before);
    assert_eq!(noise_after, oracle_after);
    assert_case_equations(fixture, case, expected_mode)?;
    for (actual, expected) in trace.latents.iter().zip(case.latents.iter()) {
        assert_close(
            &tensor_values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    for (actual, expected) in trace
        .denoiser_evaluations
        .iter()
        .zip(fixture.primary_denoised.iter())
    {
        assert_close(
            &tensor_values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    for (step, expected) in case.steps.iter().enumerate() {
        assert_close(
            &tensor_values(
                &backend,
                trace.latents.get(step + 1).ok_or("missing next latent")?,
                &context,
            )?,
            &expected.next,
            fixture.tolerance,
        );
    }

    let callbacks = callbacks.into_inner();
    assert_eq!(callbacks.len(), case.steps.len());
    for (step, actual) in callbacks.iter().enumerate() {
        let sigma = case.sigmas.get(step).ok_or("missing callback sigma")?;
        let next_sigma = case
            .sigmas
            .get(step + 1)
            .ok_or("missing callback next sigma")?;
        let current = case
            .latents
            .get(step)
            .ok_or("missing callback current latent")?;
        let denoised = fixture
            .primary_denoised
            .get(step)
            .ok_or("missing callback denoised value")?;
        assert_eq!(actual.step, u32::try_from(step)?);
        assert_eq!(actual.total_steps, u32::try_from(case.steps.len())?);
        assert_eq!(actual.sigma, *sigma);
        assert_eq!(actual.next_sigma, *next_sigma);
        assert_close(&actual.current, current, fixture.tolerance);
        assert_close(&actual.denoised, denoised, fixture.tolerance);
    }
    let mut expected_events = Vec::new();
    for step in 0..case.steps.len() {
        expected_events.push(format!("Primary:{step}"));
        expected_events.push(format!("Callback:{step}"));
        if case
            .steps
            .get(step)
            .is_some_and(|step| step.sigma_mid.is_some())
        {
            expected_events.push(format!("Midpoint:{step}"));
        }
    }
    assert_eq!(events.into_inner(), expected_events);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok((trace, mode, noise_before, noise_after))
}

#[test]
fn val_sampler_001_dpm_2_ancestral_matches_standard_and_flow_intermediates()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let standard = run_case(&fixture, &fixture.standard, Dpm2AncestralMode::Standard)?;
    let flow = run_case(&fixture, &fixture.flow, Dpm2AncestralMode::RectifiedFlow)?;
    assert_eq!(standard.2, flow.2);
    assert_eq!(standard.3, flow.3);
    Ok(())
}

#[test]
fn val_rng_001_dpm_2_ancestral_replays_and_eta_zero_still_consumes_noise()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let first = run_case(&fixture, &fixture.standard, Dpm2AncestralMode::Standard)?;
    let replay = run_case(&fixture, &fixture.standard, Dpm2AncestralMode::Standard)?;
    assert_eq!(first.2, replay.2);
    assert_eq!(first.3, replay.3);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let case = &fixture.standard;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    let (trace, mode, noise_before, noise_after) = sample_dpm_2_ancestral(
        &backend,
        plan(case, fixture.seed)?,
        &profile(case)?,
        initial,
        &case.sigmas,
        noise_request(),
        Dpm2AncestralOptions {
            eta: 0.0,
            ..Dpm2AncestralOptions::default()
        },
        &context,
        |input, _, _, _| Ok(input.clone()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(mode, Dpm2AncestralMode::Standard);
    assert_eq!(trace.sigmas, case.sigmas);
    assert_eq!(trace.latents.len(), case.steps.len() + 1);
    let (oracle_before, oracle_after) = oracle_noise(&fixture, case, &cancellation)?;
    assert_eq!(noise_before, oracle_before);
    assert_eq!(noise_after, oracle_after);
    assert_ne!(noise_before, noise_after);
    assert!(case.steps.iter().take(2).all(|step| step.noise.is_some()));
    assert!(case.steps.last().is_some_and(|step| step.noise.is_none()));
    Ok(())
}

#[test]
fn dpm_2_ancestral_failures_are_typed_atomic_and_cancel_before_midpoint_and_rng()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let case = &fixture.standard;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    let wrong_plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::new(case.profile.clone())?,
        fixture.seed,
        3,
        1.0,
        1.0,
    )?;
    assert!(matches!(
        sample_dpm_2_ancestral(
            &backend,
            wrong_plan,
            &profile(case)?,
            initial.clone(),
            &case.sigmas,
            noise_request(),
            options(case),
            &context,
            |input, _, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2AncestralError::WrongSampler(value)) if value == "euler"
    ));
    assert!(matches!(
        sample_dpm_2_ancestral(
            &backend,
            plan(case, fixture.seed)?,
            &profile(case)?,
            initial.clone(),
            &[2.0, 1.25, 1.25, 0.0],
            noise_request(),
            options(case),
            &context,
            |input, _, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2AncestralError::Sampling(SamplingError::InvalidSigma {
            step: 1,
            ..
        }))
    ));
    assert!(matches!(
        sample_dpm_2_ancestral(
            &backend,
            plan(case, fixture.seed)?,
            &profile(case)?,
            initial,
            &case.sigmas,
            noise_request(),
            Dpm2AncestralOptions {
                eta: f32::NAN,
                ..options(case)
            },
            &context,
            |input, _, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2AncestralError::InvalidOption { name: "eta", .. })
    ));

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_initial = tensor_from_f32(&backend, &[4], &fixture.initial, &callback_context)?;
    let events = RefCell::new(Vec::new());
    let result = sample_dpm_2_ancestral(
        &backend,
        plan(case, fixture.seed)?,
        &profile(case)?,
        callback_initial.clone(),
        &case.sigmas,
        noise_request(),
        options(case),
        &callback_context,
        |input, _, step, stage| {
            events.borrow_mut().push(format!("{stage:?}:{step}"));
            Ok(input.clone())
        },
        |_, _, _| {
            events.borrow_mut().push("Callback:0".to_owned());
            callback_cancellation.cancel();
            Ok::<(), String>(())
        },
    );
    assert!(matches!(
        result,
        Err(Dpm2AncestralError::Sampling(SamplingError::Cancelled))
    ));
    assert_eq!(events.into_inner(), ["Primary:0", "Callback:0"]);
    assert_eq!(
        tensor_values(&backend, &callback_initial, &context)?,
        fixture.initial
    );
    assert_eq!(callback_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn dpm_2_ancestral_is_only_an_equation_adapter_over_canonical_owners() {
    for required in [
        "SamplingSession::new",
        "session.observe_step",
        "observation.commit",
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction",
        "tensor_to_f32",
        "tensor_from_f32",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing canonical owner {required}"
        );
    }
    for forbidden in [
        "struct Dpm2AncestralTrace",
        "struct Dpm2AncestralStepTrace",
        "struct SamplingTrace",
        "struct SamplingPlan",
        "struct SamplingSession",
        "struct CancellationToken",
        "struct RngStream",
        "struct TensorDescriptor",
        "OutputCommitter",
        "ExecutionQueue",
        "std::fs",
        "serde_json",
        "Python",
        "Command::new",
        "unwrap(",
        "expect(",
        "panic!(",
        "todo!(",
        "unimplemented!(",
        "CompatibilityRngTransaction::open",
        "RngCompatibilityRequest::new",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "row introduced forbidden owner or construct {forbidden}"
        );
    }
}
