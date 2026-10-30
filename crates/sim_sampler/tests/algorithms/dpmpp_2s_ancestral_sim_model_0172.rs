use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity, SamplingProgress,
    SamplingSnrMode,
    generated_dpmpp_2s_ancestral_sim_model_0172::{
        DEFINITION, DPMPP_2S_ANCESTRAL_FEATURE_ID, DPMPP_2S_ANCESTRAL_NOISE_CONTRACT_ID,
        DPMPP_2S_ANCESTRAL_SAMPLER_ID, DPMPP_2S_ANCESTRAL_SOURCE_ORDINAL,
        Dpmpp2sAncestralDenoiserStage, Dpmpp2sAncestralError, Dpmpp2sAncestralMode,
        Dpmpp2sAncestralOptions, sample_dpmpp_2s_ancestral,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, RetryRngPolicy, RngCheckpoint, RngCompatibilityPhase,
    RngCompatibilityRequest, RngExecutionScope, RngGenerationPlacement, RngSeedTransform, StreamId,
    Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_2s_ancestral_comfy_model_0172/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/dpmpp_2s_ancestral_sim_model_0172.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    source: SourceFixture,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    seed: u64,
    shape: Vec<u64>,
    initial: Vec<f32>,
    rng: RngFixture,
    cases: Vec<CaseFixture>,
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
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct RngFixture {
    contract: String,
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    execution_ordinal: u64,
    batch: u64,
    retry: u32,
    retry_policy: String,
    seed_transform: String,
    placement: String,
}

#[derive(Debug, Deserialize)]
struct CaseFixture {
    name: String,
    profile: String,
    prediction: String,
    profile_noise_scale: f32,
    eta: f32,
    noise_scale: f32,
    sigmas: Vec<f32>,
    steps: Vec<StepFixture>,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    sigma: f32,
    next_sigma: f32,
    sigma_down: f32,
    stochastic_coefficient: f32,
    second_sigma: Option<f32>,
    latent_before: Vec<f32>,
    primary_denoised: Vec<f32>,
    second_input: Option<Vec<f32>>,
    second_denoised: Option<Vec<f32>>,
    deterministic: Vec<f32>,
    noise: Option<Vec<f32>>,
    latent_after: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct CallbackRecord {
    progress: SamplingProgress,
    latent: Vec<f32>,
    denoised: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn workspace_root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "workspace root is unavailable".into())
}

fn digest(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
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

fn profile(case: &CaseFixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    let (prediction, snr_mode) = match case.prediction.as_str() {
        "epsilon" => (PredictionInterpretation::Epsilon, SamplingSnrMode::Standard),
        "flow" => (
            PredictionInterpretation::Flow,
            SamplingSnrMode::ConstantFlow { shift: 1.0 },
        ),
        value => return Err(format!("unknown prediction {value:?}").into()),
    };
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new(case.profile.clone())?,
        prediction,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
        snr_mode,
        case.profile_noise_scale,
    )?)
}

fn plan(case: &CaseFixture, identity: &str, seed: u64) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new(case.profile.clone())?,
        seed,
        u32::try_from(case.steps.len())?,
        1.0,
        1.0,
    )?)
}

fn noise_request(fixture: &Fixture) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        &fixture.rng.workflow,
        &fixture.rng.attempt,
        &fixture.rng.node,
        fixture.rng.output,
        fixture.rng.execution_ordinal,
        fixture.rng.batch,
        fixture.rng.retry,
        RetryRngPolicy::Replay,
    )
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
    for (element, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "element {element}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

fn assert_scalar(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

#[test]
fn val_sampler_001_dpmpp_2s_ancestral_definition_source_and_owner_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2S_ANCESTRAL_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2S_ANCESTRAL_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_2S_ANCESTRAL_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 13);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2s_ancestral_comfy_model_0172"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2S_ANCESTRAL_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        registry
            .resolve(&SamplerIdentity::new("dpmpp_2s_ancestral_alias")?)
            .is_err()
    );

    let contract = rng_compatibility_contract(DPMPP_2S_ANCESTRAL_NOISE_CONTRACT_ID)
        .ok_or("canonical normal-noise contract is unavailable")?;
    assert_eq!(
        contract.phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(fixture.rng.contract, DPMPP_2S_ANCESTRAL_NOISE_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.seed_transform, "add-one-on-cpu");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");

    let root = workspace_root()?;
    assert_eq!(
        digest(&root.join(&fixture.source.sampling_path))?,
        fixture.source.sampling_sha256
    );
    assert_eq!(
        digest(&root.join(&fixture.source.samplers_path))?,
        fixture.source.samplers_sha256
    );
    assert_eq!(
        digest(&root.join(&fixture.source.catalog_path))?,
        fixture.source.catalog_sha256
    );
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let standard = source_range(&sampling, fixture.source.standard_lines);
    let flow = source_range(&sampling, fixture.source.flow_lines);
    let ancestral = source_range(&sampling, fixture.source.ancestral_lines);
    let noise = source_range(&sampling, fixture.source.noise_lines);
    for fragment in [
        "sample_dpmpp_2s_ancestral",
        "get_ancestral_step",
        "callback",
        "sigma_fn",
        "denoised_2",
        "noise_sampler",
    ] {
        assert!(
            standard.contains(fragment),
            "missing standard source {fragment}"
        );
    }
    for fragment in [
        "sample_dpmpp_2s_ancestral_RF",
        "downstep_ratio",
        "renoise_coeff",
        "sigma_s_i_ratio",
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
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains(DPMPP_2S_ANCESTRAL_SAMPLER_ID))
    );
    let registry_literal = samplers
        .split_once("KSAMPLER_NAMES = [")
        .and_then(|(_, tail)| tail.split_once(']'))
        .map(|(literal, _)| literal)
        .ok_or("KSAMPLER_NAMES literal is unavailable")?;
    let registry_names = registry_literal
        .split('"')
        .skip(1)
        .step_by(2)
        .collect::<Vec<_>>();
    let source_ordinal = registry_names
        .iter()
        .position(|identity| *identity == DPMPP_2S_ANCESTRAL_SAMPLER_ID)
        .ok_or("sampler is absent from KSAMPLER_NAMES")?;
    assert_eq!(
        u16::try_from(source_ordinal)?,
        DPMPP_2S_ANCESTRAL_SOURCE_ORDINAL
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    let catalog_line = catalog
        .lines()
        .nth(fixture.source.catalog_line - 1)
        .ok_or("catalog line is unavailable")?;
    assert!(catalog_line.contains(DPMPP_2S_ANCESTRAL_SAMPLER_ID));
    assert!(catalog_line.contains(DPMPP_2S_ANCESTRAL_FEATURE_ID));

    assert!(IMPLEMENTATION.contains("noise_request.open_transaction("));
    for forbidden in [
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream::",
        "::open(",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "row bypasses its canonical owner with {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2s_ancestral_matches_both_equation_branches_and_rng()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    for case in &fixture.cases {
        verify_fixture_equations(case, fixture.tolerance);
        let expected_mode = match case.name.as_str() {
            "standard" => Dpmpp2sAncestralMode::Standard,
            "rectified_flow" => Dpmpp2sAncestralMode::RectifiedFlow,
            value => return Err(format!("unknown fixture case {value:?}").into()),
        };
        let profile = profile(case)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
        let denoiser_calls = RefCell::new(Vec::new());
        let callbacks = RefCell::new(Vec::new());
        let (trace, mode, noise_before, noise_after) = sample_dpmpp_2s_ancestral(
            &backend,
            plan(case, &fixture.identity, fixture.seed)?,
            &profile,
            initial,
            &case.sigmas,
            noise_request(&fixture),
            Dpmpp2sAncestralOptions {
                eta: case.eta,
                noise_scale: case.noise_scale,
            },
            &context,
            |latent, sigma, step, stage| {
                let expected = case
                    .steps
                    .get(step)
                    .ok_or_else(|| "unexpected denoiser step".to_owned())?;
                let (expected_sigma, expected_input, expected_output) = match stage {
                    Dpmpp2sAncestralDenoiserStage::Primary => (
                        expected.sigma,
                        expected.latent_before.as_slice(),
                        expected.primary_denoised.as_slice(),
                    ),
                    Dpmpp2sAncestralDenoiserStage::SecondOrder => (
                        expected
                            .second_sigma
                            .ok_or_else(|| "unexpected second-order call".to_owned())?,
                        expected
                            .second_input
                            .as_deref()
                            .ok_or_else(|| "missing second-order input".to_owned())?,
                        expected
                            .second_denoised
                            .as_deref()
                            .ok_or_else(|| "missing second-order output".to_owned())?,
                    ),
                };
                assert_scalar(sigma, expected_sigma, fixture.tolerance);
                let actual =
                    tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
                assert_close(&actual, expected_input, fixture.tolerance);
                denoiser_calls.borrow_mut().push((step, stage));
                tensor_from_f32(&backend, &fixture.shape, expected_output, &context)
                    .map_err(|error| error.to_string())
            },
            |progress, latent, denoised| {
                callbacks.borrow_mut().push(CallbackRecord {
                    progress: *progress,
                    latent: tensor_to_f32(&backend, latent, &context)
                        .map_err(|error| error.to_string())?
                        .to_vec(),
                    denoised: tensor_to_f32(&backend, denoised, &context)
                        .map_err(|error| error.to_string())?
                        .to_vec(),
                });
                Ok::<_, String>(())
            },
        )?;
        assert_eq!(mode, expected_mode);
        assert_eq!(trace.sigmas, case.sigmas);
        assert_eq!(trace.denoiser_evaluations.len(), case.steps.len());
        assert_eq!(trace.latents.len(), case.steps.len() + 1);
        for (step, expected) in case.steps.iter().enumerate() {
            assert_close(
                &values(&backend, &trace.latents[step], &context)?,
                &expected.latent_before,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, &trace.denoiser_evaluations[step], &context)?,
                &expected.primary_denoised,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, &trace.latents[step + 1], &context)?,
                &expected.latent_after,
                fixture.tolerance,
            );
        }
        let expected_calls = case
            .steps
            .iter()
            .enumerate()
            .flat_map(|(step, expected)| {
                let mut stages = vec![(step, Dpmpp2sAncestralDenoiserStage::Primary)];
                if expected.second_sigma.is_some() {
                    stages.push((step, Dpmpp2sAncestralDenoiserStage::SecondOrder));
                }
                stages
            })
            .collect::<Vec<_>>();
        assert_eq!(denoiser_calls.into_inner(), expected_calls);
        let callbacks = callbacks.into_inner();
        assert_eq!(callbacks.len(), case.steps.len());
        for (step, callback) in callbacks.iter().enumerate() {
            let expected = &case.steps[step];
            assert_eq!(usize::try_from(callback.progress.step)?, step);
            assert_eq!(
                callback.progress.total_steps,
                u32::try_from(case.steps.len())?
            );
            assert_scalar(callback.progress.sigma, expected.sigma, fixture.tolerance);
            assert_scalar(
                callback.progress.next_sigma,
                expected.next_sigma,
                fixture.tolerance,
            );
            assert_close(&callback.latent, &expected.latent_before, fixture.tolerance);
            assert_close(
                &callback.denoised,
                &expected.primary_denoised,
                fixture.tolerance,
            );
        }
        verify_rng(&fixture, case, noise_before, noise_after, &cancellation)?;
    }
    Ok(())
}

fn verify_rng(
    fixture: &Fixture,
    case: &CaseFixture,
    before: RngCheckpoint,
    after: RngCheckpoint,
    cancellation: &CancellationToken,
) -> Result<(), Box<dyn Error>> {
    let mut oracle = CompatibilityRngTransaction::open(
        DPMPP_2S_ANCESTRAL_NOISE_CONTRACT_ID,
        RngCompatibilityRequest::new(
            &fixture.rng.workflow,
            &fixture.rng.attempt,
            &fixture.rng.node,
            fixture.rng.output,
            fixture.rng.execution_ordinal,
            fixture.rng.batch,
            fixture.rng.retry,
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
    assert_eq!(before, oracle.checkpoint());
    for expected in case.steps.iter().filter_map(|step| step.noise.as_ref()) {
        let actual = oracle
            .draw_normal(expected.len(), cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_close(&actual, expected, fixture.tolerance);
    }
    assert_eq!(after, oracle.commit());
    Ok(())
}

fn verify_fixture_equations(case: &CaseFixture, tolerance: f32) {
    for expected in &case.steps {
        if case.name == "standard" {
            let sigma_squared = expected.sigma * expected.sigma;
            let next_squared = expected.next_sigma * expected.next_sigma;
            let sigma_up = expected.next_sigma.min(
                case.eta * (next_squared * (sigma_squared - next_squared) / sigma_squared).sqrt(),
            );
            let sigma_down = (next_squared - sigma_up * sigma_up).sqrt();
            assert_scalar(sigma_down, expected.sigma_down, tolerance);
            assert_scalar(sigma_up, expected.stochastic_coefficient, tolerance);
            if let (Some(second_sigma), Some(second_input), Some(second_denoised)) = (
                expected.second_sigma,
                expected.second_input.as_ref(),
                expected.second_denoised.as_ref(),
            ) {
                let current_time = -expected.sigma.ln();
                let next_time = -expected.sigma_down.ln();
                let step_size = next_time - current_time;
                assert_scalar(
                    (-(current_time + 0.5 * step_size)).exp(),
                    second_sigma,
                    tolerance,
                );
                let input_scale = second_sigma / expected.sigma;
                let denoised_scale = -(-0.5 * step_size).exp_m1();
                let calculated = expected
                    .latent_before
                    .iter()
                    .zip(&expected.primary_denoised)
                    .map(|(latent, denoised)| input_scale * latent + denoised_scale * denoised)
                    .collect::<Vec<_>>();
                assert_close(&calculated, second_input, tolerance);
                let output_scale = expected.sigma_down / expected.sigma;
                let denoised_scale = -(-step_size).exp_m1();
                let calculated = expected
                    .latent_before
                    .iter()
                    .zip(second_denoised)
                    .map(|(latent, denoised)| output_scale * latent + denoised_scale * denoised)
                    .collect::<Vec<_>>();
                assert_close(&calculated, &expected.deterministic, tolerance);
            }
        } else {
            let downstep_ratio = 1.0 + (expected.next_sigma / expected.sigma - 1.0) * case.eta;
            let sigma_down = expected.next_sigma * downstep_ratio;
            let alpha_next = 1.0 - expected.next_sigma;
            let alpha_down = 1.0 - sigma_down;
            let renoise = (expected.next_sigma * expected.next_sigma
                - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down))
                .sqrt();
            assert_scalar(sigma_down, expected.sigma_down, tolerance);
            assert_scalar(renoise, expected.stochastic_coefficient, tolerance);
            if let (Some(second_sigma), Some(second_input), Some(second_denoised)) = (
                expected.second_sigma,
                expected.second_input.as_ref(),
                expected.second_denoised.as_ref(),
            ) {
                let calculated_sigma = if expected.sigma == 1.0 {
                    0.9999
                } else {
                    let current_lambda = ((1.0 - expected.sigma) / expected.sigma).ln();
                    let down_lambda = ((1.0 - sigma_down) / sigma_down).ln();
                    1.0 / ((current_lambda + 0.5 * (down_lambda - current_lambda)).exp() + 1.0)
                };
                assert_scalar(calculated_sigma, second_sigma, tolerance);
                let second_ratio = second_sigma / expected.sigma;
                let calculated = expected
                    .latent_before
                    .iter()
                    .zip(&expected.primary_denoised)
                    .map(|(latent, denoised)| {
                        second_ratio * latent + (1.0 - second_ratio) * denoised
                    })
                    .collect::<Vec<_>>();
                assert_close(&calculated, second_input, tolerance);
                let down_ratio = sigma_down / expected.sigma;
                let calculated = expected
                    .latent_before
                    .iter()
                    .zip(second_denoised)
                    .map(|(latent, denoised)| down_ratio * latent + (1.0 - down_ratio) * denoised)
                    .map(|value| alpha_next / alpha_down * value)
                    .collect::<Vec<_>>();
                assert_close(&calculated, &expected.deterministic, tolerance);
            }
        }
        let calculated = match (&expected.noise, expected.next_sigma > 0.0) {
            (Some(noise), true) => expected
                .deterministic
                .iter()
                .zip(noise)
                .map(|(deterministic, noise)| {
                    deterministic
                        + noise
                            * case.noise_scale
                            * case.profile_noise_scale
                            * expected.stochastic_coefficient
                })
                .collect::<Vec<_>>(),
            _ => expected.deterministic.clone(),
        };
        assert_close(&calculated, &expected.latent_after, tolerance);
    }
}

#[test]
fn val_rng_001_dpmpp_2s_ancestral_is_replay_deterministic() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let case = fixture.cases.first().ok_or("missing standard case")?;
    let profile = profile(case)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let mut runs = Vec::new();
    for _ in 0..2 {
        let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
        let (trace, mode, before, after) = sample_dpmpp_2s_ancestral(
            &backend,
            plan(case, &fixture.identity, fixture.seed)?,
            &profile,
            initial,
            &case.sigmas,
            noise_request(&fixture),
            Dpmpp2sAncestralOptions::default(),
            &context,
            |_, _, step, stage| {
                let expected = &case.steps[step];
                let output = match stage {
                    Dpmpp2sAncestralDenoiserStage::Primary => &expected.primary_denoised,
                    Dpmpp2sAncestralDenoiserStage::SecondOrder => expected
                        .second_denoised
                        .as_ref()
                        .ok_or_else(|| "missing second-order output".to_owned())?,
                };
                tensor_from_f32(&backend, &fixture.shape, output, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<_, String>(()),
        )?;
        let latents = trace
            .latents
            .iter()
            .map(|tensor| values(&backend, tensor, &context))
            .collect::<Result<Vec<_>, _>>()?;
        runs.push((latents, mode, before, after));
    }
    assert_eq!(runs[0], runs[1]);
    Ok(())
}

#[test]
fn val_sampling_foundation_001_dpmpp_2s_ancestral_errors_cancel_and_rollback()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let case = fixture.cases.first().ok_or("missing standard case")?;
    let profile = profile(case)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let invalid = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        initial.clone(),
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions {
            eta: f32::NAN,
            noise_scale: 1.0,
        },
        &context,
        |_, _, _, _| Err("invalid options must fail before denoising".to_owned()),
        |_, _, _| Err::<(), _>("invalid options must fail before callback"),
    )
    .expect_err("non-finite eta must fail closed");
    assert!(matches!(
        invalid,
        Dpmpp2sAncestralError::InvalidOption { name: "eta", .. }
    ));

    let coefficient_callbacks = RefCell::new(Vec::new());
    let invalid_coefficient = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        initial.clone(),
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions {
            eta: -10.0,
            noise_scale: -1.0,
        },
        &context,
        |_, _, step, stage| {
            let expected = &case.steps[step];
            let output = match stage {
                Dpmpp2sAncestralDenoiserStage::Primary => &expected.primary_denoised,
                Dpmpp2sAncestralDenoiserStage::SecondOrder => expected
                    .second_denoised
                    .as_ref()
                    .ok_or_else(|| "missing second-order output".to_owned())?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, _, _| {
            coefficient_callbacks.borrow_mut().push(progress.step);
            Ok::<(), String>(())
        },
    )
    .expect_err("invalid source coefficients must fail before callback publication");
    assert!(matches!(
        invalid_coefficient,
        Dpmpp2sAncestralError::InvalidCoefficient {
            step: 0,
            coefficient: "standard ancestral step",
            ..
        }
    ));
    assert!(coefficient_callbacks.borrow().is_empty());

    let (signed, _, _, _) = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        initial.clone(),
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions {
            eta: -0.5,
            noise_scale: -0.25,
        },
        &context,
        |_, _, step, stage| {
            let expected = &case.steps[step];
            let output = match stage {
                Dpmpp2sAncestralDenoiserStage::Primary => &expected.primary_denoised,
                Dpmpp2sAncestralDenoiserStage::SecondOrder => expected
                    .second_denoised
                    .as_ref()
                    .ok_or_else(|| "missing second-order output".to_owned())?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(signed.latents.len(), case.sigmas.len());

    let wrong = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, "dpmpp_2m", fixture.seed)?,
        &profile,
        initial.clone(),
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions::default(),
        &context,
        |_, _, _, _| Err("wrong identity must fail before denoising".to_owned()),
        |_, _, _| Err::<(), _>("wrong identity must fail before callback"),
    )
    .expect_err("wrong sampler must fail closed");
    assert!(matches!(wrong, Dpmpp2sAncestralError::WrongSampler(_)));

    let callback_error = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        initial.clone(),
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions::default(),
        &context,
        |_, _, step, stage| {
            let expected = &case.steps[step];
            let output = match stage {
                Dpmpp2sAncestralDenoiserStage::Primary => &expected.primary_denoised,
                Dpmpp2sAncestralDenoiserStage::SecondOrder => expected
                    .second_denoised
                    .as_ref()
                    .ok_or_else(|| "missing second-order output".to_owned())?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Err::<(), _>("injected callback failure"),
    )
    .expect_err("callback error must abort before a committed step");
    assert!(matches!(
        callback_error,
        Dpmpp2sAncestralError::Sampling(SamplingError::Callback(reason))
            if reason == "injected callback failure"
    ));

    let second_order_error = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        initial.clone(),
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions::default(),
        &context,
        |_, _, step, stage| match stage {
            Dpmpp2sAncestralDenoiserStage::Primary => tensor_from_f32(
                &backend,
                &fixture.shape,
                &case.steps[step].primary_denoised,
                &context,
            )
            .map_err(|error| error.to_string()),
            Dpmpp2sAncestralDenoiserStage::SecondOrder => {
                Err("injected second-order failure".to_owned())
            }
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("second-order error must abort before commit");
    assert!(matches!(
        second_order_error,
        Dpmpp2sAncestralError::Denoiser {
            step: 0,
            stage: Dpmpp2sAncestralDenoiserStage::SecondOrder,
            reason,
        } if reason == "injected second-order failure"
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancelled_error = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        initial,
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions::default(),
        &cancelled_context,
        |_, _, _, _| Err("cancelled sampler must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("cancelled sampler must not callback"),
    )
    .expect_err("pre-cancelled sampler must fail");
    assert!(matches!(
        cancelled_error,
        Dpmpp2sAncestralError::Tensor(TensorError::Cancelled)
            | Dpmpp2sAncestralError::Sampling(SamplingError::Cancelled)
    ));

    let retry_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (retry, _, before, after) = sample_dpmpp_2s_ancestral(
        &backend,
        plan(case, &fixture.identity, fixture.seed)?,
        &profile,
        retry_initial,
        &case.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralOptions::default(),
        &context,
        |_, _, step, stage| {
            let expected = &case.steps[step];
            let output = match stage {
                Dpmpp2sAncestralDenoiserStage::Primary => &expected.primary_denoised,
                Dpmpp2sAncestralDenoiserStage::SecondOrder => expected
                    .second_denoised
                    .as_ref()
                    .ok_or_else(|| "missing second-order output".to_owned())?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_close(
        &values(
            &backend,
            retry.latents.last().ok_or("missing retry terminal")?,
            &context,
        )?,
        &case.steps.last().ok_or("missing final step")?.latent_after,
        fixture.tolerance,
    );
    verify_rng(&fixture, case, before, after, &cancellation)?;
    Ok(())
}
