use sim_sampler::{
    BrownianNoiseIntervalAddress, CompatibilityNoiseRequest, DiscreteSamplingProfile,
    PredictionInterpretation, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileIdentity, SamplingProgress, SamplingSnrMode,
    generated_dpmpp_sde_sim_model_0176::{
        DEFINITION, DPMPP_SDE_BROWNIAN_CONTRACT_ID, DPMPP_SDE_FEATURE_ID, DPMPP_SDE_SAMPLER_ID,
        DPMPP_SDE_SOURCE_ORDINAL, DpmppSdeDenoiserStage, DpmppSdeError, DpmppSdeOptions,
        sample_dpmpp_sde,
    },
};
use sim_tensor::{
    BrownianTree, CancellationToken, CompatibilityRngTransaction, CpuBackend,
    CpuWorkspaceAuthority, DeviceId, ExecutionContext, RetryRngPolicy, RngCheckpoint,
    RngCompatibilityPhase, RngCompatibilityRequest, RngExecutionScope, RngGenerationPlacement,
    RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_sde_comfy_model_0176/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/dpmpp_sde_sim_model_0176.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    shape: Vec<u64>,
    initial: Vec<f32>,
    seed: u64,
    eta: f32,
    noise_scale: f32,
    r: f32,
    rng: RngFixture,
    profile: String,
    sigmas: Vec<f32>,
    primary_denoised: Vec<Vec<f32>>,
    intermediate_denoised: Vec<Vec<f32>>,
    steps: Vec<StepFixture>,
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
    equation_lines: [usize; 2],
    ancestral_lines: [usize; 2],
    brownian_lines: [usize; 2],
    registry_lines: [usize; 2],
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
    placement: String,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    lambda_s: Option<f32>,
    lambda_t: Option<f32>,
    h: Option<f32>,
    lambda_s_1: Option<f32>,
    sigma_s_1: Option<f32>,
    alpha_s: Option<f32>,
    alpha_s_1: Option<f32>,
    alpha_t: Option<f32>,
    step_1_sigma_down: Option<f32>,
    step_1_sigma_up: Option<f32>,
    step_1_h: Option<f32>,
    step_1_latent_weight: Option<f32>,
    step_1_denoised_weight: Option<f32>,
    step_1_noise_scale: Option<f32>,
    step_1_deterministic: Option<Vec<f32>>,
    step_1_brownian: Option<Vec<f32>>,
    intermediate_input: Option<Vec<f32>>,
    combined_denoised: Option<Vec<f32>>,
    step_2_sigma_down: Option<f32>,
    step_2_sigma_up: Option<f32>,
    step_2_h: Option<f32>,
    step_2_latent_weight: Option<f32>,
    step_2_denoised_weight: Option<f32>,
    step_2_noise_scale: Option<f32>,
    step_2_deterministic: Option<Vec<f32>>,
    step_2_brownian: Option<Vec<f32>>,
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

fn standard_profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new(fixture.profile.clone())?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
        SamplingSnrMode::Standard,
        1.0,
    )?)
}

fn plan(
    identity: &str,
    profile: &DiscreteSamplingProfile,
    seed: u64,
    steps: usize,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile.identity().clone(),
        seed,
        u32::try_from(steps)?,
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

fn assert_scalar(actual: f32, expected: Option<f32>, tolerance: f32) {
    let expected = expected.expect("fixture coefficient is missing");
    assert!((actual - expected).abs() <= tolerance);
}

#[test]
fn val_sampler_001_dpmpp_sde_definition_source_ordinal_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_SDE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_SDE_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_SDE_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 15);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_sde_comfy_model_0176"
    );
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(DPMPP_SDE_SAMPLER_ID)?)?,
        &DEFINITION
    );
    let contract = rng_compatibility_contract(DPMPP_SDE_BROWNIAN_CONTRACT_ID)
        .ok_or("canonical Brownian contract is unavailable")?;
    assert_eq!(
        contract.phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(fixture.rng.contract, DPMPP_SDE_BROWNIAN_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");

    let root = workspace_root()?;
    for (path, expected) in [
        (
            &fixture.source.sampling_path,
            &fixture.source.sampling_sha256,
        ),
        (
            &fixture.source.samplers_path,
            &fixture.source.samplers_sha256,
        ),
        (&fixture.source.catalog_path, &fixture.source.catalog_sha256),
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "sample_dpmpp_sde",
        "BrownianTreeNoiseSampler",
        "cpu=True",
        "offset_first_sigma_for_snr",
        "lambda_s_1",
        "denoised_2",
        "denoised_d",
        "noise_sampler",
        "callback",
    ] {
        assert!(equations.contains(fragment), "missing source {fragment}");
    }
    let ancestral = source_range(&sampling, fixture.source.ancestral_lines);
    assert!(ancestral.contains("sigma_up"));
    assert!(ancestral.contains("sigma_down"));
    let brownian = source_range(&sampling, fixture.source.brownian_lines);
    assert!(brownian.contains("BatchedBrownianTree"));
    assert!(brownian.contains("BrownianTree"));
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    let registry = source_range(&samplers, fixture.source.registry_lines);
    assert!(registry.contains("\"dpmpp_sde\""));
    assert!(registry.contains("\"dpmpp_sde_gpu\""));
    let names = samplers
        .split_once("KSAMPLER_NAMES = [")
        .and_then(|(_, tail)| tail.split_once(']'))
        .map(|(literal, _)| literal)
        .ok_or("KSAMPLER_NAMES is unavailable")?
        .split('"')
        .skip(1)
        .step_by(2)
        .collect::<Vec<_>>();
    let ordinal = names
        .iter()
        .position(|identity| *identity == DPMPP_SDE_SAMPLER_ID)
        .ok_or("dpmpp_sde is absent")?;
    assert_eq!(u16::try_from(ordinal)?, DPMPP_SDE_SOURCE_ORDINAL);
    assert_eq!(names.get(ordinal + 1), Some(&"dpmpp_sde_gpu"));
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    let catalog_line = catalog
        .lines()
        .nth(fixture.source.catalog_line - 1)
        .ok_or("catalog line is unavailable")?;
    assert!(catalog_line.contains(DPMPP_SDE_SAMPLER_ID));
    assert!(catalog_line.contains(DPMPP_SDE_FEATURE_ID));

    assert!(IMPLEMENTATION.contains("noise_request.open_transaction("));
    assert!(IMPLEMENTATION.contains("pub(crate) fn sample_dpmpp_sde_with_generation_placement"));
    assert!(IMPLEMENTATION.contains("RngGenerationPlacement::CpuSeededTransfer"));
    for forbidden in [
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream::",
        "::open(",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "forbidden owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_sde_matches_every_equation_noise_callback_and_intermediate()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = standard_profile(&fixture)?;
    verify_equations(&fixture, &profile)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let calls = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let (trace, checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(
            &fixture.identity,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        initial,
        &fixture.sigmas,
        DpmppSdeOptions {
            eta: fixture.eta,
            noise_scale: fixture.noise_scale,
            r: fixture.r,
        },
        noise_request(&fixture),
        &context,
        |latent, sigma, step, stage| {
            let expected = &fixture.steps[step];
            let (expected_sigma, expected_input, expected_output) = match stage {
                DpmppSdeDenoiserStage::Primary => (
                    expected.sigma,
                    expected.latent_before.as_slice(),
                    fixture.primary_denoised[step].as_slice(),
                ),
                DpmppSdeDenoiserStage::Intermediate => (
                    expected
                        .sigma_s_1
                        .ok_or_else(|| "unexpected intermediate call".to_owned())?,
                    expected
                        .intermediate_input
                        .as_deref()
                        .ok_or_else(|| "missing intermediate input".to_owned())?,
                    fixture.intermediate_denoised[step].as_slice(),
                ),
            };
            assert_scalar(sigma, Some(expected_sigma), fixture.tolerance);
            let actual =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&actual, expected_input, fixture.tolerance);
            calls.borrow_mut().push((step, stage));
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
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_close(
            &values(&backend, &trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.denoiser_evaluations[step], &context)?,
            &fixture.primary_denoised[step],
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
    }
    let expected_calls = fixture
        .steps
        .iter()
        .enumerate()
        .flat_map(|(step, expected)| {
            let mut stages = vec![(step, DpmppSdeDenoiserStage::Primary)];
            if expected.intermediate_input.is_some() {
                stages.push((step, DpmppSdeDenoiserStage::Intermediate));
            }
            stages
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.into_inner(), expected_calls);
    let callbacks = callbacks.into_inner();
    assert_eq!(callbacks.len(), fixture.steps.len());
    for (step, callback) in callbacks.iter().enumerate() {
        assert_eq!(usize::try_from(callback.progress.step)?, step);
        assert_close(
            &callback.latent,
            &fixture.steps[step].latent_before,
            fixture.tolerance,
        );
        assert_close(
            &callback.denoised,
            &fixture.primary_denoised[step],
            fixture.tolerance,
        );
    }
    verify_rng(
        &fixture,
        checkpoints.ok_or("missing Brownian checkpoints")?,
        &cancellation,
    )?;
    Ok(())
}

fn verify_equations(
    fixture: &Fixture,
    profile: &impl SamplingProfile,
) -> Result<(), Box<dyn Error>> {
    let effective_noise = profile.scale_sampler_noise(fixture.noise_scale)?;
    for (step, expected) in fixture.steps.iter().enumerate() {
        if expected.next_sigma == 0.0 {
            assert_close(
                &expected.latent_after,
                &fixture.primary_denoised[step],
                fixture.tolerance,
            );
            continue;
        }
        let lambda_s = profile.half_log_snr(expected.sigma)?;
        let lambda_t = profile.half_log_snr(expected.next_sigma)?;
        let h = lambda_t - lambda_s;
        let lambda_s_1 = lambda_s + fixture.r * h;
        let sigma_s_1 = profile.sigma_from_half_log_snr(lambda_s_1)?;
        let alpha_s = expected.sigma * lambda_s.exp();
        let alpha_s_1 = sigma_s_1 * lambda_s_1.exp();
        let alpha_t = expected.next_sigma * lambda_t.exp();
        for (actual, pinned) in [
            (lambda_s, expected.lambda_s),
            (lambda_t, expected.lambda_t),
            (h, expected.h),
            (lambda_s_1, expected.lambda_s_1),
            (sigma_s_1, expected.sigma_s_1),
            (alpha_s, expected.alpha_s),
            (alpha_s_1, expected.alpha_s_1),
            (alpha_t, expected.alpha_t),
        ] {
            assert_scalar(actual, pinned, fixture.tolerance);
        }
        let (down_1, up_1) = ancestral_step((-lambda_s).exp(), (-lambda_s_1).exp(), fixture.eta);
        let adjusted_h_1 = -down_1.ln() - lambda_s;
        let latent_weight_1 = alpha_s_1 / alpha_s * (-adjusted_h_1).exp();
        let denoised_weight_1 = -alpha_s_1 * (-adjusted_h_1).exp_m1();
        let noise_scale_1 = alpha_s_1 * effective_noise * up_1;
        for (actual, pinned) in [
            (down_1, expected.step_1_sigma_down),
            (up_1, expected.step_1_sigma_up),
            (adjusted_h_1, expected.step_1_h),
            (latent_weight_1, expected.step_1_latent_weight),
            (denoised_weight_1, expected.step_1_denoised_weight),
            (noise_scale_1, expected.step_1_noise_scale),
        ] {
            assert_scalar(actual, pinned, fixture.tolerance);
        }
        let deterministic_1 = expected
            .latent_before
            .iter()
            .zip(&fixture.primary_denoised[step])
            .map(|(latent, denoised)| latent_weight_1 * latent + denoised_weight_1 * denoised)
            .collect::<Vec<_>>();
        assert_close(
            &deterministic_1,
            expected
                .step_1_deterministic
                .as_deref()
                .ok_or("missing step 1")?,
            fixture.tolerance,
        );
        let intermediate_input = deterministic_1
            .iter()
            .zip(
                expected
                    .step_1_brownian
                    .as_deref()
                    .ok_or("missing noise 1")?,
            )
            .map(|(deterministic, noise)| deterministic + noise_scale_1 * noise)
            .collect::<Vec<_>>();
        assert_close(
            &intermediate_input,
            expected
                .intermediate_input
                .as_deref()
                .ok_or("missing input")?,
            fixture.tolerance,
        );

        let fac = 1.0 / (2.0 * fixture.r);
        let combined = fixture.primary_denoised[step]
            .iter()
            .zip(&fixture.intermediate_denoised[step])
            .map(|(primary, intermediate)| (1.0 - fac) * primary + fac * intermediate)
            .collect::<Vec<_>>();
        assert_close(
            &combined,
            expected
                .combined_denoised
                .as_deref()
                .ok_or("missing combination")?,
            fixture.tolerance,
        );
        let (down_2, up_2) = ancestral_step((-lambda_s).exp(), (-lambda_t).exp(), fixture.eta);
        let adjusted_h_2 = -down_2.ln() - lambda_s;
        let latent_weight_2 = alpha_t / alpha_s * (-adjusted_h_2).exp();
        let denoised_weight_2 = -alpha_t * (-adjusted_h_2).exp_m1();
        let noise_scale_2 = alpha_t * effective_noise * up_2;
        for (actual, pinned) in [
            (down_2, expected.step_2_sigma_down),
            (up_2, expected.step_2_sigma_up),
            (adjusted_h_2, expected.step_2_h),
            (latent_weight_2, expected.step_2_latent_weight),
            (denoised_weight_2, expected.step_2_denoised_weight),
            (noise_scale_2, expected.step_2_noise_scale),
        ] {
            assert_scalar(actual, pinned, fixture.tolerance);
        }
        let deterministic_2 = expected
            .latent_before
            .iter()
            .zip(&combined)
            .map(|(latent, denoised)| latent_weight_2 * latent + denoised_weight_2 * denoised)
            .collect::<Vec<_>>();
        assert_close(
            &deterministic_2,
            expected
                .step_2_deterministic
                .as_deref()
                .ok_or("missing step 2")?,
            fixture.tolerance,
        );
        let next = deterministic_2
            .iter()
            .zip(
                expected
                    .step_2_brownian
                    .as_deref()
                    .ok_or("missing noise 2")?,
            )
            .map(|(deterministic, noise)| deterministic + noise_scale_2 * noise)
            .collect::<Vec<_>>();
        assert_close(&next, &expected.latent_after, fixture.tolerance);
    }
    Ok(())
}

fn ancestral_step(from: f32, to: f32, eta: f32) -> (f32, f32) {
    let from_squared = from * from;
    let to_squared = to * to;
    let up = to.min(eta * (to_squared * (from_squared - to_squared) / from_squared).sqrt());
    ((to_squared - up * up).sqrt(), up)
}

fn verify_rng(
    fixture: &Fixture,
    checkpoints: (RngCheckpoint, RngCheckpoint),
    cancellation: &CancellationToken,
) -> Result<(), Box<dyn Error>> {
    let mut transaction = CompatibilityRngTransaction::open(
        DPMPP_SDE_BROWNIAN_CONTRACT_ID,
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
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
            RngExecutionScope::Production,
        ),
        None,
        cancellation,
    )?;
    assert_eq!(checkpoints.0, transaction.checkpoint());
    let mut tree =
        transaction.brownian_tree(0.5, vec![0.0; fixture.initial.len()], 2.0, cancellation)?;
    for (step, expected) in fixture.steps.iter().enumerate() {
        if expected.next_sigma == 0.0 {
            continue;
        }
        let first = normalized_increment(
            &mut tree,
            expected.sigma,
            expected.sigma_s_1.ok_or("missing intermediate sigma")?,
            step,
            cancellation,
        )?;
        assert_close(
            &first,
            expected
                .step_1_brownian
                .as_deref()
                .ok_or("missing noise 1")?,
            fixture.tolerance,
        );
        let second = normalized_increment(
            &mut tree,
            expected.sigma,
            expected.next_sigma,
            step,
            cancellation,
        )?;
        assert_close(
            &second,
            expected
                .step_2_brownian
                .as_deref()
                .ok_or("missing noise 2")?,
            fixture.tolerance,
        );
    }
    assert_eq!(checkpoints.1, transaction.commit());
    Ok(())
}

fn normalized_increment(
    tree: &mut BrownianTree,
    start: f32,
    end: f32,
    step: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let address = BrownianNoiseIntervalAddress::new(start, end, u32::try_from(step)?)?;
    let (lower, upper) = address.canonical_interval();
    let sign = if address.reverse { -1.0_f64 } else { 1.0 };
    let normalization = f64::from(upper - lower).sqrt();
    Ok(tree
        .increment(f64::from(lower), f64::from(upper), cancellation)?
        .into_iter()
        .map(|value| (sign * value / normalization) as f32)
        .collect())
}

#[test]
fn val_sampling_foundation_001_dpmpp_sde_flow_short_errors_cancel_and_retry_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let standard = standard_profile(&fixture)?;
    let flow = DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("dpmpp-sde-flow-v1")?,
        PredictionInterpretation::Flow,
        Arc::from([0.01_f32, 0.2, 0.5, 0.8]),
        SamplingSnrMode::ConstantFlow { shift: 1.0 },
        1.25,
    )?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;

    let (short, checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(DPMPP_SDE_SAMPLER_ID, &standard, fixture.seed, 1)?,
        &standard,
        initial.clone(),
        &[2.0],
        DpmppSdeOptions::default(),
        noise_request(&fixture),
        &context,
        |_, _, _, _| Err("short schedule must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("short schedule must not callback"),
    )?;
    assert!(checkpoints.is_none());
    assert_eq!(short.latents.len(), 1);

    let (short_with_non_finite_options, checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(DPMPP_SDE_SAMPLER_ID, &standard, fixture.seed, 1)?,
        &standard,
        initial.clone(),
        &[2.0],
        DpmppSdeOptions {
            eta: f32::NAN,
            noise_scale: f32::NAN,
            r: f32::NAN,
        },
        noise_request(&fixture),
        &context,
        |_, _, _, _| Err("short schedule must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("short schedule must not callback"),
    )?;
    assert!(checkpoints.is_none());
    assert_eq!(short_with_non_finite_options.latents.len(), 1);

    let zero_r_callbacks = RefCell::new(0_usize);
    let invalid = sample_dpmpp_sde(
        &backend,
        plan(
            DPMPP_SDE_SAMPLER_ID,
            &standard,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &standard,
        initial.clone(),
        &fixture.sigmas,
        DpmppSdeOptions {
            r: 0.0,
            ..DpmppSdeOptions::default()
        },
        noise_request(&fixture),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| {
            *zero_r_callbacks.borrow_mut() += 1;
            Ok::<(), String>(())
        },
    )
    .expect_err("zero r must fail at the source equation boundary");
    assert!(matches!(
        invalid,
        DpmppSdeError::InvalidCoefficient {
            step: 0,
            coefficient: "combination factor",
            ..
        }
    ));
    assert_eq!(*zero_r_callbacks.borrow(), 1);

    let (signed_noise_trace, signed_noise_checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(
            DPMPP_SDE_SAMPLER_ID,
            &standard,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &standard,
        initial.clone(),
        &fixture.sigmas,
        DpmppSdeOptions {
            eta: 0.0,
            noise_scale: -0.5,
            r: 0.5,
        },
        noise_request(&fixture),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(signed_noise_trace.latents.len(), fixture.sigmas.len());
    assert!(signed_noise_checkpoints.is_some());

    let signed_r_error = sample_dpmpp_sde(
        &backend,
        plan(
            DPMPP_SDE_SAMPLER_ID,
            &standard,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &standard,
        initial.clone(),
        &fixture.sigmas,
        DpmppSdeOptions {
            eta: 0.0,
            noise_scale: -0.5,
            r: -0.5,
        },
        noise_request(&fixture),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<(), String>(()),
    )
    .expect_err("negative r must reach source equation validation");
    assert!(matches!(
        signed_r_error,
        DpmppSdeError::InvalidCoefficient {
            step: 0,
            coefficient: "first adjusted step",
            ..
        }
    ));

    let callback_error = sample_dpmpp_sde(
        &backend,
        plan(
            DPMPP_SDE_SAMPLER_ID,
            &standard,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &standard,
        initial.clone(),
        &fixture.sigmas,
        DpmppSdeOptions {
            eta: fixture.eta,
            noise_scale: fixture.noise_scale,
            r: fixture.r,
        },
        noise_request(&fixture),
        &context,
        |_, _, step, stage| {
            let output = match stage {
                DpmppSdeDenoiserStage::Primary => &fixture.primary_denoised[step],
                DpmppSdeDenoiserStage::Intermediate => &fixture.intermediate_denoised[step],
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Err::<(), _>("injected callback failure"),
    )
    .expect_err("callback failure must abort");
    assert!(matches!(
        callback_error,
        DpmppSdeError::Sampling(SamplingError::Callback(reason))
            if reason == "injected callback failure"
    ));

    let retry_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (retry_trace, retry_checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(
            DPMPP_SDE_SAMPLER_ID,
            &standard,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &standard,
        retry_initial,
        &fixture.sigmas,
        DpmppSdeOptions {
            eta: fixture.eta,
            noise_scale: fixture.noise_scale,
            r: fixture.r,
        },
        noise_request(&fixture),
        &context,
        |_, _, step, stage| {
            let output = match stage {
                DpmppSdeDenoiserStage::Primary => &fixture.primary_denoised[step],
                DpmppSdeDenoiserStage::Intermediate => &fixture.intermediate_denoised[step],
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_close(
        &values(
            &backend,
            retry_trace
                .latents
                .last()
                .ok_or("retry trace has no terminal latent")?,
            &context,
        )?,
        &fixture
            .steps
            .last()
            .ok_or("fixture has no terminal step")?
            .latent_after,
        fixture.tolerance,
    );
    verify_rng(
        &fixture,
        retry_checkpoints.ok_or("retry is missing Brownian checkpoints")?,
        &cancellation,
    )?;

    let flow_sigmas = [0.8_f32, 0.5, 0.0];
    let flow_primary = [[0.32_f32, -0.01, 0.7, -0.41], [0.1, 0.0, 0.2, -0.1]];
    let flow_intermediate = [0.2_f32, 0.1, 0.55, -0.25];
    let expected_flow = deterministic_step(
        &fixture.initial,
        &flow_primary[0],
        &flow_intermediate,
        flow_sigmas[0],
        flow_sigmas[1],
        fixture.r,
        &flow,
    )?;
    let expected_flow_intermediate = deterministic_intermediate(
        &fixture.initial,
        &flow_primary[0],
        flow_sigmas[0],
        flow_sigmas[1],
        fixture.r,
        &flow,
    )?;
    let flow_lambda_source = flow.half_log_snr(flow_sigmas[0])?;
    let flow_lambda_target = flow.half_log_snr(flow_sigmas[1])?;
    let expected_flow_intermediate_sigma = flow.sigma_from_half_log_snr(
        flow_lambda_source + fixture.r * (flow_lambda_target - flow_lambda_source),
    )?;
    let flow_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (flow_trace, flow_checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(DPMPP_SDE_SAMPLER_ID, &flow, fixture.seed, 2)?,
        &flow,
        flow_initial,
        &flow_sigmas,
        DpmppSdeOptions {
            eta: 0.0,
            noise_scale: 0.0,
            r: fixture.r,
        },
        noise_request(&fixture),
        &context,
        |latent, sigma, step, stage| {
            if step == 0 && stage == DpmppSdeDenoiserStage::Intermediate {
                let actual =
                    tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
                assert_close(&actual, &expected_flow_intermediate, fixture.tolerance);
                assert_scalar(
                    sigma,
                    Some(expected_flow_intermediate_sigma),
                    fixture.tolerance,
                );
            }
            let output = match (step, stage) {
                (0, DpmppSdeDenoiserStage::Primary) => flow_primary[0].as_slice(),
                (0, DpmppSdeDenoiserStage::Intermediate) => flow_intermediate.as_slice(),
                (1, DpmppSdeDenoiserStage::Primary) => flow_primary[1].as_slice(),
                _ => return Err("unexpected flow denoiser call".to_owned()),
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert!(flow_checkpoints.is_some());
    assert_close(
        &values(&backend, &flow_trace.latents[1], &context)?,
        &expected_flow,
        fixture.tolerance,
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancelled_error = sample_dpmpp_sde(
        &backend,
        plan(
            DPMPP_SDE_SAMPLER_ID,
            &standard,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &standard,
        initial,
        &fixture.sigmas,
        DpmppSdeOptions::default(),
        noise_request(&fixture),
        &cancelled_context,
        |_, _, _, _| Err("cancelled sampler must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("cancelled sampler must not callback"),
    )
    .expect_err("pre-cancelled sampler must fail");
    assert!(matches!(
        cancelled_error,
        DpmppSdeError::Tensor(TensorError::Cancelled)
    ));
    Ok(())
}

fn deterministic_intermediate(
    current: &[f32],
    denoised: &[f32],
    sigma: f32,
    next_sigma: f32,
    r: f32,
    profile: &impl SamplingProfile,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let lambda_s = profile.half_log_snr(sigma)?;
    let lambda_t = profile.half_log_snr(next_sigma)?;
    let lambda_s_1 = lambda_s + r * (lambda_t - lambda_s);
    let sigma_s_1 = profile.sigma_from_half_log_snr(lambda_s_1)?;
    let alpha_s = sigma * lambda_s.exp();
    let alpha_s_1 = sigma_s_1 * lambda_s_1.exp();
    let h = lambda_s_1 - lambda_s;
    Ok(current
        .iter()
        .zip(denoised)
        .map(|(latent, denoised)| {
            alpha_s_1 / alpha_s * (-h).exp() * latent - alpha_s_1 * (-h).exp_m1() * denoised
        })
        .collect())
}

fn deterministic_step(
    current: &[f32],
    primary: &[f32],
    intermediate: &[f32],
    sigma: f32,
    next_sigma: f32,
    r: f32,
    profile: &impl SamplingProfile,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let lambda_s = profile.half_log_snr(sigma)?;
    let lambda_t = profile.half_log_snr(next_sigma)?;
    let alpha_s = sigma * lambda_s.exp();
    let alpha_t = next_sigma * lambda_t.exp();
    let h = lambda_t - lambda_s;
    let fac = 1.0 / (2.0 * r);
    Ok(current
        .iter()
        .zip(primary.iter().zip(intermediate))
        .map(|(latent, (primary, intermediate))| {
            let combined = (1.0 - fac) * primary + fac * intermediate;
            alpha_t / alpha_s * (-h).exp() * latent - alpha_t * (-h).exp_m1() * combined
        })
        .collect())
}
