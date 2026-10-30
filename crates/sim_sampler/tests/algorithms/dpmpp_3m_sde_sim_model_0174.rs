use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_dpmpp_3m_sde_sim_model_0174::{
        DEFINITION, DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID, DPMPP_3M_SDE_FEATURE_ID,
        DPMPP_3M_SDE_SAMPLER_ID, DPMPP_3M_SDE_SOURCE_ORDINAL, Dpmpp3mSdeError, Dpmpp3mSdeOptions,
        sample_dpmpp_3m_sde,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngGenerationPlacement, RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_3m_sde_comfy_model_0174/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/dpmpp_3m_sde_sim_model_0174.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    rng_contract_id: String,
    source: SourceFixture,
    seed: u64,
    eta: f32,
    noise_scale: f32,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    adjusted_sigmas: Vec<f32>,
    raw_brownian_bounds: Vec<f32>,
    initial: Vec<f32>,
    rng: RngFixture,
    steps: Vec<StepFixture>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    sampling_path: String,
    sampling_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
    equation_lines: Vec<usize>,
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct RngFixture {
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
    step: usize,
    order: u8,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    lambda_source: Option<f32>,
    lambda_target: Option<f32>,
    step_size: Option<f32>,
    eta_step_size: Option<f32>,
    alpha_target: Option<f32>,
    latent_weight: Option<f32>,
    denoised_weight: Option<f32>,
    ratio_0: Option<f32>,
    ratio_1: Option<f32>,
    phi_2: Option<f32>,
    phi_3: Option<f32>,
    brownian_noise: Vec<f32>,
    stochastic_scale: Option<f32>,
    latent_after: Vec<f32>,
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("dpmpp-3m-sde-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
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

fn raw_bounds_noise_request() -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "dpmpp-3m-sde-raw-bounds-v1",
        "attempt-raw-bounds",
        "KSampler-23",
        23,
        175,
        0,
        0,
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
            "element {element}: expected {expected}, got {actual}"
        );
    }
}

fn assert_optional(actual: Option<f32>, expected: Option<f32>, tolerance: f32) {
    assert_eq!(actual.is_some(), expected.is_some());
    if let (Some(actual), Some(expected)) = (actual, expected) {
        assert!((actual - expected).abs() <= tolerance);
    }
}

#[test]
fn val_sampler_001_dpmpp_3m_sde_definition_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_3M_SDE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_3M_SDE_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_3M_SDE_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_3m_sde_comfy_model_0174"
    );
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(DPMPP_3M_SDE_SAMPLER_ID)?)?,
        &DEFINITION
    );

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
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_dpmpp_3m_sde(",
        "cpu=True",
        "denoised_1, denoised_2 = None, None",
        "h, h_1, h_2 = None, None, None",
        "if h_2 is not None:",
        "d1_0 = (denoised - denoised_1) / r0",
        "d1_1 = (denoised_1 - denoised_2) / r1",
        "phi_2 = h_eta.neg().expm1() / h_eta + 1",
        "phi_3 = phi_2 / h_eta - 0.5",
        "elif h_1 is not None:",
    ] {
        assert!(
            equations.contains(fragment),
            "missing source fragment {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_3m_sde\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,dpmpp_3m_sde,")
                && line.ends_with(",COMFY-MODEL-0174"))
    );
    assert!(IMPLEMENTATION.contains("pub(crate) fn sample_dpmpp_3m_sde_with_generation_placement"));
    assert!(IMPLEMENTATION.contains("CompatibilityNoiseRequest"));
    assert!(IMPLEMENTATION.contains("RngGenerationPlacement::CpuSeededTransfer"));
    for forbidden in [
        "struct Dpmpp3mSdeTrace",
        "struct Dpmpp3mSdeProgress",
        "struct Dpmpp3mSdeObservation",
        "struct Dpmpp3mSdeNoiseRequest",
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream",
        "struct CancellationToken",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_3m_sde_matches_every_order_intermediate_callback_and_rng()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let mut adjusted_sigmas = fixture.sigmas.clone();
    profile.adjust_first_sigma_for_snr(&mut adjusted_sigmas)?;
    assert_eq!(adjusted_sigmas, fixture.adjusted_sigmas);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let callbacks = RefCell::new(Vec::new());
    let (trace, checkpoints) = sample_dpmpp_3m_sde(
        &backend,
        plan(
            &fixture.identity,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        Dpmpp3mSdeOptions::new(fixture.eta, fixture.noise_scale)?,
        noise_request(&fixture),
        &context,
        |latent, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| "unexpected denoiser step".to_owned())?;
            assert!((sigma - expected.sigma).abs() <= fixture.tolerance);
            let latent =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&latent, &expected.latent_before, fixture.tolerance);
            let biases = [0.07_f32, -0.03, 0.05];
            let denoised = latent
                .iter()
                .zip(biases)
                .map(|(value, bias)| 0.61 * value + sigma * bias + step as f32 * 0.015)
                .collect::<Vec<_>>();
            assert_close(&denoised, &expected.denoised, fixture.tolerance);
            tensor_from_f32(&backend, &fixture.shape, &denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            callbacks.borrow_mut().push((
                *progress,
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<_, String>(())
        },
    )?;

    assert_eq!(trace.sigmas, fixture.adjusted_sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    let mut step_size_1: Option<f32> = None;
    let mut step_size_2: Option<f32> = None;
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_close(
            &values(&backend, &trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.denoiser_evaluations[step], &context)?,
            &expected.denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
        if expected.next_sigma == 0.0 {
            assert_eq!(expected.order, 0);
            assert_close(
                &expected.latent_after,
                &expected.denoised,
                fixture.tolerance,
            );
            continue;
        }

        let lambda_source = profile.half_log_snr(expected.sigma)?;
        let lambda_target = profile.half_log_snr(expected.next_sigma)?;
        let step_size = lambda_target - lambda_source;
        let eta_step_size = step_size * (fixture.eta + 1.0);
        let alpha_target = expected.next_sigma * lambda_target.exp();
        let latent_weight = expected.next_sigma / expected.sigma * (-step_size * fixture.eta).exp();
        let denoised_weight = alpha_target * -(-eta_step_size).exp_m1();
        let phi_2 = (-eta_step_size).exp_m1() / eta_step_size + 1.0;
        let phi_3 = phi_2 / eta_step_size - 0.5;
        let ratio_0 = step_size_1.map(|value| value / step_size);
        let ratio_1 = step_size_2.map(|value| value / step_size);
        let stochastic_scale = expected.next_sigma
            * (-(-2.0 * step_size * fixture.eta).exp_m1()).sqrt()
            * profile.scale_sampler_noise(fixture.noise_scale)?;
        assert_optional(
            Some(lambda_source),
            expected.lambda_source,
            fixture.tolerance,
        );
        assert_optional(
            Some(lambda_target),
            expected.lambda_target,
            fixture.tolerance,
        );
        assert_optional(Some(step_size), expected.step_size, fixture.tolerance);
        assert_optional(
            Some(eta_step_size),
            expected.eta_step_size,
            fixture.tolerance,
        );
        assert_optional(Some(alpha_target), expected.alpha_target, fixture.tolerance);
        assert_optional(
            Some(latent_weight),
            expected.latent_weight,
            fixture.tolerance,
        );
        assert_optional(
            Some(denoised_weight),
            expected.denoised_weight,
            fixture.tolerance,
        );
        assert_optional(ratio_0, expected.ratio_0, fixture.tolerance);
        assert_optional(ratio_1, expected.ratio_1, fixture.tolerance);
        assert_optional(Some(phi_2), expected.phi_2, fixture.tolerance);
        assert_optional(Some(phi_3), expected.phi_3, fixture.tolerance);
        assert_optional(
            Some(stochastic_scale),
            expected.stochastic_scale,
            fixture.tolerance,
        );

        let reconstructed = expected
            .latent_before
            .iter()
            .zip(&expected.denoised)
            .zip(&expected.brownian_noise)
            .enumerate()
            .map(|(element, ((latent, denoised), noise))| {
                let correction = match (step_size_1, step_size_2) {
                    (None, None) => {
                        assert_eq!(expected.order, 1);
                        0.0
                    }
                    (Some(first), None) => {
                        assert_eq!(expected.order, 2);
                        let ratio = first / step_size;
                        let previous = fixture.steps[step - 1].denoised[element];
                        alpha_target * phi_2 * (denoised - previous) / ratio
                    }
                    (Some(first), Some(second)) => {
                        assert_eq!(expected.order, 3);
                        let ratio_0 = first / step_size;
                        let ratio_1 = second / step_size;
                        let previous_1 = fixture.steps[step - 1].denoised[element];
                        let previous_2 = fixture.steps[step - 2].denoised[element];
                        let difference_0 = (denoised - previous_1) / ratio_0;
                        let difference_1 = (previous_1 - previous_2) / ratio_1;
                        let difference_sum = ratio_0 + ratio_1;
                        let first_derivative =
                            difference_0 + (difference_0 - difference_1) * ratio_0 / difference_sum;
                        let second_derivative = (difference_0 - difference_1) / difference_sum;
                        alpha_target * phi_2 * first_derivative
                            - alpha_target * phi_3 * second_derivative
                    }
                    (None, Some(_)) => unreachable!("second history cannot exist alone"),
                };
                latent_weight * latent
                    + denoised_weight * denoised
                    + correction
                    + stochastic_scale * noise
            })
            .collect::<Vec<_>>();
        assert_close(&reconstructed, &expected.latent_after, fixture.tolerance);
        step_size_2 = step_size_1;
        step_size_1 = Some(step_size);
    }
    assert_close(
        &values(
            &backend,
            trace.latents.last().ok_or("missing terminal")?,
            &context,
        )?,
        &fixture.terminal,
        fixture.tolerance,
    );

    let callbacks = callbacks.into_inner();
    assert_eq!(callbacks.len(), fixture.steps.len());
    for (step, (progress, latent, denoised)) in callbacks.iter().enumerate() {
        let expected = &fixture.steps[step];
        assert_eq!(usize::try_from(progress.step)?, step);
        assert!((progress.sigma - expected.sigma).abs() <= fixture.tolerance);
        assert!((progress.sigma_hat - expected.sigma).abs() <= fixture.tolerance);
        assert!((progress.next_sigma - expected.next_sigma).abs() <= fixture.tolerance);
        assert_close(latent, &expected.latent_before, fixture.tolerance);
        assert_close(denoised, &expected.denoised, fixture.tolerance);
    }

    assert_eq!(fixture.raw_brownian_bounds, [0.25, 4.0]);
    let (before, after) = checkpoints.ok_or("missing Brownian checkpoints")?;
    let mut oracle = noise_request(&fixture).open_transaction(
        DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::TorchSigned64,
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    let mut tree = oracle.brownian_tree(
        f64::from(fixture.raw_brownian_bounds[0]),
        vec![0.0; fixture.initial.len()],
        f64::from(fixture.raw_brownian_bounds[1]),
        &cancellation,
    )?;
    for expected in fixture.steps.iter().filter(|step| step.next_sigma > 0.0) {
        let raw = tree.increment(
            f64::from(expected.next_sigma),
            f64::from(expected.sigma),
            &cancellation,
        )?;
        let divisor = f64::from(expected.sigma - expected.next_sigma).sqrt();
        let normalized = raw
            .into_iter()
            .map(|value| (-value / divisor) as f32)
            .collect::<Vec<_>>();
        assert_close(&normalized, &expected.brownian_noise, fixture.tolerance);
    }
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn val_rng_001_cpu_brownian_uses_raw_bounds_before_constant_flow_adjustment()
-> Result<(), Box<dyn Error>> {
    let profile = DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("dpmpp-3m-sde-raw-bounds-v1")?,
        PredictionInterpretation::Denoised,
        Arc::from([0.05_f32, 0.1, 0.3, 0.7, 1.0]),
        SamplingSnrMode::ConstantFlow { shift: 1.3 },
        1.0,
    )?;
    let sigmas = [1.0_f32, 0.7, 0.0];
    let mut adjusted = sigmas;
    profile.adjust_first_sigma_for_snr(&mut adjusted)?;
    assert_ne!(adjusted[0].to_bits(), sigmas[0].to_bits());
    let initial_values = [0.4_f32, -0.2, 0.8];
    let eta = 0.5_f32;
    let noise_scale = 0.8_f32;
    let seed = 175_u64;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let (trace, checkpoints) = sample_dpmpp_3m_sde(
        &backend,
        plan(DPMPP_3M_SDE_SAMPLER_ID, &profile, seed, 2)?,
        &profile,
        tensor_from_f32(&backend, &[3], &initial_values, &context)?,
        &sigmas,
        Dpmpp3mSdeOptions::new(eta, noise_scale)?,
        raw_bounds_noise_request(),
        &context,
        |_, _, _| tensor_from_f32(&backend, &[3], &[0.0; 3], &context).map_err(|e| e.to_string()),
        |_, _, _| Ok::<_, String>(()),
    )?;
    let (before, after) = checkpoints.ok_or("missing Brownian checkpoints")?;
    let mut oracle = raw_bounds_noise_request().open_transaction(
        DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::TorchSigned64,
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    let mut raw_bounds_tree = oracle.brownian_tree(0.7, vec![0.0; 3], 1.0, &cancellation)?;
    let raw = raw_bounds_tree.increment(0.7, f64::from(adjusted[0]), &cancellation)?;
    let divisor = f64::from(adjusted[0] - adjusted[1]).sqrt();
    let normalized = raw
        .into_iter()
        .map(|value| (-value / divisor) as f32)
        .collect::<Vec<_>>();
    let lambda_source = profile.half_log_snr(adjusted[0])?;
    let lambda_target = profile.half_log_snr(adjusted[1])?;
    let step_size = lambda_target - lambda_source;
    let latent_weight = adjusted[1] / adjusted[0] * (-step_size * eta).exp();
    let stochastic_scale = adjusted[1]
        * (-(-2.0 * step_size * eta).exp_m1()).sqrt()
        * profile.scale_sampler_noise(noise_scale)?;
    let expected = initial_values
        .iter()
        .zip(normalized)
        .map(|(latent, noise)| latent_weight * latent + stochastic_scale * noise)
        .collect::<Vec<_>>();
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &expected,
        0.000002,
    );
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn val_sampling_foundation_001_short_failures_and_cancellation_are_typed_and_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let defaults = Dpmpp3mSdeOptions::source_defaults();
    assert_eq!(defaults, Dpmpp3mSdeOptions::default());
    assert_eq!(defaults.eta(), 1.0);
    assert_eq!(defaults.noise_scale(), 1.0);
    for invalid in [
        Dpmpp3mSdeOptions::new(f32::NAN, 1.0),
        Dpmpp3mSdeOptions::new(1.0, f32::INFINITY),
    ] {
        assert!(matches!(
            invalid,
            Err(Dpmpp3mSdeError::InvalidOption { .. })
        ));
    }
    let signed = Dpmpp3mSdeOptions::new(-1.0, -1.0)?;
    assert_eq!(signed.eta(), -1.0);
    assert_eq!(signed.noise_scale(), -1.0);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (short, checkpoints) = sample_dpmpp_3m_sde(
        &backend,
        plan(DPMPP_3M_SDE_SAMPLER_ID, &profile, 1, 1)?,
        &profile,
        initial.clone(),
        &[1.0],
        defaults,
        noise_request(&fixture),
        &context,
        |_, _, _| Err("short schedule called denoiser".to_owned()),
        |_, _, _| Err("short schedule called callback"),
    )?;
    assert_eq!(short.sigmas, [1.0]);
    assert!(short.denoiser_evaluations.is_empty());
    assert_eq!(short.latents.len(), 1);
    assert!(checkpoints.is_none());

    let (signed_trace, signed_checkpoints) = sample_dpmpp_3m_sde(
        &backend,
        plan(DPMPP_3M_SDE_SAMPLER_ID, &profile, 1, 2)?,
        &profile,
        initial.clone(),
        &[2.0, 1.0, 0.0],
        signed,
        noise_request(&fixture),
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(signed_trace.latents.len(), 3);
    let (signed_before, signed_after) =
        signed_checkpoints.ok_or("missing signed Brownian checkpoints")?;
    assert_ne!(signed_before, signed_after);

    let error = sample_dpmpp_3m_sde(
        &backend,
        plan("dpmpp_2m_sde", &profile, fixture.seed, fixture.steps.len())?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        Dpmpp3mSdeOptions::default(),
        noise_request(&fixture),
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("a different registered identity must not be substituted");
    assert!(matches!(error, Dpmpp3mSdeError::WrongSampler(_)));

    let error = sample_dpmpp_3m_sde(
        &backend,
        plan(
            DPMPP_3M_SDE_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        Dpmpp3mSdeOptions::default(),
        noise_request(&fixture),
        &context,
        |_, _, step| Err(format!("failure-{step}")),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("denoiser failure must remain typed");
    assert!(matches!(error, Dpmpp3mSdeError::Denoiser { step: 0, .. }));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_dpmpp_3m_sde(
        &backend,
        plan(
            DPMPP_3M_SDE_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        initial,
        &fixture.sigmas,
        Dpmpp3mSdeOptions::default(),
        noise_request(&fixture),
        &cancelled_context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, Dpmpp3mSdeError::Tensor(_)));
    Ok(())
}
