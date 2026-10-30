use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation,
    SamplerIdentity, SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity,
    SamplingSnrMode,
    generated_dpmpp_2m_sde_gpu_sim_model_0169::{
        DEFINITION, DPMPP_2M_SDE_GPU_BROWNIAN_CONTRACT_ID, DPMPP_2M_SDE_GPU_FEATURE_ID,
        DPMPP_2M_SDE_GPU_SAMPLER_ID, DPMPP_2M_SDE_GPU_SOURCE_ORDINAL, Dpmpp2mSdeGpuError,
        Dpmpp2mSdeGpuOptions, Dpmpp2mSdeSolverType, sample_dpmpp_2m_sde_gpu,
        validate_dpmpp_2m_sde_gpu_generation_device,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, RetryRngPolicy, RngCompatibilityRequest, RngExecutionScope,
    RngGenerationPlacement, RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_2m_sde_gpu_comfy_model_0169/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/dpmpp_2m_sde_gpu_sim_model_0169.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    eta: f32,
    noise_scale: f32,
    seed: u64,
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
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    lambda_s: Option<f32>,
    lambda_t: Option<f32>,
    step_size: Option<f32>,
    combined_step: Option<f32>,
    alpha_t: Option<f32>,
    latent_scale: Option<f32>,
    denoised_scale: Option<f32>,
    correction_scale: Option<f32>,
    noise_scale: Option<f32>,
    brownian_noise: Vec<f32>,
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
        SamplingProfileIdentity::new("dpmpp-2m-sde-gpu-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
    )?)
}

fn plan(identity: &str, seed: u64, steps: usize) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile()?.identity().clone(),
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
fn val_sampler_001_dpmpp_2m_sde_gpu_definition_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2M_SDE_GPU_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2M_SDE_GPU_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_2M_SDE_GPU_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2m_sde_gpu_comfy_model_0169"
    );
    assert_eq!(fixture.rng.contract, DPMPP_2M_SDE_GPU_BROWNIAN_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.placement, "native-input-device");
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2M_SDE_GPU_SAMPLER_ID)?)?,
        &DEFINITION
    );
    let cpu_definition = registry.resolve(&SamplerIdentity::new("dpmpp_2m_sde")?)?;
    assert_eq!(cpu_definition.identity, "dpmpp_2m_sde");
    assert_eq!(cpu_definition.feature_id, "COMFY-MODEL-0168");
    assert_ne!(cpu_definition, &DEFINITION);

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
        "def sample_dpmpp_2m_sde(",
        "solver_type not in {'heun', 'midpoint'}",
        "sigma_to_half_log_snr",
        "offset_first_sigma_for_snr",
        "h_eta = h * (eta + 1)",
        "if solver_type == 'heun'",
        "elif solver_type == 'midpoint'",
        "noise_sampler(sigmas[i], sigmas[i + 1])",
        "def sample_dpmpp_2m_sde_gpu(",
        "cpu=False",
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
            .is_some_and(|line| line.contains("\"dpmpp_2m_sde_gpu\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,dpmpp_2m_sde_gpu,")
                && line.ends_with(",COMFY-MODEL-0169"))
    );
    assert!(IMPLEMENTATION.contains("SamplingTrace"));
    assert!(IMPLEMENTATION.contains("CompatibilityNoiseRequest"));
    assert!(IMPLEMENTATION.contains("sample_dpmpp_2m_sde_with_generation_placement("));
    assert!(IMPLEMENTATION.contains("RngGenerationPlacement::Native(device)"));
    let short_guard = IMPLEMENTATION
        .find("if sigmas.len() > 1 {")
        .ok_or("missing source-order short-schedule guard")?;
    let device_validation = IMPLEMENTATION
        .find("validate_dpmpp_2m_sde_gpu_generation_device(device)?;")
        .ok_or("missing native-device validation")?;
    assert!(short_guard < device_validation);
    for forbidden in [
        "struct Dpmpp2mSdeGpuTrace",
        "struct Dpmpp2mSdeGpuProgress",
        "struct Dpmpp2mSdeGpuObservation",
        "RngStream::new",
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "pub struct CompatibilityNoiseRequest",
        "struct BrownianTree",
        "BrownianTree",
        "BrownianNoiseIntervalAddress",
        "SamplingSession",
        "profile.half_log_snr(",
        "exp_m1()",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_sde_gpu_matches_noise_equations_callbacks_and_terminal()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let callbacks = RefCell::new(Vec::new());
    let (trace, checkpoints) = sample_dpmpp_2m_sde_gpu(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2mSdeGpuOptions {
            eta: fixture.eta,
            noise_scale: fixture.noise_scale,
            solver_type: Dpmpp2mSdeSolverType::Midpoint,
        },
        &context,
        |latent, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| "unexpected denoiser step".to_owned())?;
            assert!((sigma - expected.sigma).abs() <= fixture.tolerance);
            let latent_values =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&latent_values, &expected.latent_before, fixture.tolerance);
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
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

    let (noise_before, noise_after) = checkpoints.ok_or("missing Brownian checkpoints")?;
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_close(
            &values(&backend, &trace.denoiser_evaluations[step], &context)?,
            &expected.denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
        if expected.next_sigma == 0.0 {
            assert_close(
                &expected.latent_after,
                &expected.denoised,
                fixture.tolerance,
            );
        } else {
            let lambda_s = profile.half_log_snr(expected.sigma)?;
            let lambda_t = profile.half_log_snr(expected.next_sigma)?;
            let step_size = lambda_t - lambda_s;
            let combined_step = step_size * (fixture.eta + 1.0);
            let alpha_t = expected.next_sigma * lambda_t.exp();
            let latent_scale =
                expected.next_sigma / expected.sigma * (-step_size * fixture.eta).exp();
            let denoised_scale = -(-combined_step).exp_m1();
            let correction_scale = if step == 0 {
                0.0
            } else {
                let previous_step_size = fixture.steps[step - 1]
                    .step_size
                    .ok_or("missing previous step size")?;
                0.5 * alpha_t * denoised_scale / (previous_step_size / step_size)
            };
            let noise_scale = expected.next_sigma
                * (-(-2.0 * step_size * fixture.eta).exp_m1()).sqrt()
                * profile.scale_sampler_noise(fixture.noise_scale)?;
            assert_optional(Some(lambda_s), expected.lambda_s, fixture.tolerance);
            assert_optional(Some(lambda_t), expected.lambda_t, fixture.tolerance);
            assert_optional(Some(step_size), expected.step_size, fixture.tolerance);
            assert_optional(
                Some(combined_step),
                expected.combined_step,
                fixture.tolerance,
            );
            assert_optional(Some(alpha_t), expected.alpha_t, fixture.tolerance);
            assert_optional(Some(latent_scale), expected.latent_scale, fixture.tolerance);
            assert_optional(
                Some(denoised_scale),
                expected.denoised_scale,
                fixture.tolerance,
            );
            assert_optional(
                Some(correction_scale),
                expected.correction_scale,
                fixture.tolerance,
            );
            assert_optional(Some(noise_scale), expected.noise_scale, fixture.tolerance);
        }
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
        assert!((progress.next_sigma - expected.next_sigma).abs() <= fixture.tolerance);
        assert_close(latent, &expected.latent_before, fixture.tolerance);
        assert_close(denoised, &expected.denoised, fixture.tolerance);
    }

    let mut oracle = CompatibilityRngTransaction::open(
        DPMPP_2M_SDE_GPU_BROWNIAN_CONTRACT_ID,
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
            RngGenerationPlacement::Native(DeviceId::CPU),
            RngExecutionScope::Production,
        ),
        None,
        &cancellation,
    )?;
    assert_eq!(noise_before, oracle.checkpoint());
    let mut tree =
        oracle.brownian_tree(0.5, vec![0.0; fixture.initial.len()], 2.0, &cancellation)?;
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
    assert_eq!(noise_after, oracle.commit());
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_sde_gpu_heun_option_uses_exact_correction() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (trace, checkpoints) = sample_dpmpp_2m_sde_gpu(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2mSdeGpuOptions {
            eta: fixture.eta,
            noise_scale: fixture.noise_scale,
            solver_type: Dpmpp2mSdeSolverType::Heun,
        },
        &context,
        |latent, sigma, step| {
            let latent =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            let denoised = latent
                .iter()
                .enumerate()
                .map(|(element, value)| {
                    0.65_f32.mul_add(
                        *value,
                        sigma * (0.03 * (element + 1) as f32) + step as f32 * 0.02,
                    )
                })
                .collect::<Vec<_>>();
            tensor_from_f32(&backend, &fixture.shape, &denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert!(checkpoints.is_some());
    let first_after = values(&backend, &trace.latents[1], &context)?;
    let first_denoised = values(&backend, &trace.denoiser_evaluations[0], &context)?;
    let second_denoised = values(&backend, &trace.denoiser_evaluations[1], &context)?;
    let expected_step = &fixture.steps[1];
    let combined_step = expected_step.combined_step.ok_or("missing combined step")?;
    let denoised_scale = expected_step
        .denoised_scale
        .ok_or("missing denoised scale")?;
    let alpha_t = expected_step.alpha_t.ok_or("missing alpha")?;
    let correction_scale = alpha_t * (denoised_scale / -combined_step + 1.0);
    let latent_scale = expected_step.latent_scale.ok_or("missing latent scale")?;
    let noise_scale = expected_step.noise_scale.ok_or("missing noise scale")?;
    let expected = first_after
        .iter()
        .zip(&second_denoised)
        .zip(&first_denoised)
        .zip(&expected_step.brownian_noise)
        .map(|(((latent, denoised), previous), noise)| {
            latent_scale * latent
                + alpha_t * denoised_scale * denoised
                + correction_scale * (denoised - previous)
                + noise_scale * noise
        })
        .collect::<Vec<_>>();
    assert_close(
        &values(&backend, &trace.latents[2], &context)?,
        &expected,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_sde_gpu_consumes_canonical_constant_flow_profile()
-> Result<(), Box<dyn Error>> {
    let flow_profile = DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("dpmpp-2m-sde-gpu-flow-v1")?,
        PredictionInterpretation::Flow,
        Arc::from([0.01_f32, 0.5, 0.999]),
        SamplingSnrMode::ConstantFlow { shift: 1.2 },
        0.4,
    )?;
    let mut expected_sigmas = vec![1.0_f32, 0.5, 0.0];
    flow_profile.adjust_first_sigma_for_snr(&mut expected_sigmas)?;
    assert!(expected_sigmas[0] < 1.0);
    assert!((flow_profile.scale_sampler_noise(0.5)? - 0.2).abs() <= 1.0e-7);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[1, 2], &[0.2, -0.3], &context)?;
    let plan = SamplingPlan::new(
        DPMPP_2M_SDE_GPU_SAMPLER_ID,
        "normal",
        flow_profile.identity().clone(),
        7,
        2,
        1.0,
        1.0,
    )?;
    let (trace, checkpoints) = sample_dpmpp_2m_sde_gpu(
        &backend,
        plan,
        &flow_profile,
        initial,
        &[1.0, 0.5, 0.0],
        CompatibilityNoiseRequest::new(
            "flow-workflow",
            "attempt-1",
            "sampler-169",
            0,
            0,
            0,
            0,
            RetryRngPolicy::Replay,
        ),
        Dpmpp2mSdeGpuOptions {
            eta: 1.0,
            noise_scale: 0.5,
            solver_type: Dpmpp2mSdeSolverType::Midpoint,
        },
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert!(checkpoints.is_some());
    assert_eq!(trace.sigmas, expected_sigmas);

    let mut oracle = CompatibilityRngTransaction::open(
        DPMPP_2M_SDE_GPU_BROWNIAN_CONTRACT_ID,
        RngCompatibilityRequest::new(
            "flow-workflow",
            "attempt-1",
            "sampler-169",
            0,
            0,
            0,
            0,
            RetryRngPolicy::Replay,
            7,
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(DeviceId::CPU),
            RngExecutionScope::Production,
        ),
        None,
        &cancellation,
    )?;
    let mut tree = oracle.brownian_tree(0.5, vec![0.0; 2], 1.0, &cancellation)?;
    let raw = tree.increment(
        f64::from(expected_sigmas[1]),
        f64::from(expected_sigmas[0]),
        &cancellation,
    )?;
    let divisor = f64::from(expected_sigmas[0] - expected_sigmas[1]).sqrt();
    let lambda_s = flow_profile.half_log_snr(expected_sigmas[0])?;
    let lambda_t = flow_profile.half_log_snr(expected_sigmas[1])?;
    let step_size = lambda_t - lambda_s;
    let alpha_t = expected_sigmas[1] * lambda_t.exp();
    let latent_scale = expected_sigmas[1] / expected_sigmas[0] * (-step_size).exp();
    let denoised_scale = -(-2.0 * step_size).exp_m1();
    let noise_scale = expected_sigmas[1]
        * (-(-2.0 * step_size).exp_m1()).sqrt()
        * flow_profile.scale_sampler_noise(0.5)?;
    let expected_first = [0.2_f32, -0.3]
        .iter()
        .zip(raw)
        .map(|(value, noise)| {
            (latent_scale + alpha_t * denoised_scale) * value
                + noise_scale * (-noise / divisor) as f32
        })
        .collect::<Vec<_>>();
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &expected_first,
        2.0e-6,
    );
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_sde_gpu_rejects_unavailable_device_and_is_failure_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let cuda = DeviceId::from_source_device("cuda:0")?;
    let error = validate_dpmpp_2m_sde_gpu_generation_device(cuda)
        .expect_err("unavailable CUDA must fail closed");
    assert!(matches!(
        error,
        Dpmpp2mSdeGpuError::DeviceUnavailable { device, .. } if device == cuda
    ));
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (short, checkpoints) = sample_dpmpp_2m_sde_gpu(
        &backend,
        plan(&fixture.identity, fixture.seed, 1)?,
        &profile,
        initial.clone(),
        &[fixture.sigmas[0]],
        noise_request(&fixture),
        Dpmpp2mSdeGpuOptions {
            eta: f32::NAN,
            ..Dpmpp2mSdeGpuOptions::default()
        },
        &context,
        |_, _, _| Err("short schedule must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("short schedule must not publish a callback"),
    )?;
    assert!(checkpoints.is_none());
    assert!(short.denoiser_evaluations.is_empty());
    assert_eq!(short.latents.len(), 1);
    assert_close(
        &values(&backend, &short.latents[0], &context)?,
        &fixture.initial,
        0.0,
    );
    let error = sample_dpmpp_2m_sde_gpu(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2mSdeGpuOptions {
            eta: f32::NAN,
            ..Dpmpp2mSdeGpuOptions::default()
        },
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("invalid options must fail before publication");
    assert!(matches!(
        error,
        Dpmpp2mSdeGpuError::EquationFamily(
            sim_sampler::generated_dpmpp_2m_sde_sim_model_0168::Dpmpp2mSdeSamplerError::InvalidOption {
                name: "eta",
                ..
            }
        )
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let error = sample_dpmpp_2m_sde_gpu(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2mSdeGpuOptions::default(),
        &cancelled_context,
        |_, _, _| Err("cancelled sampling must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("cancelled sampling must not publish a callback"),
    )
    .expect_err("pre-cancelled sampling must fail atomically");
    assert!(matches!(
        error,
        Dpmpp2mSdeGpuError::Tensor(TensorError::Cancelled)
    ));
    Ok(())
}
