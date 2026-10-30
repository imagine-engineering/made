use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity,
    generated_dpmpp_2m_sde_sim_model_0168::{
        DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID, DPMPP_2M_SDE_SAMPLER_ID, Dpmpp2mSdeOptions,
        Dpmpp2mSdeSolverType, sample_dpmpp_2m_sde,
    },
    generated_dpmpp_2m_sde_heun_sim_model_0170::{
        DEFINITION, DPMPP_2M_SDE_HEUN_FEATURE_ID, DPMPP_2M_SDE_HEUN_SAMPLER_ID,
        DPMPP_2M_SDE_HEUN_SOURCE_ORDINAL, Dpmpp2mSdeHeunError, sample_dpmpp_2m_sde_heun,
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
    "/../sim_test_support/fixtures/samplers/dpmpp_2m_sde_heun_comfy_model_0170/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/dpmpp_2m_sde_heun_sim_model_0170.rs");

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
        SamplingProfileIdentity::new("dpmpp-2m-sde-heun-row-v1")?,
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
fn val_sampler_001_dpmpp_2m_sde_heun_provenance_registry_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2M_SDE_HEUN_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2M_SDE_HEUN_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_2M_SDE_HEUN_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2m_sde_heun_comfy_model_0170"
    );
    assert_eq!(fixture.rng.contract, DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2M_SDE_HEUN_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert_ne!(DPMPP_2M_SDE_HEUN_SAMPLER_ID, DPMPP_2M_SDE_SAMPLER_ID);

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
        "if solver_type == 'heun'",
        "def sample_dpmpp_2m_sde_heun(",
        "return sample_dpmpp_2m_sde(",
        "cpu=True",
        "solver_type=solver_type",
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
            .is_some_and(|line| line.contains("\"dpmpp_2m_sde_heun\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,dpmpp_2m_sde_heun,")
                && line.ends_with(",COMFY-MODEL-0170"))
    );

    assert!(IMPLEMENTATION.contains("sample_dpmpp_2m_sde("));
    assert!(IMPLEMENTATION.contains("Dpmpp2mSdeSolverType::Heun"));
    assert!(IMPLEMENTATION.contains("CompatibilityNoiseRequest"));
    for forbidden in [
        "struct Dpmpp2mSdeHeunTrace",
        "struct Dpmpp2mSdeHeunProgress",
        "struct Dpmpp2mSdeHeunObservation",
        "struct Dpmpp2mSdeHeunNoiseRequest",
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream",
        "BrownianTree",
        "half_log_snr(",
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
fn val_sampler_001_dpmpp_2m_sde_heun_matches_every_intermediate_callback_and_terminal()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let callbacks = RefCell::new(Vec::new());
    let (trace, checkpoints) = sample_dpmpp_2m_sde_heun(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
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
            let denoised = latent_values
                .iter()
                .enumerate()
                .map(|(element, value)| {
                    0.65_f32.mul_add(
                        *value,
                        sigma * (0.03 * (element + 1) as f32) + step as f32 * 0.02,
                    )
                })
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

    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
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
            assert_close(
                &expected.latent_after,
                &expected.denoised,
                fixture.tolerance,
            );
            continue;
        }

        let lambda_s = profile.half_log_snr(expected.sigma)?;
        let lambda_t = profile.half_log_snr(expected.next_sigma)?;
        let step_size = lambda_t - lambda_s;
        let combined_step = step_size * (fixture.eta + 1.0);
        let alpha_t = expected.next_sigma * lambda_t.exp();
        let latent_scale = expected.next_sigma / expected.sigma * (-step_size * fixture.eta).exp();
        let denoised_scale = -(-combined_step).exp_m1();
        let correction_scale = if step == 0 {
            0.0
        } else {
            let previous_step_size = fixture.steps[step - 1]
                .step_size
                .ok_or("missing previous step size")?;
            alpha_t * (denoised_scale / -combined_step + 1.0) / (previous_step_size / step_size)
        };
        let stochastic_scale = expected.next_sigma
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
        assert_optional(
            Some(stochastic_scale),
            expected.noise_scale,
            fixture.tolerance,
        );

        let previous_denoised = step
            .checked_sub(1)
            .and_then(|previous| fixture.steps.get(previous))
            .map(|previous| previous.denoised.as_slice());
        let reconstructed = expected
            .latent_before
            .iter()
            .zip(&expected.denoised)
            .zip(&expected.brownian_noise)
            .enumerate()
            .map(|(element, ((latent, denoised), brownian))| {
                let correction = previous_denoised.map_or(0.0, |previous| {
                    correction_scale * (denoised - previous[element])
                });
                latent_scale * latent
                    + alpha_t * denoised_scale * denoised
                    + correction
                    + stochastic_scale * brownian
            })
            .collect::<Vec<_>>();
        assert_close(&reconstructed, &expected.latent_after, fixture.tolerance);
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

    let (noise_before, noise_after) = checkpoints.ok_or("missing Brownian RNG checkpoints")?;
    let mut oracle = noise_request(&fixture).open_transaction(
        DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::TorchSigned64,
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
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
fn val_sampling_foundation_001_adapter_is_exactly_the_canonical_heun_equation_family()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let adapter_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let family_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let denoise = |latent: &Tensor, sigma: f32, step: usize| {
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
    };
    let (adapter, adapter_checkpoints) = sample_dpmpp_2m_sde_heun(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        adapter_initial,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &context,
        denoise,
        |_, _, _| Ok::<_, String>(()),
    )?;
    let (family, family_checkpoints) = sample_dpmpp_2m_sde(
        &backend,
        plan(DPMPP_2M_SDE_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
        &profile,
        family_initial,
        &fixture.sigmas,
        Dpmpp2mSdeOptions::new_with_solver_type(
            fixture.eta,
            fixture.noise_scale,
            Dpmpp2mSdeSolverType::Heun,
        )?,
        noise_request(&fixture),
        &context,
        denoise,
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(adapter_checkpoints, family_checkpoints);
    assert_eq!(adapter.sigmas, family.sigmas);
    assert_eq!(adapter.latents.len(), family.latents.len());
    for (adapter, family) in adapter.latents.iter().zip(&family.latents) {
        assert_close(
            &values(&backend, adapter, &context)?,
            &values(&backend, family, &context)?,
            fixture.tolerance,
        );
    }
    Ok(())
}

#[test]
fn val_rng_001_adapter_rejects_wrong_identity_invalid_options_and_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let (short, checkpoints) = sample_dpmpp_2m_sde_heun(
        &backend,
        plan(&fixture.identity, fixture.seed, 1)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &[fixture.sigmas[0]],
        noise_request(&fixture),
        f32::NAN,
        f32::NAN,
        &context,
        |_, _, _| Err("short schedule must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("short schedule must not callback"),
    )?;
    assert_eq!(short.latents.len(), 1);
    assert!(checkpoints.is_none());
    let error = sample_dpmpp_2m_sde_heun(
        &backend,
        plan(DPMPP_2M_SDE_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("a distinct registered sampler identity must not be substituted");
    assert!(matches!(error, Dpmpp2mSdeHeunError::WrongSampler(_)));

    let error = sample_dpmpp_2m_sde_heun(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        f32::NAN,
        fixture.noise_scale,
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("invalid equation-family options must fail closed");
    assert!(matches!(error, Dpmpp2mSdeHeunError::EquationFamily(_)));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_dpmpp_2m_sde_heun(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &cancelled_context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, Dpmpp2mSdeHeunError::Tensor(_)));
    Ok(())
}
