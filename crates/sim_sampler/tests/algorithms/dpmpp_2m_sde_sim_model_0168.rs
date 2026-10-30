use sim_sampler::{
    BrownianNoiseIntervalAddress, CompatibilityNoiseRequest, DiscreteSamplingProfile,
    PredictionInterpretation, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_dpmpp_2m_sde_sim_model_0168::{
        DEFINITION, DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID, DPMPP_2M_SDE_FEATURE_ID,
        DPMPP_2M_SDE_SAMPLER_ID, DPMPP_2M_SDE_SOURCE_ORDINAL, Dpmpp2mSdeOptions,
        Dpmpp2mSdeSamplerError, Dpmpp2mSdeSolverType, sample_dpmpp_2m_sde,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngCompatibilityOperation, RngCompatibilityPhase, RngGenerationPlacement,
    RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_2m_sde_comfy_model_0168/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/dpmpp_2m_sde_sim_model_0168.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    rng_contract_id: String,
    placement: String,
    source: SourceFixture,
    cases: Vec<CaseFixture>,
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
}

#[derive(Debug, Deserialize)]
struct CaseFixture {
    name: String,
    seed: u64,
    eta: f32,
    noise_scale: f32,
    model_noise_scale: f32,
    snr_mode: String,
    snr_shift: Option<f32>,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    adjusted_sigmas: Vec<f32>,
    initial: Vec<f32>,
    steps: Vec<StepFixture>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    lambda_source: Option<f32>,
    lambda_target: Option<f32>,
    step_size: Option<f32>,
    eta_step_size: Option<f32>,
    alpha_target: Option<f32>,
    latent_weight: f32,
    denoised_weight: f32,
    step_ratio: Option<f32>,
    correction_weight: f32,
    brownian_noise: Option<Vec<f32>>,
    stochastic_scale: f32,
    deterministic: Vec<f32>,
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

fn profile(case: &CaseFixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    let snr_mode = match (case.snr_mode.as_str(), case.snr_shift) {
        ("standard", None) => SamplingSnrMode::Standard,
        ("constant_flow", Some(shift)) => SamplingSnrMode::ConstantFlow { shift },
        _ => return Err(format!("invalid fixture SNR mode for {}", case.name).into()),
    };
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new(format!("task294-{}", case.name))?,
        PredictionInterpretation::Denoised,
        Arc::from([0.05_f32, 0.1, 0.3, 0.7, 1.0, 2.0, 4.0]),
        snr_mode,
        case.model_noise_scale,
    )?)
}

fn plan(
    identity: &str,
    profile: &DiscreteSamplingProfile,
    seed: u64,
    steps: u32,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile.identity().clone(),
        seed,
        steps,
        1.0,
        1.0,
    )?)
}

fn noise_request(retry: u32, retry_policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "dpmpp-2m-sde-fixture-v1",
        "attempt-0168",
        "KSampler-19",
        19,
        168,
        0,
        retry,
        retry_policy,
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

fn assert_scalar(actual: f32, expected: f32, tolerance: f32, role: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{role}: expected {expected}, got {actual}"
    );
}

fn required(value: Option<f32>, role: &str, step: usize) -> Result<f32, Box<dyn Error>> {
    value.ok_or_else(|| format!("missing {role} at step {step}").into())
}

#[test]
fn val_sampler_001_dpmpp_2m_sde_definition_rng_and_source_provenance_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2M_SDE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2M_SDE_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_2M_SDE_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID);
    assert_eq!(fixture.placement, "cpu-seeded-transfer");
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2m_sde_comfy_model_0168"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2M_SDE_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        registry
            .resolve(&SamplerIdentity::new("dpmpp2m_sde")?)
            .is_err()
    );
    assert!(SamplerIdentity::new("DPMPP_2M_SDE").is_err());
    let rng = rng_compatibility_contract(DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID)
        .ok_or("Brownian RNG contract is unavailable")?;
    assert_eq!(rng.operation(), RngCompatibilityOperation::BrownianTree);
    assert_eq!(rng.phase(), RngCompatibilityPhase::SamplingNoiseAndSolver);
    assert_eq!(rng.symbol(), "torchsde.BrownianTree");

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
        "def sample_dpmpp_2m_sde",
        "if len(sigmas) <= 1",
        "sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()",
        "BrownianTreeNoiseSampler",
        "lambda_fn = partial(sigma_to_half_log_snr",
        "offset_first_sigma_for_snr",
        "s_noise = s_noise * getattr(model_sampling, \"noise_scale\", 1.0)",
        "callback({'x': x",
        "h_eta = h * (eta + 1)",
        "alpha_t = sigmas[i + 1] * lambda_t.exp()",
        "0.5 * alpha_t",
        "noise_sampler(sigmas[i], sigmas[i + 1])",
        "old_denoised = denoised",
    ] {
        assert!(
            equations.contains(fragment),
            "missing source equation {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_2m_sde\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(catalog.lines().any(|line| {
        line.contains("sampler,dpmpp_2m_sde,") && line.ends_with(",COMFY-MODEL-0168")
    }));
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_sde_matches_standard_and_constant_flow_intermediates()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    for case in &fixture.cases {
        let profile = profile(case)?;
        let mut adjusted = case.sigmas.clone();
        profile.adjust_first_sigma_for_snr(&mut adjusted)?;
        assert_eq!(adjusted, case.adjusted_sigmas);
        let effective_noise_scale = profile.scale_sampler_noise(case.noise_scale)?;
        assert_scalar(
            effective_noise_scale,
            case.noise_scale * case.model_noise_scale,
            0.0,
            "effective noise scale",
        );
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &case.shape, &case.initial, &context)?;
        let initial_alias = initial.clone();
        let events = RefCell::new(Vec::new());
        let (trace, checkpoints) = sample_dpmpp_2m_sde(
            &backend,
            plan(
                DPMPP_2M_SDE_SAMPLER_ID,
                &profile,
                case.seed,
                u32::try_from(case.steps.len())?,
            )?,
            &profile,
            initial,
            &case.sigmas,
            Dpmpp2mSdeOptions::new(case.eta, case.noise_scale)?,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |input, sigma, step| {
                let expected = case
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
                events.borrow_mut().push(format!("denoiser-{step}"));
                assert_eq!(sigma.to_bits(), expected.sigma.to_bits());
                let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
                assert_close(&input, &expected.latent_before, case.tolerance);
                let biases = [0.11_f32, -0.07, 0.03];
                let offsets = [-0.02_f32, 0.04, -0.06];
                let analytical = input
                    .iter()
                    .zip(biases)
                    .zip(offsets)
                    .map(|((value, bias), offset)| 0.61 * value + sigma * bias + offset)
                    .collect::<Vec<_>>();
                assert_close(&analytical, &expected.denoised, case.tolerance);
                tensor_from_f32(&backend, &case.shape, &expected.denoised, &context)
                    .map_err(|error| error.to_string())
            },
            |progress, latent, denoised| {
                let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
                let expected = case
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("unexpected callback step {step}"))?;
                events.borrow_mut().push(format!("callback-{step}"));
                assert_eq!(
                    progress.total_steps,
                    u32::try_from(case.steps.len()).map_err(|error| error.to_string())?
                );
                assert_eq!(progress.sigma.to_bits(), expected.sigma.to_bits());
                assert_eq!(progress.sigma_hat.to_bits(), expected.sigma.to_bits());
                assert_eq!(progress.next_sigma.to_bits(), expected.next_sigma.to_bits());
                assert_close(
                    &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                    &expected.latent_before,
                    case.tolerance,
                );
                assert_close(
                    &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                    &expected.denoised,
                    case.tolerance,
                );
                Ok::<(), String>(())
            },
        )?;
        let expected_events = (0..case.steps.len())
            .flat_map(|step| [format!("denoiser-{step}"), format!("callback-{step}")])
            .collect::<Vec<_>>();
        assert_eq!(events.into_inner(), expected_events);
        assert_eq!(trace.sigmas, case.adjusted_sigmas);
        assert_eq!(trace.denoiser_evaluations.len(), case.steps.len());
        assert_eq!(trace.latents.len(), case.steps.len() + 1);
        assert_close(
            &values(&backend, &initial_alias, &context)?,
            &case.initial,
            0.0,
        );

        let (before, after) = checkpoints.ok_or("missing Brownian checkpoints")?;
        let mut oracle = noise_request(0, RetryRngPolicy::Replay).open_transaction(
            DPMPP_2M_SDE_BROWNIAN_CONTRACT_ID,
            i128::from(case.seed),
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
            None,
            context.cancellation,
        )?;
        assert_eq!(before, oracle.checkpoint());
        let mut positive = case.sigmas.iter().copied().filter(|sigma| *sigma > 0.0);
        let first = positive.next().ok_or("missing positive Brownian sigma")?;
        let (minimum, maximum) = positive.fold((first, first), |(minimum, maximum), sigma| {
            (minimum.min(sigma), maximum.max(sigma))
        });
        let mut brownian = oracle.brownian_tree(
            f64::from(minimum),
            vec![0.0; case.initial.len()],
            f64::from(maximum),
            &cancellation,
        )?;
        assert_eq!(after, oracle.commit());

        let mut previous_denoised: Option<&[f32]> = None;
        let mut previous_step_size: Option<f32> = None;
        for (step, expected) in case.steps.iter().enumerate() {
            assert_eq!(expected.step, step);
            assert_close(
                &values(&backend, &trace.latents[step], &context)?,
                &expected.latent_before,
                case.tolerance,
            );
            assert_close(
                &values(&backend, &trace.denoiser_evaluations[step], &context)?,
                &expected.denoised,
                case.tolerance,
            );
            assert_close(
                &values(&backend, &trace.latents[step + 1], &context)?,
                &expected.latent_after,
                case.tolerance,
            );
            if expected.next_sigma == 0.0 {
                assert!(expected.lambda_source.is_none());
                assert!(expected.lambda_target.is_none());
                assert!(expected.brownian_noise.is_none());
                assert_close(&expected.deterministic, &expected.denoised, 0.0);
                assert_close(&expected.latent_after, &expected.denoised, 0.0);
                previous_denoised = Some(&expected.denoised);
                continue;
            }
            let lambda_source = profile.half_log_snr(expected.sigma)?;
            let lambda_target = profile.half_log_snr(expected.next_sigma)?;
            let step_size = lambda_target - lambda_source;
            let eta_step_size = step_size * (case.eta + 1.0);
            let alpha_target = expected.next_sigma * lambda_target.exp();
            let latent_weight =
                expected.next_sigma / expected.sigma * (-step_size * case.eta).exp();
            let denoised_weight = alpha_target * -(-eta_step_size).exp_m1();
            assert_scalar(
                lambda_source,
                required(expected.lambda_source, "lambda source", step)?,
                case.tolerance,
                "lambda source",
            );
            assert_scalar(
                lambda_target,
                required(expected.lambda_target, "lambda target", step)?,
                case.tolerance,
                "lambda target",
            );
            assert_scalar(
                step_size,
                required(expected.step_size, "step size", step)?,
                case.tolerance,
                "step size",
            );
            assert_scalar(
                eta_step_size,
                required(expected.eta_step_size, "eta step size", step)?,
                case.tolerance,
                "eta step size",
            );
            assert_scalar(
                alpha_target,
                required(expected.alpha_target, "alpha target", step)?,
                case.tolerance,
                "alpha target",
            );
            assert_scalar(
                latent_weight,
                expected.latent_weight,
                case.tolerance,
                "latent weight",
            );
            assert_scalar(
                denoised_weight,
                expected.denoised_weight,
                case.tolerance,
                "denoised weight",
            );
            let correction_weight = match previous_step_size {
                Some(previous) => {
                    let ratio = previous / step_size;
                    assert_scalar(
                        ratio,
                        required(expected.step_ratio, "step ratio", step)?,
                        case.tolerance,
                        "step ratio",
                    );
                    0.5 * denoised_weight / ratio
                }
                None => {
                    assert!(expected.step_ratio.is_none());
                    0.0
                }
            };
            assert_scalar(
                correction_weight,
                expected.correction_weight,
                case.tolerance,
                "correction weight",
            );
            let deterministic = expected
                .latent_before
                .iter()
                .zip(&expected.denoised)
                .enumerate()
                .map(|(element, (latent, denoised))| {
                    let correction = previous_denoised.map_or(0.0, |previous| {
                        correction_weight * (denoised - previous[element])
                    });
                    latent_weight * latent + denoised_weight * denoised + correction
                })
                .collect::<Vec<_>>();
            assert_close(&deterministic, &expected.deterministic, case.tolerance);
            let address = BrownianNoiseIntervalAddress::new(
                expected.sigma,
                expected.next_sigma,
                u32::try_from(step)?,
            )?;
            let (lower, upper) = address.canonical_interval();
            let sign = if address.reverse { -1.0 } else { 1.0 };
            let noise = brownian
                .increment(f64::from(lower), f64::from(upper), &cancellation)?
                .into_iter()
                .map(|value| (value * sign / f64::from(upper - lower).sqrt()) as f32)
                .collect::<Vec<_>>();
            let expected_noise = expected
                .brownian_noise
                .as_deref()
                .ok_or("missing Brownian noise")?;
            assert_close(&noise, expected_noise, case.tolerance);
            let stochastic_scale = expected.next_sigma
                * (-(-2.0 * step_size * case.eta).exp_m1()).sqrt()
                * effective_noise_scale;
            assert_scalar(
                stochastic_scale,
                expected.stochastic_scale,
                case.tolerance,
                "stochastic scale",
            );
            let analytical_next = deterministic
                .iter()
                .zip(&noise)
                .map(|(deterministic, noise)| deterministic + stochastic_scale * noise)
                .collect::<Vec<_>>();
            assert_close(&analytical_next, &expected.latent_after, case.tolerance);
            previous_denoised = Some(&expected.denoised);
            previous_step_size = Some(step_size);
        }
        assert_close(
            &values(
                &backend,
                trace.latents.last().ok_or("missing terminal latent")?,
                &context,
            )?,
            &case.terminal,
            case.tolerance,
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }
    Ok(())
}

#[test]
fn boundaries_retry_cancellation_and_failures_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let case = fixture.cases.first().ok_or("missing standard case")?;
    let profile = profile(case)?;
    let defaults = Dpmpp2mSdeOptions::source_defaults();
    assert_eq!(defaults, Dpmpp2mSdeOptions::default());
    assert_eq!(defaults.eta(), 1.0);
    assert_eq!(defaults.noise_scale(), 1.0);
    assert_eq!(defaults.solver_type(), Dpmpp2mSdeSolverType::Midpoint);
    assert_eq!(
        Dpmpp2mSdeOptions::new(case.eta, case.noise_scale)?,
        Dpmpp2mSdeOptions::new_with_solver_type(
            case.eta,
            case.noise_scale,
            Dpmpp2mSdeSolverType::Midpoint,
        )?
    );
    for invalid in [
        Dpmpp2mSdeOptions::new(f32::NAN, 1.0),
        Dpmpp2mSdeOptions::new(1.0, f32::INFINITY),
    ] {
        assert!(matches!(
            invalid,
            Err(Dpmpp2mSdeSamplerError::InvalidOption { .. })
        ));
    }
    let signed = Dpmpp2mSdeOptions::new(-1.0, -1.0)?;
    assert_eq!(signed.eta(), -1.0);
    assert_eq!(signed.noise_scale(), -1.0);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &case.shape, &case.initial, &context)?;
    let (short, short_checkpoints) = sample_dpmpp_2m_sde(
        &backend,
        plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 1, 1)?,
        &profile,
        initial.clone(),
        &[1.0],
        defaults,
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |_, _, _| Err("short schedule called denoiser".to_owned()),
        |_, _, _| Err("short schedule called callback"),
    )?;
    assert_eq!(short.sigmas, [1.0]);
    assert!(short.denoiser_evaluations.is_empty());
    assert_eq!(short.latents.len(), 1);
    assert!(short_checkpoints.is_none());

    assert!(matches!(
        sample_dpmpp_2m_sde(
            &backend,
            plan("dpmpp_2m", &profile, 1, 1)?,
            &profile,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mSdeSamplerError::WrongSampler(value)) if value == "dpmpp_2m"
    ));
    assert!(matches!(
        sample_dpmpp_2m_sde(
            &backend,
            plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 1, 1)?,
            &profile,
            initial.clone(),
            &[1.0, 1.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mSdeSamplerError::Sampling(
            SamplingError::InvalidSigma { .. }
        ))
    ));

    let run = |retry, policy| {
        sample_dpmpp_2m_sde(
            &backend,
            plan(
                DPMPP_2M_SDE_SAMPLER_ID,
                &profile,
                case.seed,
                u32::try_from(case.steps.len()).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
            &profile,
            tensor_from_f32(&backend, &case.shape, &case.initial, &context)
                .map_err(|error| error.to_string())?,
            &case.sigmas,
            Dpmpp2mSdeOptions::new(case.eta, case.noise_scale)
                .map_err(|error| error.to_string())?,
            noise_request(retry, policy),
            &context,
            |_, _, step| {
                let denoised = case
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("missing denoiser fixture {step}"))?
                    .denoised
                    .as_slice();
                tensor_from_f32(&backend, &case.shape, denoised, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )
        .map_err(|error| error.to_string())
    };
    let (replay_a, replay_a_checkpoints) = run(0, RetryRngPolicy::Replay)?;
    let (replay_b, replay_b_checkpoints) = run(7, RetryRngPolicy::Replay)?;
    assert_eq!(replay_a_checkpoints, replay_b_checkpoints);
    for (left, right) in replay_a.latents.iter().zip(&replay_b.latents) {
        assert_eq!(
            values(&backend, left, &context)?,
            values(&backend, right, &context)?
        );
    }
    let (_, advance_checkpoints) = run(7, RetryRngPolicy::Advance)?;
    assert_ne!(replay_a_checkpoints, advance_checkpoints);

    let (deterministic, deterministic_checkpoints) = sample_dpmpp_2m_sde(
        &backend,
        plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 5, 2)?,
        &profile,
        initial.clone(),
        &[2.0, 1.0, 0.0],
        Dpmpp2mSdeOptions::new(0.0, 1.0)?,
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |value, _, _| Ok(value.clone()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    let (deterministic_before, deterministic_after) =
        deterministic_checkpoints.ok_or("missing deterministic Brownian checkpoints")?;
    assert_ne!(deterministic_before, deterministic_after);
    assert_eq!(deterministic.latents.len(), 3);

    let (signed_trace, signed_checkpoints) = sample_dpmpp_2m_sde(
        &backend,
        plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 5, 2)?,
        &profile,
        initial.clone(),
        &[2.0, 1.0, 0.0],
        signed,
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |value, _, _| Ok(value.clone()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(signed_trace.latents.len(), 3);
    assert!(signed_checkpoints.is_some());

    let pre_cancelled = CancellationToken::default();
    pre_cancelled.cancel();
    let pre_cancelled_context = execution_context(&backend, &authority, &pre_cancelled)?;
    assert!(matches!(
        sample_dpmpp_2m_sde(
            &backend,
            plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 1, 2)?,
            &profile,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &pre_cancelled_context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mSdeSamplerError::Tensor(TensorError::Cancelled))
    ));
    assert!(matches!(
        sample_dpmpp_2m_sde(
            &backend,
            plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 1, 2)?,
            &profile,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Err("callback fault"),
        ),
        Err(Dpmpp2mSdeSamplerError::Sampling(SamplingError::Callback(reason))) if reason == "callback fault"
    ));
    assert!(matches!(
        sample_dpmpp_2m_sde(
            &backend,
            plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 1, 2)?,
            &profile,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |_, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mSdeSamplerError::DenoiserContract { step: 0 })
    ));
    assert!(matches!(
        sample_dpmpp_2m_sde(
            &backend,
            plan(DPMPP_2M_SDE_SAMPLER_ID, &profile, 1, 2)?,
            &profile,
            initial,
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, step| {
                if step == 0 {
                    Ok(value.clone())
                } else {
                    tensor_from_f32(&backend, &[3], &[f32::NAN, 0.0, 0.0], &context)
                        .map_err(|error| error.to_string())
                }
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mSdeSamplerError::NonFinite {
            step: 1,
            stage: "terminal denoiser",
            element: 0,
        })
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn row_is_only_an_equation_adapter_over_canonical_sampling_profile_tensor_and_rng_owners() {
    for forbidden in [
        "struct Dpmpp2mSdeTrace",
        "struct Dpmpp2mSdeProgress",
        "struct Dpmpp2mSdeObservation",
        "pub struct SamplingSession",
        "pub struct SamplingProgress",
        "pub struct SamplingTrace",
        "pub struct ObservedSamplingStep",
        "pub struct BrownianTree",
        "BrownianTree::new",
        "RngStream::new",
        "sigma.log().neg()",
        "sigma.logit().neg()",
        "percent_to_sigma",
        "std::process",
        "Command::new",
        "pyo3",
        "python",
        "javascript",
        "unsafe {",
        "todo!",
        "unimplemented!",
        "panic!",
        ".unwrap(",
        ".expect(",
        "let _ =",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "row contains forbidden owner, fallback, or placeholder {forbidden}"
        );
    }
    for canonical_adapter in [
        "pub(crate) fn sample_dpmpp_2m_sde_with_generation_placement",
        "RngGenerationPlacement::CpuSeededTransfer",
        "SamplingSession::new",
        ".observe_step(",
        "profile.half_log_snr(",
        "profile.adjust_first_sigma_for_snr(",
        "profile.scale_sampler_noise(",
        "transaction.brownian_tree(",
        "tree.increment(",
        "BrownianNoiseIntervalAddress::new",
        "backend.workspace_vec",
        "context.check()",
    ] {
        assert!(
            IMPLEMENTATION.contains(canonical_adapter),
            "row does not delegate through {canonical_adapter}"
        );
    }
    let bounds = IMPLEMENTATION.find("brownian_bounds(sigmas)");
    let clone = IMPLEMENTATION.find("adjusted_sigmas.extend_from_slice(sigmas)");
    let adjustment = IMPLEMENTATION.find("profile.adjust_first_sigma_for_snr");
    assert!(
        matches!(
            (bounds, clone, adjustment),
            (Some(bounds), Some(clone), Some(adjustment)) if bounds < clone && clone < adjustment
        ),
        "Brownian bounds must precede canonical profile schedule adjustment"
    );
}
