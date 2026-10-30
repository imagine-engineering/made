use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_exp_heun_2_x0_sim_model_0183::{ExpHeun2X0Error, sample_exp_heun_2_x0},
    generated_exp_heun_2_x0_sde_sim_model_0184::{
        DEFINITION, EXP_HEUN_2_X0_SDE_FEATURE_ID, EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID,
        EXP_HEUN_2_X0_SDE_SAMPLER_ID, EXP_HEUN_2_X0_SDE_SOURCE_ORDINAL, ExpHeun2X0DenoiserStage,
        ExpHeun2X0SdeError, ExpHeun2X0SdeOptions, ExpHeun2X0SolverType,
        exp_heun_2_x0_sde_rng_profile, sample_exp_heun_2_x0_sde,
        validate_exp_heun_2_x0_sde_generation_device,
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
    "/../sim_test_support/fixtures/samplers/exp_heun_2_x0_sde_comfy_model_0184/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/exp_heun_2_x0_sde_sim_model_0184.rs"
));
const FAMILY_IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/seeds_2_sim_model_0199.rs"
));

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
    solver_type: String,
    eta: f32,
    sampler_noise_scale: f32,
    profile_noise_scale: f32,
    effective_noise_scale: f32,
    signed_controls: SignedControlsFixture,
    seed: u64,
    rng: RngFixture,
    steps: Vec<StepFixture>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SignedControlsFixture {
    eta_above_negative_one: f32,
    eta_below_negative_one: f32,
    singular_eta: f32,
    negative_noise_scale: f32,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    sampling_path: String,
    sampling_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
    wrapper_lines: [usize; 2],
    equation_lines: [usize; 2],
    phi_lines: [usize; 2],
    profile_lines: [usize; 2],
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
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    primary: Vec<f32>,
    lambda_source: Option<f32>,
    lambda_target: Option<f32>,
    step_size: Option<f32>,
    eta_step_size: Option<f32>,
    intermediate_sigma: Option<f32>,
    alpha_target: Option<f32>,
    phi_1: Option<f32>,
    phi_2: Option<f32>,
    b1: Option<f32>,
    b2: Option<f32>,
    first_noise_root: Option<f32>,
    predictor_noise_coefficient: Option<f32>,
    first_noise: Option<Vec<f32>>,
    second_noise: Option<Vec<f32>>,
    second_noise_weight: Option<f32>,
    predictor: Option<Vec<f32>>,
    corrector: Option<Vec<f32>>,
    deterministic: Option<Vec<f32>>,
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

fn source_range(source: &str, range: [usize; 2]) -> String {
    source
        .lines()
        .skip(range[0].saturating_sub(1))
        .take(range[1].saturating_sub(range[0]) + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

fn sampler_names(source: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let assignment = source
        .find("KSAMPLER_NAMES = [")
        .ok_or("KSAMPLER_NAMES assignment is missing")?;
    let source = source.get(assignment..).ok_or("invalid assignment start")?;
    let end = source.find(']').ok_or("KSAMPLER_NAMES is unterminated")?;
    let source = source.get(..end).ok_or("invalid assignment range")?;
    let mut names = Vec::new();
    let mut characters = source.chars();
    while let Some(character) = characters.next() {
        if character == '"' {
            names.push(
                characters
                    .by_ref()
                    .take_while(|character| *character != '"')
                    .collect(),
            );
        }
    }
    Ok(names)
}

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("exp-heun-2-x0-sde-row-v1")?,
        PredictionInterpretation::Denoised,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
        SamplingSnrMode::Standard,
        fixture.profile_noise_scale,
    )?)
}

fn plan(
    fixture: &Fixture,
    identity: &str,
    profile: &DiscreteSamplingProfile,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile.identity().clone(),
        fixture.seed,
        u32::try_from(fixture.steps.len())?,
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

fn signed_source_oracle(fixture: &Fixture, eta: f32) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut current = fixture.initial.clone();
    let mut latents = vec![current.clone()];
    for expected in &fixture.steps {
        if expected.next_sigma == 0.0 {
            current = expected.primary.clone();
        } else {
            let lambda_source = -expected.sigma.ln();
            let lambda_target = -expected.next_sigma.ln();
            let step_size = lambda_target - lambda_source;
            let eta_step_size = step_size * (eta + 1.0);
            let phi_one = (-eta_step_size).exp_m1();
            let phi_two = (phi_one + eta_step_size) / -eta_step_size;
            let primary_weight = phi_one - phi_two;
            let corrector_weight = phi_two;
            let latent_weight = expected.next_sigma / expected.sigma * (-step_size * eta).exp();
            let corrector = expected
                .corrector
                .as_deref()
                .ok_or("missing signed-control corrector")?;
            current = current
                .iter()
                .zip(&expected.primary)
                .zip(corrector)
                .map(|((latent, primary), corrector)| {
                    latent_weight * latent
                        - (primary_weight * primary + corrector_weight * corrector)
                })
                .collect();
        }
        latents.push(current.clone());
    }
    Ok(latents)
}

#[test]
fn val_sampler_001_exp_heun_2_x0_sde_definition_provenance_and_family_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, EXP_HEUN_2_X0_SDE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, EXP_HEUN_2_X0_SDE_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, EXP_HEUN_2_X0_SDE_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 7);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/exp_heun_2_x0_sde_comfy_model_0184"
    );
    assert_eq!(fixture.rng.contract, EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.seed_transform, "add-one-on-cpu");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(EXP_HEUN_2_X0_SDE_SAMPLER_ID)?)?,
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
    let wrapper = source_range(&sampling, fixture.source.wrapper_lines);
    for fragment in [
        "def sample_exp_heun_2_x0_sde(",
        "eta=eta",
        "s_noise=s_noise",
        "r=1.0",
        "solver_type=solver_type",
    ] {
        assert!(wrapper.contains(fragment), "missing wrapper {fragment}");
    }
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "inject_noise = eta > 0 and s_noise > 0",
        "callback({'x': x",
        "# Step 1",
        "noise_sampler(sigmas[i], sigma_s_1)",
        "denoised_2 = model(x_2",
        "# Step 2",
        "if solver_type == \"phi_1\"",
        "elif solver_type == \"phi_2\"",
        "segment_factor = (r - 1) * h * eta",
        "noise_sampler(sigma_s_1, sigmas[i + 1])",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let phi = source_range(&sampling, fixture.source.phi_lines);
    assert!(phi.contains("return torch.expm1(h)"));
    assert!(phi.contains("return (torch.expm1(h) - h) / h"));
    let profile_source = source_range(&sampling, fixture.source.profile_lines);
    assert!(profile_source.contains("def sigma_to_half_log_snr"));
    assert!(profile_source.contains("def half_log_snr_to_sigma"));
    let noise = source_range(&sampling, fixture.source.noise_lines);
    for fragment in ["seed += 1", "torch.Generator", "torch.randn"] {
        assert!(noise.contains(fragment), "missing noise {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"exp_heun_2_x0_sde\""))
    );
    let names = sampler_names(&samplers)?;
    assert_eq!(
        names
            .iter()
            .position(|name| name == EXP_HEUN_2_X0_SDE_SAMPLER_ID),
        Some(7)
    );
    assert_eq!(
        names
            .iter()
            .filter(|name| *name == EXP_HEUN_2_X0_SDE_SAMPLER_ID)
            .count(),
        1
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,exp_heun_2_x0_sde,")
                && line.ends_with(",COMFY-MODEL-0184"))
    );

    for adapter in [
        "sample_seeds_2_stochastic_family(",
        "intermediate_step_ratio: 1.0",
        "map_solver_type(options.solver_type)",
        "map_denoiser_stage(stage)",
    ] {
        assert!(
            IMPLEMENTATION.contains(adapter),
            "missing family adapter {adapter}"
        );
    }
    for owner in [
        "fn sample_seeds_2_family<",
        "SamplingSession::new",
        ".observe_step(",
        "profile.adjust_first_sigma_for_snr(",
        ".draw_normal(",
    ] {
        assert!(
            FAMILY_IMPLEMENTATION.contains(owner),
            "family does not own {owner}"
        );
        assert!(
            !IMPLEMENTATION.contains(owner),
            "adapter duplicates {owner}"
        );
    }
    assert_eq!(FAMILY_IMPLEMENTATION.matches(".draw_normal(").count(), 2);
    assert_eq!(IMPLEMENTATION.matches(".draw_normal(").count(), 0);
    let cuda = DeviceId::from_source_device("cuda:3")?;
    assert_eq!(
        exp_heun_2_x0_sde_rng_profile(DeviceId::CPU),
        (
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
        )
    );
    assert_eq!(
        exp_heun_2_x0_sde_rng_profile(cuda),
        (
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(cuda),
        )
    );
    let cancellation = CancellationToken::default();
    let request = |placement| {
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
            if placement == RngGenerationPlacement::Native(cuda) {
                RngSeedTransform::TorchSigned64
            } else {
                RngSeedTransform::Add(1)
            },
            placement,
            RngExecutionScope::Production,
        )
    };
    let cpu = CompatibilityRngTransaction::open(
        EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID,
        request(RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        }),
        None,
        &cancellation,
    )?;
    let native = CompatibilityRngTransaction::open(
        EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID,
        request(RngGenerationPlacement::Native(cuda)),
        None,
        &cancellation,
    )?;
    assert_ne!(cpu.checkpoint(), native.checkpoint());
    for forbidden in [
        "profile.half_log_snr(",
        "profile.sigma_from_half_log_snr(",
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream",
        "struct ExpHeun2X0SdeTrace",
        "struct ExpHeun2X0SdeProgress",
        "Command::new",
        "include!(",
        "#[path",
        "todo!",
        "unimplemented!",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner or escape {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_exp_heun_2_x0_sde_matches_every_stage_callback_and_rng_draw()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.solver_type, "phi_2");
    let profile = profile(&fixture)?;
    assert!(
        (profile.scale_sampler_noise(fixture.sampler_noise_scale)? - fixture.effective_noise_scale)
            .abs()
            <= fixture.tolerance
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let events = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let (trace, noise_before, noise_after) = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: fixture.eta,
            noise_scale: fixture.sampler_noise_scale,
            solver_type: ExpHeun2X0SolverType::Phi2,
        },
        &context,
        |latent, sigma, step, stage| {
            events.borrow_mut().push(format!("{stage:?}:{step}"));
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            let (expected_sigma, expected_latent, output) = match stage {
                ExpHeun2X0DenoiserStage::Primary => (
                    expected.sigma,
                    expected.latent_before.as_slice(),
                    expected.primary.as_slice(),
                ),
                ExpHeun2X0DenoiserStage::Corrector => (
                    expected
                        .intermediate_sigma
                        .ok_or("missing intermediate sigma")?,
                    expected.predictor.as_deref().ok_or("missing predictor")?,
                    expected.corrector.as_deref().ok_or("missing corrector")?,
                ),
            };
            if (sigma - expected_sigma).abs() > fixture.tolerance {
                return Err("sigma mismatch".to_owned());
            }
            let actual =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&actual, expected_latent, fixture.tolerance);
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            events
                .borrow_mut()
                .push(format!("Callback:{}", progress.step));
            callbacks.borrow_mut().push((
                *progress,
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<_, String>(())
        },
    )?;
    assert_eq!(
        events.into_inner(),
        [
            "Primary:0",
            "Callback:0",
            "Corrector:0",
            "Primary:1",
            "Callback:1",
            "Corrector:1",
            "Primary:2",
            "Callback:2"
        ]
    );
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_close(
            &values(&backend, &trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.denoiser_evaluations[step], &context)?,
            &expected.primary,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
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
    for (step, (progress, latent, denoised)) in callbacks.into_inner().iter().enumerate() {
        let expected = &fixture.steps[step];
        assert_eq!(usize::try_from(progress.step)?, step);
        assert_eq!(usize::try_from(progress.total_steps)?, fixture.steps.len());
        assert!((progress.sigma - expected.sigma).abs() <= fixture.tolerance);
        assert!((progress.sigma_hat - expected.sigma).abs() <= fixture.tolerance);
        assert!((progress.next_sigma - expected.next_sigma).abs() <= fixture.tolerance);
        assert_close(latent, &expected.latent_before, fixture.tolerance);
        assert_close(denoised, &expected.primary, fixture.tolerance);
    }

    let mut oracle = CompatibilityRngTransaction::open(
        EXP_HEUN_2_X0_SDE_NOISE_CONTRACT_ID,
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
        &cancellation,
    )?;
    assert_eq!(noise_before, oracle.checkpoint());
    for expected in fixture.steps.iter().filter(|step| step.next_sigma > 0.0) {
        for expected_noise in [
            expected
                .first_noise
                .as_deref()
                .ok_or("missing first noise")?,
            expected
                .second_noise
                .as_deref()
                .ok_or("missing second noise")?,
        ] {
            let actual = oracle
                .draw_normal(expected_noise.len(), &cancellation)?
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            assert_close(&actual, expected_noise, 0.0);
        }
    }
    assert_eq!(noise_after, oracle.commit());
    Ok(())
}

#[test]
fn val_sampling_foundation_001_exp_heun_2_x0_sde_fixture_reconstructs_every_equation()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    for expected in &fixture.steps {
        if expected.next_sigma == 0.0 {
            assert_close(&expected.latent_after, &expected.primary, fixture.tolerance);
            continue;
        }
        let lambda_source = -expected.sigma.ln();
        let lambda_target = -expected.next_sigma.ln();
        let step_size = lambda_target - lambda_source;
        let eta_step_size = step_size * (fixture.eta + 1.0);
        let intermediate_sigma = (-lambda_target).exp();
        let alpha_target = expected.next_sigma * lambda_target.exp();
        let phi_one = (-eta_step_size).exp_m1();
        let phi_two = (phi_one + eta_step_size) / -eta_step_size;
        let b2 = phi_two;
        let b1 = phi_one - b2;
        let first_noise_root = (-(-2.0 * step_size * fixture.eta).exp_m1()).sqrt();
        let r = 1.0_f32;
        let second_noise_weight = (-((2.0 * (r - 1.0) * step_size * fixture.eta).exp_m1())).sqrt();
        for (actual, fixture_value) in [
            (
                lambda_source,
                expected.lambda_source.ok_or("missing source lambda")?,
            ),
            (
                lambda_target,
                expected.lambda_target.ok_or("missing target lambda")?,
            ),
            (step_size, expected.step_size.ok_or("missing step size")?),
            (
                eta_step_size,
                expected.eta_step_size.ok_or("missing eta step size")?,
            ),
            (
                intermediate_sigma,
                expected
                    .intermediate_sigma
                    .ok_or("missing intermediate sigma")?,
            ),
            (
                alpha_target,
                expected.alpha_target.ok_or("missing target alpha")?,
            ),
            (phi_one, expected.phi_1.ok_or("missing phi one")?),
            (phi_two, expected.phi_2.ok_or("missing phi two")?),
            (b1, expected.b1.ok_or("missing b1")?),
            (b2, expected.b2.ok_or("missing b2")?),
            (
                first_noise_root,
                expected
                    .first_noise_root
                    .ok_or("missing first noise root")?,
            ),
            (
                first_noise_root * intermediate_sigma * fixture.effective_noise_scale,
                expected
                    .predictor_noise_coefficient
                    .ok_or("missing predictor noise coefficient")?,
            ),
            (
                second_noise_weight,
                expected
                    .second_noise_weight
                    .ok_or("missing second noise weight")?,
            ),
        ] {
            assert!((actual - fixture_value).abs() <= fixture.tolerance);
        }
        assert_eq!(second_noise_weight, 0.0);
        let first_noise = expected
            .first_noise
            .as_deref()
            .ok_or("missing first noise")?;
        let second_noise = expected
            .second_noise
            .as_deref()
            .ok_or("missing second noise")?;
        let predictor = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .zip(first_noise)
            .map(|((latent, primary), noise)| {
                intermediate_sigma / expected.sigma * (-step_size * fixture.eta).exp() * latent
                    - alpha_target * phi_one * primary
                    + first_noise_root * noise * intermediate_sigma * fixture.effective_noise_scale
            })
            .collect::<Vec<_>>();
        assert_close(
            &predictor,
            expected.predictor.as_deref().ok_or("missing predictor")?,
            fixture.tolerance,
        );
        let corrector = expected.corrector.as_deref().ok_or("missing corrector")?;
        let deterministic = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .zip(corrector)
            .map(|((latent, primary), corrector)| {
                expected.next_sigma / expected.sigma * (-step_size * fixture.eta).exp() * latent
                    - alpha_target * (b1 * primary + b2 * corrector)
            })
            .collect::<Vec<_>>();
        assert_close(
            &deterministic,
            expected
                .deterministic
                .as_deref()
                .ok_or("missing deterministic")?,
            fixture.tolerance,
        );
        let output = deterministic
            .iter()
            .zip(first_noise)
            .zip(second_noise)
            .map(|((value, first), second)| {
                value
                    + (first_noise_root * first + second_noise_weight * second)
                        * expected.next_sigma
                        * fixture.effective_noise_scale
            })
            .collect::<Vec<_>>();
        assert_close(&output, &expected.latent_after, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_sampler_001_signed_eta_and_noise_scale_match_source_gating() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    for (eta, noise_scale) in [
        (fixture.signed_controls.eta_above_negative_one, 1.0),
        (fixture.signed_controls.eta_below_negative_one, 1.0),
        (fixture.eta, fixture.signed_controls.negative_noise_scale),
    ] {
        let expected_latents = signed_source_oracle(&fixture, eta)?;
        let (trace, before, after) = sample_exp_heun_2_x0_sde(
            &backend,
            plan(&fixture, &fixture.identity, &profile)?,
            &profile,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            noise_request(&fixture),
            ExpHeun2X0SdeOptions {
                eta,
                noise_scale,
                solver_type: ExpHeun2X0SolverType::Phi2,
            },
            &context,
            |_, _, step, stage| {
                let expected = &fixture.steps[step];
                let values = match stage {
                    ExpHeun2X0DenoiserStage::Primary => &expected.primary,
                    ExpHeun2X0DenoiserStage::Corrector => expected
                        .corrector
                        .as_ref()
                        .ok_or_else(|| "missing signed-control corrector".to_owned())?,
                };
                tensor_from_f32(&backend, &fixture.shape, values, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<_, String>(()),
        )?;
        assert_eq!(before, after);
        assert_eq!(trace.latents.len(), expected_latents.len());
        for (actual, expected) in trace.latents.iter().zip(expected_latents) {
            assert_close(
                &values(&backend, actual, &context)?,
                &expected,
                fixture.tolerance,
            );
        }
    }

    let singular = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: fixture.signed_controls.singular_eta,
            noise_scale: 1.0,
            solver_type: ExpHeun2X0SolverType::Phi2,
        },
        &context,
        |_, _, step, _| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].primary,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("eta == -1 must fail at the source equation singularity");
    assert!(matches!(
        singular,
        ExpHeun2X0SdeError::Family(ExpHeun2X0Error::InvalidCoefficient {
            coefficient: "phi two",
            ..
        })
    ));
    Ok(())
}

#[test]
fn val_rng_001_exp_heun_2_x0_sde_boundaries_failures_cancellation_and_replay_are_atomic()
-> Result<(), Box<dyn Error>> {
    let cuda = DeviceId::from_source_device("cuda:0")?;
    assert!(matches!(
        validate_exp_heun_2_x0_sde_generation_device(cuda),
        Err(ExpHeun2X0SdeError::DeviceUnavailable { device, .. }) if device == cuda
    ));
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);
    let error = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, "dpmpp_2m", &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions::default(),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("another registered sampler must not substitute this row");
    assert!(matches!(error, ExpHeun2X0SdeError::WrongSampler(_)));
    let error = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: f32::NAN,
            ..Default::default()
        },
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-finite eta must fail before RNG publication");
    assert!(matches!(
        error,
        ExpHeun2X0SdeError::InvalidOption { name: "eta", .. }
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions::default(),
        &cancelled_context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, ExpHeun2X0SdeError::Tensor(_)));

    let deterministic_outputs = [
        [0.3_f32, -0.2],
        [0.25, -0.1],
        [0.2, -0.05],
        [0.15, 0.0],
        [0.1, 0.02],
    ];
    let mut deterministic_call = 0_usize;
    let deterministic = sample_exp_heun_2_x0(
        &backend,
        SamplingPlan::new(
            "exp_heun_2_x0",
            "normal",
            profile.identity().clone(),
            fixture.seed,
            u32::try_from(fixture.steps.len())?,
            1.0,
            1.0,
        )?,
        &profile,
        initial()?,
        &fixture.sigmas,
        ExpHeun2X0SolverType::Phi2,
        &context,
        |_, _, _, _| {
            let output = deterministic_outputs
                .get(deterministic_call)
                .ok_or("unexpected deterministic call")?;
            deterministic_call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    let mut sde_call = 0_usize;
    let (eta_zero, eta_zero_before, eta_zero_after) = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: 0.0,
            noise_scale: 1.0,
            solver_type: ExpHeun2X0SolverType::Phi2,
        },
        &context,
        |_, _, _, _| {
            let output = deterministic_outputs
                .get(sde_call)
                .ok_or("unexpected eta-zero call")?;
            sde_call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(eta_zero_before, eta_zero_after);
    assert_eq!(deterministic.latents.len(), eta_zero.latents.len());
    for (deterministic, stochastic) in deterministic.latents.iter().zip(&eta_zero.latents) {
        assert_close(
            &values(&backend, deterministic, &context)?,
            &values(&backend, stochastic, &context)?,
            0.0,
        );
    }

    let mut zero_noise_call = 0_usize;
    let (_, zero_noise_before, zero_noise_after) = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: fixture.eta,
            noise_scale: 0.0,
            solver_type: ExpHeun2X0SolverType::Phi1,
        },
        &context,
        |_, _, _, _| {
            let output = deterministic_outputs
                .get(zero_noise_call)
                .ok_or("unexpected zero-noise call")?;
            zero_noise_call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(zero_noise_before, zero_noise_after);

    let failed_call = RefCell::new(0_usize);
    let failure = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: fixture.eta,
            noise_scale: fixture.sampler_noise_scale,
            solver_type: ExpHeun2X0SolverType::Phi2,
        },
        &context,
        |_, _, step, stage| {
            if step == 0 && stage == ExpHeun2X0DenoiserStage::Corrector {
                return Err("injected corrector failure".to_owned());
            }
            let mut call = failed_call.borrow_mut();
            let output = deterministic_outputs
                .get(*call)
                .ok_or("unexpected failure call")?;
            *call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("failure after first noise draw must abort the RNG transaction");
    assert!(matches!(
        failure,
        ExpHeun2X0SdeError::Denoiser {
            step: 0,
            stage: ExpHeun2X0DenoiserStage::Corrector,
            ..
        }
    ));

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let mut callback_cancel_call = 0_usize;
    let cancelled = sample_exp_heun_2_x0_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(
            &backend,
            &fixture.shape,
            &fixture.initial,
            &callback_context,
        )?,
        &fixture.sigmas,
        noise_request(&fixture),
        ExpHeun2X0SdeOptions {
            eta: fixture.eta,
            noise_scale: fixture.sampler_noise_scale,
            solver_type: ExpHeun2X0SolverType::Phi2,
        },
        &callback_context,
        |_, _, _, _| {
            let output = deterministic_outputs
                .get(callback_cancel_call)
                .ok_or("unexpected callback-cancellation call")?;
            callback_cancel_call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &callback_context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| {
            callback_cancellation.cancel();
            Ok::<_, String>(())
        },
    )
    .expect_err("callback cancellation must abort before the first RNG draw");
    assert!(matches!(
        cancelled,
        ExpHeun2X0SdeError::Sampling(sim_sampler::SamplingError::Cancelled)
    ));

    let run = || -> Result<_, Box<dyn Error>> {
        let mut call = 0_usize;
        Ok(sample_exp_heun_2_x0_sde(
            &backend,
            plan(&fixture, &fixture.identity, &profile)?,
            &profile,
            initial()?,
            &fixture.sigmas,
            noise_request(&fixture),
            ExpHeun2X0SdeOptions {
                eta: fixture.eta,
                noise_scale: fixture.sampler_noise_scale,
                solver_type: ExpHeun2X0SolverType::Phi2,
            },
            &context,
            |_, _, _, _| {
                let output = deterministic_outputs
                    .get(call)
                    .ok_or("unexpected replay call")?;
                call += 1;
                tensor_from_f32(&backend, &fixture.shape, output, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<_, String>(()),
        )?)
    };
    let (first, first_before, first_after) = run()?;
    let (replay, replay_before, replay_after) = run()?;
    assert_eq!((first_before, first_after), (replay_before, replay_after));
    for (first, replay) in first.latents.iter().zip(&replay.latents) {
        assert_close(
            &values(&backend, first, &context)?,
            &values(&backend, replay, &context)?,
            0.0,
        );
    }
    Ok(())
}
