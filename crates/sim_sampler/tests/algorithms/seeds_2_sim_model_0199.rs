use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_exp_heun_2_x0_sim_model_0183::{
        ExpHeun2X0DenoiserStage, ExpHeun2X0SolverType, sample_exp_heun_2_x0,
    },
    generated_seeds_2_sim_model_0199::{
        DEFINITION, SEEDS_2_FEATURE_ID, SEEDS_2_NOISE_CONTRACT_ID, SEEDS_2_SAMPLER_ID,
        SEEDS_2_SOURCE_ORDINAL, Seeds2DenoiserStage, Seeds2Error, Seeds2Options, Seeds2SolverType,
        sample_seeds_2, seeds_2_rng_profile, validate_seeds_2_generation_device,
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
    "/../sim_test_support/fixtures/samplers/seeds_2_comfy_model_0199/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
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
    intermediate_step_ratio: f32,
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
    intermediate_lambda: Option<f32>,
    intermediate_sigma: Option<f32>,
    intermediate_alpha: Option<f32>,
    alpha_target: Option<f32>,
    intermediate_phi_1: Option<f32>,
    phi_1: Option<f32>,
    phi_2: Option<f32>,
    b1: Option<f32>,
    b2: Option<f32>,
    predictor_latent_weight: Option<f32>,
    predictor_denoised_weight: Option<f32>,
    first_noise_root: Option<f32>,
    predictor_noise_coefficient: Option<f32>,
    first_noise: Option<Vec<f32>>,
    second_noise: Option<Vec<f32>>,
    first_segment_noise_weight: Option<f32>,
    second_segment_noise_weight: Option<f32>,
    predictor_deterministic: Option<Vec<f32>>,
    predictor: Option<Vec<f32>>,
    intermediate: Option<Vec<f32>>,
    deterministic: Option<Vec<f32>>,
    sde_noise: Option<Vec<f32>>,
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

fn source_range(source: &str, lines: [usize; 2]) -> String {
    source
        .lines()
        .skip(lines[0].saturating_sub(1))
        .take(lines[1].saturating_sub(lines[0]) + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

fn sampler_names(source: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let start = source
        .find("KSAMPLER_NAMES = [")
        .ok_or("missing registry")?;
    let source = source.get(start..).ok_or("invalid registry start")?;
    let end = source.find(']').ok_or("unterminated registry")?;
    let source = source.get(..end).ok_or("invalid registry range")?;
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
        SamplingProfileIdentity::new("seeds-2-row-v1")?,
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

fn noise_request(fixture: &Fixture, retry: u32) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        &fixture.rng.workflow,
        &fixture.rng.attempt,
        &fixture.rng.node,
        fixture.rng.output,
        fixture.rng.execution_ordinal,
        fixture.rng.batch,
        retry,
        RetryRngPolicy::Replay,
    )
}

fn options(fixture: &Fixture, solver_type: Seeds2SolverType) -> Seeds2Options {
    Seeds2Options {
        eta: fixture.eta,
        noise_scale: fixture.sampler_noise_scale,
        intermediate_step_ratio: fixture.intermediate_step_ratio,
        solver_type,
    }
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

#[test]
fn val_sampler_001_seeds_2_definition_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, SEEDS_2_SAMPLER_ID);
    assert_eq!(fixture.feature_id, SEEDS_2_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, SEEDS_2_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 37);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/seeds_2_comfy_model_0199"
    );
    assert_eq!(fixture.rng.contract, SEEDS_2_NOISE_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.seed_transform, "add-one-on-cpu");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(SEEDS_2_SAMPLER_ID)?)?,
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
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "def sample_seeds_2(",
        "solver_type not in {\"phi_1\", \"phi_2\"}",
        "inject_noise = eta > 0 and s_noise > 0",
        "sigmas = offset_first_sigma_for_snr",
        "callback({'x': x",
        "lambda_s_1 = torch.lerp(lambda_s, lambda_t, r)",
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
            .is_some_and(|line| line.contains("\"seeds_2\""))
    );
    let names = sampler_names(&samplers)?;
    assert_eq!(
        names.iter().position(|name| name == SEEDS_2_SAMPLER_ID),
        Some(37)
    );
    assert_eq!(
        names
            .iter()
            .filter(|name| *name == SEEDS_2_SAMPLER_ID)
            .count(),
        1
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(
                |line| line.contains("sampler,seeds_2,") && line.ends_with(",COMFY-MODEL-0199")
            )
    );

    assert_eq!(IMPLEMENTATION.matches(".draw_normal(").count(), 2);
    for required in [
        "SamplingSession::new(",
        "profile.adjust_first_sigma_for_snr(",
        "request.open_transaction(",
        "backend.workspace_vec::<f32>(",
        "observed.commit(",
        "CompatibilityRngTransaction::commit",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing canonical owner use {required}"
        );
    }
    for forbidden in [
        "struct SamplingSession",
        "struct CompatibilityRngTransaction",
        "struct RngStream",
        "struct Seeds2Trace",
        "struct Seeds2Progress",
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
    let cuda = DeviceId::from_source_device("cuda:3")?;
    assert_eq!(
        seeds_2_rng_profile(DeviceId::CPU),
        (
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
        )
    );
    assert_eq!(
        seeds_2_rng_profile(cuda),
        (
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(cuda),
        )
    );
    Ok(())
}

#[test]
fn val_sampler_001_seeds_2_matches_every_intermediate_callback_and_rng_draw()
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
    let (trace, noise_before, noise_after) = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture, fixture.rng.retry),
        options(&fixture, Seeds2SolverType::Phi2),
        &context,
        |latent, sigma, step, stage| {
            events.borrow_mut().push(format!("{stage:?}:{step}"));
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            let (expected_sigma, expected_latent, output) = match stage {
                Seeds2DenoiserStage::Primary => (
                    expected.sigma,
                    expected.latent_before.as_slice(),
                    expected.primary.as_slice(),
                ),
                Seeds2DenoiserStage::Intermediate => (
                    expected
                        .intermediate_sigma
                        .ok_or("missing intermediate sigma")?,
                    expected.predictor.as_deref().ok_or("missing predictor")?,
                    expected
                        .intermediate
                        .as_deref()
                        .ok_or("missing intermediate")?,
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
            "Intermediate:0",
            "Primary:1",
            "Callback:1",
            "Intermediate:1",
            "Primary:2",
            "Callback:2",
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
        SEEDS_2_NOISE_CONTRACT_ID,
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
fn val_sampling_foundation_001_seeds_2_fixture_reconstructs_both_solver_branches()
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
        let intermediate_lambda =
            lambda_source + (lambda_target - lambda_source) * fixture.intermediate_step_ratio;
        let intermediate_sigma = (-intermediate_lambda).exp();
        let intermediate_alpha = intermediate_sigma * intermediate_lambda.exp();
        let alpha_target = expected.next_sigma * lambda_target.exp();
        let intermediate_phi_one = (-fixture.intermediate_step_ratio * eta_step_size).exp_m1();
        let phi_one = (-eta_step_size).exp_m1();
        let phi_two = (phi_one + eta_step_size) / -eta_step_size;
        let b2 = phi_two / fixture.intermediate_step_ratio;
        let b1 = phi_one - b2;
        let predictor_latent_weight = intermediate_sigma / expected.sigma
            * (-fixture.intermediate_step_ratio * step_size * fixture.eta).exp();
        let predictor_denoised_weight = -intermediate_alpha * intermediate_phi_one;
        let first_noise_root =
            (-(-2.0 * fixture.intermediate_step_ratio * step_size * fixture.eta).exp_m1()).sqrt();
        let segment_factor = (fixture.intermediate_step_ratio - 1.0) * step_size * fixture.eta;
        let first_segment_noise_weight = segment_factor.exp();
        let second_segment_noise_weight = (-(2.0 * segment_factor).exp_m1()).sqrt();
        for (actual, fixture_value) in [
            (
                lambda_source,
                expected.lambda_source.ok_or("missing lambda source")?,
            ),
            (
                lambda_target,
                expected.lambda_target.ok_or("missing lambda target")?,
            ),
            (step_size, expected.step_size.ok_or("missing step size")?),
            (
                eta_step_size,
                expected.eta_step_size.ok_or("missing eta step")?,
            ),
            (
                intermediate_lambda,
                expected
                    .intermediate_lambda
                    .ok_or("missing intermediate lambda")?,
            ),
            (
                intermediate_sigma,
                expected
                    .intermediate_sigma
                    .ok_or("missing intermediate sigma")?,
            ),
            (
                intermediate_alpha,
                expected
                    .intermediate_alpha
                    .ok_or("missing intermediate alpha")?,
            ),
            (
                alpha_target,
                expected.alpha_target.ok_or("missing target alpha")?,
            ),
            (
                intermediate_phi_one,
                expected
                    .intermediate_phi_1
                    .ok_or("missing intermediate phi one")?,
            ),
            (phi_one, expected.phi_1.ok_or("missing phi one")?),
            (phi_two, expected.phi_2.ok_or("missing phi two")?),
            (b1, expected.b1.ok_or("missing b1")?),
            (b2, expected.b2.ok_or("missing b2")?),
            (
                predictor_latent_weight,
                expected
                    .predictor_latent_weight
                    .ok_or("missing predictor latent weight")?,
            ),
            (
                predictor_denoised_weight,
                expected
                    .predictor_denoised_weight
                    .ok_or("missing predictor denoised weight")?,
            ),
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
                first_segment_noise_weight,
                expected
                    .first_segment_noise_weight
                    .ok_or("missing first segment weight")?,
            ),
            (
                second_segment_noise_weight,
                expected
                    .second_segment_noise_weight
                    .ok_or("missing second segment weight")?,
            ),
        ] {
            assert!((actual - fixture_value).abs() <= fixture.tolerance);
        }
        let first_noise = expected
            .first_noise
            .as_deref()
            .ok_or("missing first noise")?;
        let second_noise = expected
            .second_noise
            .as_deref()
            .ok_or("missing second noise")?;
        let predictor_deterministic = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .map(|(latent, primary)| {
                predictor_latent_weight * latent + predictor_denoised_weight * primary
            })
            .collect::<Vec<_>>();
        assert_close(
            &predictor_deterministic,
            expected
                .predictor_deterministic
                .as_deref()
                .ok_or("missing predictor deterministic")?,
            fixture.tolerance,
        );
        let predictor = predictor_deterministic
            .iter()
            .zip(first_noise)
            .map(|(value, noise)| {
                value
                    + first_noise_root * intermediate_sigma * fixture.effective_noise_scale * noise
            })
            .collect::<Vec<_>>();
        assert_close(
            &predictor,
            expected.predictor.as_deref().ok_or("missing predictor")?,
            fixture.tolerance,
        );
        let intermediate = expected
            .intermediate
            .as_deref()
            .ok_or("missing intermediate")?;
        let deterministic = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .zip(intermediate)
            .map(|((latent, primary), intermediate)| {
                expected.next_sigma / expected.sigma * (-step_size * fixture.eta).exp() * latent
                    - alpha_target * (b1 * primary + b2 * intermediate)
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
        let sde_noise = first_noise
            .iter()
            .zip(second_noise)
            .map(|(first, second)| {
                first_noise_root * first * first_segment_noise_weight
                    + second_segment_noise_weight * second
            })
            .collect::<Vec<_>>();
        assert_close(
            &sde_noise,
            expected.sde_noise.as_deref().ok_or("missing sde noise")?,
            fixture.tolerance,
        );
        let output = deterministic
            .iter()
            .zip(&sde_noise)
            .map(|(value, noise)| {
                value + noise * expected.next_sigma * fixture.effective_noise_scale
            })
            .collect::<Vec<_>>();
        assert_close(&output, &expected.latent_after, fixture.tolerance);

        let phi_one_factor = 1.0 / (2.0 * fixture.intermediate_step_ratio);
        let phi_one_primary = phi_one * (1.0 - phi_one_factor);
        let phi_one_intermediate = phi_one * phi_one_factor;
        let phi_one_output = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .zip(intermediate)
            .map(|((latent, primary), intermediate)| {
                expected.next_sigma / expected.sigma * (-step_size * fixture.eta).exp() * latent
                    - alpha_target
                        * (phi_one_primary * primary + phi_one_intermediate * intermediate)
            })
            .collect::<Vec<_>>();
        assert!(phi_one_output.iter().all(|value| value.is_finite()));
        assert_ne!(phi_one_output, deterministic);
    }
    Ok(())
}

fn run_with_fixture_outputs(
    backend: &CpuBackend,
    fixture: &Fixture,
    profile: &DiscreteSamplingProfile,
    context: &ExecutionContext<'_>,
    retry: u32,
    options: Seeds2Options,
) -> Result<
    (
        sim_sampler::SamplingTrace,
        sim_tensor::RngCheckpoint,
        sim_tensor::RngCheckpoint,
    ),
    Seeds2Error,
> {
    sample_seeds_2(
        backend,
        SamplingPlan::new(
            &fixture.identity,
            "normal",
            profile.identity().clone(),
            fixture.seed,
            u32::try_from(fixture.steps.len()).map_err(|_| {
                sim_sampler::SamplingError::OutOfMemory("fixture step conversion")
            })?,
            1.0,
            1.0,
        )?,
        profile,
        tensor_from_f32(backend, &fixture.shape, &fixture.initial, context)?,
        &fixture.sigmas,
        noise_request(fixture, retry),
        options,
        context,
        |_, _, step, stage| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| "unexpected fixture step".to_owned())?;
            let output = match stage {
                Seeds2DenoiserStage::Primary => &expected.primary,
                Seeds2DenoiserStage::Intermediate => expected
                    .intermediate
                    .as_ref()
                    .ok_or_else(|| "missing fixture intermediate".to_owned())?,
            };
            tensor_from_f32(backend, &fixture.shape, output, context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
}

#[test]
fn val_sampler_001_seeds_2_phi_one_and_exp_heun_specialization_preserve_semantics()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let (phi_one, _, _) = run_with_fixture_outputs(
        &backend,
        &fixture,
        &profile,
        &context,
        0,
        options(&fixture, Seeds2SolverType::Phi1),
    )?;
    let (phi_two, _, _) = run_with_fixture_outputs(
        &backend,
        &fixture,
        &profile,
        &context,
        0,
        options(&fixture, Seeds2SolverType::Phi2),
    )?;
    assert!(
        phi_one
            .latents
            .iter()
            .zip(&phi_two.latents)
            .any(|(first, second)| values(&backend, first, &context).ok()
                != values(&backend, second, &context).ok())
    );

    let deterministic_outputs = [
        [0.3_f32, -0.2],
        [0.25, -0.1],
        [0.2, -0.05],
        [0.15, 0.0],
        [0.1, 0.02],
    ];
    let mut seeds_call = 0_usize;
    let (seeds, before, after) = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        Seeds2Options {
            eta: 0.0,
            noise_scale: 1.0,
            intermediate_step_ratio: 1.0,
            solver_type: Seeds2SolverType::Phi2,
        },
        &context,
        |_, _, _, _| {
            let output = deterministic_outputs
                .get(seeds_call)
                .ok_or("unexpected SEEDS specialization call")?;
            seeds_call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(before, after);
    let mut exp_call = 0_usize;
    let exp = sample_exp_heun_2_x0(
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
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        ExpHeun2X0SolverType::Phi2,
        &context,
        |_, _, _, stage| {
            let expected_stage = if exp_call.is_multiple_of(2) {
                ExpHeun2X0DenoiserStage::Primary
            } else {
                ExpHeun2X0DenoiserStage::Corrector
            };
            if exp_call < deterministic_outputs.len() - 1 && stage != expected_stage {
                return Err("unexpected exp-Heun stage".to_owned());
            }
            let output = deterministic_outputs
                .get(exp_call)
                .ok_or("unexpected exp-Heun call")?;
            exp_call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    for (seeds, exp) in seeds.latents.iter().zip(&exp.latents) {
        assert_close(
            &values(&backend, seeds, &context)?,
            &values(&backend, exp, &context)?,
            fixture.tolerance,
        );
    }
    Ok(())
}

#[test]
fn val_rng_001_seeds_2_failures_cancellation_retry_and_commits_are_atomic()
-> Result<(), Box<dyn Error>> {
    let cuda = DeviceId::from_source_device("cuda:0")?;
    assert!(matches!(
        validate_seeds_2_generation_device(cuda),
        Err(Seeds2Error::DeviceUnavailable { device, .. }) if device == cuda
    ));
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);

    let error = sample_seeds_2(
        &backend,
        plan(&fixture, "dpmpp_2m", &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        Seeds2Options::default(),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("another sampler identity must not substitute SEEDS-2");
    assert!(matches!(error, Seeds2Error::WrongSampler { .. }));
    for invalid in [
        Seeds2Options {
            eta: f32::NAN,
            ..Default::default()
        },
        Seeds2Options {
            noise_scale: f32::INFINITY,
            ..Default::default()
        },
        Seeds2Options {
            intermediate_step_ratio: 0.0,
            ..Default::default()
        },
    ] {
        let error = sample_seeds_2(
            &backend,
            plan(&fixture, &fixture.identity, &profile)?,
            &profile,
            initial()?,
            &fixture.sigmas,
            noise_request(&fixture, 0),
            invalid,
            &context,
            |latent, _, _, _| Ok(latent.clone()),
            |_, _, _| Ok::<_, String>(()),
        )
        .expect_err("invalid option must fail before RNG publication");
        assert!(matches!(error, Seeds2Error::InvalidOption { .. }));
    }

    let singular = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        Seeds2Options {
            eta: -1.0,
            noise_scale: 1.0,
            intermediate_step_ratio: 0.5,
            solver_type: Seeds2SolverType::Phi2,
        },
        &context,
        |_, _, step, stage| {
            let expected = &fixture.steps[step];
            let output = match stage {
                Seeds2DenoiserStage::Primary => &expected.primary,
                Seeds2DenoiserStage::Intermediate => expected
                    .intermediate
                    .as_ref()
                    .ok_or("missing intermediate")?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("phi two is singular at eta == -1");
    assert!(matches!(
        singular,
        Seeds2Error::InvalidCoefficient {
            coefficient: "phi two",
            ..
        }
    ));

    let descriptor = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        options(&fixture, Seeds2SolverType::Phi2),
        &context,
        |_, _, _, _| {
            tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("descriptor changes must be typed failures");
    assert!(matches!(
        descriptor,
        Seeds2Error::DenoiserContract {
            step: 0,
            stage: Seeds2DenoiserStage::Primary,
        }
    ));

    let callback = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        options(&fixture, Seeds2SolverType::Phi2),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Err::<(), _>("injected callback failure"),
    )
    .expect_err("callback failure must abort before noise publication");
    assert!(matches!(callback, Seeds2Error::Sampling(_)));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        options(&fixture, Seeds2SolverType::Phi2),
        &cancelled_context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, Seeds2Error::Tensor(_)));

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let cancelled = sample_seeds_2(
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
        noise_request(&fixture, 0),
        options(&fixture, Seeds2SolverType::Phi2),
        &callback_context,
        |_, _, step, _| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].primary,
                &callback_context,
            )
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
        Seeds2Error::Sampling(sim_sampler::SamplingError::Cancelled)
    ));

    let failed = sample_seeds_2(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture, 0),
        options(&fixture, Seeds2SolverType::Phi2),
        &context,
        |_, _, step, stage| {
            if step == 0 && stage == Seeds2DenoiserStage::Intermediate {
                return Err("injected intermediate failure".to_owned());
            }
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
    .expect_err("failure after the first draw must abort the transaction");
    assert!(matches!(
        failed,
        Seeds2Error::Denoiser {
            step: 0,
            stage: Seeds2DenoiserStage::Intermediate,
            ..
        }
    ));

    for (eta, noise_scale, solver_type) in [
        (0.0, 1.0, Seeds2SolverType::Phi2),
        (fixture.eta, 0.0, Seeds2SolverType::Phi1),
        (-0.5, 1.0, Seeds2SolverType::Phi1),
        (fixture.eta, -1.0, Seeds2SolverType::Phi2),
    ] {
        let (_, before, after) = run_with_fixture_outputs(
            &backend,
            &fixture,
            &profile,
            &context,
            0,
            Seeds2Options {
                eta,
                noise_scale,
                intermediate_step_ratio: fixture.intermediate_step_ratio,
                solver_type,
            },
        )?;
        assert_eq!(before, after, "disabled noise must consume no draws");
    }

    let first = run_with_fixture_outputs(
        &backend,
        &fixture,
        &profile,
        &context,
        0,
        options(&fixture, Seeds2SolverType::Phi2),
    )?;
    let replay = run_with_fixture_outputs(
        &backend,
        &fixture,
        &profile,
        &context,
        7,
        options(&fixture, Seeds2SolverType::Phi2),
    )?;
    assert_eq!((first.1, first.2), (replay.1, replay.2));
    for (first, replay) in first.0.latents.iter().zip(&replay.0.latents) {
        assert_close(
            &values(&backend, first, &context)?,
            &values(&backend, replay, &context)?,
            0.0,
        );
    }
    Ok(())
}
