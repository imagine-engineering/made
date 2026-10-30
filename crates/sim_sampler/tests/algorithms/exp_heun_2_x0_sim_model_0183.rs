use sim_sampler::{
    DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity, SamplerRegistry,
    SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_exp_heun_2_x0_sim_model_0183::{
        DEFINITION, EXP_HEUN_2_X0_FEATURE_ID, EXP_HEUN_2_X0_SAMPLER_ID,
        EXP_HEUN_2_X0_SOURCE_ORDINAL, ExpHeun2X0DenoiserStage, ExpHeun2X0Error,
        ExpHeun2X0SolverType, sample_exp_heun_2_x0,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/exp_heun_2_x0_comfy_model_0183/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/exp_heun_2_x0_sim_model_0183.rs"
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
    wrapper_lines: [usize; 2],
    equation_lines: [usize; 2],
    phi_lines: [usize; 2],
    profile_lines: [usize; 2],
    registry_line: usize,
    catalog_line: usize,
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
    intermediate_sigma: Option<f32>,
    alpha_target: Option<f32>,
    phi_1: Option<f32>,
    phi_2: Option<f32>,
    b1: Option<f32>,
    b2: Option<f32>,
    predictor: Option<Vec<f32>>,
    corrector: Option<Vec<f32>>,
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("exp-heun-2-x0-row-v1")?,
        PredictionInterpretation::Denoised,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
        SamplingSnrMode::Standard,
        1.0,
    )?)
}

fn plan(
    identity: &str,
    profile: &DiscreteSamplingProfile,
    steps: usize,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile.identity().clone(),
        183,
        u32::try_from(steps)?,
        1.0,
        1.0,
    )?)
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
fn val_sampler_001_exp_heun_2_x0_definition_source_ordinal_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, EXP_HEUN_2_X0_SAMPLER_ID);
    assert_eq!(fixture.feature_id, EXP_HEUN_2_X0_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, EXP_HEUN_2_X0_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 6);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/exp_heun_2_x0_comfy_model_0183"
    );
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(EXP_HEUN_2_X0_SAMPLER_ID)?)?,
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
        "def sample_exp_heun_2_x0(",
        "eta=0.0",
        "s_noise=0.0",
        "r=1.0",
    ] {
        assert!(wrapper.contains(fragment), "missing wrapper {fragment}");
    }
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "solver_type not in {\"phi_1\", \"phi_2\"}",
        "callback({'x': x",
        "lambda_s_1 = torch.lerp",
        "# Step 1",
        "denoised_2 = model(x_2",
        "# Step 2",
        "if solver_type == \"phi_1\"",
        "elif solver_type == \"phi_2\"",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let phi = source_range(&sampling, fixture.source.phi_lines);
    assert!(phi.contains("return torch.expm1(h)"));
    assert!(phi.contains("return (torch.expm1(h) - h) / h"));
    let profile_source = source_range(&sampling, fixture.source.profile_lines);
    assert!(profile_source.contains("def sigma_to_half_log_snr"));
    assert!(profile_source.contains("def offset_first_sigma_for_snr"));

    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"exp_heun_2_x0\""))
    );
    let names = sampler_names(&samplers)?;
    let ordinal = names
        .iter()
        .position(|name| name == EXP_HEUN_2_X0_SAMPLER_ID)
        .ok_or("sampler is absent from KSAMPLER_NAMES")?;
    assert_eq!(u16::try_from(ordinal)?, EXP_HEUN_2_X0_SOURCE_ORDINAL);
    assert_eq!(
        names
            .iter()
            .filter(|name| *name == EXP_HEUN_2_X0_SAMPLER_ID)
            .count(),
        1
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,exp_heun_2_x0,")
                && line.ends_with(",COMFY-MODEL-0183"))
    );

    for required in [
        "sample_seeds_2_deterministic_family(",
        "eta: 0.0",
        "noise_scale: 0.0",
        "intermediate_step_ratio: 1.0",
        "map_solver_type(solver_type)",
        "map_denoiser_stage(stage)",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing owner {required}"
        );
    }
    for forbidden in [
        "CompatibilityNoiseRequest",
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStream",
        "noise_transaction",
        "SamplingSession::new",
        ".observe_step(",
        "profile.adjust_first_sigma_for_snr(",
        "profile.half_log_snr(",
        "profile.sigma_from_half_log_snr(",
        ".draw_normal(",
        "struct ExpHeun2X0Trace",
        "struct ExpHeun2X0Progress",
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
fn val_sampler_001_exp_heun_2_x0_matches_predictors_correctors_and_callback_order()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.solver_type, "phi_2");
    let profile = profile()?;
    assert_eq!(profile.prediction(), PredictionInterpretation::Denoised);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let events = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let trace = sample_exp_heun_2_x0(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial,
        &fixture.sigmas,
        ExpHeun2X0SolverType::Phi2,
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
        assert!((progress.next_sigma - expected.next_sigma).abs() <= fixture.tolerance);
        assert_close(latent, &expected.latent_before, fixture.tolerance);
        assert_close(denoised, &expected.primary, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_exp_heun_2_x0_fixture_reconstructs_every_phi_two_equation()
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
        let intermediate_sigma = (-(lambda_source + step_size)).exp();
        let alpha_target = expected.next_sigma * lambda_target.exp();
        let phi_one = (-step_size).exp_m1();
        let phi_two = (phi_one + step_size) / -step_size;
        let b2 = phi_two;
        let b1 = phi_one - b2;
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
        ] {
            assert!((actual - fixture_value).abs() <= fixture.tolerance);
        }
        let predictor = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .map(|(latent, denoised)| {
                intermediate_sigma / expected.sigma * latent - alpha_target * phi_one * denoised
            })
            .collect::<Vec<_>>();
        assert_close(
            &predictor,
            expected.predictor.as_deref().ok_or("missing predictor")?,
            fixture.tolerance,
        );
        let corrector = expected.corrector.as_deref().ok_or("missing corrector")?;
        let output = expected
            .latent_before
            .iter()
            .zip(&expected.primary)
            .zip(corrector)
            .map(|((latent, primary), corrector)| {
                expected.next_sigma / expected.sigma * latent
                    - alpha_target * (b1 * primary + b2 * corrector)
            })
            .collect::<Vec<_>>();
        assert_close(&output, &expected.latent_after, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_sampler_001_exp_heun_2_x0_phi_one_failures_and_cancellation_are_typed()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);
    let error = sample_exp_heun_2_x0(
        &backend,
        plan("dpmpp_2m", &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        ExpHeun2X0SolverType::Phi2,
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("another registered sampler must not substitute this row");
    assert!(matches!(error, ExpHeun2X0Error::WrongSampler(_)));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_exp_heun_2_x0(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        ExpHeun2X0SolverType::Phi2,
        &cancelled_context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before evaluation");
    assert!(matches!(error, ExpHeun2X0Error::Tensor(_)));

    let sigmas = [2.0, 1.0, 0.0];
    let outputs = [[0.3, -0.2], [0.25, -0.1], [0.0, 0.0]];
    let mut call = 0_usize;
    let trace = sample_exp_heun_2_x0(
        &backend,
        plan(&fixture.identity, &profile, 2)?,
        &profile,
        initial()?,
        &sigmas,
        ExpHeun2X0SolverType::Phi1,
        &context,
        |_, _, _, _| {
            let output = outputs.get(call).ok_or("unexpected phi-one call")?;
            call += 1;
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(call, 3);
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &[0.5125, -0.575],
        fixture.tolerance,
    );
    assert_close(
        &values(
            &backend,
            trace.latents.last().ok_or("missing phi-one terminal")?,
            &context,
        )?,
        &[0.0, 0.0],
        fixture.tolerance,
    );
    Ok(())
}
