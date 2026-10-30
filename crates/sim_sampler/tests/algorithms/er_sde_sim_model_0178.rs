use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_er_sde_sim_model_0178::{
        DEFINITION, ER_SDE_FEATURE_ID, ER_SDE_NOISE_CONTRACT_ID, ER_SDE_SAMPLER_ID,
        ER_SDE_SOURCE_ORDINAL, ErSdeError, ErSdeOptions, er_sde_rng_profile, sample_er_sde,
        sample_er_sde_with_noise_scaler, validate_er_sde_generation_device,
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
    "/../sim_test_support/fixtures/samplers/er_sde_comfy_model_0178/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/er_sde_sim_model_0178.rs"
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
    max_stage: usize,
    sampler_noise_scale: f32,
    profile_noise_scale: f32,
    effective_noise_scale: f32,
    seed: u64,
    rng: RngFixture,
    custom_noise_scaler: CustomNoiseScalerFixture,
    steps: Vec<StepFixture>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CustomNoiseScalerFixture {
    kind: String,
    noise_scale: f32,
    max_stage: usize,
    scaler_inputs: Vec<f32>,
    latents: Vec<Vec<f32>>,
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
    stage: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    er_lambda_source: Option<f32>,
    er_lambda_target: Option<f32>,
    alpha_source: Option<f32>,
    alpha_target: Option<f32>,
    noise_scaler_source: Option<f32>,
    noise_scaler_target: Option<f32>,
    ratio: Option<f32>,
    dt: Option<f32>,
    lambda_step_size: Option<f32>,
    integration_s: Option<f32>,
    integration_s_u: Option<f32>,
    denoised_d: Option<Vec<f32>>,
    denoised_u: Option<Vec<f32>>,
    stage_one: Option<Vec<f32>>,
    stage_two: Option<Vec<f32>>,
    stage_three: Option<Vec<f32>>,
    noise: Option<Vec<f32>>,
    noise_coefficient: Option<f32>,
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

fn source_line_containing(source: &str, needle: &str) -> Result<usize, Box<dyn Error>> {
    source
        .lines()
        .position(|line| line.contains(needle))
        .and_then(|index| index.checked_add(1))
        .ok_or_else(|| format!("source line containing {needle:?} is missing").into())
}

fn sampler_names(source: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let assignment = source
        .find("KSAMPLER_NAMES = [")
        .ok_or("KSAMPLER_NAMES assignment is missing")?;
    let source = source
        .get(assignment..)
        .ok_or("KSAMPLER_NAMES start is invalid")?;
    let end = source.find(']').ok_or("KSAMPLER_NAMES is unterminated")?;
    let source = source.get(..end).ok_or("KSAMPLER_NAMES range is invalid")?;
    let mut names = Vec::new();
    let mut characters = source.chars();
    while let Some(character) = characters.next() {
        if character != '"' {
            continue;
        }
        let name = characters
            .by_ref()
            .take_while(|character| *character != '"')
            .collect();
        names.push(name);
    }
    Ok(names)
}

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("er-sde-row-v1")?,
        PredictionInterpretation::Epsilon,
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

fn assert_optional_close(actual: Option<f32>, expected: Option<f32>, tolerance: f32) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert!((actual - expected).abs() <= tolerance),
        (None, None) => {}
        values => panic!("optional values differ: {values:?}"),
    }
}

fn noise_scaler(value: f32) -> f32 {
    value * ((value.powf(0.3)).exp() + 10.0)
}

#[test]
fn val_sampler_001_er_sde_definition_provenance_ordinal_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, ER_SDE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, ER_SDE_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, ER_SDE_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/er_sde_comfy_model_0178"
    );
    assert_eq!(fixture.rng.contract, ER_SDE_NOISE_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.seed_transform, "add-one-on-cpu");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(ER_SDE_SAMPLER_ID)?)?,
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
        "def sample_er_sde(",
        "default_er_sde_noise_scaler",
        "num_integration_points = 200.0",
        "stage_used = min(max_stage, i + 1)",
        "# Stage 1 Euler",
        "# Stage 2",
        "# Stage 3",
        "noise_sampler(sigmas[i], sigmas[i + 1])",
    ] {
        assert!(equations.contains(fragment), "missing source {fragment}");
    }
    let profile_source = source_range(&sampling, fixture.source.profile_lines);
    for fragment in [
        "def sigma_to_half_log_snr",
        "isinstance(model_sampling, sim.model_sampling.CONST)",
        "def offset_first_sigma_for_snr",
        "model_sampling.percent_to_sigma(percent_offset)",
    ] {
        assert!(
            profile_source.contains(fragment),
            "missing profile {fragment}"
        );
    }
    let noise_source = source_range(&sampling, fixture.source.noise_lines);
    for fragment in ["seed += 1", "torch.Generator", "torch.randn"] {
        assert!(noise_source.contains(fragment), "missing noise {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert_eq!(
        source_line_containing(&samplers, "\"er_sde\"")?,
        fixture.source.registry_line
    );
    let names = sampler_names(&samplers)?;
    let derived_ordinal = names
        .iter()
        .position(|name| name == ER_SDE_SAMPLER_ID)
        .ok_or("er_sde is missing from KSAMPLER_NAMES")?;
    assert_eq!(u16::try_from(derived_ordinal)?, ER_SDE_SOURCE_ORDINAL);
    assert_eq!(
        names
            .iter()
            .filter(|name| *name == ER_SDE_SAMPLER_ID)
            .count(),
        1
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(
                |line| line.contains("sampler,er_sde,") && line.ends_with(",COMFY-MODEL-0178")
            )
    );

    for required in [
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction(",
        "SamplingSession::new",
        ".observe_step(",
        "profile.adjust_first_sigma_for_snr(",
        "profile.half_log_snr(",
        "profile.scale_sampler_noise(",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing owner {required}"
        );
    }
    for forbidden in [
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream",
        "::open",
        "struct ErSdeTrace",
        "struct ErSdeProgress",
        "struct ErSdeObservation",
        "struct ErSdeNoiseRequest",
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
fn val_sampler_001_er_sde_matches_all_three_stages_callbacks_and_noise()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    assert_eq!(profile.prediction(), PredictionInterpretation::Epsilon);
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
    let (trace, noise_before, noise_after) = sample_er_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        ErSdeOptions {
            noise_scale: fixture.sampler_noise_scale,
            max_stage: fixture.max_stage,
        },
        &context,
        |latent, sigma, step| {
            events.borrow_mut().push(format!("Denoiser:{step}"));
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            if (sigma - expected.sigma).abs() > fixture.tolerance {
                return Err("sigma mismatch".to_owned());
            }
            let actual =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&actual, &expected.latent_before, fixture.tolerance);
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
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
            "Denoiser:0",
            "Callback:0",
            "Denoiser:1",
            "Callback:1",
            "Denoiser:2",
            "Callback:2",
            "Denoiser:3",
            "Callback:3",
        ]
    );
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_eq!(expected.stage, fixture.max_stage.min(step + 1));
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
        assert_close(denoised, &expected.denoised, fixture.tolerance);
    }

    let mut oracle = CompatibilityRngTransaction::open(
        ER_SDE_NOISE_CONTRACT_ID,
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
    for expected in fixture.steps.iter().filter_map(|step| step.noise.as_ref()) {
        let actual = oracle
            .draw_normal(expected.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_close(&actual, expected, 0.0);
    }
    assert_eq!(noise_after, oracle.commit());
    Ok(())
}

#[test]
fn val_sampling_foundation_001_er_sde_fixture_independently_reconstructs_every_equation()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let mut old_denoised: Option<&[f32]> = None;
    let mut old_denoised_d: Option<Vec<f32>> = None;
    for expected in &fixture.steps {
        if expected.next_sigma == 0.0 {
            assert_close(
                &expected.latent_after,
                &expected.denoised,
                fixture.tolerance,
            );
            continue;
        }
        let lambda_source = expected.sigma;
        let lambda_target = expected.next_sigma;
        let alpha_source = expected.sigma / lambda_source;
        let alpha_target = expected.next_sigma / lambda_target;
        let scaler_source = noise_scaler(lambda_source);
        let scaler_target = noise_scaler(lambda_target);
        let ratio = scaler_target / scaler_source;
        let dt = lambda_target - lambda_source;
        let lambda_step_size = -dt / 200.0;
        for (actual, fixture_value) in [
            (Some(lambda_source), expected.er_lambda_source),
            (Some(lambda_target), expected.er_lambda_target),
            (Some(alpha_source), expected.alpha_source),
            (Some(alpha_target), expected.alpha_target),
            (Some(scaler_source), expected.noise_scaler_source),
            (Some(scaler_target), expected.noise_scaler_target),
            (Some(ratio), expected.ratio),
            (Some(dt), expected.dt),
            (Some(lambda_step_size), expected.lambda_step_size),
        ] {
            assert_optional_close(actual, fixture_value, fixture.tolerance);
        }
        let mut result = expected
            .latent_before
            .iter()
            .zip(&expected.denoised)
            .map(|(latent, denoised)| {
                alpha_target / alpha_source * ratio * latent
                    + alpha_target * (1.0 - ratio) * denoised
            })
            .collect::<Vec<_>>();
        assert_close(
            &result,
            expected.stage_one.as_deref().ok_or("missing stage one")?,
            fixture.tolerance,
        );
        let mut denoised_d = None;
        if expected.stage >= 2 {
            let mut inverse_sum = 0.0_f32;
            let mut weighted_sum = 0.0_f32;
            for point in 0..200 {
                let position = lambda_target + point as f32 * lambda_step_size;
                let scaled = noise_scaler(position);
                inverse_sum += 1.0 / scaled;
                weighted_sum += (position - lambda_source) / scaled;
            }
            let integration_s = inverse_sum * lambda_step_size;
            assert_optional_close(
                Some(integration_s),
                expected.integration_s,
                fixture.tolerance,
            );
            let previous = old_denoised.ok_or("missing old denoised")?;
            let previous_lambda = fixture.steps[expected.step - 1]
                .er_lambda_source
                .ok_or("missing previous lambda")?;
            let derivative = expected
                .denoised
                .iter()
                .zip(previous)
                .map(|(current, previous)| (current - previous) / (lambda_source - previous_lambda))
                .collect::<Vec<_>>();
            assert_close(
                &derivative,
                expected.denoised_d.as_deref().ok_or("missing derivative")?,
                fixture.tolerance,
            );
            let coefficient = alpha_target * (dt + integration_s * scaler_target);
            for (value, derivative) in result.iter_mut().zip(&derivative) {
                *value += coefficient * derivative;
            }
            assert_close(
                &result,
                expected.stage_two.as_deref().ok_or("missing stage two")?,
                fixture.tolerance,
            );
            if expected.stage >= 3 {
                let integration_s_u = weighted_sum * lambda_step_size;
                assert_optional_close(
                    Some(integration_s_u),
                    expected.integration_s_u,
                    fixture.tolerance,
                );
                let previous_derivative =
                    old_denoised_d.as_deref().ok_or("missing old derivative")?;
                let previous_two_lambda = fixture.steps[expected.step - 2]
                    .er_lambda_source
                    .ok_or("missing two-step lambda")?;
                let derivative_u = derivative
                    .iter()
                    .zip(previous_derivative)
                    .map(|(current, previous)| {
                        (current - previous) / ((lambda_source - previous_two_lambda) / 2.0)
                    })
                    .collect::<Vec<_>>();
                assert_close(
                    &derivative_u,
                    expected
                        .denoised_u
                        .as_deref()
                        .ok_or("missing derivative u")?,
                    fixture.tolerance,
                );
                let coefficient =
                    alpha_target * (dt.powi(2) / 2.0 + integration_s_u * scaler_target);
                for (value, derivative) in result.iter_mut().zip(&derivative_u) {
                    *value += coefficient * derivative;
                }
                assert_close(
                    &result,
                    expected
                        .stage_three
                        .as_deref()
                        .ok_or("missing stage three")?,
                    fixture.tolerance,
                );
            }
            denoised_d = Some(derivative);
        }
        let radicand = lambda_target.powi(2) - lambda_source.powi(2) * ratio.powi(2);
        let coefficient = alpha_target * fixture.effective_noise_scale * radicand.sqrt();
        assert_optional_close(
            Some(coefficient),
            expected.noise_coefficient,
            fixture.tolerance,
        );
        for (value, noise) in result
            .iter_mut()
            .zip(expected.noise.as_deref().ok_or("missing noise")?)
        {
            *value += coefficient * noise;
        }
        assert_close(&result, &expected.latent_after, fixture.tolerance);
        old_denoised = Some(&expected.denoised);
        if denoised_d.is_some() {
            old_denoised_d = denoised_d;
        }
    }
    Ok(())
}

#[test]
fn val_rng_001_er_sde_constant_flow_boundaries_failures_and_zero_noise_are_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);
    let error = sample_er_sde(
        &backend,
        plan(&fixture, "dpmpp_2m", &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ErSdeOptions::default(),
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("another registered sampler must not substitute er_sde");
    assert!(matches!(error, ErSdeError::WrongSampler(_)));

    let error = sample_er_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ErSdeOptions {
            noise_scale: f32::NAN,
            max_stage: 3,
        },
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-finite noise scale must fail before RNG publication");
    assert!(matches!(error, ErSdeError::SamplingProfile(_)));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_er_sde(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        ErSdeOptions::default(),
        &cancelled_context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, ErSdeError::Tensor(_)));

    let flow_profile = DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("er-sde-constant-flow-v1")?,
        PredictionInterpretation::Flow,
        Arc::from([0.01_f32, 0.2, 0.5, 0.9]),
        SamplingSnrMode::ConstantFlow { shift: 2.0 },
        1.0,
    )?;
    let flow_sigmas = [1.0, 0.5, 0.0];
    let mut adjusted = flow_sigmas;
    flow_profile.adjust_first_sigma_for_snr(&mut adjusted)?;
    assert!(adjusted[0] < 1.0);
    let flow_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let flow_plan = SamplingPlan::new(
        ER_SDE_SAMPLER_ID,
        "normal",
        flow_profile.identity().clone(),
        fixture.seed,
        2,
        1.0,
        1.0,
    )?;
    let (trace, before, after) = sample_er_sde(
        &backend,
        flow_plan,
        &flow_profile,
        flow_initial,
        &flow_sigmas,
        noise_request(&fixture),
        ErSdeOptions {
            noise_scale: fixture.custom_noise_scaler.noise_scale,
            max_stage: fixture.custom_noise_scaler.max_stage,
        },
        &context,
        |_, _, step| {
            let denoised = if step == 0 { [0.2, -0.1] } else { [0.0, 0.0] };
            tensor_from_f32(&backend, &fixture.shape, &denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(trace.sigmas, adjusted);
    assert_eq!(before, after, "zero noise must consume no normal values");
    assert_close(
        &values(
            &backend,
            trace.latents.last().ok_or("missing flow terminal")?,
            &context,
        )?,
        &[0.0, 0.0],
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn val_sampler_001_er_sde_custom_noise_scaler_and_device_placement_are_source_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let scaler_inputs = RefCell::new(Vec::new());
    let (trace, before, after) = sample_er_sde_with_noise_scaler(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        ErSdeOptions {
            noise_scale: 0.0,
            max_stage: 1,
        },
        &context,
        |_, _, step| {
            tensor_from_f32(&backend, &fixture.shape, &fixture.steps[step].denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
        |value| {
            scaler_inputs.borrow_mut().push(value);
            Ok(value)
        },
    )?;
    assert_eq!(before, after, "zero source noise scale must consume no values");

    assert_eq!(fixture.custom_noise_scaler.kind, "identity");
    for (step, expected) in fixture.custom_noise_scaler.latents.iter().skip(1).enumerate() {
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            expected,
            fixture.tolerance,
        );
    }
    assert_eq!(
        scaler_inputs.into_inner(),
        fixture.custom_noise_scaler.scaler_inputs
    );

    assert_eq!(
        er_sde_rng_profile(DeviceId::CPU),
        (
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
        )
    );
    let cuda = DeviceId::from_source_device("cuda:0")?;
    assert_eq!(
        er_sde_rng_profile(cuda),
        (
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(cuda),
        )
    );
    assert!(matches!(
        validate_er_sde_generation_device(cuda),
        Err(ErSdeError::DeviceUnavailable { device, .. }) if device == cuda
    ));

    let scaler_failure = sample_er_sde_with_noise_scaler(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        ErSdeOptions {
            noise_scale: 0.0,
            max_stage: 1,
        },
        &context,
        |_, _, step| {
            tensor_from_f32(&backend, &fixture.shape, &fixture.steps[step].denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
        |_| Err("custom scaler rejected input".to_owned()),
    )
    .expect_err("custom scaler errors must remain typed and abort the transaction");
    assert!(matches!(
        scaler_failure,
        ErSdeError::NoiseScaler { step: 0, ref reason, .. }
            if reason == "custom scaler rejected input"
    ));
    Ok(())
}
