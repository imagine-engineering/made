use sim_sampler::{
    DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingProgress,
    SamplingSnrMode,
    generated_gradient_estimation_sim_model_0185::{
        DEFINITION, GRADIENT_ESTIMATION_FEATURE_ID, GRADIENT_ESTIMATION_SAMPLER_ID,
        GRADIENT_ESTIMATION_SOURCE_ORDINAL, GradientEstimationError, GradientEstimationOptions,
        sample_gradient_estimation,
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
    "/../sim_test_support/fixtures/samplers/gradient_estimation_comfy_model_0185/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/gradient_estimation_sim_model_0185.rs"
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
    gamma: f32,
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
    family_lines: [usize; 2],
    cfg_pp_wrapper_lines: [usize; 2],
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    derivative: Vec<f32>,
    euler: Option<Vec<f32>>,
    correction: Option<Vec<f32>>,
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("gradient-estimation-row-v1")?,
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
        185,
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
fn val_sampler_001_gradient_estimation_definition_source_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, GRADIENT_ESTIMATION_SAMPLER_ID);
    assert_eq!(fixture.feature_id, GRADIENT_ESTIMATION_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, GRADIENT_ESTIMATION_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 34);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/gradient_estimation_comfy_model_0185"
    );
    assert_eq!(
        GradientEstimationOptions::source_defaults().gamma,
        fixture.gamma
    );
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(GRADIENT_ESTIMATION_SAMPLER_ID)?)?,
        &DEFINITION
    );

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
    let family = source_range(&sampling, fixture.source.family_lines);
    for fragment in [
        "def sample_gradient_estimation(",
        "ge_gamma=2.",
        "d = to_d(x, sigmas[i], denoised)",
        "callback({'x': x",
        "x = x + d * dt",
        "d_bar = (ge_gamma - 1) * (d - old_d)",
        "x = x + d_bar * dt",
    ] {
        assert!(
            family.contains(fragment),
            "missing source equation {fragment}"
        );
    }
    let cfg_pp_wrapper = source_range(&sampling, fixture.source.cfg_pp_wrapper_lines);
    assert!(cfg_pp_wrapper.contains("return sample_gradient_estimation("));
    assert!(cfg_pp_wrapper.contains("cfg_pp=True"));
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"gradient_estimation\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,gradient_estimation,")
                && line.ends_with(",COMFY-MODEL-0185"))
    );

    for required in [
        "SamplerRegistry::foundational()",
        "SchedulerRegistry::foundational()",
        "SamplingSession::new",
        "session.observe_step(",
        "observed.commit(",
        "backend.workspace_vec::<f32>",
        "validate_cfg_pp_denoiser_output",
        "sample_gradient_estimation_family(",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing owner mapping {required}"
        );
    }
    for forbidden in [
        "struct SamplingSession",
        "struct SamplingProgress",
        "struct SamplingTrace",
        "struct CancellationToken",
        "struct ExecutionContext",
        "struct CpuWorkspaceAuthority",
        "struct CfgPpDenoiserOutput",
        "CompatibilityRngTransaction",
        "CompatibilityNoiseRequest",
        "RngStream",
        "std::fs",
        "serde",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_every_gradient_intermediate_callback_and_boundary_match_fixture()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let callbacks = RefCell::new(Vec::<(SamplingProgress, Vec<f32>, Vec<f32>)>::new());
    let denoiser_inputs = RefCell::new(Vec::<Vec<f32>>::new());
    let trace = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        GradientEstimationOptions {
            gamma: fixture.gamma,
        },
        &context,
        |latent, sigma, step| {
            denoiser_inputs
                .borrow_mut()
                .push(values(&backend, latent, &context).map_err(|error| error.to_string())?);
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            assert!((sigma - expected.sigma).abs() <= fixture.tolerance);
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

    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    let denoiser_inputs = denoiser_inputs.borrow();
    let callbacks = callbacks.borrow();
    for expected in &fixture.steps {
        assert_eq!(expected.step, callbacks[expected.step].0.step as usize);
        assert_eq!(callbacks[expected.step].0.total_steps, 3);
        assert_eq!(callbacks[expected.step].0.sigma, expected.sigma);
        assert_eq!(callbacks[expected.step].0.sigma_hat, expected.sigma);
        assert_eq!(callbacks[expected.step].0.next_sigma, expected.next_sigma);
        assert_close(
            &denoiser_inputs[expected.step],
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &callbacks[expected.step].1,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &callbacks[expected.step].2,
            &expected.denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                &trace.denoiser_evaluations[expected.step],
                &context,
            )?,
            &expected.denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[expected.step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );

        let derivative = expected
            .latent_before
            .iter()
            .zip(&expected.denoised)
            .map(|(latent, denoised)| (latent - denoised) / expected.sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &expected.derivative, fixture.tolerance);
        if let Some(euler) = expected.euler.as_ref() {
            let calculated = expected
                .latent_before
                .iter()
                .zip(&derivative)
                .map(|(latent, derivative)| {
                    latent + derivative * (expected.next_sigma - expected.sigma)
                })
                .collect::<Vec<_>>();
            assert_close(&calculated, euler, fixture.tolerance);
            if let Some(correction) = expected.correction.as_ref() {
                let corrected = euler
                    .iter()
                    .zip(correction)
                    .map(|(value, correction)| value + correction)
                    .collect::<Vec<_>>();
                assert_close(&corrected, &expected.latent_after, fixture.tolerance);
            }
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
    Ok(())
}

#[test]
fn val_sampler_001_gamma_is_finite_signed_and_terminal_denoising_skips_correction()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    for (gamma, expected_second) in [
        (1.0, [0.3625, -0.325]),
        (0.0, [0.4125, -0.4]),
        (-1.0, [0.4625, -0.475]),
    ] {
        let trace = sample_gradient_estimation(
            &backend,
            plan(&fixture.identity, &profile, fixture.steps.len())?,
            &profile,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            GradientEstimationOptions { gamma },
            &context,
            |_, _, step| {
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<_, String>(()),
        )?;
        assert_close(
            &values(&backend, &trace.latents[2], &context)?,
            &expected_second,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace.latents.last().ok_or("missing terminal")?,
                &context,
            )?,
            &fixture.terminal,
            fixture.tolerance,
        );
    }

    let single_sigmas = [1.0, 0.0];
    let single_output = [0.25, -0.125];
    let trace = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, 1)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &single_sigmas,
        GradientEstimationOptions { gamma: -10.0 },
        &context,
        |_, _, _| {
            tensor_from_f32(&backend, &fixture.shape, &single_output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, _, _| {
            assert_eq!(progress.step, 0);
            assert_eq!(progress.next_sigma, 0.0);
            Ok::<_, String>(())
        },
    )?;
    assert_eq!(trace.latents.len(), 2);
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &single_output,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn val_sampler_001_failures_are_typed_and_callback_commit_is_cancellation_safe()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);

    let error = sample_gradient_estimation(
        &backend,
        plan("dpmpp_2m", &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("a registered sampler must not alias this row");
    assert!(matches!(
        error,
        GradientEstimationError::WrongSampler { .. }
    ));

    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions { gamma: f32::NAN },
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-finite gamma must fail before evaluation");
    assert!(matches!(error, GradientEstimationError::InvalidGamma(value) if value.is_nan()));

    let invalid_sigmas = [2.0, 2.0, 0.5, 0.0];
    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &invalid_sigmas,
        GradientEstimationOptions::default(),
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-descending schedules must be rejected by the session owner");
    assert!(matches!(
        error,
        GradientEstimationError::Sampling(SamplingError::InvalidSigma { step: 0, .. })
    ));

    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &context,
        |_, _, step| Err(format!("fault-{step}")),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("denoiser faults must retain their step");
    assert!(matches!(
        error,
        GradientEstimationError::Denoiser { step: 0, .. }
    ));

    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &context,
        |_, _, _| {
            tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("descriptor changes must fail before callbacks");
    assert!(matches!(
        error,
        GradientEstimationError::DenoiserContract { step: 0, .. }
    ));

    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &context,
        |_, _, _| {
            tensor_from_f32(&backend, &fixture.shape, &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-finite derivatives must fail before callbacks");
    assert!(matches!(
        error,
        GradientEstimationError::NonFinite {
            step: 0,
            stage: "guided denoiser",
            element: 0
        }
    ));

    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &context,
        |_, _, step| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_, _, _| Err::<(), _>("callback-fault"),
    )
    .expect_err("callback faults must be projected by the session owner");
    assert!(matches!(
        error,
        GradientEstimationError::Sampling(SamplingError::Callback(reason))
            if reason == "callback-fault"
    ));

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        tensor_from_f32(
            &backend,
            &fixture.shape,
            &fixture.initial,
            &callback_context,
        )?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &callback_context,
        |_, _, step| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &callback_context,
            )
            .map_err(|error| error.to_string())
        },
        |_, _, _| {
            callback_cancellation.cancel();
            Ok::<_, String>(())
        },
    )
    .expect_err("cancellation after callback must prevent the commit");
    assert!(matches!(
        error,
        GradientEstimationError::Sampling(SamplingError::Cancelled)
    ));

    let pre_cancelled = CancellationToken::default();
    pre_cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &pre_cancelled)?;
    let error = sample_gradient_estimation(
        &backend,
        plan(&fixture.identity, &profile, fixture.steps.len())?,
        &profile,
        initial()?,
        &fixture.sigmas,
        GradientEstimationOptions::default(),
        &cancelled_context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before denoising");
    assert!(matches!(error, GradientEstimationError::Tensor(_)));
    Ok(())
}

#[test]
fn val_rng_001_gradient_estimation_is_deterministic_and_draws_no_rng_phase()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let run = || {
        sample_gradient_estimation(
            &backend,
            plan(&fixture.identity, &profile, fixture.steps.len())
                .map_err(|error| error.to_string())?,
            &profile,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)
                .map_err(|error| error.to_string())?,
            &fixture.sigmas,
            GradientEstimationOptions::default(),
            &context,
            |_, _, step| {
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<_, String>(()),
        )
        .map_err(|error| error.to_string())
    };
    let first = run()?;
    let second = run()?;
    assert_eq!(first.sigmas, second.sigmas);
    assert_eq!(first.latents.len(), second.latents.len());
    for (first, second) in first.latents.iter().zip(&second.latents) {
        assert_eq!(
            values(&backend, first, &context)?,
            values(&backend, second, &context)?
        );
    }
    assert!(!DEFINITION.stochastic);
    for forbidden in [
        "CompatibilityNoiseRequest",
        "CompatibilityRngTransaction",
        "draw_normal",
        "RngCheckpoint",
        "RngStream",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "unexpected RNG owner {forbidden}"
        );
    }
    Ok(())
}
