use sim_sampler::{
    DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity, SamplerRegistry,
    SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingProgress, SamplingSnrMode,
    generated_gradient_estimation_sim_model_0185::{
        GradientEstimationError, GradientEstimationOptions,
    },
    generated_gradient_estimation_cfg_pp_sim_model_0186::{
        DEFINITION, GRADIENT_ESTIMATION_CFG_PP_FEATURE_ID,
        GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, GRADIENT_ESTIMATION_CFG_PP_SOURCE_ORDINAL,
        GradientEstimationCfgPpDenoiserOutput, sample_gradient_estimation_cfg_pp,
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
    "/../sim_test_support/fixtures/samplers/gradient_estimation_cfg_pp_comfy_model_0186/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/gradient_estimation_cfg_pp_sim_model_0186.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    family_identity: String,
    gamma: f32,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    source: SourceFixture,
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
    wrapper_lines: [usize; 2],
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    guided: Vec<f32>,
    unconditional: Vec<f32>,
    derivative: Vec<f32>,
    cfg_pp_base: Option<Vec<f32>>,
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("gradient-estimation-cfg-pp-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
        SamplingSnrMode::Standard,
        1.0,
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
        19,
        u32::try_from(fixture.steps.len())?,
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
fn val_sampler_001_definition_provenance_and_adapter_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID);
    assert_eq!(fixture.feature_id, GRADIENT_ESTIMATION_CFG_PP_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, GRADIENT_ESTIMATION_CFG_PP_SOURCE_ORDINAL);
    assert_eq!(fixture.family_identity, "gradient_estimation");
    assert_eq!(fixture.gamma, GradientEstimationOptions::source_defaults().gamma);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 35);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID)?)?,
        &DEFINITION
    );

    let root = workspace_root()?;
    for (path, expected) in [
        (&fixture.source.sampling_path, &fixture.source.sampling_sha256),
        (&fixture.source.samplers_path, &fixture.source.samplers_sha256),
        (&fixture.source.catalog_path, &fixture.source.catalog_sha256),
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let wrapper = sampling
        .lines()
        .skip(fixture.source.wrapper_lines[0].saturating_sub(1))
        .take(fixture.source.wrapper_lines[1] - fixture.source.wrapper_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_gradient_estimation_cfg_pp(",
        "return sample_gradient_estimation(",
        "ge_gamma=ge_gamma",
        "cfg_pp=True",
    ] {
        assert!(wrapper.contains(fragment), "missing source fragment {fragment}");
    }
    let family = sampling
        .lines()
        .skip(fixture.source.family_lines[0].saturating_sub(1))
        .take(fixture.source.family_lines[1] - fixture.source.family_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "d = to_d(x, sigmas[i], uncond_denoised)",
        "x = denoised + d * sigmas[i + 1]",
        "d_bar = (ge_gamma - 1) * (d - old_d)",
    ] {
        assert!(family.contains(fragment), "missing family equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"gradient_estimation_cfg_pp\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,gradient_estimation_cfg_pp,")
                && line.ends_with(",COMFY-MODEL-0186"))
    );
    assert!(IMPLEMENTATION.contains("sample_gradient_estimation_family("));
    assert!(IMPLEMENTATION.contains("        true,"));
    for forbidden in [
        "SamplingSession::new",
        "observe_step(",
        "tensor_to_f32(",
        "workspace_vec(",
        "validate_cfg_pp_denoiser_output(",
        "gradient_estimation_update(",
        "RngStream",
        "SchedulerRegistry",
    ] {
        assert!(!IMPLEMENTATION.contains(forbidden), "duplicate owner {forbidden}");
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_every_cfg_pp_intermediate_and_callback()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let calls = RefCell::new(Vec::<Vec<f32>>::new());
    let callbacks = RefCell::new(Vec::<(SamplingProgress, Vec<f32>, Vec<f32>)>::new());
    let trace = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        GradientEstimationOptions { gamma: fixture.gamma },
        &context,
        |latent, sigma, step| {
            let expected = &fixture.steps[step];
            calls.borrow_mut().push(
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
            );
            assert!((sigma - expected.sigma).abs() <= fixture.tolerance);
            Ok(GradientEstimationCfgPpDenoiserOutput {
                denoised: tensor_from_f32(&backend, &fixture.shape, &expected.guided, &context)
                    .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &expected.unconditional,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
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

    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_close(&values(&backend, &trace.latents[0], &context)?, &fixture.initial, fixture.tolerance);
    let calls = calls.into_inner();
    let callbacks = callbacks.into_inner();
    let mut previous_derivative: Option<&[f32]> = None;
    for (index, step) in fixture.steps.iter().enumerate() {
        assert_eq!(step.step, index);
        assert_close(&calls[index], &step.latent_before, fixture.tolerance);
        assert_close(&callbacks[index].1, &step.latent_before, fixture.tolerance);
        assert_close(&callbacks[index].2, &step.guided, fixture.tolerance);
        assert_eq!(usize::try_from(callbacks[index].0.step)?, index);
        assert!((callbacks[index].0.sigma - step.sigma).abs() <= fixture.tolerance);
        assert!((callbacks[index].0.next_sigma - step.next_sigma).abs() <= fixture.tolerance);

        let derivative = step
            .latent_before
            .iter()
            .zip(&step.unconditional)
            .map(|(latent, unconditional)| (latent - unconditional) / step.sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &step.derivative, fixture.tolerance);
        if step.next_sigma == 0.0 {
            assert!(step.cfg_pp_base.is_none());
            assert!(step.correction.is_none());
            assert_close(&step.latent_after, &step.guided, fixture.tolerance);
        } else {
            let base = step
                .guided
                .iter()
                .zip(&derivative)
                .map(|(guided, derivative)| guided + derivative * step.next_sigma)
                .collect::<Vec<_>>();
            assert_close(
                &base,
                step.cfg_pp_base.as_deref().ok_or("missing CFG++ base")?,
                fixture.tolerance,
            );
            let expected_after = if let Some(previous) = previous_derivative {
                let delta_sigma = step.next_sigma - step.sigma;
                let correction = derivative
                    .iter()
                    .zip(previous)
                    .map(|(current, previous)| {
                        (fixture.gamma - 1.0) * (current - previous) * delta_sigma
                    })
                    .collect::<Vec<_>>();
                assert_close(
                    &correction,
                    step.correction.as_deref().ok_or("missing correction")?,
                    fixture.tolerance,
                );
                base.iter()
                    .zip(correction)
                    .map(|(base, correction)| base + correction)
                    .collect::<Vec<_>>()
            } else {
                assert!(step.correction.is_none());
                base
            };
            assert_close(&expected_after, &step.latent_after, fixture.tolerance);
            if index == 0 {
                let guided_derivative = step
                    .latent_before
                    .iter()
                    .zip(&step.guided)
                    .map(|(latent, guided)| (latent - guided) / step.sigma)
                    .collect::<Vec<_>>();
                let guided_derivative_base = step
                    .guided
                    .iter()
                    .zip(guided_derivative)
                    .map(|(guided, derivative)| guided + derivative * step.next_sigma)
                    .collect::<Vec<_>>();
                assert!(
                    guided_derivative_base
                        .iter()
                        .zip(&step.latent_after)
                        .any(|(guided_base, actual)| {
                            (guided_base - actual).abs() > fixture.tolerance
                        }),
                    "the fixture must distinguish guided and unconditional derivatives"
                );
            }
        }
        assert_close(
            &values(&backend, &trace.latents[index + 1], &context)?,
            &step.latent_after,
            fixture.tolerance,
        );
        previous_derivative = Some(&step.derivative);
    }
    assert_close(
        &values(&backend, trace.latents.last().ok_or("missing terminal")?, &context)?,
        &fixture.terminal,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn val_rng_001_rejects_wrong_identity_invalid_gamma_and_pre_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let denoiser = |latent: &Tensor, _: f32, _: usize| {
        Ok(GradientEstimationCfgPpDenoiserOutput {
            denoised: latent.clone(),
            unconditional_denoised: latent.clone(),
        })
    };
    let wrong = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, "gradient_estimation", &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        GradientEstimationOptions::source_defaults(),
        &context,
        denoiser,
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("wrong identity must fail");
    assert!(matches!(wrong, GradientEstimationError::WrongSampler { .. }));

    let invalid = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        GradientEstimationOptions { gamma: f32::NAN },
        &context,
        denoiser,
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-finite gamma must fail");
    assert!(matches!(invalid, GradientEstimationError::InvalidGamma(value) if value.is_nan()));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        GradientEstimationOptions::source_defaults(),
        &cancelled_context,
        denoiser,
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancellation must fail");
    assert!(matches!(error, GradientEstimationError::Tensor(_)));
    Ok(())
}

#[test]
fn unconditional_descriptor_mismatch_precedes_callback() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let callback_count = RefCell::new(0_usize);
    let error = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        GradientEstimationOptions::source_defaults(),
        &context,
        |latent, _, _| {
            Ok(GradientEstimationCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: tensor_from_f32(&backend, &[1], &[0.0], &context)
                    .map_err(|error| error.to_string())?,
            })
        },
        |_, _, _| {
            *callback_count.borrow_mut() += 1;
            Ok::<_, String>(())
        },
    )
    .expect_err("unconditional descriptor mismatch must fail");
    assert!(matches!(
        error,
        GradientEstimationError::DenoiserContract {
            step: 0,
            output: "unconditional denoiser output"
        }
    ));
    assert_eq!(*callback_count.borrow(), 0);
    Ok(())
}

#[test]
fn callback_error_and_callback_cancellation_are_failure_atomic() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let callback_error_count = RefCell::new(0_usize);
    let callback_error = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        GradientEstimationOptions::source_defaults(),
        &context,
        |latent, _, _| {
            Ok(GradientEstimationCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: latent.clone(),
            })
        },
        |_, _, _| {
            *callback_error_count.borrow_mut() += 1;
            Err::<(), _>("blocked")
        },
    )
    .expect_err("callback failure must abort the observed step");
    assert!(matches!(
        callback_error,
        GradientEstimationError::Sampling(sim_sampler::SamplingError::Callback(reason))
            if reason == "blocked"
    ));
    assert_eq!(*callback_error_count.borrow(), 1);
    assert_close(
        &values(&backend, &initial, &context)?,
        &fixture.initial,
        fixture.tolerance,
    );

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_cancellation_count = RefCell::new(0_usize);
    let cancellation_error = sample_gradient_estimation_cfg_pp(
        &backend,
        plan(&fixture, GRADIENT_ESTIMATION_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        GradientEstimationOptions::source_defaults(),
        &callback_context,
        |latent, _, _| {
            Ok(GradientEstimationCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: latent.clone(),
            })
        },
        |_, _, _| {
            *callback_cancellation_count.borrow_mut() += 1;
            assert!(callback_cancellation.cancel());
            Ok::<_, String>(())
        },
    )
    .expect_err("callback-triggered cancellation must abort before commit");
    assert!(matches!(
        cancellation_error,
        GradientEstimationError::Sampling(sim_sampler::SamplingError::Cancelled)
    ));
    assert_eq!(*callback_cancellation_count.borrow(), 1);
    assert_close(
        &values(&backend, &initial, &context)?,
        &fixture.initial,
        fixture.tolerance,
    );
    Ok(())
}
