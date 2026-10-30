use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_seeds_3_sim_model_0200::{
        DEFINITION, SEEDS_3_FEATURE_ID, SEEDS_3_NOISE_CONTRACT_ID, SEEDS_3_SAMPLER_ID,
        SEEDS_3_SOURCE_ORDINAL, Seeds3DenoiserStage, Seeds3Error, Seeds3Options, sample_seeds_3,
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
    "/../sim_test_support/fixtures/samplers/seeds_3_comfy_model_0200/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/seeds_3_sim_model_0200.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    rng_contract_id: String,
    seed: u64,
    eta: f32,
    sampler_noise_scale: f32,
    profile_noise_scale: f32,
    effective_noise_scale: f32,
    r_1: f32,
    r_2: f32,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
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
    equation_lines: [usize; 2],
    phi_lines: [usize; 2],
    profile_lines: [usize; 2],
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
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    current: Vec<f32>,
    primary: Vec<f32>,
    lambda_source: Option<f32>,
    lambda_target: Option<f32>,
    h: Option<f32>,
    h_eta: Option<f32>,
    first_sigma: Option<f32>,
    second_sigma: Option<f32>,
    first_noise: Option<Vec<f32>>,
    second_noise: Option<Vec<f32>>,
    third_noise: Option<Vec<f32>>,
    stage_two_input: Option<Vec<f32>>,
    stage_two: Option<Vec<f32>>,
    stage_three_input: Option<Vec<f32>>,
    stage_three: Option<Vec<f32>>,
    next: Vec<f32>,
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

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("seeds-3-row-v1")?,
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

fn request(fixture: &Fixture) -> CompatibilityNoiseRequest {
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

#[test]
fn val_sampler_001_definition_provenance_and_owner_boundaries_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, SEEDS_3_SAMPLER_ID);
    assert_eq!(fixture.feature_id, SEEDS_3_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, SEEDS_3_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, SEEDS_3_NOISE_CONTRACT_ID);
    assert_eq!(fixture.effective_noise_scale, 0.6);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.source_ordinal, 38);
    assert!(DEFINITION.stochastic);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(SEEDS_3_SAMPLER_ID)?)?,
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
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "def sample_seeds_3(",
        "lambda_s_1 = torch.lerp(lambda_s, lambda_t, r_1)",
        "a3_2 = r_2 / r_1 * ei_h_phi_2(-r_2 * h_eta)",
        "b3 = ei_h_phi_2(-h_eta) / r_2",
        "noise_sampler(sigma_s_2, sigmas[i + 1])",
    ] {
        assert!(
            equations.contains(fragment),
            "missing source equation {fragment}"
        );
    }
    assert!(source_range(&sampling, fixture.source.phi_lines).contains("torch.expm1(h) - h"));
    assert!(
        source_range(&sampling, fixture.source.profile_lines)
            .contains("offset_first_sigma_for_snr")
    );
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("\"seeds_3\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(|line| {
                line.starts_with("sampler,seeds_3,") && line.ends_with(",COMFY-MODEL-0200")
            })
    );
    for forbidden in [
        "struct SamplingSession",
        "struct CancellationToken",
        "struct CpuWorkspaceAuthority",
        "fn validate_euler_noise_generation_device",
        "std::fs",
        "rusqlite",
        "sqlx",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_every_stage_callback_and_rng_checkpoint()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let options = Seeds3Options::new(
        fixture.eta,
        fixture.sampler_noise_scale,
        fixture.r_1,
        fixture.r_2,
    )?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let events = RefCell::new(Vec::new());
    let (trace, (before, after)) = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture),
        options,
        &context,
        |input, sigma, step, stage| {
            let expected = fixture.steps.get(step).ok_or("unexpected step")?;
            let (expected_input, expected_sigma, output) = match stage {
                Seeds3DenoiserStage::Primary => (
                    expected.current.as_slice(),
                    fixture.sigmas[step],
                    expected.primary.as_slice(),
                ),
                Seeds3DenoiserStage::StageTwo => (
                    expected
                        .stage_two_input
                        .as_deref()
                        .ok_or("missing stage-two input")?,
                    expected.first_sigma.ok_or("missing first sigma")?,
                    expected
                        .stage_two
                        .as_deref()
                        .ok_or("missing stage-two output")?,
                ),
                Seeds3DenoiserStage::StageThree => (
                    expected
                        .stage_three_input
                        .as_deref()
                        .ok_or("missing stage-three input")?,
                    expected.second_sigma.ok_or("missing second sigma")?,
                    expected
                        .stage_three
                        .as_deref()
                        .ok_or("missing stage-three output")?,
                ),
            };
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                expected_input,
                fixture.tolerance,
            );
            assert!((sigma - expected_sigma).abs() <= fixture.tolerance);
            events
                .borrow_mut()
                .push(format!("denoiser-{step}-{stage:?}"));
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, current, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture.steps.get(step).ok_or("unexpected callback")?;
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.steps.len()).unwrap_or(0)
            );
            assert_eq!(progress.sigma, fixture.sigmas[step]);
            assert_eq!(progress.sigma_hat, fixture.sigmas[step]);
            assert_eq!(progress.next_sigma, fixture.sigmas[step + 1]);
            assert_close(
                &values(&backend, current, &context).map_err(|error| error.to_string())?,
                &expected.current,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.primary,
                fixture.tolerance,
            );
            events.borrow_mut().push(format!("callback-{step}"));
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(
        events.into_inner(),
        [
            "denoiser-0-Primary",
            "callback-0",
            "denoiser-0-StageTwo",
            "denoiser-0-StageThree",
            "denoiser-1-Primary",
            "callback-1",
        ]
    );
    for (index, expected) in fixture.steps.iter().enumerate() {
        assert_close(
            &values(&backend, &trace.latents[index], &context)?,
            &expected.current,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[index + 1], &context)?,
            &expected.next,
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
    let mut oracle = request(&fixture).open_transaction(
        SEEDS_3_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    for expected in [
        fixture.steps[0]
            .first_noise
            .as_deref()
            .ok_or("missing first noise")?,
        fixture.steps[0]
            .second_noise
            .as_deref()
            .ok_or("missing second noise")?,
        fixture.steps[0]
            .third_noise
            .as_deref()
            .ok_or("missing third noise")?,
    ] {
        let actual = oracle.draw_normal(fixture.initial.len(), &cancellation)?;
        assert_close(
            &actual.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            expected,
            fixture.tolerance,
        );
    }
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn analytical_fixture_reconstructs_three_stage_coefficients_and_noise() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let expected = &fixture.steps[0];
    let lambda_source = -fixture.sigmas[0].ln();
    let lambda_target = -fixture.sigmas[1].ln();
    let h = lambda_target - lambda_source;
    let h_eta = h * (fixture.eta + 1.0);
    assert!(
        (lambda_source - expected.lambda_source.ok_or("missing source lambda")?).abs()
            <= fixture.tolerance
    );
    assert!(
        (lambda_target - expected.lambda_target.ok_or("missing target lambda")?).abs()
            <= fixture.tolerance
    );
    assert!((h - expected.h.ok_or("missing h")?).abs() <= fixture.tolerance);
    assert!((h_eta - expected.h_eta.ok_or("missing h eta")?).abs() <= fixture.tolerance);
    let lambda_one = lambda_source + h * fixture.r_1;
    let lambda_two = lambda_source + h * fixture.r_2;
    let sigma_one = (-lambda_one).exp();
    let sigma_two = (-lambda_two).exp();
    assert!(
        (sigma_one - expected.first_sigma.ok_or("missing first sigma")?).abs() <= fixture.tolerance
    );
    assert!(
        (sigma_two - expected.second_sigma.ok_or("missing second sigma")?).abs()
            <= fixture.tolerance
    );
    let phi_one = |value: f32| value.exp_m1();
    let phi_two = |value: f32| (value.exp_m1() - value) / value;
    let noise_one = expected
        .first_noise
        .as_deref()
        .ok_or("missing first noise")?;
    let noise_two = expected
        .second_noise
        .as_deref()
        .ok_or("missing second noise")?;
    let noise_three = expected
        .third_noise
        .as_deref()
        .ok_or("missing third noise")?;
    let first_root = (-(-2.0 * fixture.r_1 * h * fixture.eta).exp_m1()).sqrt();
    let mut accumulated = noise_one
        .iter()
        .map(|noise| first_root * noise)
        .collect::<Vec<_>>();
    let stage_two_input = fixture
        .initial
        .iter()
        .zip(&expected.primary)
        .zip(&accumulated)
        .map(|((latent, primary), noise)| {
            sigma_one / fixture.sigmas[0] * (-fixture.r_1 * h * fixture.eta).exp() * latent
                - sigma_one * lambda_one.exp() * phi_one(-fixture.r_1 * h_eta) * primary
                + noise * sigma_one * fixture.effective_noise_scale
        })
        .collect::<Vec<_>>();
    assert_close(
        &stage_two_input,
        expected
            .stage_two_input
            .as_deref()
            .ok_or("missing stage-two input")?,
        fixture.tolerance,
    );
    let second_old = ((fixture.r_1 - fixture.r_2) * h * fixture.eta).exp();
    let second_new = (-((2.0 * (fixture.r_1 - fixture.r_2) * h * fixture.eta).exp_m1())).sqrt();
    for (accumulated, noise) in accumulated.iter_mut().zip(noise_two) {
        *accumulated = *accumulated * second_old + second_new * noise;
    }
    let a32 = fixture.r_2 / fixture.r_1 * phi_two(-fixture.r_2 * h_eta);
    let a31 = phi_one(-fixture.r_2 * h_eta) - a32;
    let stage_three_input = fixture
        .initial
        .iter()
        .zip(&expected.primary)
        .zip(
            expected
                .stage_two
                .as_deref()
                .ok_or("missing stage-two output")?,
        )
        .zip(&accumulated)
        .map(|(((latent, primary), stage_two), noise)| {
            sigma_two / fixture.sigmas[0] * (-fixture.r_2 * h * fixture.eta).exp() * latent
                - sigma_two * lambda_two.exp() * (a31 * primary + a32 * stage_two)
                + noise * sigma_two * fixture.effective_noise_scale
        })
        .collect::<Vec<_>>();
    assert_close(
        &stage_three_input,
        expected
            .stage_three_input
            .as_deref()
            .ok_or("missing stage-three input")?,
        fixture.tolerance,
    );
    let output_old = ((fixture.r_2 - 1.0) * h * fixture.eta).exp();
    let output_new = (-((2.0 * (fixture.r_2 - 1.0) * h * fixture.eta).exp_m1())).sqrt();
    for (accumulated, noise) in accumulated.iter_mut().zip(noise_three) {
        *accumulated = *accumulated * output_old + output_new * noise;
    }
    let b3 = phi_two(-h_eta) / fixture.r_2;
    let b1 = phi_one(-h_eta) - b3;
    let output = fixture
        .initial
        .iter()
        .zip(&expected.primary)
        .zip(
            expected
                .stage_three
                .as_deref()
                .ok_or("missing stage-three output")?,
        )
        .zip(&accumulated)
        .map(|(((latent, primary), stage_three), noise)| {
            fixture.sigmas[1] / fixture.sigmas[0] * (-h * fixture.eta).exp() * latent
                - fixture.sigmas[1] * lambda_target.exp() * (b1 * primary + b3 * stage_three)
                + noise * fixture.sigmas[1] * fixture.effective_noise_scale
        })
        .collect::<Vec<_>>();
    assert_close(&output, &expected.next, fixture.tolerance);
    Ok(())
}

#[test]
fn val_rng_001_options_failures_cancellation_and_noise_gating_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert!(Seeds3Options::new(f32::NAN, 1.0, 1.0 / 3.0, 2.0 / 3.0).is_err());
    assert!(Seeds3Options::new(1.0, 1.0, 0.75, 0.5).is_err());
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let error = sample_seeds_3(
        &backend,
        plan(&fixture, "seeds_2", &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture),
        Seeds3Options::default(),
        &context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(error, Err(Seeds3Error::WrongSampler { .. })));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture),
        Seeds3Options::default(),
        &cancelled_context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        error,
        Err(Seeds3Error::Tensor(TensorError::Cancelled))
    ));
    let deterministic_options =
        Seeds3Options::new(0.0, fixture.sampler_noise_scale, fixture.r_1, fixture.r_2)?;
    let (_, (before, after)) = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        request(&fixture),
        deterministic_options,
        &context,
        |_input, _sigma, step, stage| {
            let expected = &fixture.steps[step];
            let output = match stage {
                Seeds3DenoiserStage::Primary => expected.primary.as_slice(),
                Seeds3DenoiserStage::StageTwo => {
                    expected.stage_two.as_deref().ok_or("missing stage two")?
                }
                Seeds3DenoiserStage::StageThree => expected
                    .stage_three
                    .as_deref()
                    .ok_or("missing stage three")?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    assert_eq!(before, after, "eta zero must not consume source noise");
    Ok(())
}

#[test]
fn val_sampler_001_source_boundaries_and_atomic_failure_paths_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert!(Seeds3Options::new(1.0, 1.0, 0.5, 0.5).is_ok());
    assert!(Seeds3Options::new(1.0, 1.0, 0.5, 1.0).is_ok());
    assert!(Seeds3Options::new(1.0, 1.0, 0.0, 0.5).is_err());
    assert!(Seeds3Options::new(1.0, 1.0, 0.5, 1.01).is_err());
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let callback_error = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture),
        Seeds3Options::default(),
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
        |_, _, _| Err("injected callback failure".to_owned()),
    )
    .expect_err("callback failure must abort before any RNG draw");
    assert!(matches!(
        callback_error,
        Seeds3Error::Sampling(sim_sampler::SamplingError::Callback(_))
    ));

    for failed_stage in [
        Seeds3DenoiserStage::StageTwo,
        Seeds3DenoiserStage::StageThree,
    ] {
        let result = sample_seeds_3(
            &backend,
            plan(&fixture, &fixture.identity, &profile)?,
            &profile,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            request(&fixture),
            Seeds3Options::default(),
            &context,
            |_input, _sigma, step, stage| {
                if stage == failed_stage {
                    return Err("injected stage failure".to_owned());
                }
                let values = match stage {
                    Seeds3DenoiserStage::Primary => fixture.steps[step].primary.as_slice(),
                    Seeds3DenoiserStage::StageTwo => fixture.steps[step]
                        .stage_two
                        .as_deref()
                        .ok_or_else(|| "missing stage two".to_owned())?,
                    Seeds3DenoiserStage::StageThree => {
                        fixture.steps[step]
                            .stage_three
                            .as_deref()
                            .ok_or_else(|| "missing stage three".to_owned())?
                    }
                };
                tensor_from_f32(&backend, &fixture.shape, values, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        );
        assert!(matches!(
            result,
            Err(Seeds3Error::Denoiser { step: 0, stage, .. }) if stage == failed_stage
        ));
    }

    let wrong_descriptor = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture),
        Seeds3Options::default(),
        &context,
        |_, _, _, _| {
            tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        wrong_descriptor,
        Err(Seeds3Error::DenoiserContract {
            step: 0,
            stage: Seeds3DenoiserStage::Primary
        })
    ));

    let non_finite_terminal = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture),
        Seeds3Options::new(0.0, 0.0, fixture.r_1, fixture.r_2)?,
        &context,
        |_input, _sigma, step, stage| {
            let output = if step == 1 && stage == Seeds3DenoiserStage::Primary {
                vec![f32::NAN, 0.0]
            } else {
                match stage {
                    Seeds3DenoiserStage::Primary => fixture.steps[step].primary.clone(),
                    Seeds3DenoiserStage::StageTwo => fixture.steps[step]
                        .stage_two
                        .clone()
                        .ok_or_else(|| "missing stage two".to_owned())?,
                    Seeds3DenoiserStage::StageThree => fixture.steps[step]
                        .stage_three
                        .clone()
                        .ok_or_else(|| "missing stage three".to_owned())?,
                }
            };
            tensor_from_f32(&backend, &fixture.shape, &output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        non_finite_terminal,
        Err(Seeds3Error::NonFinite {
            step: 1,
            stage: "terminal denoiser",
            element: 0
        })
    ));

    let boundary_options = Seeds3Options::new(1.0, 1.0, 0.5, 1.0)?;
    let stage_three_sigmas = RefCell::new(Vec::new());
    let (_, (before, after)) = sample_seeds_3(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture),
        boundary_options,
        &context,
        |_input, sigma, step, stage| {
            if stage == Seeds3DenoiserStage::StageThree {
                stage_three_sigmas.borrow_mut().push(sigma);
            }
            let output = match stage {
                Seeds3DenoiserStage::Primary => fixture.steps[step].primary.as_slice(),
                Seeds3DenoiserStage::StageTwo => fixture.steps[step]
                    .stage_two
                    .as_deref()
                    .ok_or_else(|| "missing stage two".to_owned())?,
                Seeds3DenoiserStage::StageThree => fixture.steps[step]
                    .stage_three
                    .as_deref()
                    .ok_or_else(|| "missing stage three".to_owned())?,
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_close(
        &stage_three_sigmas.into_inner(),
        &[fixture.sigmas[1]],
        fixture.tolerance,
    );
    let mut oracle = request(&fixture).open_transaction(
        SEEDS_3_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    for _ in 0..3 {
        oracle.draw_normal(fixture.initial.len(), &cancellation)?;
    }
    assert_eq!(
        after,
        oracle.commit(),
        "r_2 == 1 still consumes the third source draw"
    );
    Ok(())
}
