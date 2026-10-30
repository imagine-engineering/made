use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation,
    SamplerIdentity, SamplerRegistry, SamplingPlan, SamplingProfile, SamplingSnrMode,
    generated_lcm_sim_model_0191::{
        DEFINITION, LCM_FEATURE_ID, LCM_NOISE_CONTRACT_ID, LCM_SAMPLER_ID, LCM_SOURCE_ORDINAL,
        LcmError, LcmOptions, lcm_rng_profile, sample_lcm,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/lcm_comfy_model_0191/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/lcm_sim_model_0191.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    tolerance: f32,
    seed: u64,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    profile_sigmas: Vec<f32>,
    profile_noise_scale: f32,
    options: OptionsFixture,
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

#[derive(Clone, Copy, Debug, Deserialize)]
struct OptionsFixture {
    noise_scale_start: f32,
    noise_scale_end: f32,
    noise_clip_standard_deviations: f32,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    raw_noise: Option<Vec<f32>>,
    noise_standard_deviation: Option<f32>,
    noise_clip: Option<f32>,
    interpolation: Option<f32>,
    noise_scale: Option<f32>,
    clipped_scaled_noise: Option<Vec<f32>>,
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

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        sim_sampler::SamplingProfileIdentity::new("lcm-flow-row-v1")?,
        PredictionInterpretation::Flow,
        Arc::<[f32]>::from(fixture.profile_sigmas.clone()),
        SamplingSnrMode::ConstantFlow { shift: 1.0 },
        fixture.profile_noise_scale,
    )?)
}

fn plan(fixture: &Fixture, identity: &str) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile(fixture)?.identity().clone(),
        fixture.seed,
        u32::try_from(fixture.steps.len())?,
        1.0,
        1.0,
    )?)
}

fn options(fixture: &Fixture) -> LcmOptions {
    LcmOptions {
        noise_scale_start: fixture.options.noise_scale_start,
        noise_scale_end: Some(fixture.options.noise_scale_end),
        noise_clip_standard_deviations: fixture.options.noise_clip_standard_deviations,
    }
}

fn noise_request(retry: u32, retry_policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "lcm-fixture-v1",
        "attempt-0191",
        "KSampler-26",
        26,
        191,
        7,
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

#[test]
fn generated_lcm_sim_model_0191_definition_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, LCM_SAMPLER_ID);
    assert_eq!(fixture.feature_id, LCM_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, LCM_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 26);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/lcm_comfy_model_0191"
    );
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(LCM_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new("latent-consistency")?)
            .is_err()
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
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_lcm(",
        "n_steps = max(1, len(sigmas) - 1)",
        "callback({'x': x",
        "x = denoised",
        "if sigmas[i + 1] > 0:",
        "noise = noise_sampler(sigmas[i], sigmas[i + 1])",
        "clip_val = noise_clip_std * noise.std()",
        "noise = noise.clamp(min=-clip_val, max=clip_val)",
        "s_noise_i = s_start + (s_end - s_start) * t",
        "x = model_sampling.noise_scaling(sigmas[i + 1], noise, x)",
    ] {
        assert!(equations.contains(fragment), "missing source equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"ddpm\", \"lcm\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| {
                line.starts_with("sampler,lcm,") && line.ends_with(",COMFY-MODEL-0191")
            })
    );
    for forbidden in [
        "struct SamplingSession",
        "struct SamplingProgress",
        "struct SamplingTrace",
        "struct CancellationToken",
        "struct CompatibilityNoiseRequest",
        "struct CompatibilityRngTransaction",
        "struct RngStream",
        "CpuWorkspaceAuthority",
        "authorize_workspace",
        "fn validate_sigmas",
        "unsafe {",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn generated_lcm_sim_model_0191_matches_noise_callback_and_latent_intermediates()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;

    let (seed_transform, placement) = lcm_rng_profile(DeviceId::CPU);
    let mut oracle = noise_request(2, RetryRngPolicy::Replay).open_transaction(
        LCM_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        seed_transform,
        placement,
        None,
        &cancellation,
    )?;
    let expected_before = oracle.checkpoint();
    for step in &fixture.steps {
        if let Some(expected_noise) = &step.raw_noise {
            let actual = oracle
                .draw_normal(expected_noise.len(), &cancellation)?
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            assert_close(&actual, expected_noise, fixture.tolerance);
        }
    }
    let expected_after = oracle.commit();

    let denoiser_inputs = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let (trace, noise_before, noise_after) = sample_lcm(
        &backend,
        plan(&fixture, LCM_SAMPLER_ID)?,
        &profile(&fixture)?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(2, RetryRngPolicy::Replay),
        options(&fixture),
        &context,
        |latent, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            assert_eq!(sigma.to_bits(), expected.sigma.to_bits());
            denoiser_inputs
                .borrow_mut()
                .push(values(&backend, latent, &context).map_err(|error| error.to_string())?);
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            callbacks.borrow_mut().push((
                *progress,
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<(), String>(())
        },
    )?;

    assert_eq!(noise_before, expected_before);
    assert_eq!(noise_after, expected_after);
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    assert_close(
        &values(&backend, trace.latents.first().ok_or("missing initial")?, &context)?,
        &fixture.initial,
        fixture.tolerance,
    );
    for (index, step) in fixture.steps.iter().enumerate() {
        assert_eq!(step.step, index);
        let scheduled_next = fixture
            .sigmas
            .get(index + 1)
            .copied()
            .ok_or("missing scheduled next sigma")?;
        assert_eq!(step.next_sigma.to_bits(), scheduled_next.to_bits());
        assert_close(
            denoiser_inputs.borrow().get(index).ok_or("missing input")?,
            &step.latent_before,
            fixture.tolerance,
        );
        let (progress, callback_latent, callback_denoised) = callbacks
            .borrow()
            .get(index)
            .cloned()
            .ok_or("missing callback")?;
        assert_eq!(progress.step, u32::try_from(index)?);
        assert_eq!(progress.total_steps, u32::try_from(fixture.steps.len())?);
        assert_eq!(progress.sigma.to_bits(), step.sigma.to_bits());
        assert_eq!(progress.sigma_hat.to_bits(), step.sigma.to_bits());
        assert_eq!(progress.next_sigma.to_bits(), step.next_sigma.to_bits());
        assert_close(&callback_latent, &step.latent_before, fixture.tolerance);
        assert_close(&callback_denoised, &step.denoised, fixture.tolerance);
        assert_close(
            &values(
                &backend,
                trace.latents.get(index + 1).ok_or("missing latent")?,
                &context,
            )?,
            &step.latent_after,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace
                    .denoiser_evaluations
                    .get(index)
                    .ok_or("missing denoised")?,
                &context,
            )?,
            &step.denoised,
            fixture.tolerance,
        );
        if let (Some(raw), Some(standard_deviation), Some(clip), Some(interpolation), Some(scale), Some(prepared)) = (
            &step.raw_noise,
            step.noise_standard_deviation,
            step.noise_clip,
            step.interpolation,
            step.noise_scale,
            &step.clipped_scaled_noise,
        ) {
            let mean = raw.iter().copied().sum::<f32>() / raw.len() as f32;
            let degrees_of_freedom = raw
                .len()
                .checked_sub(1)
                .ok_or("noise fixture must have at least two elements")?;
            let variance = raw
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f32>()
                / degrees_of_freedom as f32;
            assert!((variance.sqrt() - standard_deviation).abs() <= fixture.tolerance);
            assert!(
                (fixture.options.noise_clip_standard_deviations * standard_deviation - clip).abs()
                    <= fixture.tolerance
            );
            let interpolation_denominator = fixture
                .steps
                .len()
                .checked_sub(1)
                .ok_or("LCM fixture needs a nonterminal interpolation denominator")?;
            let expected_interpolation = index as f32 / interpolation_denominator as f32;
            assert!((interpolation - expected_interpolation).abs() <= fixture.tolerance);
            let expected_scale = fixture.options.noise_scale_start
                + (fixture.options.noise_scale_end - fixture.options.noise_scale_start)
                    * interpolation;
            assert!((scale - expected_scale).abs() <= fixture.tolerance);
            let expected_prepared = raw
                .iter()
                .map(|value| value.clamp(-clip, clip) * scale)
                .collect::<Vec<_>>();
            assert_close(&expected_prepared, prepared, fixture.tolerance);
        } else {
            assert_eq!(step.next_sigma, 0.0);
        }
    }
    assert_close(
        &values(&backend, trace.latents.last().ok_or("missing terminal")?, &context)?,
        &fixture.terminal,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn generated_lcm_sim_model_0191_terminal_step_draws_no_noise_and_retry_is_deterministic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let run = |retry, retry_policy| {
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            noise_request(retry, retry_policy),
            options(&fixture),
            &context,
            |_latent, _sigma, step| {
                let denoised = fixture
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &denoised.denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        )
        .map_err(Box::<dyn Error>::from)
    };
    let replay_a = run(2, RetryRngPolicy::Replay)?;
    let replay_b = run(2, RetryRngPolicy::Replay)?;
    assert_eq!(replay_a.1, replay_b.1);
    assert_eq!(replay_a.2, replay_b.2);
    for (left, right) in replay_a.0.latents.iter().zip(&replay_b.0.latents) {
        assert_close(
            &values(&backend, left, &context)?,
            &values(&backend, right, &context)?,
            0.0,
        );
    }
    let advanced = run(3, RetryRngPolicy::Advance)?;
    assert_ne!(advanced.1, replay_a.1);
    assert_ne!(advanced.2, replay_a.2);

    let terminal_sigmas = [0.5, 0.0];
    let terminal_plan = SamplingPlan::new(
        LCM_SAMPLER_ID,
        "normal",
        profile(&fixture)?.identity().clone(),
        fixture.seed,
        1,
        1.0,
        1.0,
    )?;
    let (terminal, before, after) = sample_lcm(
        &backend,
        terminal_plan,
        &profile(&fixture)?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &terminal_sigmas,
        noise_request(2, RetryRngPolicy::Replay),
        LcmOptions::default(),
        &context,
        |_latent, _sigma, _step| {
            tensor_from_f32(&backend, &fixture.shape, &fixture.terminal, &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    )?;
    assert_eq!(before, after);
    assert_close(
        &values(&backend, terminal.latents.last().ok_or("missing terminal")?, &context)?,
        &fixture.terminal,
        0.0,
    );
    Ok(())
}

#[test]
fn generated_lcm_sim_model_0191_failures_and_cancellation_are_typed_and_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);

    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, "euler")?,
            &profile(&fixture)?,
            initial()?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &context,
            |_latent, _sigma, _step| Err("must not run".to_owned()),
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::WrongSampler { .. })
    ));
    let mut invalid_options = options(&fixture);
    invalid_options.noise_scale_start = f32::NAN;
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            initial()?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            invalid_options,
            &context,
            |_latent, _sigma, _step| Err("must not run".to_owned()),
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::InvalidOption { name: "s_noise", .. })
    ));
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            initial()?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &context,
            |_latent, _sigma, _step| Err("fixture failure".to_owned()),
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::Denoiser { step: 0, .. })
    ));
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            initial()?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &context,
            |_latent, _sigma, _step| {
                tensor_from_f32(&backend, &[1, 1, 1, 2], &[0.0, 1.0], &context)
                    .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::DenoiserContract { step: 0 })
    ));
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            initial()?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &context,
            |_latent, _sigma, _step| {
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &[f32::NAN, 0.0, 0.0, 0.0],
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::NonFinite {
            step: 0,
            stage: "denoiser output",
            element: 0
        })
    ));

    let singleton_shape = [1_u64, 1, 1, 1];
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            tensor_from_f32(&backend, &singleton_shape, &[0.0], &context)?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &context,
            |_latent, _sigma, _step| {
                tensor_from_f32(&backend, &singleton_shape, &[0.0], &context)
                    .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::NoiseClipCardinality { step: 0 })
    ));

    let denoiser_calls = RefCell::new(0_usize);
    let callback_calls = RefCell::new(0_usize);
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            initial()?,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &context,
            |_latent, _sigma, step| {
                *denoiser_calls.borrow_mut() += 1;
                let denoised = fixture
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &denoised.denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| {
                *callback_calls.borrow_mut() += 1;
                Err("callback failure".to_owned())
            },
        ),
        Err(LcmError::Sampling(sim_sampler::SamplingError::Callback(_)))
    ));
    assert_eq!(*denoiser_calls.borrow(), 1);
    assert_eq!(*callback_calls.borrow(), 1);

    let callback_cancellation = CancellationToken::default();
    let callback_cancellation_context =
        execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_cancellation_initial = tensor_from_f32(
        &backend,
        &fixture.shape,
        &fixture.initial,
        &callback_cancellation_context,
    )?;
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            callback_cancellation_initial,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &callback_cancellation_context,
            |_latent, _sigma, step| {
                let denoised = fixture
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &denoised.denoised,
                    &callback_cancellation_context,
                )
                .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| {
                callback_cancellation.cancel();
                Ok::<(), String>(())
            },
        ),
        Err(LcmError::Sampling(sim_sampler::SamplingError::Cancelled))
    ));

    let cancelled = CancellationToken::default();
    let cancelled_initial = initial()?;
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    assert!(matches!(
        sample_lcm(
            &backend,
            plan(&fixture, LCM_SAMPLER_ID)?,
            &profile(&fixture)?,
            cancelled_initial,
            &fixture.sigmas,
            noise_request(2, RetryRngPolicy::Replay),
            options(&fixture),
            &cancelled_context,
            |_latent, _sigma, _step| Err("must not run".to_owned()),
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        ),
        Err(LcmError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}
