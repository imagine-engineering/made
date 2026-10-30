use sim_sampler::{
    CompatibilityNoiseRequest, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfileIdentity,
    generated_heun_sim_model_0187::{
        DEFINITION, HEUN_CHURN_NOISE_CONTRACT_ID, HEUN_FEATURE_ID, HEUN_SAMPLER_ID,
        HEUN_SOURCE_ORDINAL, HeunDenoiserStage, HeunOptions, HeunSamplerError, sample_heun,
    },
    generated_native_diffusion::NativeDiffusionSamplerError,
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
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/heun_comfy_model_0187/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/heun_sim_model_0187.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    rng_contract_id: String,
    seed: u64,
    options: OptionsFixture,
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
    equation_lines: Vec<usize>,
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct OptionsFixture {
    s_churn: f32,
    s_tmin: f32,
    s_tmax: f32,
    s_noise: f32,
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
    sigma: f32,
    next_sigma: f32,
    gamma: f32,
    sigma_hat: f32,
    noise: Option<Vec<f32>>,
    noise_bits: Option<Vec<u32>>,
    churned: Vec<f32>,
    primary_denoised: Vec<f32>,
    primary_derivative: Vec<f32>,
    predictor: Vec<f32>,
    correction_denoised: Option<Vec<f32>>,
    correction_derivative: Option<Vec<f32>>,
    average_derivative: Option<Vec<f32>>,
    next_latent: Vec<f32>,
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

fn profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("analytical-heun-row-v1")?)
}

fn plan(identity: &str, seed: u64, steps: usize) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile()?,
        seed,
        u32::try_from(steps)?,
        1.0,
        1.0,
    )?)
}

fn options(fixture: &Fixture) -> Result<HeunOptions, NativeDiffusionSamplerError> {
    HeunOptions::new(
        fixture.options.s_churn,
        fixture.options.s_tmin,
        fixture.options.s_tmax,
        fixture.options.s_noise,
    )
}

fn noise_request(
    fixture: &Fixture,
    retry: u32,
    retry_policy: RetryRngPolicy,
) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        &fixture.rng.workflow,
        &fixture.rng.attempt,
        &fixture.rng.node,
        fixture.rng.output,
        fixture.rng.execution_ordinal,
        fixture.rng.batch,
        retry,
        retry_policy,
    )
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

fn item<'a, T>(values: &'a [T], index: usize, role: &str) -> Result<&'a T, Box<dyn Error>> {
    values
        .get(index)
        .ok_or_else(|| format!("missing {role} at index {index}").into())
}

#[test]
fn definition_registry_source_and_rng_provenance_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, HEUN_SAMPLER_ID);
    assert_eq!(fixture.feature_id, HEUN_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, HEUN_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/heun_comfy_model_0187"
    );

    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(HEUN_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(registry.resolve(&SamplerIdentity::new("heun2")?).is_err());
    assert!(SamplerIdentity::new("Heun").is_err());

    assert_eq!(fixture.rng_contract_id, HEUN_CHURN_NOISE_CONTRACT_ID);
    let contract = rng_compatibility_contract(HEUN_CHURN_NOISE_CONTRACT_ID)
        .ok_or("Heun churn RNG contract is unavailable")?;
    assert_eq!(contract.operation(), RngCompatibilityOperation::NormalLike);
    assert_eq!(
        contract.phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(contract.symbol(), "torch.randn_like");

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
        "def sample_heun",
        "s_churn / (len(sigmas) - 1)",
        "torch.randn_like(x) * s_noise",
        "d = to_d(x, sigma_hat, denoised)",
        "callback({'x': x",
        "x_2 = x + d * dt",
        "denoised_2 = model(x_2",
        "d_2 = to_d(x_2",
        "d_prime = (d + d_2) / 2",
        "x = x + d_prime * dt",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("KSAMPLER_NAMES") && line.contains("\"heun\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(
                |line| line.starts_with("sampler,heun,") && line.ends_with(",COMFY-MODEL-0187")
            )
    );
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_heun_intermediate_rng_and_callback() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let (sampling, checkpoints) = sample_heun(
        &backend,
        plan(HEUN_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        options(&fixture)?,
        noise_request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay),
        &context,
        |input, sigma, step, stage| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            match stage {
                HeunDenoiserStage::Primary => {
                    events.borrow_mut().push(format!("primary-{step}"));
                    assert_eq!(sigma.to_bits(), expected.sigma_hat.to_bits());
                    assert_close(
                        &values(&backend, input, &context).map_err(|error| error.to_string())?,
                        &expected.churned,
                        fixture.tolerance,
                    );
                    tensor_from_f32(
                        &backend,
                        &fixture.shape,
                        &expected.primary_denoised,
                        &context,
                    )
                    .map_err(|error| error.to_string())
                }
                HeunDenoiserStage::Correction => {
                    events.borrow_mut().push(format!("correction-{step}"));
                    let correction = expected
                        .correction_denoised
                        .as_deref()
                        .ok_or_else(|| format!("unexpected terminal correction at step {step}"))?;
                    assert_eq!(sigma.to_bits(), expected.next_sigma.to_bits());
                    assert_close(
                        &values(&backend, input, &context).map_err(|error| error.to_string())?,
                        &expected.predictor,
                        fixture.tolerance,
                    );
                    tensor_from_f32(&backend, &fixture.shape, correction, &context)
                        .map_err(|error| error.to_string())
                }
            }
        },
        |progress, current, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected callback step {step}"))?;
            events.borrow_mut().push(format!("callback-{step}"));
            assert_eq!(progress.sigma.to_bits(), expected.sigma.to_bits());
            assert_eq!(progress.next_sigma.to_bits(), expected.next_sigma.to_bits());
            assert_eq!(progress.sigma_hat.to_bits(), expected.sigma_hat.to_bits());
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.steps.len()).map_err(|error| error.to_string())?
            );
            assert_close(
                &values(&backend, current, &context).map_err(|error| error.to_string())?,
                &expected.churned,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.primary_denoised,
                fixture.tolerance,
            );
            Ok::<(), String>(())
        },
    )?;

    assert_eq!(
        events.into_inner(),
        [
            "primary-0",
            "callback-0",
            "correction-0",
            "primary-1",
            "callback-1",
            "correction-1",
            "primary-2",
            "callback-2",
        ]
    );
    assert_close(
        &values(&backend, &initial_alias, &context)?,
        &fixture.initial,
        0.0,
    );
    assert_eq!(sampling.sigmas, fixture.sigmas);
    assert_eq!(sampling.latents.len(), fixture.steps.len() + 1);
    assert_eq!(sampling.denoiser_evaluations.len(), fixture.steps.len());

    let (noise_before, noise_after) = checkpoints.ok_or("missing churn RNG checkpoints")?;
    let mut oracle = noise_request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay)
        .open_transaction(
            HEUN_CHURN_NOISE_CONTRACT_ID,
            i128::from(fixture.seed),
            RngSeedTransform::TorchSigned64,
            RngGenerationPlacement::Native(DeviceId::CPU),
            None,
            context.cancellation,
        )?;
    assert_eq!(noise_before, oracle.checkpoint());
    for (step, expected) in fixture.steps.iter().enumerate() {
        let current = values(
            &backend,
            item(&sampling.latents, step, "pre-step latent")?,
            &context,
        )?;
        let analytical_gamma = if fixture.options.s_churn > 0.0
            && fixture.options.s_tmin <= expected.sigma
            && expected.sigma <= fixture.options.s_tmax
        {
            (fixture.options.s_churn / fixture.steps.len() as f32).min(2.0_f32.sqrt() - 1.0)
        } else {
            0.0
        };
        assert!((analytical_gamma - expected.gamma).abs() <= fixture.tolerance);
        assert_eq!(
            (expected.sigma * (analytical_gamma + 1.0)).to_bits(),
            expected.sigma_hat.to_bits()
        );
        match (expected.noise.as_deref(), expected.noise_bits.as_deref()) {
            (Some(expected_noise), Some(expected_bits)) => {
                let actual_noise = oracle
                    .draw_normal(fixture.initial.len(), &cancellation)?
                    .into_iter()
                    .map(|value| value as f32 * fixture.options.s_noise)
                    .collect::<Vec<_>>();
                assert_close(&actual_noise, expected_noise, 0.0);
                assert_eq!(
                    actual_noise
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected_bits
                );
                let scale = (expected.sigma_hat.powi(2) - expected.sigma.powi(2)).sqrt();
                let churned = current
                    .iter()
                    .zip(actual_noise.iter())
                    .map(|(current, noise)| noise.mul_add(scale, *current))
                    .collect::<Vec<_>>();
                assert_close(&churned, &expected.churned, fixture.tolerance);
            }
            (None, None) => assert_close(&current, &expected.churned, fixture.tolerance),
            _ => return Err(format!("churn fixture mismatch at step {step}").into()),
        }

        let primary_derivative = expected
            .churned
            .iter()
            .zip(expected.primary_denoised.iter())
            .map(|(input, denoised)| (input - denoised) / expected.sigma_hat)
            .collect::<Vec<_>>();
        assert_close(
            &primary_derivative,
            &expected.primary_derivative,
            fixture.tolerance,
        );
        let delta = expected.next_sigma - expected.sigma_hat;
        let predictor = expected
            .churned
            .iter()
            .zip(primary_derivative.iter())
            .map(|(input, derivative)| derivative.mul_add(delta, *input))
            .collect::<Vec<_>>();
        assert_close(&predictor, &expected.predictor, fixture.tolerance);

        match (
            expected.correction_denoised.as_deref(),
            expected.correction_derivative.as_deref(),
            expected.average_derivative.as_deref(),
        ) {
            (Some(denoised), Some(correction), Some(average)) => {
                let analytical_correction = predictor
                    .iter()
                    .zip(denoised.iter())
                    .map(|(input, denoised)| (input - denoised) / expected.next_sigma)
                    .collect::<Vec<_>>();
                assert_close(&analytical_correction, correction, fixture.tolerance);
                let analytical_average = primary_derivative
                    .iter()
                    .zip(analytical_correction.iter())
                    .map(|(primary, correction)| (primary + correction) * 0.5)
                    .collect::<Vec<_>>();
                assert_close(&analytical_average, average, fixture.tolerance);
            }
            (None, None, None) => assert_eq!(expected.next_sigma, 0.0),
            _ => return Err(format!("correction fixture mismatch at step {step}").into()),
        }
        assert_close(
            &values(
                &backend,
                item(&sampling.latents, step + 1, "next latent")?,
                &context,
            )?,
            &expected.next_latent,
            fixture.tolerance,
        );
    }
    assert_eq!(noise_after, oracle.commit());
    assert_ne!(noise_before, noise_after);
    assert_close(
        &values(
            &backend,
            sampling.latents.last().ok_or("missing terminal latent")?,
            &context,
        )?,
        &fixture.terminal,
        fixture.tolerance,
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn boundaries_retry_cancellation_and_failures_are_transactional() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;
    let defaults = HeunOptions::source_defaults();
    assert_eq!(defaults, HeunOptions::default());
    assert_eq!(defaults.s_churn(), 0.0);
    assert_eq!(defaults.s_tmin(), 0.0);
    assert_eq!(defaults.s_tmax(), f32::INFINITY);
    assert_eq!(defaults.s_noise(), 1.0);

    let stages = RefCell::new(Vec::new());
    let (terminal_sampling, terminal_checkpoints) = sample_heun(
        &backend,
        plan(HEUN_SAMPLER_ID, 7, 1)?,
        &profile()?,
        initial.clone(),
        &[1.0, 0.0],
        defaults,
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        &context,
        |_, _, _, stage| {
            stages.borrow_mut().push(stage);
            tensor_from_f32(&backend, &[2], &[0.25, 0.5], &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(stages.into_inner(), [HeunDenoiserStage::Primary]);
    assert!(terminal_checkpoints.is_none());
    assert_close(
        &values(
            &backend,
            terminal_sampling
                .latents
                .get(1)
                .ok_or("missing terminal-step latent")?,
            &context,
        )?,
        &[0.25, 0.5],
        0.0,
    );

    for invalid in [
        HeunOptions::new(f32::NAN, 0.0, 1.0, 1.0),
        HeunOptions::new(0.0, f32::INFINITY, 1.0, 1.0),
        HeunOptions::new(0.0, 0.0, f32::NEG_INFINITY, 1.0),
        HeunOptions::new(0.0, 0.0, 1.0, f32::NAN),
    ] {
        assert!(matches!(
            invalid,
            Err(NativeDiffusionSamplerError::InvalidEulerOption { .. })
        ));
    }
    assert!(matches!(
        sample_heun(
            &backend,
            plan("ddpm", 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunSamplerError::WrongSampler(value)) if value == "ddpm"
    ));
    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 1.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunSamplerError::Sampling(
            SamplingError::InvalidSigma { .. }
        ))
    ));

    let run_fixture = |retry, retry_policy| {
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, fixture.seed, fixture.steps.len())
                .map_err(|error| error.to_string())?,
            &profile().map_err(|error| error.to_string())?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)
                .map_err(|error| error.to_string())?,
            &fixture.sigmas,
            options(&fixture).map_err(|error| error.to_string())?,
            noise_request(&fixture, retry, retry_policy),
            &context,
            |_, _, step, stage| {
                let expected = fixture
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("missing fixture step {step}"))?;
                let output = match stage {
                    HeunDenoiserStage::Primary => expected.primary_denoised.as_slice(),
                    HeunDenoiserStage::Correction => expected
                        .correction_denoised
                        .as_deref()
                        .ok_or_else(|| format!("missing correction output {step}"))?,
                };
                tensor_from_f32(&backend, &fixture.shape, output, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )
        .map_err(|error| error.to_string())
    };
    let (replay_a, replay_a_checkpoints) = run_fixture(0, RetryRngPolicy::Replay)?;
    let (replay_b, replay_b_checkpoints) = run_fixture(9, RetryRngPolicy::Replay)?;
    assert_eq!(replay_a_checkpoints, replay_b_checkpoints);
    for (left, right) in replay_a.latents.iter().zip(replay_b.latents.iter()) {
        assert_eq!(
            values(&backend, left, &context)?,
            values(&backend, right, &context)?
        );
    }
    let (_, advance_checkpoints) = run_fixture(9, RetryRngPolicy::Advance)?;
    let (replay_before, replay_after) =
        replay_a_checkpoints.ok_or("missing replay RNG checkpoints")?;
    let (advance_before, advance_after) =
        advance_checkpoints.ok_or("missing advance RNG checkpoints")?;
    assert_ne!(replay_before, advance_before);
    assert_ne!(replay_after, advance_after);

    let pre_cancelled = CancellationToken::default();
    pre_cancelled.cancel();
    let pre_cancelled_context = execution_context(&backend, &authority, &pre_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &pre_cancelled_context,
            |value, _, _, _| {
                events.borrow_mut().push("denoiser");
                Ok(value.clone())
            },
            |_, _, _| {
                events.borrow_mut().push("callback");
                Ok::<(), String>(())
            }
        ),
        Err(HeunSamplerError::Tensor(TensorError::Cancelled))
    ));
    assert!(events.borrow().is_empty());

    let callback_cancelled = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &callback_context,
            |value, _, _, stage| {
                events.borrow_mut().push(match stage {
                    HeunDenoiserStage::Primary => "primary",
                    HeunDenoiserStage::Correction => "correction",
                });
                Ok(value.clone())
            },
            |_, _, _| {
                events.borrow_mut().push("callback");
                callback_cancelled.cancel();
                Ok::<(), String>(())
            }
        ),
        Err(HeunSamplerError::EulerFoundation(
            NativeDiffusionSamplerError::Sampling(SamplingError::Cancelled)
        ))
    ));
    assert_eq!(events.into_inner(), ["primary", "callback"]);

    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Err("callback fault")
        ),
        Err(HeunSamplerError::EulerFoundation(
            NativeDiffusionSamplerError::Sampling(SamplingError::Callback(reason))
        )) if reason == "callback fault"
    ));
    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, step, stage| if step == 0 && stage == HeunDenoiserStage::Correction {
                Err("correction fault".to_owned())
            } else {
                Ok(value.clone())
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunSamplerError::Denoiser {
            step: 0,
            stage: HeunDenoiserStage::Correction,
            reason
        }) if reason == "correction fault"
    ));
    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |_, _, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunSamplerError::DenoiserContract {
            step: 0,
            stage: HeunDenoiserStage::Primary
        })
    ));
    assert!(matches!(
        sample_heun(
            &backend,
            plan(HEUN_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial,
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |_, _, _, _| tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunSamplerError::NonFinite {
            step: 0,
            stage: "primary denoiser",
            element: 0
        })
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn row_is_only_an_equation_and_compatibility_adapter() {
    for forbidden in [
        "pub struct HeunTrace",
        "struct HeunTrace",
        "sigma_hats:",
        "churn_noises:",
        "pub struct SamplingSession",
        concat!("pub struct Cancellation", "Token"),
        "pub struct CompatibilityRngTransaction",
        "RngStream::new",
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
            "row contains forbidden owner or placeholder {forbidden}"
        );
    }
    for canonical_adapter in [
        "SamplingSession::new",
        "observe_euler_denoised",
        "advance_euler",
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction",
        "tensor_from_f32",
        "tensor_to_f32",
    ] {
        assert!(IMPLEMENTATION.contains(canonical_adapter));
    }
    assert!(!IMPLEMENTATION.contains("CompatibilityRngTransaction::open"));
    assert!(!IMPLEMENTATION.contains("RngCompatibilityRequest::new"));
    let primary = IMPLEMENTATION.find("let primary_denoised = denoiser(");
    let callback = IMPLEMENTATION.find("let observed = observe_euler_denoised(");
    let correction = IMPLEMENTATION.find("let correction_denoised =");
    let average = IMPLEMENTATION.find("let average_derivative = average_derivatives(");
    assert!(
        matches!(
            (primary, callback, correction, average),
            (Some(primary), Some(callback), Some(correction), Some(average))
                if primary < callback && callback < correction && correction < average
        ),
        "primary, callback, correction, and average order diverged"
    );
}
