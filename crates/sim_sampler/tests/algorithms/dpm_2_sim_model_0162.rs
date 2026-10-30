use sim_sampler::{
    CompatibilityNoiseRequest, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfileIdentity,
    generated_dpm_2_sim_model_0162::{
        DEFINITION, DPM_2_CHURN_NOISE_CONTRACT_ID, DPM_2_FEATURE_ID, DPM_2_SAMPLER_ID,
        DPM_2_SOURCE_ORDINAL, Dpm2DenoiserStage, Dpm2Options, Dpm2SamplerError, sample_dpm_2,
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
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpm_2_comfy_model_0162/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/dpm_2_sim_model_0162.rs");

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
    steps: Vec<StepFixture>,
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
struct OptionsFixture {
    s_churn: f32,
    s_tmin: f32,
    s_tmax: f32,
    s_noise: f32,
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
    midpoint_sigma: Option<f32>,
    midpoint_input: Option<Vec<f32>>,
    midpoint_denoised: Option<Vec<f32>>,
    midpoint_derivative: Option<Vec<f32>>,
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

fn plan(identity: &str, seed: u64, steps: u32) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new("analytical-epsilon-v1")?,
        seed,
        steps,
        1.0,
        1.0,
    )?)
}

fn profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("analytical-epsilon-v1")?)
}

fn noise_request(retry: u32, retry_policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "dpm2-fixture-v1",
        "attempt-0162",
        "KSampler-8",
        8,
        162,
        3,
        retry,
        retry_policy,
    )
}

fn options(fixture: &Fixture) -> Result<Dpm2Options, Dpm2SamplerError> {
    Dpm2Options::new(
        fixture.options.s_churn,
        fixture.options.s_tmin,
        fixture.options.s_tmax,
        fixture.options.s_noise,
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
    assert_eq!(fixture.identity, DPM_2_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPM_2_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPM_2_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpm_2_comfy_model_0162"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPM_2_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(registry.resolve(&SamplerIdentity::new("dpm2")?).is_err());
    assert!(SamplerIdentity::new("DPM_2").is_err());

    assert_eq!(fixture.rng_contract_id, DPM_2_CHURN_NOISE_CONTRACT_ID);
    let contract = rng_compatibility_contract(DPM_2_CHURN_NOISE_CONTRACT_ID)
        .ok_or("DPM2 churn RNG contract is unavailable")?;
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
        "def sample_dpm_2",
        "s_churn / (len(sigmas) - 1)",
        "torch.randn_like(x) * s_noise",
        "callback({'x': x",
        "sigma_mid = sigma_hat.log().lerp",
        "denoised_2 = model",
        "x = x + d_2 * dt_2",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpm_2\"") && line.contains("KSAMPLER_NAMES"))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .any(|line| { line.contains("sampler,dpm_2,") && line.ends_with(",COMFY-MODEL-0162") })
    );
    Ok(())
}

#[test]
fn val_sampler_001_dpm_2_matches_every_analytical_intermediate_and_callback_order()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let (sampling, checkpoints) = sample_dpm_2(
        &backend,
        plan(
            DPM_2_SAMPLER_ID,
            fixture.seed,
            u32::try_from(fixture.steps.len())?,
        )?,
        &profile()?,
        initial,
        &fixture.sigmas,
        options(&fixture)?,
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |input, sigma, step, stage| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            match stage {
                Dpm2DenoiserStage::Primary => {
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
                Dpm2DenoiserStage::Midpoint => {
                    events.borrow_mut().push(format!("midpoint-{step}"));
                    let midpoint_sigma = expected
                        .midpoint_sigma
                        .ok_or_else(|| format!("missing midpoint sigma at {step}"))?;
                    let midpoint_input = expected
                        .midpoint_input
                        .as_deref()
                        .ok_or_else(|| format!("missing midpoint input at {step}"))?;
                    let midpoint_denoised = expected
                        .midpoint_denoised
                        .as_deref()
                        .ok_or_else(|| format!("missing midpoint output at {step}"))?;
                    assert!((sigma - midpoint_sigma).abs() <= fixture.tolerance);
                    assert_close(
                        &values(&backend, input, &context).map_err(|error| error.to_string())?,
                        midpoint_input,
                        fixture.tolerance,
                    );
                    tensor_from_f32(&backend, &fixture.shape, midpoint_denoised, &context)
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
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.steps.len()).map_err(|error| error.to_string())?
            );
            assert_eq!(progress.sigma.to_bits(), expected.sigma.to_bits());
            assert_eq!(progress.next_sigma.to_bits(), expected.next_sigma.to_bits());
            assert_eq!(progress.sigma_hat.to_bits(), expected.sigma_hat.to_bits());
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
            "midpoint-0",
            "primary-1",
            "callback-1",
            "midpoint-1",
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
    assert_close(
        &values(
            &backend,
            sampling
                .latents
                .first()
                .ok_or("missing initial trace latent")?,
            &context,
        )?,
        &fixture.initial,
        0.0,
    );
    let (noise_before, noise_after) = checkpoints.ok_or("missing churn RNG checkpoints")?;
    let mut oracle = noise_request(0, RetryRngPolicy::Replay).open_transaction(
        DPM_2_CHURN_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::TorchSigned64,
        RngGenerationPlacement::Native(DeviceId::CPU),
        None,
        context.cancellation,
    )?;
    assert_eq!(noise_before, oracle.checkpoint());
    for (step, expected) in fixture.steps.iter().enumerate() {
        let gamma = if fixture.options.s_churn > 0.0
            && fixture.options.s_tmin <= expected.sigma
            && expected.sigma <= fixture.options.s_tmax
        {
            (fixture.options.s_churn / fixture.steps.len() as f32).min(2.0_f32.sqrt() - 1.0)
        } else {
            0.0
        };
        assert!((gamma - expected.gamma).abs() <= fixture.tolerance);
        let sigma_hat = expected.sigma * (gamma + 1.0);
        assert_eq!(sigma_hat.to_bits(), expected.sigma_hat.to_bits());
        assert!((sigma_hat / expected.sigma - 1.0 - expected.gamma).abs() <= fixture.tolerance);
        let current = values(
            &backend,
            item(&sampling.latents, step, "canonical pre-step latent")?,
            &context,
        )?;
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
                let perturbation_scale =
                    (expected.sigma_hat.powi(2) - expected.sigma.powi(2)).sqrt();
                let churned = current
                    .iter()
                    .zip(actual_noise.iter())
                    .map(|(current, noise)| noise.mul_add(perturbation_scale, *current))
                    .collect::<Vec<_>>();
                assert_close(&churned, &expected.churned, fixture.tolerance);
            }
            (None, None) => assert_close(&current, &expected.churned, fixture.tolerance),
            _ => return Err(format!("churn-noise fixture mismatch at step {step}").into()),
        }
        assert_close(
            &values(
                &backend,
                item(
                    &sampling.denoiser_evaluations,
                    step,
                    "primary denoiser evaluation",
                )?,
                &context,
            )?,
            &expected.primary_denoised,
            0.0,
        );
        let primary_derivative = expected
            .churned
            .iter()
            .zip(expected.primary_denoised.iter())
            .map(|(current, denoised)| (current - denoised) / expected.sigma_hat)
            .collect::<Vec<_>>();
        assert_close(
            &primary_derivative,
            &expected.primary_derivative,
            fixture.tolerance,
        );
        let next_index = step.checked_add(1).ok_or("next latent index overflowed")?;
        assert_close(
            &values(
                &backend,
                item(&sampling.latents, next_index, "next latent")?,
                &context,
            )?,
            &expected.next_latent,
            fixture.tolerance,
        );
        match (
            expected.midpoint_sigma,
            expected.midpoint_input.as_deref(),
            expected.midpoint_denoised.as_deref(),
            expected.midpoint_derivative.as_deref(),
        ) {
            (Some(sigma), Some(input), Some(denoised), Some(derivative)) => {
                let analytical_sigma = (expected.sigma_hat.ln()
                    + (expected.next_sigma.ln() - expected.sigma_hat.ln()) * 0.5)
                    .exp();
                assert!((analytical_sigma - sigma).abs() <= fixture.tolerance);
                let analytical_derivative = input
                    .iter()
                    .zip(denoised.iter())
                    .map(|(input, denoised)| (input - denoised) / sigma)
                    .collect::<Vec<_>>();
                assert_close(&analytical_derivative, derivative, fixture.tolerance);
            }
            (None, None, None, None) => assert_eq!(expected.next_sigma, 0.0),
            _ => return Err(format!("midpoint fixture mismatch at step {step}").into()),
        }
    }
    assert_eq!(noise_after, oracle.commit());
    assert_ne!(noise_before, noise_after);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn defaults_boundaries_retry_cancellation_and_failures_are_transactional()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;

    let defaults = Dpm2Options::source_defaults();
    assert_eq!(defaults, Dpm2Options::default());
    assert_eq!(defaults.s_churn(), 0.0);
    assert_eq!(defaults.s_tmin(), 0.0);
    assert_eq!(defaults.s_tmax(), f32::INFINITY);
    assert_eq!(defaults.s_noise(), 1.0);
    let (default_sampling, default_checkpoints) = sample_dpm_2(
        &backend,
        plan(DPM_2_SAMPLER_ID, 7, 1)?,
        &profile()?,
        initial.clone(),
        &[1.0, 0.0],
        defaults,
        noise_request(0, RetryRngPolicy::Replay),
        &context,
        |_, _, _, _| {
            tensor_from_f32(&backend, &[2], &[0.25, 0.5], &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert!(default_checkpoints.is_none());
    assert_close(
        &values(
            &backend,
            default_sampling
                .latents
                .get(1)
                .ok_or("missing default terminal latent")?,
            &context,
        )?,
        &[0.25, 0.5],
        0.0,
    );

    for invalid in [
        Dpm2Options::new(f32::NAN, 0.0, 1.0, 1.0),
        Dpm2Options::new(0.0, f32::INFINITY, 1.0, 1.0),
        Dpm2Options::new(0.0, 0.0, f32::NEG_INFINITY, 1.0),
        Dpm2Options::new(0.0, 0.0, 1.0, f32::NAN),
    ] {
        assert!(matches!(
            invalid,
            Err(Dpm2SamplerError::InvalidOption { .. })
        ));
    }
    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan("ddpm", 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2SamplerError::WrongSampler(value)) if value == "ddpm"
    ));
    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 1.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2SamplerError::Sampling(
            SamplingError::InvalidSigma { .. }
        ))
    ));

    let run_fixture = |retry, retry_policy| {
        sample_dpm_2(
            &backend,
            plan(
                DPM_2_SAMPLER_ID,
                fixture.seed,
                u32::try_from(fixture.steps.len()).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
            &profile().map_err(|error| error.to_string())?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)
                .map_err(|error| error.to_string())?,
            &fixture.sigmas,
            options(&fixture).map_err(|error| error.to_string())?,
            noise_request(retry, retry_policy),
            &context,
            |_, _, step, stage| {
                let expected = fixture
                    .steps
                    .get(step)
                    .ok_or_else(|| format!("missing fixture step {step}"))?;
                let output = match stage {
                    Dpm2DenoiserStage::Primary => expected.primary_denoised.as_slice(),
                    Dpm2DenoiserStage::Midpoint => expected
                        .midpoint_denoised
                        .as_deref()
                        .ok_or_else(|| format!("missing midpoint output {step}"))?,
                };
                tensor_from_f32(&backend, &fixture.shape, output, &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )
        .map_err(|error| error.to_string())
    };
    let (replay_a_sampling, replay_a_checkpoints) = run_fixture(0, RetryRngPolicy::Replay)?;
    let (replay_b_sampling, replay_b_checkpoints) = run_fixture(9, RetryRngPolicy::Replay)?;
    assert_eq!(replay_a_checkpoints, replay_b_checkpoints);
    assert_eq!(
        replay_a_sampling.latents.len(),
        replay_b_sampling.latents.len()
    );
    for (left, right) in replay_a_sampling
        .latents
        .iter()
        .zip(&replay_b_sampling.latents)
    {
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
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
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
        Err(Dpm2SamplerError::Tensor(TensorError::Cancelled))
    ));
    assert!(events.borrow().is_empty());

    let callback_cancelled = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &callback_context,
            |value, _, _, stage| {
                events.borrow_mut().push(stage);
                Ok(value.clone())
            },
            |_, _, _| {
                events.borrow_mut().push(Dpm2DenoiserStage::Primary);
                callback_cancelled.cancel();
                Ok::<(), String>(())
            }
        ),
        Err(Dpm2SamplerError::Sampling(SamplingError::Cancelled))
    ));
    assert_eq!(
        events.into_inner(),
        [Dpm2DenoiserStage::Primary, Dpm2DenoiserStage::Primary]
    );

    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Err("callback fault")
        ),
        Err(Dpm2SamplerError::Sampling(SamplingError::Callback(reason))) if reason == "callback fault"
    ));
    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |value, _, step, stage| if step == 0 && stage == Dpm2DenoiserStage::Midpoint { Err("midpoint fault".to_owned()) } else { Ok(value.clone()) },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2SamplerError::Denoiser { step: 0, stage: Dpm2DenoiserStage::Midpoint, reason }) if reason == "midpoint fault"
    ));
    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |_, _, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2SamplerError::DenoiserContract {
            step: 0,
            stage: Dpm2DenoiserStage::Primary
        })
    ));
    assert!(matches!(
        sample_dpm_2(
            &backend,
            plan(DPM_2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial,
            &[1.0, 0.0],
            defaults,
            noise_request(0, RetryRngPolicy::Replay),
            &context,
            |_, _, _, _| tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpm2SamplerError::NonFinite {
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
        "pub struct Dpm2Trace",
        "struct Dpm2Trace",
        "sigma_hats:",
        "churn_noises:",
        "midpoint_inputs:",
        "primary_derivatives:",
        "midpoint_derivatives:",
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
        ".observe_step(",
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction",
        "tensor_from_f32",
        "tensor_to_f32",
    ] {
        assert!(IMPLEMENTATION.contains(canonical_adapter));
    }
    assert!(!IMPLEMENTATION.contains("CompatibilityRngTransaction::open"));
    assert!(!IMPLEMENTATION.contains("RngCompatibilityRequest::new"));
    let primary_derivative = IMPLEMENTATION.find("let primary_derivative =");
    let callback = IMPLEMENTATION.find("let observed = session.observe_step(");
    let midpoint = IMPLEMENTATION.find("let midpoint_denoised = denoiser(");
    assert!(
        matches!(
            (primary_derivative, callback, midpoint),
            (Some(primary_derivative), Some(callback), Some(midpoint))
                if primary_derivative < callback && callback < midpoint
        ),
        "primary derivative, callback, and midpoint denoiser order diverged"
    );
}
