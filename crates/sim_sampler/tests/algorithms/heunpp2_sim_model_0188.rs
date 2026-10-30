use sim_sampler::{
    CompatibilityNoiseRequest, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfileIdentity,
    generated_heunpp2_sim_model_0188::{
        DEFINITION, HEUNPP2_FEATURE_ID, HEUNPP2_NOISE_CONTRACT_ID, HEUNPP2_SAMPLER_ID,
        HEUNPP2_SOURCE_ORDINAL, HeunPp2DenoiserStage, HeunPp2Options, HeunPp2SamplerError,
        sample_heunpp2,
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
    "/../sim_test_support/fixtures/samplers/heunpp2_comfy_model_0188/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/heunpp2_sim_model_0188.rs");

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
    branch: String,
    sigma: f32,
    next_sigma: f32,
    gamma: f32,
    sigma_hat: f32,
    noise: Vec<f32>,
    noise_bits: Vec<u32>,
    churned: Vec<f32>,
    primary_denoised: Vec<f32>,
    primary_derivative: Vec<f32>,
    correction_input: Option<Vec<f32>>,
    correction_denoised: Option<Vec<f32>>,
    correction_derivative: Option<Vec<f32>>,
    lookahead_sigma: Option<f32>,
    lookahead_input: Option<Vec<f32>>,
    lookahead_denoised: Option<Vec<f32>>,
    lookahead_derivative: Option<Vec<f32>>,
    weights: Option<Vec<f32>>,
    weighted_derivative: Option<Vec<f32>>,
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
    Ok(SamplingProfileIdentity::new("analytical-heunpp2-row-v1")?)
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

fn options(fixture: &Fixture) -> Result<HeunPp2Options, NativeDiffusionSamplerError> {
    HeunPp2Options::new(
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

fn derivative(input: &[f32], denoised: &[f32], sigma: f32) -> Vec<f32> {
    input
        .iter()
        .zip(denoised)
        .map(|(input, denoised)| (input - denoised) / sigma)
        .collect()
}

fn advance(input: &[f32], derivative: &[f32], delta: f32) -> Vec<f32> {
    input
        .iter()
        .zip(derivative)
        .map(|(input, derivative)| derivative.mul_add(delta, *input))
        .collect()
}

fn weighted(derivatives: &[&[f32]], weights: &[f32]) -> Vec<f32> {
    (0..derivatives.first().map_or(0, |values| values.len()))
        .map(|element| {
            derivatives
                .iter()
                .zip(weights)
                .fold(0.0, |value, (derivative, weight)| {
                    derivative[element].mul_add(*weight, value)
                })
        })
        .collect()
}

#[test]
fn definition_registry_source_and_rng_provenance_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, HEUNPP2_SAMPLER_ID);
    assert_eq!(fixture.feature_id, HEUNPP2_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, HEUNPP2_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/heunpp2_comfy_model_0188"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(HEUNPP2_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        registry
            .resolve(&SamplerIdentity::new("heun_pp2")?)
            .is_err()
    );
    assert!(SamplerIdentity::new("Heunpp2").is_err());

    assert_eq!(fixture.rng_contract_id, HEUNPP2_NOISE_CONTRACT_ID);
    let contract = rng_compatibility_contract(HEUNPP2_NOISE_CONTRACT_ID)
        .ok_or("Heun++2 RNG contract is unavailable")?;
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
        "def sample_heunpp2",
        "eps = torch.randn_like(x) * s_noise",
        "if sigmas[i + 1] == s_end",
        "elif sigmas[i + 2] == s_end",
        "w = 2 * sigmas[0]",
        "d_prime = d * w1 + d_2 * w2",
        "dt_2 = sigmas[i + 2] - sigmas[i + 1]",
        "x_3 = x_2 + d_2 * dt_2",
        "w = 3 * sigmas[0]",
        "d_prime = w1 * d + w2 * d_2 + w3 * d_3",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("KSAMPLER_NAMES") && line.contains("\"heunpp2\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| {
                line.starts_with("sampler,heunpp2,") && line.ends_with(",COMFY-MODEL-0188")
            })
    );
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_branch_intermediate_rng_and_callback() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let (sampling, (noise_before, noise_after)) = sample_heunpp2(
        &backend,
        plan(HEUNPP2_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
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
            let (event, expected_sigma, expected_input, output) = match stage {
                HeunPp2DenoiserStage::Primary => (
                    format!("primary-{step}"),
                    expected.sigma_hat,
                    expected.churned.as_slice(),
                    expected.primary_denoised.as_slice(),
                ),
                HeunPp2DenoiserStage::Correction => (
                    format!("correction-{step}"),
                    expected.next_sigma,
                    expected
                        .correction_input
                        .as_deref()
                        .ok_or_else(|| format!("unexpected correction at step {step}"))?,
                    expected
                        .correction_denoised
                        .as_deref()
                        .ok_or_else(|| format!("missing correction at step {step}"))?,
                ),
                HeunPp2DenoiserStage::Lookahead => (
                    format!("lookahead-{step}"),
                    expected
                        .lookahead_sigma
                        .ok_or_else(|| format!("unexpected lookahead at step {step}"))?,
                    expected
                        .lookahead_input
                        .as_deref()
                        .ok_or_else(|| format!("missing lookahead input at step {step}"))?,
                    expected
                        .lookahead_denoised
                        .as_deref()
                        .ok_or_else(|| format!("missing lookahead output at step {step}"))?,
                ),
            };
            events.borrow_mut().push(event);
            assert_eq!(sigma.to_bits(), expected_sigma.to_bits());
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                expected_input,
                fixture.tolerance,
            );
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
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
            "lookahead-0",
            "primary-1",
            "callback-1",
            "correction-1",
            "lookahead-1",
            "primary-2",
            "callback-2",
            "correction-2",
            "primary-3",
            "callback-3"
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

    let mut oracle = noise_request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay)
        .open_transaction(
            HEUNPP2_NOISE_CONTRACT_ID,
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
            sampling.latents.get(step).ok_or("missing current latent")?,
            &context,
        )?;
        let noise = oracle
            .draw_normal(fixture.initial.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32 * fixture.options.s_noise)
            .collect::<Vec<_>>();
        assert_close(&noise, &expected.noise, 0.0);
        assert_eq!(
            noise
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected.noise_bits
        );
        let analytical_gamma = if fixture.options.s_tmin <= expected.sigma
            && expected.sigma <= fixture.options.s_tmax
        {
            (fixture.options.s_churn / fixture.steps.len() as f32).min(2.0_f32.sqrt() - 1.0)
        } else {
            0.0
        };
        assert_eq!(analytical_gamma.to_bits(), expected.gamma.to_bits());
        assert_eq!(
            (expected.sigma * (analytical_gamma + 1.0)).to_bits(),
            expected.sigma_hat.to_bits()
        );
        let analytical_churned = if expected.gamma > 0.0 {
            let scale = (expected.sigma_hat.powi(2) - expected.sigma.powi(2)).sqrt();
            current
                .iter()
                .zip(&noise)
                .map(|(current, noise)| noise.mul_add(scale, *current))
                .collect::<Vec<_>>()
        } else {
            current
        };
        assert_close(&analytical_churned, &expected.churned, fixture.tolerance);
        let primary = derivative(
            &expected.churned,
            &expected.primary_denoised,
            expected.sigma_hat,
        );
        assert_close(&primary, &expected.primary_derivative, fixture.tolerance);
        let delta = expected.next_sigma - expected.sigma_hat;
        match expected.branch.as_str() {
            "heunpp" => {
                let correction_input = advance(&expected.churned, &primary, delta);
                assert_close(
                    &correction_input,
                    expected
                        .correction_input
                        .as_deref()
                        .ok_or("missing correction input")?,
                    fixture.tolerance,
                );
                let correction = derivative(
                    &correction_input,
                    expected
                        .correction_denoised
                        .as_deref()
                        .ok_or("missing correction")?,
                    expected.next_sigma,
                );
                assert_close(
                    &correction,
                    expected
                        .correction_derivative
                        .as_deref()
                        .ok_or("missing correction derivative")?,
                    fixture.tolerance,
                );
                let lookahead_sigma = expected.lookahead_sigma.ok_or("missing lookahead sigma")?;
                let lookahead_input = advance(
                    &correction_input,
                    &correction,
                    lookahead_sigma - expected.next_sigma,
                );
                assert_close(
                    &lookahead_input,
                    expected
                        .lookahead_input
                        .as_deref()
                        .ok_or("missing lookahead input")?,
                    fixture.tolerance,
                );
                let lookahead = derivative(
                    &lookahead_input,
                    expected
                        .lookahead_denoised
                        .as_deref()
                        .ok_or("missing lookahead")?,
                    lookahead_sigma,
                );
                assert_close(
                    &lookahead,
                    expected
                        .lookahead_derivative
                        .as_deref()
                        .ok_or("missing lookahead derivative")?,
                    fixture.tolerance,
                );
                let weights = expected
                    .weights
                    .as_deref()
                    .ok_or("missing Heun++ weights")?;
                let combined = weighted(&[&primary, &correction, &lookahead], weights);
                assert_close(
                    &combined,
                    expected
                        .weighted_derivative
                        .as_deref()
                        .ok_or("missing weighted derivative")?,
                    fixture.tolerance,
                );
                assert_close(
                    &advance(&expected.churned, &combined, delta),
                    &expected.next_latent,
                    fixture.tolerance,
                );
            }
            "heun" => {
                let correction_input = advance(&expected.churned, &primary, delta);
                assert_close(
                    &correction_input,
                    expected
                        .correction_input
                        .as_deref()
                        .ok_or("missing correction input")?,
                    fixture.tolerance,
                );
                let correction = derivative(
                    &correction_input,
                    expected
                        .correction_denoised
                        .as_deref()
                        .ok_or("missing correction")?,
                    expected.next_sigma,
                );
                let weights = expected.weights.as_deref().ok_or("missing Heun weights")?;
                let combined = weighted(&[&primary, &correction], weights);
                assert_close(
                    &combined,
                    expected
                        .weighted_derivative
                        .as_deref()
                        .ok_or("missing weighted derivative")?,
                    fixture.tolerance,
                );
                assert_close(
                    &advance(&expected.churned, &combined, delta),
                    &expected.next_latent,
                    fixture.tolerance,
                );
            }
            "euler" => assert_close(
                &advance(&expected.churned, &primary, delta),
                &expected.next_latent,
                fixture.tolerance,
            ),
            branch => return Err(format!("unknown fixture branch {branch}").into()),
        }
        assert_close(
            &values(
                &backend,
                sampling
                    .latents
                    .get(step + 1)
                    .ok_or("missing next latent")?,
                &context,
            )?,
            &expected.next_latent,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                sampling
                    .denoiser_evaluations
                    .get(step)
                    .ok_or("missing primary evaluation")?,
                &context,
            )?,
            &expected.primary_denoised,
            fixture.tolerance,
        );
    }
    assert_eq!(noise_after, oracle.commit());
    assert_ne!(noise_before, noise_after);
    assert_close(
        &values(
            &backend,
            sampling.latents.last().ok_or("missing terminal")?,
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
    let defaults = HeunPp2Options::source_defaults();
    let stages = RefCell::new(Vec::new());
    let (terminal, (before, after)) = sample_heunpp2(
        &backend,
        plan(HEUNPP2_SAMPLER_ID, 7, 1)?,
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
    assert_eq!(stages.into_inner(), [HeunPp2DenoiserStage::Primary]);
    assert_ne!(before, after);
    assert_close(
        &values(
            &backend,
            terminal.latents.get(1).ok_or("missing terminal latent")?,
            &context,
        )?,
        &[0.25, 0.5],
        0.0,
    );

    let negative_churn = HeunPp2Options::new(-0.5, 0.0, f32::INFINITY, 1.0)?;
    let sigma_hats = RefCell::new(Vec::new());
    sample_heunpp2(
        &backend,
        plan(HEUNPP2_SAMPLER_ID, 8, 1)?,
        &profile()?,
        initial.clone(),
        &[1.0, 0.0],
        negative_churn,
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        &context,
        |value, sigma, _, _| {
            sigma_hats.borrow_mut().push(sigma);
            Ok(value.clone())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(sigma_hats.into_inner(), [0.5]);

    let invalid_sigma_hat = HeunPp2Options::new(-2.0, 0.0, f32::INFINITY, 1.0)?;
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 8, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            invalid_sigma_hat,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunPp2SamplerError::InvalidSigmaHat {
            step: 0,
            sigma_hat
        }) if sigma_hat == -1.0
    ));

    assert!(matches!(
        sample_heunpp2(
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
        Err(HeunPp2SamplerError::WrongSampler(value)) if value == "ddpm"
    ));
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 1.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunPp2SamplerError::Sampling(
            SamplingError::InvalidSigma { .. }
        ))
    ));

    let run_fixture = |retry, retry_policy| {
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, fixture.seed, fixture.steps.len())
                .map_err(|error| error.to_string())?,
            &profile().map_err(|error| error.to_string())?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)
                .map_err(|error| error.to_string())?,
            &fixture.sigmas,
            options(&fixture).map_err(|error| error.to_string())?,
            noise_request(&fixture, retry, retry_policy),
            &context,
            |_, _, step, stage| {
                let expected = fixture.steps.get(step).ok_or("missing fixture step")?;
                let output = match stage {
                    HeunPp2DenoiserStage::Primary => expected.primary_denoised.as_slice(),
                    HeunPp2DenoiserStage::Correction => expected
                        .correction_denoised
                        .as_deref()
                        .ok_or("missing correction")?,
                    HeunPp2DenoiserStage::Lookahead => expected
                        .lookahead_denoised
                        .as_deref()
                        .ok_or("missing lookahead")?,
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
    assert_ne!(replay_a_checkpoints.0, advance_checkpoints.0);
    assert_ne!(replay_a_checkpoints.1, advance_checkpoints.1);

    let pre_cancelled = CancellationToken::default();
    pre_cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &pre_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &cancelled_context,
            |value, _, _, _| {
                events.borrow_mut().push("denoiser");
                Ok(value.clone())
            },
            |_, _, _| {
                events.borrow_mut().push("callback");
                Ok::<(), String>(())
            }
        ),
        Err(HeunPp2SamplerError::Tensor(TensorError::Cancelled))
    ));
    assert!(events.borrow().is_empty());

    let callback_cancelled = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &callback_context,
            |value, _, _, stage| {
                events.borrow_mut().push(match stage {
                    HeunPp2DenoiserStage::Primary => "primary",
                    HeunPp2DenoiserStage::Correction => "correction",
                    HeunPp2DenoiserStage::Lookahead => "lookahead",
                });
                Ok(value.clone())
            },
            |_, _, _| {
                events.borrow_mut().push("callback");
                callback_cancelled.cancel();
                Ok::<(), String>(())
            }
        ),
        Err(HeunPp2SamplerError::EulerFoundation(
            NativeDiffusionSamplerError::Sampling(SamplingError::Cancelled)
        ))
    ));
    assert_eq!(events.into_inner(), ["primary", "callback"]);

    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, step, stage| if step == 0
                && stage == HeunPp2DenoiserStage::Correction {
                Err("correction fault".to_owned())
            } else {
                Ok(value.clone())
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunPp2SamplerError::Denoiser {
            step: 0,
            stage: HeunPp2DenoiserStage::Correction,
            reason
        }) if reason == "correction fault"
    ));

    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 3)?,
            &profile()?,
            initial.clone(),
            &[3.0, 2.0, 1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, step, stage| if step == 0 && stage == HeunPp2DenoiserStage::Lookahead {
                Err("lookahead fault".to_owned())
            } else {
                Ok(value.clone())
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(HeunPp2SamplerError::Denoiser {
            step: 0,
            stage: HeunPp2DenoiserStage::Lookahead,
            reason
        }) if reason == "lookahead fault"
    ));
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            defaults,
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |value, _, _, _| Ok(value.clone()),
            |_, _, _| Err("callback fault")
        ),
        Err(HeunPp2SamplerError::EulerFoundation(
            NativeDiffusionSamplerError::Sampling(SamplingError::Callback(reason))
        )) if reason == "callback fault"
    ));
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 1)?,
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
        Err(HeunPp2SamplerError::DenoiserContract {
            step: 0,
            stage: HeunPp2DenoiserStage::Primary
        })
    ));
    assert!(matches!(
        sample_heunpp2(
            &backend,
            plan(HEUNPP2_SAMPLER_ID, 1, 1)?,
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
        Err(HeunPp2SamplerError::NonFinite {
            step: 0,
            stage: "denoiser",
            element: 0
        })
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn row_is_only_an_equation_and_compatibility_adapter() {
    for forbidden in [
        "pub struct HeunPp2Trace",
        "struct HeunPp2Trace",
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
        "backend.workspace_vec",
    ] {
        assert!(IMPLEMENTATION.contains(canonical_adapter));
    }
    assert_eq!(IMPLEMENTATION.matches(".draw_normal(").count(), 1);
    assert!(!IMPLEMENTATION.contains("CompatibilityRngTransaction::open"));
    assert!(!IMPLEMENTATION.contains("RngCompatibilityRequest::new"));
    let draw = IMPLEMENTATION.find("let churned = draw_and_apply_churn(");
    let primary = IMPLEMENTATION.find("let primary_denoised = evaluate_denoiser(");
    let callback = IMPLEMENTATION.find("let observed = observe_euler_denoised(");
    let correction = IMPLEMENTATION.find("let correction_denoised = evaluate_denoiser(");
    let lookahead = IMPLEMENTATION.find("let lookahead_denoised = evaluate_denoiser(");
    assert!(matches!(
        (draw, primary, callback, correction, lookahead),
        (Some(draw), Some(primary), Some(callback), Some(correction), Some(lookahead))
            if draw < primary && primary < callback && callback < correction && correction < lookahead
    ));
    assert!(IMPLEMENTATION.contains("let normal = transaction.draw_normal("));
}
