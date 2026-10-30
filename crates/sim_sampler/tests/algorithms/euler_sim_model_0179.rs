use sim_sampler::{
    CompatibilityNoiseRequest, EULER_SAMPLER_ID, SamplerIdentity, SamplerRegistry, SamplingError,
    SamplingPlan, SamplingProfileIdentity,
    generated_euler_sim_model_0179::{
        DEFINITION, EULER_FEATURE_ID, EULER_SOURCE_ORDINAL, sample_euler_sim_model_0179,
    },
    generated_native_diffusion::{
        EULER_CHURN_NOISE_CONTRACT_ID, EulerOptions, NativeDiffusionSamplerError,
        validate_euler_noise_generation_device,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngCompatibilityOperation, RngCompatibilityPhase, RngGenerationPlacement,
    RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32}, rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/euler_comfy_model_0179/trajectory.json"
));
const ADAPTER_IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/euler_sim_model_0179.rs");
const EQUATION_IMPLEMENTATION: &str = include_str!("../../src/algorithms/native_diffusion.rs");

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
    denoised: Vec<f32>,
    derivative: Vec<f32>,
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

fn pinned_ksampler_names(source: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let (_, after_marker) = source
        .split_once("KSAMPLER_NAMES = [")
        .ok_or("KSAMPLER_NAMES literal is unavailable")?;
    let (literal, _) = after_marker
        .split_once(']')
        .ok_or("KSAMPLER_NAMES literal is unterminated")?;
    let names = literal
        .split('"')
        .enumerate()
        .filter_map(|(index, value)| (!index.is_multiple_of(2)).then(|| value.to_owned()))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err("KSAMPLER_NAMES literal contains no identities".into());
    }
    Ok(names)
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
    Ok(SamplingProfileIdentity::new("analytical-euler-row-v1")?)
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

fn options(fixture: &Fixture) -> Result<EulerOptions, NativeDiffusionSamplerError> {
    EulerOptions::new(
        fixture.options.s_churn,
        fixture.options.s_tmin,
        fixture.options.s_tmax,
        fixture.options.s_noise,
    )
}

fn noise_request(fixture: &Fixture, retry_policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        &fixture.rng.workflow,
        &fixture.rng.attempt,
        &fixture.rng.node,
        fixture.rng.output,
        fixture.rng.execution_ordinal,
        fixture.rng.batch,
        fixture.rng.retry,
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

#[test]
fn definition_ordinal_provenance_and_single_owner_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, EULER_SAMPLER_ID);
    assert_eq!(fixture.feature_id, EULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, EULER_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(!DEFINITION.stochastic);
    assert_eq!(DEFINITION.implementation_module, "algorithms/native_diffusion");

    let registry = SamplerRegistry::foundational()?;
    assert_eq!(registry.default_definition(), &DEFINITION);
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(EULER_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert_eq!(
        registry
            .definitions()
            .iter()
            .filter(|definition| definition.identity == EULER_SAMPLER_ID)
            .count(),
        1
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
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_euler",
        "s_churn / (len(sigmas) - 1)",
        "torch.randn_like(x) * s_noise",
        "x = x + eps",
        "d = to_d(x, sigma_hat, denoised)",
        "callback({'x': x",
        "dt = sigmas[i + 1] - sigma_hat",
        "x = x + d * dt",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert_eq!(
        pinned_ksampler_names(&samplers)?
            .iter()
            .position(|identity| identity == EULER_SAMPLER_ID),
        Some(usize::from(EULER_SOURCE_ORDINAL))
    );
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("KSAMPLER_NAMES") && line.contains("\"euler\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.starts_with("sampler,euler,")
                && line.ends_with(",COMFY-MODEL-0179"))
    );

    assert!(ADAPTER_IMPLEMENTATION.contains("sample_euler_with_options"));
    for forbidden in ["windows(2)", "draw_normal", "observe_step", "mul_add"] {
        assert!(!ADAPTER_IMPLEMENTATION.contains(forbidden));
    }
    assert_eq!(
        EQUATION_IMPLEMENTATION
            .matches("fn sample_euler_canonical")
            .count(),
        1
    );
    assert!(EQUATION_IMPLEMENTATION.contains("apply_euler_churn"));
    assert!(EQUATION_IMPLEMENTATION.contains("session.observe_step"));
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_euler_intermediate_rng_and_callback() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.rng_contract_id, EULER_CHURN_NOISE_CONTRACT_ID);
    let contract = rng_compatibility_contract(EULER_CHURN_NOISE_CONTRACT_ID)
        .ok_or("Euler churn RNG contract is unavailable")?;
    assert_eq!(contract.operation(), RngCompatibilityOperation::NormalLike);
    assert_eq!(contract.phase(), RngCompatibilityPhase::SamplingNoiseAndSolver);
    assert_eq!(contract.symbol(), "torch.randn_like");

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let events = RefCell::new(Vec::new());
    let (trace, checkpoints) = sample_euler_sim_model_0179(
        &backend,
        plan(EULER_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        options(&fixture)?,
        noise_request(&fixture, RetryRngPolicy::Replay),
        &context,
        |input, sigma_hat, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            events.borrow_mut().push(format!("denoiser-{step}"));
            assert_eq!(sigma_hat.to_bits(), expected.sigma_hat.to_bits());
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                &expected.churned,
                fixture.tolerance,
            );
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected callback step {step}"))?;
            events.borrow_mut().push(format!("callback-{step}"));
            assert_eq!(progress.sigma.to_bits(), expected.sigma.to_bits());
            assert_eq!(progress.sigma_hat.to_bits(), expected.sigma_hat.to_bits());
            assert_eq!(progress.next_sigma.to_bits(), expected.next_sigma.to_bits());
            assert_close(
                &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                &expected.churned,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.denoised,
                fixture.tolerance,
            );
            Ok::<(), String>(())
        },
    )?;

    assert_eq!(
        events.into_inner(),
        (0..fixture.steps.len())
            .flat_map(|step| [format!("denoiser-{step}"), format!("callback-{step}")])
            .collect::<Vec<_>>()
    );
    let (before, after) = checkpoints.ok_or("churn RNG checkpoints are unavailable")?;
    assert_ne!(before, after);
    assert_eq!(before.device, DeviceId::CPU);
    let actual_latents = trace
        .latents
        .iter()
        .map(|tensor| values(&backend, tensor, &context))
        .collect::<Result<Vec<_>, _>>()?;
    assert_close(
        actual_latents.first().ok_or("initial latent is unavailable")?,
        &fixture.initial,
        fixture.tolerance,
    );
    for (actual, expected) in actual_latents.iter().skip(1).zip(&fixture.steps) {
        assert_close(actual, &expected.next_latent, fixture.tolerance);
        if let (Some(noise), Some(noise_bits)) = (&expected.noise, &expected.noise_bits) {
            assert_eq!(
                noise.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                *noise_bits
            );
        }
        for (actual, expected) in expected.derivative.iter().zip(
            expected
                .churned
                .iter()
                .zip(&expected.denoised)
                .map(|(latent, denoised)| (latent - denoised) / expected.sigma_hat),
        ) {
            assert!((*actual - expected).abs() <= fixture.tolerance);
        }
        let expected_gamma = if fixture.options.s_churn > 0.0
            && fixture.options.s_tmin <= expected.sigma
            && expected.sigma <= fixture.options.s_tmax
        {
            (fixture.options.s_churn / fixture.steps.len() as f32)
                .min(2.0_f32.sqrt() - 1.0)
        } else {
            0.0
        };
        assert_eq!(expected.gamma.to_bits(), expected_gamma.to_bits());
    }
    assert_close(
        actual_latents.last().ok_or("terminal latent is unavailable")?,
        &fixture.terminal,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn failures_are_typed_and_source_defaults_do_not_consume_rng() -> Result<(), Box<dyn Error>> {
    let cuda = DeviceId::from_source_device("cuda:0")?;
    assert!(matches!(
        validate_euler_noise_generation_device(cuda),
        Err(NativeDiffusionSamplerError::DeviceUnavailable { device, .. }) if device == cuda
    ));
    for (name, options) in [
        ("s_churn", EulerOptions::new(f32::NAN, 0.0, f32::INFINITY, 1.0)),
        ("s_tmin", EulerOptions::new(0.0, f32::INFINITY, f32::INFINITY, 1.0)),
        ("s_tmax", EulerOptions::new(0.0, 0.0, f32::NEG_INFINITY, 1.0)),
        ("s_noise", EulerOptions::new(0.0, 0.0, f32::INFINITY, f32::NAN)),
    ] {
        assert!(matches!(
            options,
            Err(NativeDiffusionSamplerError::InvalidEulerOption { name: actual, .. })
                if actual == name
        ));
    }

    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    assert!(matches!(
        sample_euler_sim_model_0179(
            &backend,
            plan("ddim", fixture.seed, fixture.steps.len())?,
            &profile()?,
            initial.clone(),
            &fixture.sigmas,
            options(&fixture)?,
            noise_request(&fixture, RetryRngPolicy::Replay),
            &context,
            |input, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(NativeDiffusionSamplerError::WrongEulerSampler(identity)) if identity == "ddim"
    ));
    assert!(matches!(
        sample_euler_sim_model_0179(
            &backend,
            plan(EULER_SAMPLER_ID, fixture.seed, 1)?,
            &profile()?,
            initial.clone(),
            &[f32::NAN, 0.0],
            EulerOptions::default(),
            noise_request(&fixture, RetryRngPolicy::Replay),
            &context,
            |input, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(NativeDiffusionSamplerError::InvalidSigma { .. })
    ));
    let (_, checkpoints) = sample_euler_sim_model_0179(
        &backend,
        plan(EULER_SAMPLER_ID, fixture.seed, 1)?,
        &profile()?,
        initial,
        &[1.0, 0.0],
        EulerOptions::default(),
        noise_request(&fixture, RetryRngPolicy::Replay),
        &context,
        |input, _, _| Ok(input.clone()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert!(checkpoints.is_none());
    Ok(())
}

#[test]
fn cancellation_and_replay_are_transactional() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let initial = tensor_from_f32(
        &backend,
        &fixture.shape,
        &fixture.initial,
        &execution_context(&backend, &authority, &CancellationToken::default())?,
    )?;
    assert!(matches!(
        sample_euler_sim_model_0179(
            &backend,
            plan(EULER_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
            &profile()?,
            initial,
            &fixture.sigmas,
            options(&fixture)?,
            noise_request(&fixture, RetryRngPolicy::Replay),
            &cancelled_context,
            |input, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(NativeDiffusionSamplerError::Tensor(TensorError::Cancelled))
    ));

    let run = |cancellation: &CancellationToken| {
        let context = execution_context(&backend, &authority, cancellation)?;
        let (trace, checkpoints) = sample_euler_sim_model_0179(
            &backend,
            plan(EULER_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
            &profile()?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            options(&fixture)?,
            noise_request(&fixture, RetryRngPolicy::Replay),
            &context,
            |_, _, step| {
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps.get(step).ok_or("missing step")?.denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )?;
        let terminal = values(
            &backend,
            trace.latents.last().ok_or("terminal latent is unavailable")?,
            &context,
        )?;
        Ok::<_, Box<dyn Error>>((terminal, checkpoints))
    };
    let first_cancellation = CancellationToken::default();
    let second_cancellation = CancellationToken::default();
    let first = run(&first_cancellation)?;
    let second = run(&second_cancellation)?;
    assert_eq!(first, second);

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    assert!(matches!(
        sample_euler_sim_model_0179(
            &backend,
            plan(EULER_SAMPLER_ID, fixture.seed, fixture.steps.len())?,
            &profile()?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &callback_context)?,
            &fixture.sigmas,
            options(&fixture)?,
            noise_request(&fixture, RetryRngPolicy::Replay),
            &callback_context,
            |_, _, step| tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps.get(step).ok_or("missing step")?.denoised,
                &callback_context,
            )
            .map_err(|error| error.to_string()),
            |_, _, _| {
                callback_cancellation.cancel();
                Ok::<(), String>(())
            },
        ),
        Err(NativeDiffusionSamplerError::Sampling(SamplingError::Cancelled))
    ));
    Ok(())
}

#[test]
fn canonical_rng_contract_parameters_are_fixed() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let request = noise_request(&fixture, RetryRngPolicy::Replay);
    let cancellation = CancellationToken::default();
    let transaction = request.open_transaction(
        EULER_CHURN_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::TorchSigned64,
        RngGenerationPlacement::Native(DeviceId::CPU),
        None,
        &cancellation,
    )?;
    assert_eq!(transaction.generation_device(), DeviceId::CPU);
    assert_eq!(transaction.output_device(), DeviceId::CPU);
    Ok(())
}
