use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfileIdentity,
    generated_res_multistep_sim_model_0193::{
        DEFINITION, RES_MULTISTEP_FEATURE_ID, RES_MULTISTEP_NOISE_CONTRACT_ID,
        RES_MULTISTEP_SAMPLER_ID, RES_MULTISTEP_SOURCE_ORDINAL, ResMultistepOptions,
        ResMultistepSamplerError, sample_res_multistep,
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
    "/../sim_test_support/fixtures/samplers/res_multistep_comfy_model_0193/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/res_multistep_sim_model_0193.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    rng_contract_id: String,
    seed: u64,
    noise_scale: f32,
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
    current: Vec<f32>,
    denoised: Vec<f32>,
    derivative: Option<Vec<f32>>,
    h: Option<f32>,
    c2: Option<f32>,
    phi1: Option<f32>,
    phi2: Option<f32>,
    b1: Option<f32>,
    b2: Option<f32>,
    denoised_mix: Option<Vec<f32>>,
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("analytical-res-multistep-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.1_f32, 0.5, 1.0, 2.0, 4.0]),
    )?)
}

fn plan(identity: &str, seed: u64, steps: usize) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new("analytical-res-multistep-row-v1")?,
        seed,
        u32::try_from(steps)?,
        1.0,
        1.0,
    )?)
}

fn request(fixture: &Fixture, retry: u32, policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        &fixture.rng.workflow,
        &fixture.rng.attempt,
        &fixture.rng.node,
        fixture.rng.output,
        fixture.rng.execution_ordinal,
        fixture.rng.batch,
        retry,
        policy,
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
fn definition_registry_and_pinned_source_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, RES_MULTISTEP_SAMPLER_ID);
    assert_eq!(fixture.feature_id, RES_MULTISTEP_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, RES_MULTISTEP_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, RES_MULTISTEP_NOISE_CONTRACT_ID);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.source_ordinal, 30);
    assert!(DEFINITION.stochastic);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(RES_MULTISTEP_SAMPLER_ID)?)?,
        &DEFINITION
    );
    let root = workspace_root()?;
    assert_eq!(digest(&root.join(&fixture.source.sampling_path))?, fixture.source.sampling_sha256);
    assert_eq!(digest(&root.join(&fixture.source.samplers_path))?, fixture.source.samplers_sha256);
    assert_eq!(digest(&root.join(&fixture.source.catalog_path))?, fixture.source.catalog_sha256);
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def res_multistep",
        "phi1_fn = lambda t: torch.expm1(t) / t",
        "c2 = (t_prev - t_old) / h",
        "b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)",
        "x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)",
        "noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up",
        "eta=0., cfg_pp=False",
    ] {
        assert!(equations.contains(fragment), "missing source equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(samplers.contains("KSAMPLER_NAMES = ["));
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("\"res_multistep\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(catalog.lines().nth(fixture.source.catalog_line - 1).is_some_and(|line| {
        line.starts_with("sampler,res_multistep,") && line.ends_with(",COMFY-MODEL-0193")
    }));
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_intermediate_callback_and_rng_draw() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let (trace, (before, after)) = sample_res_multistep(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay),
        ResMultistepOptions::new(fixture.noise_scale)?,
        &context,
        |input, sigma, step| {
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            assert_eq!(sigma.to_bits(), fixture.sigmas[step].to_bits());
            assert_close(&values(&backend, input, &context).map_err(|error| error.to_string())?, &expected.current, fixture.tolerance);
            events.borrow_mut().push(format!("denoiser-{step}"));
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, current, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture.steps.get(step).ok_or("unexpected callback step")?;
            assert_close(&values(&backend, current, &context).map_err(|error| error.to_string())?, &expected.current, fixture.tolerance);
            assert_close(&values(&backend, denoised, &context).map_err(|error| error.to_string())?, &expected.denoised, fixture.tolerance);
            events.borrow_mut().push(format!("callback-{step}"));
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(events.into_inner(), ["denoiser-0", "callback-0", "denoiser-1", "callback-1", "denoiser-2", "callback-2", "denoiser-3", "callback-3"]);
    assert_close(&values(&backend, &alias, &context)?, &fixture.initial, 0.0);
    for (index, expected) in fixture.steps.iter().enumerate() {
        assert_close(&values(&backend, trace.latents.get(index).ok_or("missing latent")?, &context)?, &expected.current, fixture.tolerance);
        assert_close(&values(&backend, trace.latents.get(index + 1).ok_or("missing next latent")?, &context)?, &expected.next, fixture.tolerance);
        assert!(matches!(expected.branch.as_str(), "euler" | "multistep"));
    }
    assert_close(&values(&backend, trace.latents.last().ok_or("missing terminal")?, &context)?, &fixture.terminal, fixture.tolerance);
    let mut oracle = request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay).open_transaction(
        RES_MULTISTEP_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer { output_device: DeviceId::CPU },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    for _ in 0..fixture.steps.len() - 1 {
        oracle.draw_normal(fixture.initial.len(), &cancellation)?;
    }
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn analytical_fixture_reconstructs_euler_and_multistep_equations() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    for (step, expected) in fixture.steps.iter().enumerate() {
        let sigma = fixture.sigmas[step];
        let next_sigma = fixture.sigmas[step + 1];
        if expected.branch == "euler" {
            let derivative = expected.current.iter().zip(&expected.denoised).map(|(current, denoised)| (current - denoised) / sigma).collect::<Vec<_>>();
            assert_close(&derivative, expected.derivative.as_deref().ok_or("missing derivative")?, fixture.tolerance);
            let next = expected.current.iter().zip(&derivative).map(|(current, derivative)| current + derivative * (next_sigma - sigma)).collect::<Vec<_>>();
            assert_close(&next, &expected.next, fixture.tolerance);
        } else {
            let h = expected.h.ok_or("missing h")?;
            let c2 = expected.c2.ok_or("missing c2")?;
            let phi1 = (-h).exp_m1() / -h;
            let phi2 = (phi1 - 1.0) / -h;
            assert!((phi1 - expected.phi1.ok_or("missing phi1")?).abs() <= fixture.tolerance);
            assert!((phi2 - expected.phi2.ok_or("missing phi2")?).abs() <= fixture.tolerance);
            let b1 = phi1 - phi2 / c2;
            let b2 = phi2 / c2;
            assert!((b1 - expected.b1.ok_or("missing b1")?).abs() <= fixture.tolerance);
            assert!((b2 - expected.b2.ok_or("missing b2")?).abs() <= fixture.tolerance);
            let previous = &fixture.steps[step - 1].denoised;
            let mix = expected.denoised.iter().zip(previous).map(|(denoised, previous)| b1 * denoised + b2 * previous).collect::<Vec<_>>();
            assert_close(&mix, expected.denoised_mix.as_deref().ok_or("missing mix")?, fixture.tolerance);
            let next = expected.current.iter().zip(&mix).map(|(current, mix)| (-h).exp() * current + h * mix).collect::<Vec<_>>();
            assert_close(&next, &expected.next, fixture.tolerance);
        }
    }
    Ok(())
}

#[test]
fn failures_cancellation_retry_and_ownership_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert!(ResMultistepOptions::new(f32::NAN).is_err());
    for forbidden in ["struct SamplingSession", "struct CancellationToken", "struct CpuWorkspaceAuthority", "fn standard_ancestral_step", "std::fs", "sqlx", "rusqlite"] {
        assert!(!IMPLEMENTATION.contains(forbidden), "row owns forbidden service {forbidden}");
    }
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let error = sample_res_multistep(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepOptions::default(),
        &context,
        |_input, _sigma, step| Err(format!("failure-{step}")),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(error, Err(ResMultistepSamplerError::Denoiser { step: 0, .. })));
    let error = sample_res_multistep(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepOptions::default(),
        &context,
        |_input, _sigma, _step| tensor_from_f32(&backend, &[1, 2], &[0.0, 0.0], &context).map_err(|error| error.to_string()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(error, Err(ResMultistepSamplerError::DenoiserContract { step: 0 })));
    let error = sample_res_multistep(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepOptions::default(),
        &context,
        |_input, _sigma, step| tensor_from_f32(&backend, &fixture.shape, &fixture.steps[step].denoised, &context).map_err(|error| error.to_string()),
        |_progress, _current, _denoised| Err::<(), _>("callback-failure"),
    );
    assert!(matches!(error, Err(ResMultistepSamplerError::Sampling(_))));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let error = sample_res_multistep(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, 1, RetryRngPolicy::Advance),
        ResMultistepOptions::default(),
        &cancelled_context,
        |_input, _sigma, _step| Err("must not execute".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(error, Err(ResMultistepSamplerError::Tensor(TensorError::Cancelled))));
    Ok(())
}
