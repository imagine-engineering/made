use sim_sampler::{
    CfgPpDenoiserOutput, CompatibilityNoiseRequest, DiscreteSamplingProfile,
    PredictionInterpretation, SamplerIdentity, SamplerRegistry, SamplingPlan,
    SamplingProfileIdentity,
    generated_res_multistep_ancestral_sim_model_0194::{
        DEFINITION, RES_MULTISTEP_ANCESTRAL_FEATURE_ID, RES_MULTISTEP_ANCESTRAL_NOISE_CONTRACT_ID,
        RES_MULTISTEP_ANCESTRAL_SAMPLER_ID, RES_MULTISTEP_ANCESTRAL_SOURCE_ORDINAL,
        ResMultistepAncestralOptions, sample_res_multistep_ancestral,
    },
    generated_res_multistep_sim_model_0193::{
        ResMultistepFamilyOptions, ResMultistepSamplerError, sample_res_multistep_family,
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
    "/../sim_test_support/fixtures/samplers/res_multistep_ancestral_comfy_model_0194/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/res_multistep_ancestral_sim_model_0194.rs");
const FAMILY_OWNER: &str = include_str!("../../src/algorithms/res_multistep_sim_model_0193.rs");

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
    family_lines: [usize; 2],
    wrapper_lines: [usize; 2],
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
    sigma_down: f32,
    sigma_up: f32,
    h: Option<f32>,
    c2: Option<f32>,
    phi1: Option<f32>,
    phi2: Option<f32>,
    b1: Option<f32>,
    b2: Option<f32>,
    deterministic: Vec<f32>,
    noise: Option<Vec<f32>>,
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
    let [start, end] = range;
    source
        .lines()
        .skip(start.saturating_sub(1))
        .take(end.saturating_sub(start) + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

fn pinned_ksampler_names(source: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let (_, after_marker) = source
        .split_once("KSAMPLER_NAMES = [")
        .ok_or("KSAMPLER_NAMES literal is unavailable")?;
    let (literal, _) = after_marker
        .split_once(']')
        .ok_or("KSAMPLER_NAMES literal is unterminated")?;
    Ok(literal
        .split('"')
        .enumerate()
        .filter_map(|(index, value)| (!index.is_multiple_of(2)).then(|| value.to_owned()))
        .collect())
}

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("analytical-res-multistep-ancestral-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.1_f32, 0.5, 1.0, 2.0, 4.0]),
    )?)
}

fn plan(identity: &str, seed: u64, steps: usize) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new("analytical-res-multistep-ancestral-row-v1")?,
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

fn run_fixture(
    fixture: &Fixture,
    retry: u32,
    policy: RetryRngPolicy,
) -> Result<
    (
        Vec<Vec<f32>>,
        sim_tensor::RngCheckpoint,
        sim_tensor::RngCheckpoint,
    ),
    Box<dyn Error>,
> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (trace, (before, after)) = sample_res_multistep_ancestral(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(fixture, retry, policy),
        ResMultistepAncestralOptions::new(fixture.eta, fixture.noise_scale)?,
        &context,
        |_input, _sigma, step| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    let mut latents = Vec::new();
    for tensor in &trace.latents {
        latents.push(values(&backend, tensor, &context)?);
    }
    Ok((latents, before, after))
}

#[test]
fn definition_provenance_and_family_ownership_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, RES_MULTISTEP_ANCESTRAL_SAMPLER_ID);
    assert_eq!(fixture.feature_id, RES_MULTISTEP_ANCESTRAL_FEATURE_ID);
    assert_eq!(
        fixture.source_ordinal,
        RES_MULTISTEP_ANCESTRAL_SOURCE_ORDINAL
    );
    assert_eq!(
        fixture.rng_contract_id,
        RES_MULTISTEP_ANCESTRAL_NOISE_CONTRACT_ID
    );
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/res_multistep_ancestral_comfy_model_0194"
    );
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(RES_MULTISTEP_ANCESTRAL_SAMPLER_ID)?)?,
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
    let family = source_range(&sampling, fixture.source.family_lines);
    for fragment in [
        "def res_multistep",
        "get_ancestral_step",
        "if sigma_down == 0 or old_denoised is None",
        "phi1_val, phi2_val",
        "noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up",
    ] {
        assert!(
            family.contains(fragment),
            "missing family source {fragment}"
        );
    }
    let wrapper = source_range(&sampling, fixture.source.wrapper_lines);
    for fragment in [
        "def sample_res_multistep_ancestral",
        "eta=1.",
        "s_noise=1.",
        "eta=eta, cfg_pp=False",
    ] {
        assert!(
            wrapper.contains(fragment),
            "missing wrapper source {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert_eq!(
        pinned_ksampler_names(&samplers)?
            .iter()
            .position(|identity| identity == RES_MULTISTEP_ANCESTRAL_SAMPLER_ID),
        Some(usize::from(RES_MULTISTEP_ANCESTRAL_SOURCE_ORDINAL))
    );
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("\"res_multistep_ancestral\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(|line| {
                line.starts_with("sampler,res_multistep_ancestral,")
                    && line.ends_with(",COMFY-MODEL-0194")
            })
    );

    for required in [
        "sample_res_multistep_family(",
        "ResMultistepFamilyOptions::new(self.eta, self.noise_scale, false)",
        "unconditional_denoised: denoised.clone()",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing adapter mapping {required}"
        );
    }
    for forbidden in [
        "SamplingSession",
        "CompatibilityRngTransaction",
        "standard_ancestral_step",
        "draw_normal",
        "tensor_to_f32",
        "fn multistep",
        "fn euler_step",
        "fn phi",
        ".commit(",
        "std::fs",
        "sqlx",
        "rusqlite",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "adapter owns forbidden behavior {forbidden}"
        );
    }
    for owned in [
        "SamplingSession::new",
        "standard_ancestral_step",
        "noise_request.open_transaction",
        "fn multistep",
        "fn euler_step",
        "observed.commit",
    ] {
        assert!(
            FAMILY_OWNER.contains(owned),
            "family owner is missing {owned}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_branch_noise_draw_and_callback() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let (trace, (before, after)) = sample_res_multistep_ancestral(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay),
        ResMultistepAncestralOptions::new(fixture.eta, fixture.noise_scale)?,
        &context,
        |input, sigma, step| {
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            assert_eq!(sigma.to_bits(), fixture.sigmas[step].to_bits());
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                &expected.current,
                fixture.tolerance,
            );
            events.borrow_mut().push(format!("denoiser-{step}"));
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, current, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture.steps.get(step).ok_or("unexpected callback step")?;
            assert_close(
                &values(&backend, current, &context).map_err(|error| error.to_string())?,
                &expected.current,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.denoised,
                fixture.tolerance,
            );
            events.borrow_mut().push(format!("callback-{step}"));
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(
        events.into_inner(),
        [
            "denoiser-0",
            "callback-0",
            "denoiser-1",
            "callback-1",
            "denoiser-2",
            "callback-2",
            "denoiser-3",
            "callback-3",
        ]
    );
    assert_close(&values(&backend, &alias, &context)?, &fixture.initial, 0.0);
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_close(
            &values(
                &backend,
                trace.latents.get(step).ok_or("missing current latent")?,
                &context,
            )?,
            &expected.current,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace.latents.get(step + 1).ok_or("missing next latent")?,
                &context,
            )?,
            &expected.next,
            fixture.tolerance,
        );
        assert!(matches!(expected.branch.as_str(), "euler" | "multistep"));
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

    let mut oracle = request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay)
        .open_transaction(
            RES_MULTISTEP_ANCESTRAL_NOISE_CONTRACT_ID,
            i128::from(fixture.seed),
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
            None,
            &cancellation,
        )?;
    assert_eq!(before, oracle.checkpoint());
    for expected in fixture.steps.iter().filter(|step| step.noise.is_some()) {
        let noise = oracle.draw_normal(fixture.initial.len(), &cancellation)?;
        assert_close(
            &noise
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            expected.noise.as_deref().ok_or("missing fixture noise")?,
            fixture.tolerance,
        );
    }
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn analytical_fixture_reconstructs_euler_multistep_and_ancestral_noise()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(ResMultistepAncestralOptions::default().eta(), 1.0);
    assert_eq!(ResMultistepAncestralOptions::default().noise_scale(), 1.0);
    for (step, expected) in fixture.steps.iter().enumerate() {
        let sigma = fixture.sigmas[step];
        let next_sigma = fixture.sigmas[step + 1];
        let sigma_up = if fixture.eta == 0.0 {
            0.0
        } else {
            next_sigma.min(
                fixture.eta
                    * (next_sigma.powi(2) * (sigma.powi(2) - next_sigma.powi(2)) / sigma.powi(2))
                        .sqrt(),
            )
        };
        let sigma_down = (next_sigma.powi(2) - sigma_up.powi(2)).sqrt();
        assert!((sigma_down - expected.sigma_down).abs() <= fixture.tolerance);
        assert!((sigma_up - expected.sigma_up).abs() <= fixture.tolerance);

        let deterministic = if expected.branch == "euler" {
            expected
                .current
                .iter()
                .zip(&expected.denoised)
                .map(|(current, denoised)| {
                    current + ((current - denoised) / sigma) * (sigma_down - sigma)
                })
                .collect::<Vec<_>>()
        } else {
            let h = expected.h.ok_or("missing h")?;
            let c2 = expected.c2.ok_or("missing c2")?;
            let phi1 = (-h).exp_m1() / -h;
            let phi2 = (phi1 - 1.0) / -h;
            let b1 = phi1 - phi2 / c2;
            let b2 = phi2 / c2;
            assert!((phi1 - expected.phi1.ok_or("missing phi1")?).abs() <= fixture.tolerance);
            assert!((phi2 - expected.phi2.ok_or("missing phi2")?).abs() <= fixture.tolerance);
            assert!((b1 - expected.b1.ok_or("missing b1")?).abs() <= fixture.tolerance);
            assert!((b2 - expected.b2.ok_or("missing b2")?).abs() <= fixture.tolerance);
            expected
                .current
                .iter()
                .zip(&expected.denoised)
                .zip(&fixture.steps[step - 1].denoised)
                .map(|((current, denoised), previous)| {
                    (-h).exp() * current + h * (b1 * denoised + b2 * previous)
                })
                .collect::<Vec<_>>()
        };
        assert_close(&deterministic, &expected.deterministic, fixture.tolerance);
        let next = if let Some(noise) = &expected.noise {
            deterministic
                .iter()
                .zip(noise)
                .map(|(deterministic, noise)| {
                    deterministic + noise * fixture.noise_scale * sigma_up
                })
                .collect::<Vec<_>>()
        } else {
            deterministic
        };
        assert_close(&next, &expected.next, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn guided_output_mapping_matches_the_family_owner() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (adapter_latents, adapter_before, adapter_after) =
        run_fixture(&fixture, fixture.rng.retry, RetryRngPolicy::Replay)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (trace, (family_before, family_after)) = sample_res_multistep_family(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        RES_MULTISTEP_ANCESTRAL_SAMPLER_ID,
        initial,
        &fixture.sigmas,
        request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay),
        ResMultistepFamilyOptions::new(fixture.eta, fixture.noise_scale, false)?,
        &context,
        |_input, _sigma, step| {
            let denoised = tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &context,
            )
            .map_err(|error| error.to_string())?;
            Ok(CfgPpDenoiserOutput {
                unconditional_denoised: denoised.clone(),
                denoised,
            })
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    let mut family_latents = Vec::new();
    for tensor in &trace.latents {
        family_latents.push(values(&backend, tensor, &context)?);
    }
    assert_eq!(adapter_latents, family_latents);
    assert_eq!(adapter_before, family_before);
    assert_eq!(adapter_after, family_after);
    Ok(())
}

#[test]
fn failures_cancellation_retry_and_atomicity_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert!(ResMultistepAncestralOptions::new(f32::NAN, 1.0).is_err());
    assert!(ResMultistepAncestralOptions::new(1.0, f32::INFINITY).is_err());
    let (replay, replay_before, replay_after) = run_fixture(&fixture, 0, RetryRngPolicy::Replay)?;
    let (retry_replay, retry_before, retry_after) =
        run_fixture(&fixture, 7, RetryRngPolicy::Replay)?;
    assert_eq!(replay, retry_replay);
    assert_eq!(replay_before, retry_before);
    assert_eq!(replay_after, retry_after);
    let (advanced, advanced_before, advanced_after) =
        run_fixture(&fixture, 1, RetryRngPolicy::Advance)?;
    assert_ne!(replay, advanced);
    assert_ne!(replay_before, advanced_before);
    assert_ne!(replay_after, advanced_after);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let alias = initial.clone();
    let error = sample_res_multistep_ancestral(
        &backend,
        plan("res_multistep", fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralOptions::default(),
        &context,
        |_input, _sigma, _step| Err("must not execute".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        error,
        Err(ResMultistepSamplerError::WrongSampler { .. })
    ));
    let error = sample_res_multistep_ancestral(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralOptions::default(),
        &context,
        |_input, _sigma, step| Err(format!("failure-{step}")),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        error,
        Err(ResMultistepSamplerError::Denoiser { step: 0, .. })
    ));
    let error = sample_res_multistep_ancestral(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralOptions::default(),
        &context,
        |_input, _sigma, _step| {
            tensor_from_f32(&backend, &[1, 2], &[0.0, 0.0], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        error,
        Err(ResMultistepSamplerError::DenoiserContract { step: 0 })
    ));
    let error = sample_res_multistep_ancestral(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralOptions::default(),
        &context,
        |_input, _sigma, step| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Err::<(), _>("callback-failure"),
    );
    assert!(matches!(error, Err(ResMultistepSamplerError::Sampling(_))));
    assert_close(&values(&backend, &alias, &context)?, &fixture.initial, 0.0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let error = sample_res_multistep_ancestral(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralOptions::default(),
        &cancelled_context,
        |_input, _sigma, _step| Err("must not execute".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        error,
        Err(ResMultistepSamplerError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}
