use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfileIdentity,
    generated_res_multistep_ancestral_cfg_pp_sim_model_0195::{
        DEFINITION, RES_MULTISTEP_ANCESTRAL_CFG_PP_FEATURE_ID,
        RES_MULTISTEP_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        RES_MULTISTEP_ANCESTRAL_CFG_PP_SAMPLER_ID, RES_MULTISTEP_ANCESTRAL_CFG_PP_SOURCE_ORDINAL,
        ResMultistepAncestralCfgPpDenoiserOutput, ResMultistepAncestralCfgPpOptions,
        sample_res_multistep_ancestral_cfg_pp,
    },
    generated_res_multistep_sim_model_0193::{
        RES_MULTISTEP_NOISE_CONTRACT_ID, ResMultistepSamplerError,
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
    "/../sim_test_support/fixtures/samplers/res_multistep_ancestral_cfg_pp_comfy_model_0195/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/res_multistep_ancestral_cfg_pp_sim_model_0195.rs"
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
    equation_lines: [usize; 2],
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
    sigma_down: f32,
    sigma_up: f32,
    current: Vec<f32>,
    guided: Vec<f32>,
    unconditional: Vec<f32>,
    derivative: Option<Vec<f32>>,
    h: Option<f32>,
    c2: Option<f32>,
    phi1: Option<f32>,
    phi2: Option<f32>,
    b1: Option<f32>,
    b2: Option<f32>,
    corrected: Option<Vec<f32>>,
    denoised_mix: Option<Vec<f32>>,
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("analytical-res-multistep-ancestral-cfg-pp-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.1_f32, 0.5, 1.0, 2.0, 4.0]),
    )?)
}

fn plan(identity: &str, seed: u64, steps: usize) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new("analytical-res-multistep-ancestral-cfg-pp-row-v1")?,
        seed,
        u32::try_from(steps)?,
        1.0,
        1.0,
    )?)
}

fn request(
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
fn val_sampler_001_definition_provenance_and_thin_family_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, RES_MULTISTEP_ANCESTRAL_CFG_PP_SAMPLER_ID);
    assert_eq!(
        fixture.feature_id,
        RES_MULTISTEP_ANCESTRAL_CFG_PP_FEATURE_ID
    );
    assert_eq!(
        fixture.source_ordinal,
        RES_MULTISTEP_ANCESTRAL_CFG_PP_SOURCE_ORDINAL
    );
    assert_eq!(
        fixture.rng_contract_id,
        RES_MULTISTEP_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID
    );
    assert_eq!(
        RES_MULTISTEP_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        RES_MULTISTEP_NOISE_CONTRACT_ID
    );
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 33);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(
            RES_MULTISTEP_ANCESTRAL_CFG_PP_SAMPLER_ID,
        )?)?,
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
    let equations = sampling
        .lines()
        .skip(fixture.source.equation_lines[0].saturating_sub(1))
        .take(fixture.source.equation_lines[1] - fixture.source.equation_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def res_multistep(",
        "s_noise = s_noise * getattr(",
        "sigma_down, sigma_up = get_ancestral_step(",
        "d = to_d(x, sigmas[i], uncond_denoised)",
        "x = denoised + d * sigma_down",
        "x = x + (denoised - uncond_denoised)",
        "b1 * uncond_denoised + b2 * old_denoised",
        "noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up",
        "old_denoised = uncond_denoised",
        "def sample_res_multistep_ancestral_cfg_pp(",
        "eta=eta, cfg_pp=True",
    ] {
        assert!(
            equations.contains(fragment),
            "missing source equation {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("\"res_multistep_ancestral_cfg_pp\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(|line| {
                line.starts_with("sampler,res_multistep_ancestral_cfg_pp,")
                    && line.ends_with(",COMFY-MODEL-0195")
            })
    );

    assert!(IMPLEMENTATION.contains("sample_res_multistep_family("));
    assert!(
        IMPLEMENTATION.contains("ResMultistepFamilyOptions::new(self.eta, self.noise_scale, true)")
    );
    assert!(
        IMPLEMENTATION
            .contains("pub type ResMultistepAncestralCfgPpDenoiserOutput = CfgPpDenoiserOutput;")
    );
    for forbidden in [
        "fn multistep(",
        "fn euler_step(",
        "standard_ancestral_step(",
        "validate_cfg_pp_denoiser_output(",
        "SamplingSession::new",
        "CompatibilityRngTransaction",
        "draw_normal(",
        "tensor_to_f32(",
        ".commit(",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate family owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_every_cfg_pp_callback_latent_and_rng_draw()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let immutable_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let (trace, (before, after)) = sample_res_multistep_ancestral_cfg_pp(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay),
        ResMultistepAncestralCfgPpOptions::new(fixture.eta, fixture.noise_scale)?,
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
            Ok(ResMultistepAncestralCfgPpDenoiserOutput {
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
                &expected.guided,
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
    assert_close(
        &values(&backend, &immutable_alias, &context)?,
        &fixture.initial,
        0.0,
    );
    for (index, expected) in fixture.steps.iter().enumerate() {
        assert_close(
            &values(
                &backend,
                trace.latents.get(index).ok_or("missing latent")?,
                &context,
            )?,
            &expected.current,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace.latents.get(index + 1).ok_or("missing next latent")?,
                &context,
            )?,
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

    let mut oracle = request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay)
        .open_transaction(
            RES_MULTISTEP_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
            i128::from(fixture.seed),
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
            None,
            &cancellation,
        )?;
    assert_eq!(before, oracle.checkpoint());
    for expected in fixture
        .steps
        .iter()
        .filter_map(|step| step.noise.as_deref())
    {
        let drawn = oracle
            .draw_normal(expected.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_close(&drawn, expected, fixture.tolerance);
    }
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn analytical_fixture_reconstructs_ancestral_cfg_pp_euler_multistep_and_noise()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let mut previous_unconditional: Option<&[f32]> = None;
    let mut previous_sigma_down: Option<f32> = None;
    for (step, expected) in fixture.steps.iter().enumerate() {
        let sigma = *fixture.sigmas.get(step).ok_or("missing sigma")?;
        let next_sigma = *fixture.sigmas.get(step + 1).ok_or("missing next sigma")?;
        let sigma_up = next_sigma.min(
            fixture.eta
                * (next_sigma.powi(2) * (sigma.powi(2) - next_sigma.powi(2)) / sigma.powi(2))
                    .sqrt(),
        );
        let sigma_down = (next_sigma.powi(2) - sigma_up.powi(2)).sqrt();
        assert!((sigma_down - expected.sigma_down).abs() <= fixture.tolerance);
        assert!((sigma_up - expected.sigma_up).abs() <= fixture.tolerance);

        let deterministic = if sigma_down == 0.0 || previous_unconditional.is_none() {
            assert_eq!(expected.branch, "euler_cfg_pp");
            let derivative = expected
                .current
                .iter()
                .zip(&expected.unconditional)
                .map(|(current, unconditional)| (current - unconditional) / sigma)
                .collect::<Vec<_>>();
            assert_close(
                &derivative,
                expected.derivative.as_deref().ok_or("missing derivative")?,
                fixture.tolerance,
            );
            expected
                .guided
                .iter()
                .zip(&derivative)
                .map(|(guided, derivative)| guided + derivative * sigma_down)
                .collect::<Vec<_>>()
        } else {
            assert_eq!(expected.branch, "multistep_cfg_pp");
            let previous_sigma = *fixture
                .sigmas
                .get(step.saturating_sub(1))
                .ok_or("missing previous sigma")?;
            let step_size = -sigma_down.ln() - -sigma.ln();
            let c2 = (-previous_sigma.ln()
                - -previous_sigma_down
                    .ok_or("missing previous sigma down")?
                    .ln())
                / step_size;
            let negative_step = -step_size;
            let phi1 = negative_step.exp_m1() / negative_step;
            let phi2 = (phi1 - 1.0) / negative_step;
            let b1 = phi1 - phi2 / c2;
            let b2 = phi2 / c2;
            for (actual, pinned) in [
                (step_size, expected.h.ok_or("missing h")?),
                (c2, expected.c2.ok_or("missing c2")?),
                (phi1, expected.phi1.ok_or("missing phi1")?),
                (phi2, expected.phi2.ok_or("missing phi2")?),
                (b1, expected.b1.ok_or("missing b1")?),
                (b2, expected.b2.ok_or("missing b2")?),
            ] {
                assert!((actual - pinned).abs() <= fixture.tolerance);
            }
            let previous_unconditional =
                previous_unconditional.ok_or("missing previous unconditional denoiser")?;
            let denoised_mix = expected
                .unconditional
                .iter()
                .zip(previous_unconditional)
                .map(|(unconditional, previous)| b1 * unconditional + b2 * previous)
                .collect::<Vec<_>>();
            assert_close(
                &denoised_mix,
                expected
                    .denoised_mix
                    .as_deref()
                    .ok_or("missing denoised mix")?,
                fixture.tolerance,
            );
            let corrected = expected
                .current
                .iter()
                .zip(expected.guided.iter().zip(&expected.unconditional))
                .map(|(current, (guided, unconditional))| current + guided - unconditional)
                .collect::<Vec<_>>();
            assert_close(
                &corrected,
                expected.corrected.as_deref().ok_or("missing correction")?,
                fixture.tolerance,
            );
            corrected
                .iter()
                .zip(&denoised_mix)
                .map(|(current, denoised_mix)| {
                    (-step_size).exp() * current + step_size * denoised_mix
                })
                .collect::<Vec<_>>()
        };
        assert_close(&deterministic, &expected.deterministic, fixture.tolerance);
        let next = if let Some(noise) = expected.noise.as_deref() {
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
        previous_unconditional = Some(&expected.unconditional);
        previous_sigma_down = Some(sigma_down);
    }
    Ok(())
}

#[test]
fn val_rng_001_options_failures_cancellation_and_atomicity_are_exact() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let defaults = ResMultistepAncestralCfgPpOptions::source_defaults();
    assert_eq!(defaults.eta(), 1.0);
    assert_eq!(defaults.noise_scale(), 1.0);
    assert!(ResMultistepAncestralCfgPpOptions::new(f32::NAN, 1.0).is_err());
    assert!(ResMultistepAncestralCfgPpOptions::new(1.0, f32::INFINITY).is_err());

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let immutable_alias = initial.clone();
    let wrong_identity = sample_res_multistep_ancestral_cfg_pp(
        &backend,
        plan("res_multistep", fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralCfgPpOptions::default(),
        &context,
        |_input, _sigma, _step| Err("must not run".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        wrong_identity,
        Err(ResMultistepSamplerError::WrongSampler { .. })
    ));

    let denoiser_error = sample_res_multistep_ancestral_cfg_pp(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralCfgPpOptions::default(),
        &context,
        |_input, _sigma, step| Err(format!("failure-{step}")),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        denoiser_error,
        Err(ResMultistepSamplerError::Denoiser { step: 0, .. })
    ));

    let descriptor_error = sample_res_multistep_ancestral_cfg_pp(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralCfgPpOptions::default(),
        &context,
        |_input, _sigma, _step| {
            Ok(ResMultistepAncestralCfgPpDenoiserOutput {
                denoised: initial.clone(),
                unconditional_denoised: tensor_from_f32(&backend, &[1], &[0.0], &context)
                    .map_err(|error| error.to_string())?,
            })
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        descriptor_error,
        Err(ResMultistepSamplerError::DenoiserContract { step: 0 })
    ));

    let callback_error = sample_res_multistep_ancestral_cfg_pp(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        ResMultistepAncestralCfgPpOptions::default(),
        &context,
        |_input, _sigma, step| {
            let expected = fixture.steps.get(step).ok_or("unexpected step")?;
            Ok(ResMultistepAncestralCfgPpDenoiserOutput {
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
        |_progress, _current, _denoised| Err::<(), _>("callback-failure"),
    );
    assert!(matches!(
        callback_error,
        Err(ResMultistepSamplerError::Sampling(_))
    ));
    assert_close(
        &values(&backend, &immutable_alias, &context)?,
        &fixture.initial,
        0.0,
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancellation_error = sample_res_multistep_ancestral_cfg_pp(
        &backend,
        plan(&fixture.identity, fixture.seed, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        request(&fixture, 1, RetryRngPolicy::Advance),
        ResMultistepAncestralCfgPpOptions::default(),
        &cancelled_context,
        |_input, _sigma, _step| Err("must not run".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        cancellation_error,
        Err(ResMultistepSamplerError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}
