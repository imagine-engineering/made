use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingSnrMode,
    generated_dpmpp_2s_ancestral_cfg_pp_sim_model_0173::{
        DEFINITION, DPMPP_2S_ANCESTRAL_CFG_PP_FEATURE_ID,
        DPMPP_2S_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID, DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID,
        DPMPP_2S_ANCESTRAL_CFG_PP_SOURCE_ORDINAL, Dpmpp2sAncestralCfgPpDenoiserOutput,
        Dpmpp2sAncestralCfgPpDenoiserStage, Dpmpp2sAncestralCfgPpError,
        Dpmpp2sAncestralCfgPpOptions, sample_dpmpp_2s_ancestral_cfg_pp,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, RetryRngPolicy, RngCompatibilityRequest, RngExecutionScope,
    RngGenerationPlacement, RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_2s_ancestral_cfg_pp_comfy_model_0173/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/dpmpp_2s_ancestral_cfg_pp_sim_model_0173.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    eta: f32,
    sampler_noise_scale: f32,
    profile_noise_scale: f32,
    effective_noise_scale: f32,
    seed: u64,
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
    ancestral_lines: [usize; 2],
    noise_lines: [usize; 2],
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct RngFixture {
    contract: String,
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    execution_ordinal: u64,
    batch: u64,
    retry: u32,
    retry_policy: String,
    seed_transform: String,
    placement: String,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    guided: Vec<f32>,
    unconditional: Vec<f32>,
    sigma_down: f32,
    sigma_up: f32,
    primary_derivative: Vec<f32>,
    time: Option<f32>,
    next_time: Option<f32>,
    step_size: Option<f32>,
    midpoint_sigma: Option<f32>,
    cfg_delta: Option<Vec<f32>>,
    midpoint_latent: Option<Vec<f32>>,
    midpoint_guided: Option<Vec<f32>>,
    midpoint_unconditional: Option<Vec<f32>>,
    deterministic: Vec<f32>,
    noise: Option<Vec<f32>>,
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
        SamplingProfileIdentity::new("dpmpp-2s-ancestral-cfg-pp-row-v1")?,
        PredictionInterpretation::Epsilon,
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

fn noise_request(fixture: &Fixture) -> CompatibilityNoiseRequest {
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
fn val_sampler_001_dpmpp_2s_ancestral_cfg_pp_definition_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2S_ANCESTRAL_CFG_PP_FEATURE_ID);
    assert_eq!(
        fixture.source_ordinal,
        DPMPP_2S_ANCESTRAL_CFG_PP_SOURCE_ORDINAL
    );
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 14);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2s_ancestral_cfg_pp_comfy_model_0173"
    );
    assert_eq!(
        fixture.rng.contract,
        DPMPP_2S_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID
    );
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.seed_transform, "add-one-on-cpu");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert_ne!(DPMPP_2S_ANCESTRAL_CFG_PP_SAMPLER_ID, "dpmpp_2s_ancestral");

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
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "def sample_dpmpp_2s_ancestral_cfg_pp(",
        "uncond_denoised",
        "disable_cfg1_optimization=True",
        "get_ancestral_step",
        "x + (denoised - temp[0])",
        "denoised_2",
        "noise_sampler(sigmas[i], sigmas[i + 1])",
    ] {
        assert!(equations.contains(fragment), "missing source {fragment}");
    }
    let ancestral = source_range(&sampling, fixture.source.ancestral_lines);
    for fragment in ["sigma_up", "sigma_down", "eta"] {
        assert!(ancestral.contains(fragment), "missing ancestral {fragment}");
    }
    let noise = source_range(&sampling, fixture.source.noise_lines);
    for fragment in ["seed += 1", "torch.Generator", "torch.randn"] {
        assert!(noise.contains(fragment), "missing noise {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_2s_ancestral_cfg_pp\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,dpmpp_2s_ancestral_cfg_pp,")
                && line.ends_with(",COMFY-MODEL-0173"))
    );

    for required in [
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction(",
        "SamplingSession::new",
        ".observe_step(",
        "profile.scale_sampler_noise(",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing owner {required}"
        );
    }
    for forbidden in [
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream",
        "::open",
        "struct Dpmpp2sAncestralCfgPpTrace",
        "struct Dpmpp2sAncestralCfgPpProgress",
        "struct Dpmpp2sAncestralCfgPpObservation",
        "struct Dpmpp2sAncestralCfgPpNoiseRequest",
        "Command::new",
        "include!(",
        "#[path",
        "todo!",
        "unimplemented!",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner or escape {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2s_ancestral_cfg_pp_matches_every_intermediate_and_callback()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    assert!(
        (profile.scale_sampler_noise(fixture.sampler_noise_scale)? - fixture.effective_noise_scale)
            .abs()
            <= fixture.tolerance
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let events = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let (trace, noise_before, noise_after) = sample_dpmpp_2s_ancestral_cfg_pp(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralCfgPpOptions {
            eta: fixture.eta,
            noise_scale: fixture.sampler_noise_scale,
        },
        &context,
        |latent, sigma, step, stage| {
            events.borrow_mut().push(format!("{stage:?}:{step}"));
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| "unexpected denoiser step".to_owned())?;
            let (expected_sigma, expected_latent, guided, unconditional) = match stage {
                Dpmpp2sAncestralCfgPpDenoiserStage::Primary => (
                    expected.sigma,
                    expected.latent_before.as_slice(),
                    expected.guided.as_slice(),
                    expected.unconditional.as_slice(),
                ),
                Dpmpp2sAncestralCfgPpDenoiserStage::Midpoint => (
                    expected
                        .midpoint_sigma
                        .ok_or_else(|| "missing midpoint sigma".to_owned())?,
                    expected
                        .midpoint_latent
                        .as_deref()
                        .ok_or_else(|| "missing midpoint latent".to_owned())?,
                    expected
                        .midpoint_guided
                        .as_deref()
                        .ok_or_else(|| "missing midpoint guided".to_owned())?,
                    expected
                        .midpoint_unconditional
                        .as_deref()
                        .ok_or_else(|| "missing midpoint unconditional".to_owned())?,
                ),
            };
            if (sigma - expected_sigma).abs() > fixture.tolerance {
                return Err(format!("sigma mismatch at step {step}"));
            }
            let actual =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&actual, expected_latent, fixture.tolerance);
            Ok(Dpmpp2sAncestralCfgPpDenoiserOutput {
                denoised: tensor_from_f32(&backend, &fixture.shape, guided, &context)
                    .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    unconditional,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
        },
        |progress, latent, denoised| {
            events
                .borrow_mut()
                .push(format!("Callback:{}", progress.step));
            callbacks.borrow_mut().push((
                *progress,
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<_, String>(())
        },
    )?;

    assert_eq!(
        events.into_inner(),
        [
            "Primary:0",
            "Callback:0",
            "Midpoint:0",
            "Primary:1",
            "Callback:1",
            "Midpoint:1",
            "Primary:2",
            "Callback:2",
        ]
    );
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_close(
            &values(&backend, &trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.denoiser_evaluations[step], &context)?,
            &expected.guided,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            &expected.latent_after,
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
    let callbacks = callbacks.into_inner();
    assert_eq!(callbacks.len(), fixture.steps.len());
    for (step, (progress, latent, denoised)) in callbacks.iter().enumerate() {
        let expected = &fixture.steps[step];
        assert_eq!(usize::try_from(progress.step)?, step);
        assert_eq!(usize::try_from(progress.total_steps)?, fixture.steps.len());
        assert!((progress.sigma - expected.sigma).abs() <= fixture.tolerance);
        assert!((progress.sigma_hat - expected.sigma).abs() <= fixture.tolerance);
        assert!((progress.next_sigma - expected.next_sigma).abs() <= fixture.tolerance);
        assert_close(latent, &expected.latent_before, fixture.tolerance);
        assert_close(denoised, &expected.guided, fixture.tolerance);
    }

    let mut oracle = CompatibilityRngTransaction::open(
        DPMPP_2S_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        RngCompatibilityRequest::new(
            &fixture.rng.workflow,
            &fixture.rng.attempt,
            &fixture.rng.node,
            fixture.rng.output,
            fixture.rng.execution_ordinal,
            fixture.rng.batch,
            fixture.rng.retry,
            RetryRngPolicy::Replay,
            i128::from(fixture.seed),
            RngSeedTransform::Add(1),
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: DeviceId::CPU,
            },
            RngExecutionScope::Production,
        ),
        None,
        &cancellation,
    )?;
    assert_eq!(noise_before, oracle.checkpoint());
    for expected in fixture.steps.iter().filter_map(|step| step.noise.as_ref()) {
        let actual = oracle
            .draw_normal(expected.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_close(&actual, expected, 0.0);
    }
    assert_eq!(noise_after, oracle.commit());
    Ok(())
}

#[test]
fn val_sampling_foundation_001_dpmpp_2s_ancestral_cfg_pp_equations_are_analytical()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    for expected in &fixture.steps {
        let sigma_squared = expected.sigma * expected.sigma;
        let next_squared = expected.next_sigma * expected.next_sigma;
        let sigma_up = if fixture.eta == 0.0 {
            0.0
        } else {
            expected.next_sigma.min(
                fixture.eta
                    * (next_squared * (sigma_squared - next_squared) / sigma_squared).sqrt(),
            )
        };
        let sigma_down = (next_squared - sigma_up * sigma_up).sqrt();
        assert!((sigma_down - expected.sigma_down).abs() <= fixture.tolerance);
        assert!((sigma_up - expected.sigma_up).abs() <= fixture.tolerance);
        let derivative = expected
            .latent_before
            .iter()
            .zip(&expected.unconditional)
            .map(|(latent, unconditional)| (latent - unconditional) / expected.sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &expected.primary_derivative, fixture.tolerance);
        if sigma_down == 0.0 {
            assert!(expected.time.is_none());
            assert!(expected.midpoint_latent.is_none());
            let terminal = expected
                .guided
                .iter()
                .zip(&derivative)
                .map(|(guided, derivative)| guided + derivative * sigma_down)
                .collect::<Vec<_>>();
            assert_close(&terminal, &expected.deterministic, fixture.tolerance);
            continue;
        }
        let time = -expected.sigma.ln();
        let next_time = -sigma_down.ln();
        let step_size = next_time - time;
        let midpoint_sigma = (-(time + 0.5 * step_size)).exp();
        assert!(
            expected
                .time
                .is_some_and(|value| (value - time).abs() <= fixture.tolerance)
        );
        assert!(
            expected
                .next_time
                .is_some_and(|value| (value - next_time).abs() <= fixture.tolerance)
        );
        assert!(
            expected
                .step_size
                .is_some_and(|value| (value - step_size).abs() <= fixture.tolerance)
        );
        assert!(
            expected
                .midpoint_sigma
                .is_some_and(|value| (value - midpoint_sigma).abs() <= fixture.tolerance)
        );
        let cfg_delta = expected
            .guided
            .iter()
            .zip(&expected.unconditional)
            .map(|(guided, unconditional)| guided - unconditional)
            .collect::<Vec<_>>();
        assert_close(
            &cfg_delta,
            expected.cfg_delta.as_deref().ok_or("missing CFG delta")?,
            fixture.tolerance,
        );
        let midpoint = expected
            .latent_before
            .iter()
            .zip(&cfg_delta)
            .zip(&expected.guided)
            .map(|((latent, delta), guided)| {
                midpoint_sigma / expected.sigma * (latent + delta)
                    - (-0.5 * step_size).exp_m1() * guided
            })
            .collect::<Vec<_>>();
        assert_close(
            &midpoint,
            expected
                .midpoint_latent
                .as_deref()
                .ok_or("missing midpoint")?,
            fixture.tolerance,
        );
        let midpoint_guided = expected
            .midpoint_guided
            .as_deref()
            .ok_or("missing midpoint denoiser")?;
        let deterministic = expected
            .latent_before
            .iter()
            .zip(&cfg_delta)
            .zip(midpoint_guided)
            .map(|((latent, delta), midpoint_guided)| {
                sigma_down / expected.sigma * (latent + delta)
                    - (-step_size).exp_m1() * midpoint_guided
            })
            .collect::<Vec<_>>();
        assert_close(&deterministic, &expected.deterministic, fixture.tolerance);
        let noise = expected.noise.as_deref().ok_or("missing normal noise")?;
        let next = deterministic
            .iter()
            .zip(noise)
            .map(|(deterministic, noise)| {
                deterministic + noise * fixture.effective_noise_scale * sigma_up
            })
            .collect::<Vec<_>>();
        assert_close(&next, &expected.latent_after, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_rng_001_dpmpp_2s_ancestral_cfg_pp_failures_are_typed_and_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = || tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context);
    let error = sample_dpmpp_2s_ancestral_cfg_pp(
        &backend,
        plan(&fixture, "dpmpp_2m", &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralCfgPpOptions::default(),
        &context,
        |latent, _, _, _| {
            Ok(Dpmpp2sAncestralCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: latent.clone(),
            })
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("a different registered sampler must not be substituted");
    assert!(matches!(error, Dpmpp2sAncestralCfgPpError::WrongSampler(_)));

    let error = sample_dpmpp_2s_ancestral_cfg_pp(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralCfgPpOptions {
            eta: f32::NAN,
            noise_scale: 1.0,
        },
        &context,
        |latent, _, _, _| {
            Ok(Dpmpp2sAncestralCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: latent.clone(),
            })
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("non-finite eta must fail before RNG publication");
    assert!(matches!(
        error,
        Dpmpp2sAncestralCfgPpError::InvalidOption { .. }
    ));

    let (signed, _, _) = sample_dpmpp_2s_ancestral_cfg_pp(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralCfgPpOptions {
            eta: 0.0,
            noise_scale: -1.0,
        },
        &context,
        |latent, _, _, _| {
            Ok(Dpmpp2sAncestralCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: latent.clone(),
            })
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(signed.latents.len(), fixture.sigmas.len());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_dpmpp_2s_ancestral_cfg_pp(
        &backend,
        plan(&fixture, &fixture.identity, &profile)?,
        &profile,
        initial()?,
        &fixture.sigmas,
        noise_request(&fixture),
        Dpmpp2sAncestralCfgPpOptions::default(),
        &cancelled_context,
        |latent, _, _, _| {
            Ok(Dpmpp2sAncestralCfgPpDenoiserOutput {
                denoised: latent.clone(),
                unconditional_denoised: latent.clone(),
            })
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, Dpmpp2sAncestralCfgPpError::Tensor(_)));
    Ok(())
}
