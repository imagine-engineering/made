use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingError, SamplingPlan, SamplingProfile, SamplingProfileIdentity,
    SamplingProgress, SamplingSnrMode, standard_ancestral_step,
    generated_euler_ancestral_cfg_pp_sim_model_0181::{
        DEFINITION, EULER_ANCESTRAL_CFG_PP_FEATURE_ID, EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, EULER_ANCESTRAL_CFG_PP_SOURCE_ORDINAL,
        EulerAncestralCfgPpDenoiserOutput, EulerAncestralCfgPpError,
        EulerAncestralCfgPpOptions, sample_euler_ancestral_cfg_pp,
    },
    generated_native_diffusion::NativeDiffusionSamplerError,
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
    "/../sim_test_support/fixtures/samplers/euler_ancestral_cfg_pp_comfy_model_0181/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/euler_ancestral_cfg_pp_sim_model_0181.rs"
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
    constant_flow_case: ConstantFlowFixture,
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
struct ConstantFlowFixture {
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    guided: Vec<Vec<f32>>,
    unconditional: Vec<Vec<f32>>,
    eta: f32,
    sampler_noise_scale: f32,
    alpha_source: f32,
    alpha_target: f32,
    normalized_sigma_source: f32,
    normalized_sigma_target: f32,
    normalized_sigma_down: f32,
    normalized_sigma_up: f32,
    physical_sigma_down: f32,
    derivative: Vec<f32>,
    first_latent: Vec<f32>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    guided: Vec<f32>,
    unconditional: Vec<f32>,
    alpha_source: Option<f32>,
    alpha_target: Option<f32>,
    sigma_down: Option<f32>,
    sigma_up: Option<f32>,
    derivative: Option<Vec<f32>>,
    deterministic: Vec<f32>,
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
        SamplingProfileIdentity::new("euler-ancestral-cfg-pp-row-v1")?,
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
fn val_sampler_001_definition_provenance_registry_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID);
    assert_eq!(fixture.feature_id, EULER_ANCESTRAL_CFG_PP_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, EULER_ANCESTRAL_CFG_PP_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 3);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(fixture.rng.contract, EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID);
    assert_eq!(fixture.rng.retry_policy, "replay");
    assert_eq!(fixture.rng.seed_transform, "add-one-on-cpu");
    assert_eq!(fixture.rng.placement, "cpu-seeded-transfer");
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(EULER_ANCESTRAL_CFG_PP_SAMPLER_ID)?)?,
        &DEFINITION
    );

    let root = workspace_root()?;
    for (path, expected) in [
        (&fixture.source.sampling_path, &fixture.source.sampling_sha256),
        (&fixture.source.samplers_path, &fixture.source.samplers_sha256),
        (&fixture.source.catalog_path, &fixture.source.catalog_sha256),
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let equations = source_range(&sampling, fixture.source.equation_lines);
    for fragment in [
        "def sample_euler_ancestral_cfg_pp(",
        "uncond_denoised",
        "disable_cfg1_optimization=True",
        "alpha_s",
        "alpha_t",
        "get_ancestral_step",
        "alpha_t * denoised + sigma_down * d",
        "alpha_t * noise_sampler",
    ] {
        assert!(equations.contains(fragment), "missing source {fragment}");
    }
    let ancestral = source_range(&sampling, fixture.source.ancestral_lines);
    assert!(ancestral.contains("sigma_down"));
    assert!(ancestral.contains("sigma_up"));
    let noise = source_range(&sampling, fixture.source.noise_lines);
    assert!(noise.contains("seed += 1"));
    assert!(noise.contains("torch.Generator"));
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"euler_ancestral_cfg_pp\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,euler_ancestral_cfg_pp,")
                && line.ends_with(",COMFY-MODEL-0181"))
    );
    for required in [
        "standard_ancestral_step(",
        "profile.half_log_snr(",
        "profile.scale_sampler_noise(",
        "validate_euler_noise_generation_device(",
        "SamplingSession::new",
        "observe_euler_denoised(",
    ] {
        assert!(IMPLEMENTATION.contains(required), "missing owner mapping {required}");
    }
    for forbidden in ["struct SamplingProgress", "struct SamplingTrace", "struct CancellationToken"] {
        assert!(!IMPLEMENTATION.contains(forbidden), "duplicate owner {forbidden}");
    }
    Ok(())
}

#[test]
fn val_sampler_001_every_equation_callback_and_rng_draw_match_the_fixture()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let callbacks = RefCell::new(Vec::<(SamplingProgress, Vec<f32>, Vec<f32>)>::new());
    let denoiser_inputs = RefCell::new(Vec::<Vec<f32>>::new());
    let (trace, before, after) = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions {
            eta: fixture.eta,
            noise_scale: fixture.sampler_noise_scale,
        },
        &context,
        |latent, sigma, step| {
            denoiser_inputs.borrow_mut().push(
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
            );
            assert!((sigma - fixture.steps[step].sigma).abs() <= fixture.tolerance);
            Ok(EulerAncestralCfgPpDenoiserOutput {
                denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].guided,
                    &context,
                )
                .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].unconditional,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
        },
        |progress, latent, denoised| {
            callbacks.borrow_mut().push((
                *progress,
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<_, String>(())
        },
    )?;

    let first = &fixture.steps[0];
    let (sigma_down, sigma_up) = standard_ancestral_step(first.sigma, first.next_sigma, fixture.eta)?;
    assert!((sigma_down - first.sigma_down.ok_or("missing sigma down")?).abs() <= fixture.tolerance);
    assert!((sigma_up - first.sigma_up.ok_or("missing sigma up")?).abs() <= fixture.tolerance);
    assert_eq!(first.alpha_source, Some(1.0));
    assert_eq!(first.alpha_target, Some(1.0));
    let derivative = fixture
        .initial
        .iter()
        .zip(&first.unconditional)
        .map(|(current, unconditional)| (current - unconditional) / first.sigma)
        .collect::<Vec<_>>();
    assert_close(
        &derivative,
        first.derivative.as_deref().ok_or("missing derivative")?,
        fixture.tolerance,
    );
    let deterministic = first
        .guided
        .iter()
        .zip(&derivative)
        .map(|(guided, derivative)| guided + sigma_down * derivative)
        .collect::<Vec<_>>();
    assert_close(&deterministic, &first.deterministic, fixture.tolerance);

    let mut oracle = noise_request(&fixture).open_transaction(
        EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    let noise = oracle.draw_normal(fixture.initial.len(), &cancellation)?;
    let expected_after_first = deterministic
        .iter()
        .zip(noise)
        .map(|(deterministic, noise)| {
            (noise as f32).mul_add(fixture.effective_noise_scale * sigma_up, *deterministic)
        })
        .collect::<Vec<_>>();
    assert_eq!(after, oracle.commit());
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_close(
        &values(&backend, &trace.latents[0], &context)?,
        &fixture.initial,
        fixture.tolerance,
    );
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &expected_after_first,
        fixture.tolerance,
    );
    assert_close(
        &values(&backend, &trace.latents[2], &context)?,
        &fixture.terminal,
        fixture.tolerance,
    );
    assert_close(&fixture.steps[1].deterministic, &fixture.terminal, fixture.tolerance);
    assert!(fixture.steps[1].alpha_source.is_none());
    assert!(fixture.steps[1].alpha_target.is_none());
    assert!(fixture.steps[1].sigma_down.is_none());
    assert!(fixture.steps[1].sigma_up.is_none());
    assert!(fixture.steps[1].derivative.is_none());

    let callbacks = callbacks.into_inner();
    assert_eq!(callbacks.len(), fixture.steps.len());
    assert_close(&callbacks[0].1, &fixture.initial, fixture.tolerance);
    assert_close(&callbacks[0].2, &first.guided, fixture.tolerance);
    assert_close(&callbacks[1].1, &expected_after_first, fixture.tolerance);
    assert_close(&callbacks[1].2, &fixture.steps[1].guided, fixture.tolerance);
    let denoiser_inputs = denoiser_inputs.into_inner();
    assert_close(&denoiser_inputs[0], &fixture.initial, fixture.tolerance);
    assert_close(&denoiser_inputs[1], &expected_after_first, fixture.tolerance);
    for (step, (progress, _, _)) in callbacks.iter().enumerate() {
        assert_eq!(usize::try_from(progress.step)?, fixture.steps[step].step);
        assert!((progress.sigma - fixture.steps[step].sigma).abs() <= fixture.tolerance);
        assert!((progress.next_sigma - fixture.steps[step].next_sigma).abs() <= fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_rng_001_eta_zero_skips_draws_but_preserves_the_source_generator_address()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let (trace, before, after) = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions {
            eta: 0.0,
            noise_scale: 0.0,
        },
        &context,
        |_, _, step| {
            Ok(EulerAncestralCfgPpDenoiserOutput {
                denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].guided,
                    &context,
                )
                .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].unconditional,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
        },
        |_, _, _| Ok::<_, String>(()),
    )?;
    let expected_first = fixture.steps[0]
        .guided
        .iter()
        .zip(
            fixture
                .initial
                .iter()
                .zip(&fixture.steps[0].unconditional)
                .map(|(current, unconditional)| (current - unconditional) / fixture.steps[0].sigma),
        )
        .map(|(guided, derivative)| guided + fixture.steps[0].next_sigma * derivative)
        .collect::<Vec<_>>();
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &expected_first,
        fixture.tolerance,
    );
    let oracle = noise_request(&fixture).open_transaction(
        EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(before, oracle.checkpoint());
    assert_eq!(after, oracle.commit());
    Ok(())
}

#[test]
fn val_sampler_001_constant_flow_and_negative_noise_follow_source_boundaries()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let case = &fixture.constant_flow_case;
    let profile = DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("euler-ancestral-cfg-pp-constant-flow-v1")?,
        PredictionInterpretation::Flow,
        Arc::from([0.01_f32, 0.1, 0.4, 0.8]),
        SamplingSnrMode::ConstantFlow { shift: 1.0 },
        fixture.profile_noise_scale,
    )?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let (trace, before, after) = sample_euler_ancestral_cfg_pp(
        &backend,
        SamplingPlan::new(
            EULER_ANCESTRAL_CFG_PP_SAMPLER_ID,
            "normal",
            profile.identity().clone(),
            fixture.seed,
            u32::try_from(case.guided.len())?,
            1.0,
            1.0,
        )?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &case.initial, &context)?,
        &case.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions {
            eta: case.eta,
            noise_scale: case.sampler_noise_scale,
        },
        &context,
        |_, _, step| {
            Ok(EulerAncestralCfgPpDenoiserOutput {
                denoised: tensor_from_f32(&backend, &fixture.shape, &case.guided[step], &context)
                    .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &case.unconditional[step],
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
        },
        |_, _, _| Ok::<_, String>(()),
    )?;

    let alpha_source = case.sigmas[0] * profile.half_log_snr(case.sigmas[0])?.exp();
    let alpha_target = case.sigmas[1] * profile.half_log_snr(case.sigmas[1])?.exp();
    assert!((alpha_source - case.alpha_source).abs() <= fixture.tolerance);
    assert!((alpha_target - case.alpha_target).abs() <= fixture.tolerance);
    let normalized_source = case.sigmas[0] / alpha_source;
    let normalized_target = case.sigmas[1] / alpha_target;
    assert!((normalized_source - case.normalized_sigma_source).abs() <= fixture.tolerance);
    assert!((normalized_target - case.normalized_sigma_target).abs() <= fixture.tolerance);
    let (normalized_down, normalized_up) =
        standard_ancestral_step(normalized_source, normalized_target, case.eta)?;
    assert!((normalized_down - case.normalized_sigma_down).abs() <= fixture.tolerance);
    assert!((normalized_up - case.normalized_sigma_up).abs() <= fixture.tolerance);
    assert!(
        (alpha_target * normalized_down - case.physical_sigma_down).abs()
            <= fixture.tolerance
    );
    assert_close(
        &values(&backend, &trace.latents[1], &context)?,
        &case.first_latent,
        fixture.tolerance,
    );
    assert_close(
        &values(&backend, trace.latents.last().ok_or("missing terminal latent")?, &context)?,
        &case.terminal,
        fixture.tolerance,
    );
    assert_close(
        &case
            .initial
            .iter()
            .zip(&case.unconditional[0])
            .map(|(current, unconditional)| (current - alpha_source * unconditional) / case.sigmas[0])
            .collect::<Vec<_>>(),
        &case.derivative,
        fixture.tolerance,
    );
    assert_eq!(before, after, "negative source noise scale must consume no RNG values");
    Ok(())
}

#[test]
fn val_sampling_foundation_001_failures_are_typed_cancellable_and_failure_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let wrong = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, "euler_ancestral", &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions::default(),
        &context,
        |latent, _, _| Ok(EulerAncestralCfgPpDenoiserOutput {
            denoised: latent.clone(),
            unconditional_denoised: latent.clone(),
        }),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("wrong identity must fail");
    assert!(matches!(wrong, EulerAncestralCfgPpError::WrongSampler { .. }));

    let invalid = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions {
            eta: f32::NAN,
            noise_scale: 1.0,
        },
        &context,
        |latent, _, _| Ok(EulerAncestralCfgPpDenoiserOutput {
            denoised: latent.clone(),
            unconditional_denoised: latent.clone(),
        }),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("invalid option must fail");
    assert!(matches!(invalid, EulerAncestralCfgPpError::InvalidOption { .. }));

    let mismatched = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions::default(),
        &context,
        |_, _, _| {
            Ok(EulerAncestralCfgPpDenoiserOutput {
                denoised: tensor_from_f32(&backend, &[1], &[0.0], &context)
                    .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.initial,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
        },
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("descriptor mismatch must fail");
    assert!(matches!(
        mismatched,
        EulerAncestralCfgPpError::DenoiserContract {
            output: "guided denoiser output",
            ..
        }
    ));

    let callback_failed = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions::default(),
        &context,
        |_, _, step| {
            Ok(EulerAncestralCfgPpDenoiserOutput {
                denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].guided,
                    &context,
                )
                .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].unconditional,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
        },
        |_, _, _| Err::<(), _>("callback-stop"),
    )
    .expect_err("callback failure must abort the step");
    assert!(matches!(
        callback_failed,
        EulerAncestralCfgPpError::EulerFoundation(NativeDiffusionSamplerError::Sampling(
            SamplingError::Callback(reason)
        )) if reason == "callback-stop"
    ));

    let clean_run = || {
        let (trace, before, after) = sample_euler_ancestral_cfg_pp(
            &backend,
            plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
            &profile,
            initial.clone(),
            &fixture.sigmas,
            noise_request(&fixture),
            EulerAncestralCfgPpOptions::default(),
            &context,
            |_, _, step| {
                Ok(EulerAncestralCfgPpDenoiserOutput {
                    denoised: tensor_from_f32(
                        &backend,
                        &fixture.shape,
                        &fixture.steps[step].guided,
                        &context,
                    )
                    .map_err(|error| error.to_string())?,
                    unconditional_denoised: tensor_from_f32(
                        &backend,
                        &fixture.shape,
                        &fixture.steps[step].unconditional,
                        &context,
                    )
                    .map_err(|error| error.to_string())?,
                })
            },
            |_, _, _| Ok::<_, String>(()),
        )?;
        let latents = trace
            .latents
            .iter()
            .map(|tensor| values(&backend, tensor, &context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok::<_, Box<dyn Error>>((latents, before, after))
    };
    assert_eq!(clean_run()?, clean_run()?, "retry must replay after abort");

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_euler_ancestral_cfg_pp(
        &backend,
        plan(&fixture, EULER_ANCESTRAL_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        EulerAncestralCfgPpOptions::default(),
        &cancelled_context,
        |latent, _, _| Ok(EulerAncestralCfgPpDenoiserOutput {
            denoised: latent.clone(),
            unconditional_denoised: latent.clone(),
        }),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancellation must fail");
    assert!(matches!(error, EulerAncestralCfgPpError::Tensor(_)));
    Ok(())
}
