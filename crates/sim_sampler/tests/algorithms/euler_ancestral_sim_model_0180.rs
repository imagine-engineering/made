use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation,
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    SamplingProgress, SamplingTrace,
    generated_euler_ancestral_sim_model_0180::{
        DEFINITION, EULER_ANCESTRAL_FEATURE_ID, EULER_ANCESTRAL_NOISE_CONTRACT_ID,
        EULER_ANCESTRAL_SAMPLER_ID, EULER_ANCESTRAL_SOURCE_ORDINAL, EulerAncestralError,
        EulerAncestralMode, EulerAncestralOptions, sample_euler_ancestral,
        validate_euler_ancestral_generation_device,
    },
    generated_native_diffusion::NativeDiffusionSamplerError,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngCompatibilityOperation, RngCompatibilityPhase, StreamId, Tensor,
    TensorError, generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/euler_ancestral_comfy_model_0180/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/euler_ancestral_sim_model_0180.rs");
const EULER_OWNER: &str = include_str!("../../src/algorithms/native_diffusion.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    rng_contract_id: String,
    seed: u64,
    initial: Vec<f32>,
    denoised: Vec<Vec<f32>>,
    standard: CaseFixture,
    flow: CaseFixture,
    signed_controls: SignedControlsFixture,
    tolerance: f32,
}

#[derive(Debug, Deserialize)]
struct SignedControlsFixture {
    negative_eta: f32,
    negative_noise_scale: f32,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    sampling_path: String,
    sampling_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
    standard_lines: [usize; 2],
    flow_lines: [usize; 2],
    ancestral_lines: [usize; 2],
    noise_lines: [usize; 2],
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct CaseFixture {
    profile: String,
    prediction: String,
    eta: f32,
    noise_scale: f32,
    sigmas: Vec<f32>,
    latents: Vec<Vec<f32>>,
    steps: Vec<StepFixture>,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    sigma_down: f32,
    sigma_up: f32,
    derivative: Vec<f32>,
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

fn profile(case: &CaseFixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    let prediction = match case.prediction.as_str() {
        "epsilon" => PredictionInterpretation::Epsilon,
        "flow" => PredictionInterpretation::Flow,
        value => return Err(format!("unknown prediction {value:?}").into()),
    };
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new(case.profile.clone())?,
        prediction,
        Arc::from([0.1_f32, 0.5, 1.0, 2.0]),
    )?)
}

fn plan(case: &CaseFixture, identity: &str, seed: u64) -> Result<SamplingPlan, Box<dyn Error>> {
    plan_with_steps(case, identity, seed, case.steps.len())
}

fn plan_with_steps(
    case: &CaseFixture,
    identity: &str,
    seed: u64,
    steps: usize,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new(case.profile.clone())?,
        seed,
        u32::try_from(steps)?,
        1.0,
        1.0,
    )?)
}

fn noise_request() -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "dpm2-ancestral-fixture-workflow",
        "dpm2-ancestral-fixture-attempt",
        "KSampler",
        0,
        0,
        0,
        0,
        RetryRngPolicy::Replay,
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

fn standard_source_oracle(
    fixture: &Fixture,
    eta: f32,
    noise_scale: f32,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let case = &fixture.standard;
    let mut current = fixture.initial.clone();
    let mut latents = vec![current.clone()];
    for (step, pair) in case.sigmas.windows(2).enumerate() {
        let sigma = pair[0];
        let next_sigma = pair[1];
        if next_sigma == 0.0 {
            current = fixture.denoised[step].clone();
        } else {
            let sigma_up = next_sigma.min(
                eta * (next_sigma.powi(2) * (sigma.powi(2) - next_sigma.powi(2))
                    / sigma.powi(2))
                .sqrt(),
            );
            let sigma_down = (next_sigma.powi(2) - sigma_up.powi(2)).sqrt();
            let noise = case.steps[step]
                .noise
                .as_deref()
                .ok_or("missing standard signed-control noise")?;
            current = current
                .iter()
                .zip(&fixture.denoised[step])
                .zip(noise)
                .map(|((latent, denoised), noise)| {
                    latent
                        + ((latent - denoised) / sigma) * (sigma_down - sigma)
                        + noise * noise_scale * sigma_up
                })
                .collect();
        }
        latents.push(current.clone());
    }
    Ok(latents)
}

fn flow_source_oracle(
    fixture: &Fixture,
    eta: f32,
    noise_scale: f32,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let case = &fixture.flow;
    let mut current = fixture.initial.clone();
    let mut latents = vec![current.clone()];
    for (step, pair) in case.sigmas.windows(2).enumerate() {
        let sigma = pair[0];
        let next_sigma = pair[1];
        if next_sigma == 0.0 {
            current = fixture.denoised[step].clone();
        } else {
            let downstep_ratio = 1.0 + (next_sigma / sigma - 1.0) * eta;
            let sigma_down = next_sigma * downstep_ratio;
            let alpha_next = 1.0 - next_sigma;
            let alpha_down = 1.0 - sigma_down;
            let renoise_coefficient = (next_sigma.powi(2)
                - sigma_down.powi(2) * alpha_next.powi(2) / alpha_down.powi(2))
            .sqrt();
            let ratio = sigma_down / sigma;
            let noise = case.steps[step]
                .noise
                .as_deref()
                .ok_or("missing flow signed-control noise")?;
            current = current
                .iter()
                .zip(&fixture.denoised[step])
                .zip(noise)
                .map(|((latent, denoised), noise)| {
                    let deterministic = ratio * latent + (1.0 - ratio) * denoised;
                    alpha_next / alpha_down * deterministic
                        + noise * noise_scale * fixture.flow.noise_scale * renoise_coefficient
                })
                .collect();
        }
        latents.push(current.clone());
    }
    Ok(latents)
}

#[test]
fn definition_ordinal_provenance_and_euler_ownership_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, EULER_ANCESTRAL_SAMPLER_ID);
    assert_eq!(fixture.feature_id, EULER_ANCESTRAL_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, EULER_ANCESTRAL_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/euler_ancestral_comfy_model_0180"
    );
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(EULER_ANCESTRAL_SAMPLER_ID)?)?,
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
    let standard = source_range(&sampling, fixture.source.standard_lines);
    let flow = source_range(&sampling, fixture.source.flow_lines);
    let ancestral = source_range(&sampling, fixture.source.ancestral_lines);
    let noise = source_range(&sampling, fixture.source.noise_lines);
    for fragment in [
        "sample_euler_ancestral",
        "get_ancestral_step",
        "callback({'x': x",
        "if sigma_down == 0",
        "x = x + d * dt + noise_sampler",
    ] {
        assert!(standard.contains(fragment), "missing standard source {fragment}");
    }
    for fragment in [
        "sample_euler_ancestral_RF",
        "downstep_ratio",
        "renoise_coeff",
        "sigma_down_i_ratio",
        "alpha_ip1 / alpha_down",
    ] {
        assert!(flow.contains(fragment), "missing flow source {fragment}");
    }
    for fragment in ["sigma_up", "sigma_down", "eta"] {
        assert!(ancestral.contains(fragment));
    }
    for fragment in ["seed += 1", "torch.Generator", "torch.randn"] {
        assert!(noise.contains(fragment));
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert_eq!(
        pinned_ksampler_names(&samplers)?
            .iter()
            .position(|identity| identity == EULER_ANCESTRAL_SAMPLER_ID),
        Some(usize::from(EULER_ANCESTRAL_SOURCE_ORDINAL))
    );
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("KSAMPLER_NAMES"))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.starts_with("sampler,euler_ancestral,")
                && line.ends_with(",COMFY-MODEL-0180"))
    );

    assert!(IMPLEMENTATION.contains("observe_euler_prediction"));
    assert!(IMPLEMENTATION.contains("observe_euler_denoised"));
    assert!(IMPLEMENTATION.contains("advance_euler"));
    assert!(!IMPLEMENTATION.contains("(current - denoised) / sigma"));
    assert_eq!(EULER_OWNER.matches("fn euler_derivative").count(), 1);
    assert_eq!(EULER_OWNER.matches("fn advance_euler").count(), 1);
    Ok(())
}

fn run_case(
    fixture: &Fixture,
    case: &CaseFixture,
    expected_mode: EulerAncestralMode,
) -> Result<
    (
        SamplingTrace,
        sim_tensor::RngCheckpoint,
        sim_tensor::RngCheckpoint,
    ),
    Box<dyn Error>,
> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let events = RefCell::new(Vec::new());
    let (trace, mode, before, after) = sample_euler_ancestral(
        &backend,
        plan(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed)?,
        &profile(case)?,
        tensor_from_f32(&backend, &[4], &fixture.initial, &context)?,
        &case.sigmas,
        noise_request(),
        EulerAncestralOptions::new(case.eta, case.noise_scale)?,
        &context,
        |input, sigma, step| {
            events.borrow_mut().push(format!("denoiser-{step}"));
            assert_eq!(sigma.to_bits(), case.sigmas[step].to_bits());
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                &case.latents[step],
                fixture.tolerance,
            );
            tensor_from_f32(&backend, &[4], &fixture.denoised[step], &context)
                .map_err(|error| error.to_string())
        },
        |progress: &SamplingProgress, input, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            events.borrow_mut().push(format!("callback-{step}"));
            assert_eq!(progress.sigma.to_bits(), case.sigmas[step].to_bits());
            assert_eq!(progress.sigma_hat.to_bits(), progress.sigma.to_bits());
            assert_eq!(progress.next_sigma.to_bits(), case.sigmas[step + 1].to_bits());
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                &case.latents[step],
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &fixture.denoised[step],
                fixture.tolerance,
            );
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(mode, expected_mode);
    assert_eq!(
        events.into_inner(),
        (0..case.steps.len())
            .flat_map(|step| [format!("denoiser-{step}"), format!("callback-{step}")])
            .collect::<Vec<_>>()
    );
    for (actual, expected) in trace.latents.iter().zip(&case.latents) {
        assert_close(
            &values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    for (step, expected) in case.steps.iter().enumerate() {
        let current = &case.latents[step];
        let sigma = case.sigmas[step];
        let derivative = current
            .iter()
            .zip(&fixture.denoised[step])
            .map(|(current, denoised)| (current - denoised) / sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &expected.derivative, fixture.tolerance);
        assert_close(
            &values(&backend, &trace.latents[step + 1], &context)?,
            &expected.next,
            fixture.tolerance,
        );
        assert!(expected.sigma_down.is_finite());
        assert!(expected.sigma_up.is_finite());
        if let Some(noise) = &expected.noise {
            assert!(noise.iter().all(|value| value.is_finite()));
        }
    }
    Ok((trace, before, after))
}

#[test]
fn val_sampler_001_matches_standard_and_rectified_flow_intermediates()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.rng_contract_id, EULER_ANCESTRAL_NOISE_CONTRACT_ID);
    let contract = rng_compatibility_contract(EULER_ANCESTRAL_NOISE_CONTRACT_ID)
        .ok_or("Euler ancestral RNG contract is unavailable")?;
    assert_eq!(contract.operation(), RngCompatibilityOperation::Normal);
    assert_eq!(contract.phase(), RngCompatibilityPhase::SamplingNoiseAndSolver);
    assert_eq!(contract.symbol(), "torch.randn");
    let standard = run_case(&fixture, &fixture.standard, EulerAncestralMode::Standard)?;
    let flow = run_case(
        &fixture,
        &fixture.flow,
        EulerAncestralMode::RectifiedFlow,
    )?;
    assert_eq!(standard.1, flow.1);
    assert_eq!(standard.2, flow.2);
    assert_ne!(standard.1, standard.2);
    assert_eq!(standard.1.device, DeviceId::CPU);
    Ok(())
}

#[test]
fn val_sampler_001_signed_controls_match_source_equations() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let cases = [
        (
            &fixture.standard,
            EulerAncestralMode::Standard,
            fixture.signed_controls.negative_eta,
            1.0,
            standard_source_oracle(
                &fixture,
                fixture.signed_controls.negative_eta,
                1.0,
            )?,
        ),
        (
            &fixture.standard,
            EulerAncestralMode::Standard,
            fixture.standard.eta,
            fixture.signed_controls.negative_noise_scale,
            standard_source_oracle(
                &fixture,
                fixture.standard.eta,
                fixture.signed_controls.negative_noise_scale,
            )?,
        ),
        (
            &fixture.flow,
            EulerAncestralMode::RectifiedFlow,
            fixture.flow.eta,
            fixture.signed_controls.negative_noise_scale,
            flow_source_oracle(
                &fixture,
                fixture.flow.eta,
                fixture.signed_controls.negative_noise_scale,
            )?,
        ),
    ];

    for (case, expected_mode, eta, noise_scale, expected_latents) in cases {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let (trace, mode, before, after) = sample_euler_ancestral(
            &backend,
            plan(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed)?,
            &profile(case)?,
            tensor_from_f32(&backend, &[4], &fixture.initial, &context)?,
            &case.sigmas,
            noise_request(),
            EulerAncestralOptions::new(eta, noise_scale)?,
            &context,
            |_, _, step| {
                tensor_from_f32(&backend, &[4], &fixture.denoised[step], &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )?;
        assert_eq!(mode, expected_mode);
        assert_ne!(before, after);
        assert_eq!(trace.latents.len(), expected_latents.len());
        for (actual, expected) in trace.latents.iter().zip(expected_latents) {
            assert_close(
                &values(&backend, actual, &context)?,
                &expected,
                fixture.tolerance,
            );
        }
    }
    Ok(())
}

#[test]
fn val_rng_001_replay_and_eta_zero_draw_boundaries_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let first = run_case(&fixture, &fixture.standard, EulerAncestralMode::Standard)?;
    let replay = run_case(&fixture, &fixture.standard, EulerAncestralMode::Standard)?;
    assert_eq!(first.1, replay.1);
    assert_eq!(first.2, replay.2);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    for (case, expected_mode, consumes_noise) in [
        (&fixture.standard, EulerAncestralMode::Standard, true),
        (&fixture.flow, EulerAncestralMode::RectifiedFlow, false),
    ] {
        let (_, mode, before, after) = sample_euler_ancestral(
            &backend,
            plan(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed)?,
            &profile(case)?,
            tensor_from_f32(&backend, &[4], &fixture.initial, &context)?,
            &case.sigmas,
            noise_request(),
            EulerAncestralOptions::new(0.0, 1.0)?,
            &context,
            |_, _, step| tensor_from_f32(&backend, &[4], &fixture.denoised[step], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(()),
        )?;
        assert_eq!(mode, expected_mode);
        assert_eq!(before != after, consumes_noise);
    }
    Ok(())
}

#[test]
fn failures_and_cancellation_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let cuda = DeviceId::from_source_device("cuda:0")?;
    assert!(matches!(
        validate_euler_ancestral_generation_device(cuda),
        Err(EulerAncestralError::DeviceUnavailable { device, .. }) if device == cuda
    ));
    assert!(matches!(
        EulerAncestralOptions::new(f32::NAN, 1.0),
        Err(EulerAncestralError::InvalidOption { name: "eta", .. })
    ));
    assert_eq!(EulerAncestralOptions::new(-0.5, 1.0)?.eta(), -0.5);
    assert_eq!(EulerAncestralOptions::new(1.0, -1.0)?.noise_scale(), -1.0);
    let fixture = fixture()?;
    let case = &fixture.standard;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    assert!(matches!(
        sample_euler_ancestral(
            &backend,
            plan(case, "euler", fixture.seed)?,
            &profile(case)?,
            initial.clone(),
            &case.sigmas,
            noise_request(),
            EulerAncestralOptions::default(),
            &context,
            |input, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(EulerAncestralError::WrongSampler(identity)) if identity == "euler"
    ));
    assert!(matches!(
        sample_euler_ancestral(
            &backend,
            plan(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed)?,
            &profile(case)?,
            initial,
            &case.sigmas,
            noise_request(),
            EulerAncestralOptions::default(),
            &context,
            |_, _, _| Err("fixture denoiser failure".to_owned()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(EulerAncestralError::EulerFoundation(
            NativeDiffusionSamplerError::Denoiser { ref reason, .. }
        )) if reason == "fixture denoiser failure"
    ));

    for invalid_sigmas in [
        Vec::new(),
        vec![1.0],
        vec![f32::NAN, 0.0],
        vec![f32::INFINITY, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![1.0, -1.0],
    ] {
        let result = sample_euler_ancestral(
            &backend,
            plan_with_steps(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed, 1)?,
            &profile(case)?,
            tensor_from_f32(&backend, &[4], &fixture.initial, &context)?,
            &invalid_sigmas,
            noise_request(),
            EulerAncestralOptions::default(),
            &context,
            |input, _, _| Ok(input.clone()),
            |_, _, _| Ok::<(), String>(()),
        );
        if invalid_sigmas.len() == 2 {
            assert!(matches!(
                result,
                Err(EulerAncestralError::Sampling(
                    SamplingError::InvalidSigma { .. }
                ))
            ));
        } else {
            assert!(matches!(
                result,
                Err(EulerAncestralError::Sampling(
                    SamplingError::ScheduleLength {
                        expected: 2,
                        actual
                    }
                )) if actual == invalid_sigmas.len()
            ));
        }
    }

    let (terminal_trace, _, terminal_before, terminal_after) = sample_euler_ancestral(
        &backend,
        plan_with_steps(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed, 1)?,
        &profile(case)?,
        tensor_from_f32(&backend, &[4], &fixture.initial, &context)?,
        &[1.0, 0.0],
        noise_request(),
        EulerAncestralOptions::default(),
        &context,
        |_, _, _| tensor_from_f32(&backend, &[4], &fixture.denoised[0], &context)
            .map_err(|error| error.to_string()),
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_eq!(terminal_before, terminal_after);
    assert_close(
        &values(
            &backend,
            terminal_trace
                .latents
                .last()
                .ok_or("terminal-only latent is unavailable")?,
            &context,
        )?,
        &fixture.denoised[0],
        fixture.tolerance,
    );

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    assert!(matches!(
        sample_euler_ancestral(
            &backend,
            plan(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed)?,
            &profile(case)?,
            tensor_from_f32(&backend, &[4], &fixture.initial, &callback_context)?,
            &case.sigmas,
            noise_request(),
            EulerAncestralOptions::default(),
            &callback_context,
            |_, _, step| tensor_from_f32(
                &backend,
                &[4],
                &fixture.denoised[step],
                &callback_context,
            )
            .map_err(|error| error.to_string()),
            |progress, _, _| {
                if progress.step == 1 {
                    callback_cancellation.cancel();
                }
                Ok::<(), String>(())
            },
        ),
        Err(EulerAncestralError::EulerFoundation(
            NativeDiffusionSamplerError::Sampling(SamplingError::Cancelled)
        ))
    ));

    let clean_run = || {
        let retry_cancellation = CancellationToken::default();
        let retry_context = execution_context(&backend, &authority, &retry_cancellation)?;
        let (trace, _, before, after) = sample_euler_ancestral(
            &backend,
            plan(case, EULER_ANCESTRAL_SAMPLER_ID, fixture.seed)?,
            &profile(case)?,
            tensor_from_f32(&backend, &[4], &fixture.initial, &retry_context)?,
            &case.sigmas,
            noise_request(),
            EulerAncestralOptions::default(),
            &retry_context,
            |_, _, step| {
                tensor_from_f32(&backend, &[4], &fixture.denoised[step], &retry_context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )?;
        let latents = trace
            .latents
            .iter()
            .map(|tensor| values(&backend, tensor, &retry_context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok::<_, Box<dyn Error>>((latents, before, after))
    };
    let clean = clean_run()?;
    let retry_after_cancel = clean_run()?;
    assert_eq!(retry_after_cancel, clean);
    Ok(())
}
