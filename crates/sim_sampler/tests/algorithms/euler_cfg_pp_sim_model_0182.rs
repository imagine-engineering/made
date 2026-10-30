use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity, SamplingProgress,
    SamplingSnrMode,
    generated_euler_ancestral_cfg_pp_sim_model_0181::{
        EULER_ANCESTRAL_CFG_PP_NOISE_CONTRACT_ID, EulerAncestralCfgPpError,
    },
    generated_euler_cfg_pp_sim_model_0182::{
        DEFINITION, EULER_CFG_PP_FEATURE_ID, EULER_CFG_PP_SAMPLER_ID,
        EULER_CFG_PP_SOURCE_ORDINAL, EulerCfgPpDenoiserOutput, sample_euler_cfg_pp,
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
    "/../sim_test_support/fixtures/samplers/euler_cfg_pp_comfy_model_0182/trajectory.json"
));
const CANONICAL_FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/euler_ancestral_cfg_pp_comfy_model_0181/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/euler_cfg_pp_sim_model_0182.rs"
));

#[derive(Debug, Deserialize)]
struct AdapterFixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    family_identity: String,
    eta: f32,
    noise_scale: f32,
    canonical_fixture_path: String,
    canonical_fixture_sha256: String,
    source: SourceFixture,
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
struct CanonicalFixture {
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    profile_noise_scale: f32,
    seed: u64,
    rng: RngFixture,
    steps: Vec<StepFixture>,
    terminal: Vec<f32>,
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
    step: usize,
    sigma: f32,
    next_sigma: f32,
    guided: Vec<f32>,
    unconditional: Vec<f32>,
}

fn fixture() -> Result<AdapterFixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn canonical_fixture() -> Result<CanonicalFixture, Box<dyn Error>> {
    Ok(serde_json::from_str(CANONICAL_FIXTURE_JSON)?)
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

fn profile(fixture: &CanonicalFixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("euler-cfg-pp-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
        SamplingSnrMode::Standard,
        fixture.profile_noise_scale,
    )?)
}

fn plan(
    fixture: &CanonicalFixture,
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

fn noise_request(fixture: &CanonicalFixture) -> CompatibilityNoiseRequest {
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
fn val_sampler_001_definition_provenance_and_thin_family_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, EULER_CFG_PP_SAMPLER_ID);
    assert_eq!(fixture.feature_id, EULER_CFG_PP_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, EULER_CFG_PP_SOURCE_ORDINAL);
    assert_eq!(fixture.family_identity, "euler_ancestral_cfg_pp");
    assert_eq!(fixture.eta, 0.0);
    assert_eq!(fixture.noise_scale, 0.0);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 1);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(EULER_CFG_PP_SAMPLER_ID)?)?,
        &DEFINITION
    );
    let root = workspace_root()?;
    for (path, expected) in [
        (&fixture.source.sampling_path, &fixture.source.sampling_sha256),
        (&fixture.source.samplers_path, &fixture.source.samplers_sha256),
        (&fixture.source.catalog_path, &fixture.source.catalog_sha256),
        (&fixture.canonical_fixture_path, &fixture.canonical_fixture_sha256),
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let source = sampling
        .lines()
        .skip(fixture.source.equation_lines[0].saturating_sub(1))
        .take(fixture.source.equation_lines[1] - fixture.source.equation_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_euler_cfg_pp(",
        "return sample_euler_ancestral_cfg_pp(",
        "eta=0.0",
        "s_noise=0.0",
        "noise_sampler=None",
    ] {
        assert!(source.contains(fragment), "missing source {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"euler_cfg_pp\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,euler_cfg_pp,")
                && line.ends_with(",COMFY-MODEL-0182"))
    );
    assert!(IMPLEMENTATION.contains("sample_euler_cfg_pp_family("));
    assert!(IMPLEMENTATION.contains("eta: 0.0"));
    assert!(IMPLEMENTATION.contains("noise_scale: 0.0"));
    for forbidden in [
        "half_log_snr(",
        "standard_ancestral_step(",
        "draw_normal(",
        "SamplingSession::new",
        "observe_euler_denoised(",
        "tensor_to_f32(",
    ] {
        assert!(!IMPLEMENTATION.contains(forbidden), "duplicate family owner {forbidden}");
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_eta_zero_adapter_matches_every_family_intermediate()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let calls = RefCell::new(Vec::<Vec<f32>>::new());
    let callbacks = RefCell::new(Vec::<(SamplingProgress, Vec<f32>, Vec<f32>)>::new());
    let (trace, before, after) = sample_euler_cfg_pp(
        &backend,
        plan(&fixture, EULER_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        &context,
        |latent, sigma, step| {
            calls.borrow_mut().push(
                values(&backend, latent, &context).map_err(|error| error.to_string())?,
            );
            assert!((sigma - fixture.steps[step].sigma).abs() <= fixture.tolerance);
            Ok(EulerCfgPpDenoiserOutput {
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
    let expected_first = first
        .guided
        .iter()
        .zip(
            fixture
                .initial
                .iter()
                .zip(&first.unconditional)
                .map(|(current, unconditional)| (current - unconditional) / first.sigma),
        )
        .map(|(guided, derivative)| guided + first.next_sigma * derivative)
        .collect::<Vec<_>>();
    assert_close(&values(&backend, &trace.latents[1], &context)?, &expected_first, fixture.tolerance);
    assert_close(
        &values(&backend, trace.latents.last().ok_or("missing terminal")?, &context)?,
        &fixture.terminal,
        fixture.tolerance,
    );
    let calls = calls.into_inner();
    assert_close(&calls[0], &fixture.initial, fixture.tolerance);
    assert_close(&calls[1], &expected_first, fixture.tolerance);
    let callbacks = callbacks.into_inner();
    assert_close(&callbacks[0].1, &fixture.initial, fixture.tolerance);
    assert_close(&callbacks[0].2, &first.guided, fixture.tolerance);
    assert_close(&callbacks[1].1, &expected_first, fixture.tolerance);
    assert_close(&callbacks[1].2, &fixture.steps[1].guided, fixture.tolerance);
    for (step, (progress, _, _)) in callbacks.iter().enumerate() {
        assert_eq!(usize::try_from(progress.step)?, fixture.steps[step].step);
        assert!((progress.sigma - fixture.steps[step].sigma).abs() <= fixture.tolerance);
        assert!((progress.next_sigma - fixture.steps[step].next_sigma).abs() <= fixture.tolerance);
    }
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
fn val_rng_001_adapter_rejects_wrong_identity_and_pre_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let wrong = sample_euler_cfg_pp(
        &backend,
        plan(&fixture, "euler_ancestral_cfg_pp", &profile)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        &context,
        |latent, _, _| Ok(EulerCfgPpDenoiserOutput {
            denoised: latent.clone(),
            unconditional_denoised: latent.clone(),
        }),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("wrong identity must fail");
    assert!(matches!(wrong, EulerAncestralCfgPpError::WrongSampler { .. }));
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_euler_cfg_pp(
        &backend,
        plan(&fixture, EULER_CFG_PP_SAMPLER_ID, &profile)?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        &cancelled_context,
        |latent, _, _| Ok(EulerCfgPpDenoiserOutput {
            denoised: latent.clone(),
            unconditional_denoised: latent.clone(),
        }),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancellation must fail");
    assert!(matches!(error, EulerAncestralCfgPpError::Tensor(_)));
    Ok(())
}
