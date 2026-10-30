use sim_sampler::{
    BrownianNoiseIntervalAddress, CompatibilityNoiseRequest, DiscreteSamplingProfile,
    PredictionInterpretation, SamplerIdentity, SamplerRegistry, SamplingPlan, SamplingProfile,
    SamplingProfileIdentity,
    generated_dpmpp_3m_sde_sim_model_0174::{
        DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID, DPMPP_3M_SDE_SAMPLER_ID, Dpmpp3mSdeOptions,
        sample_dpmpp_3m_sde,
    },
    generated_dpmpp_3m_sde_gpu_sim_model_0175::{
        DEFINITION, DPMPP_3M_SDE_GPU_FEATURE_ID, DPMPP_3M_SDE_GPU_SAMPLER_ID,
        DPMPP_3M_SDE_GPU_SOURCE_ORDINAL, Dpmpp3mSdeGpuError, sample_dpmpp_3m_sde_gpu,
        validate_dpmpp_3m_sde_gpu_generation_device,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngCheckpoint, RngGenerationPlacement, RngSeedTransform, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_3m_sde_gpu_comfy_model_0175/trajectory.json"
));
const CANONICAL_FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_3m_sde_comfy_model_0174/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/dpmpp_3m_sde_gpu_sim_model_0175.rs");

#[derive(Debug, Deserialize)]
struct AdapterFixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    rng_contract_id: String,
    placement: String,
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
    equation_lines: Vec<usize>,
    registry_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct CanonicalFixture {
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
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    brownian_noise: Vec<f32>,
    latent_after: Vec<f32>,
}

struct OracleStep {
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    brownian_noise: Vec<f32>,
    latent_after: Vec<f32>,
}

struct FamilyOracle {
    steps: Vec<OracleStep>,
    terminal: Vec<f32>,
    checkpoints: (RngCheckpoint, RngCheckpoint),
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

fn profile() -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("dpmpp-3m-sde-gpu-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
    )?)
}

fn plan(
    identity: &str,
    profile: &DiscreteSamplingProfile,
    seed: u64,
    steps: usize,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile.identity().clone(),
        seed,
        u32::try_from(steps)?,
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

fn normalized_increment(
    tree: &mut sim_tensor::BrownianTree,
    sigma: f32,
    next_sigma: f32,
    step: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let address =
        BrownianNoiseIntervalAddress::new(sigma, next_sigma, u32::try_from(step)?)?;
    let (lower, upper) = address.canonical_interval();
    let normalization = f64::from(upper - lower).sqrt();
    let sign = if address.reverse { -1.0 } else { 1.0 };
    Ok(tree
        .increment(f64::from(lower), f64::from(upper), cancellation)?
        .into_iter()
        .map(|value| (value * sign / normalization) as f32)
        .collect())
}

fn family_oracle(
    fixture: &CanonicalFixture,
    profile: &impl SamplingProfile,
    placement: RngGenerationPlacement,
    cancellation: &CancellationToken,
) -> Result<FamilyOracle, Box<dyn Error>> {
    let mut transaction = noise_request(fixture).open_transaction(
        DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::TorchSigned64,
        placement,
        None,
        cancellation,
    )?;
    let before = transaction.checkpoint();
    let minimum = fixture
        .sigmas
        .iter()
        .copied()
        .filter(|sigma| *sigma > 0.0)
        .reduce(f32::min)
        .ok_or("Brownian minimum is unavailable")?;
    let maximum = fixture
        .sigmas
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or("Brownian maximum is unavailable")?;
    let mut tree = transaction.brownian_tree(
        f64::from(minimum),
        vec![0.0; fixture.initial.len()],
        f64::from(maximum),
        cancellation,
    )?;
    let after = transaction.commit();

    let mut sigmas = fixture.sigmas.clone();
    profile.adjust_first_sigma_for_snr(&mut sigmas)?;
    let effective_noise_scale = profile.scale_sampler_noise(fixture.noise_scale)?;
    let mut current = fixture.initial.clone();
    let mut denoised_1: Option<Vec<f32>> = None;
    let mut denoised_2: Option<Vec<f32>> = None;
    let mut step_size_1: Option<f32> = None;
    let mut step_size_2: Option<f32> = None;
    let mut steps = Vec::with_capacity(sigmas.len().saturating_sub(1));
    for (step, pair) in sigmas.windows(2).enumerate() {
        let sigma = pair[0];
        let next_sigma = pair[1];
        let latent_before = current.clone();
        let biases = [0.07_f32, -0.03, 0.05];
        let denoised = latent_before
            .iter()
            .zip(biases)
            .map(|(value, bias)| 0.61 * value + sigma * bias + step as f32 * 0.015)
            .collect::<Vec<_>>();
        let mut brownian_noise = Vec::new();
        let current_step_size = if next_sigma == 0.0 {
            current = denoised.clone();
            None
        } else {
            let lambda_source = profile.half_log_snr(sigma)?;
            let lambda_target = profile.half_log_snr(next_sigma)?;
            let step_size = lambda_target - lambda_source;
            let eta_step_size = step_size * (fixture.eta + 1.0);
            let alpha_target = next_sigma * lambda_target.exp();
            let latent_weight = next_sigma / sigma * (-step_size * fixture.eta).exp();
            let denoised_weight = alpha_target * -(-eta_step_size).exp_m1();
            let phi_2 = (-eta_step_size).exp_m1() / eta_step_size + 1.0;
            let phi_3 = phi_2 / eta_step_size - 0.5;
            let stochastic_scale = next_sigma
                * (-(-2.0 * step_size * fixture.eta).exp_m1()).sqrt()
                * effective_noise_scale;
            brownian_noise = normalized_increment(
                &mut tree,
                sigma,
                next_sigma,
                step,
                cancellation,
            )?;
            current = latent_before
                .iter()
                .zip(&denoised)
                .zip(&brownian_noise)
                .enumerate()
                .map(|(element, ((latent, denoised), noise))| {
                    let correction = match (
                        denoised_1.as_deref(),
                        denoised_2.as_deref(),
                        step_size_1,
                        step_size_2,
                    ) {
                        (Some(previous_1), Some(previous_2), Some(ratio_0), Some(ratio_1)) => {
                            let ratio_0 = ratio_0 / step_size;
                            let ratio_1 = ratio_1 / step_size;
                            let difference_0 = (denoised - previous_1[element]) / ratio_0;
                            let difference_1 =
                                (previous_1[element] - previous_2[element]) / ratio_1;
                            let ratio_sum = ratio_0 + ratio_1;
                            let first_derivative = difference_0
                                + (difference_0 - difference_1) * ratio_0 / ratio_sum;
                            let second_derivative = (difference_0 - difference_1) / ratio_sum;
                            alpha_target * phi_2 * first_derivative
                                - alpha_target * phi_3 * second_derivative
                        }
                        (Some(previous_1), None, Some(ratio), None) => {
                            alpha_target * phi_2 * (denoised - previous_1[element])
                                / (ratio / step_size)
                        }
                        (None, None, None, None) => 0.0,
                        _ => unreachable!("DPM++ 3M history is internally ordered"),
                    };
                    latent_weight * latent
                        + denoised_weight * denoised
                        + correction
                        + stochastic_scale * noise
                })
                .collect();
            Some(step_size)
        };
        steps.push(OracleStep {
            latent_before,
            denoised: denoised.clone(),
            brownian_noise,
            latent_after: current.clone(),
        });
        denoised_2 = denoised_1;
        denoised_1 = Some(denoised);
        step_size_2 = step_size_1;
        step_size_1 = current_step_size;
    }
    Ok(FamilyOracle {
        steps,
        terminal: current,
        checkpoints: (before, after),
    })
}

#[test]
fn val_sampler_001_dpmpp_3m_sde_gpu_definition_ordinal_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_3M_SDE_GPU_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_3M_SDE_GPU_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_3M_SDE_GPU_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, DPMPP_3M_SDE_BROWNIAN_CONTRACT_ID);
    assert_eq!(fixture.placement, "native-input-device");
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_3m_sde_gpu_comfy_model_0175"
    );
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(DPMPP_3M_SDE_GPU_SAMPLER_ID)?)?,
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
    assert_eq!(
        digest(&root.join(&fixture.canonical_fixture_path))?,
        fixture.canonical_fixture_sha256
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
        "def sample_dpmpp_3m_sde(",
        "if h_2 is not None:",
        "phi_3 = phi_2 / h_eta - 0.5",
        "def sample_dpmpp_3m_sde_gpu(",
        "cpu=False",
        "return sample_dpmpp_3m_sde(",
    ] {
        assert!(
            equations.contains(fragment),
            "missing source fragment {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    let names = pinned_ksampler_names(&samplers)?;
    assert_eq!(
        names
            .iter()
            .position(|identity| identity == DPMPP_3M_SDE_GPU_SAMPLER_ID),
        Some(usize::from(DPMPP_3M_SDE_GPU_SOURCE_ORDINAL))
    );
    assert_eq!(DPMPP_3M_SDE_GPU_SOURCE_ORDINAL, 24);
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_3m_sde_gpu\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,dpmpp_3m_sde_gpu,")
                && line.ends_with(",COMFY-MODEL-0175"))
    );

    assert!(IMPLEMENTATION.contains("sample_dpmpp_3m_sde_with_generation_placement("));
    assert!(IMPLEMENTATION.contains("RngGenerationPlacement::Native(device)"));
    assert!(IMPLEMENTATION.contains("CompatibilityNoiseRequest"));
    let short_guard = IMPLEMENTATION
        .find("if sigmas.len() > 1 {")
        .ok_or("missing source-order short-schedule guard")?;
    let device_validation = IMPLEMENTATION
        .find("validate_dpmpp_3m_sde_gpu_generation_device(device)?;")
        .ok_or("missing native-device validation")?;
    assert!(short_guard < device_validation);
    for forbidden in [
        "struct Dpmpp3mSdeGpuTrace",
        "struct Dpmpp3mSdeGpuProgress",
        "struct Dpmpp3mSdeGpuObservation",
        "struct Dpmpp3mSdeGpuNoiseRequest",
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "RngStream",
        "BrownianTree",
        "half_log_snr(",
        "phi_2",
        "phi_3",
        "exp_m1()",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_gpu_adapter_matches_every_canonical_intermediate_callback_and_rng()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let oracle = family_oracle(
        &fixture,
        &profile,
        RngGenerationPlacement::Native(DeviceId::CPU),
        &cancellation,
    )?;
    let callbacks = RefCell::new(Vec::new());
    let (trace, checkpoints) = sample_dpmpp_3m_sde_gpu(
        &backend,
        plan(
            DPMPP_3M_SDE_GPU_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &context,
        |latent, sigma, step| {
            let expected = oracle
                .steps
                .get(step)
                .ok_or_else(|| "unexpected denoiser step".to_owned())?;
            assert!((sigma - fixture.sigmas[step]).abs() <= fixture.tolerance);
            let latent =
                tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
            assert_close(&latent, &expected.latent_before, fixture.tolerance);
            let biases = [0.07_f32, -0.03, 0.05];
            let denoised = latent
                .iter()
                .zip(biases)
                .map(|(value, bias)| 0.61 * value + sigma * bias + step as f32 * 0.015)
                .collect::<Vec<_>>();
            assert_close(&denoised, &expected.denoised, fixture.tolerance);
            tensor_from_f32(&backend, &fixture.shape, &denoised, &context)
                .map_err(|error| error.to_string())
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
    assert_eq!(trace.latents.len(), oracle.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), oracle.steps.len());
    for (step, expected) in oracle.steps.iter().enumerate() {
        assert_close(
            &values(&backend, &trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &trace.denoiser_evaluations[step], &context)?,
            &expected.denoised,
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
        &oracle.terminal,
        fixture.tolerance,
    );
    let callbacks = callbacks.into_inner();
    assert_eq!(callbacks.len(), oracle.steps.len());
    for (step, (progress, latent, denoised)) in callbacks.iter().enumerate() {
        let expected = &oracle.steps[step];
        assert_eq!(usize::try_from(progress.step)?, step);
        assert!((progress.sigma - fixture.sigmas[step]).abs() <= fixture.tolerance);
        assert!((progress.next_sigma - fixture.sigmas[step + 1]).abs() <= fixture.tolerance);
        assert_close(latent, &expected.latent_before, fixture.tolerance);
        assert_close(denoised, &expected.denoised, fixture.tolerance);
    }

    assert_eq!(
        checkpoints.ok_or("missing Brownian checkpoints")?,
        oracle.checkpoints
    );
    Ok(())
}

#[test]
fn val_sampling_foundation_001_adapter_is_exactly_the_crate_private_3m_family()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let denoise = |latent: &Tensor, sigma: f32, step: usize| {
        let latent =
            tensor_to_f32(&backend, latent, &context).map_err(|error| error.to_string())?;
        let biases = [0.07_f32, -0.03, 0.05];
        let denoised = latent
            .iter()
            .zip(biases)
            .map(|(value, bias)| 0.61 * value + sigma * bias + step as f32 * 0.015)
            .collect::<Vec<_>>();
        tensor_from_f32(&backend, &fixture.shape, &denoised, &context)
            .map_err(|error| error.to_string())
    };
    let native_oracle = family_oracle(
        &fixture,
        &profile,
        RngGenerationPlacement::Native(DeviceId::CPU),
        &cancellation,
    )?;
    let cpu_oracle = family_oracle(
        &fixture,
        &profile,
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        &cancellation,
    )?;
    let (adapter, adapter_checkpoints) = sample_dpmpp_3m_sde_gpu(
        &backend,
        plan(
            DPMPP_3M_SDE_GPU_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &context,
        denoise,
        |_, _, _| Ok::<_, String>(()),
    )?;
    let (family, family_checkpoints) = sample_dpmpp_3m_sde(
        &backend,
        plan(
            DPMPP_3M_SDE_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        Dpmpp3mSdeOptions::new(fixture.eta, fixture.noise_scale)?,
        noise_request(&fixture),
        &context,
        denoise,
        |_, _, _| Ok::<_, String>(()),
    )?;
    assert_eq!(adapter_checkpoints, Some(native_oracle.checkpoints.clone()));
    assert_eq!(family_checkpoints, Some(cpu_oracle.checkpoints.clone()));
    assert_ne!(
        native_oracle.checkpoints.0.address_digest,
        cpu_oracle.checkpoints.0.address_digest,
        "native and CPU-transfer ABI addresses must remain distinct"
    );
    assert_eq!(adapter.sigmas, family.sigmas);
    assert_eq!(adapter.latents.len(), native_oracle.steps.len() + 1);
    assert_eq!(family.latents.len(), cpu_oracle.steps.len() + 1);
    for (step, expected) in native_oracle.steps.iter().enumerate() {
        assert_close(
            &values(&backend, &adapter.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &adapter.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
    }
    for (step, expected) in cpu_oracle.steps.iter().enumerate() {
        assert!((fixture.sigmas[step] - fixture.steps[step].sigma).abs() <= fixture.tolerance);
        assert!(
            (fixture.sigmas[step + 1] - fixture.steps[step].next_sigma).abs()
                <= fixture.tolerance
        );
        assert_close(
            &values(&backend, &family.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &family.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
        assert_close(
            &expected.latent_before,
            &fixture.steps[step].latent_before,
            fixture.tolerance,
        );
        assert_close(
            &expected.latent_after,
            &fixture.steps[step].latent_after,
            fixture.tolerance,
        );
        assert_close(
            &expected.denoised,
            &fixture.steps[step].denoised,
            fixture.tolerance,
        );
        assert_close(
            &expected.brownian_noise,
            &fixture.steps[step].brownian_noise,
            fixture.tolerance,
        );
    }
    assert_close(&cpu_oracle.terminal, &fixture.terminal, fixture.tolerance);
    Ok(())
}

#[test]
fn val_rng_001_gpu_adapter_rejects_wrong_identity_invalid_options_and_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let cuda = DeviceId::from_source_device("cuda:0")?;
    let error = validate_dpmpp_3m_sde_gpu_generation_device(cuda)
        .expect_err("unavailable CUDA must fail closed through the backend capability owner");
    assert!(matches!(
        error,
        Dpmpp3mSdeGpuError::DeviceUnavailable { device, .. } if device == cuda
    ));
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let (short, checkpoints) = sample_dpmpp_3m_sde_gpu(
        &backend,
        plan(DPMPP_3M_SDE_GPU_SAMPLER_ID, &profile, fixture.seed, 1)?,
        &profile,
        initial.clone(),
        &[fixture.sigmas[0]],
        noise_request(&fixture),
        f32::NAN,
        f32::NAN,
        &context,
        |_, _, _| Err("short schedule must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("short schedule must not callback"),
    )?;
    assert_eq!(short.latents.len(), 1);
    assert!(checkpoints.is_none());
    let error = sample_dpmpp_3m_sde_gpu(
        &backend,
        plan(
            DPMPP_3M_SDE_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("a distinct registered identity must not be substituted");
    assert!(matches!(error, Dpmpp3mSdeGpuError::WrongSampler(_)));

    let error = sample_dpmpp_3m_sde_gpu(
        &backend,
        plan(
            DPMPP_3M_SDE_GPU_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        noise_request(&fixture),
        f32::NAN,
        fixture.noise_scale,
        &context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("invalid equation options must fail closed");
    assert!(matches!(error, Dpmpp3mSdeGpuError::EquationFamily(_)));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_dpmpp_3m_sde_gpu(
        &backend,
        plan(
            DPMPP_3M_SDE_GPU_SAMPLER_ID,
            &profile,
            fixture.seed,
            fixture.steps.len(),
        )?,
        &profile,
        initial,
        &fixture.sigmas,
        noise_request(&fixture),
        fixture.eta,
        fixture.noise_scale,
        &cancelled_context,
        |latent, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled execution must fail before RNG publication");
    assert!(matches!(error, Dpmpp3mSdeGpuError::Tensor(_)));
    Ok(())
}
