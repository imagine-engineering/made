use sim_sampler::{
    BrownianNoiseIntervalAddress, CompatibilityNoiseRequest, DiscreteSamplingProfile,
    PredictionInterpretation, SamplerIdentity, SamplerRegistry, SamplingPlan, SamplingProfile,
    SamplingProfileIdentity, SamplingProgress,
    generated_dpmpp_sde_sim_model_0176::{
        DPMPP_SDE_BROWNIAN_CONTRACT_ID, DPMPP_SDE_SAMPLER_ID, DpmppSdeDenoiserStage,
        DpmppSdeOptions, sample_dpmpp_sde,
    },
    generated_dpmpp_sde_gpu_sim_model_0177::{
        DEFINITION, DPMPP_SDE_GPU_FEATURE_ID, DPMPP_SDE_GPU_SAMPLER_ID,
        DPMPP_SDE_GPU_SOURCE_ORDINAL, DpmppSdeGpuError, sample_dpmpp_sde_gpu,
        validate_dpmpp_sde_gpu_generation_device,
    },
};
use sim_tensor::{
    BrownianTree, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngCheckpoint, RngGenerationPlacement, RngSeedTransform, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_sde_gpu_comfy_model_0177/trajectory.json"
));
const CANONICAL_FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_sde_comfy_model_0176/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/dpmpp_sde_gpu_sim_model_0177.rs");

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
    shape: Vec<u64>,
    initial: Vec<f32>,
    seed: u64,
    eta: f32,
    noise_scale: f32,
    r: f32,
    rng: RngFixture,
    profile: String,
    sigmas: Vec<f32>,
    primary_denoised: Vec<Vec<f32>>,
    intermediate_denoised: Vec<Vec<f32>>,
    steps: Vec<StepFixture>,
    tolerance: f32,
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
    sigma_s_1: Option<f32>,
    intermediate_input: Option<Vec<f32>>,
    latent_after: Vec<f32>,
}

#[derive(Debug)]
struct DenoiserCall {
    step: usize,
    stage: DpmppSdeDenoiserStage,
    sigma: f32,
    latent: Vec<f32>,
}

#[derive(Debug)]
struct CallbackCall {
    progress: SamplingProgress,
    latent: Vec<f32>,
    denoised: Vec<f32>,
}

struct OracleStep {
    latent_before: Vec<f32>,
    intermediate_sigma: Option<f32>,
    intermediate_input: Option<Vec<f32>>,
    latent_after: Vec<f32>,
}

struct FamilyOracle {
    steps: Vec<OracleStep>,
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

fn profile(fixture: &CanonicalFixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new(fixture.profile.clone())?,
        PredictionInterpretation::Epsilon,
        Arc::from([0.01_f32, 0.1, 0.5, 1.0, 2.0]),
    )?)
}

fn plan(
    identity: &str,
    profile: &DiscreteSamplingProfile,
    fixture: &CanonicalFixture,
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
            "element {element}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

fn assert_checkpoint_placement_invariants(actual: &RngCheckpoint, expected: &RngCheckpoint) {
    assert_eq!(actual.profile, expected.profile);
    assert_eq!(actual.algorithm, expected.algorithm);
    assert_eq!(actual.device, expected.device);
}

fn ancestral_step(from: f32, to: f32, eta: f32) -> (f32, f32) {
    if eta == 0.0 {
        return (to, 0.0);
    }
    let from_squared = from * from;
    let to_squared = to * to;
    let up = to.min(eta * (to_squared * (from_squared - to_squared) / from_squared).sqrt());
    ((to_squared - up * up).sqrt(), up)
}

fn normalized_increment(
    tree: &mut BrownianTree,
    start: f32,
    end: f32,
    step: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let address = BrownianNoiseIntervalAddress::new(start, end, u32::try_from(step)?)?;
    let (lower, upper) = address.canonical_interval();
    let sign = if address.reverse { -1.0_f64 } else { 1.0 };
    let normalization = f64::from(upper - lower).sqrt();
    Ok(tree
        .increment(f64::from(lower), f64::from(upper), cancellation)?
        .into_iter()
        .map(|value| (sign * value / normalization) as f32)
        .collect())
}

fn family_oracle(
    fixture: &CanonicalFixture,
    profile: &impl SamplingProfile,
    placement: RngGenerationPlacement,
    cancellation: &CancellationToken,
) -> Result<FamilyOracle, Box<dyn Error>> {
    let mut transaction = noise_request(fixture).open_transaction(
        DPMPP_SDE_BROWNIAN_CONTRACT_ID,
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
    let mut sigmas = fixture.sigmas.clone();
    profile.adjust_first_sigma_for_snr(&mut sigmas)?;
    let effective_noise_scale = profile.scale_sampler_noise(fixture.noise_scale)?;
    let mut current = fixture.initial.clone();
    let mut steps = Vec::with_capacity(fixture.steps.len());
    for (step, pair) in sigmas.windows(2).enumerate() {
        let sigma = pair[0];
        let next_sigma = pair[1];
        let latent_before = current.clone();
        let primary = &fixture.primary_denoised[step];
        if next_sigma == 0.0 {
            current = primary.clone();
            steps.push(OracleStep {
                latent_before,
                intermediate_sigma: None,
                intermediate_input: None,
                latent_after: current.clone(),
            });
            continue;
        }

        let lambda_source = profile.half_log_snr(sigma)?;
        let lambda_target = profile.half_log_snr(next_sigma)?;
        let intermediate_lambda = lambda_source + fixture.r * (lambda_target - lambda_source);
        let combination_factor = 1.0 / (2.0 * fixture.r);
        let intermediate_sigma = profile.sigma_from_half_log_snr(intermediate_lambda)?;
        let alpha_source = sigma * lambda_source.exp();
        let alpha_intermediate = intermediate_sigma * intermediate_lambda.exp();
        let alpha_target = next_sigma * lambda_target.exp();

        let (first_down, first_up) = ancestral_step(
            (-lambda_source).exp(),
            (-intermediate_lambda).exp(),
            fixture.eta,
        );
        let first_step = -first_down.ln() - lambda_source;
        let first_latent_weight = alpha_intermediate / alpha_source * (-first_step).exp();
        let first_denoised_weight = -alpha_intermediate * (-first_step).exp_m1();
        let mut intermediate_input = latent_before
            .iter()
            .zip(primary)
            .map(|(latent, denoised)| {
                first_latent_weight * latent + first_denoised_weight * denoised
            })
            .collect::<Vec<_>>();
        if fixture.eta > 0.0 && effective_noise_scale > 0.0 {
            let noise =
                normalized_increment(&mut tree, sigma, intermediate_sigma, step, cancellation)?;
            let scale = alpha_intermediate * effective_noise_scale * first_up;
            for (value, noise) in intermediate_input.iter_mut().zip(noise) {
                *value += scale * noise;
            }
        }

        let intermediate = &fixture.intermediate_denoised[step];
        let combined = primary
            .iter()
            .zip(intermediate)
            .map(|(primary, intermediate)| {
                (1.0 - combination_factor) * primary + combination_factor * intermediate
            })
            .collect::<Vec<_>>();
        let (second_down, second_up) =
            ancestral_step((-lambda_source).exp(), (-lambda_target).exp(), fixture.eta);
        let second_step = -second_down.ln() - lambda_source;
        let second_latent_weight = alpha_target / alpha_source * (-second_step).exp();
        let second_denoised_weight = -alpha_target * (-second_step).exp_m1();
        current = latent_before
            .iter()
            .zip(&combined)
            .map(|(latent, denoised)| {
                second_latent_weight * latent + second_denoised_weight * denoised
            })
            .collect();
        if fixture.eta > 0.0 && effective_noise_scale > 0.0 {
            let noise = normalized_increment(&mut tree, sigma, next_sigma, step, cancellation)?;
            let scale = alpha_target * effective_noise_scale * second_up;
            for (value, noise) in current.iter_mut().zip(noise) {
                *value += scale * noise;
            }
        }
        steps.push(OracleStep {
            latent_before,
            intermediate_sigma: Some(intermediate_sigma),
            intermediate_input: Some(intermediate_input),
            latent_after: current.clone(),
        });
    }
    let after = transaction.commit();
    Ok(FamilyOracle {
        steps,
        checkpoints: (before, after),
    })
}

#[test]
fn val_sampler_001_dpmpp_sde_gpu_definition_ordinal_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_SDE_GPU_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_SDE_GPU_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_SDE_GPU_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, DPMPP_SDE_BROWNIAN_CONTRACT_ID);
    assert_eq!(fixture.placement, "native-input-device");
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 16);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_sde_gpu_comfy_model_0177"
    );
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(DPMPP_SDE_GPU_SAMPLER_ID)?)?,
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
        (
            &fixture.canonical_fixture_path,
            &fixture.canonical_fixture_sha256,
        ),
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_dpmpp_sde(",
        "cpu=True",
        "lambda_s_1",
        "denoised_2",
        "denoised_d",
        "def sample_dpmpp_sde_gpu(",
        "cpu=False",
        "return sample_dpmpp_sde(",
    ] {
        assert!(equations.contains(fragment), "missing source {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    let names = pinned_ksampler_names(&samplers)?;
    let ordinal = names
        .iter()
        .position(|identity| identity == DPMPP_SDE_GPU_SAMPLER_ID)
        .ok_or("dpmpp_sde_gpu is absent")?;
    assert_eq!(u16::try_from(ordinal)?, DPMPP_SDE_GPU_SOURCE_ORDINAL);
    assert_eq!(
        names.get(ordinal.saturating_sub(1)).map(String::as_str),
        Some(DPMPP_SDE_SAMPLER_ID)
    );
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_sde_gpu\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("sampler,dpmpp_sde_gpu,")
                && line.ends_with(",COMFY-MODEL-0177"))
    );

    assert!(IMPLEMENTATION.contains("sample_dpmpp_sde_with_generation_placement("));
    assert!(IMPLEMENTATION.contains("RngGenerationPlacement::Native(device)"));
    assert!(!IMPLEMENTATION.contains("pub fn sample_dpmpp_sde_with_generation_placement"));
    let short_guard = IMPLEMENTATION
        .find("if sigmas.len() > 1 {")
        .ok_or("missing source-order short-schedule guard")?;
    let device_validation = IMPLEMENTATION
        .find("validate_dpmpp_sde_gpu_generation_device(device)?;")
        .ok_or("missing native-device validation")?;
    assert!(short_guard < device_validation);
    for forbidden in [
        "CompatibilityRngTransaction",
        "RngCompatibilityRequest",
        "RngStreamAddress",
        "BrownianTree",
        "BrownianNoiseIntervalAddress",
        "half_log_snr(",
        "ancestral_step(",
        "exp_m1()",
        "SamplingSession",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_gpu_adapter_matches_every_family_intermediate_callback_and_checkpoint()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile(&fixture)?;
    let options = DpmppSdeOptions {
        eta: fixture.eta,
        noise_scale: fixture.noise_scale,
        r: fixture.r,
    };
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let adapter_calls = RefCell::new(Vec::new());
    let adapter_callbacks = RefCell::new(Vec::new());
    let (adapter_trace, adapter_checkpoints) = sample_dpmpp_sde_gpu(
        &backend,
        plan(DPMPP_SDE_GPU_SAMPLER_ID, &profile, &fixture)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        options,
        noise_request(&fixture),
        &context,
        |latent, sigma, step, stage| {
            adapter_calls.borrow_mut().push(DenoiserCall {
                step,
                stage,
                sigma,
                latent: tensor_to_f32(&backend, latent, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
            });
            let output = match stage {
                DpmppSdeDenoiserStage::Primary => &fixture.primary_denoised[step],
                DpmppSdeDenoiserStage::Intermediate => &fixture.intermediate_denoised[step],
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            adapter_callbacks.borrow_mut().push(CallbackCall {
                progress: *progress,
                latent: tensor_to_f32(&backend, latent, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
                denoised: tensor_to_f32(&backend, denoised, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
            });
            Ok::<_, String>(())
        },
    )?;

    let family_calls = RefCell::new(Vec::new());
    let family_callbacks = RefCell::new(Vec::new());
    let (family_trace, family_checkpoints) = sample_dpmpp_sde(
        &backend,
        plan(DPMPP_SDE_SAMPLER_ID, &profile, &fixture)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        options,
        noise_request(&fixture),
        &context,
        |latent, sigma, step, stage| {
            family_calls.borrow_mut().push(DenoiserCall {
                step,
                stage,
                sigma,
                latent: tensor_to_f32(&backend, latent, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
            });
            let output = match stage {
                DpmppSdeDenoiserStage::Primary => &fixture.primary_denoised[step],
                DpmppSdeDenoiserStage::Intermediate => &fixture.intermediate_denoised[step],
            };
            tensor_from_f32(&backend, &fixture.shape, output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            family_callbacks.borrow_mut().push(CallbackCall {
                progress: *progress,
                latent: tensor_to_f32(&backend, latent, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
                denoised: tensor_to_f32(&backend, denoised, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec(),
            });
            Ok::<_, String>(())
        },
    )?;

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
    assert_eq!(adapter_trace.sigmas, family_trace.sigmas);
    assert_eq!(adapter_trace.latents.len(), native_oracle.steps.len() + 1);
    assert_eq!(family_trace.latents.len(), cpu_oracle.steps.len() + 1);
    assert_eq!(
        adapter_trace.denoiser_evaluations.len(),
        family_trace.denoiser_evaluations.len()
    );
    for (step, expected) in native_oracle.steps.iter().enumerate() {
        assert_close(
            &values(&backend, &adapter_trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &adapter_trace.latents[step + 1], &context)?,
            &expected.latent_after,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                &adapter_trace.denoiser_evaluations[step],
                &context,
            )?,
            &fixture.primary_denoised[step],
            fixture.tolerance,
        );
    }
    for (step, expected) in cpu_oracle.steps.iter().enumerate() {
        assert_close(
            &values(&backend, &family_trace.latents[step], &context)?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(&backend, &family_trace.latents[step + 1], &context)?,
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
        if let Some(expected_intermediate) = expected.intermediate_input.as_deref() {
            assert!(
                (expected
                    .intermediate_sigma
                    .ok_or("oracle intermediate sigma is missing")?
                    - fixture.steps[step]
                        .sigma_s_1
                        .ok_or("canonical intermediate sigma is missing")?)
                .abs()
                    <= fixture.tolerance
            );
            assert_close(
                expected_intermediate,
                fixture.steps[step]
                    .intermediate_input
                    .as_deref()
                    .ok_or("canonical fixture intermediate is missing")?,
                fixture.tolerance,
            );
        }
    }

    let adapter_calls = adapter_calls.into_inner();
    let family_calls = family_calls.into_inner();
    assert_eq!(adapter_calls.len(), family_calls.len());
    for (adapter, family) in adapter_calls.iter().zip(&family_calls) {
        assert_eq!(adapter.step, family.step);
        assert_eq!(adapter.stage, family.stage);
        assert!((adapter.sigma - family.sigma).abs() <= fixture.tolerance);
        let native_expected = &native_oracle.steps[adapter.step];
        let cpu_expected = &cpu_oracle.steps[family.step];
        match adapter.stage {
            DpmppSdeDenoiserStage::Primary => {
                assert!(
                    (adapter.sigma - fixture.steps[adapter.step].sigma).abs() <= fixture.tolerance
                );
                assert_close(
                    &adapter.latent,
                    &native_expected.latent_before,
                    fixture.tolerance,
                );
                assert_close(
                    &family.latent,
                    &cpu_expected.latent_before,
                    fixture.tolerance,
                );
            }
            DpmppSdeDenoiserStage::Intermediate => {
                assert!(
                    (adapter.sigma
                        - native_expected
                            .intermediate_sigma
                            .ok_or("native intermediate sigma is missing")?)
                    .abs()
                        <= fixture.tolerance
                );
                assert_close(
                    &adapter.latent,
                    native_expected
                        .intermediate_input
                        .as_deref()
                        .ok_or("native intermediate is missing")?,
                    fixture.tolerance,
                );
                assert_close(
                    &family.latent,
                    cpu_expected
                        .intermediate_input
                        .as_deref()
                        .ok_or("CPU intermediate is missing")?,
                    fixture.tolerance,
                );
            }
        }
    }

    let adapter_callbacks = adapter_callbacks.into_inner();
    let family_callbacks = family_callbacks.into_inner();
    assert_eq!(adapter_callbacks.len(), family_callbacks.len());
    for (step, (adapter, family)) in adapter_callbacks.iter().zip(&family_callbacks).enumerate() {
        assert_eq!(adapter.progress, family.progress);
        assert!(
            (adapter.progress.next_sigma - fixture.steps[step].next_sigma).abs()
                <= fixture.tolerance
        );
        assert_close(
            &adapter.latent,
            &native_oracle.steps[step].latent_before,
            fixture.tolerance,
        );
        assert_close(
            &family.latent,
            &cpu_oracle.steps[step].latent_before,
            fixture.tolerance,
        );
        assert_close(&adapter.denoised, &family.denoised, fixture.tolerance);
        assert_close(
            &adapter.denoised,
            &fixture.primary_denoised[step],
            fixture.tolerance,
        );
    }

    let adapter_checkpoints = adapter_checkpoints.ok_or("adapter checkpoints are missing")?;
    let family_checkpoints = family_checkpoints.ok_or("family checkpoints are missing")?;
    assert_eq!(adapter_checkpoints, native_oracle.checkpoints);
    assert_eq!(family_checkpoints, cpu_oracle.checkpoints);
    assert_checkpoint_placement_invariants(&adapter_checkpoints.0, &family_checkpoints.0);
    assert_checkpoint_placement_invariants(&adapter_checkpoints.1, &family_checkpoints.1);
    assert_ne!(
        adapter_checkpoints.0.address_digest, family_checkpoints.0.address_digest,
        "native and CPU-transfer ABI addresses must remain distinct"
    );
    Ok(())
}

#[test]
fn val_sampling_foundation_001_gpu_adapter_rejects_wrong_identity_options_and_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = canonical_fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let cuda = DeviceId::from_source_device("cuda:0")?;
    let unavailable = validate_dpmpp_sde_gpu_generation_device(cuda)
        .expect_err("unavailable CUDA must fail closed through the backend capability owner");
    assert!(matches!(
        unavailable,
        DpmppSdeGpuError::DeviceUnavailable { device, .. } if device == cuda
    ));
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;

    let short_plan = SamplingPlan::new(
        DPMPP_SDE_GPU_SAMPLER_ID,
        "normal",
        profile.identity().clone(),
        fixture.seed,
        1,
        1.0,
        1.0,
    )?;
    let (short, checkpoints) = sample_dpmpp_sde_gpu(
        &backend,
        short_plan,
        &profile,
        initial.clone(),
        &[fixture.sigmas[0]],
        DpmppSdeOptions {
            eta: f32::NAN,
            noise_scale: f32::NAN,
            r: f32::NAN,
        },
        noise_request(&fixture),
        &context,
        |_, _, _, _| Err("short schedule must not denoise".to_owned()),
        |_, _, _| Err::<(), _>("short schedule must not callback"),
    )?;
    assert_eq!(short.latents.len(), 1);
    assert!(checkpoints.is_none());

    let wrong_identity = sample_dpmpp_sde_gpu(
        &backend,
        plan(DPMPP_SDE_SAMPLER_ID, &profile, &fixture)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        DpmppSdeOptions::default(),
        noise_request(&fixture),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("canonical identity must not substitute for GPU identity");
    assert!(matches!(wrong_identity, DpmppSdeGpuError::WrongSampler(_)));

    let invalid_options = sample_dpmpp_sde_gpu(
        &backend,
        plan(DPMPP_SDE_GPU_SAMPLER_ID, &profile, &fixture)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        DpmppSdeOptions {
            eta: f32::NAN,
            ..DpmppSdeOptions::default()
        },
        noise_request(&fixture),
        &context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("invalid family options must fail closed");
    assert!(matches!(
        invalid_options,
        DpmppSdeGpuError::EquationFamily(_)
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancelled_error = sample_dpmpp_sde_gpu(
        &backend,
        plan(DPMPP_SDE_GPU_SAMPLER_ID, &profile, &fixture)?,
        &profile,
        initial,
        &fixture.sigmas,
        DpmppSdeOptions::default(),
        noise_request(&fixture),
        &cancelled_context,
        |latent, _, _, _| Ok(latent.clone()),
        |_, _, _| Ok::<_, String>(()),
    )
    .expect_err("pre-cancelled adapter must fail before the family transaction opens");
    assert!(matches!(cancelled_error, DpmppSdeGpuError::Tensor(_)));
    Ok(())
}
