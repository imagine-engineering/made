use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    generated_ipndm_v_sim_model_0190::{
        DEFINITION, IPNDM_V_FEATURE_ID, IPNDM_V_MAX_ORDER, IPNDM_V_SAMPLER_ID,
        IPNDM_V_SOURCE_ORDINAL, IpndmVError, sample_ipndm_v,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/ipndm_v_comfy_model_0190/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/ipndm_v_sim_model_0190.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    max_order: usize,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
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
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    derivative: Vec<f32>,
    order: usize,
    coefficients: Vec<f32>,
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

fn profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("ipndm-v-row-v1")?)
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
fn val_sampler_001_definition_source_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, IPNDM_V_SAMPLER_ID);
    assert_eq!(fixture.feature_id, IPNDM_V_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, IPNDM_V_SOURCE_ORDINAL);
    assert_eq!(fixture.max_order, IPNDM_V_MAX_ORDER);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 28);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/ipndm_v_comfy_model_0190"
    );
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(IPNDM_V_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new("ipndm-variable")?)
            .is_err()
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
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_ipndm_v(",
        "callback({'x': x",
        "d_cur = (x_cur - denoised) / t_cur",
        "order = min(max_order, i+1)",
        "x_next = denoised",
        "coeff1 = (2 + (h_n / h_n_1)) / 2",
        "coeff3 = temp * h_n_1 / h_n_2",
        "coeff4 = -temp2",
        "buffer_model[-1] = d_cur.detach()",
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
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"ipndm_v\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(
                |line| line.starts_with("sampler,ipndm_v,") && line.ends_with(",COMFY-MODEL-0190")
            )
    );
    for forbidden in [
        "struct SamplingSession",
        "struct CancellationToken",
        "struct RngStream",
        "CpuWorkspaceAuthority",
        "authorize_workspace",
        "fn validate_sigmas",
        "unsafe {",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_every_order_intermediate_and_callback()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let denoiser_inputs = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let trace = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, 190, fixture.steps.len())?,
        &profile()?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        &context,
        |latent, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            assert_eq!(sigma.to_bits(), expected.sigma.to_bits());
            denoiser_inputs
                .borrow_mut()
                .push(values(&backend, latent, &context).map_err(|error| error.to_string())?);
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, callback_latent, denoised| {
            callbacks.borrow_mut().push((
                *progress,
                values(&backend, callback_latent, &context).map_err(|error| error.to_string())?,
                values(&backend, denoised, &context).map_err(|error| error.to_string())?,
            ));
            Ok::<_, String>(())
        },
    )?;

    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    let denoiser_inputs = denoiser_inputs.into_inner();
    let callbacks = callbacks.into_inner();
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_eq!(expected.order, IPNDM_V_MAX_ORDER.min(step + 1));
        assert_close(
            denoiser_inputs.get(step).ok_or("missing denoiser input")?,
            &expected.latent_before,
            fixture.tolerance,
        );
        let callback = callbacks.get(step).ok_or("missing callback")?;
        assert_eq!(usize::try_from(callback.0.step)?, step);
        assert_eq!(callback.0.sigma.to_bits(), expected.sigma.to_bits());
        assert_eq!(callback.0.sigma_hat.to_bits(), expected.sigma.to_bits());
        assert_eq!(
            callback.0.next_sigma.to_bits(),
            expected.next_sigma.to_bits()
        );
        assert_close(&callback.1, &fixture.initial, fixture.tolerance);
        assert_close(&callback.2, &expected.denoised, fixture.tolerance);
        let derivative = expected
            .latent_before
            .iter()
            .zip(&expected.denoised)
            .map(|(latent, denoised)| (latent - denoised) / expected.sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &expected.derivative, fixture.tolerance);
        if expected.next_sigma == 0.0 {
            assert!(expected.coefficients.is_empty());
            assert_close(
                &expected.latent_after,
                &expected.denoised,
                fixture.tolerance,
            );
        } else {
            assert_eq!(expected.coefficients.len(), expected.order);
            assert!(expected.coefficients.iter().all(|value| value.is_finite()));
        }
        assert_close(
            &values(
                &backend,
                trace.latents.get(step + 1).ok_or("missing latent")?,
                &context,
            )?,
            &expected.latent_after,
            fixture.tolerance,
        );
    }
    assert_close(
        &values(
            &backend,
            trace.latents.last().ok_or("missing terminal latent")?,
            &context,
        )?,
        &fixture.terminal,
        fixture.tolerance,
    );
    Ok(())
}

fn run_terminal(seed: u64) -> Result<Vec<f32>, Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let trace = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, seed, fixture.steps.len())?,
        &profile()?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        &context,
        |_latent, _sigma, step| {
            let denoised = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("missing step {step}"))?;
            tensor_from_f32(&backend, &fixture.shape, &denoised.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<_, String>(()),
    )?;
    values(
        &backend,
        trace.latents.last().ok_or("missing terminal latent")?,
        &context,
    )
}

#[test]
fn val_rng_001_is_deterministic_and_seed_independent() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let first = run_terminal(1)?;
    let second = run_terminal(u64::MAX)?;
    assert_close(&first, &second, fixture.tolerance);
    assert_close(&first, &fixture.terminal, fixture.tolerance);
    Ok(())
}

#[test]
fn failures_cancellation_and_callback_commit_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();

    let wrong = sample_ipndm_v(
        &backend,
        plan("heun", 0, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_latent, _sigma, _step| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<_, String>(()),
    );
    assert!(matches!(wrong, Err(IpndmVError::WrongSampler { .. })));

    let failed = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, 0, fixture.steps.len())?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_latent, _sigma, step| Err(format!("failure-{step}")),
        |_progress, _latent, _denoised| Ok::<_, String>(()),
    );
    assert!(matches!(failed, Err(IpndmVError::Denoiser { step: 0, .. })));

    let callback_failed = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, 0, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        &context,
        |_latent, _sigma, step| {
            let denoised = fixture.steps.get(step).ok_or("missing step")?;
            tensor_from_f32(&backend, &fixture.shape, &denoised.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Err::<(), _>("callback-failed"),
    );
    assert!(matches!(
        callback_failed,
        Err(IpndmVError::Sampling(SamplingError::Callback(reason)))
            if reason == "callback-failed"
    ));
    assert_close(
        &values(&backend, &initial_alias, &context)?,
        &fixture.initial,
        fixture.tolerance,
    );

    let allocation_cancellation = CancellationToken::default();
    let allocation_context = execution_context(&backend, &authority, &allocation_cancellation)?;
    let cancelled_initial = tensor_from_f32(
        &backend,
        &fixture.shape,
        &fixture.initial,
        &allocation_context,
    )?;
    assert!(cancellation.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancellation)?;
    let cancelled = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, 0, fixture.steps.len())?,
        &profile()?,
        cancelled_initial,
        &fixture.sigmas,
        &cancelled_context,
        |_latent, _sigma, _step| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<_, String>(()),
    );
    assert!(matches!(
        cancelled,
        Err(IpndmVError::Tensor(TensorError::Cancelled))
    ));

    let mismatch_cancellation = CancellationToken::default();
    let mismatch_context = execution_context(&backend, &authority, &mismatch_cancellation)?;
    let mismatch = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, 0, fixture.steps.len())?,
        &profile()?,
        tensor_from_f32(
            &backend,
            &fixture.shape,
            &fixture.initial,
            &mismatch_context,
        )?,
        &fixture.sigmas,
        &mismatch_context,
        |_latent, _sigma, _step| {
            tensor_from_f32(&backend, &[1], &[0.0], &mismatch_context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<_, String>(()),
    );
    assert!(matches!(
        mismatch,
        Err(IpndmVError::DenoiserContract { step: 0 })
    ));
    Ok(())
}

#[test]
fn input_and_history_are_immutable_aliases() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let alias = initial.clone();
    let _trace = sample_ipndm_v(
        &backend,
        plan(IPNDM_V_SAMPLER_ID, 0, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        &context,
        |_latent, _sigma, step| {
            let denoised = fixture.steps.get(step).ok_or("missing step")?;
            tensor_from_f32(&backend, &fixture.shape, &denoised.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<_, String>(()),
    )?;
    assert_close(
        &values(&backend, &alias, &context)?,
        &fixture.initial,
        fixture.tolerance,
    );
    Ok(())
}
