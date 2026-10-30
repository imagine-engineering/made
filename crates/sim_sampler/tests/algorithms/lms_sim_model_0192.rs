use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    generated_lms_sim_model_0192::{
        DEFINITION, LMS_FEATURE_ID, LMS_MAX_ORDER, LMS_SAMPLER_ID, LMS_SOURCE_ORDINAL,
        LmsSamplerError, linear_multistep_coefficient, sample_lms, sample_lms_with_order,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    error::Error,
    fs,
    path::Path,
};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/lms_comfy_model_0192/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/lms_sim_model_0192.rs"
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

#[derive(Deserialize)]
struct HighOrderFixture {
    schema_version: u16,
    sources: std::collections::BTreeMap<String, String>,
    relative_tolerance: f32,
    cases: Vec<HighOrderCoefficients>,
    trajectories: Vec<HighOrderTrajectory>,
}

#[derive(Deserialize)]
struct HighOrderCoefficients {
    name: String,
    order: usize,
    step: usize,
    sigmas: Vec<f32>,
    coefficients: Vec<f32>,
}

#[derive(Deserialize)]
struct HighOrderTrajectory {
    order: usize,
    sigmas: Vec<f32>,
    callbacks: Vec<Vec<f32>>,
    terminal: Vec<f32>,
}

#[test]
fn val_sampling_foundation_001_high_orders_match_source_with_full_history()
-> Result<(), Box<dyn Error>> {
    let fixture: HighOrderFixture = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/samplers/lms_comfy_model_0192/source_high_orders.json"
    )))?;
    assert_eq!(fixture.schema_version, 1);
    for (path, expected) in &fixture.sources {
        assert_eq!(digest(&workspace_root()?.join(path))?, *expected);
    }
    assert_eq!(fixture.cases.len(), 24);
    for case in &fixture.cases {
        assert_eq!(case.coefficients.len(), case.order);
        for (index, expected) in case.coefficients.iter().copied().enumerate() {
            let actual = linear_multistep_coefficient(case.order, &case.sigmas, case.step, index)?;
            assert!(
                (actual - expected).abs() <= fixture.relative_tolerance * expected.abs().max(1.0),
                "{} order {} coefficient {index}: expected {expected}, got {actual}",
                case.name,
                case.order,
            );
        }
    }
    assert_eq!(fixture.trajectories.len(), 4);
    for trajectory in &fixture.trajectories {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
        let callbacks = Cell::new(0);
        let denoiser_calls = Cell::new(0);
        let trace = sample_lms_with_order(
            &backend,
            plan(LMS_SAMPLER_ID, 0x0192, trajectory.sigmas.len() - 1)?,
            &profile()?,
            initial,
            &trajectory.sigmas,
            trajectory.order,
            &context,
            |latent, sigma, step| {
                denoiser_calls.set(denoiser_calls.get() + 1);
                if step == 99 {
                    let mut denoised =
                        values(&backend, latent, &context).map_err(|error| error.to_string())?;
                    for value in &mut denoised {
                        *value -= sigma;
                    }
                    tensor_from_f32(&backend, &[1], &denoised, &context)
                        .map_err(|error| error.to_string())
                } else {
                    Ok(latent.clone())
                }
            },
            |progress, latent, _| {
                let expected = trajectory
                    .callbacks
                    .get(progress.step as usize)
                    .ok_or_else(|| "unexpected callback step".to_owned())?;
                let tolerance = fixture.relative_tolerance
                    * expected.iter().copied().map(f32::abs).fold(1.0, f32::max);
                assert_close(
                    &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                    expected,
                    tolerance,
                );
                callbacks.set(callbacks.get() + 1);
                Ok::<_, String>(())
            },
        )?;
        assert_eq!(callbacks.get(), 101);
        assert_eq!(denoiser_calls.get(), 101);
        assert_eq!(trace.latents.len(), 102);
        let terminal = trace.latents.last().ok_or("missing terminal latent")?;
        let tolerance = fixture.relative_tolerance
            * trajectory
                .terminal
                .iter()
                .copied()
                .map(f32::abs)
                .fold(1.0, f32::max);
        assert_close(
            &values(&backend, terminal, &context)?,
            &trajectory.terminal,
            tolerance,
        );
    }
    Ok(())
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
    Ok(SamplingProfileIdentity::new("analytical-lms-row-v1")?)
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

fn item<'a, T>(values: &'a [T], index: usize, role: &str) -> Result<&'a T, Box<dyn Error>> {
    values
        .get(index)
        .ok_or_else(|| format!("missing {role} at index {index}").into())
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
    assert_eq!(fixture.identity, LMS_SAMPLER_ID);
    assert_eq!(fixture.feature_id, LMS_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, LMS_SOURCE_ORDINAL);
    assert_eq!(fixture.max_order, LMS_MAX_ORDER);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, 10);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/lms_comfy_model_0192"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(LMS_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(
        registry
            .resolve(&SamplerIdentity::new("linear_multistep")?)
            .is_err()
    );
    assert!(SamplerIdentity::new("LMS").is_err());

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
        "def linear_multistep_coeff(",
        "prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])",
        "integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)",
        "def sample_lms(",
        "d = to_d(x, sigmas[i], denoised)",
        "ds.append(d)",
        "ds.pop(0)",
        "callback({'x': x",
        "x = denoised",
        "cur_order = min(i + 1, order)",
        "linear_multistep_coeff(cur_order, sigmas_cpu, i, j)",
        "zip(coeffs, reversed(ds))",
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
            .is_some_and(|line| line.contains("\"lms\"") && line.contains("\"dpm_fast\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| {
                line.starts_with("sampler,lms,") && line.ends_with(",COMFY-MODEL-0192")
            })
    );

    for forbidden in [
        "struct SamplingSession",
        "struct CancellationToken",
        "struct ExecutionContext",
        "struct SamplingTrace",
        "CompatibilityRngTransaction",
        "CpuWorkspaceAuthority",
        "authorize_workspace",
        "std::fs",
        "serde",
        "unsafe {",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    for delegated in [
        "SamplingSession::new",
        ".observe_step(",
        ".commit(next, context.cancellation)",
        "ExecutionContext<'_>",
        "context.cancellation",
        "backend.workspace_vec",
        "tensor_from_f32",
        "tensor_to_f32",
    ] {
        assert!(
            IMPLEMENTATION.contains(delegated),
            "missing canonical delegation {delegated}"
        );
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_coefficients_intermediates_and_callbacks()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());

    let trace = sample_lms(
        &backend,
        plan(LMS_SAMPLER_ID, 0x0192, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        &context,
        |latent, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            events.borrow_mut().push(format!("denoiser-{step}"));
            assert_eq!(sigma.to_bits(), expected.sigma.to_bits());
            assert_close(
                &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                &expected.latent_before,
                fixture.tolerance,
            );
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, callback_latent, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            events.borrow_mut().push(format!("callback-{step}"));
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
    assert_close(
        &values(&backend, &initial_alias, &context)?,
        &fixture.initial,
        0.0,
    );
    let callbacks = callbacks.into_inner();
    let mut derivative_history: Vec<Vec<f32>> = Vec::new();
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_eq!(expected.order, LMS_MAX_ORDER.min(step + 1));
        let callback = item(&callbacks, step, "callback")?;
        assert_eq!(usize::try_from(callback.0.step)?, step);
        assert_eq!(callback.0.sigma.to_bits(), expected.sigma.to_bits());
        assert_eq!(callback.0.sigma_hat.to_bits(), expected.sigma.to_bits());
        assert_eq!(
            callback.0.next_sigma.to_bits(),
            expected.next_sigma.to_bits()
        );
        assert_close(&callback.1, &expected.latent_before, fixture.tolerance);
        assert_close(&callback.2, &expected.denoised, fixture.tolerance);

        let latent = values(
            &backend,
            item(&trace.latents, step, "latent before")?,
            &context,
        )?;
        let denoised = values(
            &backend,
            item(&trace.denoiser_evaluations, step, "denoised")?,
            &context,
        )?;
        let derivative = latent
            .iter()
            .zip(&denoised)
            .map(|(latent, denoised)| (latent - denoised) / expected.sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &expected.derivative, fixture.tolerance);
        derivative_history.push(derivative);
        if derivative_history.len() > LMS_MAX_ORDER {
            derivative_history.rotate_left(1);
            assert!(derivative_history.pop().is_some());
        }

        if expected.next_sigma == 0.0 {
            assert!(expected.coefficients.is_empty());
            assert_close(&expected.latent_after, &expected.denoised, 0.0);
        } else {
            assert_eq!(expected.coefficients.len(), expected.order);
            let mut analytical = latent.clone();
            for (coefficient_index, expected_coefficient) in
                expected.coefficients.iter().copied().enumerate()
            {
                let coefficient = linear_multistep_coefficient(
                    expected.order,
                    &fixture.sigmas,
                    step,
                    coefficient_index,
                )?;
                assert_close(&[coefficient], &[expected_coefficient], fixture.tolerance);
                let history = item(
                    &derivative_history,
                    derivative_history.len() - 1 - coefficient_index,
                    "derivative history",
                )?;
                for (value, derivative_value) in analytical.iter_mut().zip(history) {
                    *value += coefficient * derivative_value;
                }
            }
            assert_close(&analytical, &expected.latent_after, fixture.tolerance);
        }
        assert_close(
            &values(
                &backend,
                item(&trace.latents, step + 1, "latent after")?,
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
    let expected_events = (0..fixture.steps.len())
        .flat_map(|step| [format!("denoiser-{step}"), format!("callback-{step}")])
        .collect::<Vec<_>>();
    assert_eq!(events.into_inner(), expected_events);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

fn run_fixture(seed: u64) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let trace = sample_lms(
        &backend,
        plan(LMS_SAMPLER_ID, seed, fixture.steps.len())?,
        &profile()?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        &context,
        |_, _, step| {
            let denoised = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            tensor_from_f32(&backend, &fixture.shape, &denoised.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    trace
        .latents
        .iter()
        .map(|tensor| values(&backend, tensor, &context))
        .collect()
}

#[test]
fn val_rng_001_lms_is_seed_independent_and_deterministic() -> Result<(), Box<dyn Error>> {
    assert!(!DEFINITION.stochastic);
    assert_eq!(run_fixture(0)?, run_fixture(u64::MAX)?);
    Ok(())
}

#[test]
fn boundaries_failures_and_cancellation_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;

    let terminal = sample_lms(
        &backend,
        plan(LMS_SAMPLER_ID, 0, 1)?,
        &profile()?,
        initial.clone(),
        &[1.0, 0.0],
        &context,
        |_, _, _| {
            tensor_from_f32(&backend, &[2], &[0.25, 0.5], &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    )?;
    assert_close(
        &values(
            &backend,
            item(&terminal.latents, 1, "terminal latent")?,
            &context,
        )?,
        &[0.25, 0.5],
        0.0,
    );

    assert!(matches!(
        linear_multistep_coefficient(0, &[1.0, 0.5], 0, 0),
        Err(LmsSamplerError::InvalidOrder { .. })
    ));
    assert!(matches!(
        linear_multistep_coefficient(2, &[1.0, 0.5], 0, 0),
        Err(LmsSamplerError::InvalidOrder { .. })
    ));
    assert!(matches!(
        linear_multistep_coefficient(2, &[1.0, 1.0, 0.5], 1, 0),
        Err(LmsSamplerError::SingularCoefficient { step: 1, .. })
    ));
    assert!(matches!(
        sample_lms(
            &backend,
            plan("ddpm", 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::WrongSampler(value)) if value == "ddpm"
    ));
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 2)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::Sampling(SamplingError::ScheduleLength {
            expected: 3,
            actual: 2
        }))
    ));
    let wrong_profile = SamplingProfileIdentity::new("wrong-lms-profile-v1")?;
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &wrong_profile,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::Sampling(
            SamplingError::ProfileMismatch { .. }
        ))
    ));
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 1.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::Sampling(SamplingError::InvalidSigma {
            step: 0,
            ..
        }))
    ));
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| Err("fixture denoiser fault".to_owned()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::Denoiser { step: 0, .. })
    ));
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::DenoiserContract { step: 0 })
    ));
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Err("fixture callback fault")
        ),
        Err(LmsSamplerError::Sampling(SamplingError::Callback(reason)))
            if reason == "fixture callback fault"
    ));

    let nan_callbacks = Cell::new(0_usize);
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| {
                nan_callbacks.set(nan_callbacks.get() + 1);
                Ok::<(), String>(())
            }
        ),
        Err(LmsSamplerError::NonFinite {
            step: 0,
            stage: "denoiser",
            element: 0
        })
    ));
    assert_eq!(nan_callbacks.get(), 0);

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_token = callback_cancellation.clone();
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &callback_context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| {
                assert!(callback_token.cancel());
                Ok::<(), String>(())
            }
        ),
        Err(LmsSamplerError::Cancelled { step: 0 })
    ));

    let pre_cancellation = CancellationToken::default();
    assert!(pre_cancellation.cancel());
    let pre_context = execution_context(&backend, &authority, &pre_cancellation)?;
    let denoiser_calls = Cell::new(0_usize);
    assert!(matches!(
        sample_lms(
            &backend,
            plan(LMS_SAMPLER_ID, 0, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &pre_context,
            |value, _, _| {
                denoiser_calls.set(denoiser_calls.get() + 1);
                Ok(value.clone())
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(LmsSamplerError::Cancelled { step: 0 })
    ));
    assert_eq!(denoiser_calls.get(), 0);
    assert_close(&values(&backend, &initial, &context)?, &[1.0, -1.0], 0.0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(callback_context.scratch.in_use_bytes(), 0);
    assert_eq!(pre_context.scratch.in_use_bytes(), 0);
    Ok(())
}
