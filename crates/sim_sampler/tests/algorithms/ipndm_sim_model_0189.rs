use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    generated_ipndm_sim_model_0189::{
        DEFINITION, IPNDM_FEATURE_ID, IPNDM_HISTORY_CAPACITY, IPNDM_MAX_ORDER, IPNDM_SAMPLER_ID,
        IPNDM_SOURCE_ORDINAL, IpndmSamplerError, sample_ipndm,
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
    "/../sim_test_support/fixtures/samplers/ipndm_comfy_model_0189/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/ipndm_sim_model_0189.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    max_order: usize,
    history_capacity: usize,
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
    sigma: f32,
    next_sigma: f32,
    order: usize,
    denoised: Vec<f32>,
    derivative: Vec<f32>,
    effective_derivative: Option<Vec<f32>>,
    next_latent: Vec<f32>,
    history_after: Vec<Vec<f32>>,
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
    Ok(SamplingProfileIdentity::new("analytical-ipndm-row-v1")?)
}

fn plan(identity: &str, steps: usize) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile()?,
        0x0189,
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

fn item<'a, T>(values: &'a [T], index: usize, role: &str) -> Result<&'a T, Box<dyn Error>> {
    values
        .get(index)
        .ok_or_else(|| format!("missing {role} at index {index}").into())
}

#[test]
fn definition_registry_and_pinned_source_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, IPNDM_SAMPLER_ID);
    assert_eq!(fixture.feature_id, IPNDM_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, IPNDM_SOURCE_ORDINAL);
    assert_eq!(fixture.max_order, IPNDM_MAX_ORDER);
    assert_eq!(fixture.history_capacity, IPNDM_HISTORY_CAPACITY);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert!(DEFINITION.aliases.is_empty());
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/ipndm_comfy_model_0189"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(IPNDM_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(registry.resolve(&SamplerIdentity::new("pndm")?).is_err());
    assert!(SamplerIdentity::new("IPNDM").is_err());

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
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_ipndm",
        "x_next = x",
        "buffer_model = []",
        "denoised = model(x_cur",
        "callback({'x': x",
        "d_cur = (x_cur - denoised) / t_cur",
        "order = min(max_order, i+1)",
        "x_next = denoised",
        "(3 * d_cur - buffer_model[-1]) / 2",
        "(23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12",
        "(55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24",
        "if len(buffer_model) == max_order - 1",
        "buffer_model[k] = buffer_model[k+1]",
        "buffer_model[-1] = d_cur",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"ipndm\"") && line.contains("\"ipndm_v\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| {
                line.starts_with("sampler,ipndm,") && line.ends_with(",COMFY-MODEL-0189")
            })
    );
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_ipndm_intermediate_and_callback() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let trace = sample_ipndm(
        &backend,
        plan(IPNDM_SAMPLER_ID, fixture.steps.len())?,
        &profile()?,
        initial,
        &fixture.sigmas,
        &context,
        |input, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            events.borrow_mut().push(format!("denoiser-{step}"));
            assert_eq!(sigma.to_bits(), expected.sigma.to_bits());
            let expected_input = if step == 0 {
                fixture.initial.as_slice()
            } else {
                fixture
                    .steps
                    .get(step - 1)
                    .ok_or_else(|| format!("missing input for step {step}"))?
                    .next_latent
                    .as_slice()
            };
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                expected_input,
                fixture.tolerance,
            );
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, callback_latent, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected callback step {step}"))?;
            events.borrow_mut().push(format!("callback-{step}"));
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.steps.len()).map_err(|error| error.to_string())?
            );
            assert_eq!(progress.sigma.to_bits(), expected.sigma.to_bits());
            assert_eq!(progress.sigma_hat.to_bits(), expected.sigma.to_bits());
            assert_eq!(progress.next_sigma.to_bits(), expected.next_sigma.to_bits());
            assert_close(
                &values(&backend, callback_latent, &context).map_err(|error| error.to_string())?,
                &fixture.initial,
                0.0,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.denoised,
                fixture.tolerance,
            );
            Ok::<(), String>(())
        },
    )?;

    let expected_events = (0..fixture.steps.len())
        .flat_map(|step| [format!("denoiser-{step}"), format!("callback-{step}")])
        .collect::<Vec<_>>();
    assert_eq!(events.into_inner(), expected_events);
    assert_close(
        &values(&backend, &initial_alias, &context)?,
        &fixture.initial,
        0.0,
    );
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);

    let mut analytical_history: Vec<Vec<f32>> = Vec::new();
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.order, IPNDM_MAX_ORDER.min(step + 1));
        let current = values(
            &backend,
            item(&trace.latents, step, "current latent")?,
            &context,
        )?;
        let denoised = values(
            &backend,
            item(&trace.denoiser_evaluations, step, "denoiser output")?,
            &context,
        )?;
        let derivative = current
            .iter()
            .zip(denoised.iter())
            .map(|(current, denoised)| (current - denoised) / expected.sigma)
            .collect::<Vec<_>>();
        assert_close(&derivative, &expected.derivative, fixture.tolerance);

        if let Some(expected_effective) = expected.effective_derivative.as_deref() {
            let effective = derivative
                .iter()
                .enumerate()
                .map(|(element, current)| match expected.order {
                    1 => *current,
                    2 => {
                        (3.0 * current
                            - analytical_history.last().expect("order 2 history")[element])
                            / 2.0
                    }
                    3 => {
                        (23.0 * current
                            - 16.0
                                * analytical_history.last().expect("order 3 newest history")
                                    [element]
                            + 5.0
                                * analytical_history
                                    .get(analytical_history.len() - 2)
                                    .expect("order 3 older history")[element])
                            / 12.0
                    }
                    4 => {
                        (55.0 * current
                            - 59.0
                                * analytical_history.last().expect("order 4 newest history")
                                    [element]
                            + 37.0
                                * analytical_history
                                    .get(analytical_history.len() - 2)
                                    .expect("order 4 middle history")[element]
                            - 9.0
                                * analytical_history
                                    .get(analytical_history.len() - 3)
                                    .expect("order 4 oldest history")[element])
                            / 24.0
                    }
                    order => panic!("unexpected fixture order {order}"),
                })
                .collect::<Vec<_>>();
            assert_close(&effective, expected_effective, fixture.tolerance);
        } else {
            assert_eq!(expected.next_sigma, 0.0);
            assert_close(&expected.next_latent, &expected.denoised, 0.0);
        }
        assert_close(
            &values(
                &backend,
                item(&trace.latents, step + 1, "next latent")?,
                &context,
            )?,
            &expected.next_latent,
            fixture.tolerance,
        );

        if analytical_history.len() == IPNDM_HISTORY_CAPACITY {
            analytical_history.rotate_left(1);
            if let Some(newest) = analytical_history.last_mut() {
                *newest = derivative;
            }
        } else {
            analytical_history.push(derivative);
        }
        assert_eq!(analytical_history.len(), expected.history_after.len());
        for (actual, expected) in analytical_history.iter().zip(&expected.history_after) {
            assert_close(actual, expected, fixture.tolerance);
        }
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
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn boundaries_failures_and_cancellation_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;

    let terminal = sample_ipndm(
        &backend,
        plan(IPNDM_SAMPLER_ID, 1)?,
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
        sample_ipndm(
            &backend,
            plan("ddpm", 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(IpndmSamplerError::WrongSampler(value)) if value == "ddpm"
    ));
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 2)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(IpndmSamplerError::Sampling(SamplingError::ScheduleLength {
            expected: 3,
            actual: 2
        }))
    ));
    let wrong_profile = SamplingProfileIdentity::new("wrong-ipndm-profile-v1")?;
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
            &wrong_profile,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(IpndmSamplerError::Sampling(
            SamplingError::ProfileMismatch { .. }
        ))
    ));
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 1.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(IpndmSamplerError::Sampling(SamplingError::InvalidSigma {
            step: 0,
            ..
        }))
    ));
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(IpndmSamplerError::DenoiserContract { step: 0 })
    ));
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| Err("fixture denoiser fault".to_owned()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(IpndmSamplerError::Denoiser { step: 0, .. })
    ));
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Err("fixture callback fault")
        ),
        Err(IpndmSamplerError::Sampling(SamplingError::Callback(reason))) if reason == "fixture callback fault"
    ));
    let callbacks = RefCell::new(0_usize);
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| {
                *callbacks.borrow_mut() += 1;
                Ok::<(), String>(())
            }
        ),
        Err(IpndmSamplerError::NonFinite {
            step: 0,
            stage: "denoiser",
            element: 0
        })
    ));
    assert_eq!(*callbacks.borrow(), 1);

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_token = callback_cancellation.clone();
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
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
        Err(IpndmSamplerError::Cancelled { step: 0 })
    ));

    let pre_cancellation = CancellationToken::default();
    assert!(pre_cancellation.cancel());
    let pre_context = execution_context(&backend, &authority, &pre_cancellation)?;
    let denoiser_calls = Cell::new(0_usize);
    assert!(matches!(
        sample_ipndm(
            &backend,
            plan(IPNDM_SAMPLER_ID, 1)?,
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
        Err(IpndmSamplerError::Cancelled { step: 0 })
    ));
    assert_eq!(denoiser_calls.get(), 0);
    assert_close(&values(&backend, &initial, &context)?, &[1.0, -1.0], 0.0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(callback_context.scratch.in_use_bytes(), 0);
    assert_eq!(pre_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn row_is_an_equation_adapter_over_authoritative_owners() {
    for forbidden in [
        "CompatibilityNoiseRequest",
        "CompatibilityRngTransaction",
        "CancellationToken",
        "struct SamplingProgress",
        "struct SamplingTrace",
        "struct SamplingSession",
        "struct ExecutionContext",
        "authorize_workspace",
        "CpuWorkspaceAuthority",
        "std::fs",
        "serde",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "row introduced forbidden owner {forbidden}"
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
            "missing delegation {delegated}"
        );
    }
}
