use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    generated_dpmpp_2m_sim_model_0166::{
        DEFINITION, DPMPP_2M_FEATURE_ID, DPMPP_2M_SAMPLER_ID, DPMPP_2M_SOURCE_ORDINAL,
        Dpmpp2mSamplerError, sample_dpmpp_2m,
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
    "/../sim_test_support/fixtures/samplers/dpmpp_2m_comfy_model_0166/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/dpmpp_2m_sim_model_0166.rs");

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
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: usize,
    sigma: f32,
    next_sigma: f32,
    latent_before: Vec<f32>,
    denoised: Vec<f32>,
    time: f32,
    next_time: Option<f32>,
    step_size: Option<f32>,
    previous_step_size: Option<f32>,
    step_ratio: Option<f32>,
    latent_ratio: f32,
    current_denoised_weight: f32,
    previous_denoised_weight: f32,
    transformed_denoised: Vec<f32>,
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
    Ok(SamplingProfileIdentity::new("dpmpp-2m-row-v1")?)
}

fn plan(identity: &str, steps: u32) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile()?,
        166,
        steps,
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

fn assert_scalar(actual: f32, expected: f32, tolerance: f32, role: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{role}: expected {expected}, got {actual}"
    );
}

fn assert_optional(actual: Option<f32>, expected: Option<f32>, tolerance: f32, role: &str) {
    assert_eq!(
        actual.is_some(),
        expected.is_some(),
        "{role}: optional coefficient presence changed"
    );
    if let (Some(actual), Some(expected)) = (actual, expected) {
        assert_scalar(actual, expected, tolerance, role);
    }
}

#[test]
fn val_sampler_001_dpmpp_2m_definition_and_source_provenance_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2M_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2M_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_2M_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2m_comfy_model_0166"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2M_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(registry.resolve(&SamplerIdentity::new("dpmpp2m")?).is_err());
    assert!(SamplerIdentity::new("DPMPP_2M").is_err());

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
        "def sample_dpmpp_2m",
        "old_denoised = None",
        "denoised = model",
        "callback({'x': x",
        "h = t_next - t",
        "old_denoised is None or sigmas[i + 1] == 0",
        "h_last = t - t_fn(sigmas[i - 1])",
        "r = h_last / h",
        "denoised_d = (1 + 1 / (2 * r))",
        "old_denoised = denoised",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_2m\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog.lines().any(|line| {
            line.contains("sampler,dpmpp_2m,") && line.ends_with(",COMFY-MODEL-0166")
        })
    );
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_matches_every_intermediate_callback_and_terminal_boundary()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let trace = sample_dpmpp_2m(
        &backend,
        plan(DPMPP_2M_SAMPLER_ID, u32::try_from(fixture.steps.len())?)?,
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
            assert_eq!(expected.step, step);
            assert_eq!(sigma.to_bits(), expected.sigma.to_bits());
            let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
            assert_close(&input, &expected.latent_before, fixture.tolerance);
            let biases = [0.11_f32, -0.07, 0.03];
            let offsets = [-0.02_f32, 0.04, -0.06];
            let analytical = input
                .iter()
                .zip(biases)
                .zip(offsets)
                .map(|((value, bias), offset)| 0.62 * value + sigma * bias + offset)
                .collect::<Vec<_>>();
            assert_close(&analytical, &expected.denoised, fixture.tolerance);
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
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
            assert_eq!(progress.next_sigma.to_bits(), expected.next_sigma.to_bits());
            assert_close(
                &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                &expected.latent_before,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.denoised,
                0.0,
            );
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
        &values(&backend, &initial_alias, &context)?,
        &fixture.initial,
        0.0,
    );
    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(trace.latents.len(), fixture.steps.len() + 1);
    assert_eq!(trace.denoiser_evaluations.len(), fixture.steps.len());
    for (step, expected) in fixture.steps.iter().enumerate() {
        let time = -expected.sigma.ln();
        assert_scalar(time, expected.time, fixture.tolerance, "time");
        let next_time = (expected.next_sigma != 0.0).then_some(-expected.next_sigma.ln());
        assert_optional(
            next_time,
            expected.next_time,
            fixture.tolerance,
            "next time",
        );
        let step_size = next_time.map(|next_time| next_time - time);
        assert_optional(
            step_size,
            expected.step_size,
            fixture.tolerance,
            "step size",
        );
        let previous_step_size = if step > 0 && expected.next_sigma != 0.0 {
            let previous_sigma = fixture
                .sigmas
                .get(step - 1)
                .copied()
                .ok_or("missing previous fixture sigma")?;
            Some(time - -previous_sigma.ln())
        } else {
            None
        };
        assert_optional(
            previous_step_size,
            expected.previous_step_size,
            fixture.tolerance,
            "previous step size",
        );
        let step_ratio = previous_step_size
            .zip(step_size)
            .map(|(previous, current)| previous / current);
        assert_optional(
            step_ratio,
            expected.step_ratio,
            fixture.tolerance,
            "step ratio",
        );
        let latent_ratio = next_time
            .map(|next_time| (-next_time).exp() / (-time).exp())
            .unwrap_or(0.0);
        assert_scalar(
            latent_ratio,
            expected.latent_ratio,
            fixture.tolerance,
            "latent ratio",
        );
        let inverse_double_ratio = step_ratio.map(|ratio| 1.0 / (2.0 * ratio));
        let current_denoised_weight = inverse_double_ratio
            .map(|coefficient| 1.0 + coefficient)
            .unwrap_or(1.0);
        let previous_denoised_weight = inverse_double_ratio.map(|value| -value).unwrap_or(0.0);
        assert_scalar(
            current_denoised_weight,
            expected.current_denoised_weight,
            fixture.tolerance,
            "current denoised weight",
        );
        assert_scalar(
            previous_denoised_weight,
            expected.previous_denoised_weight,
            fixture.tolerance,
            "previous denoised weight",
        );
        let previous_denoised = step
            .checked_sub(1)
            .and_then(|previous| fixture.steps.get(previous))
            .map(|previous| previous.denoised.as_slice());
        let transformed_denoised = expected
            .denoised
            .iter()
            .enumerate()
            .map(|(element, denoised)| {
                current_denoised_weight * denoised
                    + previous_denoised_weight
                        * previous_denoised
                            .and_then(|values| values.get(element))
                            .copied()
                            .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        assert_close(
            &transformed_denoised,
            &expected.transformed_denoised,
            fixture.tolerance,
        );
        let analytical_after = if let Some(step_size) = step_size {
            expected
                .latent_before
                .iter()
                .zip(&transformed_denoised)
                .map(|(latent, denoised)| {
                    latent_ratio * latent - (-step_size).exp_m1() * denoised
                })
                .collect::<Vec<_>>()
        } else {
            expected.denoised.clone()
        };
        assert_close(
            &analytical_after,
            &expected.latent_after,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace
                    .denoiser_evaluations
                    .get(step)
                    .ok_or("missing denoiser trace")?,
                &context,
            )?,
            &expected.denoised,
            0.0,
        );
        assert_close(
            &values(
                &backend,
                trace
                    .latents
                    .get(step + 1)
                    .ok_or("missing latent trace")?,
                &context,
            )?,
            &expected.latent_after,
            fixture.tolerance,
        );
    }
    assert_close(
        &values(
            &backend,
            trace
                .latents
                .last()
                .ok_or("missing terminal latent")?,
            &context,
        )?,
        &fixture.terminal,
        fixture.tolerance,
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn dpmpp_2m_failures_and_cancellation_are_failure_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;

    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan("ddpm", 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpmpp2mSamplerError::WrongSampler(identity)) if identity == "ddpm"
    ));
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 1)?,
            &SamplingProfileIdentity::new("different-profile-v1")?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpmpp2mSamplerError::Sampling(
            SamplingError::ProfileMismatch { .. }
        ))
    ));
    for sigmas in [&[1.0, 1.0][..], &[1.0, 2.0][..], &[f32::NAN, 0.0][..]] {
        assert!(matches!(
            sample_dpmpp_2m(
                &backend,
                plan(DPMPP_2M_SAMPLER_ID, 1)?,
                &profile()?,
                initial.clone(),
                sigmas,
                &context,
                |value, _, _| Ok(value.clone()),
                |_, _, _| Ok::<(), String>(())
            ),
            Err(Dpmpp2mSamplerError::Sampling(
                SamplingError::InvalidSigma { .. }
            ))
        ));
    }
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 2)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpmpp2mSamplerError::Sampling(
            SamplingError::ScheduleLength { .. }
        ))
    ));

    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, step| {
                events.borrow_mut().push(format!("denoiser-{step}"));
                Err("model fault".to_owned())
            },
            |_, _, _| {
                events.borrow_mut().push("callback".to_owned());
                Ok::<(), String>(())
            }
        ),
        Err(Dpmpp2mSamplerError::Denoiser { step: 0, reason }) if reason == "model fault"
    ));
    assert_eq!(events.into_inner(), ["denoiser-0"]);

    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| {
                tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(Dpmpp2mSamplerError::DenoiserContract { step: 0 })
    ));

    let callbacks = RefCell::new(0_u32);
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| {
                *callbacks.borrow_mut() += 1;
                Err("callback fault")
            }
        ),
        Err(Dpmpp2mSamplerError::Sampling(SamplingError::Callback(reason)))
            if reason == "callback fault"
    ));
    assert_eq!(*callbacks.borrow(), 1);

    let non_finite_callbacks = RefCell::new(0_u32);
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| {
                tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| {
                *non_finite_callbacks.borrow_mut() += 1;
                Ok::<(), String>(())
            }
        ),
        Err(Dpmpp2mSamplerError::NonFinite {
            step: 0,
            stage: "denoiser",
            element: 0
        })
    ));
    assert_eq!(*non_finite_callbacks.borrow(), 1);

    let pre_cancelled = CancellationToken::default();
    assert!(pre_cancelled.cancel());
    let pre_cancelled_context = execution_context(&backend, &authority, &pre_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &pre_cancelled_context,
            |value, _, _| {
                events.borrow_mut().push("denoiser");
                Ok(value.clone())
            },
            |_, _, _| {
                events.borrow_mut().push("callback");
                Ok::<(), String>(())
            }
        ),
        Err(Dpmpp2mSamplerError::Tensor(TensorError::Cancelled))
    ));
    assert!(events.borrow().is_empty());

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpmpp_2m(
            &backend,
            plan(DPMPP_2M_SAMPLER_ID, 2)?,
            &profile()?,
            initial,
            &[2.0, 1.0, 0.0],
            &callback_context,
            |value, _, step| {
                events.borrow_mut().push(format!("denoiser-{step}"));
                Ok(value.clone())
            },
            |progress, _, _| {
                events
                    .borrow_mut()
                    .push(format!("callback-{}", progress.step));
                callback_cancellation.cancel();
                Ok::<(), String>(())
            }
        ),
        Err(Dpmpp2mSamplerError::Sampling(SamplingError::Cancelled))
    ));
    assert_eq!(events.into_inner(), ["denoiser-0", "callback-0"]);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn dpmpp_2m_is_only_an_equation_adapter_over_authoritative_sampling_owners() {
    for required in [
        "SamplingPlan",
        "SamplingSession::new",
        ".observe_step(",
        "observed.commit(",
        "SamplingTrace",
        "ExecutionContext",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing owner delegation {required}"
        );
    }
    for forbidden in [
        "struct SamplingTrace",
        "struct SamplingProgress",
        "struct CancellationToken",
        "struct RngCheckpoint",
        "CompatibilityRngTransaction",
        "RngStream::new",
        "fn commit_step",
        "fn observe_step",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "DPM++ 2M duplicates authoritative owner {forbidden}"
        );
    }
}
