use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    generated_dpmpp_2m_cfg_pp_sim_model_0167::{
        DEFINITION, DPMPP_2M_CFG_PP_FEATURE_ID, DPMPP_2M_CFG_PP_SAMPLER_ID,
        DPMPP_2M_CFG_PP_SOURCE_ORDINAL, Dpmpp2mCfgPpDenoiserOutput, Dpmpp2mCfgPpSamplerError,
        sample_dpmpp_2m_cfg_pp,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpmpp_2m_cfg_pp_comfy_model_0167/trajectory.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/algorithms/dpmpp_2m_cfg_pp_sim_model_0167.rs");

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
    guided_scale: f32,
    guided_sigma_weights: Vec<f32>,
    guided_offsets: Vec<f32>,
    unconditional_scale: f32,
    unconditional_sigma_weights: Vec<f32>,
    unconditional_offsets: Vec<f32>,
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
    guided_denoised: Vec<f32>,
    unconditional_denoised: Vec<f32>,
    time: f32,
    next_time: Option<f32>,
    step_size: Option<f32>,
    previous_step_size: Option<f32>,
    step_ratio: Option<f32>,
    latent_ratio: f32,
    unconditional_weight: f32,
    history_weight: f32,
    history_delta: Option<Vec<f32>>,
    denoised_mix: Vec<f32>,
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
    Ok(SamplingProfileIdentity::new("dpmpp-2m-cfg-pp-row-v1")?)
}

fn plan(identity: &str, steps: u32) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        profile()?,
        167,
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

fn analytical_values(
    input: &[f32],
    sigma: f32,
    scale: f32,
    sigma_weights: &[f32],
    offsets: &[f32],
) -> Vec<f32> {
    input
        .iter()
        .zip(sigma_weights)
        .zip(offsets)
        .map(|((value, sigma_weight), offset)| scale * value + sigma * sigma_weight + offset)
        .collect()
}

fn assert_fixture_equations(fixture: &Fixture) -> Result<(), Box<dyn Error>> {
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_eq!(expected.step, step);
        assert_eq!(fixture.sigmas.get(step).copied(), Some(expected.sigma));
        assert_eq!(
            fixture.sigmas.get(step + 1).copied(),
            Some(expected.next_sigma)
        );
        let guided = analytical_values(
            &expected.latent_before,
            expected.sigma,
            fixture.guided_scale,
            &fixture.guided_sigma_weights,
            &fixture.guided_offsets,
        );
        let unconditional = analytical_values(
            &expected.latent_before,
            expected.sigma,
            fixture.unconditional_scale,
            &fixture.unconditional_sigma_weights,
            &fixture.unconditional_offsets,
        );
        assert_close(&guided, &expected.guided_denoised, fixture.tolerance);
        assert_close(
            &unconditional,
            &expected.unconditional_denoised,
            fixture.tolerance,
        );

        let time = -expected.sigma.ln();
        assert_scalar(time, expected.time, fixture.tolerance, "fixture time");
        let (next_time, step_size, latent_ratio) = if expected.next_sigma == 0.0 {
            (None, None, 0.0)
        } else {
            let next_time = -expected.next_sigma.ln();
            let step_size = next_time - time;
            (Some(next_time), Some(step_size), (-step_size).exp())
        };
        assert_optional(
            next_time,
            expected.next_time,
            fixture.tolerance,
            "fixture next time",
        );
        assert_optional(
            step_size,
            expected.step_size,
            fixture.tolerance,
            "fixture step size",
        );
        assert_scalar(
            latent_ratio,
            expected.latent_ratio,
            fixture.tolerance,
            "fixture latent ratio",
        );
        assert_scalar(
            -latent_ratio,
            expected.unconditional_weight,
            fixture.tolerance,
            "fixture unconditional weight",
        );

        let (previous_step_size, step_ratio, history_weight, history_delta) =
            if step > 0 && expected.next_sigma != 0.0 {
                let previous = fixture
                    .steps
                    .get(step - 1)
                    .ok_or("missing preceding fixture step")?;
                let previous_time = -previous.sigma.ln();
                let previous_step_size = time - previous_time;
                let current_step_size = step_size.ok_or("missing non-terminal step size")?;
                let step_ratio = previous_step_size / current_step_size;
                let history_weight = -(-current_step_size).exp_m1() * (1.0 / (2.0 * step_ratio));
                let history_delta = expected
                    .guided_denoised
                    .iter()
                    .zip(&previous.unconditional_denoised)
                    .map(|(guided, old_unconditional)| guided - old_unconditional)
                    .collect::<Vec<_>>();
                (
                    Some(previous_step_size),
                    Some(step_ratio),
                    history_weight,
                    Some(history_delta),
                )
            } else {
                (None, None, 0.0, None)
            };
        assert_optional(
            previous_step_size,
            expected.previous_step_size,
            fixture.tolerance,
            "fixture previous step size",
        );
        assert_optional(
            step_ratio,
            expected.step_ratio,
            fixture.tolerance,
            "fixture step ratio",
        );
        assert_scalar(
            history_weight,
            expected.history_weight,
            fixture.tolerance,
            "fixture history weight",
        );
        assert_eq!(history_delta.is_some(), expected.history_delta.is_some());
        if let (Some(actual), Some(expected)) =
            (history_delta.as_ref(), expected.history_delta.as_ref())
        {
            assert_close(actual, expected, fixture.tolerance);
        }

        let denoised_mix = expected
            .unconditional_denoised
            .iter()
            .enumerate()
            .map(|(element, unconditional)| {
                let unconditional_term = -latent_ratio * unconditional;
                history_delta
                    .as_ref()
                    .and_then(|values| values.get(element))
                    .copied()
                    .map(|history| unconditional_term + history_weight * history)
                    .unwrap_or(unconditional_term)
            })
            .collect::<Vec<_>>();
        assert_close(&denoised_mix, &expected.denoised_mix, fixture.tolerance);
        let latent_after = expected
            .guided_denoised
            .iter()
            .zip(&denoised_mix)
            .zip(&expected.latent_before)
            .map(|((guided, mix), latent)| (guided + mix) + latent_ratio * latent)
            .collect::<Vec<_>>();
        assert_close(&latent_after, &expected.latent_after, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_cfg_pp_definition_and_source_provenance_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPMPP_2M_CFG_PP_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DPMPP_2M_CFG_PP_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DPMPP_2M_CFG_PP_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert!(!DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpmpp_2m_cfg_pp_comfy_model_0167"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPMPP_2M_CFG_PP_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(registry.resolve(&SamplerIdentity::new("dpmpp_2m")?).is_ok());
    assert!(
        registry
            .resolve(&SamplerIdentity::new("dpmpp2m_cfgpp")?)
            .is_err()
    );
    assert!(SamplerIdentity::new("DPMPP_2M_CFG_PP").is_err());

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
        "def sample_dpmpp_2m_cfg_pp",
        "old_uncond_denoised = None",
        "uncond_denoised = None",
        "uncond_denoised = args[\"uncond_denoised\"]",
        "disable_cfg1_optimization=True",
        "denoised = model",
        "callback({'x': x",
        "h = t_next - t",
        "old_uncond_denoised is None or sigmas[i + 1] == 0",
        "-torch.exp(-h) * uncond_denoised",
        "r = h_last / h",
        "(denoised - old_uncond_denoised)",
        "x = denoised + denoised_mix + torch.exp(-h) * x",
        "old_uncond_denoised = uncond_denoised",
    ] {
        assert!(equations.contains(fragment), "missing equation {fragment}");
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"dpmpp_2m_cfg_pp\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(catalog.lines().any(|line| {
        line.contains("sampler,dpmpp_2m_cfg_pp,") && line.ends_with(",COMFY-MODEL-0167")
    }));
    Ok(())
}

#[test]
fn val_sampler_001_dpmpp_2m_cfg_pp_matches_every_intermediate_and_callback()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_fixture_equations(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let trace = sample_dpmpp_2m_cfg_pp(
        &backend,
        plan(
            DPMPP_2M_CFG_PP_SAMPLER_ID,
            u32::try_from(fixture.steps.len())?,
        )?,
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
            let guided = analytical_values(
                &input,
                sigma,
                fixture.guided_scale,
                &fixture.guided_sigma_weights,
                &fixture.guided_offsets,
            );
            let unconditional = analytical_values(
                &input,
                sigma,
                fixture.unconditional_scale,
                &fixture.unconditional_sigma_weights,
                &fixture.unconditional_offsets,
            );
            assert_close(&guided, &expected.guided_denoised, fixture.tolerance);
            assert_close(
                &unconditional,
                &expected.unconditional_denoised,
                fixture.tolerance,
            );
            Ok(Dpmpp2mCfgPpDenoiserOutput {
                denoised: tensor_from_f32(&backend, &fixture.shape, &guided, &context)
                    .map_err(|error| error.to_string())?,
                unconditional_denoised: tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &unconditional,
                    &context,
                )
                .map_err(|error| error.to_string())?,
            })
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
                &expected.guided_denoised,
                fixture.tolerance,
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
        assert_close(
            &values(
                &backend,
                trace
                    .denoiser_evaluations
                    .get(step)
                    .ok_or("missing canonical denoiser observation")?,
                &context,
            )?,
            &expected.guided_denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace
                    .latents
                    .get(step + 1)
                    .ok_or("missing canonical latent observation")?,
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
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

fn same_output(value: &Tensor) -> Dpmpp2mCfgPpDenoiserOutput {
    Dpmpp2mCfgPpDenoiserOutput {
        denoised: value.clone(),
        unconditional_denoised: value.clone(),
    }
}

#[test]
fn dpmpp_2m_cfg_pp_rejects_invalid_contracts_and_schedules() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;

    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan("dpmpp_2m", 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(same_output(value)),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mCfgPpSamplerError::WrongSampler(identity)) if identity == "dpmpp_2m"
    ));
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
            &SamplingProfileIdentity::new("different-profile-v1")?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(same_output(value)),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mCfgPpSamplerError::Sampling(
            SamplingError::ProfileMismatch { .. }
        ))
    ));
    for sigmas in [&[1.0, 1.0][..], &[1.0, 2.0][..], &[f32::NAN, 0.0][..]] {
        assert!(matches!(
            sample_dpmpp_2m_cfg_pp(
                &backend,
                plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
                &profile()?,
                initial.clone(),
                sigmas,
                &context,
                |value, _, _| Ok(same_output(value)),
                |_, _, _| Ok::<(), String>(()),
            ),
            Err(Dpmpp2mCfgPpSamplerError::Sampling(
                SamplingError::InvalidSigma { .. }
            ))
        ));
    }
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 2)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(same_output(value)),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mCfgPpSamplerError::Sampling(
            SamplingError::ScheduleLength { .. }
        ))
    ));

    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| {
                Ok(Dpmpp2mCfgPpDenoiserOutput {
                    denoised: tensor_from_f32(&backend, &[1], &[0.0], &context)
                        .map_err(|error| error.to_string())?,
                    unconditional_denoised: initial.clone(),
                })
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mCfgPpSamplerError::DenoiserContract {
            step: 0,
            output: "guided denoiser output"
        })
    ));
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
            &profile()?,
            initial,
            &[1.0, 0.0],
            &context,
            |value, _, _| {
                Ok(Dpmpp2mCfgPpDenoiserOutput {
                    denoised: value.clone(),
                    unconditional_denoised: tensor_from_f32(&backend, &[1], &[0.0], &context)
                        .map_err(|error| error.to_string())?,
                })
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mCfgPpSamplerError::DenoiserContract {
            step: 0,
            output: "unconditional denoiser output"
        })
    ));
    Ok(())
}

#[test]
fn dpmpp_2m_cfg_pp_failures_and_cancellation_are_failure_atomic() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;
    let initial_alias = initial.clone();

    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
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
            },
        ),
        Err(Dpmpp2mCfgPpSamplerError::Denoiser { step: 0, reason }) if reason == "model fault"
    ));
    assert_eq!(events.into_inner(), ["denoiser-0"]);

    let callbacks = RefCell::new(0_u32);
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(same_output(value)),
            |_, _, _| {
                *callbacks.borrow_mut() += 1;
                Err("callback fault")
            },
        ),
        Err(Dpmpp2mCfgPpSamplerError::Sampling(SamplingError::Callback(reason)))
            if reason == "callback fault"
    ));
    assert_eq!(*callbacks.borrow(), 1);

    for (non_finite_output, expected_stage) in [
        ("guided", "guided denoiser"),
        ("unconditional", "unconditional denoiser"),
    ] {
        let callbacks = RefCell::new(0_u32);
        assert!(matches!(
            sample_dpmpp_2m_cfg_pp(
                &backend,
                plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
                &profile()?,
                initial.clone(),
                &[1.0, 0.0],
                &context,
                |value, _, _| {
                    let non_finite = tensor_from_f32(
                        &backend,
                        &[2],
                        &[f32::NAN, 0.0],
                        &context,
                    )
                    .map_err(|error| error.to_string())?;
                    Ok(Dpmpp2mCfgPpDenoiserOutput {
                        denoised: if non_finite_output == "guided" {
                            non_finite.clone()
                        } else {
                            value.clone()
                        },
                        unconditional_denoised: if non_finite_output == "unconditional" {
                            non_finite
                        } else {
                            value.clone()
                        },
                    })
                },
                |_, _, _| {
                    *callbacks.borrow_mut() += 1;
                    Ok::<(), String>(())
                },
            ),
            Err(Dpmpp2mCfgPpSamplerError::NonFinite {
                step: 0,
                stage,
                element: 0
            }) if stage == expected_stage
        ));
        assert_eq!(*callbacks.borrow(), 1);
    }

    let pre_cancelled = CancellationToken::default();
    assert!(pre_cancelled.cancel());
    let pre_cancelled_context = execution_context(&backend, &authority, &pre_cancelled)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
            &profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &pre_cancelled_context,
            |value, _, _| {
                events.borrow_mut().push("denoiser");
                Ok(same_output(value))
            },
            |_, _, _| {
                events.borrow_mut().push("callback");
                Ok::<(), String>(())
            },
        ),
        Err(Dpmpp2mCfgPpSamplerError::Tensor(TensorError::Cancelled))
    ));
    assert!(events.borrow().is_empty());

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let events = RefCell::new(Vec::new());
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 2)?,
            &profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            &callback_context,
            |value, _, step| {
                events.borrow_mut().push(format!("denoiser-{step}"));
                Ok(same_output(value))
            },
            |progress, _, _| {
                events
                    .borrow_mut()
                    .push(format!("callback-{}", progress.step));
                callback_cancellation.cancel();
                Ok::<(), String>(())
            },
        ),
        Err(Dpmpp2mCfgPpSamplerError::Sampling(SamplingError::Cancelled))
    ));
    assert_eq!(events.into_inner(), ["denoiser-0", "callback-0"]);

    assert_close(
        &values(&backend, &initial_alias, &context)?,
        &[1.0, -1.0],
        0.0,
    );

    let zero_workspace = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    assert!(matches!(
        sample_dpmpp_2m_cfg_pp(
            &backend,
            plan(DPMPP_2M_CFG_PP_SAMPLER_ID, 1)?,
            &profile()?,
            initial,
            &[1.0, 0.0],
            &zero_workspace,
            |value, _, _| Ok(same_output(value)),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(Dpmpp2mCfgPpSamplerError::TensorKernel(
            NativeDiffusionTensorError::Tensor(TensorError::WorkspaceAuthorizationExceeded { .. })
        ))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn dpmpp_2m_cfg_pp_is_only_an_equation_adapter_over_authoritative_owners() {
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
        "Vec<Dpmpp2mCfgPpStep",
        "Dpmpp2mCfgPpTrace",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "DPM++ 2M CFG++ duplicates authoritative owner {forbidden}"
        );
    }
}
