use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfile, SamplingProfileIdentity,
    generated_sa_solver_sim_model_0197::{
        DEFINITION, SA_SOLVER_FEATURE_ID, SA_SOLVER_NOISE_CONTRACT_ID, SA_SOLVER_SAMPLER_ID,
        SA_SOLVER_SOURCE_ORDINAL, SaSolverError, SaSolverEvaluation, SaSolverFamilyOptions,
        SaSolverOptions, sample_sa_solver, sample_sa_solver_family, stochastic_adams_coefficients,
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
    "/../sim_test_support/fixtures/samplers/sa_solver_comfy_model_0197/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/sa_solver_sim_model_0197.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    rng_contract_id: String,
    seed: u64,
    noise_scale: f32,
    tolerance: f32,
    shape: Vec<u64>,
    profile_sigmas: Vec<f32>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    rng: RngFixture,
    steps: Vec<StepFixture>,
    terminal: Vec<f32>,
    stochastic_noise: Vec<Vec<f32>>,
    stochastic_latents: Vec<Vec<f32>>,
    long_schedule: LongScheduleFixture,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    sampling_path: String,
    sampling_sha256: String,
    coefficient_path: String,
    coefficient_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
    sampling_lines: Vec<usize>,
    coefficient_lines: Vec<usize>,
    registry_line: usize,
    catalog_line: usize,
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
    current: Vec<f32>,
    denoised: Vec<f32>,
    corrector_order: usize,
    corrector_coefficients: Vec<f32>,
    corrected: Vec<f32>,
    predictor_order: usize,
    predictor_coefficients: Vec<f32>,
    next: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct LongScheduleFixture {
    sigmas: Vec<f32>,
    orders: Vec<(usize, usize)>,
    latents: Vec<Vec<f32>>,
    pece_latents: Vec<Vec<f32>>,
    pece_corrected_inputs: Vec<Vec<f32>>,
    simple_predictor_order_2: Vec<f32>,
    simple_corrector_order_2: Vec<f32>,
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

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("analytical-sa-solver-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas.clone()),
    )?)
}

fn plan(fixture: &Fixture, identity: &str) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new("analytical-sa-solver-row-v1")?,
        fixture.seed,
        u32::try_from(fixture.steps.len())?,
        1.0,
        1.0,
    )?)
}

fn request(fixture: &Fixture, retry: u32, policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        &fixture.rng.workflow,
        &fixture.rng.attempt,
        &fixture.rng.node,
        fixture.rng.output,
        fixture.rng.execution_ordinal,
        fixture.rng.batch,
        retry,
        policy,
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
fn definition_registry_pinned_sources_and_family_ownership_are_exact() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, SA_SOLVER_SAMPLER_ID);
    assert_eq!(fixture.feature_id, SA_SOLVER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, SA_SOLVER_SOURCE_ORDINAL);
    assert_eq!(fixture.rng_contract_id, SA_SOLVER_NOISE_CONTRACT_ID);
    assert_eq!(DEFINITION.source_ordinal, 39);
    assert!(DEFINITION.stochastic);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(SA_SOLVER_SAMPLER_ID)?)?,
        &DEFINITION
    );

    let root = workspace_root()?;
    for (path, expected) in [
        (
            &fixture.source.sampling_path,
            &fixture.source.sampling_sha256,
        ),
        (
            &fixture.source.coefficient_path,
            &fixture.source.coefficient_sha256,
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
    let sampling_evidence = fixture
        .source
        .sampling_lines
        .iter()
        .filter_map(|line| sampling.lines().nth(line - 1))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def sample_sa_solver(",
        "offset_first_sigma_for_snr",
        "sigma_to_half_log_snr",
        "corrector_order_used = 0",
        "compute_stochastic_adams_b_coeffs",
        "x = sigmas[i] / sigmas[i - 1]",
        "if use_pece:",
        "x_pred = sigmas[i + 1] / sigmas[i]",
        "noise_sampler(sigmas[i], sigmas[i + 1])",
        "use_pece=True",
    ] {
        assert!(
            sampling_evidence.contains(fragment),
            "missing source fragment {fragment}"
        );
    }
    let coefficients = fs::read_to_string(root.join(&fixture.source.coefficient_path))?;
    let coefficient_evidence = fixture
        .source
        .coefficient_lines
        .iter()
        .filter_map(|line| coefficients.lines().nth(line - 1))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def compute_exponential_coeffs",
        "tau_mul = 1 + tau_t ** 2",
        "product_terms_factored",
        "torch.linalg.solve",
        "alpha_t = sigma_next * lambda_t.exp()",
        "def get_tau_interval_func",
    ] {
        assert!(
            coefficient_evidence.contains(fragment),
            "missing coefficient source {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| {
                line.contains("\"sa_solver\"") && line.contains("\"sa_solver_pece\"")
            })
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(|line| {
                line.starts_with("sampler,sa_solver,") && line.ends_with(",COMFY-MODEL-0197")
            })
    );

    for forbidden in [
        "struct SamplingSession",
        "struct CancellationToken",
        "struct CpuWorkspaceAuthority",
        "struct CompatibilityRngTransaction",
        "fn half_log_snr(",
        "std::fs",
        "sqlx",
        "rusqlite",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "row owns forbidden service {forbidden}"
        );
    }
    assert!(IMPLEMENTATION.contains("pub fn sample_sa_solver_family"));
    assert!(IMPLEMENTATION.contains("use_pece: bool"));
    assert!(IMPLEMENTATION.contains("SaSolverEvaluation::Corrected"));
    Ok(())
}

#[test]
fn val_sampler_001_matches_every_intermediate_callback_and_default_tau()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let initial_alias = initial.clone();
    let events = RefCell::new(Vec::new());
    let options = SaSolverOptions::new(fixture.noise_scale, 3, 4, false)?;
    let (trace, (before, after)) = sample_sa_solver(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        initial,
        &fixture.sigmas,
        request(&fixture, fixture.rng.retry, RetryRngPolicy::Replay),
        options,
        &context,
        |input, sigma, step| {
            let expected = fixture.steps.get(step).ok_or("unexpected denoiser step")?;
            assert_eq!(sigma.to_bits(), fixture.sigmas[step].to_bits());
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                &expected.current,
                fixture.tolerance,
            );
            events.borrow_mut().push(format!("denoiser-{step}"));
            tensor_from_f32(&backend, &fixture.shape, &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, current, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture.steps.get(step).ok_or("unexpected callback step")?;
            assert_close(
                &values(&backend, current, &context).map_err(|error| error.to_string())?,
                &expected.current,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.denoised,
                fixture.tolerance,
            );
            events.borrow_mut().push(format!("callback-{step}"));
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
            "callback-3"
        ]
    );
    assert_eq!(before, after);
    assert_close(
        &values(&backend, &initial_alias, &context)?,
        &fixture.initial,
        0.0,
    );
    for (step, expected) in fixture.steps.iter().enumerate() {
        assert_close(
            &values(
                &backend,
                trace.latents.get(step).ok_or("missing latent")?,
                &context,
            )?,
            &expected.current,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace.latents.get(step + 1).ok_or("missing next latent")?,
                &context,
            )?,
            &expected.next,
            fixture.tolerance,
        );
    }
    assert_close(
        &values(
            &backend,
            trace.latents.last().ok_or("missing terminal")?,
            &context,
        )?,
        &fixture.terminal,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn analytical_fixture_reconstructs_coefficients_and_state_equations() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let lambdas = fixture
        .sigmas
        .iter()
        .map(|sigma| {
            if *sigma == 0.0 {
                f32::INFINITY
            } else {
                -sigma.ln()
            }
        })
        .collect::<Vec<_>>();
    let mut corrected_state = fixture.initial.clone();
    let mut history: Vec<Vec<f32>> = Vec::new();
    let mut previous_h = 0.0_f32;
    for (step, expected) in fixture.steps.iter().enumerate() {
        history.push(expected.denoised.clone());
        if history.len() > 4 {
            history.remove(0);
        }
        let corrected = if expected.corrector_order == 0 {
            expected.current.clone()
        } else {
            let order = expected.corrector_order;
            let coefficients = stochastic_adams_coefficients(
                fixture.sigmas[step],
                &lambdas[step + 1 - order..=step],
                lambdas[step - 1],
                lambdas[step],
                1.0,
                false,
                true,
                step,
            )?;
            assert_close(
                &coefficients,
                &expected.corrector_coefficients,
                fixture.tolerance,
            );
            let scale = fixture.sigmas[step] / fixture.sigmas[step - 1] * (-previous_h).exp();
            (0..fixture.initial.len())
                .map(|element| {
                    scale * corrected_state[element]
                        + coefficients
                            .iter()
                            .zip(history[history.len() - order..].iter())
                            .map(|(coefficient, prediction)| coefficient * prediction[element])
                            .sum::<f32>()
                })
                .collect()
        };
        assert_close(&corrected, &expected.corrected, fixture.tolerance);
        if expected.predictor_order > 0 {
            let order = expected.predictor_order;
            let coefficients = stochastic_adams_coefficients(
                fixture.sigmas[step + 1],
                &lambdas[step + 1 - order..=step],
                lambdas[step],
                lambdas[step + 1],
                1.0,
                false,
                false,
                step,
            )?;
            assert_close(
                &coefficients,
                &expected.predictor_coefficients,
                fixture.tolerance,
            );
            let h = lambdas[step + 1] - lambdas[step];
            let scale = fixture.sigmas[step + 1] / fixture.sigmas[step] * (-h).exp();
            let next = (0..fixture.initial.len())
                .map(|element| {
                    scale * corrected[element]
                        + coefficients
                            .iter()
                            .zip(history[history.len() - order..].iter())
                            .map(|(coefficient, prediction)| coefficient * prediction[element])
                            .sum::<f32>()
                })
                .collect::<Vec<_>>();
            assert_close(&next, &expected.next, fixture.tolerance);
            previous_h = h;
        }
        corrected_state = corrected;
    }
    Ok(())
}

#[test]
fn long_history_full_orders_simple_order_two_and_pece_states_match_oracle()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let make_plan = || -> Result<SamplingPlan, Box<dyn Error>> {
        Ok(SamplingPlan::new(
            SA_SOLVER_SAMPLER_ID,
            "normal",
            profile.identity().clone(),
            fixture.seed,
            u32::try_from(fixture.long_schedule.sigmas.len() - 1)?,
            1.0,
            1.0,
        )?)
    };
    let model = |input: &Tensor, sigma: f32| -> Result<Tensor, String> {
        let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
        let first = input.first().copied().ok_or("missing first input")?;
        let second = input.get(1).copied().ok_or("missing second input")?;
        tensor_from_f32(
            &backend,
            &fixture.shape,
            &[
                0.25 * first + 0.05 * sigma + 0.1,
                -0.2 * second + 0.03 * sigma - 0.05,
            ],
            &context,
        )
        .map_err(|error| error.to_string())
    };
    let deterministic = sample_sa_solver_family(
        &backend,
        make_plan()?,
        &profile,
        SA_SOLVER_SAMPLER_ID,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.long_schedule.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverFamilyOptions::new(SaSolverOptions::new(0.0, 3, 4, false)?, false),
        &context,
        |_sigma, _step| Ok(0.0),
        |input, sigma, _step, _evaluation| model(input, sigma),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    assert_eq!(
        deterministic.0.latents.len(),
        fixture.long_schedule.latents.len()
    );
    for (actual, expected) in deterministic
        .0
        .latents
        .iter()
        .zip(&fixture.long_schedule.latents)
    {
        assert_close(
            &values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    let total_steps = fixture.long_schedule.sigmas.len() - 1;
    let mut available = 0_usize;
    let derived_orders = (0..total_steps)
        .map(|step| {
            available = (available + 1).min(4);
            let predictor = 3.min(available).min(total_steps - 1 - step);
            let corrector = if step == 0 || step + 1 == total_steps {
                0
            } else {
                4.min(available).min(total_steps - step)
            };
            (predictor, corrector)
        })
        .collect::<Vec<_>>();
    assert_eq!(derived_orders, fixture.long_schedule.orders);

    let corrected_inputs = RefCell::new(Vec::new());
    let pece = sample_sa_solver_family(
        &backend,
        make_plan()?,
        &profile,
        SA_SOLVER_SAMPLER_ID,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.long_schedule.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverFamilyOptions::new(SaSolverOptions::new(0.0, 3, 4, false)?, true),
        &context,
        |_sigma, _step| Ok(0.0),
        |input, sigma, _step, evaluation| {
            if evaluation == SaSolverEvaluation::Corrected {
                corrected_inputs
                    .borrow_mut()
                    .push(values(&backend, input, &context).map_err(|error| error.to_string())?);
            }
            model(input, sigma)
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    for (actual, expected) in pece
        .0
        .latents
        .iter()
        .zip(&fixture.long_schedule.pece_latents)
    {
        assert_close(
            &values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    for (actual, expected) in corrected_inputs
        .into_inner()
        .iter()
        .zip(&fixture.long_schedule.pece_corrected_inputs)
    {
        assert_close(actual, expected, fixture.tolerance);
    }

    let lambdas = fixture
        .long_schedule
        .sigmas
        .iter()
        .take(2)
        .map(|sigma| -sigma.ln())
        .collect::<Vec<_>>();
    let predictor = stochastic_adams_coefficients(
        3.0,
        &lambdas,
        -3.5_f32.ln(),
        -3.0_f32.ln(),
        0.0,
        true,
        false,
        1,
    )?;
    assert_close(
        &predictor,
        &fixture.long_schedule.simple_predictor_order_2,
        fixture.tolerance,
    );
    let corrector = stochastic_adams_coefficients(
        3.5,
        &lambdas,
        -4.0_f32.ln(),
        -3.5_f32.ln(),
        0.0,
        true,
        true,
        1,
    )?;
    assert_close(
        &corrector,
        &fixture.long_schedule.simple_corrector_order_2,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn rng_pece_failures_cancellation_retry_and_atomicity_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert!(SaSolverOptions::new(f32::NAN, 3, 4, false).is_err());
    assert!(SaSolverOptions::new(1.0, 0, 4, false).is_err());
    assert_eq!(SaSolverOptions::new(1.0, 3, 0, false)?.corrector_order(), 0);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;

    let run = |retry, policy| -> Result<_, Box<dyn Error>> {
        let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
        sample_sa_solver_family(
            &backend,
            plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
            &profile(&fixture)?,
            SA_SOLVER_SAMPLER_ID,
            initial,
            &fixture.sigmas,
            request(&fixture, retry, policy),
            SaSolverFamilyOptions::new(SaSolverOptions::new(0.75, 3, 4, false)?, false),
            &context,
            |_sigma, _step| Ok(1.0),
            |_input, _sigma, step, evaluation| {
                assert_eq!(evaluation, SaSolverEvaluation::Predictor);
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_progress, _current, _denoised| Ok::<(), String>(()),
        )
        .map_err(Box::<dyn Error>::from)
    };
    let replay_a = run(2, RetryRngPolicy::Replay)?;
    for (actual, expected) in replay_a.0.latents.iter().zip(&fixture.stochastic_latents) {
        assert_close(
            &values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    let replay_b = run(2, RetryRngPolicy::Replay)?;
    assert_eq!(replay_a.1, replay_b.1);
    for (left, right) in replay_a.0.latents.iter().zip(replay_b.0.latents.iter()) {
        assert_close(
            &values(&backend, left, &context)?,
            &values(&backend, right, &context)?,
            0.0,
        );
    }
    let advanced = run(3, RetryRngPolicy::Advance)?;
    assert_ne!(replay_a.1, advanced.1);
    let mut oracle = request(&fixture, 2, RetryRngPolicy::Replay).open_transaction(
        SA_SOLVER_NOISE_CONTRACT_ID,
        i128::from(fixture.seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(replay_a.1.0, oracle.checkpoint());
    for expected in &fixture.stochastic_noise {
        let actual = oracle.draw_normal(fixture.initial.len(), &cancellation)?;
        assert_close(
            &actual.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            expected,
            fixture.tolerance,
        );
    }
    assert_eq!(replay_a.1.1, oracle.commit());

    let pece_events = RefCell::new(Vec::new());
    let pece_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    sample_sa_solver_family(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        SA_SOLVER_SAMPLER_ID,
        pece_initial,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverFamilyOptions::new(SaSolverOptions::new(0.0, 3, 4, false)?, true),
        &context,
        |_sigma, _step| Ok(1.0),
        |_input, _sigma, step, evaluation| {
            pece_events.borrow_mut().push((step, evaluation));
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    assert_eq!(
        pece_events.into_inner(),
        [
            (0, SaSolverEvaluation::Predictor),
            (1, SaSolverEvaluation::Predictor),
            (1, SaSolverEvaluation::Corrected),
            (2, SaSolverEvaluation::Predictor),
            (2, SaSolverEvaluation::Corrected),
            (3, SaSolverEvaluation::Predictor),
            (3, SaSolverEvaluation::Corrected),
        ]
    );

    for short_sigmas in [&fixture.sigmas[..1], &fixture.sigmas[..0]] {
        let denoiser_called = RefCell::new(false);
        let callback_called = RefCell::new(false);
        let short = sample_sa_solver(
            &backend,
            plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
            &profile(&fixture)?,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            short_sigmas,
            request(&fixture, 0, RetryRngPolicy::Replay),
            SaSolverOptions::default(),
            &context,
            |_input, _sigma, _step| {
                *denoiser_called.borrow_mut() = true;
                Err("short schedule must not evaluate".to_owned())
            },
            |_progress, _current, _denoised| {
                *callback_called.borrow_mut() = true;
                Ok::<(), String>(())
            },
        )?;
        assert_eq!(short.0.sigmas, short_sigmas);
        assert!(short.0.denoiser_evaluations.is_empty());
        assert_eq!(short.0.latents.len(), 1);
        assert_close(
            &values(&backend, &short.0.latents[0], &context)?,
            &fixture.initial,
            0.0,
        );
        assert_eq!(short.1.0, short.1.1);
        assert!(!denoiser_called.into_inner());
        assert!(!callback_called.into_inner());
    }

    let wrong = sample_sa_solver(
        &backend,
        plan(&fixture, "lms")?,
        &profile(&fixture)?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverOptions::default(),
        &context,
        |_input, _sigma, _step| Err("must not run".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(wrong, Err(SaSolverError::WrongSampler { .. })));

    let callback_alias = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let callback_input = callback_alias.clone();
    let callback_error = sample_sa_solver(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        callback_input,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverOptions::default(),
        &context,
        |_input, _sigma, step| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Err::<(), _>("callback failed"),
    );
    assert!(matches!(callback_error, Err(SaSolverError::Sampling(_))));
    assert_close(
        &values(&backend, &callback_alias, &context)?,
        &fixture.initial,
        0.0,
    );

    let descriptor_error = sample_sa_solver(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverOptions::default(),
        &context,
        |_input, _sigma, _step| {
            tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        descriptor_error,
        Err(SaSolverError::DenoiserContract { .. })
    ));

    for tau_mode in ["failure", "nan", "negative"] {
        let tau_error = sample_sa_solver_family(
            &backend,
            plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
            &profile(&fixture)?,
            SA_SOLVER_SAMPLER_ID,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            request(&fixture, 0, RetryRngPolicy::Replay),
            SaSolverFamilyOptions::new(SaSolverOptions::default(), false),
            &context,
            move |_sigma, _step| match tau_mode {
                "failure" => Err("injected tau failure".to_owned()),
                "nan" => Ok(f32::NAN),
                "negative" => Ok(-1.0),
                _ => Ok(0.0),
            },
            |_input, _sigma, step, _evaluation| {
                tensor_from_f32(
                    &backend,
                    &fixture.shape,
                    &fixture.steps[step].denoised,
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_progress, _current, _denoised| Ok::<(), String>(()),
        );
        match tau_mode {
            "failure" => assert!(matches!(
                tau_error,
                Err(SaSolverError::TauFunction { step: 0, .. })
            )),
            _ => assert!(matches!(
                tau_error,
                Err(SaSolverError::InvalidCoefficient {
                    step: 0,
                    coefficient: "tau",
                    ..
                })
            )),
        }
    }

    let terminal_non_finite = sample_sa_solver(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverOptions::new(0.0, 3, 4, false)?,
        &context,
        |_input, _sigma, step| {
            let output = if step + 1 == fixture.steps.len() {
                vec![f32::NAN, 0.0]
            } else {
                fixture.steps[step].denoised.clone()
            };
            tensor_from_f32(&backend, &fixture.shape, &output, &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        terminal_non_finite,
        Err(SaSolverError::NonFinite {
            step: 3,
            stage: "terminal denoiser",
            element: 0
        })
    ));

    let callback_cancellation = CancellationToken::default();
    let callback_cancellation_context =
        execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_cancelled = sample_sa_solver(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        tensor_from_f32(
            &backend,
            &fixture.shape,
            &fixture.initial,
            &callback_cancellation_context,
        )?,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverOptions::default(),
        &callback_cancellation_context,
        |_input, _sigma, step| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.steps[step].denoised,
                &callback_cancellation_context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| {
            callback_cancellation.cancel();
            Ok::<(), String>(())
        },
    );
    assert!(matches!(
        callback_cancelled,
        Err(SaSolverError::Sampling(
            sim_sampler::SamplingError::Cancelled
        ))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancelled_result = sample_sa_solver(
        &backend,
        plan(&fixture, SA_SOLVER_SAMPLER_ID)?,
        &profile(&fixture)?,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(&fixture, 0, RetryRngPolicy::Replay),
        SaSolverOptions::default(),
        &cancelled_context,
        |_input, _sigma, _step| Err("must not run".to_owned()),
        |_progress, _current, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        cancelled_result,
        Err(SaSolverError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}
