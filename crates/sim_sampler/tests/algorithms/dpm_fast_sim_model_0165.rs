use sim_sampler::{
    CompatibilityNoiseRequest, SamplerIdentity, SamplerRegistry, SamplingPlan,
    SamplingProfileIdentity,
    generated_dpm_fast_sim_model_0165::{
        DEFINITION, DPM_FAST_NOISE_CONTRACT_ID, DPM_FAST_SAMPLER_ID, DpmFastOptions,
        DpmFastSamplerError, sample_dpm_fast,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, RetryRngPolicy, RngCompatibilityError, RngCompatibilityOperation,
    RngCompatibilityPhase, RngGenerationPlacement, RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::PathBuf};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/dpm_fast_comfy_model_0165/trajectory.json"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    tolerance: f32,
    seed: u64,
    rng: RngFixture,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    options: OptionsFixture,
    internal_times: Vec<f32>,
    intervals: Vec<IntervalFixture>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct RngFixture {
    contract_id: String,
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    execution_ordinal: u64,
    batch: u64,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    wrapper_path: String,
    wrapper_sha256: String,
    equation_path: String,
    equation_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct OptionsFixture {
    eta: f32,
    noise_scale: f32,
}

#[derive(Debug, Deserialize)]
struct IntervalFixture {
    interval: u32,
    order: u8,
    time: f32,
    next_time: f32,
    solver_next_time: f32,
    sigma: f32,
    next_sigma: f32,
    sigma_down: f32,
    sigma_up: f32,
    latent_before: Vec<f32>,
    base_denoised: Vec<f32>,
    evaluations: Vec<EvaluationFixture>,
    solved: Vec<f32>,
    noise: Vec<f32>,
    latent_after: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EvaluationFixture {
    evaluation: u8,
    sigma: f32,
    input: Vec<f32>,
    denoised: Vec<f32>,
    epsilon: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn digest(path: &str) -> Result<String, Box<dyn Error>> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(workspace_root()?.join(path))?)
    ))
}

fn profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("dpm-fast-row-v1")?)
}

fn plan(seed: u64, steps: u32) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        DPM_FAST_SAMPLER_ID,
        "normal",
        profile()?,
        seed,
        steps,
        1.0,
        1.0,
    )?)
}

fn noise_request(
    fixture: &Fixture,
    retry: u32,
    policy: RetryRngPolicy,
) -> CompatibilityNoiseRequest {
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

fn open_noise_transaction(
    request: CompatibilityNoiseRequest,
    seed: u64,
    cancellation: &CancellationToken,
) -> Result<CompatibilityRngTransaction, RngCompatibilityError> {
    request.open_transaction(
        DPM_FAST_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        cancellation,
    )
}

fn execution_context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
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
    for (element, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {element}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

fn analytical_denoised(input: &[f32], sigma: f32) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(element, value)| 0.72_f32 * value + sigma * if element == 0 { 0.11 } else { -0.18 })
        .collect()
}

fn analytical_ancestral_target(
    sigma_from: f32,
    sigma_to: f32,
    terminal_time: f32,
    eta: f32,
) -> (f32, f32, f32) {
    let variance = sigma_to.powi(2) * (sigma_from.powi(2) - sigma_to.powi(2)) / sigma_from.powi(2);
    let sigma_up = sigma_to.min(eta * variance.max(0.0).sqrt());
    let sigma_down = (sigma_to.powi(2) - sigma_up.powi(2)).max(0.0).sqrt();
    let solver_time = (-sigma_down.ln()).min(terminal_time);
    let solver_sigma = (-solver_time).exp();
    let stochastic_scale = (sigma_to.powi(2) - solver_sigma.powi(2)).max(0.0).sqrt();
    (sigma_down, solver_time, stochastic_scale)
}

#[test]
fn val_sampler_001_dpm_fast_definition_source_and_rng_provenance_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DPM_FAST_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DEFINITION.feature_id);
    assert_eq!(fixture.source_ordinal, DEFINITION.source_ordinal);
    assert_eq!(fixture.rng.contract_id, DPM_FAST_NOISE_CONTRACT_ID);
    assert_eq!(fixture.rng.workflow, "dpm-fast-fixture-v1");
    assert_eq!(fixture.rng.attempt, "attempt-0165");
    assert_eq!(fixture.rng.node, "KSampler-11");
    assert_eq!(fixture.rng.output, 11);
    assert_eq!(fixture.rng.execution_ordinal, 165);
    assert_eq!(fixture.rng.batch, 74);
    assert_eq!(DEFINITION.source_ordinal, 11);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/dpm_fast_comfy_model_0165"
    );
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SamplerIdentity::new(DPM_FAST_SAMPLER_ID)?)?,
        &DEFINITION
    );
    assert!(registry.resolve(&SamplerIdentity::new("dpmfast")?).is_err());
    assert_eq!(
        digest(&fixture.source.wrapper_path)?,
        fixture.source.wrapper_sha256
    );
    assert_eq!(
        digest(&fixture.source.equation_path)?,
        fixture.source.equation_sha256
    );
    assert_eq!(
        digest(&fixture.source.catalog_path)?,
        fixture.source.catalog_sha256
    );
    let source = fs::read_to_string(workspace_root()?.join(&fixture.source.equation_path))?;
    for fragment in [
        "def dpm_solver_fast",
        "m = math.floor(nfe / 3) + 1",
        "orders = [3] * (m - 2) + [2, 1]",
        "self.dpm_solver_3_step",
        "x = x + su * s_noise * noise_sampler",
        "def sample_dpm_fast",
    ] {
        assert!(
            source.contains(fragment),
            "missing pinned equation {fragment}"
        );
    }
    let contract = rng_compatibility_contract(DPM_FAST_NOISE_CONTRACT_ID)
        .ok_or("DPM fast RNG contract is unavailable")?;
    assert_eq!(contract.operation(), RngCompatibilityOperation::Normal);
    assert_eq!(
        contract.phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    Ok(())
}

#[test]
fn dpm_fast_is_only_an_equation_adapter_over_authoritative_sampling_and_rng_owners()
-> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(
        workspace_root()?.join("crates/sim_sampler/src/algorithms/dpm_fast_sim_model_0165.rs"),
    )?;
    for required in [
        "SamplingPlan",
        "SamplingSession::new",
        ".observe_step(",
        "observed.commit(",
        "CompatibilityNoiseRequest",
        "noise_request.open_transaction",
        "noise_transaction.commit()",
        "dpm_solver_first_order",
        "dpm_solver_first_intermediate",
        "dpm_solver_second_order",
        "dpm_solver_third_order",
    ] {
        assert!(
            source.contains(required),
            "missing owner delegation {required}"
        );
    }
    for forbidden in [
        "struct DpmFastTrace",
        "struct SamplingTrace",
        "struct SamplingProgress",
        "struct CancellationToken",
        "struct RngCheckpoint",
        "RngStream::new",
        "fn commit_step",
        "fn observe_step",
        "CompatibilityRngTransaction::open",
        "RngCompatibilityRequest::new",
        ".exp_m1()",
    ] {
        assert!(
            !source.contains(forbidden),
            "DPM fast duplicates authoritative owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampler_001_dpm_fast_matches_every_fixed_order_intermediate_and_callback()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &fixture.initial, &context)?;
    let events = RefCell::new(Vec::new());
    let evaluation_count = RefCell::new(0_usize);
    let options = DpmFastOptions::new(fixture.options.eta, fixture.options.noise_scale)?;
    let (trace, checkpoints) = sample_dpm_fast(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        &profile()?,
        initial,
        &fixture.sigmas,
        options,
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        &context,
        |input, sigma, interval, evaluation| {
            let expected_interval = fixture
                .intervals
                .get(usize::try_from(interval).map_err(|error| error.to_string())?)
                .ok_or_else(|| format!("unexpected interval {interval}"))?;
            let expected = expected_interval
                .evaluations
                .get(usize::from(evaluation))
                .ok_or_else(|| format!("unexpected evaluation {evaluation}"))?;
            if expected.evaluation != evaluation {
                return Err(format!(
                    "evaluation address mismatch at {interval}:{evaluation}"
                ));
            }
            events
                .borrow_mut()
                .push(format!("denoiser-{interval}-{evaluation}"));
            *evaluation_count.borrow_mut() += 1;
            if (sigma - expected.sigma).abs() > fixture.tolerance {
                return Err(format!("sigma mismatch at {interval}:{evaluation}"));
            }
            let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
            assert_close(&input, &expected.input, fixture.tolerance);
            let analytical = analytical_denoised(&input, sigma);
            assert_close(&analytical, &expected.denoised, fixture.tolerance);
            let epsilon = input
                .iter()
                .zip(expected.denoised.iter())
                .map(|(input, denoised)| (input - denoised) / sigma)
                .collect::<Vec<_>>();
            assert_close(&epsilon, &expected.epsilon, fixture.tolerance);
            tensor_from_f32(&backend, &[2], &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            let interval = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            let expected = fixture
                .intervals
                .get(interval)
                .ok_or_else(|| format!("unexpected callback {interval}"))?;
            events.borrow_mut().push(format!("callback-{interval}"));
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.sigmas.len() - 1).map_err(|error| error.to_string())?
            );
            assert!((progress.sigma - expected.sigma).abs() <= fixture.tolerance);
            assert!((progress.next_sigma - expected.next_sigma).abs() <= fixture.tolerance);
            assert_close(
                &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                &expected.latent_before,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.base_denoised,
                fixture.tolerance,
            );
            Ok::<(), String>(())
        },
    )?;

    assert_eq!(
        events.into_inner(),
        [
            "denoiser-0-0",
            "callback-0",
            "denoiser-0-1",
            "denoiser-0-2",
            "denoiser-1-0",
            "callback-1",
            "denoiser-1-1",
        ]
    );
    assert_eq!(*evaluation_count.borrow(), fixture.sigmas.len() - 1);
    assert_close(
        &trace.sigmas,
        &fixture
            .intervals
            .iter()
            .map(|value| value.sigma)
            .chain(fixture.intervals.last().map(|value| value.next_sigma))
            .collect::<Vec<_>>(),
        fixture.tolerance,
    );
    assert_eq!(trace.denoiser_evaluations.len(), fixture.intervals.len());
    assert_eq!(trace.latents.len(), fixture.intervals.len() + 1);
    for (index, expected) in fixture.intervals.iter().enumerate() {
        assert_eq!(expected.interval, u32::try_from(index)?);
        assert_eq!(expected.order, if index == 0 { 3 } else { 2 });
        let time = fixture
            .internal_times
            .get(index)
            .ok_or("missing fixture interval time")?;
        let next_time = fixture
            .internal_times
            .get(index + 1)
            .ok_or("missing fixture next interval time")?;
        assert!((*time - expected.time).abs() <= fixture.tolerance);
        assert!((*next_time - expected.next_time).abs() <= fixture.tolerance);
        let (sigma_down, solver_next_time, sigma_up) = analytical_ancestral_target(
            expected.sigma,
            expected.next_sigma,
            *fixture
                .internal_times
                .last()
                .ok_or("missing fixture terminal time")?,
            fixture.options.eta,
        );
        assert!((sigma_down - expected.sigma_down).abs() <= fixture.tolerance);
        assert!((solver_next_time - expected.solver_next_time).abs() <= fixture.tolerance);
        assert!((sigma_up - expected.sigma_up).abs() <= fixture.tolerance);
        assert_close(
            &values(
                &backend,
                trace
                    .latents
                    .get(index)
                    .ok_or("missing traced pre-step latent")?,
                &context,
            )?,
            &expected.latent_before,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace
                    .denoiser_evaluations
                    .get(index)
                    .ok_or("missing traced denoiser output")?,
                &context,
            )?,
            &expected.base_denoised,
            fixture.tolerance,
        );
        assert_close(
            &values(
                &backend,
                trace
                    .latents
                    .get(index + 1)
                    .ok_or("missing traced post-step latent")?,
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

    let (actual_before, actual_after) = checkpoints.ok_or("missing RNG checkpoints")?;
    let mut oracle = open_noise_transaction(
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        fixture.seed,
        &cancellation,
    )?;
    assert_eq!(actual_before, oracle.checkpoint());
    for expected in &fixture.intervals {
        let noise = oracle
            .draw_normal(fixture.initial.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_close(&noise, &expected.noise, 0.0);
        let reconstructed = expected
            .solved
            .iter()
            .zip(noise.iter())
            .map(|(solved, noise)| solved + expected.sigma_up * fixture.options.noise_scale * noise)
            .collect::<Vec<_>>();
        assert_close(&reconstructed, &expected.latent_after, fixture.tolerance);
    }
    assert_eq!(actual_after, oracle.commit());
    Ok(())
}

#[test]
fn dpm_fast_partitions_every_nfe_remainder_into_exact_solver_orders() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let expected_orders: &[&[u8]] = &[
        &[1],
        &[2],
        &[2, 1],
        &[3, 1],
        &[3, 2],
        &[3, 2, 1],
        &[3, 3, 1],
    ];
    for (offset, expected) in expected_orders.iter().enumerate() {
        let steps = u32::try_from(offset + 1)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &[2], &[0.8, -1.1], &context)?;
        let mut sigmas = Vec::new();
        sigmas.try_reserve_exact(usize::try_from(steps)? + 1)?;
        for index in 0..=steps {
            sigmas.push(2.0 - 1.5 * index as f32 / steps as f32);
        }
        let observed = RefCell::new(vec![0_u8; expected.len()]);
        let (trace, checkpoints) = sample_dpm_fast(
            &backend,
            plan(71, steps)?,
            &profile()?,
            initial,
            &sigmas,
            DpmFastOptions::default(),
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            &context,
            |input, sigma, interval, evaluation| {
                let interval = usize::try_from(interval).map_err(|error| error.to_string())?;
                let mut observed = observed.borrow_mut();
                let slot = observed
                    .get_mut(interval)
                    .ok_or_else(|| format!("unexpected solver interval {interval}"))?;
                *slot = slot
                    .checked_add(1)
                    .ok_or_else(|| "evaluation count overflow".to_owned())?;
                let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
                let denoised = analytical_denoised(&input, sigma);
                if evaluation >= 3 {
                    return Err(format!("unexpected evaluation ordinal {evaluation}"));
                }
                tensor_from_f32(&backend, &[2], &denoised, &context)
                    .map_err(|error| error.to_string())
            },
            |progress, _, _| {
                assert_eq!(progress.total_steps, steps);
                Ok::<(), String>(())
            },
        )?;
        assert_eq!(observed.into_inner(), *expected);
        assert_eq!(trace.denoiser_evaluations.len(), expected.len());
        let (actual_before, actual_after) =
            checkpoints.ok_or("missing fixed-step RNG checkpoints")?;
        let mut oracle = open_noise_transaction(
            noise_request(&fixture, 0, RetryRngPolicy::Replay),
            71,
            &cancellation,
        )?;
        assert_eq!(actual_before, oracle.checkpoint());
        for _ in *expected {
            oracle.draw_normal(2, &cancellation)?;
        }
        assert_eq!(actual_after, oracle.commit());
    }
    Ok(())
}

#[test]
fn dpm_fast_short_schedule_and_failures_publish_no_partial_state() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &fixture.initial, &context)?;
    let (short, checkpoints) = sample_dpm_fast(
        &backend,
        plan(fixture.seed, 1)?,
        &profile()?,
        initial.clone(),
        &[2.0],
        DpmFastOptions::default(),
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        &context,
        |_, _, _, _| Err("denoiser must not run".to_owned()),
        |_, _, _| Err::<(), _>("callback must not run"),
    )?;
    assert!(checkpoints.is_none());
    assert!(short.denoiser_evaluations.is_empty());
    assert_eq!(short.latents.len(), 1);
    assert_close(
        &values(
            &backend,
            short.latents.first().ok_or("missing short latent")?,
            &context,
        )?,
        &fixture.initial,
        0.0,
    );

    let error = match sample_dpm_fast(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        &profile()?,
        initial.clone(),
        &fixture.sigmas,
        DpmFastOptions::new(fixture.options.eta, fixture.options.noise_scale)?,
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        &context,
        |_, _, interval, evaluation| {
            if interval == 0 && evaluation == 1 {
                Err("injected denoiser failure".to_owned())
            } else {
                Ok(initial.clone())
            }
        },
        |_, _, _| Ok::<(), String>(()),
    ) {
        Ok(_) => return Err("injected denoiser failure unexpectedly succeeded".into()),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        DpmFastSamplerError::Denoiser {
            interval: 0,
            evaluation: 1,
            ..
        }
    ));

    cancellation.cancel();
    let cancelled = match sample_dpm_fast(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        &profile()?,
        initial,
        &fixture.sigmas,
        DpmFastOptions::default(),
        noise_request(&fixture, 0, RetryRngPolicy::Replay),
        &context,
        |_, _, _, _| Err("denoiser must not run".to_owned()),
        |_, _, _| Ok::<(), String>(()),
    ) {
        Ok(_) => return Err("pre-cancelled sampling unexpectedly succeeded".into()),
        Err(error) => error,
    };
    assert!(matches!(
        cancelled,
        DpmFastSamplerError::Tensor(TensorError::Cancelled)
    ));
    Ok(())
}

#[test]
fn dpm_fast_retry_policy_replays_or_advances_only_the_canonical_stream()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let run = |retry, policy| -> Result<Vec<f32>, Box<dyn Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = execution_context(&backend, &authority, &cancellation)?;
        let initial = tensor_from_f32(&backend, &[2], &fixture.initial, &context)?;
        let (trace, _) = sample_dpm_fast(
            &backend,
            plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
            &profile()?,
            initial,
            &fixture.sigmas,
            DpmFastOptions::new(fixture.options.eta, fixture.options.noise_scale)?,
            noise_request(&fixture, retry, policy),
            &context,
            |input, sigma, _, _| {
                let input = values(&backend, input, &context).map_err(|error| error.to_string())?;
                tensor_from_f32(
                    &backend,
                    &[2],
                    &analytical_denoised(&input, sigma),
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        )?;
        values(
            &backend,
            trace.latents.last().ok_or("missing output")?,
            &context,
        )
    };
    let replay_zero = run(0, RetryRngPolicy::Replay)?;
    let replay_three = run(3, RetryRngPolicy::Replay)?;
    assert_close(&replay_zero, &replay_three, 0.0);
    let advance = run(3, RetryRngPolicy::Advance)?;
    assert_ne!(
        replay_zero
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        advance
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}
