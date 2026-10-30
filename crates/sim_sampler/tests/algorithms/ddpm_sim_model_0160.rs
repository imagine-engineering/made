use sim_sampler::{
    CompatibilityNoiseRequest, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfileIdentity,
    generated_ddpm_sim_model_0160::{
        DDPM_NOISE_CONTRACT_ID, DDPM_SAMPLER_ID, DEFINITION, DdpmSamplerError, sample_ddpm,
    },
};
use sim_tensor::{
    CancellationToken, CompatibilityRngTransaction, CpuBackend, CpuWorkspaceAuthority, DeviceId,
    ExecutionContext, RetryRngPolicy, RngCompatibilityError, RngCompatibilityPhase,
    RngGenerationPlacement, RngSeedTransform, StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, path::Path};

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    source: SourceFixture,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    seed: u64,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    denoised: Vec<Vec<f32>>,
    noises: Vec<Vec<f32>>,
    latents: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    path: String,
    sha256: String,
    step_lines: [u32; 2],
    loop_lines: [u32; 2],
    noise_lines: [u32; 2],
}

fn fixture() -> Result<Fixture, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/samplers/ddpm_comfy_model_0160/oracle.json"
    )))?)
}

fn workspace_root() -> Result<&'static Path, Box<dyn std::error::Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "workspace root unavailable".into())
}

fn execution_context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
    bytes: u64,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(bytes)?,
        cancellation,
    ))
}

fn values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32(backend, tensor, context)?.to_vec())
}

fn assert_f32_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {index}: actual {actual:?}, expected {expected:?}"
        );
    }
}

fn base_noise_request() -> CompatibilityNoiseRequest {
    noise_request(2, RetryRngPolicy::Replay)
}

fn noise_request(retry: u32, retry_policy: RetryRngPolicy) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "ddpm-fixture-v1",
        "attempt-0160",
        "KSampler-25",
        25,
        160,
        99,
        retry,
        retry_policy,
    )
}

fn open_noise_transaction(
    request: CompatibilityNoiseRequest,
    seed: u64,
    cancellation: &CancellationToken,
) -> Result<CompatibilityRngTransaction, RngCompatibilityError> {
    request.open_transaction(
        DDPM_NOISE_CONTRACT_ID,
        i128::from(seed),
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        cancellation,
    )
}

fn plan(seed: u64, steps: u32) -> Result<SamplingPlan, Box<dyn std::error::Error>> {
    Ok(SamplingPlan::new(
        "ddpm",
        "normal",
        SamplingProfileIdentity::new("analytical-epsilon-v1")?,
        seed,
        steps,
        1.0,
        1.0,
    )?)
}

#[test]
fn ddpm_definition_and_oracle_provenance_are_source_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DDPM_SAMPLER_ID);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert!(DEFINITION.aliases.is_empty());
    assert!(DEFINITION.stochastic);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/ddpm_comfy_model_0160"
    );
    assert_eq!(fixture.source.step_lines, [984, 991]);
    assert_eq!(fixture.source.loop_lines, [993, 1012]);
    assert_eq!(fixture.source.noise_lines, [77, 88]);
    let source_path = workspace_root()?.join(&fixture.source.path);
    assert_eq!(
        format!("{:x}", Sha256::digest(std::fs::read(source_path)?)),
        fixture.source.sha256
    );
    let registry = SamplerRegistry::foundational()?;
    let resolved = registry.resolve(&sim_sampler::SamplerIdentity::new("ddpm")?)?;
    assert_eq!(resolved, &DEFINITION);
    Ok(())
}

#[test]
fn ddpm_matches_every_analytical_intermediate_noise_and_callback()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 512 * 1024)?;
    let initial = tensor_from_f32(&backend, &[1, 1, 2, 2], &fixture.initial, &context)?;
    let base_request = base_noise_request();
    let mut oracle_transaction =
        open_noise_transaction(base_request.clone(), fixture.seed, &cancellation)?;
    assert_eq!(
        oracle_transaction.contract().rng_id(),
        DDPM_NOISE_CONTRACT_ID
    );
    assert_eq!(
        oracle_transaction.contract().phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(
        oracle_transaction.contract().phase().as_str(),
        RngCompatibilityPhase::SamplingNoiseAndSolver.as_str()
    );
    let oracle_before = oracle_transaction.checkpoint();
    let mut observed_noises = Vec::new();
    observed_noises.try_reserve_exact(fixture.noises.len())?;
    for expected in &fixture.noises {
        let actual = oracle_transaction
            .draw_normal(expected.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_f32_bits(&actual, expected);
        observed_noises.push(actual);
    }
    let oracle_after = oracle_transaction.commit();
    let mut replay = open_noise_transaction(
        noise_request(0, RetryRngPolicy::Replay),
        fixture.seed,
        &cancellation,
    )?;
    assert_eq!(replay.checkpoint(), oracle_before);
    for expected in &observed_noises {
        let actual = replay
            .draw_normal(expected.len(), &cancellation)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        assert_f32_bits(&actual, expected);
    }
    assert_eq!(replay.commit(), oracle_after);

    let events = RefCell::new(Vec::new());
    let denoised = &fixture.denoised;
    let (sampling, noise_before, noise_after) = sample_ddpm(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        initial,
        &fixture.sigmas,
        base_request,
        &context,
        |_, _, step| {
            events.borrow_mut().push(format!("denoiser-{step}"));
            tensor_from_f32(&backend, &[1, 1, 2, 2], &denoised[step], &context)
                .map_err(|error| error.to_string())
        },
        |progress, current, denoised| {
            let step = usize::try_from(progress.step).map_err(|error| error.to_string())?;
            events.borrow_mut().push(format!("callback-{step}"));
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.sigmas.len() - 1)
                    .map_err(|error| error.to_string())?
            );
            assert_eq!(progress.sigma.to_bits(), fixture.sigmas[step].to_bits());
            assert_eq!(
                progress.next_sigma.to_bits(),
                fixture.sigmas[step + 1].to_bits()
            );
            assert_f32_bits(
                &values(&backend, current, &context).map_err(|error| error.to_string())?,
                &fixture.latents[step],
            );
            assert_f32_bits(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &fixture.denoised[step],
            );
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(noise_before, oracle_before);
    assert_eq!(noise_after, oracle_after);
    assert_eq!(sampling.sigmas, fixture.sigmas);
    assert_eq!(
        events.into_inner(),
        [
            "denoiser-0",
            "callback-0",
            "denoiser-1",
            "callback-1",
            "denoiser-2",
            "callback-2"
        ]
    );
    assert_eq!(sampling.denoiser_evaluations.len(), fixture.denoised.len());
    assert_eq!(sampling.latents.len(), fixture.latents.len());
    for (actual, expected) in sampling.denoiser_evaluations.iter().zip(&fixture.denoised) {
        assert_f32_bits(&values(&backend, actual, &context)?, expected);
    }
    for (actual, expected) in sampling.latents.iter().zip(&fixture.latents) {
        let actual = values(&backend, actual, &context)?;
        assert_f32_bits(&actual, expected);
    }
    Ok(())
}

#[test]
fn ddpm_failures_are_typed_cancelled_and_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 512 * 1024)?;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    let callback_count = RefCell::new(0_usize);
    let denoiser_count = RefCell::new(0_usize);
    let result = sample_ddpm(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        initial,
        &fixture.sigmas,
        base_noise_request(),
        &context,
        |_, _, step| {
            *denoiser_count.borrow_mut() += 1;
            tensor_from_f32(&backend, &[4], &fixture.denoised[step], &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| {
            *callback_count.borrow_mut() += 1;
            Err("injected callback failure")
        },
    );
    assert!(matches!(
        result,
        Err(DdpmSamplerError::Sampling(SamplingError::Callback(reason)))
            if reason == "injected callback failure"
    ));
    assert_eq!(*callback_count.borrow(), 1);
    assert_eq!(*denoiser_count.borrow(), 1);
    let mut rollback_replay =
        open_noise_transaction(base_noise_request(), fixture.seed, &cancellation)?;
    let replayed = rollback_replay
        .draw_normal(fixture.initial.len(), &cancellation)?
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    assert_f32_bits(&replayed, &fixture.noises[0]);

    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    let wrong_sampler_denoiser_ran = RefCell::new(false);
    let wrong_plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::new("analytical-epsilon-v1")?,
        fixture.seed,
        u32::try_from(fixture.sigmas.len() - 1)?,
        1.0,
        1.0,
    )?;
    assert!(matches!(
        sample_ddpm(
            &backend,
            wrong_plan,
            initial,
            &fixture.sigmas,
            base_noise_request(),
            &context,
            |value, _, _| {
                *wrong_sampler_denoiser_ran.borrow_mut() = true;
                Ok(value.clone())
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(DdpmSamplerError::WrongSampler(identity)) if identity == "euler"
    ));
    assert!(!*wrong_sampler_denoiser_ran.borrow());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled, 512 * 1024)?;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    let denoiser_ran = RefCell::new(false);
    let result = sample_ddpm(
        &backend,
        plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
        initial,
        &fixture.sigmas,
        base_noise_request(),
        &cancelled_context,
        |_, _, _| {
            *denoiser_ran.borrow_mut() = true;
            Err("must not run".to_owned())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(result, Err(DdpmSamplerError::Tensor(TensorError::Cancelled))));
    assert!(!*denoiser_ran.borrow());

    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    assert!(matches!(
        sample_ddpm(
            &backend,
            plan(fixture.seed, 1)?,
            initial,
            &[1.0, 1.0],
            base_noise_request(),
            &context,
            |_, _, _| Err("invalid schedule must fail first".to_owned()),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(DdpmSamplerError::Sampling(SamplingError::InvalidSigma { step: 0, .. }))
    ));

    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    assert!(matches!(
        sample_ddpm(
            &backend,
            plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
            initial,
            &fixture.sigmas,
            base_noise_request(),
            &context,
            |_, _, step| Err(format!("fixture denoiser failure {step}")),
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(DdpmSamplerError::Denoiser { step: 0, .. })
    ));

    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    assert!(matches!(
        sample_ddpm(
            &backend,
            plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
            initial,
            &fixture.sigmas,
            base_noise_request(),
            &context,
            |_, _, _| {
                tensor_from_f32(&backend, &[2, 2], &fixture.denoised[0], &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(DdpmSamplerError::Sampling(SamplingError::TensorContract {
            role: "denoiser output",
            ..
        }))
    ));

    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    assert!(matches!(
        sample_ddpm(
            &backend,
            plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
            initial,
            &fixture.sigmas,
            base_noise_request(),
            &context,
            |_, _, _| {
                tensor_from_f32(
                    &backend,
                    &[4],
                    &[f32::NAN, 0.0, 0.0, 0.0],
                    &context,
                )
                .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(DdpmSamplerError::NonFinite {
            step: 0,
            element: 0,
        })
    ));

    let constrained_context = execution_context(&backend, &authority, &cancellation, 15)?;
    let initial = tensor_from_f32(&backend, &[4], &fixture.initial, &context)?;
    assert!(matches!(
        sample_ddpm(
            &backend,
            plan(fixture.seed, u32::try_from(fixture.sigmas.len() - 1)?)?,
            initial,
            &fixture.sigmas,
            base_noise_request(),
            &constrained_context,
            |_, _, step| {
                tensor_from_f32(&backend, &[4], &fixture.denoised[step], &context)
                    .map_err(|error| error.to_string())
            },
            |_, _, _| Ok::<(), String>(()),
        ),
        Err(DdpmSamplerError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded {
                requested: 16,
                authorized: 15,
                ..
            }
        ))
    ));
    Ok(())
}
