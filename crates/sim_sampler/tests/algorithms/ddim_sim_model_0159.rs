use sim_sampler::{
    CompatibilityNoiseRequest, SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan,
    SamplingProfileIdentity,
    generated_ddim_sim_model_0159::{
        DDIM_FEATURE_ID, DDIM_INPAINT_NOISE_CONTRACT_ID, DDIM_SAMPLER_ID, DDIM_SOURCE_ORDINAL,
        DEFINITION, DdimError, ddim_inpaint_replacement_noise, sample_ddim,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, RetryRngPolicy,
    RngAlgorithm, RngCompatibilityError, RngCompatibilityPhase, RngProfileVersion, StreamId,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    rng_compatibility_contract,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, error::Error, fs, path::Path};

const FIXTURE_BYTES: &[u8] = include_bytes!(
    "../../../sim_test_support/fixtures/samplers/ddim_comfy_model_0159/fixture.json"
);
const IMPLEMENTATION: &str = include_str!("../../src/algorithms/ddim_sim_model_0159.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    feature_id: String,
    sampler: String,
    source: SourceFixture,
    plan: PlanFixture,
    inpaint_replacement_noise: NoiseFixture,
    steps: Vec<StepFixture>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    samplers_path: String,
    samplers_sha256: String,
    sampling_path: String,
    sampling_sha256: String,
    dispatch_line: usize,
    equation_lines: Vec<usize>,
    inpaint_noise_lines: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct PlanFixture {
    scheduler: String,
    profile: String,
    seed: u64,
    sigmas: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct NoiseFixture {
    base_seed: i128,
    effective_seed: u64,
    phase: String,
    shape: Vec<u64>,
    values: Vec<f32>,
    value_bits: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct StepFixture {
    step: u32,
    sigma: f32,
    next_sigma: f32,
    latent: Vec<f32>,
    denoised: Vec<f32>,
    next_latent: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct CallbackObservation {
    step: u32,
    total_steps: u32,
    sigma: f32,
    next_sigma: f32,
    latent: Vec<f32>,
    denoised: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_slice(FIXTURE_BYTES)?)
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

fn noise_request() -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "fixture-workflow",
        "fixture-attempt",
        "KSampler",
        0,
        0,
        0,
        0,
        RetryRngPolicy::Replay,
    )
}

fn plan(fixture: &Fixture) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        fixture.sampler.clone(),
        fixture.plan.scheduler.clone(),
        SamplingProfileIdentity::new(fixture.plan.profile.clone())?,
        fixture.plan.seed,
        u32::try_from(fixture.steps.len()).map_err(|_| SamplingError::Overflow("fixture steps"))?,
        1.0,
        1.0,
    )?)
}

#[test]
fn definition_and_source_provenance_are_exact_and_unaliased() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.feature_id, DDIM_FEATURE_ID);
    assert_eq!(fixture.sampler, DDIM_SAMPLER_ID);
    assert_eq!(DDIM_INPAINT_NOISE_CONTRACT_ID, "COMFY-RNG-B35F0F617BFA");
    let noise_contract = rng_compatibility_contract(DDIM_INPAINT_NOISE_CONTRACT_ID)
        .ok_or("missing canonical DDIM noise contract")?;
    assert_eq!(
        noise_contract.phase(),
        RngCompatibilityPhase::SamplingNoiseAndSolver
    );
    assert_eq!(DEFINITION.identity, DDIM_SAMPLER_ID);
    assert_eq!(DEFINITION.feature_id, DDIM_FEATURE_ID);
    assert_eq!(DEFINITION.source_ordinal, DDIM_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/ddim_comfy_model_0159"
    );
    assert!(DEFINITION.stochastic);

    let registry = SamplerRegistry::foundational()?;
    let resolved = registry.resolve(&SamplerIdentity::new(DDIM_SAMPLER_ID)?)?;
    assert_eq!(resolved, &DEFINITION);
    assert!(
        registry
            .resolve(&SamplerIdentity::new("ddim_alias")?)
            .is_err()
    );
    assert!(SamplerIdentity::new("DDIM").is_err());

    let root = workspace_root()?;
    assert_eq!(
        digest(&root.join(&fixture.source.samplers_path))?,
        fixture.source.samplers_sha256
    );
    assert_eq!(
        digest(&root.join(&fixture.source.sampling_path))?,
        fixture.source.sampling_sha256
    );
    let samplers_source = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    let sampling_source = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    assert!(
        samplers_source
            .lines()
            .nth(fixture.source.dispatch_line - 1)
            .is_some_and(
                |line| line.contains("ksampler(\"euler\", inpaint_options={\"random\": True})")
            )
    );
    assert_eq!(
        fixture.source.equation_lines,
        [205, 206, 207, 208, 209, 211]
    );
    let equations = fixture
        .source
        .equation_lines
        .iter()
        .filter_map(|line_number| sampling_source.lines().nth(*line_number - 1))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "denoised = model",
        "to_d(",
        "callback(",
        "dt =",
        "x = x + d * dt",
    ] {
        assert!(
            equations.contains(fragment),
            "missing source equation {fragment}"
        );
    }
    assert_eq!(fixture.source.inpaint_noise_lines, [986, 987, 988]);
    let inpaint_noise = fixture
        .source
        .inpaint_noise_lines
        .iter()
        .filter_map(|line_number| samplers_source.lines().nth(*line_number - 1))
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in ["random", "torch.manual_seed", " + 1", "torch.randn"] {
        assert!(
            inpaint_noise.contains(fragment),
            "missing inpaint noise source {fragment}"
        );
    }
    Ok(())
}

#[test]
fn analytical_trajectory_noise_callbacks_and_aliases_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.inpaint_replacement_noise.effective_seed, 42);
    assert_eq!(
        fixture.inpaint_replacement_noise.phase,
        RngCompatibilityPhase::SamplingNoiseAndSolver.as_str()
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let replacement_noise = ddim_inpaint_replacement_noise(
        &backend,
        &fixture.inpaint_replacement_noise.shape,
        noise_request(),
        fixture.inpaint_replacement_noise.base_seed,
        &context,
    )?;
    assert_eq!(replacement_noise.before.algorithm, RngAlgorithm::Mt19937);
    assert_eq!(replacement_noise.before.profile, RngProfileVersion::V1);
    assert_ne!(replacement_noise.before, replacement_noise.after);
    assert_eq!(
        &*tensor_to_f32(&backend, &replacement_noise.noise, &context)?,
        fixture.inpaint_replacement_noise.values
    );
    assert_eq!(
        tensor_to_f32(&backend, &replacement_noise.noise, &context)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        fixture.inpaint_replacement_noise.value_bits
    );
    let replay_noise = ddim_inpaint_replacement_noise(
        &backend,
        &fixture.inpaint_replacement_noise.shape,
        noise_request(),
        fixture.inpaint_replacement_noise.base_seed,
        &context,
    )?;
    assert_eq!(replacement_noise.before, replay_noise.before);
    assert_eq!(replacement_noise.after, replay_noise.after);
    assert_eq!(
        &*tensor_to_f32(&backend, &replay_noise.noise, &context)?,
        fixture.inpaint_replacement_noise.values
    );
    drop(replay_noise);

    let initial = tensor_from_f32(&backend, &[2], &fixture.steps[0].latent, &context)?;
    let initial_alias = initial.clone();
    let initial_storage = initial.storage_id();
    let replacement_storage = replacement_noise.noise.storage_id();
    assert_ne!(initial_storage, replacement_storage);
    let denoiser_step = Cell::new(0_usize);
    let mut callbacks = Vec::new();
    let (sampling, inpaint_replacement_noise) = sample_ddim(
        &backend,
        plan(&fixture)?,
        initial,
        &fixture.plan.sigmas,
        replacement_noise,
        &context,
        |latent, noise, sigma, step| {
            let expected = fixture
                .steps
                .get(step)
                .ok_or_else(|| format!("unexpected denoiser step {step}"))?;
            if denoiser_step.get() != step
                || sigma != expected.sigma
                || tensor_to_f32(&backend, latent, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec()
                    != expected.latent
                || tensor_to_f32(&backend, noise, &context)
                    .map_err(|error| error.to_string())?
                    .to_vec()
                    != fixture.inpaint_replacement_noise.values
            {
                return Err(format!("denoiser observation diverged at step {step}"));
            }
            denoiser_step.set(step + 1);
            tensor_from_f32(&backend, &[2], &expected.denoised, &context)
                .map_err(|error| error.to_string())
        },
        |progress, denoised, latent| {
            callbacks.push(CallbackObservation {
                step: progress.step,
                total_steps: progress.total_steps,
                sigma: progress.sigma,
                next_sigma: progress.next_sigma,
                latent: tensor_to_f32(&backend, latent, &context)?.to_vec(),
                denoised: tensor_to_f32(&backend, denoised, &context)?.to_vec(),
            });
            Ok::<(), sim_tensor::generated_native_diffusion::NativeDiffusionTensorError>(())
        },
    )?;

    assert_eq!(denoiser_step.get(), fixture.steps.len());
    let total_steps = u32::try_from(fixture.steps.len())?;
    let expected_callbacks = fixture
        .steps
        .iter()
        .map(|step| CallbackObservation {
            step: step.step,
            total_steps,
            sigma: step.sigma,
            next_sigma: step.next_sigma,
            latent: step.latent.clone(),
            denoised: step.denoised.clone(),
        })
        .collect::<Vec<_>>();
    assert_eq!(callbacks, expected_callbacks);
    assert_eq!(sampling.sigmas, fixture.plan.sigmas);
    assert_eq!(sampling.denoiser_evaluations.len(), fixture.steps.len());
    assert_eq!(sampling.latents.len(), fixture.steps.len() + 1);
    for (index, latent) in sampling.latents.iter().enumerate() {
        let expected = if index == 0 {
            &fixture.steps[0].latent
        } else {
            &fixture.steps[index - 1].next_latent
        };
        assert_eq!(&*tensor_to_f32(&backend, latent, &context)?, expected);
    }
    assert_eq!(sampling.latents[0].storage_id(), initial_storage);
    assert_ne!(
        sampling
            .latents
            .last()
            .ok_or("missing final latent")?
            .storage_id(),
        initial_storage
    );
    assert_eq!(
        inpaint_replacement_noise.noise.storage_id(),
        replacement_storage
    );
    assert_eq!(
        &*tensor_to_f32(&backend, &initial_alias, &context)?,
        fixture.steps[0].latent
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn failures_cancellation_and_workspace_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let make_noise =
        || ddim_inpaint_replacement_noise(&backend, &[2], noise_request(), 41, &context);
    let initial = || tensor_from_f32(&backend, &[2], &[2.0, -2.0], &context);
    let denoise = |_: &_, _: &_, _: f32, _: usize| {
        tensor_from_f32(&backend, &[2], &[0.5, -0.5], &context).map_err(|error| error.to_string())
    };

    let wrong_plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::sd15(),
        41,
        2,
        1.0,
        1.0,
    )?;
    assert!(matches!(
        sample_ddim(
            &backend,
            wrong_plan,
            initial()?,
            &fixture.plan.sigmas,
            make_noise()?,
            &context,
            denoise,
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::WrongSampler(ref identity)) if identity == "euler"
    ));

    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &[1.0],
            make_noise()?,
            &context,
            denoise,
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::Sampling(SamplingError::ScheduleLength {
            expected: 3,
            actual: 1
        }))
    ));
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &[1.0, 1.0, 0.0],
            make_noise()?,
            &context,
            denoise,
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::Sampling(SamplingError::InvalidSigma {
            step: 0,
            ..
        }))
    ));
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &fixture.plan.sigmas,
            ddim_inpaint_replacement_noise(&backend, &[1], noise_request(), 41, &context)?,
            &context,
            denoise,
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::ReplacementNoiseContract { .. })
    ));

    let callback_count = Cell::new(0_u32);
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &fixture.plan.sigmas,
            make_noise()?,
            &context,
            denoise,
            |_, _, _| {
                callback_count.set(callback_count.get() + 1);
                Err("injected callback failure")
            }
        ),
        Err(DdimError::Sampling(SamplingError::Callback(ref reason)))
            if reason == "injected callback failure"
    ));
    assert_eq!(callback_count.get(), 1);

    let denoiser_count = Cell::new(0_u32);
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &fixture.plan.sigmas,
            make_noise()?,
            &context,
            |_, _, _, step| {
                denoiser_count.set(denoiser_count.get() + 1);
                Err(format!("denoiser-{step}"))
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::Denoiser { step: 0, ref reason }) if reason == "denoiser-0"
    ));
    assert_eq!(denoiser_count.get(), 1);

    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &fixture.plan.sigmas,
            make_noise()?,
            &context,
            |_, _, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::DenoiserContract { step: 0, .. })
    ));
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &fixture.plan.sigmas,
            make_noise()?,
            &context,
            |_, _, _, _| tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::NonFiniteOutput { step: 0, index: 0 })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancelled_denoiser_count = Cell::new(0_u32);
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            initial()?,
            &fixture.plan.sigmas,
            make_noise()?,
            &cancelled_context,
            |_, _, _, _| {
                cancelled_denoiser_count.set(cancelled_denoiser_count.get() + 1);
                Err("must not execute".to_owned())
            },
            |_, _, _| Ok::<(), String>(())
        ),
        Err(DdimError::Cancelled)
    ));
    assert_eq!(cancelled_denoiser_count.get(), 0);

    let commit_cancellation = CancellationToken::default();
    let commit_context = execution_context(&backend, &authority, &commit_cancellation)?;
    let commit_callbacks = Cell::new(0_u32);
    let later_denoisers = Cell::new(0_u32);
    assert!(matches!(
        sample_ddim(
            &backend,
            plan(&fixture)?,
            tensor_from_f32(&backend, &[2], &[2.0, -2.0], &commit_context)?,
            &fixture.plan.sigmas,
            ddim_inpaint_replacement_noise(&backend, &[2], noise_request(), 41, &commit_context,)?,
            &commit_context,
            |_, _, _, step| {
                later_denoisers.set(later_denoisers.get() + 1);
                tensor_from_f32(&backend, &[2], &[0.5, -0.5], &commit_context)
                    .map_err(|error| format!("step {step}: {error}"))
            },
            |_, _, _| {
                commit_callbacks.set(commit_callbacks.get() + 1);
                commit_cancellation.cancel();
                Ok::<(), String>(())
            }
        ),
        Err(DdimError::Sampling(SamplingError::Cancelled))
    ));
    assert_eq!(commit_callbacks.get(), 1);
    assert_eq!(later_denoisers.get(), 1);

    assert!(matches!(
        ddim_inpaint_replacement_noise(&backend, &[2], noise_request(), i128::MAX, &context,),
        Err(DdimError::RngCompatibility(
            RngCompatibilityError::InvalidSeed { .. }
        ))
    ));

    let (small_backend, small_authority) = CpuWorkspaceAuthority::create_backend(64)?;
    let small_cancellation = CancellationToken::default();
    let small_context = small_backend.execution_context(
        StreamId::DEFAULT,
        small_authority.authorize_workspace(7)?,
        &small_cancellation,
    );
    assert!(matches!(
        ddim_inpaint_replacement_noise(&small_backend, &[2], noise_request(), 41, &small_context,),
        Err(DdimError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(small_context.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    Ok(())
}

#[test]
fn row_delegates_every_foundational_owner() {
    for required in [
        "SamplingPlan",
        "sample_euler_canonical",
        "EulerOptions::source_defaults",
        "DDIM_SAMPLER_ID",
        "CompatibilityNoiseRequest",
        "request.open_transaction",
        "RngSeedTransform::Add(1)",
        "RngGenerationPlacement::CpuSeededTransfer",
        "transaction.draw_normal",
        "TensorDescriptor::contiguous",
        "backend.workspace_vec",
        ".cancellation",
        ".check()",
    ] {
        assert!(
            IMPLEMENTATION.contains(required),
            "missing owner {required}"
        );
    }
    for forbidden in [
        "pub struct samplingplan",
        "pub struct samplingsession",
        "pub struct rngstream",
        "pub struct cancellationtoken",
        "pub struct cpubackend",
        "pub struct ddimtrace",
        "fn normal_schedule",
        "fn ddim_scheduler",
        "address.phase",
        "native_rng_execution_profile",
        "rngstream::new",
        "std::fs",
        "serde_json",
        "python",
        "javascript",
        "compatibilityrngtransaction::open",
        "rngcompatibilityrequest::new",
        "sigmas.windows(2)",
        "derivative.mul_add",
        "(current - denoised) / sigma",
        "session.commit_step",
    ] {
        assert!(
            !IMPLEMENTATION.to_ascii_lowercase().contains(forbidden),
            "row duplicates or embeds forbidden owner {forbidden}"
        );
    }
}
