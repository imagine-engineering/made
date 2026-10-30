use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, PredictionInterpretation, SamplerIdentity,
    SamplerRegistry, SamplingPlan, SamplingProfileIdentity,
    generated_sa_solver_sim_model_0197::{
        SA_SOLVER_NOISE_CONTRACT_ID, SaSolverError, SaSolverEvaluation, SaSolverFamilyOptions,
        SaSolverOptions, sample_sa_solver_family, source_default_tau_interval,
    },
    generated_sa_solver_pece_sim_model_0198::{
        DEFINITION, SA_SOLVER_PECE_FEATURE_ID, SA_SOLVER_PECE_SAMPLER_ID,
        SA_SOLVER_PECE_SOURCE_ORDINAL, sample_sa_solver_pece,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, RetryRngPolicy,
    StreamId, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/sa_solver_pece_comfy_model_0198/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/sa_solver_pece_sim_model_0198.rs"
));
const FAMILY_IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/sa_solver_sim_model_0197.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    seed: u64,
    shape: Vec<u64>,
    profile_sigmas: Vec<f32>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    denoised: Vec<Vec<f32>>,
    expected_evaluations: Vec<(usize, String)>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    sampling_path: String,
    sampling_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
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
        SamplingProfileIdentity::new("analytical-sa-solver-pece-row-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas.clone()),
    )?)
}

fn plan(fixture: &Fixture, identity: &str) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        identity,
        "normal",
        SamplingProfileIdentity::new("analytical-sa-solver-pece-row-v1")?,
        fixture.seed,
        u32::try_from(fixture.denoised.len())?,
        1.0,
        1.0,
    )?)
}

fn request(retry: u32) -> CompatibilityNoiseRequest {
    CompatibilityNoiseRequest::new(
        "sa-solver-pece-fixture-v1",
        "attempt-0198",
        "KSampler-40",
        40,
        198,
        2,
        retry,
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

#[test]
fn definition_provenance_and_single_family_owner_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, SA_SOLVER_PECE_SAMPLER_ID);
    assert_eq!(fixture.feature_id, SA_SOLVER_PECE_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, SA_SOLVER_PECE_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.source_ordinal, 40);
    assert!(DEFINITION.stochastic);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert_eq!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new(SA_SOLVER_PECE_SAMPLER_ID)?)?,
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
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let sampling = fs::read_to_string(root.join(&fixture.source.sampling_path))?;
    let wrapper = sampling
        .lines()
        .skip(1838)
        .take(4)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(wrapper.contains("def sample_sa_solver_pece("));
    assert!(wrapper.contains("return sample_sa_solver("));
    assert!(wrapper.contains("use_pece=True"));
    for adapter in [
        "sample_sa_solver_family(",
        "SA_SOLVER_PECE_SAMPLER_ID",
        "SaSolverFamilyOptions::new(options, true)",
        "source_default_tau_interval(profile)",
    ] {
        assert!(
            IMPLEMENTATION.contains(adapter),
            "missing adapter {adapter}"
        );
    }
    for owner in [
        "SamplingSession::new",
        ".observe_step(",
        "stochastic_adams_coefficients(",
        ".open_transaction(",
        ".draw_normal(",
        ".commit(",
    ] {
        assert!(
            FAMILY_IMPLEMENTATION.contains(owner),
            "family missing {owner}"
        );
        assert!(
            !IMPLEMENTATION.contains(owner),
            "adapter duplicates {owner}"
        );
    }
    assert!(!IMPLEMENTATION.contains("Command::new"));
    assert!(!IMPLEMENTATION.contains("todo!"));
    assert!(!IMPLEMENTATION.contains("unimplemented!"));
    Ok(())
}

#[test]
fn adapter_matches_the_authoritative_pece_family_and_every_evaluation() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let adapter_events = RefCell::new(Vec::new());
    let adapter = sample_sa_solver_pece(
        &backend,
        plan(&fixture, SA_SOLVER_PECE_SAMPLER_ID)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(0),
        SaSolverOptions::new(0.0, 3, 4, false)?,
        &context,
        |_input, _sigma, step, evaluation| {
            adapter_events.borrow_mut().push((step, evaluation));
            tensor_from_f32(&backend, &fixture.shape, &fixture.denoised[step], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    let direct_events = RefCell::new(Vec::new());
    let (start_sigma, end_sigma) = source_default_tau_interval(&profile)?;
    let direct = sample_sa_solver_family(
        &backend,
        plan(&fixture, SA_SOLVER_PECE_SAMPLER_ID)?,
        &profile,
        SA_SOLVER_PECE_SAMPLER_ID,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(0),
        SaSolverFamilyOptions::new(SaSolverOptions::new(0.0, 3, 4, false)?, true),
        &context,
        |sigma, _step| {
            Ok(if start_sigma >= sigma && sigma >= end_sigma {
                1.0
            } else {
                0.0
            })
        },
        |_input, _sigma, step, evaluation| {
            direct_events.borrow_mut().push((step, evaluation));
            tensor_from_f32(&backend, &fixture.shape, &fixture.denoised[step], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _current, _denoised| Ok::<(), String>(()),
    )?;
    assert_eq!(adapter.1, direct.1);
    assert_eq!(adapter.0.sigmas, direct.0.sigmas);
    assert_eq!(
        adapter.0.denoiser_evaluations.len(),
        direct.0.denoiser_evaluations.len()
    );
    for (adapter, direct) in adapter
        .0
        .denoiser_evaluations
        .iter()
        .zip(&direct.0.denoiser_evaluations)
    {
        assert_eq!(
            values(&backend, adapter, &context)?,
            values(&backend, direct, &context)?
        );
    }
    for (adapter, direct) in adapter.0.latents.iter().zip(&direct.0.latents) {
        assert_eq!(
            values(&backend, adapter, &context)?,
            values(&backend, direct, &context)?
        );
    }
    assert_eq!(
        adapter_events.borrow().as_slice(),
        direct_events.borrow().as_slice()
    );
    let expected = fixture
        .expected_evaluations
        .iter()
        .map(|(step, evaluation)| {
            let evaluation = match evaluation.as_str() {
                "Predictor" => SaSolverEvaluation::Predictor,
                "Corrected" => SaSolverEvaluation::Corrected,
                value => panic!("unexpected evaluation {value}"),
            };
            (*step, evaluation)
        })
        .collect::<Vec<_>>();
    assert_eq!(adapter_events.into_inner(), expected);
    Ok(())
}

#[test]
fn failures_cancellation_and_rng_contract_remain_family_owned() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let wrong = sample_sa_solver_pece(
        &backend,
        plan(&fixture, "lms")?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(0),
        SaSolverOptions::default(),
        &context,
        |_, _, _, _| Err("must not run".to_owned()),
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(wrong, Err(SaSolverError::WrongSampler { .. })));

    let denoiser_failure = sample_sa_solver_pece(
        &backend,
        plan(&fixture, SA_SOLVER_PECE_SAMPLER_ID)?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        request(0),
        SaSolverOptions::new(0.0, 3, 4, false)?,
        &context,
        |_input, _sigma, step, evaluation| {
            if evaluation == SaSolverEvaluation::Corrected {
                return Err("injected PECE evaluation failure".to_owned());
            }
            tensor_from_f32(&backend, &fixture.shape, &fixture.denoised[step], &context)
                .map_err(|error| error.to_string())
        },
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        denoiser_failure,
        Err(SaSolverError::Denoiser {
            step: 1,
            evaluation: SaSolverEvaluation::Corrected,
            ..
        })
    ));

    let cancelled_initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    cancellation.cancel();
    let cancelled = sample_sa_solver_pece(
        &backend,
        plan(&fixture, SA_SOLVER_PECE_SAMPLER_ID)?,
        &profile,
        cancelled_initial,
        &fixture.sigmas,
        request(0),
        SaSolverOptions::default(),
        &context,
        |_, _, _, _| Err("must not run".to_owned()),
        |_, _, _| Ok::<(), String>(()),
    );
    assert!(matches!(
        cancelled,
        Err(SaSolverError::Tensor(TensorError::Cancelled))
    ));
    assert_eq!(SA_SOLVER_NOISE_CONTRACT_ID, "COMFY-RNG-B35F0F617BFA");
    Ok(())
}
