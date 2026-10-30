use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    SamplingProgress,
    generated_deis_sim_model_0161::{
        DEFINITION, DEIS_MAX_ORDER, DEIS_SAMPLER_FEATURE_ID, DEIS_SAMPLER_ID,
        DEIS_TABULATION_POINTS, DeisSamplerError, sample_deis,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::Cell, error::Error, fs, path::PathBuf};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/deis_comfy_model_0161/trajectory.json"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    mode: String,
    max_order: usize,
    tabulation_points: usize,
    tolerance: f32,
    sigmas: Vec<f32>,
    noise: Vec<f32>,
    initial: Vec<f32>,
    denoiser_outputs: Vec<Vec<f32>>,
    derivatives: Vec<Vec<f32>>,
    coefficients: Vec<Vec<f32>>,
    latents: Vec<Vec<f32>>,
    callbacks: Vec<CallbackFixture>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    coefficient_path: String,
    coefficient_sha256: String,
    sampler_path: String,
    sampler_sha256: String,
    catalog_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CallbackFixture {
    step: u32,
    sigma: f32,
    next_sigma: f32,
    current: Vec<f32>,
    denoised: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn file_digest(path: &str) -> Result<String, Box<dyn Error>> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(workspace()?.join(path))?)
    ))
}

fn plan(sampler: &str, steps: u32) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        sampler,
        "normal",
        SamplingProfileIdentity::new("deis-row-fixture-v1")?,
        0x0161,
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

fn fixture_profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("deis-row-fixture-v1")?)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}, tolerance {tolerance}"
        );
    }
}

#[test]
fn val_sampler_001_deis_definition_is_exact_and_not_an_alias() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DEIS_SAMPLER_ID);
    assert_eq!(fixture.feature_id, DEIS_SAMPLER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, DEFINITION.source_ordinal);
    assert_eq!(fixture.mode, "tab");
    assert_eq!(fixture.max_order, DEIS_MAX_ORDER);
    assert_eq!(fixture.tabulation_points, DEIS_TABULATION_POINTS);
    assert!(fixture.noise.is_empty());
    assert_eq!(DEFINITION.identity, "deis");
    assert_eq!(DEFINITION.feature_id, "COMFY-MODEL-0161");
    assert_eq!(DEFINITION.source_ordinal, 29);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        DEFINITION.implementation_module,
        "algorithms/deis_comfy_model_0161"
    );
    assert!(!DEFINITION.stochastic);
    let registry = SamplerRegistry::foundational()?;
    let resolved = registry.resolve(&SamplerIdentity::new("deis")?)?;
    assert_eq!(resolved, &DEFINITION);
    assert!(
        SamplerRegistry::foundational()?
            .resolve(&SamplerIdentity::new("deis_tab")?)
            .is_err()
    );
    assert_eq!(
        fixture.source.coefficient_path,
        "projects/comfy/ComfyUI/sim/k_diffusion/deis.py"
    );
    assert_eq!(
        fixture.source.coefficient_sha256,
        "8d4e5057a062b77ef2e33af8208a2d6a3c3fbb431315a4c37689a307d66bc807"
    );
    assert_eq!(
        fixture.source.sampler_path,
        "projects/comfy/ComfyUI/sim/k_diffusion/sampling.py"
    );
    assert_eq!(
        fixture.source.sampler_sha256,
        "cc5f944efd85c566484c3999beb74e8c19c894f2a50ca574d090c3a46ac6bd06"
    );
    assert_eq!(
        fixture.source.catalog_sha256,
        "12afc10d34d0eb7a103b648d0d7a21797bf5fc687cec73b5bb0a6a7f7220e7b4"
    );
    assert_eq!(
        file_digest(&fixture.source.coefficient_path)?,
        fixture.source.coefficient_sha256
    );
    assert_eq!(
        file_digest(&fixture.source.sampler_path)?,
        fixture.source.sampler_sha256
    );
    assert_eq!(
        file_digest("crates/sim_test_support/parity/catalogs/backend-models.csv")?,
        fixture.source.catalog_sha256
    );
    Ok(())
}

#[test]
fn val_sampler_001_deis_matches_every_analytical_intermediate() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &fixture.initial, &context)?;
    let denoiser_outputs = fixture.denoiser_outputs.clone();
    let mut denoiser_inputs = Vec::new();
    let mut callbacks = Vec::new();
    let trace = sample_deis(
        &backend,
        plan("deis", u32::try_from(fixture.sigmas.len() - 1)?)?,
        &fixture_profile()?,
        initial,
        &fixture.sigmas,
        &context,
        |current, sigma, step| {
            let values =
                tensor_to_f32(&backend, current, &context).map_err(|error| error.to_string())?;
            denoiser_inputs.push((step, sigma, values.to_vec()));
            let output = denoiser_outputs
                .get(step)
                .ok_or_else(|| format!("missing denoiser output for step {step}"))?;
            tensor_from_f32(&backend, &[2], output, &context).map_err(|error| error.to_string())
        },
        |progress: &SamplingProgress, current, denoised| {
            let current = tensor_to_f32(&backend, current, &context)
                .map_err(|error| error.to_string())?
                .to_vec();
            let denoised = tensor_to_f32(&backend, denoised, &context)
                .map_err(|error| error.to_string())?
                .to_vec();
            callbacks.push(CallbackFixture {
                step: progress.step,
                sigma: progress.sigma,
                next_sigma: progress.next_sigma,
                current,
                denoised,
            });
            Ok(())
        },
    )?;

    assert_eq!(trace.sigmas, fixture.sigmas);
    assert_eq!(denoiser_inputs.len(), fixture.denoiser_outputs.len());
    assert_eq!(
        trace.denoiser_evaluations.len(),
        fixture.denoiser_outputs.len()
    );
    assert_eq!(trace.latents.len(), fixture.latents.len());
    for (step, (input_step, sigma, input)) in denoiser_inputs.iter().enumerate() {
        assert_eq!(*input_step, step);
        assert_eq!(*sigma, fixture.sigmas[step]);
        let expected_input = fixture
            .latents
            .get(step)
            .ok_or("missing expected denoiser input")?;
        assert_close(input, expected_input, fixture.tolerance);
    }
    for (actual, expected) in trace
        .denoiser_evaluations
        .iter()
        .zip(fixture.denoiser_outputs.iter())
    {
        assert_close(
            &tensor_to_f32(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    for (actual, expected) in trace.latents.iter().zip(fixture.latents.iter()) {
        assert_close(
            &tensor_to_f32(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    for (actual, expected) in callbacks.iter().zip(fixture.callbacks.iter()) {
        assert_eq!(actual.step, expected.step);
        assert_eq!(actual.sigma, expected.sigma);
        assert_eq!(actual.next_sigma, expected.next_sigma);
        assert_close(&actual.current, &expected.current, fixture.tolerance);
        assert_close(&actual.denoised, &expected.denoised, fixture.tolerance);
    }

    assert_eq!(fixture.coefficients.len(), fixture.sigmas.len() - 1);
    assert!(fixture.coefficients.first().is_some_and(Vec::is_empty));
    assert!(fixture.coefficients.last().is_some_and(Vec::is_empty));
    for (step, expected) in fixture.derivatives.iter().enumerate() {
        let current = fixture
            .latents
            .get(step)
            .ok_or("missing derivative input latent")?;
        let denoised = fixture
            .denoiser_outputs
            .get(step)
            .ok_or("missing derivative denoiser output")?;
        let sigma = *fixture.sigmas.get(step).ok_or("missing derivative sigma")?;
        let actual = current
            .iter()
            .zip(denoised.iter())
            .map(|(current, denoised)| (current - denoised) / sigma)
            .collect::<Vec<_>>();
        assert_close(&actual, expected, fixture.tolerance);
    }
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_sampler_001_deis_failures_are_typed_atomic_and_cancellable() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &[2], &[1.0, -1.0], &context)?;

    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::ZeroSteps)
    ));
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 1.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::Sampling(SamplingError::InvalidSigma {
            step: 0,
            ..
        }))
    ));
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 2)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::Sampling(SamplingError::ScheduleLength {
            expected: 3,
            actual: 2
        }))
    ));
    let wrong_profile = SamplingProfileIdentity::new("wrong-profile-v1")?;
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &wrong_profile,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::Sampling(
            SamplingError::ProfileMismatch { .. }
        ))
    ));
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[f32::INFINITY, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::Sampling(SamplingError::InvalidSigma {
            step: 0,
            ..
        }))
    ));
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| tensor_from_f32(&backend, &[1], &[0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::DenoiserContract { step: 0 })
    ));
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| Err("fixture denoiser fault".to_owned()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::Denoiser { step: 0, .. })
    ));
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |_, _, _| tensor_from_f32(&backend, &[2], &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::NonFinite {
            step: 0,
            stage: "denoiser",
            element: 0
        })
    ));

    let callback_count = Cell::new(0_u32);
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| {
                callback_count.set(callback_count.get() + 1);
                Err("fixture callback fault".to_owned())
            }
        ),
        Err(DeisSamplerError::Sampling(SamplingError::Callback(_)))
    ));
    assert_eq!(callback_count.get(), 1);

    let callback_cancellation = CancellationToken::default();
    let callback_cancellation_context =
        execution_context(&backend, &authority, &callback_cancellation)?;
    let callback_count = Cell::new(0_u32);
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial.clone(),
            &[1.0, 0.0],
            &callback_cancellation_context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| {
                callback_count.set(callback_count.get() + 1);
                callback_cancellation.cancel();
                Ok(())
            }
        ),
        Err(DeisSamplerError::Cancelled { step: 0 })
    ));
    assert_eq!(callback_count.get(), 1);
    assert_eq!(callback_cancellation_context.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let denoiser_called = Cell::new(false);
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 2)?,
            &fixture_profile()?,
            initial.clone(),
            &[2.0, 1.0, 0.0],
            &cancelled_context,
            |value, _, _| {
                denoiser_called.set(true);
                Ok(value.clone())
            },
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::Cancelled { step: 0 })
    ));
    assert!(!denoiser_called.get());

    let constrained = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(15)?,
        &cancellation,
    );
    assert!(matches!(
        sample_deis(
            &backend,
            plan("deis", 1)?,
            &fixture_profile()?,
            initial,
            &[1.0, 0.0],
            &constrained,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::TensorKernel(_))
    ));
    assert_eq!(constrained.scratch.in_use_bytes(), 0);

    assert!(matches!(
        sample_deis(
            &backend,
            plan("euler", 1)?,
            &fixture_profile()?,
            tensor_from_f32(&backend, &[1], &[0.0], &context)?,
            &[1.0, 0.0],
            &context,
            |value, _, _| Ok(value.clone()),
            |_, _, _| Ok(())
        ),
        Err(DeisSamplerError::SamplerIdentity { .. })
    ));
    Ok(())
}
