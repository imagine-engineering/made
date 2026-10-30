use sim_sampler::{
    DiscreteSamplingProfile, PenultimateSigmaPolicy, PredictionInterpretation,
    SamplingProfileIdentity, SchedulerError, SchedulerIdentity, SchedulerRegistry,
    SchedulerRequest,
    generated_karras_sim_model_0206::{
        DEFINITION, KARRAS_SCHEDULER_FEATURE_ID, KARRAS_SCHEDULER_ID,
        KARRAS_SCHEDULER_SOURCE_ORDINAL, karras_schedule,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/schedulers/karras_comfy_model_0206/schedule.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/schedulers/karras_sim_model_0206.rs");

#[derive(Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    profile_sigmas: Vec<f32>,
    tolerance: f32,
    cases: Vec<CaseFixture>,
}

#[derive(Deserialize)]
struct SourceFixture {
    equation_path: String,
    equation_sha256: String,
    equation_lines: Vec<usize>,
    registry_path: String,
    registry_sha256: String,
    registry_line: usize,
    catalog_path: String,
    catalog_sha256: String,
    catalog_line: usize,
}

#[derive(Deserialize)]
struct CaseFixture {
    name: String,
    steps: u32,
    denoise: f32,
    start_step: Option<u32>,
    end_step: Option<u32>,
    discard_penultimate: bool,
    expected: Vec<f32>,
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

fn digest(path: &str) -> Result<String, Box<dyn Error>> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(workspace_root()?.join(path))?)
    ))
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

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("karras-analytical-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas.clone()),
    )?)
}

fn assert_close(case: &CaseFixture, actual: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), case.expected.len(), "{} length", case.name);
    for (index, (actual, expected)) in actual.iter().zip(&case.expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{} sigma {index}: expected {expected}, got {actual}",
            case.name
        );
    }
}

#[test]
fn val_scheduler_001_karras_definition_and_provenance_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, KARRAS_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, KARRAS_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, KARRAS_SCHEDULER_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.implementation_module, "schedulers/karras_comfy_model_0206");
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(fixture.source.equation_lines, [23, 30]);
    assert_eq!(fixture.source.registry_line, 1349);
    assert_eq!(fixture.source.catalog_line, 207);
    assert_eq!(digest(&fixture.source.equation_path)?, fixture.source.equation_sha256);
    assert_eq!(digest(&fixture.source.registry_path)?, fixture.source.registry_sha256);
    assert_eq!(digest(&fixture.source.catalog_path)?, fixture.source.catalog_sha256);
    let registry = SchedulerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SchedulerIdentity::new(KARRAS_SCHEDULER_ID)?)?,
        &DEFINITION
    );
    Ok(())
}

#[test]
fn val_scheduler_001_karras_matches_source_equations_and_shared_slicing()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    for case in &fixture.cases {
        let mut request = SchedulerRequest::new(&fixture.identity, case.steps, case.denoise)?
            .with_window(case.start_step, case.end_step)?;
        if case.discard_penultimate {
            request = request.with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard);
        }
        let actual = karras_schedule(&backend, &context, &registry, &profile, &request)?;
        assert_close(case, &actual, fixture.tolerance);
    }
    assert!(
        !IMPLEMENTATION.contains("workspace_vec")
            && !IMPLEMENTATION.contains("start_step")
            && !IMPLEMENTATION.contains("end_step")
            && IMPLEMENTATION.contains("build_scheduler_schedule")
    );
    Ok(())
}

#[test]
fn val_scheduler_001_karras_rejects_invalid_or_cancelled_requests() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    assert_eq!(
        karras_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new("normal", 2, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: KARRAS_SCHEDULER_ID,
            actual: "normal".to_owned(),
        })
    );
    assert_eq!(
        SchedulerRequest::new(KARRAS_SCHEDULER_ID, 0, 1.0),
        Err(SchedulerError::ZeroSteps)
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    assert_eq!(
        karras_schedule(
            &backend,
            &cancelled_context,
            &registry,
            &profile,
            &SchedulerRequest::new(KARRAS_SCHEDULER_ID, 2, 1.0)?,
        ),
        Err(SchedulerError::Cancelled)
    );
    Ok(())
}
