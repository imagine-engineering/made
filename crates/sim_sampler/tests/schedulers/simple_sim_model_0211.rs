use sim_sampler::{
    DiscreteSamplingProfile, PredictionInterpretation, SIMPLE_SCHEDULER_FEATURE_ID,
    SIMPLE_SCHEDULER_ID, SamplingProfileIdentity, SchedulerError, SchedulerIdentity,
    SchedulerRegistry, SchedulerRequest,
    generated_simple_sim_model_0211::{DEFINITION, SIMPLE_SCHEDULER_SOURCE_ORDINAL, simple_schedule},
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path, sync::Arc};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/schedulers/simple_comfy_model_0211/schedule.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/schedulers/simple_sim_model_0211.rs");

#[derive(Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    profile_sigmas: Vec<f32>,
    cases: Vec<CaseFixture>,
}

#[derive(Deserialize)]
struct SourceFixture {
    path: String,
    sha256: String,
    equation_lines: Vec<usize>,
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
    expected: Vec<f32>,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    Ok(serde_json::from_str(FIXTURE_JSON)?)
}

fn root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "workspace root is unavailable".into())
}

fn digest(path: &str) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(root()?.join(path))?)))
}

fn context<'a>(
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

#[test]
fn val_scheduler_001_simple_definition_and_source_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, SIMPLE_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, SIMPLE_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, SIMPLE_SCHEDULER_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.implementation_module, "schedulers/simple_comfy_model_0211");
    assert_eq!(fixture.source.equation_lines, [644, 651]);
    assert_eq!(fixture.source.registry_line, 1347);
    assert_eq!(fixture.source.catalog_line, 212);
    assert_eq!(digest(&fixture.source.path)?, fixture.source.sha256);
    assert_eq!(digest(&fixture.source.catalog_path)?, fixture.source.catalog_sha256);
    assert_eq!(
        SchedulerRegistry::foundational()?
            .resolve(&SchedulerIdentity::new(SIMPLE_SCHEDULER_ID)?)?,
        &DEFINITION
    );
    Ok(())
}

#[test]
fn val_scheduler_001_simple_arrays_and_failures_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("simple-analytical-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas),
    )?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution_context = context(&backend, &authority, &cancellation)?;
    for case in fixture.cases {
        let actual = simple_schedule(
            &backend,
            &execution_context,
            &registry,
            &profile,
            &SchedulerRequest::new(SIMPLE_SCHEDULER_ID, case.steps, case.denoise)?,
        )?;
        assert_eq!(actual, case.expected, "{}", case.name);
    }
    assert_eq!(
        simple_schedule(
            &backend,
            &execution_context,
            &registry,
            &profile,
            &SchedulerRequest::new("normal", 2, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: SIMPLE_SCHEDULER_ID,
            actual: "normal".to_owned(),
        })
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert_eq!(
        simple_schedule(
            &backend,
            &context(&backend, &authority, &cancelled)?,
            &registry,
            &profile,
            &SchedulerRequest::new(SIMPLE_SCHEDULER_ID, 2, 1.0)?,
        ),
        Err(SchedulerError::Cancelled)
    );
    assert!(
        !IMPLEMENTATION.contains("workspace_vec")
            && !IMPLEMENTATION.contains("start_step")
            && !IMPLEMENTATION.contains("end_step")
            && IMPLEMENTATION.contains("build_scheduler_schedule")
    );
    Ok(())
}
