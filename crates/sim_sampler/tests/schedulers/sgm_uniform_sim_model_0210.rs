use sim_sampler::{
    DiscreteSamplingProfile, PredictionInterpretation, SamplingProfileIdentity, SchedulerError,
    SchedulerIdentity, SchedulerRegistry, SchedulerRequest,
    generated_sgm_uniform_sim_model_0210::{
        DEFINITION, SGM_UNIFORM_SCHEDULER_FEATURE_ID, SGM_UNIFORM_SCHEDULER_ID,
        SGM_UNIFORM_SCHEDULER_SOURCE_ORDINAL, sgm_uniform_schedule,
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
    "/../sim_test_support/fixtures/schedulers/sgm_uniform_comfy_model_0210/schedule.json"
));
const IMPLEMENTATION: &str = include_str!("../../src/schedulers/sgm_uniform_sim_model_0210.rs");

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
    start_step: Option<u32>,
    end_step: Option<u32>,
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

fn assert_close(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!((actual - expected).abs() <= tolerance, "{name} sigma {index}");
    }
}

#[test]
fn val_scheduler_001_sgm_uniform_definition_and_source_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, SGM_UNIFORM_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, SGM_UNIFORM_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, SGM_UNIFORM_SCHEDULER_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.implementation_module, "schedulers/sgm_uniform_comfy_model_0210");
    assert_eq!(fixture.source.equation_lines, [670, 690]);
    assert_eq!(fixture.source.registry_line, 1348);
    assert_eq!(fixture.source.catalog_line, 211);
    assert_eq!(digest(&fixture.source.path)?, fixture.source.sha256);
    assert_eq!(digest(&fixture.source.catalog_path)?, fixture.source.catalog_sha256);
    assert_eq!(
        SchedulerRegistry::foundational()?
            .resolve(&SchedulerIdentity::new(SGM_UNIFORM_SCHEDULER_ID)?)?,
        &DEFINITION
    );
    Ok(())
}

#[test]
fn val_scheduler_001_sgm_uniform_delegates_normal_family_and_shared_slicing()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("sgm-uniform-analytical-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas),
    )?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution_context = context(&backend, &authority, &cancellation)?;
    for case in fixture.cases {
        let request = SchedulerRequest::new(SGM_UNIFORM_SCHEDULER_ID, case.steps, case.denoise)?
            .with_window(case.start_step, case.end_step)?;
        let actual = sgm_uniform_schedule(
            &backend,
            &execution_context,
            &registry,
            &profile,
            &request,
        )?;
        assert_close(&case.name, &actual, &case.expected, fixture.tolerance);
    }
    assert!(
        IMPLEMENTATION.contains("normal_schedule_with_mode")
            && !IMPLEMENTATION.contains("sigma_at_model_time")
            && !IMPLEMENTATION.contains("build_scheduler_schedule")
            && !IMPLEMENTATION.contains("workspace_vec")
    );
    assert_eq!(
        sgm_uniform_schedule(
            &backend,
            &execution_context,
            &registry,
            &profile,
            &SchedulerRequest::new("normal", 2, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: SGM_UNIFORM_SCHEDULER_ID,
            actual: "normal".to_owned(),
        })
    );
    Ok(())
}
