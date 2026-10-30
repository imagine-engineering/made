use sim_sampler::{
    DiscreteSamplingProfile, PenultimateSigmaPolicy, PredictionInterpretation,
    SamplingProfileIdentity, SchedulerError, SchedulerIdentity, SchedulerRegistry,
    SchedulerRequest,
    generated_kl_optimal_sim_model_0207::{
        DEFINITION, KL_OPTIMAL_SCHEDULER_FEATURE_ID, KL_OPTIMAL_SCHEDULER_ID,
        KL_OPTIMAL_SCHEDULER_SOURCE_ORDINAL, kl_optimal_schedule,
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
    "/../sim_test_support/fixtures/schedulers/kl_optimal_comfy_model_0207/schedule.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/schedulers/kl_optimal_sim_model_0207.rs");

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
    source_path: String,
    source_sha256: String,
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
        SamplingProfileIdentity::new("kl-optimal-analytical-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas.clone()),
    )?)
}

fn scheduler_names(source: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let (_, handlers) = source
        .split_once("SCHEDULER_HANDLERS = {")
        .ok_or("SCHEDULER_HANDLERS is unavailable")?;
    let (handlers, _) = handlers
        .split_once("\n}")
        .ok_or("SCHEDULER_HANDLERS is unterminated")?;
    let names = handlers
        .lines()
        .filter_map(|line| line.trim().strip_prefix('"'))
        .filter_map(|line| line.split_once('"').map(|(name, _)| name.to_owned()))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err("SCHEDULER_HANDLERS contains no identities".into());
    }
    Ok(names)
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
fn val_scheduler_001_kl_optimal_definition_and_provenance_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, KL_OPTIMAL_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, KL_OPTIMAL_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, KL_OPTIMAL_SCHEDULER_SOURCE_ORDINAL);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        DEFINITION.implementation_module,
        "schedulers/kl_optimal_comfy_model_0207"
    );

    let source = fs::read_to_string(workspace_root()?.join(&fixture.source.source_path))?;
    assert_eq!(
        scheduler_names(&source)?
            .iter()
            .position(|name| name == KL_OPTIMAL_SCHEDULER_ID),
        Some(usize::from(KL_OPTIMAL_SCHEDULER_SOURCE_ORDINAL))
    );
    let registry = SchedulerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SchedulerIdentity::new(KL_OPTIMAL_SCHEDULER_ID)?)?,
        &DEFINITION
    );
    assert_eq!(fixture.source.equation_lines, [731, 735]);
    assert_eq!(fixture.source.registry_line, 1355);
    assert_eq!(fixture.source.catalog_line, 208);
    assert_eq!(
        digest(&fixture.source.source_path)?,
        fixture.source.source_sha256
    );
    assert_eq!(
        digest(&fixture.source.catalog_path)?,
        fixture.source.catalog_sha256
    );
    Ok(())
}

#[test]
fn val_scheduler_001_kl_optimal_matches_source_equation_and_shared_slicing()
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
        let actual = kl_optimal_schedule(&backend, &context, &registry, &profile, &request)?;
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
fn val_scheduler_001_kl_optimal_rejects_degenerate_or_invalid_requests()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;

    assert_eq!(
        kl_optimal_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new("normal", 2, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: KL_OPTIMAL_SCHEDULER_ID,
            actual: "normal".to_owned(),
        })
    );
    assert_eq!(
        SchedulerRequest::new(KL_OPTIMAL_SCHEDULER_ID, 0, 1.0),
        Err(SchedulerError::ZeroSteps)
    );
    assert!(matches!(
        kl_optimal_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new(KL_OPTIMAL_SCHEDULER_ID, 1, 1.0)?,
        ),
        Err(SchedulerError::NonFiniteSigma { index: 0, value }) if value.is_nan()
    ));
    assert_eq!(
        SchedulerRequest::new(KL_OPTIMAL_SCHEDULER_ID, 2, 1.0)?
            .with_window(Some(2), Some(2)),
        Err(SchedulerError::InvalidWindow {
            start: 2,
            end: 2,
            steps: 2,
        })
    );
    Ok(())
}

#[test]
fn val_scheduler_001_kl_optimal_honors_structural_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = execution_context(&backend, &authority, &cancellation)?;
    assert_eq!(
        kl_optimal_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new(KL_OPTIMAL_SCHEDULER_ID, 512, 1.0)?,
        ),
        Err(SchedulerError::Cancelled)
    );
    Ok(())
}
