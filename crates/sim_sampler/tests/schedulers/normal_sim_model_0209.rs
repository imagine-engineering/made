use sim_sampler::{
    DiscreteSamplingProfile, NORMAL_FOUNDATION_DEFINITION, NORMAL_SCHEDULER_FEATURE_ID,
    NORMAL_SCHEDULER_ID, PenultimateSigmaPolicy, PredictionInterpretation, SamplingProfileIdentity,
    SchedulerError, SchedulerIdentity, SchedulerRegistry, SchedulerRequest,
    generated_normal_sim_model_0209::{
        DEFINITION, NORMAL_SCHEDULER_SOURCE_ORDINAL, normal_schedule_adapter,
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
    "/../sim_test_support/fixtures/schedulers/normal_comfy_model_0209/schedule.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedulers/normal_sim_model_0209.rs"
));

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
struct SourceFixture {
    equation_path: String,
    equation_sha256: String,
    equation_lines: [usize; 2],
    registry_line: usize,
    catalog_path: String,
    catalog_sha256: String,
    catalog_line: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("normal-scheduler-analytical-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas.clone()),
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

fn assert_close(case: &CaseFixture, actual: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), case.expected.len(), "{} length", case.name);
    for (index, (actual, expected)) in actual.iter().zip(&case.expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "{} sigma {index}: expected {expected}, got {actual}",
            case.name
        );
    }
}

#[test]
fn val_scheduler_001_normal_definition_provenance_and_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, NORMAL_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, NORMAL_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, NORMAL_SCHEDULER_SOURCE_ORDINAL);
    assert_eq!(DEFINITION, NORMAL_FOUNDATION_DEFINITION);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        DEFINITION.implementation_module,
        "schedulers/native_diffusion"
    );
    assert_eq!(fixture.source.equation_lines, [670, 692]);
    assert_eq!(fixture.source.registry_line, 1353);
    assert_eq!(fixture.source.catalog_line, 210);
    assert_eq!(
        digest(&fixture.source.equation_path)?,
        fixture.source.equation_sha256
    );
    assert_eq!(
        digest(&fixture.source.catalog_path)?,
        fixture.source.catalog_sha256
    );

    let source = fs::read_to_string(workspace_root()?.join(&fixture.source.equation_path))?;
    let equation = source
        .lines()
        .skip(fixture.source.equation_lines[0] - 1)
        .take(fixture.source.equation_lines[1] - fixture.source.equation_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def normal_scheduler(model_sampling, steps, sgm=False, floor=False):",
        "timesteps = torch.linspace(start, end, steps)",
        "sigs.append(float(s.sigma(ts)))",
        "sigs += [0.0]",
    ] {
        assert!(equation.contains(fragment), "missing source {fragment}");
    }
    assert!(
        source
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("\"normal\": SchedulerHandler(normal_scheduler)"))
    );
    let catalog = fs::read_to_string(workspace_root()?.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(
                |line| line.starts_with("scheduler,normal,") && line.ends_with(",COMFY-MODEL-0209")
            )
    );
    assert_eq!(
        SchedulerRegistry::foundational()?
            .resolve(&SchedulerIdentity::new(NORMAL_SCHEDULER_ID)?)?,
        &DEFINITION
    );

    assert!(IMPLEMENTATION.contains("crate::normal_schedule("));
    for forbidden in [
        "build_scheduler_schedule",
        "workspace_vec",
        "request.validate(",
        "request.denoise",
        "request.start_step",
        "request.end_step",
        "sigma_at_model_time",
        "try_push",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_scheduler_001_normal_complete_arrays_and_shared_slicing_match_fixture()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;

    for case in &fixture.cases {
        assert!(!case.name.is_empty());
        let mut request = SchedulerRequest::new(NORMAL_SCHEDULER_ID, case.steps, case.denoise)?
            .with_window(case.start_step, case.end_step)?;
        if case.discard_penultimate {
            request = request.with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard);
        }
        let actual = normal_schedule_adapter(&backend, &context, &registry, &profile, &request)?;
        assert_close(case, &actual, fixture.tolerance);
    }
    Ok(())
}

#[test]
fn val_scheduler_001_normal_typed_failures_and_cancellation_are_preserved()
-> Result<(), Box<dyn Error>> {
    assert_eq!(
        SchedulerRequest::new(NORMAL_SCHEDULER_ID, 0, 1.0),
        Err(SchedulerError::ZeroSteps)
    );
    assert_eq!(
        SchedulerRequest::new(NORMAL_SCHEDULER_ID, 4, 0.0),
        Err(SchedulerError::InvalidDenoise(0.0))
    );

    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    assert_eq!(
        normal_schedule_adapter(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new("simple", 4, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: NORMAL_SCHEDULER_ID,
            actual: "simple".to_owned(),
        })
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    assert_eq!(
        normal_schedule_adapter(
            &backend,
            &cancelled_context,
            &registry,
            &profile,
            &SchedulerRequest::new(NORMAL_SCHEDULER_ID, 4, 1.0)?,
        ),
        Err(SchedulerError::Cancelled)
    );
    Ok(())
}
