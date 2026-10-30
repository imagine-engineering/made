use sim_sampler::{
    DiscreteSamplingProfile, PenultimateSigmaPolicy, PredictionInterpretation, SamplingProfileIdentity,
    SchedulerError, SchedulerIdentity, SchedulerRegistry, SchedulerRequest,
    generated_beta_sim_model_0203::{
        BETA_SCHEDULER_FEATURE_ID, BETA_SCHEDULER_ID, BETA_SCHEDULER_SOURCE_ORDINAL,
        DEFINITION, beta_schedule,
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
    "/../sim_test_support/fixtures/schedulers/beta_comfy_model_0203/schedule.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedulers/beta_sim_model_0203.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    alpha: f64,
    beta: f64,
    source: SourceFixture,
    profile_sigmas: Vec<f32>,
    tolerance: f32,
    cases: Vec<CaseFixture>,
    short_profile_case: ShortProfileFixture,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    path: String,
    sha256: String,
    equation_lines: [usize; 2],
    registry_line: usize,
    catalog_path: String,
    catalog_sha256: String,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct CaseFixture {
    name: String,
    steps: u32,
    denoise: f32,
    start_step: Option<u32>,
    end_step: Option<u32>,
    discard_penultimate: bool,
    timestep_indices: Vec<usize>,
    expected: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ShortProfileFixture {
    profile_sigmas: Vec<f32>,
    steps: u32,
    timestep_indices: Vec<usize>,
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

fn digest(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn profile(sigmas: Vec<f32>) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("beta-scheduler-fixture-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(sigmas),
    )?)
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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "sigma {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn val_scheduler_001_beta_definition_and_pinned_source_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, BETA_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, BETA_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, BETA_SCHEDULER_SOURCE_ORDINAL);
    assert_eq!(fixture.alpha, 0.6);
    assert_eq!(fixture.beta, 0.6);
    assert_eq!(DEFINITION.identity, BETA_SCHEDULER_ID);
    assert_eq!(DEFINITION.feature_id, BETA_SCHEDULER_FEATURE_ID);
    assert_eq!(DEFINITION.source_ordinal, 5);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        SchedulerRegistry::foundational()?
            .resolve(&SchedulerIdentity::new(BETA_SCHEDULER_ID)?)?,
        &DEFINITION
    );

    let root = workspace_root()?;
    assert_eq!(digest(&root.join(&fixture.source.path))?, fixture.source.sha256);
    assert_eq!(
        digest(&root.join(&fixture.source.catalog_path))?,
        fixture.source.catalog_sha256
    );
    let source = fs::read_to_string(root.join(&fixture.source.path))?;
    let equation = source
        .lines()
        .skip(fixture.source.equation_lines[0].saturating_sub(1))
        .take(fixture.source.equation_lines[1] - fixture.source.equation_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):",
        "scipy.stats.beta.ppf(ts, alpha, beta)",
        "numpy.rint",
        "if t != last_t:",
        "sigs += [0.0]",
    ] {
        assert!(equation.contains(fragment), "missing source {fragment}");
    }
    assert!(
        source
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains("\"beta\": SchedulerHandler(beta_scheduler)"))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("scheduler,beta,")
                && line.ends_with(",COMFY-MODEL-0203"))
    );

    assert!(IMPLEMENTATION.contains("build_scheduler_schedule("));
    assert!(IMPLEMENTATION.contains("round_ties_even()"));
    for forbidden in [
        "request.validate(",
        "request.denoise",
        "request.start_step",
        "request.end_step",
        "PenultimateSigmaPolicy",
        "workspace_vec",
        "selected_schedule",
    ] {
        assert!(!IMPLEMENTATION.contains(forbidden), "duplicate owner {forbidden}");
    }
    Ok(())
}

#[test]
fn val_scheduler_001_beta_arrays_boundaries_and_slicing_match_fixture()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(fixture.profile_sigmas)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 1024 * 1024)?;
    let registry = SchedulerRegistry::foundational()?;
    for case in fixture.cases {
        assert!(!case.name.is_empty());
        assert!(!case.timestep_indices.is_empty());
        let mut request = SchedulerRequest::new(BETA_SCHEDULER_ID, case.steps, case.denoise)?
            .with_window(case.start_step, case.end_step)?;
        if case.discard_penultimate {
            request = request.with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard);
        }
        let actual = beta_schedule(&backend, &context, &registry, &profile, &request)?;
        assert_eq!(actual.len(), case.expected.len(), "case {}", case.name);
        for (index, (actual, expected)) in actual.iter().zip(&case.expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= fixture.tolerance,
                "case {} sigma {index}: expected {expected}, got {actual}",
                case.name
            );
        }
    }
    Ok(())
}

#[test]
fn val_scheduler_001_beta_adjacent_duplicate_collapse_preserves_short_schedule()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let short = fixture.short_profile_case;
    assert_eq!(short.timestep_indices, [2, 1, 0]);
    let profile = profile(short.profile_sigmas)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 1024 * 1024)?;
    let actual = beta_schedule(
        &backend,
        &context,
        &SchedulerRegistry::foundational()?,
        &profile,
        &SchedulerRequest::new(BETA_SCHEDULER_ID, short.steps, 1.0)?,
    )?;
    assert_close(&actual, &short.expected, fixture.tolerance);
    Ok(())
}

#[test]
fn val_scheduler_001_beta_typed_failures_cancellation_and_workspace_are_exact()
-> Result<(), Box<dyn Error>> {
    assert!(matches!(
        SchedulerRequest::new(BETA_SCHEDULER_ID, 0, 1.0),
        Err(SchedulerError::ZeroSteps)
    ));
    assert!(matches!(
        SchedulerRequest::new(BETA_SCHEDULER_ID, 4, 0.0),
        Err(SchedulerError::InvalidDenoise(0.0))
    ));
    assert!(matches!(
        SchedulerRequest::new(BETA_SCHEDULER_ID, 4, 1.0)?.with_window(Some(3), Some(2)),
        Err(SchedulerError::InvalidWindow { .. })
    ));

    let fixture = fixture()?;
    let profile = profile(fixture.profile_sigmas)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 1024 * 1024)?;
    assert!(matches!(
        beta_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new("normal", 4, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: BETA_SCHEDULER_ID,
            ..
        })
    ));

    assert!(cancellation.cancel());
    assert!(matches!(
        beta_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new(BETA_SCHEDULER_ID, 4, 1.0)?,
        ),
        Err(SchedulerError::Cancelled)
    ));

    let fresh_cancellation = CancellationToken::default();
    let constrained = execution_context(&backend, &authority, &fresh_cancellation, 8)?;
    assert!(matches!(
        beta_schedule(
            &backend,
            &constrained,
            &registry,
            &profile,
            &SchedulerRequest::new(BETA_SCHEDULER_ID, 4, 1.0)?,
        ),
        Err(SchedulerError::Tensor(_))
    ));
    Ok(())
}
