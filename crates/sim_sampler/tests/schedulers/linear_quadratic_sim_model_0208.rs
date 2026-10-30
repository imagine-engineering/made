use sim_sampler::{
    DiscreteSamplingProfile, PenultimateSigmaPolicy, PredictionInterpretation,
    SamplingProfileIdentity, SchedulerError, SchedulerIdentity, SchedulerRegistry,
    SchedulerRequest,
    generated_linear_quadratic_sim_model_0208::{
        DEFINITION, LINEAR_QUADRATIC_SCHEDULER_FEATURE_ID, LINEAR_QUADRATIC_SCHEDULER_ID,
        LINEAR_QUADRATIC_SCHEDULER_SOURCE_ORDINAL, linear_quadratic_schedule,
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
    "/../sim_test_support/fixtures/schedulers/linear_quadratic_comfy_model_0208/schedule.json"
));
const IMPLEMENTATION: &str =
    include_str!("../../src/schedulers/linear_quadratic_sim_model_0208.rs");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    threshold_noise: f64,
    source: SourceFixture,
    profile_sigmas: Vec<f32>,
    cases: Vec<CaseFixture>,
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

fn profile(fixture: &Fixture) -> Result<DiscreteSamplingProfile, Box<dyn Error>> {
    Ok(DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("linear-quadratic-fixture-v1")?,
        PredictionInterpretation::Epsilon,
        Arc::from(fixture.profile_sigmas.clone()),
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

#[test]
fn val_scheduler_001_linear_quadratic_definition_and_pinned_source_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, LINEAR_QUADRATIC_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, LINEAR_QUADRATIC_SCHEDULER_FEATURE_ID);
    assert_eq!(
        fixture.source_ordinal,
        LINEAR_QUADRATIC_SCHEDULER_SOURCE_ORDINAL
    );
    assert_eq!(fixture.threshold_noise, 0.025);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        DEFINITION.implementation_module,
        "schedulers/linear_quadratic_comfy_model_0208"
    );

    let root = workspace_root()?;
    assert_eq!(digest(&root.join(&fixture.source.path))?, fixture.source.sha256);
    assert_eq!(
        digest(&root.join(&fixture.source.catalog_path))?,
        fixture.source.catalog_sha256
    );
    let source = fs::read_to_string(root.join(&fixture.source.path))?;
    assert_eq!(
        scheduler_names(&source)?
            .iter()
            .position(|name| name == LINEAR_QUADRATIC_SCHEDULER_ID),
        Some(usize::from(LINEAR_QUADRATIC_SCHEDULER_SOURCE_ORDINAL))
    );
    let equation = source
        .lines()
        .skip(fixture.source.equation_lines[0].saturating_sub(1))
        .take(fixture.source.equation_lines[1] - fixture.source.equation_lines[0] + 1)
        .collect::<Vec<_>>()
        .join("\n");
    for fragment in [
        "def linear_quadratic_schedule(model_sampling, steps, threshold_noise=0.025, linear_steps=None):",
        "linear_steps = steps // 2",
        "quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)",
        "sigma_schedule = [1.0 - x for x in sigma_schedule]",
        "return torch.FloatTensor(sigma_schedule) * model_sampling.sigma_max.cpu()",
    ] {
        assert!(equation.contains(fragment), "missing source fragment {fragment}");
    }
    assert!(
        source
            .lines()
            .nth(fixture.source.registry_line.saturating_sub(1))
            .is_some_and(|line| line.contains(
                "\"linear_quadratic\": SchedulerHandler(linear_quadratic_schedule)"
            ))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line.saturating_sub(1))
            .is_some_and(|line| line.contains("scheduler,linear_quadratic,")
                && line.ends_with(",COMFY-MODEL-0208"))
    );
    assert_eq!(
        SchedulerRegistry::foundational()?
            .resolve(&SchedulerIdentity::new(LINEAR_QUADRATIC_SCHEDULER_ID)?)?,
        &DEFINITION
    );
    Ok(())
}

#[test]
fn val_scheduler_001_linear_quadratic_arrays_and_shared_slicing_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 1024 * 1024)?;
    for case in fixture.cases {
        assert!(!case.name.is_empty());
        let mut request = SchedulerRequest::new(&fixture.identity, case.steps, case.denoise)?
            .with_window(case.start_step, case.end_step)?;
        if case.discard_penultimate {
            request = request.with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard);
        }
        let actual =
            linear_quadratic_schedule(&backend, &context, &registry, &profile, &request)?;
        assert_eq!(actual, case.expected, "case {}", case.name);
    }
    Ok(())
}

#[test]
fn val_scheduler_001_linear_quadratic_delegates_shared_ownership_and_typed_failures()
-> Result<(), Box<dyn Error>> {
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
    assert!(IMPLEMENTATION.contains("build_scheduler_schedule("));

    assert_eq!(
        SchedulerRequest::new(LINEAR_QUADRATIC_SCHEDULER_ID, 0, 1.0),
        Err(SchedulerError::ZeroSteps)
    );
    assert!(matches!(
        SchedulerRequest::new(LINEAR_QUADRATIC_SCHEDULER_ID, 4, f32::NAN),
        Err(SchedulerError::InvalidDenoise(value)) if value.is_nan()
    ));
    assert_eq!(
        SchedulerRequest::new(LINEAR_QUADRATIC_SCHEDULER_ID, 4, 1.0)?
            .with_window(Some(3), Some(2)),
        Err(SchedulerError::InvalidWindow {
            start: 3,
            end: 2,
            steps: 4,
        })
    );

    let fixture = fixture()?;
    let profile = profile(&fixture)?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation, 1024 * 1024)?;
    assert_eq!(
        linear_quadratic_schedule(
            &backend,
            &context,
            &registry,
            &profile,
            &SchedulerRequest::new("normal", 4, 1.0)?,
        ),
        Err(SchedulerError::AlgorithmMismatch {
            expected: LINEAR_QUADRATIC_SCHEDULER_ID,
            actual: "normal".to_owned(),
        })
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled, 1024 * 1024)?;
    assert_eq!(
        linear_quadratic_schedule(
            &backend,
            &cancelled_context,
            &registry,
            &profile,
            &SchedulerRequest::new(LINEAR_QUADRATIC_SCHEDULER_ID, 4, 1.0)?,
        ),
        Err(SchedulerError::Cancelled)
    );

    let active = CancellationToken::default();
    let constrained_context = execution_context(&backend, &authority, &active, 8)?;
    assert!(matches!(
        linear_quadratic_schedule(
            &backend,
            &constrained_context,
            &registry,
            &profile,
            &SchedulerRequest::new(LINEAR_QUADRATIC_SCHEDULER_ID, 4, 1.0)?,
        ),
        Err(SchedulerError::Tensor(_))
    ));
    Ok(())
}
