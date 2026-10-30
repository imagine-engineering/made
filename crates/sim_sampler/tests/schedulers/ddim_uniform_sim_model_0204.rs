use sim_sampler::{
    DiscreteSamplingProfile, PenultimateSigmaPolicy, PredictionInterpretation, SamplingProfile,
    SamplingProfileError, SamplingProfileIdentity, SchedulerError, SchedulerIdentity,
    SchedulerRegistry, SchedulerRequest,
    generated_ddim_uniform_sim_model_0204::{
        DDIM_UNIFORM_SCHEDULER_FEATURE_ID, DDIM_UNIFORM_SCHEDULER_ID, DEFINITION,
        ddim_uniform_schedule,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf, sync::Arc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleFixture {
    schema_version: u32,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    profile_sigmas: Vec<f32>,
    tolerance: f32,
    cases: Vec<ScheduleCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceFixture {
    product: String,
    version: String,
    tree_fingerprint: String,
    equation_path: String,
    equation_sha256: String,
    equation_lines: [usize; 2],
    registry_lines: [usize; 3],
    catalog_path: String,
    catalog_sha256: String,
    catalog_line: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleCase {
    name: String,
    steps: u32,
    denoise: f32,
    start_step: Option<u32>,
    end_step: Option<u32>,
    discard_penultimate: bool,
    profile_sigmas: Option<Vec<f32>>,
    stride: usize,
    selected_source_indices: Vec<Option<usize>>,
    expected: Vec<f32>,
}

struct FixtureProfile {
    identity: SamplingProfileIdentity,
    sigmas: Vec<f32>,
}

impl FixtureProfile {
    fn new(sigmas: Vec<f32>) -> Result<Self, SchedulerError> {
        Ok(Self {
            identity: SamplingProfileIdentity::new("ddim-uniform-analytical-v1")?,
            sigmas,
        })
    }
}

impl SamplingProfile for FixtureProfile {
    fn identity(&self) -> &SamplingProfileIdentity {
        &self.identity
    }

    fn prediction(&self) -> PredictionInterpretation {
        PredictionInterpretation::Epsilon
    }

    fn sigma_count(&self) -> usize {
        self.sigmas.len()
    }

    fn sigma_at_model_time(&self, model_time: f32) -> Result<f32, SamplingProfileError> {
        if !model_time.is_finite() || model_time < 0.0 || model_time.fract() != 0.0 {
            return Err(SamplingProfileError::InvalidModelTime(model_time));
        }
        let index = model_time as usize;
        self.sigmas
            .get(index)
            .copied()
            .ok_or(SamplingProfileError::GridIndex(index))
    }

    fn model_time_for_sigma(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        self.sigmas
            .iter()
            .position(|candidate| candidate.to_bits() == sigma.to_bits())
            .map(|index| index as f32)
            .ok_or(SamplingProfileError::InvalidSigma(sigma))
    }

    fn sigma_min(&self) -> f32 {
        self.sigmas.first().copied().unwrap_or(0.0)
    }

    fn sigma_max(&self) -> f32 {
        self.sigmas.last().copied().unwrap_or(0.0)
    }

    fn half_log_snr(&self, sigma: f32) -> Result<f32, SamplingProfileError> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(SamplingProfileError::InvalidSnrSigma(sigma));
        }
        Ok(-sigma.ln())
    }

    fn sigma_from_half_log_snr(&self, half_log_snr: f32) -> Result<f32, SamplingProfileError> {
        if !half_log_snr.is_finite() {
            return Err(SamplingProfileError::InvalidHalfLogSnr(half_log_snr));
        }
        Ok((-half_log_snr).exp())
    }

    fn adjust_first_sigma_for_snr(&self, _sigmas: &mut [f32]) -> Result<(), SamplingProfileError> {
        Ok(())
    }

    fn scale_sampler_noise(&self, sampler_noise_scale: f32) -> Result<f32, SamplingProfileError> {
        if !sampler_noise_scale.is_finite() {
            return Err(SamplingProfileError::InvalidNoiseScale(sampler_noise_scale));
        }
        Ok(sampler_noise_scale)
    }
}

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn fixture() -> Result<ScheduleFixture, Box<dyn Error>> {
    let path = workspace()?.join(
        "crates/sim_test_support/fixtures/schedulers/ddim_uniform_comfy_model_0204/schedule.json",
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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

#[test]
fn definition_and_source_provenance_are_exact() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, DDIM_UNIFORM_SCHEDULER_ID);
    assert_eq!(fixture.feature_id, DDIM_UNIFORM_SCHEDULER_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, 4);
    assert_eq!(fixture.tolerance, 0.0);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.feature_id, fixture.feature_id);
    assert_eq!(DEFINITION.source_ordinal, fixture.source_ordinal);
    assert_eq!(DEFINITION.aliases, &[] as &[&str]);
    assert_eq!(
        DEFINITION.implementation_module,
        "schedulers/ddim_uniform_comfy_model_0204"
    );

    let root = workspace()?;
    let equation = fs::read(root.join(&fixture.source.equation_path))?;
    assert_eq!(sha256(&equation), fixture.source.equation_sha256);
    let catalog = fs::read(root.join(&fixture.source.catalog_path))?;
    assert_eq!(sha256(&catalog), fixture.source.catalog_sha256);
    let equation_text = std::str::from_utf8(&equation)?;
    assert_eq!(
        equation_text
            .lines()
            .nth(fixture.source.equation_lines[0] - 1),
        Some("def ddim_scheduler(model_sampling, steps):")
    );
    assert_eq!(
        equation_text
            .lines()
            .nth(fixture.source.registry_lines[1] - 1),
        Some("    \"ddim_uniform\": SchedulerHandler(ddim_scheduler),")
    );
    let catalog_text = std::str::from_utf8(&catalog)?;
    assert!(
        catalog_text
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(|line| line.starts_with("scheduler,ddim_uniform,"))
    );
    assert_eq!(fixture.source.product, "ComfyUI");
    assert_eq!(fixture.source.version, "0.27.1");
    assert_eq!(
        fixture.source.tree_fingerprint,
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(fixture.source.equation_lines, [653, 668]);
    assert_eq!(fixture.source.registry_lines, [1346, 1351, 1357]);
    assert_eq!(fixture.source.catalog_line, 205);

    let registry = SchedulerRegistry::foundational()?;
    assert_eq!(
        registry.resolve(&SchedulerIdentity::new(DDIM_UNIFORM_SCHEDULER_ID)?)?,
        &DEFINITION
    );
    Ok(())
}

#[test]
fn analytical_arrays_match_every_source_branch_exactly() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;

    for case in fixture.cases {
        let sigmas = case
            .profile_sigmas
            .clone()
            .unwrap_or_else(|| fixture.profile_sigmas.clone());
        let second_sigma = *sigmas.get(1).ok_or("fixture profile is too short")?;
        let mut effective_steps =
            (f64::from(case.steps) / f64::from(case.denoise)).floor() as usize;
        if case.discard_penultimate {
            effective_steps += 1;
        }
        let selection_steps = if second_sigma.abs() <= 0.00001 {
            effective_steps + 1
        } else {
            effective_steps
        };
        assert_eq!(
            case.stride,
            (sigmas.len() / selection_steps).max(1),
            "{} fixture stride",
            case.name
        );
        assert_eq!(
            case.selected_source_indices.len(),
            case.expected.len(),
            "{} fixture provenance",
            case.name
        );
        let analytical = case
            .selected_source_indices
            .iter()
            .map(|source_index| match source_index {
                Some(source_index) => sigmas
                    .get(*source_index)
                    .copied()
                    .ok_or("fixture source index is unavailable"),
                None => Ok(0.0),
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            case.expected, analytical,
            "{} analytical fixture",
            case.name
        );

        let profile = FixtureProfile::new(sigmas)?;
        let mut request =
            SchedulerRequest::new(DDIM_UNIFORM_SCHEDULER_ID, case.steps, case.denoise)?
                .with_window(case.start_step, case.end_step)?;
        if case.discard_penultimate {
            request = request.with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard);
        }
        let actual = ddim_uniform_schedule(&backend, &context, &registry, &profile, &request)?;
        assert_eq!(actual, case.expected, "{}", case.name);
    }
    Ok(())
}

#[test]
fn discrete_profile_indices_preserve_source_f32_bits() -> Result<(), Box<dyn Error>> {
    let sigmas = Arc::<[f32]>::from([0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0]);
    let profile = DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("ddim-uniform-direct-grid-v1")?,
        PredictionInterpretation::Epsilon,
        sigmas.clone(),
    )?;
    let registry = SchedulerRegistry::foundational()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let actual = ddim_uniform_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new(DDIM_UNIFORM_SCHEDULER_ID, 4, 1.0)?,
    )?;
    let mut expected = [9_usize, 7, 5, 3, 1]
        .into_iter()
        .map(|index| {
            sigmas
                .get(index)
                .copied()
                .ok_or("direct-grid fixture index is unavailable")
        })
        .collect::<Result<Vec<_>, _>>()?;
    expected.push(0.0);
    assert_eq!(actual, expected);
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn canonical_failures_are_typed_and_failure_atomic() -> Result<(), Box<dyn Error>> {
    assert!(matches!(
        SchedulerRequest::new(DDIM_UNIFORM_SCHEDULER_ID, 0, 1.0),
        Err(SchedulerError::ZeroSteps)
    ));
    assert!(matches!(
        SchedulerRequest::new(DDIM_UNIFORM_SCHEDULER_ID, 4, 0.0),
        Err(SchedulerError::InvalidDenoise(0.0))
    ));

    let registry = SchedulerRegistry::foundational()?;
    let profile = FixtureProfile::new(vec![0.01, 0.04, 1.0])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let wrong_identity = SchedulerRequest::new("normal", 1, 1.0)?;
    assert!(matches!(
        ddim_uniform_schedule(&backend, &context, &registry, &profile, &wrong_identity),
        Err(SchedulerError::AlgorithmMismatch {
            expected: DDIM_UNIFORM_SCHEDULER_ID,
            actual,
        }) if actual == "normal"
    ));

    let short = FixtureProfile::new(vec![0.01])?;
    let request = SchedulerRequest::new(DDIM_UNIFORM_SCHEDULER_ID, 1, 1.0)?;
    assert!(matches!(
        ddim_uniform_schedule(&backend, &context, &registry, &short, &request),
        Err(SchedulerError::InvalidProfile(_))
    ));

    let non_finite = FixtureProfile::new(vec![0.01, f32::NAN, 1.0])?;
    assert!(matches!(
        ddim_uniform_schedule(&backend, &context, &registry, &non_finite, &request),
        Err(SchedulerError::NonFiniteSigma { index: 0, value }) if value.is_nan()
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    assert!(matches!(
        ddim_uniform_schedule(&backend, &cancelled_context, &registry, &profile, &request),
        Err(SchedulerError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(20)?;
    let limited_cancellation = CancellationToken::default();
    let limited_context = limited_backend.execution_context(
        StreamId::DEFAULT,
        limited_authority.authorize_workspace(20)?,
        &limited_cancellation,
    );
    let fixture_profile = FixtureProfile::new(fixture()?.profile_sigmas)?;
    assert!(matches!(
        ddim_uniform_schedule(
            &limited_backend,
            &limited_context,
            &registry,
            &fixture_profile,
            &SchedulerRequest::new(DDIM_UNIFORM_SCHEDULER_ID, 4, 1.0)?,
        ),
        Err(SchedulerError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(limited_backend.memory_snapshot().current_bytes, 0);
    Ok(())
}
