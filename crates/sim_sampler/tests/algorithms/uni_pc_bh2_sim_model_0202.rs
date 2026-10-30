use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingError, SamplingPlan, SamplingProfileIdentity,
    generated_uni_pc_bh2_sim_model_0202::{
        DEFINITION, UNI_PC_BH2_FEATURE_ID, UNI_PC_BH2_SAMPLER_ID,
        UNI_PC_BH2_SOURCE_ORDINAL, UniPcDenoiserStage, UniPcError, sample_uni_pc_bh2,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/uni_pc_bh2_comfy_model_0202/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/uni_pc_bh2_sim_model_0202.rs"
));

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u16,
    identity: String,
    feature_id: String,
    source_ordinal: u16,
    source: SourceFixture,
    terminal_sigma_replacement: f32,
    maximum_order: usize,
    tolerance: f32,
    shape: Vec<u64>,
    sigmas: Vec<f32>,
    initial: Vec<f32>,
    model_scale: f32,
    model_sigma_scale: f32,
    model_offsets: Vec<f32>,
    model_calls: Vec<ModelCallFixture>,
    callbacks: Vec<CallbackFixture>,
    trace_latents: Vec<Vec<f32>>,
    terminal: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    uni_pc_path: String,
    uni_pc_sha256: String,
    samplers_path: String,
    samplers_sha256: String,
    catalog_path: String,
    catalog_sha256: String,
    bh_lines: [usize; 2],
    wrapper_lines: [usize; 2],
    registry_line: usize,
    selector_line: usize,
    discard_policy_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct ModelCallFixture {
    target_index: usize,
    stage: String,
    sigma: f32,
    input: Vec<f32>,
    predictor: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CallbackFixture {
    step: usize,
    latent: Vec<f32>,
    denoised: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct ObservedCall {
    step: usize,
    stage: UniPcDenoiserStage,
    sigma: f32,
    input: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct ObservedCallback {
    step: usize,
    total_steps: u32,
    sigma: f32,
    sigma_hat: f32,
    next_sigma: f32,
    latent: Vec<f32>,
    denoised: Vec<f32>,
}

#[derive(Debug, PartialEq)]
struct ObservedRun {
    calls: Vec<ObservedCall>,
    callbacks: Vec<ObservedCallback>,
    latents: Vec<Vec<f32>>,
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

fn source_range(source: &str, range: [usize; 2]) -> String {
    source
        .lines()
        .skip(range[0].saturating_sub(1))
        .take(range[1].saturating_sub(range[0]) + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

fn profile() -> Result<SamplingProfileIdentity, Box<dyn Error>> {
    Ok(SamplingProfileIdentity::new("uni-pc-bh2-row-v1")?)
}

fn plan(
    sampler: &str,
    profile: &SamplingProfileIdentity,
    steps: u32,
    seed: u64,
) -> Result<SamplingPlan, Box<dyn Error>> {
    Ok(SamplingPlan::new(
        sampler,
        "normal",
        profile.clone(),
        seed,
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

fn values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor_to_f32(backend, tensor, context)?.to_vec())
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (element, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "element {element}: expected {expected}, got {actual}"
        );
    }
}

fn execute(
    fixture: &Fixture,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    seed: u64,
) -> Result<ObservedRun, Box<dyn Error>> {
    let profile = profile()?;
    let calls = RefCell::new(Vec::new());
    let callbacks = RefCell::new(Vec::new());
    let trace = sample_uni_pc_bh2(
        backend,
        plan(
            &fixture.identity,
            &profile,
            u32::try_from(fixture.sigmas.len().saturating_sub(1))?,
            seed,
        )?,
        &profile,
        tensor_from_f32(backend, &fixture.shape, &fixture.initial, context)?,
        &fixture.sigmas,
        context,
        |input, sigma, step, stage| {
            let input = tensor_to_f32(backend, input, context)
                .map_err(|error| error.to_string())?
                .to_vec();
            let mut output = Vec::new();
            output
                .try_reserve_exact(input.len())
                .map_err(|_| "model output allocation failed".to_owned())?;
            for (element, input) in input.iter().copied().enumerate() {
                let offset = fixture
                    .model_offsets
                    .get(element)
                    .copied()
                    .ok_or_else(|| "missing model offset".to_owned())?;
                output.push(
                    fixture.model_scale * input + fixture.model_sigma_scale * sigma + offset,
                );
            }
            calls.borrow_mut().push(ObservedCall {
                step,
                stage,
                sigma,
                input,
                output: output.clone(),
            });
            tensor_from_f32(backend, &fixture.shape, &output, context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            callbacks.borrow_mut().push(ObservedCallback {
                step: usize::try_from(progress.step).map_err(|error| error.to_string())?,
                total_steps: progress.total_steps,
                sigma: progress.sigma,
                sigma_hat: progress.sigma_hat,
                next_sigma: progress.next_sigma,
                latent: values(backend, latent, context).map_err(|error| error.to_string())?,
                denoised: values(backend, denoised, context).map_err(|error| error.to_string())?,
            });
            Ok::<(), String>(())
        },
    )?;
    let latents = trace
        .latents
        .iter()
        .map(|tensor| values(backend, tensor, context))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ObservedRun {
        calls: calls.into_inner(),
        callbacks: callbacks.into_inner(),
        latents,
    })
}

#[test]
fn val_sampler_001_definition_provenance_and_adapter_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, UNI_PC_BH2_SAMPLER_ID);
    assert_eq!(fixture.feature_id, UNI_PC_BH2_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, UNI_PC_BH2_SOURCE_ORDINAL);
    assert_eq!(fixture.maximum_order, 3);
    assert_eq!(fixture.terminal_sigma_replacement, 0.001);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.source_ordinal, 43);
    assert!(!DEFINITION.stochastic);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(&fixture.identity)?)?,
        &DEFINITION
    );

    let root = workspace_root()?;
    for (path, expected) in [
        (&fixture.source.uni_pc_path, &fixture.source.uni_pc_sha256),
        (
            &fixture.source.samplers_path,
            &fixture.source.samplers_sha256,
        ),
        (&fixture.source.catalog_path, &fixture.source.catalog_sha256),
    ] {
        assert_eq!(digest(&root.join(path))?, *expected);
    }
    let source = fs::read_to_string(root.join(&fixture.source.uni_pc_path))?;
    let bh = source_range(&source, fixture.source.bh_lines);
    assert!(bh.contains("elif self.variant == 'bh2':"));
    assert!(bh.contains("B_h = torch.expm1(hh)"));
    assert!(bh.contains("rhos_c = torch.linalg.solve(R, b)"));
    let wrapper = source_range(&source, fixture.source.wrapper_lines);
    assert!(wrapper.contains("variant='bh2'"));
    assert!(wrapper.contains("timesteps[-1] = 0.001"));

    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(samplers.lines().nth(fixture.source.registry_line - 1).is_some_and(
        |line| line.contains("[\"ddim\", \"uni_pc\", \"uni_pc_bh2\"]")
    ));
    assert!(samplers.lines().nth(fixture.source.selector_line - 1).is_some_and(
        |line| line.contains("elif name == \"uni_pc_bh2\"")
    ));
    assert!(samplers
        .lines()
        .nth(fixture.source.discard_policy_line - 1)
        .is_some_and(|line| line.contains("'uni_pc_bh2'")));
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(catalog
        .lines()
        .nth(fixture.source.catalog_line - 1)
        .is_some_and(|line| {
            line.starts_with("sampler,uni_pc_bh2,") && line.ends_with(",COMFY-MODEL-0202")
        }));

    assert!(IMPLEMENTATION.contains("sample_uni_pc_variant("));
    assert!(IMPLEMENTATION.contains("UniPcVariant::Bh2"));
    for forbidden in [
        "fn multistep_update",
        "fn bh_system_rhs",
        "fn solve_vandermonde",
        "fn shift_history",
        "fn sigma_",
        "struct SamplingSession",
        "observe_step",
        ".commit(",
        "CancellationToken",
        "Rng",
        "std::fs",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "adapter duplicates owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_every_bh2_input_predictor_callback_and_latent()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let observed = execute(&fixture, &backend, &context, 17)?;

    assert_eq!(observed.calls.len(), fixture.model_calls.len());
    for (actual, expected) in observed.calls.iter().zip(&fixture.model_calls) {
        assert_eq!(actual.step, expected.target_index);
        assert_eq!(
            actual.stage,
            if expected.stage == "initial" {
                UniPcDenoiserStage::Initial
            } else {
                UniPcDenoiserStage::Corrector
            }
        );
        assert!((actual.sigma - expected.sigma).abs() <= fixture.tolerance);
        assert_close(&actual.input, &expected.input, fixture.tolerance);
        assert_close(&actual.output, &expected.output, fixture.tolerance);
        let input_scale = (1.0_f32 + actual.sigma.powi(2)).sqrt();
        let predictor = actual
            .input
            .iter()
            .map(|value| value / input_scale)
            .collect::<Vec<_>>();
        assert_close(&predictor, &expected.predictor, fixture.tolerance);
        let analytical = actual
            .input
            .iter()
            .zip(&fixture.model_offsets)
            .map(|(input, offset)| {
                fixture.model_scale * input + fixture.model_sigma_scale * actual.sigma + offset
            })
            .collect::<Vec<_>>();
        assert_close(&actual.output, &analytical, fixture.tolerance);
    }

    assert_eq!(observed.callbacks.len(), fixture.callbacks.len());
    for (index, (actual, expected)) in observed
        .callbacks
        .iter()
        .zip(&fixture.callbacks)
        .enumerate()
    {
        assert_eq!(actual.step, expected.step);
        assert_eq!(actual.total_steps, u32::try_from(fixture.callbacks.len())?);
        assert_eq!(actual.sigma, fixture.sigmas[index]);
        assert_eq!(actual.sigma_hat, fixture.sigmas[index]);
        assert_eq!(actual.next_sigma, fixture.sigmas[index + 1]);
        assert_close(&actual.latent, &expected.latent, fixture.tolerance);
        assert_close(&actual.denoised, &expected.denoised, fixture.tolerance);
    }

    assert_eq!(observed.latents.len(), fixture.trace_latents.len());
    for (actual, expected) in observed.latents.iter().zip(&fixture.trace_latents) {
        assert_close(actual, expected, fixture.tolerance);
    }
    assert_close(
        observed.latents.last().ok_or("missing terminal latent")?,
        &fixture.terminal,
        fixture.tolerance,
    );
    Ok(())
}

#[test]
fn val_rng_001_is_seed_independent_and_has_no_rng_phase() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    assert_eq!(
        execute(&fixture, &backend, &context, 17)?,
        execute(&fixture, &backend, &context, u64::MAX)?
    );
    assert!(!IMPLEMENTATION.contains("Rng"));
    assert!(!IMPLEMENTATION.contains("noise_request"));
    Ok(())
}

#[test]
fn val_sampler_001_preserves_canonical_typed_failures_cancellation_and_atomicity()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let steps = u32::try_from(fixture.sigmas.len().saturating_sub(1))?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;

    let wrong = sample_uni_pc_bh2(
        &backend,
        plan("uni_pc", &profile, steps, 17)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(wrong, Err(UniPcError::WrongSampler { expected, .. }) if expected == UNI_PC_BH2_SAMPLER_ID));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let cancelled_result = sample_uni_pc_bh2(
        &backend,
        plan(&fixture.identity, &profile, steps, 17)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &cancelled_context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(cancelled_result, Err(UniPcError::Cancelled { step: 0 })));

    let callback_count = RefCell::new(0_usize);
    let denoiser_result = sample_uni_pc_bh2(
        &backend,
        plan(&fixture.identity, &profile, steps, 17)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_input, _sigma, step, stage| Err(format!("failure at {step} {stage:?}")),
        |_progress, _latent, _denoised| {
            *callback_count.borrow_mut() += 1;
            Ok::<(), String>(())
        },
    );
    assert!(matches!(
        denoiser_result,
        Err(UniPcError::Denoiser {
            step: 0,
            stage: UniPcDenoiserStage::Initial,
            ..
        })
    ));
    assert_eq!(*callback_count.borrow(), 0);

    let callback_result = sample_uni_pc_bh2(
        &backend,
        plan(&fixture.identity, &profile, steps, 17)?,
        &profile,
        initial,
        &fixture.sigmas,
        &context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.model_calls[0].output,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Err::<(), _>("callback failure"),
    );
    assert!(matches!(
        callback_result,
        Err(UniPcError::Sampling(SamplingError::Callback(reason)))
            if reason == "callback failure"
    ));
    Ok(())
}
