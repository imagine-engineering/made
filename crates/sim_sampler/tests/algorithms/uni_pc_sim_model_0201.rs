use sim_sampler::{
    SamplerIdentity, SamplerRegistry, SamplingPlan, SamplingProfileIdentity,
    generated_uni_pc_sim_model_0201::{
        DEFINITION, UNI_PC_FEATURE_ID, UNI_PC_MAX_ORDER, UNI_PC_SAMPLER_ID, UNI_PC_SOURCE_ORDINAL,
        UNI_PC_TERMINAL_SIGMA, UniPcDenoiserStage, UniPcError, sample_uni_pc,
    },
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, error::Error, fs, path::Path};

const FIXTURE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/samplers/uni_pc_comfy_model_0201/trajectory.json"
));
const IMPLEMENTATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/algorithms/uni_pc_sim_model_0201.rs"
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
    transitions: Vec<TransitionFixture>,
    callbacks: Vec<CallbackFixture>,
    trace_latents: Vec<Vec<f32>>,
    terminal: Vec<f32>,
    bh2_terminal: Vec<f32>,
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
    traversal_lines: [usize; 2],
    wrapper_lines: [usize; 2],
    registry_line: usize,
    selector_line: usize,
    catalog_line: usize,
}

#[derive(Debug, Deserialize)]
struct ModelCallFixture {
    target_index: usize,
    stage: String,
    sigma: f32,
    input: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TransitionFixture {
    target_index: usize,
    order: usize,
    corrector: bool,
    predictor: Vec<f32>,
    corrected: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CallbackFixture {
    step: usize,
    latent: Vec<f32>,
    denoised: Vec<f32>,
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
    Ok(SamplingProfileIdentity::new("uni-pc-row-v1")?)
}

fn plan(
    sampler: &str,
    profile: &SamplingProfileIdentity,
    steps: u32,
) -> Result<SamplingPlan, Box<dyn Error>> {
    plan_with_seed(sampler, profile, steps, 17)
}

fn plan_with_seed(
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
    tensor: &sim_tensor::Tensor,
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

#[test]
fn val_sampler_001_definition_provenance_and_family_ownership_are_exact()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.identity, UNI_PC_SAMPLER_ID);
    assert_eq!(fixture.feature_id, UNI_PC_FEATURE_ID);
    assert_eq!(fixture.source_ordinal, UNI_PC_SOURCE_ORDINAL);
    assert_eq!(fixture.maximum_order, UNI_PC_MAX_ORDER);
    assert_eq!(fixture.terminal_sigma_replacement, UNI_PC_TERMINAL_SIGMA);
    assert_eq!(DEFINITION.identity, fixture.identity);
    assert_eq!(DEFINITION.source_ordinal, 42);
    assert!(!DEFINITION.stochastic);
    assert!(DEFINITION.aliases.is_empty());
    assert_eq!(
        SamplerRegistry::foundational()?.resolve(&SamplerIdentity::new(UNI_PC_SAMPLER_ID)?)?,
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
    for fragment in [
        "def multistep_uni_pc_bh_update(",
        "if self.variant == 'bh1':",
        "B_h = hh",
        "elif self.variant == 'bh2':",
        "B_h = torch.expm1(hh)",
        "rhos_c = torch.linalg.solve(R, b)",
    ] {
        assert!(bh.contains(fragment), "missing BH-family source {fragment}");
    }
    let traversal = source_range(&source, fixture.source.traversal_lines);
    for fragment in [
        "order=3",
        "lower_order_final=True",
        "use_corrector = False",
        "callback({'x': x, 'i': step_index, 'denoised': model_prev_list[-1]})",
    ] {
        assert!(
            traversal.contains(fragment),
            "missing traversal source {fragment}"
        );
    }
    let wrapper = source_range(&source, fixture.source.wrapper_lines);
    for fragment in [
        "timesteps[-1] = 0.001",
        "noise = noise / torch.sqrt(1.0 + timesteps[0] ** 2.0)",
        "order = min(3, len(timesteps) - 2)",
        "variant='bh2'",
    ] {
        assert!(
            wrapper.contains(fragment),
            "missing wrapper source {fragment}"
        );
    }
    let samplers = fs::read_to_string(root.join(&fixture.source.samplers_path))?;
    assert!(
        samplers
            .lines()
            .nth(fixture.source.registry_line - 1)
            .is_some_and(|line| line.contains("[\"ddim\", \"uni_pc\", \"uni_pc_bh2\"]"))
    );
    assert!(
        samplers
            .lines()
            .nth(fixture.source.selector_line - 1)
            .is_some_and(|line| line.contains("if name == \"uni_pc\""))
    );
    let catalog = fs::read_to_string(root.join(&fixture.source.catalog_path))?;
    assert!(
        catalog
            .lines()
            .nth(fixture.source.catalog_line - 1)
            .is_some_and(|line| {
                line.starts_with("sampler,uni_pc,") && line.ends_with(",COMFY-MODEL-0201")
            })
    );
    for forbidden in [
        "struct SamplingSession",
        "struct SamplingProgress",
        "struct CancellationToken",
        "struct RngStream",
        "CompatibilityNoiseRequest",
        "std::fs",
        "rusqlite",
        "sqlx",
    ] {
        assert!(
            !IMPLEMENTATION.contains(forbidden),
            "duplicate owner {forbidden}"
        );
    }
    assert!(IMPLEMENTATION.contains("enum UniPcVariant"));
    assert!(IMPLEMENTATION.contains("UniPcVariant::Bh1"));
    assert!(IMPLEMENTATION.contains("UniPcVariant::Bh2"));
    Ok(())
}

#[test]
fn val_sampling_foundation_001_matches_every_bh1_evaluation_callback_and_latent()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let call_index = RefCell::new(0_usize);
    let callback_index = RefCell::new(0_usize);
    let trace = sample_uni_pc(
        &backend,
        plan(
            &fixture.identity,
            &profile,
            u32::try_from(fixture.sigmas.len() - 1)?,
        )?,
        &profile,
        tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
        &fixture.sigmas,
        &context,
        |input, sigma, step, stage| {
            let index = *call_index.borrow();
            let expected = fixture
                .model_calls
                .get(index)
                .ok_or("unexpected model call")?;
            assert_eq!(step, expected.target_index);
            assert_eq!(
                stage,
                if expected.stage == "initial" {
                    UniPcDenoiserStage::Initial
                } else {
                    UniPcDenoiserStage::Corrector
                }
            );
            assert!((sigma - expected.sigma).abs() <= fixture.tolerance);
            assert_close(
                &values(&backend, input, &context).map_err(|error| error.to_string())?,
                &expected.input,
                fixture.tolerance,
            );
            *call_index.borrow_mut() += 1;
            tensor_from_f32(&backend, &fixture.shape, &expected.output, &context)
                .map_err(|error| error.to_string())
        },
        |progress, latent, denoised| {
            let index = *callback_index.borrow();
            let expected = fixture.callbacks.get(index).ok_or("unexpected callback")?;
            assert_eq!(
                usize::try_from(progress.step).map_err(|error| error.to_string())?,
                expected.step
            );
            assert_eq!(
                progress.total_steps,
                u32::try_from(fixture.callbacks.len()).map_err(|error| error.to_string())?
            );
            assert_eq!(progress.sigma, fixture.sigmas[index]);
            assert_eq!(progress.sigma_hat, fixture.sigmas[index]);
            assert_eq!(progress.next_sigma, fixture.sigmas[index + 1]);
            assert_close(
                &values(&backend, latent, &context).map_err(|error| error.to_string())?,
                &expected.latent,
                fixture.tolerance,
            );
            assert_close(
                &values(&backend, denoised, &context).map_err(|error| error.to_string())?,
                &expected.denoised,
                fixture.tolerance,
            );
            *callback_index.borrow_mut() += 1;
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(*call_index.borrow(), fixture.model_calls.len());
    assert_eq!(*callback_index.borrow(), fixture.callbacks.len());
    for call in &fixture.model_calls {
        let analytical = call
            .input
            .iter()
            .zip(&fixture.model_offsets)
            .map(|(input, offset)| {
                fixture.model_scale * input + fixture.model_sigma_scale * call.sigma + offset
            })
            .collect::<Vec<_>>();
        assert_close(&call.output, &analytical, fixture.tolerance);
    }
    assert_eq!(trace.latents.len(), fixture.trace_latents.len());
    for (actual, expected) in trace.latents.iter().zip(&fixture.trace_latents) {
        assert_close(
            &values(&backend, actual, &context)?,
            expected,
            fixture.tolerance,
        );
    }
    assert_close(
        &values(
            &backend,
            trace.latents.last().ok_or("missing terminal")?,
            &context,
        )?,
        &fixture.terminal,
        fixture.tolerance,
    );
    assert_eq!(fixture.transitions.len(), 5);
    assert_eq!(
        fixture
            .transitions
            .iter()
            .map(|transition| transition.order)
            .collect::<Vec<_>>(),
        [1, 2, 3, 2, 1]
    );
    for transition in &fixture.transitions {
        assert_eq!(transition.corrector, transition.target_index < 5);
        assert_eq!(transition.predictor.len(), fixture.initial.len());
        assert_eq!(transition.corrected.len(), fixture.initial.len());
        if transition.corrector {
            let call = fixture
                .model_calls
                .iter()
                .find(|call| {
                    call.target_index == transition.target_index && call.stage == "corrector"
                })
                .ok_or("missing predictor model call")?;
            let input_scale = (1.0_f32 + call.sigma.powi(2)).sqrt();
            let physical_predictor = transition
                .predictor
                .iter()
                .map(|value| value * input_scale)
                .collect::<Vec<_>>();
            assert_close(&call.input, &physical_predictor, fixture.tolerance);
        }
        let callback_index = match transition.target_index {
            1..=3 => Some(transition.target_index),
            5 => Some(4),
            _ => None,
        };
        if let Some(callback_index) = callback_index {
            let callback = fixture
                .callbacks
                .get(callback_index)
                .ok_or("missing corrected callback")?;
            assert_close(&callback.latent, &transition.corrected, fixture.tolerance);
        }
    }
    assert_ne!(fixture.terminal, fixture.bh2_terminal);
    Ok(())
}

#[test]
fn val_rng_001_is_deterministic_and_has_no_rng_phase() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let run = |seed| -> Result<_, Box<dyn Error>> {
        let call_index = RefCell::new(0_usize);
        Ok(sample_uni_pc(
            &backend,
            plan_with_seed(
                &fixture.identity,
                &profile,
                u32::try_from(fixture.sigmas.len() - 1)?,
                seed,
            )?,
            &profile,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            &fixture.sigmas,
            &context,
            |_input, _sigma, _step, _stage| {
                let index = *call_index.borrow();
                *call_index.borrow_mut() += 1;
                let call = fixture
                    .model_calls
                    .get(index)
                    .ok_or_else(|| "unexpected denoiser call".to_owned())?;
                tensor_from_f32(&backend, &fixture.shape, &call.output, &context)
                    .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        )?)
    };
    let first = run(17)?;
    let second = run(u64::MAX)?;
    for (first, second) in first.latents.iter().zip(&second.latents) {
        assert_eq!(
            values(&backend, first, &context)?,
            values(&backend, second, &context)?
        );
    }
    assert!(!IMPLEMENTATION.contains("Rng"));
    assert!(!IMPLEMENTATION.contains("noise_request"));
    Ok(())
}

#[test]
fn val_sampler_001_boundaries_failures_and_cancellation_are_typed_and_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let profile = profile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial = tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?;
    let steps = u32::try_from(fixture.sigmas.len() - 1)?;

    let wrong = sample_uni_pc(
        &backend,
        plan("euler", &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(wrong, Err(UniPcError::WrongSampler { .. })));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    let error = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &cancelled_context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(error, Err(UniPcError::Cancelled { step: 0 })));

    let callback_error = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
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
        callback_error,
        Err(UniPcError::Sampling(sim_sampler::SamplingError::Callback(reason)))
            if reason == "callback failure"
    ));

    let call = RefCell::new(0_usize);
    let denoiser_error = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_input, _sigma, _step, stage| {
            let index = *call.borrow();
            *call.borrow_mut() += 1;
            if stage == UniPcDenoiserStage::Corrector {
                return Err("corrector failure".to_owned());
            }
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.model_calls[index].output,
                &context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        denoiser_error,
        Err(UniPcError::Denoiser {
            step: 1,
            stage: UniPcDenoiserStage::Corrector,
            ..
        })
    ));

    let corrector_cancellation = CancellationToken::default();
    let corrector_context = execution_context(&backend, &authority, &corrector_cancellation)?;
    let call = RefCell::new(0_usize);
    let cancelled_during_corrector = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &corrector_context,
        |_input, _sigma, _step, stage| {
            let index = *call.borrow();
            *call.borrow_mut() += 1;
            let model_call = fixture
                .model_calls
                .get(index)
                .ok_or_else(|| "unexpected denoiser call".to_owned())?;
            let output = tensor_from_f32(
                &backend,
                &fixture.shape,
                &model_call.output,
                &corrector_context,
            )
            .map_err(|error| error.to_string())?;
            if stage == UniPcDenoiserStage::Corrector {
                corrector_cancellation.cancel();
            }
            Ok(output)
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        cancelled_during_corrector,
        Err(UniPcError::Cancelled { step: 1 })
    ));

    let descriptor_error = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(&backend, &[1], &[0.0], &context).map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        descriptor_error,
        Err(UniPcError::DenoiserContract {
            step: 0,
            stage: UniPcDenoiserStage::Initial
        })
    ));

    let non_finite = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(&backend, &fixture.shape, &[f32::NAN, 0.0], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        non_finite,
        Err(UniPcError::NonFinite { step: 0, .. })
    ));

    let callback_cancellation = CancellationToken::default();
    let callback_context = execution_context(&backend, &authority, &callback_cancellation)?;
    let error = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &callback_context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(
                &backend,
                &fixture.shape,
                &fixture.model_calls[0].output,
                &callback_context,
            )
            .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| {
            callback_cancellation.cancel();
            Ok::<(), String>(())
        },
    );
    assert!(matches!(error, Err(UniPcError::Cancelled { step: 0 })));

    let schedule_mismatch = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, 2)?,
        &profile,
        initial.clone(),
        &[2.0, 0.0],
        &context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        schedule_mismatch,
        Err(UniPcError::Sampling(
            sim_sampler::SamplingError::ScheduleLength {
                expected: 3,
                actual: 2
            }
        ))
    ));

    for invalid_sigmas in [
        [2.0_f32, 2.0, 0.0],
        [2.0, 3.0, 0.0],
        [2.0, -1.0, 0.0],
        [2.0, f32::NAN, 0.0],
        [2.0, f32::INFINITY, 0.0],
    ] {
        let invalid = sample_uni_pc(
            &backend,
            plan(&fixture.identity, &profile, 2)?,
            &profile,
            initial.clone(),
            &invalid_sigmas,
            &context,
            |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
            |_progress, _latent, _denoised| Ok::<(), String>(()),
        );
        assert!(matches!(
            invalid,
            Err(UniPcError::Sampling(
                sim_sampler::SamplingError::InvalidSigma { .. }
            ))
        ));
    }

    let tiny_sigma = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, 2)?,
        &profile,
        initial.clone(),
        &[2.0, f32::MIN_POSITIVE, 0.0],
        &context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(&backend, &fixture.shape, &[0.35, -0.25], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        tiny_sigma,
        Err(UniPcError::InvalidCoefficient { step: 1, .. })
    ));

    let constrained_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4)?,
        &cancellation,
    );
    let out_of_memory = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, steps)?,
        &profile,
        initial.clone(),
        &fixture.sigmas,
        &constrained_context,
        |_input, _sigma, _step, _stage| Err("must not run".to_owned()),
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    );
    assert!(matches!(
        out_of_memory,
        Err(UniPcError::Tensor(_)) | Err(UniPcError::TensorKernel(_))
    ));

    let short_sigmas = [2.0_f32, 0.0];
    let callbacks = RefCell::new(0_usize);
    let short = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, 1)?,
        &profile,
        initial.clone(),
        &short_sigmas,
        &context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(&backend, &fixture.shape, &[0.35, -0.25], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| {
            *callbacks.borrow_mut() += 1;
            Ok::<(), String>(())
        },
    )?;
    assert_eq!(*callbacks.borrow(), 1);
    let initial_scale = (1.0_f32 + short_sigmas[0].powi(2)).sqrt();
    let terminal_scale = (1.0_f32 + UNI_PC_TERMINAL_SIGMA.powi(2)).sqrt();
    let expected = fixture
        .initial
        .iter()
        .map(|value| value / initial_scale * terminal_scale)
        .collect::<Vec<_>>();
    assert_close(
        &values(
            &backend,
            short.latents.last().ok_or("missing short output")?,
            &context,
        )?,
        &expected,
        fixture.tolerance,
    );

    let nonzero_terminal_sigmas = [2.0_f32, 0.5];
    let nonzero_terminal = sample_uni_pc(
        &backend,
        plan(&fixture.identity, &profile, 1)?,
        &profile,
        initial,
        &nonzero_terminal_sigmas,
        &context,
        |_input, _sigma, _step, _stage| {
            tensor_from_f32(&backend, &fixture.shape, &[0.35, -0.25], &context)
                .map_err(|error| error.to_string())
        },
        |_progress, _latent, _denoised| Ok::<(), String>(()),
    )?;
    let nonzero_terminal_scale = (1.0_f32 + nonzero_terminal_sigmas[1].powi(2)).sqrt();
    let expected = fixture
        .initial
        .iter()
        .map(|value| value / initial_scale * nonzero_terminal_scale)
        .collect::<Vec<_>>();
    assert_close(
        &values(
            &backend,
            nonzero_terminal
                .latents
                .last()
                .ok_or("missing nonzero terminal output")?,
            &context,
        )?,
        &expected,
        fixture.tolerance,
    );

    for schedule in [
        [2.0_f32, 1.0, 0.0].as_slice(),
        [2.0, 1.2, 0.6, 0.0].as_slice(),
    ] {
        let call_count = RefCell::new(0_usize);
        let callback_count = RefCell::new(0_usize);
        sample_uni_pc(
            &backend,
            plan(
                &fixture.identity,
                &profile,
                u32::try_from(schedule.len() - 1)?,
            )?,
            &profile,
            tensor_from_f32(&backend, &fixture.shape, &fixture.initial, &context)?,
            schedule,
            &context,
            |input, sigma, _step, _stage| {
                let input =
                    tensor_to_f32(&backend, input, &context).map_err(|error| error.to_string())?;
                let output = input
                    .iter()
                    .zip(&fixture.model_offsets)
                    .map(|(input, offset)| {
                        fixture.model_scale * input + fixture.model_sigma_scale * sigma + offset
                    })
                    .collect::<Vec<_>>();
                *call_count.borrow_mut() += 1;
                tensor_from_f32(&backend, &fixture.shape, &output, &context)
                    .map_err(|error| error.to_string())
            },
            |_progress, _latent, _denoised| {
                *callback_count.borrow_mut() += 1;
                Ok::<(), String>(())
            },
        )?;
        assert_eq!(*call_count.borrow(), schedule.len() - 1);
        assert_eq!(*callback_count.borrow(), schedule.len() - 1);
    }

    assert!(matches!(
        SamplingPlan::new(&fixture.identity, "normal", profile, 17, 0, 1.0, 1.0,),
        Err(sim_sampler::SamplingError::ZeroSteps)
    ));
    Ok(())
}
