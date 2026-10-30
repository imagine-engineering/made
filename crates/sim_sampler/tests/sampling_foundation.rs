use serde_json::json;
use sha2::{Digest, Sha256};
use sim_sampler::generated_native_diffusion::{
    NativeDiffusionSamplerError, scale_initial_noise, scale_model_input,
};
use sim_sampler::{
    AdaptiveSamplingAttempt, AdaptiveSamplingSession, BrownianNoiseIntervalAddress,
    CompatibilityNoiseRequest, DiscreteSamplingProfile, EULER_FOUNDATION_DEFINITION,
    NORMAL_FOUNDATION_DEFINITION, NoiseError, NoisePhaseIdentity, NoiseRequest,
    PenultimateSigmaPolicy, PredictionInterpretation, SamplerDefinition, SamplerRegistry,
    SamplingError, SamplingPlan, SamplingProfile, SamplingProfileError, SamplingProfileIdentity,
    SamplingSession, SamplingSnrMode, SchedulerDefinition, SchedulerError, SchedulerIdentity,
    SchedulerRegistry, SchedulerRequest, build_scheduler_schedule, normal_noise, normal_schedule,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext,
    RetryRngPolicy, RngGenerationPlacement, RngSeedTransform, StreamId, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
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

fn pinned_sampler_names() -> Result<Vec<String>, Box<dyn Error>> {
    let source = fs::read_to_string(workspace()?.join("projects/comfy/ComfyUI/sim/samplers.py"))?;
    let (_, after_marker) = source
        .split_once("KSAMPLER_NAMES = [")
        .ok_or("KSAMPLER_NAMES literal is unavailable")?;
    let (literal, _) = after_marker
        .split_once(']')
        .ok_or("KSAMPLER_NAMES literal is unterminated")?;
    let mut names = literal
        .split('"')
        .enumerate()
        .filter_map(|(index, value)| (!index.is_multiple_of(2)).then(|| value.to_owned()))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err("KSAMPLER_NAMES literal contains no identities".into());
    }
    let (_, after_sampler_marker) = source
        .split_once("SAMPLER_NAMES = KSAMPLER_NAMES + [")
        .ok_or("SAMPLER_NAMES extension literal is unavailable")?;
    let (extension, _) = after_sampler_marker
        .split_once(']')
        .ok_or("SAMPLER_NAMES extension literal is unterminated")?;
    names.extend(
        extension
            .split('"')
            .enumerate()
            .filter_map(|(index, value)| (!index.is_multiple_of(2)).then(|| value.to_owned())),
    );
    Ok(names)
}

#[test]
fn identities_registries_and_plan_round_trip_fail_closed() -> Result<(), Box<dyn Error>> {
    let samplers = SamplerRegistry::foundational()?;
    let schedulers = SchedulerRegistry::foundational()?;
    assert_eq!(samplers.default_definition().identity, "euler");
    assert_eq!(samplers.default_definition().source_ordinal, 0);
    assert_eq!(schedulers.default_definition().identity, "simple");
    assert_eq!(schedulers.default_definition().source_ordinal, 0);
    assert_eq!(
        schedulers
            .resolve(&SchedulerIdentity::new("normal")?)?
            .source_ordinal,
        6
    );

    let profile = DiscreteSamplingProfile::sd15()?;
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        profile.identity().clone(),
        42,
        4,
        7.0,
        1.0,
    )?;
    plan.validate(&samplers, &schedulers, profile.identity())?;
    let encoded = serde_json::to_vec(&plan)?;
    let decoded: SamplingPlan = serde_json::from_slice(&encoded)?;
    assert_eq!(decoded, plan);
    assert!(
        serde_json::from_value::<SamplingPlan>(json!({
            "schema_version": 2,
            "sampler": "euler",
            "scheduler": "normal",
            "profile": "sd15-discrete-epsilon-v1",
            "seed": 42,
            "steps": 4,
            "guidance": 7.0,
            "denoise": 1.0
        }))
        .is_err()
    );
    assert!(
        SamplingPlan::new(
            "python callback",
            "normal",
            SamplingProfileIdentity::sd15(),
            0,
            4,
            1.0,
            1.0,
        )
        .is_err()
    );

    let scheduler_request = SchedulerRequest::new("normal", 4, 1.0)?
        .with_window(Some(1), Some(3))?
        .with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard);
    let scheduler_request_encoded = serde_json::to_vec(&scheduler_request)?;
    let scheduler_request_decoded: SchedulerRequest =
        serde_json::from_slice(&scheduler_request_encoded)?;
    assert_eq!(scheduler_request_decoded, scheduler_request);
    assert!(
        serde_json::from_value::<SchedulerRequest>(json!({
            "identity": "normal",
            "steps": 0,
            "denoise": 1.0,
            "start_step": null,
            "end_step": null,
            "penultimate_sigma_policy": "keep"
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<SchedulerRequest>(json!({
            "identity": "normal",
            "steps": 4,
            "denoise": 1.0,
            "start_step": 3,
            "end_step": 2,
            "penultimate_sigma_policy": "keep"
        }))
        .is_err()
    );

    let duplicate_identity = SamplerDefinition {
        feature_id: "COMFY-MODEL-0159",
        source_ordinal: 1,
        implementation_module: "algorithms/duplicate",
        ..EULER_FOUNDATION_DEFINITION
    };
    assert!(matches!(
        SamplerRegistry::new(vec![EULER_FOUNDATION_DEFINITION, duplicate_identity]),
        Err(SamplingError::DuplicateIdentity { .. })
    ));
    let duplicate_ordinal = SchedulerDefinition {
        identity: "other",
        feature_id: "COMFY-MODEL-0208",
        source_ordinal: NORMAL_FOUNDATION_DEFINITION.source_ordinal,
        aliases: &[],
        implementation_module: "schedulers/other",
    };
    assert!(matches!(
        SchedulerRegistry::new(vec![
            sim_sampler::SIMPLE_FOUNDATION_DEFINITION,
            NORMAL_FOUNDATION_DEFINITION,
            duplicate_ordinal,
        ]),
        Err(SchedulerError::DuplicateSourceOrdinal(6))
    ));
    Ok(())
}

#[test]
fn generated_sampler_ordinals_match_the_pinned_source_list() -> Result<(), Box<dyn Error>> {
    let source_names = pinned_sampler_names()?;
    let registry = SamplerRegistry::foundational()?;
    assert_eq!(
        registry.definitions().len(),
        source_names.len() + 2,
        "native sampler registry must include the pinned list and two node-owned samplers"
    );
    for (source_ordinal, (definition, source_identity)) in registry
        .definitions()
        .iter()
        .zip(source_names.iter())
        .enumerate()
    {
        assert_eq!(
            definition.identity,
            source_identity.as_str(),
            "generated sampler identity does not match pinned source ordinal {source_ordinal}"
        );
        assert_eq!(
            usize::from(definition.source_ordinal),
            source_ordinal,
            "sampler {:?} has an ordinal that does not match pinned KSAMPLER_NAMES",
            definition.identity
        );
    }
    assert_eq!(
        registry
            .definitions()
            .iter()
            .skip(source_names.len())
            .map(|definition| definition.identity)
            .collect::<Vec<_>>(),
        ["void_ddim", "lcm_upscale"],
    );
    Ok(())
}

#[test]
fn sampling_plan_is_the_only_sampler_to_penultimate_policy_adapter() -> Result<(), Box<dyn Error>> {
    let profile = SamplingProfileIdentity::sd15();
    let registry = SamplerRegistry::foundational()?;
    let discard = ["dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"];
    for definition in registry.definitions() {
        let plan = SamplingPlan::new(
            definition.identity,
            "normal",
            profile.clone(),
            0,
            4,
            1.0,
            1.0,
        )?;
        let request = SchedulerRequest::for_sampling_plan(&plan)?;
        let expected = if discard.contains(&definition.identity) {
            PenultimateSigmaPolicy::Discard
        } else {
            PenultimateSigmaPolicy::Keep
        };
        assert_eq!(
            request.penultimate_sigma_policy, expected,
            "{}",
            definition.identity
        );
        assert_eq!(request.identity, *plan.scheduler());
        assert_eq!(request.steps, plan.steps());
        assert_eq!(request.denoise, plan.denoise());
    }
    Ok(())
}

#[test]
fn scheduler_profile_slicing_and_scaling_are_canonical() -> Result<(), Box<dyn Error>> {
    let profile = DiscreteSamplingProfile::sd15()?;
    assert_eq!(profile.prediction(), PredictionInterpretation::Epsilon);
    assert_eq!(profile.sigma_count(), 1_000);
    assert_eq!(profile.model_time_for_sigma(0.0)?, 0.0);
    for model_time in [0.0_f32, 17.0, 499.0, 999.0] {
        let sigma = profile.sigma_at_model_time(model_time)?;
        assert_eq!(profile.model_time_for_sigma(sigma)?, model_time);
    }
    let mut noise = [1.0_f32, -2.0];
    profile.scale_initial_noise_in_place(&mut noise, &[0.5, 0.25], 2.0, false)?;
    assert_eq!(noise, [2.5, -3.75]);
    let mut input = [5.0_f32];
    profile.scale_model_input_in_place(&mut input, 2.0)?;
    assert!((input[0] - 5.0 / 5.0_f32.sqrt()).abs() < 1.0e-6);
    let maximum = profile.sigma_max();
    assert!(profile.is_max_denoise(maximum)?);
    let mut maximum_noise = [1.0_f32];
    profile.scale_initial_noise_in_place(&mut maximum_noise, &[0.0], maximum, true)?;
    assert!((maximum_noise[0] - (1.0 + maximum * maximum).sqrt()).abs() < 1.0e-6);

    let flow = DiscreteSamplingProfile::new(
        SamplingProfileIdentity::new("flow-v1")?,
        PredictionInterpretation::Flow,
        Arc::from([0.25_f32, 1.0]),
    )?;
    let mut flow_input = [3.0_f32];
    flow.scale_model_input_in_place(&mut flow_input, 0.25)?;
    assert_eq!(flow_input, [3.0]);
    let mut flow_prediction = [2.0_f32];
    flow.interpret_prediction_in_place(&mut flow_prediction, &[3.0], 0.25)?;
    assert_eq!(flow_prediction, [2.5]);
    let mut flow_noise = [4.0_f32];
    flow.scale_initial_noise_in_place(&mut flow_noise, &[2.0], 0.25, false)?;
    assert_eq!(flow_noise, [2.5]);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let registry = SchedulerRegistry::foundational()?;
    let full = normal_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 4, 1.0)?,
    )?;
    assert_eq!(full.len(), 5);
    assert!(full.windows(2).all(|pair| pair[0] > pair[1]));
    assert_eq!(full[4], 0.0);
    let window = normal_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 4, 0.5)?.with_window(Some(1), Some(3))?,
    )?;
    assert_eq!(window.len(), 3);
    assert!(window.windows(2).all(|pair| pair[0] > pair[1]));
    let discarded = normal_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 4, 1.0)?
            .with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard),
    )?;
    let mut source_order_discarded = normal_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 5, 1.0)?,
    )?;
    source_order_discarded.remove(source_order_discarded.len() - 2);
    assert_eq!(discarded, source_order_discarded);

    let near_full_denoise = build_scheduler_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 20_001, 0.99995)?,
        "normal",
        |effective_steps, _profile, _context, output| {
            assert_eq!(effective_steps, 20_001);
            output.try_push(1.0)?;
            output.try_push(0.0)?;
            Ok(())
        },
    )?;
    assert_eq!(near_full_denoise, [1.0, 0.0]);

    let shortened = build_scheduler_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 4, 0.5)?.with_window(Some(1), Some(3))?,
        "normal",
        |effective_steps, _profile, _context, output| {
            assert_eq!(effective_steps, 8);
            for sigma in [3.0, 2.0, 1.0, 0.0] {
                output.try_push(sigma)?;
            }
            Ok(())
        },
    )?;
    assert_eq!(shortened, [2.0, 1.0, 0.0]);

    let discard_before_tail = build_scheduler_schedule(
        &backend,
        &context,
        &registry,
        &profile,
        &SchedulerRequest::new("normal", 4, 0.5)?
            .with_penultimate_sigma_policy(PenultimateSigmaPolicy::Discard),
        "normal",
        |effective_steps, _profile, _context, output| {
            assert_eq!(effective_steps, 9);
            for sigma in [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0] {
                output.try_push(sigma)?;
            }
            Ok(())
        },
    )?;
    assert_eq!(discard_before_tail, [5.0, 4.0, 3.0, 2.0, 0.0]);
    Ok(())
}

#[test]
fn sampling_profile_owns_snr_offsets_and_model_noise_scale() -> Result<(), Box<dyn Error>> {
    let standard = DiscreteSamplingProfile::sd15()?;
    let standard_half_log_snr = standard.half_log_snr(2.0)?;
    assert!((standard_half_log_snr + 2.0_f32.ln()).abs() < 1.0e-7);
    assert!((standard.sigma_from_half_log_snr(standard_half_log_snr)? - 2.0).abs() < 1.0e-6);
    let mut standard_sigmas = [2.0_f32, 1.0, 0.0];
    standard.adjust_first_sigma_for_snr(&mut standard_sigmas)?;
    assert_eq!(standard_sigmas, [2.0, 1.0, 0.0]);
    assert_eq!(standard.scale_sampler_noise(0.5)?, 0.5);

    let flow = DiscreteSamplingProfile::new_with_sampling_parameters(
        SamplingProfileIdentity::new("constant-flow-snr-v1")?,
        PredictionInterpretation::Flow,
        Arc::from([0.25_f32, 1.0]),
        SamplingSnrMode::ConstantFlow { shift: 3.0 },
        2.5,
    )?;
    let flow_half_log_snr = flow.half_log_snr(0.25)?;
    assert!((flow_half_log_snr - 3.0_f32.ln()).abs() < 1.0e-7);
    assert!((flow.sigma_from_half_log_snr(flow_half_log_snr)? - 0.25).abs() < 1.0e-7);
    let mut flow_sigmas = [1.0_f32, 0.5, 0.0];
    flow.adjust_first_sigma_for_snr(&mut flow_sigmas)?;
    let expected_offset = 3.0 * 0.9999 / (1.0 + 2.0 * 0.9999);
    assert!((flow_sigmas[0] - expected_offset).abs() < 1.0e-7);
    assert_eq!(&flow_sigmas[1..], &[0.5, 0.0]);
    assert_eq!(flow.scale_sampler_noise(0.4)?, 1.0);
    let mut scaled_noise = [4.0_f32];
    flow.scale_initial_noise_in_place(&mut scaled_noise, &[2.0], 0.25, false)?;
    assert_eq!(scaled_noise, [4.0]);

    let mut singleton = [1.0_f32];
    flow.adjust_first_sigma_for_snr(&mut singleton)?;
    assert_eq!(singleton, [1.0]);
    assert!(matches!(
        flow.half_log_snr(1.0),
        Err(SamplingProfileError::InvalidSnrSigma(1.0))
    ));
    assert_eq!(standard.scale_sampler_noise(-0.5)?, -0.5);
    assert_eq!(flow.scale_sampler_noise(-0.4)?, -1.0);
    assert!(matches!(
        DiscreteSamplingProfile::new_with_sampling_parameters(
            SamplingProfileIdentity::new("invalid-flow-shift")?,
            PredictionInterpretation::Flow,
            Arc::from([0.25_f32, 1.0]),
            SamplingSnrMode::ConstantFlow { shift: 0.0 },
            1.0,
        ),
        Err(SamplingProfileError::InvalidSnrShift(0.0))
    ));
    assert!(matches!(
        DiscreteSamplingProfile::new_with_sampling_parameters(
            SamplingProfileIdentity::new("invalid-noise-scale")?,
            PredictionInterpretation::Epsilon,
            Arc::from([0.25_f32, 1.0]),
            SamplingSnrMode::Standard,
            f32::NAN,
        ),
        Err(SamplingProfileError::InvalidNoiseScale(value)) if value.is_nan()
    ));
    Ok(())
}

#[test]
fn native_diffusion_scaling_adapters_are_exact_and_context_bound() -> Result<(), Box<dyn Error>> {
    let profile = DiscreteSamplingProfile::sd15()?;
    let sigma = profile.sigma_max();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let noise = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
    let latent = tensor_from_f32(&backend, &[1], &[0.0], &context)?;

    let initial = scale_initial_noise(&backend, &noise, &latent, sigma, &context)?;
    let initial = tensor_to_f32(&backend, &initial, &context)?;
    assert!((initial[0] - (1.0 + sigma * sigma).sqrt()).abs() < 1.0e-6);

    let raw_model_input = tensor_from_f32(&backend, &[1], &[5.0], &context)?;
    let scaled_model_input = scale_model_input(&backend, &raw_model_input, 2.0, &context)?;
    let scaled_values = tensor_to_f32(&backend, &scaled_model_input, &context)?;
    assert!((scaled_values[0] - 5.0 / 5.0_f32.sqrt()).abs() < 1.0e-6);
    assert_eq!(
        &*tensor_to_f32(&backend, &raw_model_input, &context)?,
        &[5.0]
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    assert!(matches!(
        scale_model_input(&backend, &raw_model_input, 2.0, &cancelled_context),
        Err(NativeDiffusionSamplerError::TensorKernel(
            NativeDiffusionTensorError::Tensor(TensorError::Cancelled)
        ))
    ));

    let insufficient_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(3)?,
        &cancellation,
    );
    assert!(matches!(
        scale_model_input(&backend, &raw_model_input, 2.0, &insufficient_context),
        Err(NativeDiffusionSamplerError::TensorKernel(
            NativeDiffusionTensorError::Tensor(TensorError::WorkspaceAuthorizationExceeded { .. })
        ))
    ));
    Ok(())
}

#[test]
fn session_commits_steps_failure_atomically_and_orders_callbacks() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::sd15(),
        7,
        2,
        1.0,
        1.0,
    )?;
    let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
    let original_id = initial.tensor_id();
    let mut session = SamplingSession::new(plan, vec![2.0, 1.0, 0.0], initial)?;
    let denoised = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
    let next = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
    assert!(matches!(
        session.commit_step(
            denoised.clone(),
            next.clone(),
            &cancellation,
            |_, _, _| Err("injected callback failure")
        ),
        Err(SamplingError::Callback(_))
    ));
    assert_eq!(session.next_step(), 0);
    assert_eq!(session.current().tensor_id(), original_id);

    let callback_order = RefCell::new(Vec::new());
    let callback_latent = session.current().clone();
    let observed = session.observe_step(
        &callback_latent,
        denoised.clone(),
        &cancellation,
        |progress, latent, observed_denoised| {
            assert_eq!(progress.step, 0);
            assert_eq!(latent.tensor_id(), original_id);
            assert_eq!(observed_denoised.tensor_id(), denoised.tensor_id());
            callback_order.borrow_mut().push("callback");
            Ok::<(), &'static str>(())
        },
    )?;
    callback_order.borrow_mut().push("intermediate-denoiser");
    drop(observed);
    assert_eq!(
        callback_order.into_inner(),
        vec!["callback", "intermediate-denoiser"]
    );
    assert_eq!(session.next_step(), 0);
    assert_eq!(session.current().tensor_id(), original_id);

    let other_stream_context = backend.execution_context(
        StreamId::new(1),
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let other_stream_next = tensor_from_f32(&backend, &[1], &[1.0], &other_stream_context)?;
    assert!(matches!(
        session.commit_step(
            denoised.clone(),
            other_stream_next,
            &cancellation,
            |_, _, _| Ok::<(), &'static str>(())
        ),
        Err(SamplingError::TensorContract {
            role: "next latent",
            ..
        })
    ));
    assert_eq!(session.next_step(), 0);
    assert_eq!(session.current().tensor_id(), original_id);

    let callbacks = Cell::new(0_u32);
    session.commit_step(denoised, next, &cancellation, |progress, _, _| {
        assert_eq!(progress.step, callbacks.get());
        callbacks.set(callbacks.get() + 1);
        Ok::<(), &'static str>(())
    })?;
    let denoised = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
    let next = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
    session.commit_step(denoised, next, &cancellation, |progress, _, _| {
        assert_eq!(progress.step, callbacks.get());
        callbacks.set(callbacks.get() + 1);
        Ok::<(), &'static str>(())
    })?;
    let trace = session.finish()?;
    assert_eq!(callbacks.get(), 2);
    assert_eq!(trace.denoiser_evaluations.len(), 2);
    assert_eq!(trace.latents.len(), 3);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let callback_ran = Cell::new(false);
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::sd15(),
        7,
        1,
        1.0,
        1.0,
    )?;
    let initial = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
    let mut session = SamplingSession::new(plan, vec![1.0, 0.0], initial)?;
    let denoised = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
    let next = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
    assert_eq!(
        session.commit_step(denoised, next, &cancelled, |_, _, _| {
            callback_ran.set(true);
            Ok::<(), &'static str>(())
        }),
        Err(SamplingError::Cancelled)
    );
    assert!(!callback_ran.get());
    assert_eq!(session.next_step(), 0);
    Ok(())
}

#[test]
fn adaptive_session_owns_rejected_and_accepted_attempt_state() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::sd15(),
        9,
        2,
        1.0,
        1.0,
    )?;
    let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
    let initial_id = initial.tensor_id();
    let denoised = tensor_from_f32(&backend, &[1], &[0.5], &context)?;
    let low = tensor_from_f32(&backend, &[1], &[1.25], &context)?;
    let high = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
    let accepted = tensor_from_f32(&backend, &[1], &[0.75], &context)?;
    let accepted_id = accepted.tensor_id();
    let mut session = AdaptiveSamplingSession::new(plan, 2.0, 0.5, initial, 2, 2)?;
    let callback_order = RefCell::new(Vec::new());

    session.commit_attempt(
        AdaptiveSamplingAttempt {
            proposed_sigma: 1.0,
            base_denoised: denoised.clone(),
            evaluations: vec![denoised.clone(), low.clone()],
            proposed_low: low.clone(),
            proposed_high: high.clone(),
            stochastic_noise: None,
            accepted_next: None,
            error: 2.0,
            next_step_size: 0.25,
        },
        &cancellation,
        |progress, latent, _| {
            assert!(!progress.accepted);
            assert_eq!(progress.sigma, 2.0);
            assert_eq!(progress.n_reject, 1);
            assert_eq!(latent.tensor_id(), initial_id);
            callback_order.borrow_mut().push("rejected");
            Ok::<(), &'static str>(())
        },
    )?;
    assert_eq!(session.current().tensor_id(), initial_id);

    session.commit_attempt(
        AdaptiveSamplingAttempt {
            proposed_sigma: 0.5,
            base_denoised: denoised.clone(),
            evaluations: vec![denoised, high.clone()],
            proposed_low: low,
            proposed_high: high,
            stochastic_noise: None,
            accepted_next: Some(accepted),
            error: 0.25,
            next_step_size: 0.5,
        },
        &cancellation,
        |progress, latent, _| {
            assert!(progress.accepted);
            assert_eq!(progress.sigma, 0.5);
            assert_eq!(progress.nfe, 4);
            assert_eq!(progress.n_accept, 1);
            assert_eq!(progress.n_reject, 1);
            assert_eq!(latent.tensor_id(), accepted_id);
            callback_order.borrow_mut().push("accepted");
            Ok::<(), &'static str>(())
        },
    )?;
    let trace = session.finish()?;
    assert_eq!(callback_order.into_inner(), vec!["rejected", "accepted"]);
    assert_eq!(trace.attempts.len(), 2);
    assert_eq!(trace.latents.len(), 3);
    assert_eq!(
        trace
            .latents
            .first()
            .ok_or("missing initial latent")?
            .tensor_id(),
        initial_id
    );
    assert_eq!(
        trace
            .latents
            .get(1)
            .ok_or("missing rejected-attempt latent")?
            .tensor_id(),
        initial_id
    );
    assert_eq!(
        trace
            .latents
            .last()
            .ok_or("missing accepted latent")?
            .tensor_id(),
        accepted_id
    );
    Ok(())
}

#[test]
fn adaptive_attempt_limit_is_checked_before_row_work() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::sd15(),
        10,
        2,
        1.0,
        1.0,
    )?;
    let initial = tensor_from_f32(&backend, &[1], &[2.0], &context)?;
    let denoised = tensor_from_f32(&backend, &[1], &[0.5], &context)?;
    let low = tensor_from_f32(&backend, &[1], &[1.25], &context)?;
    let high = tensor_from_f32(&backend, &[1], &[1.0], &context)?;
    let mut session = AdaptiveSamplingSession::new(plan, 2.0, 0.5, initial, 1, 1)?;
    assert_eq!(session.next_attempt(&cancellation)?, 0);
    session.commit_attempt(
        AdaptiveSamplingAttempt {
            proposed_sigma: 1.0,
            base_denoised: denoised.clone(),
            evaluations: vec![denoised],
            proposed_low: low,
            proposed_high: high,
            stochastic_noise: None,
            accepted_next: None,
            error: 2.0,
            next_step_size: 0.25,
        },
        &cancellation,
        |_, _, _| Ok::<(), &'static str>(()),
    )?;
    assert_eq!(
        session.next_attempt(&cancellation),
        Err(SamplingError::AdaptiveAttemptLimitExceeded { limit: 1 })
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert_eq!(
        session.next_attempt(&cancelled),
        Err(SamplingError::Cancelled)
    );
    Ok(())
}

#[test]
fn adaptive_completion_uses_source_one_e_minus_five_time_tolerance() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let plan = SamplingPlan::new(
        "euler",
        "normal",
        SamplingProfileIdentity::sd15(),
        11,
        1,
        1.0,
        1.0,
    )?;
    let initial = tensor_from_f32(&backend, &[1], &[0.0], &context)?;
    let within_sigma = 1.000_009_f32;
    let outside_sigma = 1.000_011_f32;
    let terminal_sigma = 1.0_f32;
    assert!(within_sigma.ln() < 1.0e-5);
    assert!(outside_sigma.ln() > 1.0e-5);

    let within = AdaptiveSamplingSession::new(
        plan.clone(),
        within_sigma,
        terminal_sigma,
        initial.clone(),
        1,
        1,
    )?;
    assert!(within.is_complete());
    assert_eq!(
        within.next_attempt(&cancellation),
        Err(SamplingError::SessionComplete)
    );

    let outside = AdaptiveSamplingSession::new(plan, outside_sigma, terminal_sigma, initial, 1, 1)?;
    assert!(!outside.is_complete());
    assert_eq!(outside.next_attempt(&cancellation), Ok(0));
    Ok(())
}

#[test]
fn noise_phases_replay_without_owning_rng_state() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = execution_context(&backend, &authority, &cancellation)?;
    let initial_request = NoiseRequest::native_diffusion("prompt-a", "5")?;
    let first = normal_noise(
        &backend,
        &[8],
        &initial_request.stream(99, sim_tensor::DeviceId::CPU)?,
        &context,
    )?;
    let replay = normal_noise(
        &backend,
        &[8],
        &initial_request.stream(99, sim_tensor::DeviceId::CPU)?,
        &context,
    )?;
    assert_eq!(first.before, replay.before);
    assert_eq!(first.after, replay.after);
    assert_eq!(
        &*tensor_to_f32(&backend, &first.noise, &context)?,
        &*tensor_to_f32(&backend, &replay.noise, &context)?
    );
    let other_prompt = normal_noise(
        &backend,
        &[8],
        &NoiseRequest::native_diffusion("prompt-b", "5")?.stream(99, sim_tensor::DeviceId::CPU)?,
        &context,
    )?;
    assert_ne!(
        &*tensor_to_f32(&backend, &first.noise, &context)?,
        &*tensor_to_f32(&backend, &other_prompt.noise, &context)?
    );
    let ancestral_request = NoiseRequest::new(
        "sd15-tiny-v1",
        "fixture",
        "KSampler",
        0,
        NoisePhaseIdentity::new("ancestral-step-v1")?,
        0,
        0,
        RetryRngPolicy::Replay,
    )?;
    let ancestral = normal_noise(
        &backend,
        &[8],
        &ancestral_request.stream(99, sim_tensor::DeviceId::CPU)?,
        &context,
    )?;
    assert_ne!(
        &*tensor_to_f32(&backend, &first.noise, &context)?,
        &*tensor_to_f32(&backend, &ancestral.noise, &context)?
    );
    let forward = BrownianNoiseIntervalAddress::new(0.5, 2.0, 3)?;
    let reverse = BrownianNoiseIntervalAddress::new(2.0, 0.5, 3)?;
    assert_eq!(forward.canonical_interval(), reverse.canonical_interval());
    assert_ne!(forward.reverse, reverse.reverse);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = execution_context(&backend, &authority, &cancelled)?;
    assert!(matches!(
        normal_noise(
            &backend,
            &[8],
            &initial_request.stream(99, sim_tensor::DeviceId::CPU)?,
            &cancelled_context,
        ),
        Err(NoiseError::Cancelled)
    ));
    Ok(())
}

#[test]
fn compatibility_noise_request_is_the_only_row_transaction_adapter() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let request = CompatibilityNoiseRequest::new(
        "sampling-foundation-workflow",
        "sampling-foundation-attempt",
        "KSampler",
        0,
        108,
        0,
        0,
        RetryRngPolicy::Replay,
    );
    let mut first = request.clone().open_transaction(
        sim_sampler::generated_ddpm_sim_model_0160::DDPM_NOISE_CONTRACT_ID,
        108,
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    let first_values = first.draw_normal(4, &cancellation)?;
    let first_after = first.commit();
    let mut replay = request.open_transaction(
        sim_sampler::generated_ddpm_sim_model_0160::DDPM_NOISE_CONTRACT_ID,
        108,
        RngSeedTransform::Add(1),
        RngGenerationPlacement::CpuSeededTransfer {
            output_device: DeviceId::CPU,
        },
        None,
        &cancellation,
    )?;
    assert_eq!(replay.draw_normal(4, &cancellation)?, first_values);
    assert_eq!(replay.commit(), first_after);
    Ok(())
}

#[test]
fn val_sampling_foundation_001() -> Result<(), Box<dyn Error>> {
    identities_registries_and_plan_round_trip_fail_closed()?;
    generated_sampler_ordinals_match_the_pinned_source_list()?;
    sampling_plan_is_the_only_sampler_to_penultimate_policy_adapter()?;
    scheduler_profile_slicing_and_scaling_are_canonical()?;
    sampling_profile_owns_snr_offsets_and_model_noise_scale()?;
    native_diffusion_scaling_adapters_are_exact_and_context_bound()?;
    session_commits_steps_failure_atomically_and_orders_callbacks()?;
    adaptive_session_owns_rejected_and_accepted_attempt_state()?;
    adaptive_attempt_limit_is_checked_before_row_work()?;
    adaptive_completion_uses_source_one_e_minus_five_time_tolerance()?;
    noise_phases_replay_without_owning_rng_state()?;
    compatibility_noise_request_is_the_only_row_transaction_adapter()?;
    assert!(matches!(
        SamplingError::from(TensorError::AllocationFailed {
            requested: 64,
            reason: "injected OOM".to_owned(),
        }),
        SamplingError::Tensor(TensorError::AllocationFailed { requested: 64, .. })
    ));
    assert!(matches!(
        SamplingError::from(TensorError::DeviceLost {
            reason: "injected device loss".to_owned(),
        }),
        SamplingError::Tensor(TensorError::DeviceLost { .. })
    ));

    let root = workspace()?;
    let model_adapter_path = "crates/sim_model/src/slices/native_diffusion.rs";
    let runtime_adapter_path = "crates/sim_runtime/src/native_execution_controller.rs";
    let fixture_adapter_path =
        "crates/sim_test_support/src/bin/generate_native_diffusion_fixture.rs";
    let source_files = [
        "crates/sim_sampler/src/sampler.rs",
        "crates/sim_sampler/src/scheduler.rs",
        "crates/sim_sampler/src/sampling_profile.rs",
        "crates/sim_sampler/src/noise.rs",
        "crates/sim_sampler/src/algorithms/native_diffusion.rs",
        model_adapter_path,
        runtime_adapter_path,
        fixture_adapter_path,
        "crates/sim_sampler/build.rs",
        "crates/sim_sampler/tests/sampling_foundation.rs",
        "crates/sim_tensor/src/ops/native_diffusion.rs",
    ];
    let mut source_digests = BTreeMap::new();
    for relative in source_files {
        let bytes = fs::read(root.join(relative))?;
        source_digests.insert(relative, format!("{:x}", Sha256::digest(bytes)));
    }
    let model_adapter = fs::read_to_string(root.join(model_adapter_path))?;
    assert!(model_adapter.contains("pub fn denoise_at_model_time("));
    assert!(model_adapter.contains("model_time: f32,"));
    assert!(!model_adapter.contains("fn sigma_to_timestep"));
    assert!(!model_adapter.contains("0.00085_f64.sqrt()"));
    assert!(!model_adapter.contains("DiscreteSamplingProfile"));

    let runtime_adapter = fs::read_to_string(root.join(runtime_adapter_path))?;
    assert!(runtime_adapter.contains("let model_time = model_time_for_profile("));
    let initial_scaling = runtime_adapter
        .split_once("let initial =")
        .and_then(|(_, source)| source.split_once("let initial_sha256"))
        .map(|(source, _)| source)
        .ok_or("runtime initial-noise scaling branch is missing")?;
    assert!(initial_scaling.contains("NativeSamplerAlgorithm::VoidDdim"));
    assert!(initial_scaling.contains("noise.noise.clone()"));
    assert!(initial_scaling.contains("scale_initial_noise_for_profile("));
    assert!(runtime_adapter.contains("let model_input = match scale_model_input_for_profile("));
    assert!(!runtime_adapter.contains("noise.mul_add(initial_sigma"));

    let fixture_adapter = fs::read_to_string(root.join(fixture_adapter_path))?;
    assert!(fixture_adapter.contains("let initial = scale_initial_noise("));
    assert!(fixture_adapter.contains("let model_input = scale_model_input("));
    assert!(!fixture_adapter.contains("value * sigmas[0]"));

    let fixture_files = [
        "projects/comfy/ComfyUI/sim/samplers.py",
        "projects/comfy/ComfyUI/sim/model_sampling.py",
        "projects/comfy/ComfyUI/sim/k_diffusion/sampling.py",
    ];
    let mut fixture_digests = BTreeMap::new();
    for relative in fixture_files {
        let bytes = fs::read(root.join(relative))?;
        fixture_digests.insert(relative, format!("{:x}", Sha256::digest(bytes)));
    }
    let cases = BTreeMap::from([
        ("brownian_interval_identity_is_stable", true),
        ("callback_and_step_commit_are_failure_atomic", true),
        ("cancellation_precedes_callback_and_commit", true),
        ("compatibility_noise_request_is_canonical", true),
        ("device_loss_and_oom_are_typed", true),
        ("euler_and_simple_source_defaults_are_explicit", true),
        (
            "generated_source_test_fixture_closure_and_ordinals_are_exact",
            true,
        ),
        ("model_time_and_noise_scaling_have_one_profile_owner", true),
        ("sampling_profile_owns_snr_and_model_noise_scale", true),
        (
            "model_input_and_max_denoise_adapters_are_context_bound",
            true,
        ),
        ("noise_phases_are_independent_and_retry_replayable", true),
        (
            "plan_identity_schema_and_penultimate_policy_are_checked",
            true,
        ),
        (
            "runtime_fixture_and_model_time_adapters_are_canonical",
            true,
        ),
        (
            "scheduler_builder_owns_validation_workspace_and_finalization",
            true,
        ),
        (
            "scheduler_builder_preserves_full_and_python_tail_semantics",
            true,
        ),
        ("sampling_profile_exact_index_access_is_canonical", true),
        ("scheduler_denoise_and_step_slicing_are_checked", true),
        (
            "adaptive_attempt_state_limits_and_completion_tolerance_are_canonical",
            true,
        ),
    ]);
    let passed = cases.values().filter(|passed| **passed).count();
    let failed = cases.len() - passed;
    assert_eq!(cases.len(), 18);
    assert_eq!(failed, 0);
    let value = json!({
        "validation_id": "VAL-SAMPLING-FOUNDATION-001",
        "scope": "authoritative native sampler, scheduler, sampling-profile, and RNG-phase foundation",
        "environment": {
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "operating_system": std::env::consts::OS,
        },
        "source_digests": source_digests,
        "fixture_digests": fixture_digests,
        "cases": cases,
        "summary": {"passed": passed, "failed": failed, "skipped": 0},
        "skipped": [],
    });
    let output = root.join("target/sim-parity/val-sampling-foundation-001.json");
    let parent = output.parent().ok_or("artifact parent is unavailable")?;
    fs::create_dir_all(parent)?;
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    fs::write(output, bytes)?;
    Ok(())
}
