use serde_json::Value;
use sim_nodes::{
    NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
};
use sim_runtime::{
    NativeHandleStoreGeneration, NativeNodeOutcome, NativeOpaqueHandle, NativePrimitive,
    NativeStoredPayload, NativeValue, generated_native_node_registry_projection,
};
use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, NORMAL_SCHEDULER_ID, SamplingPlan,
    SamplingProfileIdentity,
};
use sim_tensor::generated_native_diffusion::{tensor_from_f32, tensor_to_f32};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, RetryRngPolicy, StreamId, Tensor,
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;
const LIMIT: u64 = 256 * 1024 * 1024;
const FIXTURE: &str = include_str!("../fixtures/nodes/model-sampling-comfy-node-0571/fixture.json");
fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn sim_nodes::NativeHandleStore>,
    cancellation: CancellationToken,
    scratch_bytes: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("sampling".to_owned());
    let scratch = authority.authorize_workspace(scratch_bytes)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x64703), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x64701)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn case_inputs(case: &Value) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    case["inputs"]
        .as_object()
        .ok_or("source inputs absent")?
        .iter()
        .map(|(name, value)| {
            let value = if matches!(name.as_str(), "order" | "max_stage" | "scale_steps") {
                NativePrimitive::Integer(value.as_i64().ok_or("source integer absent")?)
            } else if let Some(value) = value.as_str() {
                NativePrimitive::String(value.to_owned())
            } else {
                NativePrimitive::Number(value.as_f64().ok_or("source number absent")?)
            };
            Ok((name.clone(), NativeValue::Primitive { value }))
        })
        .collect()
}
fn values(outcome: NativeNodeOutcome) -> Result<Vec<NativeValue>, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("ordinary values absent".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    Ok(outputs)
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 5e-7 + expected.abs() * 2e-6,
        "{actual} != {expected}"
    );
}

#[derive(Debug)]
struct PublicationBoundary {
    inner: Arc<dyn sim_nodes::NativeHandleStore>,
    remaining: std::sync::atomic::AtomicUsize,
    reject: bool,
}
impl sim_nodes::NativeHandleStore for PublicationBoundary {
    fn identity(&self) -> sim_nodes::NativeHandleStoreIdentity {
        self.inner.identity()
    }
    fn attempt_id(&self) -> AttemptId {
        self.inner.attempt_id()
    }
    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected: &sim_nodes::NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<sim_nodes::NativeResolvedPayload, sim_nodes::NativeHandleStoreError> {
        self.inner.resolve(handle, expected, cancellation)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, sim_nodes::NativeHandleStoreError> {
        let trigger = self
            .remaining
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            == 1;
        if trigger && self.reject {
            return Err(sim_nodes::NativeHandleStoreError::Rejected(
                "test output capacity exhausted".to_owned(),
            ));
        }
        let handle = self.inner.publish(payload, cancellation)?;
        if trigger {
            cancellation.cancel();
        }
        Ok(handle)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), sim_nodes::NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)
    }
}
fn source_parameters(sampler: &sim_sampler::NativeSamplerPayload) -> Result<Value, Box<dyn Error>> {
    use serde_json::json;
    use sim_sampler::NativeSamplerParameters as Parameters;
    let solver_name = |solver| match solver {
        sim_sampler::generated_dpmpp_2m_sde_sim_model_0168::Dpmpp2mSdeSolverType::Midpoint => {
            "midpoint"
        }
        sim_sampler::generated_dpmpp_2m_sde_sim_model_0168::Dpmpp2mSdeSolverType::Heun => "heun",
    };
    Ok(match *sampler.parameters() {
        Parameters::DpmAdaptive(options) => {
            json!({"order":options.order,"rtol":options.relative_tolerance,"atol":options.absolute_tolerance,"h_init":options.initial_step_size,"pcoeff":options.proportional_coefficient,"icoeff":options.integral_coefficient,"dcoeff":options.derivative_coefficient,"accept_safety":options.acceptance_safety,"eta":options.eta,"s_noise":options.noise_scale})
        }
        Parameters::Dpmpp2mSde(options) => {
            json!({"eta":options.eta(),"s_noise":options.noise_scale(),"solver_type":solver_name(options.solver_type())})
        }
        Parameters::Dpmpp2mSdeGpu(options) => {
            json!({"eta":options.eta,"s_noise":options.noise_scale,"solver_type":solver_name(options.solver_type)})
        }
        Parameters::Dpmpp2sAncestral(options) => {
            json!({"eta":options.eta,"s_noise":options.noise_scale})
        }
        Parameters::Dpmpp3mSde(options) => {
            json!({"eta":options.eta(),"s_noise":options.noise_scale()})
        }
        Parameters::EtaNoise { eta, noise_scale } => json!({"eta":eta,"s_noise":noise_scale}),
        Parameters::DpmppSde(options) => {
            json!({"eta":options.eta,"s_noise":options.noise_scale,"r":options.r})
        }
        Parameters::EulerAncestral(options) => {
            json!({"eta":options.eta(),"s_noise":options.noise_scale()})
        }
        Parameters::EulerAncestralCfgPp(options) => {
            json!({"eta":options.eta,"s_noise":options.noise_scale})
        }
        Parameters::Lcm(options) => {
            json!({"s_noise":options.noise_scale_start,"s_noise_end":options.noise_scale_end,"noise_clip_std":options.noise_clip_standard_deviations})
        }
        Parameters::ErSde {
            solver,
            eta,
            noise_scale,
            max_stage,
        } => {
            let scaler = match solver {
                sim_sampler::NativeErSdeSolver::ErSde => Value::Null,
                _ => json!([
                    0.0,
                    0.25_f64.powf(f64::from(eta) + 1.0),
                    1.0,
                    2.0_f64.powf(f64::from(eta) + 1.0)
                ]),
            };
            json!({"s_noise":noise_scale,"max_stage":max_stage,"noise_scaler":scaler})
        }
        Parameters::LcmUpscale(options) => {
            use sim_sampler::generated_lcm_upscale::LcmUpscaleMethod;
            let method = match options.method() {
                LcmUpscaleMethod::Bislerp => "bislerp",
                LcmUpscaleMethod::NearestExact => "nearest-exact",
                LcmUpscaleMethod::Bilinear => "bilinear",
                LcmUpscaleMethod::Area => "area",
                LcmUpscaleMethod::Bicubic => "bicubic",
            };
            json!({"total_upscale":options.total_upscale(),"upscale_steps":options.upscale_steps(),"upscale_method":method})
        }
        _ => return Err("unexpected part-four parameters".into()),
    })
}
fn compare_source(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (name, expected) in expected {
                compare_source(
                    actual.get(name).expect("source parameter missing"),
                    expected,
                );
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                compare_source(actual, expected);
            }
        }
        (Value::Number(actual), Value::Number(expected)) => close(
            actual.as_f64().expect("finite native number"),
            expected.as_f64().expect("finite source number"),
        ),
        _ => assert_eq!(actual, expected),
    }
}
#[test]
fn task647_sampler_nodes_preserve_source_parameters_cache_and_recovery()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for case in fixture["source_reference"]["cases"]
        .as_array()
        .ok_or("source cases absent")?
    {
        let name = case["class_type"].as_str().ok_or("class absent")?;
        let node = registry.node(name).ok_or("node absent")?;
        let inputs = case_inputs(case)?;
        let persisted = serde_json::to_string(&inputs)?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64704)));
        let cancellation = CancellationToken::default();
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        let token = node.cache_change_token(&inputs)?;
        let dependencies = node.cache_dependencies(&invocation, &inputs)?;
        assert!(node.demanded_lazy_inputs(&invocation, &inputs)?.is_empty());
        let outputs = values(smol::block_on(node.execute(invocation, inputs.clone()))?)?;
        assert_eq!(outputs.len(), 1);
        let Some(NativeValue::Handle { value: handle }) = outputs.first() else {
            return Err("sampler handle absent".into());
        };
        let payload = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Sampler(sampler) = payload.as_ref() else {
            return Err("sampler payload absent".into());
        };
        sampler.validate()?;
        let expected = case["output"]["algorithm"]
            .as_str()
            .ok_or("algorithm absent")?;
        assert_eq!(
            sampler.identity().as_str(),
            if expected == "sample_lcm_upscale" {
                "lcm_upscale"
            } else {
                expected
            }
        );
        compare_source(&source_parameters(sampler)?, &case["output"]["parameters"]);
        let recovered_generation = generation.fresh_generation()?;
        let recovered_store = recovered_generation.handle_store_for_attempt(store.attempt_id());
        assert!(
            recovered_store
                .resolve(handle, handle.handle_type(), &cancellation)
                .is_err()
        );
        let recovered_inputs = serde_json::from_str(&persisted)?;
        assert_eq!(node.cache_change_token(&recovered_inputs)?, token);
        let recovered_context = context(
            backend.clone(),
            &authority,
            recovered_store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        assert_eq!(
            node.cache_dependencies(&recovered_context, &recovered_inputs)?,
            dependencies
        );
        let recovered = values(smol::block_on(
            node.execute(recovered_context, recovered_inputs),
        )?)?;
        let Some(NativeValue::Handle {
            value: recovered_handle,
        }) = recovered.first()
        else {
            return Err("recovered sampler absent".into());
        };
        assert_eq!(
            recovered_store
                .resolve(
                    recovered_handle,
                    recovered_handle.handle_type(),
                    &cancellation
                )?
                .digest_sha256(),
            payload.digest_sha256()
        );
        assert_eq!(generation.len(), 1);
        assert_eq!(recovered_generation.len(), 1);
    }
    Ok(())
}
#[test]
fn task647_sampler_nodes_reject_invalid_inputs_and_rollback_publication()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for case in fixture["source_reference"]["cases"]
        .as_array()
        .ok_or("source cases absent")?
        .iter()
        .filter(|case| case["case"] == "defaults")
    {
        let name = case["class_type"].as_str().ok_or("class absent")?;
        let node = registry.node(name).ok_or("node absent")?;
        let good = case_inputs(case)?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64705)));
        let mut invalid = Vec::new();
        let mut extra = good.clone();
        extra.insert("extra".into(), NativeValue::List { values: Vec::new() });
        invalid.push(extra);
        for name in good.keys() {
            let mut missing = good.clone();
            missing.remove(name);
            invalid.push(missing);
            for bad in [
                NativeValue::List { values: Vec::new() },
                NativeValue::Primitive {
                    value: NativePrimitive::String("invalid".into()),
                },
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(-2),
                },
                NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(20001),
                },
            ] {
                let mut inputs = good.clone();
                inputs.insert(name.clone(), bad);
                invalid.push(inputs);
            }
        }
        for inputs in invalid {
            let invocation = context(
                backend.clone(),
                &authority,
                store.clone(),
                CancellationToken::default(),
                LIMIT,
            )?;
            assert!(node.cache_change_token(&inputs).is_err(), "{name}");
            assert!(
                node.demanded_lazy_inputs(&invocation, &inputs).is_err(),
                "{name}"
            );
            assert!(
                smol::block_on(node.execute(invocation, inputs)).is_err(),
                "{name}"
            );
            assert_eq!(generation.len(), 0);
        }
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                store.clone(),
                cancellation,
                LIMIT,
            )?,
            good.clone(),
        ))
        .err()
        .ok_or("pre-cancel succeeded")?;
        assert_eq!(error.kind, sim_nodes::NativeNodeFailureKind::Interrupted);
        for reject in [false, true] {
            let boundary = Arc::new(PublicationBoundary {
                inner: store.clone(),
                remaining: std::sync::atomic::AtomicUsize::new(1),
                reject,
            });
            let error = smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    boundary,
                    CancellationToken::default(),
                    LIMIT,
                )?,
                good.clone(),
            ))
            .err()
            .ok_or("publication boundary succeeded")?;
            assert_eq!(
                error.kind,
                if reject {
                    sim_nodes::NativeNodeFailureKind::Failure
                } else {
                    sim_nodes::NativeNodeFailureKind::Interrupted
                }
            );
            assert_eq!(generation.len(), 0);
        }
        values(smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                store.clone(),
                CancellationToken::default(),
                LIMIT,
            )?,
            good,
        ))?)?;
        assert_eq!(generation.len(), 1);
    }
    Ok(())
}
struct AffineDenoiser<'backend, 'execution> {
    backend: &'backend CpuBackend,
    execution: &'execution sim_tensor::ExecutionContext<'execution>,
    calls: usize,
    cfg_calls: usize,
}
impl sim_sampler::NativeSamplerDenoiser for AffineDenoiser<'_, '_> {
    fn denoise(&mut self, input: &Tensor, _sigma: f32, _step: usize) -> Result<Tensor, String> {
        self.calls += 1;
        let values = tensor_to_f32(self.backend, input, self.execution)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|value| value * 0.2 + 0.1)
            .collect::<Vec<_>>();
        tensor_from_f32(
            self.backend,
            input.descriptor().shape(),
            &values,
            self.execution,
        )
        .map_err(|error| error.to_string())
    }
    fn denoise_cfg_pp(
        &mut self,
        input: &Tensor,
        sigma: f32,
        step: usize,
    ) -> Result<sim_sampler::CfgPpDenoiserOutput, String> {
        self.cfg_calls += 1;
        let denoised = self.denoise(input, sigma, step)?;
        let count = usize::try_from(
            input
                .descriptor()
                .element_count()
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let unconditional_denoised = tensor_from_f32(
            self.backend,
            input.descriptor().shape(),
            &vec![0.0; count],
            self.execution,
        )
        .map_err(|error| error.to_string())?;
        Ok(sim_sampler::CfgPpDenoiserOutput {
            denoised,
            unconditional_denoised,
        })
    }
}
fn engine_sampler(
    case: &Value,
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
) -> Result<Arc<sim_sampler::NativeSamplerPayload>, Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    let name = case["class_type"].as_str().ok_or("class absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let mut descriptor = registry
        .descriptor(name)
        .ok_or("descriptor absent")?
        .clone();
    descriptor.output_node = true;
    let node_id = NodeId("sampler".into());
    let plan = CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x64712)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: name.into(),
                descriptor,
                inputs: case_inputs(case)?
                    .into_iter()
                    .map(|(name, value)| (name, InputBinding::Literal { value }))
                    .collect(),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: std::collections::BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id.clone()],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    let executor = NativeImageExecutor::new_with_generated_registry(
        sim_types::ProfileId(Uuid::from_u128(0x64710)),
        BTreeMap::new(),
        true,
        backend,
    )?;
    let attempt = AttemptId(Uuid::from_u128(0x64711));
    let cancellation = CancellationToken::default();
    let result = executor.execute_blocking(
        &plan,
        attempt,
        cancellation.clone(),
        0,
        authority.authorize_workspace(LIMIT)?,
    )?;
    assert_eq!(
        result.report.state,
        AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    let Some(NativeValue::Handle { value: handle }) = result
        .report
        .outputs
        .get(&node_id)
        .and_then(|outputs| outputs.first())
    else {
        return Err("engine sampler output absent".into());
    };
    let store = executor
        .handle_store_generation_for_test()
        .handle_store_for_attempt(attempt);
    let payload = store.resolve(handle, handle.handle_type(), &cancellation)?;
    let NativeStoredPayload::Sampler(sampler) = payload.as_ref() else {
        return Err("engine sampler payload absent".into());
    };
    Ok(sampler.clone())
}
#[test]
fn task647_engine_sampler_outputs_execute_all_algorithms_and_noise_devices()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    let cases = fixture["source_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?;
    let mut selected = cases
        .iter()
        .filter(|case| case["case"] == "defaults")
        .cloned()
        .collect::<Vec<_>>();
    for name in [
        "SamplerDPMPP_2M_SDE",
        "SamplerDPMPP_3M_SDE",
        "SamplerDPMPP_SDE",
    ] {
        let mut case = cases
            .iter()
            .find(|case| case["class_type"] == name && case["case"] == "defaults")
            .ok_or("default absent")?
            .clone();
        case["inputs"]["noise_device"] = serde_json::json!("cpu");
        selected.push(case);
    }
    for method in ["bislerp", "nearest-exact", "bilinear", "area", "bicubic"] {
        let mut case = cases
            .iter()
            .find(|case| case["class_type"] == "SamplerLCMUpscale" && case["case"] == "defaults")
            .ok_or("upscale default absent")?
            .clone();
        case["inputs"]["scale_ratio"] = serde_json::json!(2.0);
        case["inputs"]["upscale_method"] = serde_json::json!(method);
        selected.push(case);
    }
    for case in selected {
        let sampler = engine_sampler(&case, backend.clone(), &authority)?;
        let initial_values = [1.0, -0.5, 0.25, 0.75];
        let initial = tensor_from_f32(&backend, &[1, 1, 2, 2], &initial_values, &execution)?;
        let mut denoiser = AffineDenoiser {
            backend: &backend,
            execution: &execution,
            calls: 0,
            cfg_calls: 0,
        };
        let mut callbacks = 0;
        let sampled = sampler.execute_checked(
            &backend,
            SamplingPlan::new(
                sampler.identity().as_str(),
                NORMAL_SCHEDULER_ID,
                SamplingProfileIdentity::sd15(),
                7,
                3,
                1.0,
                1.0,
            )?,
            &DiscreteSamplingProfile::sd15()?,
            initial.clone(),
            &[2.0, 1.0, 0.5, 0.0],
            CompatibilityNoiseRequest::new(
                "task647",
                "attempt",
                "sampler",
                0,
                0,
                0,
                0,
                RetryRngPolicy::Replay,
            ),
            &execution,
            &mut denoiser,
            &mut |_, _, _| {
                callbacks += 1;
                Ok(())
            },
        )?;
        assert!(denoiser.calls > 0 && callbacks > 0, "{case}");
        assert_eq!(
            denoiser.cfg_calls > 0,
            case["class_type"] == "SamplerEulerAncestralCFGPP"
        );
        let output = sampled.final_tensor().ok_or("sampler output absent")?;
        assert!(
            tensor_to_f32(&backend, output, &execution)?
                .iter()
                .all(|value| value.is_finite()),
            "{case}"
        );
        let dimension = if case["inputs"]["scale_ratio"] == 2.0 {
            4
        } else {
            2
        };
        assert_eq!(output.descriptor().shape(), &[1, 1, dimension, dimension]);
        assert_eq!(
            &*tensor_to_f32(&backend, &initial, &execution)?,
            &initial_values
        );
    }
    Ok(())
}
#[test]
fn task647_adaptive_zero_boundaries_match_source_or_fail_before_denoising()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let source: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-sampling-comfy-node-0571/adaptive-zero-reference.json"
    ))?;
    let default = fixture["source_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .iter()
        .find(|case| case["class_type"] == "SamplerDPMAdaptative" && case["case"] == "defaults")
        .ok_or("adaptive default absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    for reference in source["cases"].as_array().ok_or("adaptive cases absent")? {
        let mut case = default.clone();
        for (name, value) in reference["parameters"]
            .as_object()
            .ok_or("parameters absent")?
        {
            case["inputs"][name] = value.clone();
        }
        let sampler = engine_sampler(&case, backend.clone(), &authority)?;
        let mut denoiser = AffineDenoiser {
            backend: &backend,
            execution: &execution,
            calls: 0,
            cfg_calls: 0,
        };
        let mut callbacks = 0;
        let result = sampler.execute_checked(
            &backend,
            SamplingPlan::new(
                sampler.identity().as_str(),
                NORMAL_SCHEDULER_ID,
                SamplingProfileIdentity::sd15(),
                7,
                1,
                1.0,
                1.0,
            )?,
            &DiscreteSamplingProfile::sd15()?,
            tensor_from_f32(&backend, &[2], &[1.0, -0.5], &execution)?,
            &[2.0, 0.5],
            CompatibilityNoiseRequest::new(
                "task647",
                "attempt",
                "adaptive",
                0,
                0,
                0,
                0,
                RetryRngPolicy::Replay,
            ),
            &execution,
            &mut denoiser,
            &mut |_, _, _| {
                callbacks += 1;
                Ok(())
            },
        );
        if reference["status"] == "complete" {
            let sampled = result?;
            let output = tensor_to_f32(
                &backend,
                sampled.final_tensor().ok_or("adaptive output absent")?,
                &execution,
            )?;
            for (actual, expected) in output.iter().zip(
                reference["output"]
                    .as_array()
                    .ok_or("source output absent")?,
            ) {
                close(
                    f64::from(*actual),
                    expected.as_f64().ok_or("source number absent")?,
                );
            }
            assert_eq!(Some(callbacks), reference["info"]["steps"].as_u64());
            assert_eq!(
                Some(denoiser.calls as u64),
                reference["info"]["nfe"].as_u64()
            );
        } else {
            assert!(matches!(
                result,
                Err(sim_sampler::NativeSamplerPayloadError::Algorithm { .. })
            ));
            assert_eq!(callbacks, 0);
            assert_eq!(denoiser.calls, 0);
        }
    }
    Ok(())
}
