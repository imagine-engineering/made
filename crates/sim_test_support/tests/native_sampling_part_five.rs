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
const FIXTURE: &str = include_str!("../fixtures/nodes/model-sampling-comfy-node-0582/fixture.json");
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
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x64803), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x64801)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn base_model(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
) -> Result<sim_sampler::NativeDiffusionPayload, Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    let bundle = sim_test_support::NativeDiffusionFixture::checked_in()
        .load_bundle_with_context(backend.clone(), &execution)?;
    let model = Arc::new(sim_model::NativeModelPayload::sd15_model(
        bundle.model().clone(),
    )?);
    let conditioning = Arc::new(sim_sampler::NativeConditioningPayload::checked_sd15(
        bundle.model_digest(),
        bundle.model(),
        bundle.conditioning().patch_graph().clone(),
        None,
    )?);
    Ok(sim_sampler::NativeDiffusionPayload::model(
        model,
        conditioning,
    )?)
}
fn case_inputs(
    case: &Value,
    node: &sim_nodes::NativeNodeDescriptor,
    base: &sim_sampler::NativeDiffusionPayload,
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    store: &dyn sim_nodes::NativeHandleStore,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let mut inputs = BTreeMap::new();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        cancellation,
    );
    for input in &node.inputs {
        let name = input.name.as_str();
        let value = if name == "model" {
            let mut model = base.clone();
            if let Some(shift) = case["inputs"]["model_shift"].as_f64() {
                model = model.with_sampling_policy(Arc::new(
                    sim_sampler::NativeSamplingPolicy::checked(
                        sim_sampler::NativeSamplingPolicyKind::Flow {
                            shift: shift as f32,
                            multiplier: 1000.0,
                            noise_scale: 1.0,
                        },
                    )?,
                ))?;
            }
            let payload = NativeStoredPayload::Model(Arc::new(
                sim_nodes::NativeStoredModelPayload::native_diffusion(Arc::new(model))?,
            ));
            NativeValue::Handle {
                value: store.publish(payload, cancellation)?,
            }
        } else if name == "sigmas" {
            let values = case["inputs"][name]
                .as_array()
                .ok_or("sigma fixture absent")?
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|value| value as f32)
                        .ok_or("invalid sigma")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tensor = tensor_from_f32(backend, &[values.len() as u64], &values, &execution)?;
            NativeValue::Handle {
                value: store.publish(
                    NativeStoredPayload::Tensor(Arc::new(
                        sim_tensor::NativeTensorPayload::from_tensor(
                            sim_tensor::NativeTensorRole::Sigmas,
                            tensor,
                        )?,
                    )),
                    cancellation,
                )?,
            }
        } else {
            let value = &case["inputs"][name];
            let primitive = if let Some(value) = value.as_bool() {
                NativePrimitive::Boolean(value)
            } else if let Some(value) = value.as_str() {
                NativePrimitive::String(value.to_owned())
            } else if matches!(
                name,
                "order" | "steps" | "step" | "predictor_order" | "corrector_order"
            ) {
                NativePrimitive::Integer(value.as_i64().ok_or("integer fixture absent")?)
            } else {
                NativePrimitive::Number(value.as_f64().ok_or("number fixture absent")?)
            };
            NativeValue::Primitive { value: primitive }
        };
        inputs.insert(name.to_owned(), value);
    }
    Ok(inputs)
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

#[test]
fn task648_sampling_nodes_match_pinned_source_and_recover() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let base = base_model(backend.clone(), &authority)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    for case in fixture["source_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
    {
        let name = case["class_type"].as_str().ok_or("class absent")?;
        let node = registry.node(name).ok_or("node absent")?;
        let descriptor = registry.descriptor(name).ok_or("descriptor absent")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64804)));
        let inputs = case_inputs(
            case,
            descriptor,
            &base,
            &backend,
            &authority,
            store.as_ref(),
            &cancellation,
        )?;
        let serialized = serde_json::to_string(&inputs)?;
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        assert!(node.demanded_lazy_inputs(&invocation, &inputs)?.is_empty());
        let dependencies = node.cache_dependencies(&invocation, &inputs)?;
        let token = node.cache_change_token(&inputs)?;
        let result = smol::block_on(node.execute(invocation, inputs.clone()));
        if case.get("error").is_some() {
            assert!(result.is_err(), "{name}");
            assert_eq!(generation.len(), 1);
            continue;
        }
        let outputs = values(result?)?;
        let mut digests = Vec::new();
        for (index, output) in outputs.iter().enumerate() {
            match output {
                NativeValue::Primitive {
                    value: NativePrimitive::Number(value),
                } => {
                    close(
                        *value,
                        case["outputs"][index]
                            .as_f64()
                            .ok_or("expected number absent")?,
                    );
                    digests.push(format!("{value:?}"));
                }
                NativeValue::Handle { value: handle } => {
                    let payload = store.resolve(handle, handle.handle_type(), &cancellation)?;
                    digests.push(payload.digest_sha256());
                    match payload.as_ref() {
                        NativeStoredPayload::Tensor(sigmas) => {
                            let actual = tensor_to_f32(&backend, sigmas.tensor(), &execution)?;
                            let expected = case["outputs"][index]
                                .as_array()
                                .ok_or("expected sigmas absent")?;
                            assert_eq!(actual.len(), expected.len(), "{case}");
                            for (actual, expected) in actual.iter().zip(expected) {
                                close(
                                    f64::from(*actual),
                                    expected.as_f64().ok_or("expected sigma absent")?,
                                );
                            }
                        }
                        NativeStoredPayload::Sampler(sampler) => {
                            assert_eq!(
                                sampler.identity().as_str(),
                                case["outputs"][index]["algorithm"]
                                    .as_str()
                                    .ok_or("algorithm absent")?
                            );
                            match sampler.parameters() {
                                sim_sampler::NativeSamplerParameters::Lms { order } => {
                                    assert_eq!(
                                        u64::from(*order),
                                        case["inputs"]["order"].as_u64().ok_or("order absent")?
                                    )
                                }
                                sim_sampler::NativeSamplerParameters::Seeds2(options) => {
                                    close(
                                        f64::from(options.eta),
                                        case["inputs"]["eta"].as_f64().ok_or("eta absent")?,
                                    );
                                    close(
                                        f64::from(options.noise_scale),
                                        case["inputs"]["s_noise"].as_f64().ok_or("noise absent")?,
                                    );
                                    close(
                                        f64::from(options.intermediate_step_ratio),
                                        case["inputs"]["r"].as_f64().ok_or("ratio absent")?,
                                    );
                                    assert_eq!(
                                        options.solver_type,
                                        if case["inputs"]["solver_type"] == "phi_1" {
                                            sim_sampler::generated_seeds_2_sim_model_0199::Seeds2SolverType::Phi1
                                        } else {
                                            sim_sampler::generated_seeds_2_sim_model_0199::Seeds2SolverType::Phi2
                                        }
                                    );
                                }
                                sim_sampler::NativeSamplerParameters::SaSolverInterval {
                                    parameters,
                                    start_sigma,
                                    end_sigma,
                                } => {
                                    close(
                                        f64::from(*start_sigma),
                                        case["outputs"][index]["start_sigma"]
                                            .as_f64()
                                            .ok_or("start sigma absent")?,
                                    );
                                    close(
                                        f64::from(*end_sigma),
                                        case["outputs"][index]["end_sigma"]
                                            .as_f64()
                                            .ok_or("end sigma absent")?,
                                    );
                                    for pair in case["outputs"][index]["tau_samples"]
                                        .as_array()
                                        .ok_or("tau samples absent")?
                                    {
                                        let sigma = pair[0].as_f64().ok_or("sigma absent")? as f32;
                                        let actual = if sigma <= *start_sigma && sigma >= *end_sigma
                                        {
                                            parameters.eta
                                        } else {
                                            0.0
                                        };
                                        close(
                                            f64::from(actual),
                                            pair[1].as_f64().ok_or("tau absent")?,
                                        );
                                    }
                                }
                                _ => return Err("unexpected sampler parameters".into()),
                            }
                        }
                        NativeStoredPayload::Model(model) => {
                            let modified = model.diffusion().ok_or("modified diffusion absent")?;
                            verify_video_guidance(
                                case,
                                modified.guidance_modifiers(),
                                &backend,
                                &execution,
                            )?;
                        }
                        _ => return Err("unexpected source output kind".into()),
                    }
                }
                _ => return Err("unexpected source output".into()),
            }
        }
        let after = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        assert_eq!(node.cache_dependencies(&after, &inputs)?, dependencies);
        assert_eq!(
            node.cache_change_token(&serde_json::from_str(&serialized)?)?,
            token
        );
        let recovered_generation = generation.fresh_generation()?;
        let recovered_store = recovered_generation.handle_store_for_attempt(store.attempt_id());
        for input in inputs.values() {
            if let NativeValue::Handle { value } = input {
                assert!(
                    recovered_store
                        .resolve(value, value.handle_type(), &cancellation)
                        .is_err()
                );
            }
        }
        let recovered_inputs = case_inputs(
            case,
            descriptor,
            &base,
            &backend,
            &authority,
            recovered_store.as_ref(),
            &cancellation,
        )?;
        let recovered_outputs = values(smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                recovered_store.clone(),
                cancellation.clone(),
                LIMIT,
            )?,
            recovered_inputs,
        ))?)?;
        let recovered_digests = recovered_outputs
            .iter()
            .map(|value| match value {
                NativeValue::Handle { value } => Ok(recovered_store
                    .resolve(value, value.handle_type(), &cancellation)?
                    .digest_sha256()),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(value),
                } => Ok(format!("{value:?}")),
                _ => Err("unexpected recovered output".into()),
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        assert_eq!(digests, recovered_digests, "{name}");
    }
    Ok(())
}

fn verify_video_guidance(
    case: &Value,
    modifiers: &sim_sampler::NativeGuidanceModifierChain,
    backend: &CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<(), Box<dyn Error>> {
    use sim_model::conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningIdentity, ConditioningSet,
        ConditioningValue,
    };
    use sim_model::generated_native_diffusion::{
        sd15_latent_format_identity, sd15_model_family_identity,
    };
    use sim_sampler::{GuidanceBranch, GuidanceDenoiser, GuidanceError, GuidanceEvaluation};
    struct Predictions {
        conditional: Tensor,
        unconditional: Tensor,
    }
    impl GuidanceDenoiser for Predictions {
        fn evaluate_batch(
            &mut self,
            evaluations: &[GuidanceEvaluation],
            _: &sim_tensor::ExecutionContext<'_>,
        ) -> Result<Vec<Tensor>, GuidanceError> {
            Ok(evaluations
                .iter()
                .map(|evaluation| match evaluation.branch() {
                    GuidanceBranch::Conditional => self.conditional.clone(),
                    GuidanceBranch::Unconditional => self.unconditional.clone(),
                })
                .collect())
        }
    }
    let batch = case["inputs"]["batch"].as_u64().ok_or("batch absent")?;
    let array = |name: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        case["inputs"][name]
            .as_array()
            .ok_or("prediction absent")?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or_else(|| "prediction absent".into())
            })
            .collect()
    };
    let conditional = tensor_from_f32(
        backend,
        &[batch, 1, 1, 1],
        &array("conditional")?,
        execution,
    )?;
    let unconditional = tensor_from_f32(
        backend,
        &[batch, 1, 1, 1],
        &array("unconditional")?,
        execution,
    )?;
    let latent = tensor_from_f32(
        backend,
        &[batch, 1, 1, 1],
        &vec![0.0; batch as usize],
        execution,
    )?;
    let identity = ConditioningIdentity::new(
        "task648",
        sd15_model_family_identity()?,
        sd15_latent_format_identity()?,
    )?;
    let conditioning = ConditioningSet::checked(
        identity,
        vec![ConditioningEntry::checked(
            "text",
            ConditioningValue::cross_attention(tensor_from_f32(
                backend,
                &[1, 1, 1],
                &[0.0],
                execution,
            )?)?,
            ConditioningEntryOptions::default(),
        )?],
        execution.cancellation,
    )?;
    let scale = case["inputs"]["scale"].as_f64().ok_or("scale absent")? as f32;
    let result = sim_sampler::execute_guidance_with_attached_modifiers(
        backend,
        &latent,
        1.0,
        &DiscreteSamplingProfile::sd15()?,
        &SamplingPlan::new(
            "euler",
            NORMAL_SCHEDULER_ID,
            SamplingProfileIdentity::sd15(),
            7,
            1,
            scale,
            1.0,
        )?,
        &conditioning,
        &conditioning,
        sim_sampler::GuidanceOptions::default(),
        &mut Predictions {
            conditional,
            unconditional,
        },
        &sim_sampler::ExecutionHookGraph::checked(7, Vec::new())?,
        modifiers,
        7,
        execution,
    )?;
    let actual = tensor_to_f32(backend, result.guided(), execution)?;
    let expected = case["outputs"].as_array().ok_or("guidance absent")?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        close(
            f64::from(*actual),
            expected.as_f64().ok_or("guidance value absent")?,
        );
    }
    Ok(())
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
#[test]
fn task648_sampling_nodes_fail_atomically_at_source_boundaries() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let base = base_model(backend.clone(), &authority)?;
    for row in fixture["nodes"].as_array().ok_or("rows absent")? {
        let name = row["class_type"].as_str().ok_or("class absent")?;
        let case = fixture["source_reference"]["cases"]
            .as_array()
            .ok_or("cases absent")?
            .iter()
            .find(|case| case["class_type"] == name && case.get("error").is_none())
            .ok_or("successful source case absent")?;
        let descriptor = registry.descriptor(name).ok_or("descriptor absent")?;
        let node = registry.node(name).ok_or("node absent")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64805)));
        let cancellation = CancellationToken::default();
        let good = case_inputs(
            case,
            descriptor,
            &base,
            &backend,
            &authority,
            store.as_ref(),
            &cancellation,
        )?;
        let initial_count = generation.len();
        let mut invalid = Vec::new();
        let mut extra = good.clone();
        extra.insert(
            "extra".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(0),
            },
        );
        invalid.push(extra);
        for name in good.keys() {
            let mut missing = good.clone();
            missing.remove(name);
            invalid.push(missing);
            for bad in [
                NativeValue::List { values: Vec::new() },
                NativeValue::Primitive {
                    value: NativePrimitive::String("invalid".to_owned()),
                },
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(-1),
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
                cancellation.clone(),
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
            assert_eq!(generation.len(), initial_count, "{name}");
        }
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let error = smol::block_on(node.execute(
            context(backend.clone(), &authority, store.clone(), cancelled, LIMIT)?,
            good.clone(),
        ))
        .err()
        .ok_or("pre-cancel succeeded")?;
        assert_eq!(error.kind, sim_runtime::NativeNodeFailureKind::Interrupted);
        if name != "SamplingPercentToSigma" {
            for reject in [false, true] {
                for publication in 1..=descriptor.outputs.len() {
                    let cancellation = CancellationToken::default();
                    let boundary = Arc::new(PublicationBoundary {
                        inner: store.clone(),
                        remaining: std::sync::atomic::AtomicUsize::new(publication),
                        reject,
                    });
                    let error = smol::block_on(node.execute(
                        context(
                            backend.clone(),
                            &authority,
                            boundary,
                            cancellation.clone(),
                            LIMIT,
                        )?,
                        good.clone(),
                    ))
                    .err()
                    .ok_or("publication boundary succeeded")?;
                    assert_eq!(
                        error.kind,
                        if reject {
                            sim_runtime::NativeNodeFailureKind::Failure
                        } else {
                            sim_runtime::NativeNodeFailureKind::Interrupted
                        }
                    );
                    assert_eq!(
                        generation.len(),
                        initial_count,
                        "{name}, publication {publication}"
                    );
                }
            }
        }
        if name == "SetFirstSigma" {
            assert!(
                smol::block_on(node.execute(
                    context(
                        backend.clone(),
                        &authority,
                        store.clone(),
                        cancellation.clone(),
                        1
                    )?,
                    good.clone()
                ))
                .is_err(),
                "{name}"
            );
            assert_eq!(generation.len(), initial_count);
        }
        if matches!(name, "SetFirstSigma" | "SDTurboScheduler") {
            let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(1)?;
            assert!(
                smol::block_on(node.execute(
                    context(
                        Arc::new(limited_backend),
                        &limited_authority,
                        store.clone(),
                        cancellation.clone(),
                        1,
                    )?,
                    good.clone(),
                ))
                .is_err(),
                "{name} allocation limit"
            );
            assert_eq!(generation.len(), initial_count);
        }
        let handles = good
            .values()
            .filter_map(|value| {
                if let NativeValue::Handle { value } = value {
                    Some(value)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        for handle in handles {
            store.revoke(handle, &cancellation)?;
        }
        assert_eq!(generation.len(), 0);
        assert!(
            node.cache_dependencies(
                &context(
                    backend.clone(),
                    &authority,
                    store.clone(),
                    cancellation.clone(),
                    LIMIT
                )?,
                &good
            )
            .is_err()
                || initial_count == 0
        );
        drop(store);
    }
    Ok(())
}

#[test]
fn task648_sa_sampler_captures_its_input_model_interval_before_execution()
-> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("SamplerSASolver").ok_or("SA node absent")?;
    let descriptor = registry
        .descriptor("SamplerSASolver")
        .ok_or("SA descriptor absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let base = base_model(backend.clone(), &authority)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    let mut digests = Vec::new();
    for (shift, expected) in [(None, 0.78125_f64), (Some(3.0), 0.875)] {
        let mut case = serde_json::json!({"inputs":{"eta":1.0,"sde_start_percent":0.0,"sde_end_percent":1.0,"s_noise":0.0,"predictor_order":3,"corrector_order":4,"use_pece":false,"simple_order_2":false}});
        if let Some(shift) = shift {
            case["inputs"]["model_shift"] = serde_json::json!(shift);
        }
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64806)));
        let inputs = case_inputs(
            &case,
            descriptor,
            &base,
            &backend,
            &authority,
            store.as_ref(),
            &cancellation,
        )?;
        let outputs = values(smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                store.clone(),
                cancellation.clone(),
                LIMIT,
            )?,
            inputs,
        ))?)?;
        let Some(NativeValue::Handle { value: handle }) = outputs.first() else {
            return Err("SA sampler handle absent".into());
        };
        let payload = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Sampler(sampler) = payload.as_ref() else {
            return Err("SA sampler absent".into());
        };
        digests.push(payload.digest_sha256());
        let initial = tensor_from_f32(&backend, &[1], &[1.0], &execution)?;
        let denoised = tensor_from_f32(&backend, &[1], &[0.5], &execution)?;
        let mut denoiser = move |_: &Tensor, _: f32, _: usize| Ok(denoised.clone());
        let result = sampler.execute_checked(
            &backend,
            SamplingPlan::new(
                "sa_solver",
                NORMAL_SCHEDULER_ID,
                SamplingProfileIdentity::sd15(),
                7,
                1,
                1.0,
                1.0,
            )?,
            &DiscreteSamplingProfile::sd15()?,
            initial,
            &[4.0, 3.0],
            CompatibilityNoiseRequest::new(
                "task648",
                "attempt",
                "sampling",
                0,
                0,
                0,
                0,
                RetryRngPolicy::Replay,
            ),
            &execution,
            &mut denoiser,
            &mut |_, _, _| Ok(()),
        )?;
        let actual = tensor_to_f32(
            &backend,
            result.final_tensor().ok_or("SA output absent")?,
            &execution,
        )?;
        close(
            f64::from(*actual.first().ok_or("SA value absent")?),
            expected,
        );
    }
    assert_ne!(digests.first(), digests.last());
    Ok(())
}

#[test]
fn task648_video_guidance_preserves_pre_cfg_hooks_and_compiled_model_wrappers()
-> Result<(), Box<dyn Error>> {
    use sim_nodes::{NativeModelCompileBackend, NativeModelCompileRequest};
    use sim_sampler::{NativeGuidanceModifier, NativeGuidanceModifierChain};
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let apg = NativeGuidanceModifier::AdaptiveProjectedGuidance {
        eta: 1.0,
        norm_threshold: 0.0,
        momentum: 0.0,
    };
    let base = Arc::new(
        base_model(backend.clone(), &authority)?.with_guidance_modifiers(Arc::new(
            NativeGuidanceModifierChain::checked(vec![
                apg,
                NativeGuidanceModifier::CfgOverride { scale: 2.0 },
                NativeGuidanceModifier::VideoTriangleCfg { minimum_scale: 0.5 },
            ])?,
        ))?,
    );
    let registry = generated_native_node_registry_projection(None)?;
    for name in ["VideoLinearCFGGuidance", "VideoTriangleCFGGuidance"] {
        let generation = NativeHandleStoreGeneration::new()?;
        let attempt = AttemptId(Uuid::from_u128(0x64807));
        let store = generation.handle_store_for_attempt(attempt);
        let cancellation = CancellationToken::default();
        let node_id = NodeId(name.to_owned());
        let identity =
            NativeNodeServiceIdentity::checked(Uuid::from_u128(0x64808), attempt, node_id.clone())?;
        let scratch = authority.authorize_workspace(LIMIT)?;
        let compute = NativeNodeComputeSession::checked(
            identity.clone(),
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let service =
            sim_runtime::native_execution_controller::NativeRuntimeModelCompileService::new(
                identity, None,
            );
        let invocation = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x64809)),
            attempt,
            node_id,
            cancellation.clone(),
            scratch,
            store.clone(),
            NativeNodeServices::checked(None, None, Some(compute))?
                .with_model_compile(Arc::new(service))?,
        )?;
        let compiled = invocation.compile_native_model(&NativeModelCompileRequest::checked(
            base.clone(),
            NativeModelCompileBackend::Inductor,
        )?)?;
        let source = NativeStoredPayload::Model(Arc::new(
            sim_nodes::NativeStoredModelPayload::native_compiled_diffusion(
                compiled.model().clone(),
                compiled.compiled_model().clone(),
            )?,
        ));
        let source_digest = source.digest_sha256();
        let handle = store.publish(source, &cancellation)?;
        let inputs = BTreeMap::from([
            (
                "model".to_owned(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            ),
            (
                "min_cfg".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(1.0),
                },
            ),
        ]);
        let node = registry.node(name).ok_or("video guidance absent")?;
        let outputs = values(smol::block_on(node.execute(invocation, inputs))?)?;
        let Some(NativeValue::Handle { value: output }) = outputs.first() else {
            return Err("model output absent".into());
        };
        let payload = store.resolve(output, output.handle_type(), &cancellation)?;
        let NativeStoredPayload::Model(model) = payload.as_ref() else {
            return Err("modified model absent".into());
        };
        let wrapper = model.compiled_model().ok_or("compiled wrapper dropped")?;
        assert!(Arc::ptr_eq(wrapper, compiled.compiled_model()));
        assert_eq!(wrapper.policy_sha256(), compiled.policy_sha256());
        wrapper.validate()?;
        let diffusion = model.diffusion().ok_or("diffusion dropped")?;
        assert!(Arc::ptr_eq(diffusion.model_payload(), base.model_payload()));
        assert_eq!(
            diffusion.sampling_policy().semantic_digest_sha256(),
            base.sampling_policy().semantic_digest_sha256()
        );
        assert_eq!(
            diffusion.hook_graph().identity()?,
            base.hook_graph().identity()?
        );
        let expected = if name == "VideoLinearCFGGuidance" {
            NativeGuidanceModifier::VideoLinearCfg { minimum_scale: 1.0 }
        } else {
            NativeGuidanceModifier::VideoTriangleCfg { minimum_scale: 1.0 }
        };
        assert_eq!(diffusion.guidance_modifiers().modifiers(), &[apg, expected]);
        assert_ne!(payload.digest_sha256(), source_digest);
        assert_eq!(
            store
                .resolve(&handle, handle.handle_type(), &cancellation)?
                .digest_sha256(),
            source_digest
        );
    }
    Ok(())
}

#[test]
fn task648_engine_publishes_lms_and_both_executable_seeds_solvers() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    use sim_types::ProfileId;
    use std::collections::BTreeSet;
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let mut cases = fixture["sampler_consumption_reference"]["cases"]
        .as_array()
        .ok_or("sampler execution cases absent")?
        .clone();
    cases.push(serde_json::json!({"class_type":"SamplerLMS","inputs":{"order":4},"initial":[1.0],"sigmas":[2.0,1.0],"denoised":[[0.3]],"expected":[0.65]}));
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let base = base_model(backend.clone(), &authority)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    for case in cases {
        let name = case["class_type"].as_str().ok_or("sampler class absent")?;
        let executor = NativeImageExecutor::new_with_generated_registry(
            ProfileId(Uuid::from_u128(0x64810)),
            BTreeMap::new(),
            true,
            backend.clone(),
        )?;
        let attempt = AttemptId(Uuid::from_u128(0x64811));
        let store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(attempt);
        let mut descriptor = registry
            .descriptor(name)
            .ok_or("sampler descriptor absent")?
            .clone();
        let inputs = case_inputs(
            &case,
            &descriptor,
            &base,
            &backend,
            &authority,
            store.as_ref(),
            &cancellation,
        )?;
        descriptor.output_node = true;
        let node_id = NodeId("sampler".to_owned());
        let plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x64812)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                CompiledNode {
                    id: node_id.clone(),
                    class_type: name.to_owned(),
                    descriptor,
                    inputs: inputs
                        .into_iter()
                        .map(|(name, value)| (name, InputBinding::Literal { value }))
                        .collect(),
                    unknown: BTreeMap::new(),
                },
            )]),
            topological_order: vec![node_id.clone()],
            static_required_nodes: BTreeSet::from([node_id.clone()]),
            output_nodes: vec![node_id.clone()],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
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
            .and_then(|values| values.first())
        else {
            return Err("engine sampler output absent".into());
        };
        let payload = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Sampler(sampler) = payload.as_ref() else {
            return Err("executable sampler absent".into());
        };
        let predictions = case["denoised"]
            .as_array()
            .ok_or("denoiser values absent")?
            .iter()
            .map(|values| {
                let value = values[0].as_f64().ok_or("denoised scalar absent")? as f32;
                Ok(tensor_from_f32(&backend, &[1], &[value], &execution)?)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let mut predictions = predictions.into_iter();
        let mut denoiser = |_: &Tensor, _: f32, _: usize| {
            predictions
                .next()
                .ok_or_else(|| "extra denoiser call".to_owned())
        };
        let sampled = sampler.execute_checked(
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
            tensor_from_f32(&backend, &[1], &[1.0], &execution)?,
            &[2.0, 1.0],
            CompatibilityNoiseRequest::new(
                "task648",
                "attempt",
                "sampling",
                0,
                0,
                0,
                0,
                RetryRngPolicy::Replay,
            ),
            &execution,
            &mut denoiser,
            &mut |_, _, _| Ok(()),
        )?;
        assert!(predictions.next().is_none());
        let actual = tensor_to_f32(
            &backend,
            sampled.final_tensor().ok_or("sampled output absent")?,
            &execution,
        )?;
        close(
            f64::from(*actual.first().ok_or("sampled value absent")?),
            case["expected"][0]
                .as_f64()
                .ok_or("expected sample absent")?,
        );
    }
    Ok(())
}

#[test]
fn task648_source_percent_nodes_preserve_discrete_timeline_and_float_endpoints()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("SamplingPercentToSigma")
        .ok_or("percentage node absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let base = base_model(backend.clone(), &authority)?;
    let cancellation = CancellationToken::default();
    for case in fixture["discrete_percent_reference"]["cases"]
        .as_array()
        .ok_or("discrete reference absent")?
    {
        let profile = sim_sampler::NativeSamplingPolicy::checked(
            sim_sampler::NativeSamplingPolicyKind::Discrete {
                prediction: if case["profile"] == "lcm" {
                    sim_sampler::NativePredictionKind::Lcm
                } else {
                    sim_sampler::NativePredictionKind::Epsilon
                },
                zero_terminal_snr: false,
            },
        )?;
        let model = base.clone().with_sampling_policy(Arc::new(profile))?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64813)));
        let handle = store.publish(
            NativeStoredPayload::Model(Arc::new(
                sim_nodes::NativeStoredModelPayload::native_diffusion(Arc::new(model))?,
            )),
            &cancellation,
        )?;
        let percent = case["percent"].as_f64().ok_or("percentage absent")?;
        let actual = case["return_actual_sigma"]
            .as_bool()
            .ok_or("actual flag absent")?;
        let inputs = BTreeMap::from([
            ("model".to_owned(), NativeValue::Handle { value: handle }),
            (
                "sampling_percent".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(percent),
                },
            ),
            (
                "return_actual_sigma".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(actual),
                },
            ),
        ]);
        let output = values(smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                store.clone(),
                cancellation.clone(),
                LIMIT,
            )?,
            inputs,
        ))?)?;
        let Some(NativeValue::Primitive {
            value: NativePrimitive::Number(sigma),
        }) = output.first()
        else {
            return Err("source sigma scalar absent".into());
        };
        let expected = case["sigma"].as_f64().ok_or("source sigma absent")?;
        if percent == 0.0 && !actual {
            assert_eq!(*sigma, expected);
        } else {
            close(*sigma, expected);
        }
        assert_eq!(generation.len(), 1);
        drop(store);
        assert_eq!(generation.len(), 0);
    }
    Ok(())
}

#[derive(Debug)]
struct CancelOnResolve(Arc<dyn sim_nodes::NativeHandleStore>);
impl sim_nodes::NativeHandleStore for CancelOnResolve {
    fn identity(&self) -> sim_nodes::NativeHandleStoreIdentity {
        self.0.identity()
    }
    fn attempt_id(&self) -> AttemptId {
        self.0.attempt_id()
    }
    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected: &sim_nodes::NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<sim_nodes::NativeResolvedPayload, sim_nodes::NativeHandleStoreError> {
        let payload = self.0.resolve(handle, expected, cancellation)?;
        cancellation.cancel();
        Ok(payload)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, sim_nodes::NativeHandleStoreError> {
        self.0.publish(payload, cancellation)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), sim_nodes::NativeHandleStoreError> {
        self.0.revoke(handle, cancellation)
    }
}
#[test]
fn task648_cancellation_after_input_resolution_suppresses_all_outputs() -> Result<(), Box<dyn Error>>
{
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let base = base_model(backend.clone(), &authority)?;
    for row in fixture["nodes"].as_array().ok_or("rows absent")? {
        let name = row["class_type"].as_str().ok_or("class absent")?;
        if matches!(name, "SamplerLMS" | "SamplerSEEDS2") {
            continue;
        }
        let case = fixture["source_reference"]["cases"]
            .as_array()
            .ok_or("cases absent")?
            .iter()
            .find(|case| case["class_type"] == name && case.get("error").is_none())
            .ok_or("successful case absent")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64814)));
        let cancellation = CancellationToken::default();
        let inputs = case_inputs(
            case,
            registry.descriptor(name).ok_or("descriptor absent")?,
            &base,
            &backend,
            &authority,
            store.as_ref(),
            &cancellation,
        )?;
        let original = generation.len();
        let invocation = context(
            backend.clone(),
            &authority,
            Arc::new(CancelOnResolve(store.clone())),
            cancellation,
            LIMIT,
        )?;
        let error = smol::block_on(
            registry
                .node(name)
                .ok_or("node absent")?
                .execute(invocation, inputs),
        )
        .err()
        .ok_or("cancelled input resolution succeeded")?;
        assert_eq!(
            error.kind,
            sim_runtime::NativeNodeFailureKind::Interrupted,
            "{name}"
        );
        assert_eq!(generation.len(), original, "{name}");
        drop(store);
    }
    Ok(())
}
