use serde_json::Value;
use sim_nodes::{
    NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
};
use sim_runtime::{
    NativeHandleStoreGeneration, NativeNodeOutcome, NativeOpaqueHandle, NativePrimitive,
    NativeStoredPayload, NativeValue, generated_native_node_registry_projection,
};
use sim_sampler::{
    CompatibilityNoiseRequest, DiscreteSamplingProfile, NORMAL_SCHEDULER_ID,
    NativeSamplerAlgorithm, SamplingPlan, SamplingProfileIdentity,
};
use sim_tensor::generated_native_diffusion::{tensor_from_f32, tensor_to_f32};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, RetryRngPolicy, StreamId, Tensor,
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;
const LIMIT: u64 = 16 * 1024 * 1024;
const FIXTURE: &str = include_str!("../fixtures/nodes/model-sampling-comfy-node-0743/fixture.json");
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
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x64903), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x64901)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn inputs(case: &Value) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let mut inputs = BTreeMap::from([(
        "steps".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(case["steps"].as_u64().ok_or("steps absent")?),
        },
    )]);
    for name in ["beta_d", "beta_min", "eps_s"] {
        inputs.insert(
            name.to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(case[name].as_f64().ok_or("number absent")?),
            },
        );
    }
    Ok(inputs)
}
fn output(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        mut outputs,
        effects,
        ui,
    } = outcome
    else {
        return Err("values absent".into());
    };
    assert!(effects.is_empty() && ui.is_none());
    assert_eq!(outputs.len(), 1);
    let Some(NativeValue::Handle { value }) = outputs.pop() else {
        return Err("handle absent".into());
    };
    Ok(value)
}

#[test]
fn task649_vp_node_matches_source_and_reconstructs_exact_payloads() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("VPScheduler").ok_or("VP node absent")?;
    for case in fixture["vp_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
    {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64902)));
        let inputs = inputs(case)?;
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        assert!(node.demanded_lazy_inputs(&invocation, &inputs)?.is_empty());
        assert!(
            node.cache_dependencies(&invocation, &inputs)?
                .artifact_digests
                .is_empty()
        );
        let token = node.cache_change_token(&inputs)?;
        let handle = output(smol::block_on(node.execute(invocation, inputs.clone()))?)?;
        let payload = store.resolve(&handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Tensor(sigmas) = payload.as_ref() else {
            return Err("sigmas absent".into());
        };
        assert_eq!(
            sigmas.tensor().descriptor().shape(),
            &[case["steps"].as_u64().ok_or("steps absent")? + 1]
        );
        for (index, expected) in case["sigmas"]
            .as_array()
            .ok_or("reference absent")?
            .iter()
            .enumerate()
        {
            let actual = f32::from_le_bytes(
                sigmas
                    .tensor()
                    .element_bytes(&[u64::try_from(index)?])?
                    .try_into()?,
            );
            if expected.as_str() == Some("+infinity") {
                assert_eq!(actual, f32::INFINITY);
            } else {
                let expected = expected.as_f64().ok_or("expected value absent")?;
                assert!(
                    (f64::from(actual) - expected).abs() <= 5e-8 + expected.abs() * 2e-6,
                    "index {index}: {actual} versus {expected}"
                );
            }
        }
        let recovered = generation.fresh_generation()?;
        let recovered_store = recovered.handle_store_for_attempt(store.attempt_id());
        assert!(
            recovered_store
                .resolve(&handle, handle.handle_type(), &cancellation)
                .is_err()
        );
        let restored: BTreeMap<String, NativeValue> =
            serde_json::from_slice(&serde_json::to_vec(&inputs)?)?;
        assert_eq!(node.cache_change_token(&restored)?, token);
        let restored_handle = output(smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                recovered_store.clone(),
                cancellation.clone(),
                LIMIT,
            )?,
            restored,
        ))?)?;
        assert_eq!(
            recovered_store
                .resolve(
                    &restored_handle,
                    restored_handle.handle_type(),
                    &cancellation
                )?
                .digest_sha256(),
            payload.digest_sha256()
        );
        store.revoke(&handle, &cancellation)?;
        assert!(
            store
                .resolve(&handle, handle.handle_type(), &cancellation)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn task649_void_node_publishes_the_executable_alpha_space_sampler() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64904)));
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("VOIDSampler").ok_or("VOID node absent")?;
    let values = BTreeMap::new();
    let invocation = context(
        backend.clone(),
        &authority,
        store.clone(),
        cancellation.clone(),
        LIMIT,
    )?;
    assert!(node.demanded_lazy_inputs(&invocation, &values)?.is_empty());
    let handle = output(smol::block_on(node.execute(invocation, values))?)?;
    let payload = store.resolve(&handle, handle.handle_type(), &cancellation)?;
    let NativeStoredPayload::Sampler(sampler) = payload.as_ref() else {
        return Err("sampler absent".into());
    };
    assert_eq!(sampler.algorithm(), NativeSamplerAlgorithm::VoidDdim);
    assert_eq!(sampler.identity().as_str(), "void_ddim");
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    let initial = tensor_from_f32(&backend, &[1], &[2.0], &execution)?;
    let denoised = tensor_from_f32(&backend, &[1], &[0.5], &execution)?;
    let mut denoiser = move |_input: &Tensor, _sigma: f32, _step: usize| Ok(denoised.clone());
    let result = sampler.execute_checked(
        &backend,
        SamplingPlan::new(
            "void_ddim",
            NORMAL_SCHEDULER_ID,
            SamplingProfileIdentity::sd15(),
            7,
            1,
            1.0,
            1.0,
        )?,
        &DiscreteSamplingProfile::sd15()?,
        initial,
        &[2.0, 1.0],
        CompatibilityNoiseRequest::new(
            "task649",
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
    let values = tensor_to_f32(
        &backend,
        result.final_tensor().ok_or("final sample absent")?,
        &execution,
    )?;
    let actual = *values.first().ok_or("sample absent")?;
    let alpha = 0.2_f32;
    let expected =
        0.5_f32.sqrt() * 0.5 + 0.5_f32.sqrt() * ((2.0 - alpha.sqrt() * 0.5) / (1.0 - alpha).sqrt());
    assert!((actual - expected).abs() < 1e-6);
    assert!((actual - 1.25).abs() > 0.1);
    let recovered = generation.fresh_generation()?;
    let recovered_store = recovered.handle_store_for_attempt(store.attempt_id());
    assert!(
        recovered_store
            .resolve(&handle, handle.handle_type(), &cancellation)
            .is_err()
    );
    let restored: BTreeMap<String, NativeValue> = serde_json::from_str("{}")?;
    let restored_handle = output(smol::block_on(node.execute(
        context(
            backend,
            &authority,
            recovered_store.clone(),
            cancellation.clone(),
            LIMIT,
        )?,
        restored,
    ))?)?;
    assert_eq!(
        recovered_store
            .resolve(
                &restored_handle,
                restored_handle.handle_type(),
                &cancellation
            )?
            .digest_sha256(),
        payload.digest_sha256()
    );
    Ok(())
}

#[derive(Debug)]
struct CancelOnPublication {
    inner: Arc<dyn sim_nodes::NativeHandleStore>,
    remaining: std::sync::atomic::AtomicUsize,
}

impl sim_nodes::NativeHandleStore for CancelOnPublication {
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
        let output = self.inner.publish(payload, cancellation)?;
        if self
            .remaining
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            == 1
        {
            cancellation.cancel();
        }
        Ok(output)
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
fn task649_sampling_nodes_reject_invalid_inputs_and_publish_atomically()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let case = fixture["vp_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .first()
        .ok_or("case absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for class_type in ["VOIDSampler", "VPScheduler"] {
        let values = if class_type == "VOIDSampler" {
            BTreeMap::new()
        } else {
            inputs(case)?
        };
        let node = registry.node(class_type).ok_or("node absent")?;
        for boundary in [
            "pre-cancel",
            "post-publish-cancel",
            "store-capacity",
            "extra-input",
        ] {
            let cancellation = CancellationToken::default();
            let generation = NativeHandleStoreGeneration::with_capacities(
                8,
                if boundary == "store-capacity" {
                    1
                } else {
                    usize::try_from(LIMIT)?
                },
            )?;
            let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64905)));
            let invocation_store: Arc<dyn sim_nodes::NativeHandleStore> =
                if boundary == "post-publish-cancel" {
                    Arc::new(CancelOnPublication {
                        inner: store.clone(),
                        remaining: std::sync::atomic::AtomicUsize::new(1),
                    })
                } else {
                    store.clone()
                };
            let mut values = values.clone();
            if boundary == "pre-cancel" {
                cancellation.cancel();
            }
            if boundary == "extra-input" {
                values.insert(
                    "extra".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(1),
                    },
                );
            }
            let result = smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    invocation_store,
                    cancellation.clone(),
                    LIMIT,
                )?,
                values,
            ));
            let error = result.err().ok_or("boundary unexpectedly succeeded")?;
            assert_eq!(
                error.kind,
                if cancellation.is_cancelled() {
                    sim_runtime::NativeNodeFailureKind::Interrupted
                } else {
                    sim_runtime::NativeNodeFailureKind::Failure
                }
            );
            assert_eq!(generation.len(), 0, "{class_type}: {boundary}");
            drop(store);
        }
    }
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64906)));
    let cancellation = CancellationToken::default();
    let node = registry.node("VPScheduler").ok_or("VP node absent")?;
    let good = inputs(case)?;
    let mut cases = Vec::new();
    for name in ["steps", "beta_d", "beta_min", "eps_s"] {
        let mut missing = good.clone();
        missing.remove(name);
        cases.push(missing);
        for bad in [
            NativeValue::List { values: Vec::new() },
            NativeValue::Primitive {
                value: NativePrimitive::Integer(-1),
            },
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(10001),
            },
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(false),
            },
        ] {
            let mut invalid = good.clone();
            invalid.insert(name.to_owned(), bad);
            cases.push(invalid);
        }
    }
    let mut zero_steps = good.clone();
    zero_steps.insert(
        "steps".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(0),
        },
    );
    cases.push(zero_steps);
    for values in cases {
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        assert!(node.cache_change_token(&values).is_err());
        assert!(node.demanded_lazy_inputs(&invocation, &values).is_err());
        assert!(smol::block_on(node.execute(invocation, values)).is_err());
        assert_eq!(generation.len(), 0);
    }
    assert!(
        smol::block_on(node.execute(
            context(backend, &authority, store.clone(), cancellation, 1)?,
            good
        ))
        .is_err()
    );
    assert_eq!(generation.len(), 0);
    drop(store);
    Ok(())
}

#[test]
fn task649_sampling_nodes_execute_through_native_engine() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    use sim_types::ProfileId;
    use std::collections::BTreeSet;
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let case = fixture["vp_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .first()
        .ok_or("case absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for class_type in ["VOIDSampler", "VPScheduler"] {
        let executor = NativeImageExecutor::new_with_generated_registry(
            ProfileId(Uuid::from_u128(0x64907)),
            BTreeMap::new(),
            true,
            backend.clone(),
        )?;
        let attempt = AttemptId(Uuid::from_u128(0x64908));
        let store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(attempt);
        let mut descriptor = registry
            .descriptor(class_type)
            .ok_or("descriptor absent")?
            .clone();
        descriptor.output_node = true;
        let node_id = NodeId("sampling".to_owned());
        let values = if class_type == "VOIDSampler" {
            BTreeMap::new()
        } else {
            inputs(case)?
        };
        let plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x64909)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                CompiledNode {
                    id: node_id.clone(),
                    class_type: class_type.to_owned(),
                    descriptor,
                    inputs: values
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
        let values = result
            .report
            .outputs
            .get(&node_id)
            .ok_or("engine outputs absent")?;
        assert_eq!(values.len(), 1);
        let Some(NativeValue::Handle { value: handle }) = values.first() else {
            return Err("engine handle absent".into());
        };
        let payload = store.resolve(handle, handle.handle_type(), &cancellation)?;
        match payload.as_ref() {
            NativeStoredPayload::Sampler(sampler) if class_type == "VOIDSampler" => {
                assert_eq!(sampler.algorithm(), NativeSamplerAlgorithm::VoidDdim)
            }
            NativeStoredPayload::Tensor(sigmas) if class_type == "VPScheduler" => {
                assert_eq!(sigmas.tensor().descriptor().shape(), &[21]);
                assert_eq!(
                    f32::from_le_bytes(sigmas.tensor().element_bytes(&[20])?.try_into()?),
                    0.0
                );
            }
            _ => return Err("incorrect engine payload".into()),
        }
    }
    Ok(())
}
