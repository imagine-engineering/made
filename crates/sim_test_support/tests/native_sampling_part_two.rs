use serde_json::Value;
use sim_nodes::{
    NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
};
use sim_runtime::{
    NativeHandleStoreGeneration, NativeNodeOutcome, NativeOpaqueHandle, NativePrimitive,
    NativeStoredPayload, NativeValue, generated_native_node_registry_projection,
};
use sim_sampler::{SchedulerError, SigmaInterpolationSpacing};
use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};
use sim_types::{AttemptId, NodeId, PromptId};
use std::error::Error;
use std::{collections::BTreeMap, sync::Arc};
use uuid::Uuid;

const SOURCE_SCHEDULES: &str =
    include_str!("../fixtures/nodes/model-sampling-comfy-node-0139/source_schedules.json");

fn integer(inputs: &Value, name: &str) -> Result<u64, Box<dyn Error>> {
    inputs
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("source integer {name} is absent").into())
}
fn number(inputs: &Value, name: &str) -> Result<f64, Box<dyn Error>> {
    inputs
        .get(name)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("source number {name} is absent").into())
}

#[test]
fn task645_canonical_schedules_match_pinned_pytorch_outputs() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(SOURCE_SCHEDULES)?;
    let cases = fixture
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("source cases absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let mut finite_cases = 0;
    let mut rejected_cases = 0;
    for (case_index, case) in cases.iter().enumerate() {
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4 * 1024 * 1024)?,
            &cancellation,
        );
        let inputs = case.get("inputs").ok_or("source inputs absent")?;
        let name = case
            .get("class_type")
            .and_then(Value::as_str)
            .ok_or("source node absent")?;
        let sigmas = inputs
            .get("sigmas")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|value| value as f32)
                            .ok_or("source sigma absent")
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        let output = match name {
            "KarrasScheduler" => sim_sampler::generated_karras_sim_model_0206::karras_sigmas(
                &backend,
                integer(inputs, "steps")?,
                number(inputs, "sigma_min")?,
                number(inputs, "sigma_max")?,
                number(inputs, "rho")?,
                &context,
            ),
            "ExponentialScheduler" => {
                sim_sampler::generated_exponential_sim_model_0205::exponential_sigmas(
                    &backend,
                    integer(inputs, "steps")?,
                    number(inputs, "sigma_min")?,
                    number(inputs, "sigma_max")?,
                    &context,
                )
            }
            "FlipSigmas" => sim_sampler::flip_sigmas(&backend, &sigmas, &context),
            "ExtendIntermediateSigmas" => sim_sampler::extend_intermediate_sigmas(
                &backend,
                &sigmas,
                integer(inputs, "steps")?,
                number(inputs, "start_at_sigma")?,
                number(inputs, "end_at_sigma")?,
                match inputs
                    .get("spacing")
                    .and_then(Value::as_str)
                    .ok_or("source spacing absent")?
                {
                    "linear" => SigmaInterpolationSpacing::Linear,
                    "cosine" => SigmaInterpolationSpacing::Cosine,
                    "sine" => SigmaInterpolationSpacing::Sine,
                    _ => return Err("unknown source spacing".into()),
                },
                &context,
            ),
            "Flux2Scheduler" => sim_sampler::flux2_sigmas(
                &backend,
                integer(inputs, "steps")?,
                integer(inputs, "width")?,
                integer(inputs, "height")?,
                &context,
            ),
            "Ideogram4Scheduler" => sim_sampler::ideogram4_sigmas(
                &backend,
                integer(inputs, "steps")?,
                integer(inputs, "width")?,
                integer(inputs, "height")?,
                number(inputs, "mu")?,
                number(inputs, "std")?,
                &context,
            ),
            "GITSScheduler" => sim_sampler::gits_sigmas(
                &backend,
                number(inputs, "coeff")?,
                integer(inputs, "steps")?,
                number(inputs, "denoise")?,
                &context,
            ),
            _ => return Err(format!("unassigned source node {name}").into()),
        };
        if case.get("source_error").is_some() {
            assert!(
                matches!(output, Err(SchedulerError::InvalidParameters(_))),
                "{name}/{case_index}: {output:?}"
            );
            rejected_cases += 1;
        } else if case.get("finite").and_then(Value::as_bool) == Some(false) {
            assert!(
                matches!(output, Err(SchedulerError::NonFiniteSigma { .. })),
                "{name}/{case_index}: {output:?}"
            );
            rejected_cases += 1;
        } else {
            let output = output.map_err(|error| format!("{name}/{case_index}: {error}"))?;
            let actual =
                sim_tensor::generated_native_diffusion::tensor_to_f32(&backend, &output, &context)?;
            let expected = case
                .get("f32_bits")
                .and_then(Value::as_array)
                .ok_or("source sigma bits absent")?;
            assert_eq!(actual.len(), expected.len(), "{name}/{case_index}");
            for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let expected =
                    f32::from_bits(expected.as_i64().ok_or("source bits absent")? as i32 as u32);
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
                    "{name}/{case_index}/{position}: {actual} versus {expected}"
                );
            }
            finite_cases += 1;
        }
    }
    assert_eq!((finite_cases, rejected_cases), (625, 60));
    Ok(())
}

fn context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn sim_nodes::NativeHandleStore>,
    cancellation: CancellationToken,
    scratch_bytes: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("sampling".to_owned());
    let scratch = authority.authorize_workspace(scratch_bytes)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x64503), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x64501)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn registered_inputs(
    case: &Value,
    context: &NativeNodeContext,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let mut result = BTreeMap::new();
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    for (name, value) in case
        .get("inputs")
        .and_then(Value::as_object)
        .ok_or("source inputs absent")?
    {
        let value = if name == "sigmas" {
            let values = value
                .as_array()
                .ok_or("source sigmas absent")?
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|value| value as f32)
                        .ok_or("source sigma absent")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tensor = sim_tensor::generated_native_diffusion::tensor_from_f32(
                compute.backend(),
                &[values.len() as u64],
                &values,
                &execution,
            )?;
            let payload = sim_tensor::NativeTensorPayload::from_tensor(
                sim_tensor::NativeTensorRole::Sigmas,
                tensor,
            )?;
            NativeValue::Handle {
                value: context.handle_store().publish(
                    NativeStoredPayload::Tensor(Arc::new(payload)),
                    &context.cancellation,
                )?,
            }
        } else {
            NativeValue::Primitive {
                value: if let Some(value) = value.as_str() {
                    NativePrimitive::String(value.to_owned())
                } else if matches!(name.as_str(), "steps" | "width" | "height") {
                    NativePrimitive::UnsignedInteger(value.as_u64().ok_or("integer absent")?)
                } else {
                    NativePrimitive::Number(value.as_f64().ok_or("number absent")?)
                },
            }
        };
        result.insert(name.clone(), value);
    }
    Ok(result)
}
fn output_handle(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        mut outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("ordinary output absent".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    assert_eq!(outputs.len(), 1);
    let Some(NativeValue::Handle { value }) = outputs.pop() else {
        return Err("sigma output handle absent".into());
    };
    Ok(value)
}
fn assert_source_output(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    case: &Value,
) -> Result<(), Box<dyn Error>> {
    let stored =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Tensor(payload) = stored.as_ref() else {
        return Err("canonical tensor payload absent".into());
    };
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let actual = sim_tensor::generated_native_diffusion::tensor_to_f32(
        compute.backend(),
        payload.tensor(),
        &execution,
    )?;
    let expected = case
        .get("f32_bits")
        .and_then(Value::as_array)
        .ok_or("source bits absent")?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let expected = f32::from_bits(expected.as_i64().ok_or("source bits absent")? as i32 as u32);
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
            "{actual} versus {expected}"
        );
    }
    Ok(())
}
#[test]
fn task645_registered_schedules_execute_source_cases_and_preserve_identity()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(SOURCE_SCHEDULES)?;
    let cases = fixture
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("source cases absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64510 + index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            4 * 1024 * 1024,
        )?;
        let inputs = registered_inputs(case, &context)?;
        let before = serde_json::to_vec(&inputs)?;
        let name = case
            .get("class_type")
            .and_then(Value::as_str)
            .ok_or("node absent")?;
        let node = registry.node(name).ok_or("registered node absent")?;
        let token = node.cache_change_token(&inputs)?;
        let recovered = serde_json::from_slice(&before)?;
        assert_eq!(token, node.cache_change_token(&recovered)?);
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        node.cache_dependencies(&context, &inputs)?;
        let initial = generation.len();
        let result = smol::block_on(node.execute(context.clone(), recovered));
        if case.get("source_error").is_some()
            || case.get("finite").and_then(Value::as_bool) == Some(false)
        {
            let error = result
                .err()
                .ok_or("source failure unexpectedly succeeded")?;
            assert_eq!(error.kind, sim_nodes::NativeNodeFailureKind::Failure);
            assert_eq!(generation.len(), initial);
        } else {
            let handle =
                output_handle(result.map_err(|error| format!("{name}/{index}: {error}"))?)?;
            assert_source_output(&context, &handle, case)?;
            if name == "FlipSigmas"
                && case["inputs"]["sigmas"]
                    .as_array()
                    .is_some_and(Vec::is_empty)
            {
                assert_eq!(
                    inputs.get("sigmas"),
                    Some(&NativeValue::Handle {
                        value: handle.clone()
                    })
                );
                assert_eq!(generation.len(), initial);
            } else {
                assert_eq!(generation.len(), initial + 1);
            }
            let retry = output_handle(smol::block_on(
                node.execute(context.clone(), inputs.clone()),
            )?)?;
            assert_eq!(retry.digest_sha256(), handle.digest_sha256());
            assert_source_output(&context, &retry, case)?;
        }
        assert_eq!(before, serde_json::to_vec(&inputs)?);
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
fn task645_registered_schedules_reject_invalid_inputs_and_rollback_failures()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(SOURCE_SCHEDULES)?;
    let cases = fixture
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("source cases absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for name in [
        "ExponentialScheduler",
        "ExtendIntermediateSigmas",
        "FlipSigmas",
        "Flux2Scheduler",
        "GITSScheduler",
        "Ideogram4Scheduler",
        "KarrasScheduler",
    ] {
        let case = cases
            .iter()
            .find(|case| {
                case.get("class_type").and_then(Value::as_str) == Some(name)
                    && case.get("finite").and_then(Value::as_bool) == Some(true)
                    && case
                        .get("f32_bits")
                        .and_then(Value::as_array)
                        .is_some_and(|values| values.len() > 1)
                    && case
                        .get("inputs")
                        .and_then(|inputs| inputs.get("sigmas"))
                        .and_then(Value::as_array)
                        .is_none_or(|values| values.len() > 1)
            })
            .ok_or("successful source case absent")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x64520)));
        let good_context = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
            4 * 1024 * 1024,
        )?;
        let good = registered_inputs(case, &good_context)?;
        let initial = generation.len();
        let node = registry.node(name).ok_or("registered node absent")?;
        let mut invalid_cases = vec![BTreeMap::new()];
        let mut extra = good.clone();
        extra.insert(
            "unknown".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(true),
            },
        );
        invalid_cases.push(extra);
        for key in good.keys() {
            let mut missing = good.clone();
            missing.remove(key);
            invalid_cases.push(missing);
            let mut wrong = good.clone();
            wrong.insert(
                key.clone(),
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(true),
                },
            );
            invalid_cases.push(wrong);
            let mut empty = good.clone();
            empty.insert(key.clone(), NativeValue::List { values: Vec::new() });
            invalid_cases.push(empty);
            let mut later = good.clone();
            later.insert(
                key.clone(),
                NativeValue::List {
                    values: vec![
                        good.get(key).ok_or("good input absent")?.clone(),
                        NativeValue::Primitive {
                            value: NativePrimitive::Boolean(true),
                        },
                    ],
                },
            );
            invalid_cases.push(later);
        }
        for bad in invalid_cases {
            assert!(node.cache_change_token(&bad).is_err(), "{name}");
            assert!(
                node.demanded_lazy_inputs(&good_context, &bad).is_err(),
                "{name}"
            );
            assert!(
                smol::block_on(node.execute(good_context.clone(), bad)).is_err(),
                "{name}"
            );
            assert_eq!(generation.len(), initial);
        }
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let cancelled = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation,
            4 * 1024 * 1024,
        )?;
        assert_eq!(
            node.demanded_lazy_inputs(&cancelled, &BTreeMap::new())
                .expect_err("pre-cancel admitted")
                .kind,
            sim_nodes::NativeNodeFailureKind::Interrupted
        );
        assert_eq!(
            smol::block_on(node.execute(cancelled, good.clone()))
                .expect_err("pre-cancel executed")
                .kind,
            sim_nodes::NativeNodeFailureKind::Interrupted
        );
        assert_eq!(generation.len(), initial);
        assert!(
            smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    store.clone(),
                    CancellationToken::default(),
                    0
                )?,
                good.clone()
            ))
            .is_err()
        );
        assert_eq!(generation.len(), initial);
        for reject in [false, true] {
            let boundary = Arc::new(PublicationBoundary {
                inner: store.clone(),
                remaining: std::sync::atomic::AtomicUsize::new(1),
                reject,
            });
            let failure = smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    boundary,
                    CancellationToken::default(),
                    4 * 1024 * 1024,
                )?,
                good.clone(),
            ))
            .expect_err("publication boundary succeeded");
            assert_eq!(
                failure.kind,
                if reject {
                    sim_nodes::NativeNodeFailureKind::Failure
                } else {
                    sim_nodes::NativeNodeFailureKind::Interrupted
                }
            );
            assert_eq!(generation.len(), initial, "{name}");
        }
        let output = output_handle(smol::block_on(
            node.execute(good_context.clone(), good.clone()),
        )?)?;
        assert_source_output(&good_context, &output, case)?;
        if let Some(NativeValue::Handle { value }) = good.get("sigmas") {
            store.revoke(value, &good_context.cancellation)?;
            let before = generation.len();
            assert!(node.cache_dependencies(&good_context, &good).is_err());
            assert!(smol::block_on(node.execute(good_context.clone(), good)).is_err());
            assert_eq!(generation.len(), before);
        }
    }
    Ok(())
}

#[test]
fn task645_engine_schedules_preserve_mapped_order_repeat_last_and_cache_reuse()
-> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    let fixture: Value = serde_json::from_str(SOURCE_SCHEDULES)?;
    let cases = fixture
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("source cases absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for name in [
        "ExponentialScheduler",
        "ExtendIntermediateSigmas",
        "FlipSigmas",
        "Flux2Scheduler",
        "GITSScheduler",
        "Ideogram4Scheduler",
        "KarrasScheduler",
    ] {
        let executor = NativeImageExecutor::new_with_generated_registry(
            sim_types::ProfileId(Uuid::from_u128(0x64530)),
            BTreeMap::new(),
            true,
            backend.clone(),
        )?;
        let attempt = AttemptId(Uuid::from_u128(0x64531));
        let store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(attempt);
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            4 * 1024 * 1024,
        )?;
        let mut selected = Vec::new();
        let mut rows = Vec::new();
        for row in 0..3 {
            let steps = if name == "GITSScheduler" {
                [2, 3, 20]
            } else {
                [1, 2, 7]
            }
            .get(row)
            .copied()
            .ok_or("mapped step absent")?;
            let later = row > 0;
            let expected_inputs = match name {
                "KarrasScheduler" => {
                    serde_json::json!({"steps":steps,"sigma_min":0.0291675,"sigma_max":14.614642,"rho":if later{1.0}else{7.0}})
                }
                "ExponentialScheduler" => {
                    serde_json::json!({"steps":steps,"sigma_min":if later{1.0}else{0.0291675},"sigma_max":if later{1.0}else{14.614642}})
                }
                "Flux2Scheduler" => {
                    serde_json::json!({"steps":steps,"width":if later{1024}else{16},"height":if later{1024}else{16}})
                }
                "Ideogram4Scheduler" => {
                    serde_json::json!({"steps":steps,"width":if later{1024}else{256},"height":if later{1024}else{256},"mu":0.0,"std":1.75})
                }
                "GITSScheduler" => {
                    serde_json::json!({"steps":steps,"coeff":if later{1.2}else{0.8},"denoise":1.0})
                }
                "ExtendIntermediateSigmas" => {
                    serde_json::json!({"steps":steps,"sigmas":if later{vec![0.0,2.0,1.0]}else{vec![10.0,7.0,2.0,0.0]},"start_at_sigma":-1.0,"end_at_sigma":0.0,"spacing":"linear"})
                }
                "FlipSigmas" => {
                    serde_json::json!({"sigmas":match row{0=>vec![],1=>vec![0.0],_=>vec![10.0,7.0,2.0,0.0]}})
                }
                _ => return Err("unassigned schedule".into()),
            };
            let case = cases
                .iter()
                .find(|case| {
                    case.get("class_type").and_then(Value::as_str) == Some(name)
                        && case.get("inputs") == Some(&expected_inputs)
                })
                .ok_or_else(|| {
                    format!("mapped source case absent: {name}/{row}/{expected_inputs}")
                })?;
            let mut inputs = registered_inputs(case, &context)?;
            if row == 2 && name == "ExtendIntermediateSigmas" {
                let previous: &BTreeMap<String, NativeValue> =
                    rows.last().ok_or("previous row absent")?;
                inputs.insert(
                    "sigmas".to_owned(),
                    previous
                        .get("sigmas")
                        .ok_or("previous sigma input absent")?
                        .clone(),
                );
            }
            selected.push(case);
            rows.push(inputs);
        }
        let first = rows.first().ok_or("first row absent")?;
        let mut inputs = BTreeMap::new();
        for key in first.keys() {
            let mut values = rows
                .iter()
                .map(|row| row.get(key).cloned().ok_or("mapped field absent"))
                .collect::<Result<Vec<_>, _>>()?;
            let value = if values.windows(2).all(|pair| pair.first() == pair.get(1)) {
                values.pop().ok_or("scalar field absent")?
            } else {
                if values.get(1) == values.get(2) {
                    values.pop();
                }
                NativeValue::List { values }
            };
            inputs.insert(key.clone(), InputBinding::Literal { value });
        }
        let node_id = NodeId("schedule".to_owned());
        let mut descriptor = registry
            .descriptor(name)
            .ok_or("descriptor absent")?
            .clone();
        descriptor.output_node = true;
        let plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x64532)),
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
                    inputs,
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
        let plan: CompiledPlan = serde_json::from_str(&serde_json::to_string(&plan)?)?;
        let result = executor.execute_blocking(
            &plan,
            attempt,
            CancellationToken::default(),
            0,
            authority.authorize_workspace(4 * 1024 * 1024)?,
        )?;
        assert_eq!(
            result.report.state,
            AttemptState::Succeeded,
            "{name}: {:?}",
            result.report.error
        );
        let Some(NativeValue::List { values }) = result
            .report
            .outputs
            .get(&node_id)
            .and_then(|outputs| outputs.first())
        else {
            return Err("mapped sigma outputs absent".into());
        };
        assert_eq!(values.len(), 3);
        for (value, case) in values.iter().zip(&selected) {
            let NativeValue::Handle { value } = value else {
                return Err("mapped sigma handle absent".into());
            };
            assert_source_output(&context, value, case)?;
        }
        if !matches!(name, "FlipSigmas" | "ExtendIntermediateSigmas") {
            let cached = executor.execute_blocking(
                &plan,
                AttemptId(Uuid::from_u128(0x64533)),
                CancellationToken::default(),
                0,
                authority.authorize_workspace(4 * 1024 * 1024)?,
            )?;
            assert_eq!(
                cached.report.state,
                AttemptState::Succeeded,
                "{name}: {:?}",
                cached.report.error
            );
            assert_eq!(cached.report.cache_hits, 1);
            assert_eq!(cached.report.outputs, result.report.outputs);
        }
    }
    Ok(())
}

#[test]
fn task645_ksampler_uses_selected_canonical_scheduler() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, InputBinding, NativeImageExecutor, compile_native_diffusion_workflow,
    };
    use sim_sampler::{DiscreteSamplingProfile, SchedulerRegistry, SchedulerRequest};
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let mut plan = compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &Default::default(),
        provider.clone(),
    )?;
    let sampler_id = NodeId("5".to_owned());
    plan.nodes
        .get_mut(&sampler_id)
        .ok_or("KSampler missing")?
        .inputs
        .insert(
            "scheduler".to_owned(),
            InputBinding::Literal {
                value: NativeValue::Primitive {
                    value: NativePrimitive::String("karras".to_owned()),
                },
            },
        );
    let memory_limit = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        sim_types::ProfileId(Uuid::from_u128(0x645_600)),
        BTreeMap::new(),
        true,
        backend.clone(),
        provider,
    )?;
    let result = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x645_601)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(memory_limit)?,
    )?;
    assert_eq!(
        result.report.state,
        AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(memory_limit)?,
        &cancellation,
    );
    let expected = sim_sampler::generated_karras_sim_model_0206::karras_schedule(
        &backend,
        &context,
        &SchedulerRegistry::foundational()?,
        &DiscreteSamplingProfile::sd15()?,
        &SchedulerRequest::new("karras", 4, 1.0)?,
    )?;
    let actual = result
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("KSampler UI missing")?
        .get("sigmas")
        .ok_or("KSampler sigmas missing")?;
    assert_eq!(actual, &serde_json::to_value(expected)?);
    Ok(())
}

#[test]
fn task645_step_windows_preserve_source_skip_and_leftover_noise_branches()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-sampling-comfy-node-0139/source_step_windows.json"
    ))?;
    let cases = fixture["cases"]
        .as_array()
        .ok_or("step-window cases missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1 << 20)?,
        &cancellation,
    );
    for (index, case) in cases.iter().enumerate() {
        let sigmas: Vec<f32> = serde_json::from_value(case["sigmas"].clone())?;
        let start = case["start_step"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?;
        let end = case["last_step"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?;
        let actual = sim_sampler::select_ksampler_step_window(
            sigmas,
            start,
            end,
            case["force_full_denoise"]
                .as_bool()
                .ok_or("force flag missing")?,
            &context,
        )?;
        if case["output"].as_str() == Some("latent") {
            assert!(actual.is_empty(), "source skipped window {index}");
        } else {
            let expected: Vec<f32> = serde_json::from_value(case["output"]["sigmas"].clone())?;
            assert_eq!(actual, expected, "source window {index}");
        }
    }
    assert_eq!(cases.len(), 150);
    cancellation.cancel();
    assert!(matches!(
        sim_sampler::select_ksampler_step_window(
            vec![1.0, 0.0],
            Some(10000),
            None,
            false,
            &context
        ),
        Err(SchedulerError::Cancelled)
    ));
    Ok(())
}

fn sampling_only_plan(
    mut plan: sim_runtime::CompiledPlan,
) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
    let sampler = NodeId("5".to_owned());
    plan.nodes.retain(|node, _| node.0 != "6" && node.0 != "7");
    plan.topological_order
        .retain(|node| plan.nodes.contains_key(node));
    plan.static_required_nodes
        .retain(|node| plan.nodes.contains_key(node));
    plan.nodes
        .get_mut(&sampler)
        .ok_or("sampler missing")?
        .descriptor
        .output_node = true;
    plan.output_nodes = vec![sampler];
    plan.provider_execution = None;
    plan.validate_integrity()?;
    Ok(plan)
}

#[test]
fn task645_advanced_sampler_runs_real_noise_windows_and_unchanged_latents()
-> Result<(), Box<dyn Error>> {
    use serde_json::json;
    use sim_runtime::{
        AttemptState, InputBinding, NativeImageExecutor, compile_native_diffusion_workflow,
    };
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let workflow_bytes = include_bytes!("../fixtures/native_diffusion/workflow.json");
    let baseline = sampling_only_plan(compile_native_diffusion_workflow(
        workflow_bytes,
        &Default::default(),
        provider.clone(),
    )?)?;
    let mut workflow: Value = serde_json::from_slice(workflow_bytes)?;
    let node = workflow["nodes"]
        .as_array_mut()
        .ok_or("workflow nodes missing")?
        .iter_mut()
        .find(|node| node["id"].as_str() == Some("5"))
        .ok_or("workflow sampler missing")?;
    node["type"] = json!("KSamplerAdvanced");
    node["properties"]["widget_input_names"] = json!([
        "add_noise",
        "noise_seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "start_at_step",
        "end_at_step",
        "return_with_leftover_noise"
    ]);
    node["widgets_values"] = json!([
        "enable",
        81985529216486895_u64,
        4,
        7.0,
        "euler",
        "normal",
        0,
        10000,
        "disable"
    ]);
    let advanced = sampling_only_plan(compile_native_diffusion_workflow(
        &serde_json::to_vec(&workflow)?,
        &Default::default(),
        provider.clone(),
    )?)?;
    let mut advanced: sim_runtime::CompiledPlan =
        serde_json::from_slice(&serde_json::to_vec(&advanced)?)?;
    advanced.prompt_id = baseline.prompt_id;
    let memory_limit = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        sim_types::ProfileId(Uuid::from_u128(0x645_700)),
        BTreeMap::new(),
        true,
        backend.clone(),
        provider,
    )?;
    let sampler_id = NodeId("5".to_owned());
    let run = |plan: &sim_runtime::CompiledPlan, index| -> Result<_, Box<dyn Error>> {
        let result = executor.execute_blocking(
            plan,
            AttemptId(Uuid::from_u128(0x645_710 + index)),
            CancellationToken::default(),
            0,
            authority.authorize_workspace(memory_limit)?,
        )?;
        assert_eq!(
            result.report.state,
            AttemptState::Succeeded,
            "case {index}: {:?}",
            result.report.error
        );
        Ok(result)
    };
    let baseline_result = run(&baseline, 0)?;
    let baseline_ui = baseline_result
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("baseline evidence missing")?;
    let full = run(&advanced, 1)?;
    let full_ui = full
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("advanced evidence missing")?;
    for key in [
        "sigmas",
        "noise_sha256",
        "initial_sha256",
        "denoiser_sha256",
        "latent_sha256",
    ] {
        assert_eq!(
            full_ui[key], baseline_ui[key],
            "full advanced sampling changed {key}"
        );
    }
    let full_sigmas: Vec<f32> = serde_json::from_value(full_ui["sigmas"].clone())?;
    let mut disabled_noise_hash = None;
    for (index, (start, end, leftover, noise)) in [
        (1_u64, 3_u64, "enable", "enable"),
        (1, 3, "disable", "enable"),
        (0, 10000, "disable", "disable"),
        (0, 10000, "disable", "disable"),
        (0, 0, "disable", "enable"),
        (3, 1, "enable", "enable"),
        (10000, 10000, "disable", "disable"),
    ]
    .into_iter()
    .enumerate()
    {
        let mut plan = advanced.clone();
        let node = plan
            .nodes
            .get_mut(&sampler_id)
            .ok_or("advanced sampler missing")?;
        for (name, value) in [
            ("start_at_step", json!(start)),
            ("end_at_step", json!(end)),
            ("return_with_leftover_noise", json!(leftover)),
            ("add_noise", json!(noise)),
            ("noise_seed", json!(index as u64 + 19)),
        ] {
            node.inputs.insert(
                name.to_owned(),
                InputBinding::Literal {
                    value: NativeValue::Primitive {
                        value: if let Some(value) = value.as_str() {
                            NativePrimitive::String(value.to_owned())
                        } else {
                            NativePrimitive::UnsignedInteger(
                                value.as_u64().ok_or("integer input missing")?,
                            )
                        },
                    },
                },
            );
        }
        plan.validate_integrity()?;
        let result = run(&plan, index as u128 + 2)?;
        let ui = result
            .report
            .ui_outputs
            .get(&sampler_id)
            .ok_or("window evidence missing")?;
        if start >= end.min(4) {
            assert_eq!(ui["sampling_skipped"], json!(true));
            assert_eq!(ui["denoiser_sha256"], json!([]));
        } else {
            let sigmas: Vec<f32> = serde_json::from_value(ui["sigmas"].clone())?;
            assert_eq!(sigmas.len(), (end.min(4) - start + 1) as usize);
            assert_eq!(sigmas.first(), full_sigmas.get(start as usize));
            assert_eq!(
                sigmas.last().copied(),
                if leftover == "disable" {
                    Some(0.0)
                } else {
                    full_sigmas.get(end as usize).copied()
                }
            );
            assert_eq!(
                ui["denoiser_sha256"]
                    .as_array()
                    .ok_or("denoiser trace missing")?
                    .len(),
                sigmas.len() - 1
            );
            if noise == "disable" {
                if let Some(previous) = &disabled_noise_hash {
                    assert_eq!(&ui["noise_sha256"], previous);
                }
                disabled_noise_hash = Some(ui["noise_sha256"].clone());
                assert_ne!(ui["noise_sha256"], full_ui["noise_sha256"]);
            }
        }
        let output = result
            .report
            .outputs
            .get(&sampler_id)
            .and_then(|outputs| outputs.first())
            .ok_or("advanced output missing")?;
        let NativeValue::Handle { value: handle } = output else {
            return Err("advanced LATENT handle missing".into());
        };
        let store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x645_710 + index as u128 + 2)));
        let cancellation = CancellationToken::default();
        let stored = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Latent(latent) = stored.as_ref() else {
            return Err("advanced latent bundle missing".into());
        };
        assert_eq!(latent.metadata().spatial_downscale_ratio(), None);
        assert_eq!(latent.metadata().temporal_downscale_ratio(), None);
        if start >= end.min(4) {
            let context = backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(memory_limit)?,
                &cancellation,
            );
            let samples = latent
                .samples()
                .tensor()
                .ok_or("advanced samples missing")?;
            assert!(
                sim_tensor::generated_native_diffusion::tensor_to_f32(&backend, samples, &context)?
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }
    let mut zero = baseline;
    zero.nodes
        .get_mut(&sampler_id)
        .ok_or("baseline sampler missing")?
        .inputs
        .insert(
            "denoise".to_owned(),
            InputBinding::Literal {
                value: NativeValue::Primitive {
                    value: NativePrimitive::Number(0.0),
                },
            },
        );
    let skipped = run(&zero, 20)?;
    assert_eq!(
        skipped
            .report
            .ui_outputs
            .get(&sampler_id)
            .ok_or("zero denoise evidence missing")?["sampling_skipped"],
        json!(true)
    );
    let cached = run(&advanced, 21)?;
    assert_eq!(cached.report.outputs, full.report.outputs);
    assert!(cached.report.cache_hits > 0);
    Ok(())
}

#[test]
fn task645_retained_sampling_rows_preserve_exact_source_schemas() -> Result<(), Box<dyn Error>> {
    let provider = Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let registry = generated_native_node_registry_projection(Some(provider))?;
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-sampling-comfy-node-0139/fixture.json"
    ))?;
    for name in ["KSampler", "KSamplerAdvanced", "DualModelGuider"] {
        let descriptor = registry
            .descriptor(name)
            .ok_or("retained descriptor missing")?;
        descriptor.validate_exact_schema_v2()?;
        assert_eq!(
            registry
                .descriptors()
                .filter(|(identifier, _)| *identifier == name)
                .count(),
            1
        );
        assert!(registry.descriptor_is_bound(name));
        assert!(descriptor.inputs.iter().all(|input| input.cardinality
            == sim_nodes::NativePortCardinality::Mapped
            && !input.lazy
            && !input.hidden));
        assert_eq!(
            descriptor.cache,
            sim_nodes::NativeCachePolicy::InputIdentity
        );
        assert_eq!(descriptor.effect, sim_nodes::NativeEffectClass::Pure);
        let row = fixture["nodes"]
            .as_array()
            .ok_or("source rows missing")?
            .iter()
            .find(|row| row["class_type"].as_str() == Some(name))
            .ok_or("source row missing")?;
        assert_eq!(
            serde_json::to_value(sim_nodes::built_in_source_schema(name)?)?,
            row["schema"]
        );
        assert_eq!(
            *descriptor,
            serde_json::from_slice(&serde_json::to_vec(descriptor)?)?
        );
    }
    Ok(())
}

#[test]
fn task645_dual_model_guider_preserves_optional_model_and_null_conditioning()
-> Result<(), Box<dyn Error>> {
    use sim_model::conditioning::{ConditioningConstant, ConditioningValue};
    use sim_runtime::{AttemptState, NativeImageExecutor, compile_native_diffusion_workflow};
    use sim_sampler::{NativeGuiderConditioningSets, NativeGuiderKind};
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let registry = generated_native_node_registry_projection(Some(provider.clone()))?;
    let mut template = sampling_only_plan(compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &Default::default(),
        provider.clone(),
    )?)?;
    template.nodes.remove(&NodeId("4".to_owned()));
    template
        .topological_order
        .retain(|node| template.nodes.contains_key(node));
    template
        .static_required_nodes
        .retain(|node| template.nodes.contains_key(node));
    let node_id = NodeId("5".to_owned());
    let node = template.nodes.get_mut(&node_id).ok_or("sampler missing")?;
    node.class_type = "DualModelGuider".to_owned();
    node.descriptor = registry
        .descriptor("DualModelGuider")
        .ok_or("guider descriptor missing")?
        .clone();
    node.descriptor.output_node = true;
    node.inputs
        .retain(|name, _| matches!(name.as_str(), "model" | "positive" | "negative" | "cfg"));
    let memory_limit = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        sim_types::ProfileId(Uuid::from_u128(0x645_800)),
        BTreeMap::new(),
        true,
        Arc::new(backend),
        provider,
    )?;
    let mut digests = std::collections::BTreeSet::new();
    for (index, (separate_model, supplied_negative)) in
        [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
    {
        let mut plan = template.clone();
        let node = plan.nodes.get_mut(&node_id).ok_or("guider missing")?;
        if separate_model {
            node.inputs.insert(
                "model_negative".to_owned(),
                node.inputs.get("model").ok_or("model missing")?.clone(),
            );
        }
        if !supplied_negative {
            node.inputs.remove("negative");
            let unused = NodeId("3".to_owned());
            plan.nodes.remove(&unused);
            plan.topological_order.retain(|node| node != &unused);
            plan.static_required_nodes.remove(&unused);
        }
        plan.validate_integrity()?;
        let plan: sim_runtime::CompiledPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
        let attempt = AttemptId(Uuid::from_u128(0x645_810 + index as u128));
        let result = executor.execute_blocking(
            &plan,
            attempt,
            CancellationToken::default(),
            0,
            authority.authorize_workspace(memory_limit)?,
        )?;
        assert_eq!(
            result.report.state,
            AttemptState::Succeeded,
            "case {index}: {:?}",
            result.report.error
        );
        let output = result
            .report
            .outputs
            .get(&node_id)
            .and_then(|values| values.first())
            .ok_or("guider output missing")?;
        let NativeValue::Handle { value: handle } = output else {
            return Err("GUIDER handle missing".into());
        };
        let store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(attempt);
        let cancellation = CancellationToken::default();
        let stored = store.resolve(handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Guider(payload) = stored.as_ref() else {
            return Err("canonical GUIDER missing".into());
        };
        payload.validate()?;
        let guider = payload.guider();
        assert_eq!(
            guider.kind(),
            if separate_model {
                NativeGuiderKind::DualModel
            } else {
                NativeGuiderKind::Cfg
            }
        );
        assert_eq!(guider.guidance(), 7.0);
        let negative = match guider.conditioning_sets() {
            NativeGuiderConditioningSets::Cfg { negative, .. }
            | NativeGuiderConditioningSets::DualModel { negative, .. } => negative,
            _ => return Err("wrong guider strategy".into()),
        };
        let entry = negative.entries().first().ok_or("negative entry missing")?;
        if supplied_negative {
            assert!(matches!(
                entry.value(),
                ConditioningValue::CrossAttention(_)
            ));
        } else {
            assert!(matches!(
                entry.value(),
                ConditioningValue::Constant(ConditioningConstant::Null)
            ));
        }
        assert!(digests.insert(guider.semantic_digest_sha256().to_owned()));
        if let NativeGuiderConditioningSets::DualModel {
            positive, negative, ..
        } = guider.conditioning_sets()
        {
            for distinct_wrapper in [false, true] {
                let negative_model = if distinct_wrapper {
                    Arc::new(guider.model().as_ref().clone())
                } else {
                    guider.model().clone()
                };
                let aliased = sim_sampler::NativeGuiderPayload::dual_model(
                    guider.model().clone(),
                    negative_model,
                    positive.clone(),
                    negative.clone(),
                    7.0,
                )?;
                let payload = NativeStoredPayload::Guider(Arc::new(
                    sim_nodes::NativeStoredGuiderPayload::checked(Arc::new(aliased), None)?,
                ));
                assert_eq!(
                    payload.residency()?.resident_bytes()?,
                    payload.resident_bytes()?
                );
                let handle = store.publish(payload, &cancellation)?;
                store.resolve(&handle, handle.handle_type(), &cancellation)?;
            }
        }
        let retry = executor.execute_blocking(
            &plan,
            AttemptId(Uuid::from_u128(0x645_820 + index as u128)),
            CancellationToken::default(),
            0,
            authority.authorize_workspace(memory_limit)?,
        )?;
        assert_eq!(
            retry.report.state,
            AttemptState::Succeeded,
            "{:?}",
            retry.report.error
        );
        assert_eq!(retry.report.outputs, result.report.outputs);
        assert!(retry.report.cache_hits > 0);
    }
    Ok(())
}

#[test]
fn task645_ksampler_dispatches_every_scheduler_and_penultimate_policy() -> Result<(), Box<dyn Error>>
{
    use sim_runtime::{
        AttemptState, InputBinding, NativeImageExecutor, compile_native_diffusion_workflow,
    };
    use sim_sampler::{DiscreteSamplingProfile, SamplingPlan, SamplingProfile};
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let template = sampling_only_plan(compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &Default::default(),
        provider.clone(),
    )?)?;
    let memory_limit = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        sim_types::ProfileId(Uuid::from_u128(0x645_900)),
        BTreeMap::new(),
        true,
        backend.clone(),
        provider,
    )?;
    let sampler_id = NodeId("5".to_owned());
    let mut cases = sim_sampler::SchedulerRegistry::foundational()?
        .definitions()
        .iter()
        .map(|definition| (definition.identity, "euler", 1.0_f32))
        .collect::<Vec<_>>();
    cases.extend([("normal", "dpm_2", 1.0), ("karras", "euler", 0.5)]);
    assert_eq!(cases.len(), 11);
    for (index, (scheduler, sampler, denoise)) in cases.into_iter().enumerate() {
        let mut plan = template.clone();
        let node = plan.nodes.get_mut(&sampler_id).ok_or("sampler missing")?;
        for (name, value) in [
            ("scheduler", NativePrimitive::String(scheduler.to_owned())),
            ("sampler_name", NativePrimitive::String(sampler.to_owned())),
            ("denoise", NativePrimitive::Number(f64::from(denoise))),
        ] {
            node.inputs.insert(
                name.to_owned(),
                InputBinding::Literal {
                    value: NativeValue::Primitive { value },
                },
            );
        }
        plan.validate_integrity()?;
        let result = executor.execute_blocking(
            &plan,
            AttemptId(Uuid::from_u128(0x645_910 + index as u128)),
            CancellationToken::default(),
            0,
            authority.authorize_workspace(memory_limit)?,
        )?;
        assert_eq!(
            result.report.state,
            AttemptState::Succeeded,
            "{scheduler}/{sampler}/{denoise}: {:?}",
            result.report.error
        );
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(memory_limit)?,
            &cancellation,
        );
        let profile = DiscreteSamplingProfile::sd15()?;
        let sampling = SamplingPlan::new(
            sampler,
            scheduler,
            profile.identity().clone(),
            81985529216486895,
            4,
            7.0,
            denoise,
        )?;
        let expected =
            sim_sampler::schedule_for_sampling_plan(&backend, &context, &profile, &sampling)?;
        let ui = result
            .report
            .ui_outputs
            .get(&sampler_id)
            .ok_or("sampler evidence missing")?;
        assert_eq!(
            ui["sigmas"],
            serde_json::to_value(expected)?,
            "{scheduler}/{sampler}/{denoise}"
        );
        assert_eq!(ui["sampler_algorithm"].as_str(), Some(sampler));
    }
    Ok(())
}

#[test]
fn task645_sampling_nodes_map_seed_lists_and_repeat_last_guidance() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, InputBinding, NativeImageExecutor, compile_native_diffusion_workflow,
    };
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let registry = generated_native_node_registry_projection(Some(provider.clone()))?;
    let template = sampling_only_plan(compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &Default::default(),
        provider.clone(),
    )?)?;
    let memory_limit = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        sim_types::ProfileId(Uuid::from_u128(0x645_a00)),
        BTreeMap::new(),
        true,
        Arc::new(backend),
        provider,
    )?;
    let node_id = NodeId("5".to_owned());
    let scalar = |value| NativeValue::Primitive { value };
    let binding = |value| InputBinding::Literal { value };
    for (ordinal, name) in ["KSampler", "KSamplerAdvanced"].into_iter().enumerate() {
        let mut plan = template.clone();
        let node = plan.nodes.get_mut(&node_id).ok_or("sampler missing")?;
        let seed_name = if name == "KSamplerAdvanced" {
            node.class_type = name.to_owned();
            node.descriptor = registry
                .descriptor(name)
                .ok_or("advanced descriptor missing")?
                .clone();
            node.descriptor.output_node = true;
            node.inputs.remove("seed");
            node.inputs.remove("denoise");
            for (key, value) in [
                ("add_noise", NativePrimitive::String("enable".to_owned())),
                ("start_at_step", NativePrimitive::UnsignedInteger(0)),
                ("end_at_step", NativePrimitive::UnsignedInteger(10000)),
                (
                    "return_with_leftover_noise",
                    NativePrimitive::String("disable".to_owned()),
                ),
            ] {
                node.inputs.insert(key.to_owned(), binding(scalar(value)));
            }
            "noise_seed"
        } else {
            "seed"
        };
        node.inputs.insert(
            "steps".to_owned(),
            binding(scalar(NativePrimitive::UnsignedInteger(1))),
        );
        node.inputs.insert(
            seed_name.to_owned(),
            binding(NativeValue::List {
                values: [19, 20, 21]
                    .into_iter()
                    .map(|seed| scalar(NativePrimitive::UnsignedInteger(seed)))
                    .collect(),
            }),
        );
        node.inputs.insert(
            "cfg".to_owned(),
            binding(NativeValue::List {
                values: [0.0, 7.0]
                    .into_iter()
                    .map(|cfg| scalar(NativePrimitive::Number(cfg)))
                    .collect(),
            }),
        );
        plan.validate_integrity()?;
        let execute = |plan: &sim_runtime::CompiledPlan, iteration| -> Result<_, Box<dyn Error>> {
            let result = executor.execute_blocking(
                plan,
                AttemptId(Uuid::from_u128(
                    0x645_a10 + ordinal as u128 * 10 + iteration,
                )),
                CancellationToken::default(),
                0,
                authority.authorize_workspace(memory_limit)?,
            )?;
            assert_eq!(
                result.report.state,
                AttemptState::Succeeded,
                "{name}/{iteration}: {:?}",
                result.report.error
            );
            Ok(result)
        };
        let mapped = execute(&plan, 0)?;
        let NativeValue::List { values } = mapped
            .report
            .outputs
            .get(&node_id)
            .and_then(|values| values.first())
            .ok_or("mapped outputs missing")?
        else {
            return Err("mapped LATENT list missing".into());
        };
        assert_eq!(values.len(), 3);
        let rows = mapped
            .report
            .ui_outputs
            .get(&node_id)
            .and_then(Value::as_array)
            .ok_or("mapped evidence missing")?;
        assert_eq!(rows.len(), 3);
        for (index, row) in rows.iter().enumerate() {
            let mut single = plan.clone();
            let node = single.nodes.get_mut(&node_id).ok_or("sampler missing")?;
            node.inputs.insert(
                seed_name.to_owned(),
                binding(scalar(NativePrimitive::UnsignedInteger(index as u64 + 19))),
            );
            node.inputs.insert(
                "cfg".to_owned(),
                binding(scalar(NativePrimitive::Number(if index == 0 {
                    0.0
                } else {
                    7.0
                }))),
            );
            let result = execute(&single, index as u128 + 1)?;
            let expected = result
                .report
                .ui_outputs
                .get(&node_id)
                .ok_or("scalar evidence missing")?;
            for key in ["noise_sha256", "sigmas", "latent_sha256"] {
                assert_eq!(row[key], expected[key], "{name}/{index}/{key}");
            }
        }
        let cached = execute(&plan, 5)?;
        assert_eq!(cached.report.outputs, mapped.report.outputs);
        assert!(cached.report.cache_hits > 0);
    }
    Ok(())
}

#[test]
fn task645_zero_denoise_normalizes_empty_latent_channels_and_spatial_hint()
-> Result<(), Box<dyn Error>> {
    assert_normalized_empty_latent_mask(0.0)
}

#[test]
fn task645_real_sampling_preserves_original_mask_after_empty_latent_normalization()
-> Result<(), Box<dyn Error>> {
    assert_normalized_empty_latent_mask(1.0)
}

fn assert_normalized_empty_latent_mask(denoise: f64) -> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, InputBinding, NativeImageExecutor, compile_native_diffusion_workflow,
    };
    use sim_tensor::{NativeLatentBundle, NativeLatentMetadata};
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let mut plan = sampling_only_plan(compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &Default::default(),
        provider.clone(),
    )?)?;
    let unused = NodeId("4".to_owned());
    plan.nodes.remove(&unused);
    plan.topological_order.retain(|node| node != &unused);
    plan.static_required_nodes.remove(&unused);
    let memory_limit = 2 * 1024 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
    let backend = Arc::new(backend);
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        sim_types::ProfileId(Uuid::from_u128(0x645_b00)),
        BTreeMap::new(),
        true,
        backend.clone(),
        provider,
    )?;
    let attempt = AttemptId(Uuid::from_u128(0x645_b01));
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(memory_limit)?,
        &cancellation,
    );
    let source_side = if denoise == 0.0 { 4_u64 } else { 2 };
    let source_values = usize::try_from(source_side * source_side)?;
    let samples = sim_tensor::generated_native_diffusion::tensor_from_f32(
        &backend,
        &[1, 1, source_side, source_side],
        &vec![0.0; source_values],
        &context,
    )?;
    let mask_value = if denoise == 0.0 { 0.5 } else { 0.0 };
    let original_mask = sim_tensor::generated_native_diffusion::tensor_from_f32(
        &backend,
        &[1, 1, source_side, source_side],
        &vec![mask_value; source_values],
        &context,
    )?;
    let latent = NativeLatentBundle::single(
        samples,
        Some(original_mask),
        None,
        NativeLatentMetadata::checked(None, None, Some(16), None)?,
        &context,
    )?;
    let store = executor
        .handle_store_generation_for_test()
        .handle_store_for_attempt(attempt);
    let input = store.publish(NativeStoredPayload::Latent(Arc::new(latent)), &cancellation)?;
    let node_id = NodeId("5".to_owned());
    let node = plan.nodes.get_mut(&node_id).ok_or("sampler missing")?;
    node.inputs.insert(
        "latent_image".to_owned(),
        InputBinding::Literal {
            value: NativeValue::Handle { value: input },
        },
    );
    node.inputs.insert(
        "denoise".to_owned(),
        InputBinding::Literal {
            value: NativeValue::Primitive {
                value: NativePrimitive::Number(denoise),
            },
        },
    );
    plan.validate_integrity()?;
    let result = executor.execute_blocking(
        &plan,
        attempt,
        cancellation.clone(),
        0,
        authority.authorize_workspace(memory_limit)?,
    )?;
    assert_eq!(
        result.report.state,
        AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    let NativeValue::Handle { value: output } = result
        .report
        .outputs
        .get(&node_id)
        .and_then(|values| values.first())
        .ok_or("latent output missing")?
    else {
        return Err("latent handle missing".into());
    };
    let stored = store.resolve(output, output.handle_type(), &cancellation)?;
    let NativeStoredPayload::Latent(latent) = stored.as_ref() else {
        return Err("canonical latent missing".into());
    };
    let samples = latent.samples().tensor().ok_or("latent samples missing")?;
    assert_eq!(
        samples.descriptor().shape(),
        &[1, 4, source_side * 2, source_side * 2]
    );
    let Some(sim_tensor::NativeLatentNoiseMask::Tensor(mask)) = latent.noise_mask() else {
        return Err("original noise mask missing".into());
    };
    assert_eq!(mask.descriptor().shape(), &[1, 1, source_side, source_side]);
    assert_eq!(
        sim_tensor::generated_native_diffusion::tensor_to_f32(&backend, mask, &context)?.as_ref(),
        &vec![mask_value; source_values]
    );

    assert_eq!(latent.metadata().spatial_downscale_ratio(), None);
    assert!(
        sim_tensor::generated_native_diffusion::tensor_to_f32(&backend, samples, &context)?
            .iter()
            .all(|value| if denoise == 0.0 {
                *value == 0.0
            } else {
                value.abs() <= 0.00001
            })
    );
    Ok(())
}

#[test]
fn task645_empty_latent_normalization_matches_pinned_source() -> Result<(), Box<dyn Error>> {
    use sim_tensor::generated_native_diffusion::{tensor_from_f32, tensor_to_f32};
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-sampling-comfy-node-0139/source_empty_latents.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("source cases missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    for (index, case) in cases.iter().enumerate() {
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32 * 1024 * 1024)?,
            &cancellation,
        );
        let name = case["format"].as_str().ok_or("format missing")?;
        let definition = sim_model::GENERATED_LATENT_FORMATS
            .iter()
            .find(|definition| definition.identifier == name)
            .ok_or("canonical latent format missing")?;
        let shape: Vec<u64> = serde_json::from_value(case["shape"].clone())?;
        let mut values = vec![0.0; usize::try_from(shape.iter().product::<u64>())?];
        if case["nonempty"] == true {
            *values.last_mut().ok_or("empty input")? = 1.25;
        }
        let input = tensor_from_f32(&backend, &shape, &values, &context)?;
        let output = sim_model::latent_format::normalize_sampling_latent(
            definition,
            &backend,
            &input,
            case["spatial"].as_u64().map(u32::try_from).transpose()?,
            case["temporal"].as_u64().map(u32::try_from).transpose()?,
            &context,
        );
        if case.get("error").is_some() {
            assert!(output.is_err(), "source case {index}: {case}");
            continue;
        }
        let output = output?;
        let expected_shape: Vec<u64> = serde_json::from_value(case["output_shape"].clone())?;
        assert_eq!(output.descriptor().shape(), expected_shape, "case {index}");
        let values = tensor_to_f32(&backend, &output, &context)?;
        assert_eq!(
            values.iter().map(|value| f64::from(*value)).sum::<f64>(),
            case["output_sum"].as_f64().ok_or("source sum missing")?,
            "case {index}"
        );
        if case["nonempty"] == true {
            assert_eq!(values.last(), Some(&1.25), "case {index}");
            assert!(values.iter().rev().skip(1).all(|value| *value == 0.0));
        } else {
            assert!(values.iter().all(|value| *value == 0.0), "case {index}");
        }
    }
    assert_eq!(cases.len(), 360);
    cancellation.cancel();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024)?,
        &cancellation,
    );
    assert!(context.check().is_err());
    Ok(())
}

#[test]
fn task734_latent_storage_preserves_original_mask_geometry() -> Result<(), Box<dyn Error>> {
    use sim_tensor::generated_native_diffusion::{tensor_from_f32, tensor_to_f32};
    use sim_tensor::{NativeLatentBundle, NativeLatentMetadata};
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-sampling-comfy-node-0139/source_masks.json"
    ))?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    for (index, case) in fixture["cases"]
        .as_array()
        .ok_or("mask cases missing")?
        .iter()
        .enumerate()
    {
        let shape: Vec<u64> = serde_json::from_value(case["output_shape"].clone())?;
        let mask_shape: Vec<u64> = serde_json::from_value(case["mask_shape"].clone())?;
        let mask_values: Vec<f32> = serde_json::from_value(case["input_values"].clone())?;
        let samples = tensor_from_f32(
            &backend,
            &shape,
            &vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
            &context,
        )?;
        let mask = tensor_from_f32(&backend, &mask_shape, &mask_values, &context)?;
        let bundle = NativeLatentBundle::single(
            samples,
            Some(mask),
            None,
            NativeLatentMetadata::default(),
            &context,
        )
        .map_err(|error| format!("source mask case {index}: {error}"))?;
        let expected: Vec<f32> = serde_json::from_value(case["prepared_values"].clone())?;
        let Some(sim_tensor::NativeLatentNoiseMask::Tensor(original_mask)) = bundle.noise_mask()
        else {
            return Err("original mask missing".into());
        };
        let prepared = sim_sampler::prepare_native_latent_mask(
            original_mask,
            bundle.samples().tensor().ok_or("samples missing")?,
            &backend,
            &context,
        )?;
        let expected_shape: Vec<u64> = serde_json::from_value(case["prepared_shape"].clone())?;
        assert_eq!(
            prepared.descriptor().shape(),
            expected_shape,
            "mask case {index}"
        );
        let values = tensor_to_f32(&backend, &prepared, &context)?;
        assert_eq!(values.len(), expected.len());
        for (position, (value, expected)) in values.iter().zip(&expected).enumerate() {
            assert!(
                (value - expected).abs() <= 0.000001,
                "mask case {index}/{position}: {value} != {expected}"
            );
        }

        let projection: sim_tensor::NativeLatentBundleProjection =
            serde_json::from_slice(&serde_json::to_vec(bundle.projection())?)?;
        assert_eq!(&projection, bundle.projection());
        let sampling_input = sim_sampler::NativeLatentSamplingInput::checked(&bundle, &context)?;
        let ones = tensor_from_f32(
            &backend,
            &shape,
            &vec![1.0; usize::try_from(shape.iter().product::<u64>())?],
            &context,
        )?;
        let denoised = sampling_input.masked_denoised(
            &sim_tensor::NativeLatentSamples::Tensor(ones.clone()),
            &backend,
            &context,
        )?;
        let denoised_values = tensor_to_f32(
            &backend,
            denoised.tensor().ok_or("denoised missing")?,
            &context,
        )?;
        for (value, expected) in denoised_values.iter().zip(&expected) {
            assert!(
                (value - expected).abs() <= 0.000001,
                "denoised mask case {index}"
            );
        }
        let zeros = tensor_from_f32(
            &backend,
            &shape,
            &vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
            &context,
        )?;
        let model_input = sampling_input.masked_model_input(
            &sim_tensor::NativeLatentSamples::Tensor(ones),
            &sim_sampler::NativeLatentNoiseGeneration::Tensor(sim_sampler::NativeNoiseGeneration {
                noise: zeros,
                before: None,
                after: None,
            }),
            1.0,
            &backend,
            &context,
        )?;
        let model_values = tensor_to_f32(
            &backend,
            model_input.tensor().ok_or("model input missing")?,
            &context,
        )?;
        for (value, expected) in model_values.iter().zip(&expected) {
            assert!(
                (value - expected).abs() <= 0.000001,
                "model input mask case {index}"
            );
        }
        bundle.validate(&context)?;
        bundle.validate_retained()?;
        let Some(sim_tensor::NativeLatentNoiseMask::Tensor(retained)) = bundle.noise_mask() else {
            return Err("mask missing".into());
        };
        assert_eq!(retained.descriptor().shape(), mask_shape);
        assert_eq!(
            tensor_to_f32(&backend, retained, &context)?.as_ref(),
            mask_values.as_slice()
        );
    }
    Ok(())
}

#[test]
fn task734_mask_and_latent_preparation_cancel_without_mutating_inputs() -> Result<(), Box<dyn Error>>
{
    use sim_tensor::generated_native_diffusion::{tensor_from_f32, tensor_to_f32};
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let samples = tensor_from_f32(&backend, &[1, 1, 2, 2], &[0.0; 4], &context)?;
    let mask = tensor_from_f32(&backend, &[1, 1, 2, 2], &[0.0, 0.25, 0.5, 1.0], &context)?;
    let definition = sim_model::GENERATED_LATENT_FORMATS
        .iter()
        .find(|definition| definition.identifier == "SD15")
        .ok_or("SD15 latent format missing")?;
    assert!(
        sim_model::latent_format::normalize_sampling_latent(
            definition,
            &backend,
            &samples,
            Some(u32::MAX),
            None,
            &context
        )
        .is_err()
    );
    cancellation.cancel();
    assert!(sim_sampler::prepare_native_latent_mask(&mask, &samples, &backend, &context).is_err());
    assert!(
        sim_model::latent_format::normalize_sampling_latent(
            definition,
            &backend,
            &samples,
            Some(16),
            None,
            &context
        )
        .is_err()
    );
    let retry_cancellation = CancellationToken::default();
    let retry = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &retry_cancellation,
    );
    assert_eq!(
        tensor_to_f32(&backend, &samples, &retry)?.as_ref(),
        &[0.0; 4]
    );
    assert_eq!(
        tensor_to_f32(&backend, &mask, &retry)?.as_ref(),
        &[0.0, 0.25, 0.5, 1.0]
    );
    let output = sim_model::latent_format::normalize_sampling_latent(
        definition,
        &backend,
        &samples,
        Some(16),
        None,
        &retry,
    )?;
    assert_eq!(output.descriptor().shape(), &[1, 4, 4, 4]);
    assert!(sim_sampler::prepare_native_latent_mask(&mask, &output, &backend, &retry).is_ok());
    Ok(())
}
