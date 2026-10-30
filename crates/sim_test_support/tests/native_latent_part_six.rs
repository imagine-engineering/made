use serde_json::Value;
use sim_nodes::{
    NativeHandleStore, NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle,
    NativePrimitive, NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, NativeLatentBundle, NativeLatentMetadata,
    NativeLatentSamples, NativeTensorPayload, NativeTensorRole, StreamId,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;
type TestResult = Result<(), Box<dyn Error>>;
fn fixture() -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0597/source-contract.json"
    ))?)
}
fn context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("latent-part-six".to_owned());
    let scratch = authority.authorize_workspace(budget)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x63203), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x63201)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}
fn publish(
    value: &Value,
    context: &NativeNodeContext,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let samples = tensor_from_f32(
        compute.backend(),
        &serde_json::from_value::<Vec<u64>>(value["samples"]["shape"].clone())?,
        &serde_json::from_value::<Vec<f32>>(value["samples"]["values"].clone())?,
        &execution,
    )?;
    let mask = if let Some(mask) = value.get("noise_mask") {
        Some(tensor_from_f32(
            compute.backend(),
            &serde_json::from_value::<Vec<u64>>(mask["shape"].clone())?,
            &serde_json::from_value::<Vec<f32>>(mask["values"].clone())?,
            &execution,
        )?)
    } else {
        None
    };
    let indices = value
        .get("batch_index")
        .map(|v| serde_json::from_value::<Vec<u64>>(v.clone()))
        .transpose()?;
    let payload = NativeLatentBundle::single(
        samples.clone(),
        mask,
        None,
        NativeLatentMetadata::checked(
            None,
            None,
            value
                .get("downscale_ratio_spacial")
                .and_then(Value::as_u64)
                .map(u32::try_from)
                .transpose()?,
            value
                .get("downscale_ratio_temporal")
                .and_then(Value::as_u64)
                .map(u32::try_from)
                .transpose()?,
        )?,
        &execution,
    )?;
    let payload = if indices.is_some() {
        payload.replaced_samples_and_indices_source_dictionary(
            NativeLatentSamples::Tensor(samples),
            indices,
            &execution,
        )?
    } else {
        payload
    };
    Ok(context.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(payload)),
        &context.cancellation,
    )?)
}
fn output(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Values required".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    assert_eq!(outputs.len(), 1);
    let NativeValue::Handle { value } = outputs.into_iter().next().ok_or("Output")? else {
        return Err("Handle required".into());
    };
    Ok(value)
}
fn close(actual: f32, expected: &Value) -> bool {
    if let Some(special) = expected.as_str() {
        match special {
            "nan" => actual.is_nan(),
            "inf" => actual == f32::INFINITY,
            "-inf" => actual == f32::NEG_INFINITY,
            _ => false,
        }
    } else {
        expected.as_f64().is_some_and(|expected| {
            (f64::from(actual) - expected).abs() <= 2.0e-5 + expected.abs() * 2.0e-5
        })
    }
}
fn compare(
    value: &NativeLatentBundle,
    expected: &Value,
    context: &NativeNodeContext,
) -> TestResult {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let samples = value.samples().tensor().ok_or("Tensor samples")?;
    assert_eq!(
        samples.descriptor().shape(),
        serde_json::from_value::<Vec<u64>>(expected["samples"]["shape"].clone())?
    );
    let actual = tensor_to_f32(compute.backend(), samples, &execution)?;
    let expected_values = expected["samples"]["values"].as_array().ok_or("Values")?;
    assert_eq!(actual.len(), expected_values.len());
    for (index, (actual, expected)) in actual.iter().zip(expected_values).enumerate() {
        assert!(
            close(*actual, expected),
            "sample{index}: {actual} vs {expected}"
        );
    }
    assert_eq!(
        value.metadata().spatial_downscale_ratio(),
        expected
            .get("downscale_ratio_spacial")
            .and_then(Value::as_u64)
            .map(u32::try_from)
            .transpose()?
    );
    assert_eq!(
        value.metadata().temporal_downscale_ratio(),
        expected
            .get("downscale_ratio_temporal")
            .and_then(Value::as_u64)
            .map(u32::try_from)
            .transpose()?
    );
    assert_eq!(
        value.batch_indices(),
        expected
            .get("batch_index")
            .map(|v| serde_json::from_value::<Vec<u64>>(v.clone()))
            .transpose()?
            .as_deref()
    );
    if let Some(mask) = expected.get("noise_mask") {
        let actual = value.noise_mask().and_then(|v| v.tensor()).ok_or("Mask")?;
        assert_eq!(
            actual.descriptor().shape(),
            serde_json::from_value::<Vec<u64>>(mask["shape"].clone())?
        );
        let actual = tensor_to_f32(compute.backend(), actual, &execution)?;
        for (a, b) in actual
            .iter()
            .zip(mask["values"].as_array().ok_or("Mask values")?)
        {
            assert!(close(*a, b));
        }
    } else {
        assert!(value.noise_mask().is_none());
    }
    value.validate_retained()?;
    let restored: sim_tensor::NativeLatentBundleProjection =
        serde_json::from_slice(&serde_json::to_vec(value.projection())?)?;
    assert_eq!(&restored, value.projection());

    assert!(
        value.projection().resident_bytes()
            >= samples.descriptor().minimum_backing_byte_length()?
    );
    Ok(())
}
fn publish_mask(
    value: &Value,
    context: &NativeNodeContext,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let tensor = tensor_from_f32(
        compute.backend(),
        &serde_json::from_value::<Vec<u64>>(value["shape"].clone())?,
        &serde_json::from_value::<Vec<f32>>(value["values"].clone())?,
        &execution,
    )?;
    let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Mask, tensor)?;
    Ok(context.handle_store().publish(
        NativeStoredPayload::Tensor(Arc::new(payload)),
        &context.cancellation,
    )?)
}
fn inputs_for_case(
    case: &Value,
    context: &NativeNodeContext,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    if case["node"] == "StableCascade_EmptyLatentImage" {
        return ["width", "height", "compression", "batch_size"]
            .into_iter()
            .map(|name| {
                let value = case["arguments"]
                    .get(name)
                    .cloned()
                    .unwrap_or(Value::from(1));
                let primitive = if let Some(value) = value.as_i64() {
                    NativePrimitive::Integer(value)
                } else {
                    NativePrimitive::Number(value.as_f64().ok_or("Numeric source argument")?)
                };
                Ok((name.to_owned(), NativeValue::Primitive { value: primitive }))
            })
            .collect();
    }
    let mut inputs = BTreeMap::from([(
        "samples".to_owned(),
        NativeValue::Handle {
            value: publish(&case["input"], context)?,
        },
    )]);
    if case["node"] == "SetLatentNoiseMask" {
        inputs.insert(
            "mask".to_owned(),
            NativeValue::Handle {
                value: publish_mask(&case["mask"], context)?,
            },
        );
    } else {
        inputs.insert(
            "trim_amount".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(case["trim_amount"].as_i64().ok_or("Trim amount")?),
            },
        );
    }
    Ok(inputs)
}
#[test]
fn task632_two_registered_tensor_leaves_match_original_torch_dictionary_contracts() -> TestResult {
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let cases = oracle["cases"].as_array().ok_or("Source cases")?;
    assert_eq!(cases.len(), 108);
    let mut counts = [0_usize; 4];
    for (index, case) in cases.iter().enumerate() {
        let name = case["node"].as_str().ok_or("Node")?;
        let node = registry.node(name).ok_or("Production node")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632100 + index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let inputs = inputs_for_case(case, &context)?;
        let originals = inputs
            .values()
            .filter_map(|value| match value {
                NativeValue::Handle { value } => Some(value),
                _ => None,
            })
            .map(|handle| {
                Ok((
                    handle.clone(),
                    context
                        .handle_store()
                        .resolve(handle, handle.handle_type(), &context.cancellation)?
                        .digest_sha256(),
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        if case["node"] == "TrimVideoLatent"
            && case["trim_amount"]
                .as_i64()
                .is_some_and(|amount| amount < 0)
        {
            let boundary = oracle["source_schema_gaps"]
                .as_array()
                .ok_or("Schema boundaries")?
                .iter()
                .find(|boundary| boundary["case_index"].as_u64() == Some(index as u64))
                .ok_or("Original validator evidence")?;
            assert_eq!(boundary["source_prompt_validation"]["valid"], false);
            assert_eq!(
                boundary["source_prompt_validation"]["errors"][0]["type"],
                "value_smaller_than_min"
            );
            counts[3] += 1;
            let count = generation.len();
            assert_eq!(
                smol::block_on(node.execute(context.clone(), inputs))
                    .err()
                    .ok_or("Expected source prompt boundary rejection")?
                    .kind,
                NativeNodeFailureKind::Failure
            );
            assert_eq!(generation.len(), count);
            continue;
        }
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        let restored = serde_json::from_slice(&serde_json::to_vec(&inputs)?)?;
        assert_eq!(
            node.cache_change_token(&inputs)?,
            node.cache_change_token(&restored)?
        );
        assert_eq!(
            node.cache_dependencies(&context, &inputs)?,
            node.cache_dependencies(&context, &restored)?
        );
        let count = generation.len();
        let result = smol::block_on(node.execute(context.clone(), restored));
        if case.get("error").is_some() {
            counts[1] += 1;
            assert_eq!(
                result.err().ok_or("Original source failure")?.kind,
                NativeNodeFailureKind::Failure
            );
            assert_eq!(generation.len(), count);
        } else if ["samples", "noise_mask"].into_iter().any(|name| {
            case["output"][name]["shape"]
                .as_array()
                .is_some_and(|shape| shape.iter().any(|extent| extent.as_u64() == Some(0)))
        }) {
            counts[2] += 1;
            let error = result
                .err()
                .ok_or("Canonical nonempty representation gate")?;
            assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            assert!(error.message.contains("nonempty"), "{index}: {error}");
            assert_eq!(generation.len(), count);
        } else {
            counts[0] += 1;
            let handle = output(result.map_err(|error| format!("{index} {name}: {error}"))?)?;
            let payload = context.handle_store().resolve(
                &handle,
                handle.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Latent(bundle) = payload.as_ref() else {
                return Err("Real LATENT required".into());
            };
            compare(bundle, &case["output"], &context)
                .map_err(|error| format!("{index} {name}: {error}"))?;
            assert_eq!(generation.len(), count + 1);
        }
        for (handle, digest) in originals {
            assert_eq!(
                context
                    .handle_store()
                    .resolve(&handle, handle.handle_type(), &context.cancellation)?
                    .digest_sha256(),
                digest,
                "Source mutation"
            );
        }
    }
    assert_eq!(counts, [54, 9, 27, 18]);
    Ok(())
}
#[test]
fn task632_source_valid_representation_gaps_remain_visible_and_fail_typed_admission() -> TestResult
{
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let cases = oracle["representation_gaps"]
        .as_array()
        .ok_or("Source gaps")?;
    assert_eq!(cases.len(), 4);
    for (index, case) in cases.iter().enumerate() {
        assert!(case.get("output").is_some());
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632300 + index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            1024 * 1024,
        )?;
        let mask = publish_mask(&case["mask"], &context);
        if index < 3 {
            assert!(mask.is_err());
            assert_eq!(generation.len(), 0);
        } else {
            let node = registry.node("SetLatentNoiseMask").ok_or("Setter")?;
            let inputs = BTreeMap::from([
                (
                    "samples".to_owned(),
                    NativeValue::Handle {
                        value: publish(&case["input"], &context)?,
                    },
                ),
                ("mask".to_owned(), NativeValue::Handle { value: mask? }),
            ]);
            let count = generation.len();
            assert!(smol::block_on(node.execute(context.clone(), inputs)).is_err());
            assert_eq!(generation.len(), count);
        }
    }
    assert_eq!(
        oracle["source_invalid_input_errors"]
            .as_array()
            .ok_or("Source errors")?
            .len(),
        6
    );
    Ok(())
}
#[test]
fn task632_exact_three_row_claims_cache_lists_validation_and_cancellation() -> TestResult {
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let bindings = (sim_nodes::generated_model_latent_06::NATIVE_NODE_BINDINGS)()?;
    assert_eq!(bindings.len(), 5);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, binding) in bindings.iter().take(3).enumerate() {
        let name = &binding.descriptor().class_type;
        assert_eq!(
            binding.feature_id(),
            oracle["claims"][name].as_str().ok_or("Claim")?
        );
        assert_eq!(
            binding.disposition(),
            sim_nodes::NativeNodeBindingDisposition::Executable
        );
        binding.descriptor().validate_exact_schema_v2()?;
        assert!(binding.presentation().search_aliases.is_empty());
        let case = oracle["cases"]
            .as_array()
            .ok_or("Cases")?
            .iter()
            .find(|case| case["node"] == name.as_str())
            .or_else(|| {
                oracle["cascade_cases"]
                    .as_array()
                    .and_then(|cases| cases.first())
            })
            .ok_or("Case")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let cancellation = CancellationToken::default();
        let context = context(
            backend.clone(),
            &authority,
            generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632500 + index as u128))),
            cancellation.clone(),
            1024 * 1024,
        )?;
        let inputs = inputs_for_case(case, &context)?;
        let count = generation.len();
        let node = registry.node(name).ok_or("Node")?;
        let mut invalid = inputs.clone();
        invalid.insert(
            "unassigned".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(1),
            },
        );
        assert!(smol::block_on(node.execute(context.clone(), invalid)).is_err());
        let mapped_name = if name == "StableCascade_EmptyLatentImage" {
            "batch_size"
        } else {
            "samples"
        };
        for values in [
            Vec::new(),
            vec![inputs.get(mapped_name).ok_or("Mapped input")?.clone()],
        ] {
            let mut invalid = inputs.clone();
            invalid.insert(mapped_name.to_owned(), NativeValue::List { values });
            assert!(smol::block_on(node.execute(context.clone(), invalid)).is_err());
        }
        assert_eq!(generation.len(), count);
        cancellation.cancel();
        assert_eq!(
            smol::block_on(node.execute(context, inputs))
                .err()
                .ok_or("Cancellation")?
                .kind,
            NativeNodeFailureKind::Interrupted
        );
    }
    Ok(())
}

#[test]
fn task632_source_producer_gate_and_real_cascade_consumer_mapper_recovery() -> TestResult {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    assert!(registry.descriptor("EmptyLatentImage").is_none());
    assert!(registry.node("EmptyLatentImage").is_none());
    let original = oracle["mapped_cases"]
        .as_array()
        .ok_or("Preserved source scenarios")?;
    assert_eq!(original.len(), 2);
    for scenario in original {
        assert_eq!(
            scenario["outputs"]
                .as_array()
                .ok_or("Source golds retained")?
                .len(),
            3
        );
    }
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let integer = |value| NativeValue::Primitive {
        value: NativePrimitive::Integer(value),
    };
    let literal = |value| InputBinding::Literal { value };
    let scenario = oracle["cascade_mapped_cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Actual source Cascade mapping")?;
    let producer = NodeId("cascade-producer".to_owned());
    let consumers = [
        NodeId("trim-stage-0-c".to_owned()),
        NodeId("trim-stage-1-b".to_owned()),
    ];
    let node = |id: NodeId,
                name: &str,
                inputs: BTreeMap<String, InputBinding>|
     -> Result<CompiledNode, Box<dyn Error>> {
        Ok(CompiledNode {
            id,
            class_type: name.to_owned(),
            descriptor: registry
                .descriptor(name)
                .ok_or("Real generated descriptor")?
                .clone(),
            inputs,
            unknown: BTreeMap::new(),
        })
    };
    let mut producer_inputs = BTreeMap::new();
    for parameter in ["width", "height", "compression", "batch_size"] {
        producer_inputs.insert(
            parameter.to_owned(),
            literal(NativeValue::List {
                values: scenario["inputs"][parameter]
                    .as_array()
                    .ok_or("Mapped source input")?
                    .iter()
                    .map(|value| Ok(integer(value.as_i64().ok_or("Integer")?)))
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?,
            }),
        );
    }
    let mut nodes = BTreeMap::from([(
        producer.clone(),
        node(
            producer.clone(),
            "StableCascade_EmptyLatentImage",
            producer_inputs,
        )?,
    )]);
    let mut order = vec![producer.clone()];
    for (column, consumer) in consumers.iter().enumerate() {
        let mut compiled = node(
            consumer.clone(),
            "TrimVideoLatent",
            BTreeMap::from([
                (
                    "samples".to_owned(),
                    InputBinding::Link {
                        source: producer.clone(),
                        output_index: column,
                        lazy: false,
                        mode: sim_nodes::NativePortCardinality::Mapped,
                    },
                ),
                ("trim_amount".to_owned(), literal(integer(0))),
            ]),
        )?;
        compiled.descriptor.output_node = true;
        nodes.insert(consumer.clone(), compiled);
        order.push(consumer.clone());
    }
    let plan = CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x632600)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes,
        static_required_nodes: order.iter().cloned().collect(),
        topological_order: order,
        output_nodes: consumers.to_vec(),
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    let encoded = serde_json::to_vec(&plan)?;
    let mut previous: Option<Vec<NativeValue>> = None;
    for restart in 0..2 {
        let restored: CompiledPlan = serde_json::from_slice(&encoded)?;
        let executor = NativeImageExecutor::new_with_generated_registry(
            sim_types::ProfileId(Uuid::from_u128(0x632610)),
            BTreeMap::new(),
            true,
            backend.clone(),
        )?;
        let attempt = AttemptId(Uuid::from_u128(0x632700 + restart * 2));
        let result = executor.execute_blocking(
            &restored,
            attempt,
            CancellationToken::default(),
            0,
            authority.authorize_workspace(16 * 1024 * 1024)?,
        )?;
        assert_eq!(
            result.report.state,
            AttemptState::Succeeded,
            "{:?}",
            result.report.error
        );
        assert_eq!(result.report.cache_hits, 0);
        let generation = executor.handle_store_generation_for_test();
        let context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(attempt),
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let mut retained = Vec::new();
        for (column, consumer) in consumers.iter().enumerate() {
            let columns = result
                .report
                .outputs
                .get(consumer)
                .ok_or("Owned consumer outputs")?;
            assert_eq!(columns.len(), 1);
            let NativeValue::List { values } = columns.first().ok_or("Mapped column")? else {
                return Err("Real mapped list".into());
            };
            assert_eq!(values.len(), 3);
            for (value, expected) in values.iter().zip(
                scenario["columns"][column]
                    .as_array()
                    .ok_or("Actual source producer gold")?,
            ) {
                let NativeValue::Handle { value } = value else {
                    return Err("LATENT handle".into());
                };
                let payload = context.handle_store().resolve(
                    value,
                    value.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(bundle) = payload.as_ref() else {
                    return Err("LATENT bundle".into());
                };
                // The original trim body at amount zero retains the producer's full samples and dictionary.
                compare(bundle, expected, &context)?;
            }
            retained.extend(values.clone());
        }
        if let Some(previous) = &previous {
            for value in previous {
                let NativeValue::Handle { value } = value else {
                    return Err("Previous handle".into());
                };
                assert!(
                    context
                        .handle_store()
                        .resolve(value, value.handle_type(), &context.cancellation)
                        .is_err()
                );
            }
        }
        previous = Some(retained);
        let cached = executor.execute_blocking(
            &restored,
            AttemptId(Uuid::from_u128(0x632701 + restart * 2)),
            CancellationToken::default(),
            0,
            authority.authorize_workspace(16 * 1024 * 1024)?,
        )?;
        assert_eq!(cached.report.state, AttemptState::Succeeded);
        assert!(cached.report.cache_hits > 0);
    }
    Ok(())
}

#[derive(Debug)]
struct CancelAfterPublish {
    inner: Arc<dyn NativeHandleStore>,
    armed: std::sync::atomic::AtomicBool,
    revoked: std::sync::atomic::AtomicUsize,
    cancellation: CancellationToken,
}
impl NativeHandleStore for CancelAfterPublish {
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
        let handle = self.inner.publish(payload, cancellation)?;
        if self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.cancellation.cancel();
        }
        Ok(handle)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), sim_nodes::NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)?;
        self.revoked
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn task632_workspace_failure_and_postpublication_cancellation_are_atomic() -> TestResult {
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let case = oracle["cases"]
        .as_array()
        .ok_or("Cases")?
        .iter()
        .find(|case| case["node"] == "TrimVideoLatent" && case["trim_amount"] == 1)
        .ok_or("Trim case")?;
    let generation = NativeHandleStoreGeneration::new()?;
    let cancellation = CancellationToken::default();
    let store = Arc::new(CancelAfterPublish {
        inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632900))),
        armed: std::sync::atomic::AtomicBool::new(false),
        revoked: std::sync::atomic::AtomicUsize::new(0),
        cancellation: cancellation.clone(),
    });
    let first_context = context(
        backend,
        &authority,
        store.clone(),
        cancellation,
        1024 * 1024,
    )?;
    let inputs = inputs_for_case(case, &first_context)?;
    let count = generation.len();
    let node = registry.node("TrimVideoLatent").ok_or("Trimmer")?;
    let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(256)?;
    let limited_generation = NativeHandleStoreGeneration::new()?;
    let limited = context(
        Arc::new(limited_backend),
        &limited_authority,
        limited_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632901))),
        CancellationToken::default(),
        256,
    )?;
    let limited_inputs = inputs_for_case(case, &limited)?;
    let limited_count = limited_generation.len();
    assert!(smol::block_on(node.execute(limited.clone(), limited_inputs.clone())).is_err());
    for value in limited_inputs.values() {
        if let NativeValue::Handle { value } = value {
            limited
                .handle_store()
                .resolve(value, value.handle_type(), &limited.cancellation)?;
        }
    }
    assert_eq!(limited_generation.len(), limited_count);
    store.armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let result = smol::block_on(node.execute(first_context, inputs));
    assert_eq!(
        result.err().ok_or("Cancellation after publication")?.kind,
        NativeNodeFailureKind::Interrupted
    );
    assert_eq!(store.revoked.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(generation.len(), count);
    Ok(())
}

#[derive(Debug)]
struct TwoOutputFault {
    inner: Arc<dyn NativeHandleStore>,
    attempts: std::sync::atomic::AtomicUsize,
    revoked: std::sync::atomic::AtomicUsize,
    reject_at: usize,
    cancel_after: usize,
    cancellation: CancellationToken,
}
impl NativeHandleStore for TwoOutputFault {
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
        let attempt = self
            .attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if attempt == self.reject_at {
            return Err(sim_nodes::NativeHandleStoreError::Rejected(
                "scripted second-output acceptance failure".to_owned(),
            ));
        }
        let handle = self.inner.publish(payload, cancellation)?;
        if attempt == self.cancel_after {
            self.cancellation.cancel();
        }
        Ok(handle)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), sim_nodes::NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)?;
        self.revoked
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn task632_stable_cascade_original_two_outputs_schema_boundaries_and_atomic_publication()
-> TestResult {
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("StableCascade_EmptyLatentImage")
        .ok_or("Cascade production node")?;
    let descriptor = registry
        .descriptor("StableCascade_EmptyLatentImage")
        .ok_or("Cascade descriptor")?;
    assert_eq!(descriptor.outputs.len(), 2);
    assert_eq!(
        descriptor
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        ["stage_c", "stage_b"]
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let cases = oracle["cascade_cases"].as_array().ok_or("Cascade cases")?;
    assert_eq!(cases.len(), 9);
    for (index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let context = context(
            backend.clone(),
            &authority,
            generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632a00 + index as u128))),
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let inputs = inputs_for_case(case, &context)?;
        let result = smol::block_on(node.execute(context.clone(), inputs));
        if case.get("error").is_some() {
            assert_eq!(
                result.err().ok_or("Source error")?.kind,
                NativeNodeFailureKind::Failure
            );
            assert_eq!(generation.len(), 0);
        } else {
            let NativeNodeOutcome::Values {
                outputs,
                ui,
                effects,
            } = result.map_err(|error| format!("Cascade {index}: {error}"))?
            else {
                return Err("Cascade values required".into());
            };
            assert!(ui.is_none() && effects.is_empty());
            assert_eq!(outputs.len(), 2);
            assert_eq!(generation.len(), 2);
            for (value, expected) in outputs
                .iter()
                .zip(case["outputs"].as_array().ok_or("Two original outputs")?)
            {
                let NativeValue::Handle { value } = value else {
                    return Err("Real LATENT handle".into());
                };
                let payload = context.handle_store().resolve(
                    value,
                    value.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(bundle) = payload.as_ref() else {
                    return Err("Canonical LATENT bundle".into());
                };
                assert_eq!(
                    bundle
                        .samples()
                        .tensor()
                        .ok_or("Tensor")?
                        .descriptor()
                        .dtype(),
                    sim_tensor::DType::F32
                );
                compare(bundle, expected, &context)?;
            }
        }
    }
    let gaps = oracle["cascade_gaps"]
        .as_array()
        .ok_or("Cascade direct-body boundaries")?;
    assert_eq!(gaps.len(), 6);
    for (index, case) in gaps.iter().enumerate() {
        assert!(case.get("outputs").is_some());
        let generation = NativeHandleStoreGeneration::new()?;
        let context = context(
            backend.clone(),
            &authority,
            generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632b00 + index as u128))),
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let result =
            smol::block_on(node.execute(context.clone(), inputs_for_case(case, &context)?));
        if index < 5 {
            assert_eq!(case["source_prompt_validation"]["valid"], false);
            assert_eq!(
                result.err().ok_or("Source and native schema reject")?.kind,
                NativeNodeFailureKind::Failure
            );
            assert_eq!(generation.len(), 0);
        } else {
            assert_eq!(case["source_prompt_validation"]["valid"], true);
            assert_eq!(case["output_dtypes"][0], "torch.float64");
            let NativeNodeOutcome::Values { outputs, .. } = result? else {
                return Err("Native default dtype result".into());
            };
            for value in outputs {
                let NativeValue::Handle { value } = value else {
                    return Err("Handle".into());
                };
                let payload = context.handle_store().resolve(
                    &value,
                    value.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(bundle) = payload.as_ref() else {
                    return Err("Latent".into());
                };
                assert_eq!(
                    bundle
                        .samples()
                        .tensor()
                        .ok_or("Tensor")?
                        .descriptor()
                        .dtype(),
                    sim_tensor::DType::F32
                );
            }
        }
    }
    for (index, (reject_at, cancel_after, expected_revoke)) in
        [(1, 0, 0), (2, 0, 1), (0, 1, 1), (0, 2, 2)]
            .into_iter()
            .enumerate()
    {
        let generation = NativeHandleStoreGeneration::new()?;
        let cancellation = CancellationToken::default();
        let store = Arc::new(TwoOutputFault {
            inner: generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632c00 + index as u128))),
            attempts: std::sync::atomic::AtomicUsize::new(0),
            revoked: std::sync::atomic::AtomicUsize::new(0),
            reject_at,
            cancel_after,
            cancellation: cancellation.clone(),
        });
        let context = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation,
            8 * 1024 * 1024,
        )?;
        let error = smol::block_on(node.execute(
            context.clone(),
            inputs_for_case(cases.first().ok_or("Case")?, &context)?,
        ))
        .err()
        .ok_or("Publication error")?;
        assert_eq!(
            error.kind,
            if cancel_after == 0 {
                NativeNodeFailureKind::Failure
            } else {
                NativeNodeFailureKind::Interrupted
            }
        );
        assert_eq!(generation.len(), 0);
        assert_eq!(
            store.revoked.load(std::sync::atomic::Ordering::SeqCst),
            expected_revoke
        );
    }
    let generation = NativeHandleStoreGeneration::new()?;
    let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(16)?;
    let context = context(
        Arc::new(limited_backend),
        &limited_authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632d00))),
        CancellationToken::default(),
        16,
    )?;
    assert!(
        smol::block_on(node.execute(
            context.clone(),
            inputs_for_case(cases.first().ok_or("Case")?, &context)?
        ))
        .is_err()
    );
    assert_eq!(generation.len(), 0);
    Ok(())
}

#[test]
fn task632_stage_c_encoder_exact_schema_and_preexecution_cancellation() -> TestResult {
    let bindings = (sim_nodes::generated_model_latent_06::NATIVE_NODE_BINDINGS)()?;
    let binding = bindings
        .iter()
        .find(|binding| binding.descriptor().class_type == "StableCascade_StageC_VAEEncode")
        .ok_or("Stage C encoder binding absent")?;
    assert_eq!(binding.feature_id(), "COMFY-NODE-0637");
    binding.descriptor().validate_exact_schema_v2()?;
    assert_eq!(
        binding
            .descriptor()
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["image", "vae", "compression"]
    );
    assert_eq!(binding.descriptor().outputs.len(), 2);
    assert!(binding.presentation().search_aliases.is_empty());
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("StableCascade_StageC_VAEEncode")
        .ok_or("Stage C node absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let cancellation = CancellationToken::default();
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632e00))),
        cancellation.clone(),
        1024 * 1024,
    )?;
    let count = generation.len();
    assert!(smol::block_on(node.execute(context.clone(), BTreeMap::new())).is_err());
    assert_eq!(generation.len(), count);
    cancellation.cancel();
    assert_eq!(
        smol::block_on(node.execute(context, BTreeMap::new()))
            .err()
            .ok_or("Cancellation failure absent")?
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    assert_eq!(generation.len(), count);
    Ok(())
}

fn original_full_stage_c_vae(
    context: &NativeNodeContext,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    let selected = std::env::var_os("SIM_STAGE_C_ENCODER_CHECKPOINT")
        .ok_or("Required actual Stage C checkpoint absent: set SIM_STAGE_C_ENCODER_CHECKPOINT")?;
    let checkpoint = std::path::PathBuf::from(selected).canonicalize()?;
    let bytes = std::fs::read(&checkpoint)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "e23c1fe17ba8cdd6c49a5b292d72502bd4cf729e7de9f2060bb4dcc26737c0a0"
    );
    canonical_source_image_vae(
        context,
        &checkpoint,
        &sim_model::VaeKernelProfile::StableCascadeStageCEncoderV1,
    )
}

fn canonical_source_image_vae(
    context: &NativeNodeContext,
    checkpoint: &std::path::Path,
    profile: &sim_model::VaeKernelProfile,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    use sim_model::vae_image::load_image_vae_from_model_store_with_context;
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelStore, NativeModelPayload, ParserLimits,
        PatchGraph, VaeArchitectureRegistry, VaeBoundary, VaeDescriptor,
    };
    let directory = checkpoint.parent().ok_or("Checkpoint parent absent")?;
    let filename = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("UTF8 checkpoint filename required")?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "task632-source-vae",
        "checkpoint",
        directory,
        ["safetensors"],
    )?)?;
    index.refresh(&context.cancellation)?;
    let key = ArtifactKey::new("task632-source-vae", filename)?;
    let artifact = index
        .record(&key)
        .cloned()
        .ok_or("Actual checkpoint record absent")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &artifact.key, &context.cancellation)?;
    let registry = VaeArchitectureRegistry::checked()?;
    let (families, latents) = VaeArchitectureRegistry::canonical_targets()?;
    let probe = store.family_probe(&loaded, &context.cancellation)?;
    let selection = registry.select(&probe, &context.cancellation)?;
    let target =
        registry.intended_target(&selection, &families, &latents, &context.cancellation)?;
    let definition = latents
        .get(target.latent_format())
        .ok_or("Latent format absent")?;
    let patch = PatchGraph::checked_semantic(&artifact.sha256, Vec::new())?.identity();
    let descriptor = VaeDescriptor::checked_selection(
        &artifact,
        &selection,
        &target,
        &families,
        &latents,
        patch,
        VaeBoundary::image(3)?,
        [0.0, 1.0],
        &context.cancellation,
    )?;
    assert_eq!(descriptor.identity().profile(), profile);
    if profile == &sim_model::VaeKernelProfile::StableCascadeStageCEncoderV1 {
        assert_eq!(descriptor.source_image_encode_downscale_ratio()?, 32);
    }
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let vae = Arc::new(load_image_vae_from_model_store_with_context(
        compute.backend(),
        &store,
        &index,
        loaded,
        descriptor,
        definition,
        &execution,
    )?);
    let model = Arc::new(NativeModelPayload::native_vae(vae)?);
    let diffusion = Arc::new(sim_sampler::NativeDiffusionPayload::vae(model)?);
    Ok(context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(
            sim_nodes::NativeStoredModelPayload::native_diffusion(diffusion)?,
        )),
        &context.cancellation,
    )?)
}

#[test]
#[ignore = "requires hash-verified actual StabilityAI encoder in SIM_STAGE_C_ENCODER_CHECKPOINT; required Task632 source acceptance command"]
fn task632_stage_c_actual_full_resource_source_acceptance() -> TestResult {
    let oracle = fixture()?;
    let reference = &oracle["stage_c_encoder"];
    assert_eq!(
        reference["checkpoint"]["sha256"],
        "e23c1fe17ba8cdd6c49a5b292d72502bd4cf729e7de9f2060bb4dcc26737c0a0"
    );
    assert_eq!(reference["feature_block_count"], 40);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("StableCascade_StageC_VAEEncode")
        .ok_or("Stage C node absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632e01))),
        CancellationToken::default(),
        1024 * 1024 * 1024,
    )?;
    let vae = original_full_stage_c_vae(&context)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let mut numeric = 0;
    let mut source_failures = 0;
    let mut representation_gaps = 0;
    for case in reference["cases"]
        .as_array()
        .ok_or("Stage C source cases absent")?
    {
        let shape = serde_json::from_value::<Vec<u64>>(case["image"]["shape"].clone())?;
        let values = serde_json::from_value::<Vec<f32>>(case["image"]["values"].clone())?;
        let tensor = tensor_from_f32(compute.backend(), &shape, &values, &execution)?;
        let image = context.handle_store().publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                sim_tensor::ImageTensor::from_tensor(tensor)?,
            )?)),
            &context.cancellation,
        )?;
        let inputs = BTreeMap::from([
            (
                "image".to_owned(),
                NativeValue::Handle {
                    value: image.clone(),
                },
            ),
            ("vae".to_owned(), NativeValue::Handle { value: vae.clone() }),
            (
                "compression".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(
                        case["compression"].as_i64().ok_or("Compression")?,
                    ),
                },
            ),
        ]);
        let count = generation.len();
        let result = smol::block_on(node.execute(context.clone(), inputs));
        if case.get("error").is_some() {
            assert!(
                result.is_err(),
                "Source failure must remain failure: {}",
                case["label"]
            );
            assert_eq!(generation.len(), count);
            source_failures += 1;
        } else if shape.get(3) == Some(&1) || shape.get(1).is_some_and(|height| *height < 8) {
            assert!(
                result.is_err(),
                "Canonical grayscale/empty-stage-B gap: {}",
                case["label"]
            );
            assert_eq!(generation.len(), count);
            representation_gaps += 1;
        } else {
            let NativeNodeOutcome::Values {
                outputs,
                ui,
                effects,
            } = result?
            else {
                return Err("Actual Stage C tensor outputs required".into());
            };
            assert!(ui.is_none() && effects.is_empty());
            assert_eq!(outputs.len(), 2);
            for (output, gold) in outputs
                .iter()
                .zip(case["outputs"].as_array().ok_or("Golds")?)
            {
                let NativeValue::Handle { value } = output else {
                    return Err("LATENT handle required".into());
                };
                let resolved = context.handle_store().resolve(
                    value,
                    value.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(bundle) = resolved.as_ref() else {
                    return Err("LATENT bundle required".into());
                };
                assert!(bundle.noise_mask().is_none() && bundle.batch_indices().is_none());
                assert_eq!(bundle.metadata(), &NativeLatentMetadata::default());
                let samples = bundle.samples().tensor().ok_or("Tensor LATENT required")?;
                assert_eq!(
                    samples.descriptor().shape(),
                    serde_json::from_value::<Vec<u64>>(gold["samples"]["shape"].clone())?
                );
                let actual = tensor_to_f32(compute.backend(), samples, &execution)?;
                let expected =
                    serde_json::from_value::<Vec<f32>>(gold["samples"]["values"].clone())?;
                assert_eq!(actual.len(), expected.len());
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (*actual - *expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                        "Full source numerical mismatch at {index}: {actual} vs {expected}; {}",
                        case["label"]
                    );
                }
                context
                    .handle_store()
                    .revoke(value, &context.cancellation)?;
            }
            numeric += 1;
        }
        context
            .handle_store()
            .revoke(&image, &context.cancellation)?;
    }
    assert_eq!((numeric, source_failures, representation_gaps), (3, 6, 2));
    let model = context
        .handle_store()
        .resolve(&vae, vae.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Model(model) = model.as_ref() else {
        return Err("Canonical loaded model payload required".into());
    };
    let first = reference["cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Source first numeric case absent")?;
    let shape = serde_json::from_value::<Vec<u64>>(first["image"]["shape"].clone())?;
    let values = serde_json::from_value::<Vec<f32>>(first["image"]["values"].clone())?;
    for (index, (reject_at, cancel_after, revoked)) in [(1, 0, 0), (2, 0, 1), (0, 1, 1), (0, 2, 2)]
        .into_iter()
        .enumerate()
    {
        let attempt = AttemptId(Uuid::from_u128(0x632e10 + index as u128));
        let inner = generation.handle_store_for_attempt(attempt);
        let cancellation = CancellationToken::default();
        let model_handle =
            inner.publish(NativeStoredPayload::Model(model.clone()), &cancellation)?;
        let tensor = tensor_from_f32(compute.backend(), &shape, &values, &execution)?;
        let image = inner.publish(
            NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                NativeTensorRole::Image,
                sim_tensor::ImageTensor::from_tensor(tensor)?,
            )?)),
            &cancellation,
        )?;
        let fault = Arc::new(TwoOutputFault {
            inner,
            attempts: std::sync::atomic::AtomicUsize::new(0),
            revoked: std::sync::atomic::AtomicUsize::new(0),
            reject_at,
            cancel_after,
            cancellation: cancellation.clone(),
        });
        let interrupted = crate::context(
            backend.clone(),
            &authority,
            fault.clone(),
            cancellation,
            1024 * 1024 * 1024,
        )?;
        let inputs = BTreeMap::from([
            (
                "image".to_owned(),
                NativeValue::Handle {
                    value: image.clone(),
                },
            ),
            (
                "vae".to_owned(),
                NativeValue::Handle {
                    value: model_handle.clone(),
                },
            ),
            (
                "compression".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(42),
                },
            ),
        ]);
        let count = generation.len();
        let error = smol::block_on(node.execute(interrupted.clone(), inputs))
            .err()
            .ok_or("Expected actual VAE publication rollback failure")?;
        if cancel_after != 0 {
            assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        }
        assert_eq!(
            fault.revoked.load(std::sync::atomic::Ordering::SeqCst),
            revoked
        );
        assert_eq!(generation.len(), count);
        let cleanup = CancellationToken::default();
        fault.resolve(&image, image.handle_type(), &cleanup)?;
        fault.resolve(&model_handle, model_handle.handle_type(), &cleanup)?;
        fault.revoke(&image, &cleanup)?;
        fault.revoke(&model_handle, &cleanup)?;
    }
    Ok(())
}

#[test]
fn task632_vae_decode_reconciled_factory_exact_schema_and_precancel() -> TestResult {
    let binding = sim_nodes::generated_model_latent_06::vae_decode_native_binding()?;
    assert_eq!(binding.feature_id(), "COMFY-NODE-0710");
    let descriptor = binding.descriptor();
    assert_eq!(descriptor.class_type, "VAEDecode");
    assert_eq!(
        descriptor
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["samples", "vae"]
    );
    assert_eq!(descriptor.outputs.len(), 1);
    assert_eq!(
        descriptor
            .outputs
            .first()
            .ok_or("IMAGE output absent")?
            .name,
        "image"
    );
    assert_eq!(
        binding.presentation().search_aliases,
        [
            "decode",
            "decode latent",
            "latent to image",
            "render latent"
        ]
    );
    let node = sim_nodes::generated_model_latent_06::vae_decode_native_node();
    assert_eq!(node.class_type(), "VAEDecode");
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4096)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632d01))),
        cancellation,
        1024,
    )?;
    let error = smol::block_on(node.execute(context, BTreeMap::new()))
        .err()
        .ok_or("Precancellation required")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert!(generation.is_empty());
    Ok(())
}

#[test]
fn task632_vae_decode_original_pixel_model_and_source_boundaries() -> TestResult {
    use sha2::{Digest, Sha256};
    let source = fixture()?;
    let reference = &source["vae_decode"];
    let bytes = serde_json::from_value::<Vec<u8>>(reference["artifact"]["bytes"].clone())?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        reference["artifact"]["sha256"]
            .as_str()
            .ok_or("Source artifact hash absent")?
    );
    let directory = tempfile::tempdir()?;
    let checkpoint = directory.path().join("original-pixel-space.safetensors");
    std::fs::write(&checkpoint, bytes)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632d02))),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    let vae = canonical_source_image_vae(
        &context,
        &checkpoint,
        &sim_model::VaeKernelProfile::PixelSpaceV1,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("VAEDecode")
        .ok_or("Reconciled VAEDecode registry binding absent")?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let mut numeric = 0;
    let mut gaps = 0;
    let mut failures = 0;
    for case in reference["cases"]
        .as_array()
        .ok_or("Decode source cases absent")?
    {
        let count = generation.len();
        if case.get("error_type").is_some() {
            if case["kind"] == "tensor" {
                assert!(
                    publish(&serde_json::json!({"samples": case["input"]}), &context).is_err(),
                    "Source invalid/empty input must fail canonical admission: {}",
                    case["label"]
                );
            } else {
                let error = smol::block_on(node.execute(
                    context.clone(),
                    BTreeMap::from([(
                        "vae".to_owned(),
                        NativeValue::Handle { value: vae.clone() },
                    )]),
                ))
                .err()
                .ok_or("Unrepresentable/malformed source input must fail node validation")?;
                assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            }
            failures += 1;
            assert_eq!(generation.len(), count);
            continue;
        }
        if case["kind"] == "nested" {
            let components = case["input"].as_array().ok_or("Nested components absent")?;
            let first = components.first().ok_or("Nested first absent")?;
            let shape = serde_json::from_value::<Vec<u64>>(first["shape"].clone())?;
            if shape.len() == 4 {
                let video = tensor_from_f32(
                    compute.backend(),
                    &shape,
                    &serde_json::from_value::<Vec<f32>>(first["values"].clone())?,
                    &execution,
                )?;
                let audio = components.get(1).ok_or("Nested second absent")?;
                let audio = tensor_from_f32(
                    compute.backend(),
                    &serde_json::from_value::<Vec<u64>>(audio["shape"].clone())?,
                    &serde_json::from_value::<Vec<f32>>(audio["values"].clone())?,
                    &execution,
                )?;
                assert!(
                    NativeLatentBundle::checked(
                        NativeLatentSamples::AudioVideo { video, audio },
                        None,
                        None,
                        NativeLatentMetadata::default(),
                        &execution
                    )
                    .is_err()
                );
                gaps += 1;
                continue;
            }
        }
        let input = if case["kind"] == "nested" {
            case["input"]
                .as_array()
                .and_then(|parts| parts.first())
                .ok_or("First source nested component absent")?
                .clone()
        } else {
            case["input"].clone()
        };
        let latent = if case["kind"] == "nested" {
            let video = tensor_from_f32(
                compute.backend(),
                &serde_json::from_value::<Vec<u64>>(input["shape"].clone())?,
                &serde_json::from_value::<Vec<f32>>(input["values"].clone())?,
                &execution,
            )?;
            let audio = case["input"]
                .as_array()
                .and_then(|parts| parts.get(1))
                .ok_or("Original nested audio absent")?;
            let audio = tensor_from_f32(
                compute.backend(),
                &serde_json::from_value::<Vec<u64>>(audio["shape"].clone())?,
                &serde_json::from_value::<Vec<f32>>(audio["values"].clone())?,
                &execution,
            )?;
            let bundle = NativeLatentBundle::checked(
                NativeLatentSamples::AudioVideo { video, audio },
                None,
                None,
                NativeLatentMetadata::default(),
                &execution,
            )?;
            context.handle_store().publish(
                NativeStoredPayload::Latent(Arc::new(bundle)),
                &context.cancellation,
            )?
        } else {
            publish(
                &serde_json::json!({"samples": input, "batch_index": [99]}),
                &context,
            )?
        };
        let digest = context
            .handle_store()
            .resolve(&latent, latent.handle_type(), &context.cancellation)?
            .digest_sha256();
        let inputs = BTreeMap::from([
            (
                "samples".to_owned(),
                NativeValue::Handle {
                    value: latent.clone(),
                },
            ),
            ("vae".to_owned(), NativeValue::Handle { value: vae.clone() }),
        ]);
        let before = generation.len();
        let outcome = smol::block_on(node.execute(context.clone(), inputs));
        let shape = serde_json::from_value::<Vec<u64>>(input["shape"].clone())?;
        if shape.len() == 5 || shape.get(1) != Some(&3) {
            assert!(
                outcome.is_err(),
                "Explicit canonical profile/rank gap: {}",
                case["label"]
            );
            assert_eq!(generation.len(), before);
            gaps += 1;
        } else {
            let image = output(outcome?)?;
            let resolved = context.handle_store().resolve(
                &image,
                image.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
                return Err("IMAGE tensor output required".into());
            };
            assert_eq!(payload.role(), NativeTensorRole::Image);
            let tensor = payload.image().ok_or("ImageTensor required")?.tensor();
            assert_eq!(
                tensor.descriptor().shape(),
                serde_json::from_value::<Vec<u64>>(case["output"]["shape"].clone())?
            );
            let actual = tensor_to_f32(compute.backend(), tensor, &execution)?;
            let expected = case["output"]["values"]
                .as_array()
                .ok_or("Original source values absent")?;
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().copied().zip(expected) {
                assert!(
                    close(actual, expected),
                    "Original pixel decode mismatch: {}",
                    case["label"]
                );
            }
            context
                .handle_store()
                .revoke(&image, &context.cancellation)?;
            numeric += 1;
        }
        assert_eq!(
            context
                .handle_store()
                .resolve(&latent, latent.handle_type(), &context.cancellation)?
                .digest_sha256(),
            digest
        );
        context
            .handle_store()
            .revoke(&latent, &context.cancellation)?;
    }
    assert_eq!((numeric, gaps, failures), (4, 5, 5));
    let resolved =
        context
            .handle_store()
            .resolve(&vae, vae.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Model(model) = resolved.as_ref() else {
        return Err("Canonical model payload absent".into());
    };
    let first = reference["cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("First original decode case absent")?;
    for (index, (reject_at, cancel_after, expected_revoked, budget)) in [
        (1, 0, 0, 8 * 1024 * 1024),
        (0, 1, 1, 8 * 1024 * 1024),
        (0, 0, 0, 1),
    ]
    .into_iter()
    .enumerate()
    {
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x632d10 + index as u128)));
        let cancellation = CancellationToken::default();
        let model_handle =
            store.publish(NativeStoredPayload::Model(model.clone()), &cancellation)?;
        let setup = crate::context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            8 * 1024 * 1024,
        )?;
        let latent = publish(&serde_json::json!({"samples": first["input"]}), &setup)?;
        let fault = Arc::new(TwoOutputFault {
            inner: store,
            attempts: std::sync::atomic::AtomicUsize::new(0),
            revoked: std::sync::atomic::AtomicUsize::new(0),
            reject_at,
            cancel_after,
            cancellation: cancellation.clone(),
        });
        let interrupted = crate::context(
            backend.clone(),
            &authority,
            fault.clone(),
            cancellation,
            budget,
        )?;
        let before = generation.len();
        let error = smol::block_on(node.execute(
            interrupted,
            BTreeMap::from([
                (
                    "samples".to_owned(),
                    NativeValue::Handle {
                        value: latent.clone(),
                    },
                ),
                (
                    "vae".to_owned(),
                    NativeValue::Handle {
                        value: model_handle.clone(),
                    },
                ),
            ]),
        ))
        .err()
        .ok_or("Required canonical allocation/publication failure absent")?;
        if cancel_after != 0 {
            assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        }
        assert_eq!(
            fault.revoked.load(std::sync::atomic::Ordering::SeqCst),
            expected_revoked
        );
        assert_eq!(generation.len(), before);
        let cleanup = CancellationToken::default();
        fault.resolve(&latent, latent.handle_type(), &cleanup)?;
        fault.resolve(&model_handle, model_handle.handle_type(), &cleanup)?;
        fault.revoke(&latent, &cleanup)?;
        fault.revoke(&model_handle, &cleanup)?;
    }
    Ok(())
}
