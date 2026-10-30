use serde_json::Value;
use sim_nodes::{
    NativeHandleStore, NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle,
    NativePrimitive, NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, NativeLatentBundle, NativeLatentMetadata,
    NativeLatentSamples, StreamId,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;
type TestResult = Result<(), Box<dyn Error>>;
fn fixture() -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0322/source-contract.json"
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
    let node = NodeId("latent-part-four".to_owned());
    let scratch = authority.authorize_workspace(budget)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x63003), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x63001)),
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
        NativeLatentMetadata::checked(None, None, Some(8), None)?,
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
fn parameters(case: &Value) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    case["arguments"]
        .as_object()
        .ok_or("Arguments")?
        .iter()
        .map(|(name, value)| {
            let primitive = if let Some(value) = value.as_str() {
                NativePrimitive::String(value.to_owned())
            } else if ["ratio", "multiplier", "alpha", "sigma", "scale_by"].contains(&name.as_str())
            {
                NativePrimitive::Number(value.as_f64().ok_or("Number")?)
            } else {
                NativePrimitive::Integer(value.as_i64().ok_or("Integer")?)
            };
            Ok((name.clone(), NativeValue::Primitive { value: primitive }))
        })
        .collect()
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
    assert_eq!(value.metadata().spatial_downscale_ratio(), Some(8));
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
#[test]
fn task630_all_ten_native_nodes_match_actual_torch_source_and_resource_consumers() -> TestResult {
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let cases = oracle["cases"].as_array().ok_or("Cases")?;
    assert_eq!(cases.len(), 88);
    for (index, case) in cases.iter().enumerate() {
        let name = case["node"].as_str().ok_or("Node")?;
        let node = registry
            .node(name)
            .ok_or_else(|| format!("Missing {name}"))?;
        registry
            .descriptor(name)
            .ok_or("Descriptor")?
            .validate_exact_schema_v2()?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x630100 + index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let first = publish(&case["first"], &context)?;
        let first_payload =
            context
                .handle_store()
                .resolve(&first, first.handle_type(), &context.cancellation)?;
        let before = first_payload.digest_sha256();
        let mut inputs = parameters(case)?;
        let operation = name.starts_with("LatentOperation");
        if !operation {
            inputs.insert(
                if matches!(name, "LatentSubtract" | "LatentInterpolate") {
                    "samples1"
                } else {
                    "samples"
                }
                .to_owned(),
                NativeValue::Handle {
                    value: first.clone(),
                },
            );
        }
        if let Some(second) = case.get("second") {
            inputs.insert(
                "samples2".to_owned(),
                NativeValue::Handle {
                    value: publish(second, &context)?,
                },
            );
        }
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        let token = node.cache_change_token(&inputs)?;
        let dependencies = node.cache_dependencies(&context, &inputs)?;
        let restored = serde_json::from_slice(&serde_json::to_vec(&inputs)?)?;
        assert_eq!(token, node.cache_change_token(&restored)?);
        assert_eq!(dependencies, node.cache_dependencies(&context, &restored)?);
        let count = generation.len();
        let result = smol::block_on(node.execute(context.clone(), restored));
        let result = if operation {
            let operation = output(result?)?;
            let payload = context.handle_store().resolve(
                &operation,
                operation.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Model(model) = payload.as_ref() else {
                return Err("Real model operation payload".into());
            };
            assert!(model.model_payload().latent_operation_resource().is_some());
            let consumer = registry.node("LatentApplyOperation").ok_or("Consumer")?;
            smol::block_on(consumer.execute(
                context.clone(),
                BTreeMap::from([
                    (
                        "samples".to_owned(),
                        NativeValue::Handle {
                            value: first.clone(),
                        },
                    ),
                    (
                        "operation".to_owned(),
                        NativeValue::Handle { value: operation },
                    ),
                ]),
            ))
        } else {
            result
        };
        if case.get("error").is_some() {
            let error = result.err().ok_or("Expected actual source error")?;
            assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            assert_eq!(generation.len(), count + usize::from(operation));
        } else {
            let handle = output(result.map_err(|error| format!("{index} {name}: {error}"))?)?;
            let payload = context.handle_store().resolve(
                &handle,
                handle.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                return Err("Actual latent output".into());
            };
            compare(latent, &case["output"], &context)
                .map_err(|error| format!("{index} {name}: {error}"))?;
            if case["returns_original"] == true {
                assert_eq!(handle, first);
                assert_eq!(generation.len(), count);
            } else {
                assert_eq!(generation.len(), count + 1 + usize::from(operation));
            }
        }
        assert_eq!(first_payload.digest_sha256(), before, "Source changed");
    }
    Ok(())
}
#[test]
fn task630_schema_unknown_inputs_cancellation_and_budget_failures_are_atomic() -> TestResult {
    let oracle = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, name) in oracle["nodes"]
        .as_array()
        .ok_or("Nodes")?
        .iter()
        .enumerate()
    {
        let name = name.as_str().ok_or("Node")?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("Cases")?
            .iter()
            .find(|c| c["node"] == name)
            .ok_or("Case")?;
        let node = registry.node(name).ok_or("Node")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x630900 + index as u128)));
        let cancellation = CancellationToken::default();
        let context = context(
            backend.clone(),
            &authority,
            store,
            cancellation.clone(),
            256 * 1024,
        )?;
        let mut inputs = parameters(case)?;
        if !name.starts_with("LatentOperation") {
            inputs.insert(
                if matches!(name, "LatentInterpolate" | "LatentSubtract") {
                    "samples1"
                } else {
                    "samples"
                }
                .to_owned(),
                NativeValue::Handle {
                    value: publish(&case["first"], &context)?,
                },
            );
            if let Some(second) = case.get("second") {
                inputs.insert(
                    "samples2".to_owned(),
                    NativeValue::Handle {
                        value: publish(second, &context)?,
                    },
                );
            }
        }
        let count = generation.len();
        let mut invalid = inputs.clone();
        invalid.insert(
            "unassigned".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(1.0),
            },
        );
        assert!(smol::block_on(node.execute(context.clone(), invalid)).is_err());
        assert_eq!(generation.len(), count);
        cancellation.cancel();
        let error = smol::block_on(node.execute(context, inputs))
            .err()
            .ok_or("Cancellation")?;
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    }
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x630999)));
    let first_context = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        256 * 1024,
    )?;
    let source = publish(&oracle["cases"][0]["first"], &first_context)?;
    let limited = context(backend, &authority, store, CancellationToken::default(), 16)?;
    let node = registry.node("LatentMultiply").ok_or("Multiply")?;
    assert!(
        smol::block_on(node.execute(
            limited,
            BTreeMap::from([
                ("samples".to_owned(), NativeValue::Handle { value: source }),
                (
                    "multiplier".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(1.0)
                    }
                )
            ])
        ))
        .is_err()
    );
    assert_eq!(generation.len(), 1);
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
fn task630_cancellation_immediately_after_publication_revokes_exact_output() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let inner = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x630777)));
    let cancellation = CancellationToken::default();
    let store = Arc::new(CancelAfterPublish {
        inner,
        armed: std::sync::atomic::AtomicBool::new(false),
        revoked: std::sync::atomic::AtomicUsize::new(0),
        cancellation: cancellation.clone(),
    });
    let context = context(
        Arc::new(backend),
        &authority,
        store.clone(),
        cancellation,
        256 * 1024,
    )?;
    let source = publish(&fixture()?["cases"][0]["first"], &context)?;
    store.armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let node = registry.node("LatentMultiply").ok_or("Multiply")?;
    let error = smol::block_on(node.execute(
        context,
        BTreeMap::from([
            ("samples".to_owned(), NativeValue::Handle { value: source }),
            (
                "multiplier".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(2.0),
                },
            ),
        ]),
    ))
    .err()
    .ok_or("Cancellation")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(store.revoked.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(generation.len() <= 1);
    Ok(())
}

#[test]
fn task630_exact_feature_claims_and_original_source_aliases() -> TestResult {
    let bindings = (sim_nodes::generated_model_latent_04::NATIVE_NODE_BINDINGS)()?;
    assert_eq!(bindings.len(), 10);
    let oracle = fixture()?;
    for (index, binding) in bindings.iter().enumerate() {
        assert_eq!(
            binding.feature_id(),
            format!("COMFY-NODE-{:04}", 322 + index)
        );
        binding.descriptor().validate_exact_schema_v2()?;
        assert_eq!(
            serde_json::to_value(&binding.presentation().search_aliases)?,
            oracle["aliases"][&binding.descriptor().class_type]
        );
        assert_eq!(
            binding.disposition(),
            sim_nodes::NativeNodeBindingDisposition::Executable
        );
    }
    Ok(())
}
