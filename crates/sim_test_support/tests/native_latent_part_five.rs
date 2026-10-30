use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::{
    NativeModelPayload,
    model_family::NativeModelRecipeSourceForm,
    vae::{NativeVaeRecipeResource, load_native_vae_recipe},
};
use sim_nodes::{
    NativeAssetNameList, NativeAssetNameListRequest, NativeAssetNameResolution,
    NativeAssetNameResolutionRequest, NativeAssetReadRequest, NativeAssetReference,
    NativeAssetResolver, NativeAssetServiceError, NativeHandleStore, NativeHandleStoreError,
    NativeHandleStoreIdentity, NativeHandleType, NativeNamedAssetReference,
    NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle, NativePrimitive,
    NativeResolvedAsset, NativeResolvedPayload, NativeStoredModelPayload, NativeStoredPayload,
    NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_sampler::NativeDiffusionPayload;
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, NativeLatentBundle, NativeLatentMetadata,
    NativeLatentNoiseMask, NativeLatentSamples, StreamId, Tensor,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    generated_shape_layout_transform_01::tensor_expand_exact_native,
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use uuid::Uuid;

fn fixture() -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0344/source-contract.json"
    ))?)
}
fn make_context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("latent-five".into());
    let scratch = authority.authorize_workspace(budget)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x63103), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x63101)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}
fn read_tensor(context: &NativeNodeContext, value: &Value) -> Result<Tensor, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let shape = serde_json::from_value::<Vec<u64>>(value["shape"].clone())?;
    let values = serde_json::from_value::<Vec<f32>>(value["values"].clone())?;
    Ok(tensor_from_f32(
        compute.backend(),
        &shape,
        &values,
        &execution,
    )?)
}
fn read_latent(
    context: &NativeNodeContext,
    value: &Value,
) -> Result<NativeLatentBundle, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let samples = if let Some(nested) = value["samples"]["nested"].as_array() {
        NativeLatentSamples::AudioVideo {
            video: read_tensor(context, nested.first().ok_or("Missing video")?)?,
            audio: read_tensor(context, nested.get(1).ok_or("Missing audio")?)?,
        }
    } else {
        NativeLatentSamples::Tensor(read_tensor(context, &value["samples"])?)
    };
    let mask = if let Some(nested) = value["noise_mask"]["nested"].as_array() {
        Some(NativeLatentNoiseMask::AudioVideo {
            video: read_tensor(context, nested.first().ok_or("Missing video mask")?)?,
            audio: read_tensor(context, nested.get(1).ok_or("Missing audio mask")?)?,
        })
    } else if value.get("noise_mask").is_some() {
        Some(NativeLatentNoiseMask::Tensor(read_tensor(
            context,
            &value["noise_mask"],
        )?))
    } else {
        None
    };
    let indices = if let Some(indices) = value.get("batch_index") {
        Some(serde_json::from_value::<Vec<u64>>(indices.clone())?)
    } else {
        None
    };
    let mut metadata = value
        .as_object()
        .ok_or("Latent dictionary required")?
        .clone();
    metadata.remove("samples");
    metadata.remove("noise_mask");
    metadata.remove("batch_index");
    let metadata = serde_json::from_value::<NativeLatentMetadata>(Value::Object(metadata))?;
    let latent = NativeLatentBundle::checked(samples.clone(), mask, None, metadata, &execution)?;
    Ok(latent.replaced_samples_and_indices_source_dictionary(samples, indices, &execution)?)
}
fn input_value(context: &NativeNodeContext, value: &Value) -> Result<NativeValue, Box<dyn Error>> {
    if value.get("samples").is_some() {
        return Ok(NativeValue::Handle {
            value: context.handle_store().publish(
                NativeStoredPayload::Latent(Arc::new(read_latent(context, value)?)),
                &context.cancellation,
            )?,
        });
    }
    if let Some(values) = value.as_array() {
        return Ok(NativeValue::List {
            values: values
                .iter()
                .map(|value| input_value(context, value))
                .collect::<Result<Vec<_>, _>>()?,
        });
    }
    if value.is_null() {
        return Ok(NativeValue::Primitive {
            value: NativePrimitive::Null,
        });
    }
    Ok(NativeValue::Primitive {
        value: NativePrimitive::Integer(value.as_i64().ok_or("Integer input required")?),
    })
}
fn inputs(
    context: &NativeNodeContext,
    case: &Value,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    case["input"]
        .as_object()
        .ok_or("Input dictionary required")?
        .iter()
        .map(|(name, value)| Ok((name.clone(), input_value(context, value)?)))
        .collect()
}
fn encode_tensor(context: &NativeNodeContext, tensor: &Tensor) -> Result<Value, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let values = tensor_to_f32(compute.backend(), tensor, &execution)?;
    Ok(json!({"shape":tensor.descriptor().shape(),"values":&*values}))
}
fn encode_latent(
    context: &NativeNodeContext,
    latent: &NativeLatentBundle,
) -> Result<Value, Box<dyn Error>> {
    let mut value = serde_json::to_value(latent.metadata())?
        .as_object()
        .ok_or("Metadata dictionary required")?
        .clone();
    value.insert(
        "samples".into(),
        match latent.samples() {
            NativeLatentSamples::Tensor(tensor) => encode_tensor(context, tensor)?,
            NativeLatentSamples::AudioVideo { video, audio } => {
                json!({"nested":[encode_tensor(context,video)?,encode_tensor(context,audio)?]})
            }
        },
    );
    if let Some(mask) = latent.noise_mask() {
        value.insert(
            "noise_mask".into(),
            match mask {
                NativeLatentNoiseMask::Tensor(tensor) => encode_tensor(context, tensor)?,
                NativeLatentNoiseMask::AudioVideo { video, audio } => {
                    json!({"nested":[encode_tensor(context,video)?,encode_tensor(context,audio)?]})
                }
            },
        );
    }
    if let Some(indices) = latent.batch_indices() {
        value.insert("batch_index".into(), serde_json::to_value(indices)?);
    }
    latent.validate_retained()?;
    let projection: sim_tensor::NativeLatentBundleProjection =
        serde_json::from_value(serde_json::to_value(latent.projection())?)?;
    assert_eq!(
        projection.semantic_digest_sha256(),
        latent.semantic_digest_sha256()
    );
    Ok(Value::Object(value))
}
fn encode_value(context: &NativeNodeContext, value: &NativeValue) -> Result<Value, Box<dyn Error>> {
    match value {
        NativeValue::List { values } => Ok(Value::Array(
            values
                .iter()
                .map(|value| encode_value(context, value))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        NativeValue::Handle { value } => {
            let payload = context.handle_store().resolve(
                value,
                value.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                return Err("Expected latent payload".into());
            };
            encode_latent(context, latent)
        }
        _ => Err("Expected output handle/list".into()),
    }
}
fn retained_inputs(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<BTreeMap<String, Value>, Box<dyn Error>> {
    inputs
        .iter()
        .filter(|(_, value)| matches!(value, NativeValue::Handle { .. } | NativeValue::List { .. }))
        .filter(|(name, _)| name.as_str() != "batch_size")
        .map(|(name, value)| Ok((name.clone(), encode_value(context, value)?)))
        .collect()
}
#[test]
fn task631_five_registry_nodes_match_all_original_torch_successes() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let cases = fixture["cases"].as_array().ok_or("Missing source cases")?;
    assert_eq!(cases.len(), 94);
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x631100 + index as u128)));
        let context = make_context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let class_type = case["node"].as_str().ok_or("Missing node")?;
        let node = registry.node(class_type).ok_or("Missing production node")?;
        let descriptor = registry
            .descriptor(class_type)
            .ok_or("Missing production descriptor")?;
        descriptor.validate_exact_schema_v2()?;
        let inputs = inputs(&context, case)?;
        let before = retained_inputs(&context, &inputs)?;
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        node.cache_change_token(&inputs)?;
        let dependencies = node.cache_dependencies(&context, &inputs)?;
        let dependency_count = if class_type == "RebatchLatents" {
            case["input"]["latents"]
                .as_array()
                .ok_or("Missing latent list")?
                .len()
        } else {
            inputs
                .values()
                .filter(|value| matches!(value, NativeValue::Handle { .. }))
                .count()
        };
        assert_eq!(dependencies.artifact_digests.len(), dependency_count);
        let result = smol::block_on(node.execute(context.clone(), inputs.clone()));
        let outcome = result.map_err(|error| format!("case {index} {class_type}: {error}"))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("Expected values outcome".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let actual = Value::Array(
            outputs
                .iter()
                .map(|value| encode_value(&context, value))
                .collect::<Result<Vec<_>, _>>()?,
        );
        assert_eq!(actual, case["output"], "source case {index} {class_type}");
        assert_eq!(
            retained_inputs(&context, &inputs)?,
            before,
            "mutated input case {index}"
        );
        for (ordinal, identity) in case["identity"]
            .as_array()
            .ok_or("Missing identity")?
            .iter()
            .enumerate()
        {
            if let Some(name) = identity.as_str() {
                assert_eq!(
                    outputs.get(ordinal),
                    inputs.get(name),
                    "source noop identity case {index}"
                );
            }
        }
    }
    Ok(())
}
#[test]
fn task631_original_errors_fail_before_any_output_publication() -> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let errors = fixture["errors"]
        .as_array()
        .ok_or("Missing source errors")?;
    assert_eq!(errors.len(), 9);
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, case) in errors.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = Arc::new(TrackedStore::new(generation.handle_store_for_attempt(
            AttemptId(Uuid::from_u128(0x631300 + index as u128)),
        )));
        let context = make_context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let inputs = inputs(&context, case)?;
        let before = retained_inputs(&context, &inputs)?;
        let count = store.publications.load(Ordering::Acquire);
        let node = registry
            .node(case["node"].as_str().ok_or("Missing node")?)
            .ok_or("Missing production node")?;
        assert!(
            smol::block_on(node.execute(context.clone(), inputs.clone())).is_err(),
            "expected source error {index}"
        );
        assert_eq!(
            store.publications.load(Ordering::Acquire),
            count,
            "partial publication error {index}"
        );
        assert_eq!(retained_inputs(&context, &inputs)?, before);
    }
    Ok(())
}
#[derive(Debug)]
struct TrackedStore {
    inner: Arc<dyn NativeHandleStore>,
    publications: AtomicUsize,
    reject_at: AtomicUsize,
    cancel_at: AtomicUsize,
    published: Mutex<Vec<NativeOpaqueHandle>>,
}
impl TrackedStore {
    fn new(inner: Arc<dyn NativeHandleStore>) -> Self {
        Self {
            inner,
            publications: AtomicUsize::new(0),
            reject_at: AtomicUsize::new(usize::MAX),
            cancel_at: AtomicUsize::new(usize::MAX),
            published: Mutex::new(Vec::new()),
        }
    }
}
impl NativeHandleStore for TrackedStore {
    fn identity(&self) -> NativeHandleStoreIdentity {
        self.inner.identity()
    }
    fn attempt_id(&self) -> AttemptId {
        self.inner.attempt_id()
    }
    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
        self.inner.resolve(handle, expected, cancellation)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
        let ordinal = self.publications.fetch_add(1, Ordering::AcqRel) + 1;
        if ordinal == self.reject_at.load(Ordering::Acquire) {
            return Err(NativeHandleStoreError::Rejected(
                "injected second output failure".into(),
            ));
        }
        let handle = self.inner.publish(payload, cancellation)?;
        self.published
            .lock()
            .map_err(|_| NativeHandleStoreError::Rejected("Poisoned publication trace".into()))?
            .push(handle.clone());
        if ordinal == self.cancel_at.load(Ordering::Acquire) {
            cancellation.cancel();
        }
        Ok(handle)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)
    }
}
#[test]
fn task631_separate_and_rebatch_roll_back_publication_failure_and_cancellation()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for class_type in ["LTXVSeparateAVLatent", "RebatchLatents"] {
        let case = fixture["cases"]
            .as_array()
            .ok_or("Missing cases")?
            .iter()
            .find(|case| {
                case["node"] == class_type
                    && case["output"].as_array().is_some_and(|output| {
                        class_type != "RebatchLatents"
                            || output
                                .first()
                                .and_then(Value::as_array)
                                .is_some_and(|values| values.len() > 1)
                    })
            })
            .ok_or("Missing multi-output case")?;
        for cancel in [false, true] {
            let generation = NativeHandleStoreGeneration::new()?;
            let store = Arc::new(TrackedStore::new(
                generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x631400))),
            ));
            let cancellation = CancellationToken::default();
            let context = make_context(
                backend.clone(),
                &authority,
                store.clone(),
                cancellation.clone(),
                8 * 1024 * 1024,
            )?;
            let inputs = inputs(&context, case)?;
            let input_count = store.publications.load(Ordering::Acquire);
            if cancel {
                store.cancel_at.store(input_count + 2, Ordering::Release);
            } else {
                store.reject_at.store(input_count + 2, Ordering::Release);
            }
            let error = smol::block_on(
                registry
                    .node(class_type)
                    .ok_or("Missing node")?
                    .execute(context.clone(), inputs),
            )
            .err()
            .ok_or("Expected injected failure")?;
            if cancel {
                assert_eq!(
                    error.kind,
                    NativeNodeFailureKind::Interrupted,
                    "{class_type}: {error:?}"
                );
            }
            let trace = store
                .published
                .lock()
                .map_err(|_| "Poisoned trace")?
                .clone();
            let cleanup = CancellationToken::default();
            for handle in trace.iter().skip(input_count) {
                assert!(matches!(
                    store.resolve(handle, handle.handle_type(), &cleanup),
                    Err(NativeHandleStoreError::Missing(_))
                ));
            }
            for handle in trace.iter().take(input_count) {
                store.resolve(handle, handle.handle_type(), &cleanup)?;
            }
        }
    }
    Ok(())
}
#[test]
fn task631_cancel_budget_and_stale_handles_fail_without_publication() -> Result<(), Box<dyn Error>>
{
    let fixture = fixture()?;
    let case = fixture["cases"]
        .as_array()
        .ok_or("Missing cases")?
        .first()
        .ok_or("Missing repeat case")?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("RepeatLatentBatch").ok_or("Missing node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(TrackedStore::new(
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x631500))),
    ));
    let context = make_context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    let inputs = inputs(&context, case)?;
    let before = store.publications.load(Ordering::Acquire);
    let cancellation = CancellationToken::default();
    assert!(cancellation.cancel());
    let cancelled = make_context(
        backend.clone(),
        &authority,
        store.clone(),
        cancellation,
        8 * 1024 * 1024,
    )?;
    assert_eq!(
        smol::block_on(node.execute(cancelled, inputs.clone()))
            .err()
            .ok_or("Expected cancellation")?
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    let budget = make_context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        1,
    )?;
    assert!(smol::block_on(node.execute(budget, inputs.clone())).is_err());
    let replacement = NativeHandleStoreGeneration::new()?;
    let other_store = Arc::new(TrackedStore::new(
        replacement.handle_store_for_attempt(store.attempt_id()),
    ));
    let stale = make_context(
        backend,
        &authority,
        other_store.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    assert!(smol::block_on(node.execute(stale, inputs)).is_err());
    assert_eq!(other_store.publications.load(Ordering::Acquire), 0);
    assert_eq!(store.publications.load(Ordering::Acquire), before);
    Ok(())
}

#[test]
fn task631_present_null_source_dictionary_keys_remain_explicitly_unsupported()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let gaps = fixture["capability_gaps"]
        .as_array()
        .ok_or("Missing source key-presence cases")?;
    assert_eq!(gaps.len(), 2);
    assert!(
        gaps[0]["output"][0]
            .get("noise_mask")
            .is_some_and(Value::is_null)
    );
    assert!(
        gaps[0]["output"][0]
            .get("sample_rate")
            .is_some_and(Value::is_null)
    );
    assert!(
        gaps[1]["output"][0]
            .get("noise_mask")
            .is_some_and(Value::is_null)
    );
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for case in gaps {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = Arc::new(TrackedStore::new(
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x631600))),
        ));
        let context = make_context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
            8 * 1024 * 1024,
        )?;
        let inputs = case["input"]
            .as_object()
            .ok_or("Missing source inputs")?
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    NativeValue::PreservedUnknown {
                        type_name: "LATENT".into(),
                        value: value.clone(),
                    },
                )
            })
            .collect();
        let node = registry
            .node(case["node"].as_str().ok_or("Missing node")?)
            .ok_or("Missing production node")?;
        assert!(smol::block_on(node.execute(context, inputs)).is_err());
        assert_eq!(store.publications.load(Ordering::Acquire), 0);
    }
    Ok(())
}

fn empty_audio_model_payload(
    context: &NativeNodeContext,
    fixture: &Value,
) -> Result<Arc<NativeModelPayload>, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let model = &fixture["empty_audio_model"];
    assert_eq!(model["state_count"], 296);
    assert_eq!(model["contiguous_f32_bytes"], 435_889_104_u64);
    let mut state = Vec::new();
    for item in model["state"]
        .as_array()
        .ok_or("Missing original source state schema")?
    {
        assert_eq!(item["dtype"], "torch.float32");
        let name = item["name"].as_str().ok_or("Missing state name")?;
        let shape = serde_json::from_value::<Vec<i64>>(item["shape"].clone())?;
        let scalar = if name.ends_with("std-of-means") {
            1.0
        } else {
            0.0
        };
        let scalar = tensor_from_f32(compute.backend(), &[], &[scalar], &execution)?;
        let tensor = tensor_expand_exact_native(&scalar, &shape, &context.cancellation)?;
        state.push((name.to_owned(), tensor));
    }
    let configuration = serde_json::to_string(&model["configuration"])?;
    let source_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(model)?));
    let NativeVaeRecipeResource::Native(vae) = load_native_vae_recipe(
        NativeModelRecipeSourceForm::Standalone,
        &source_sha256,
        state,
        Some(&configuration),
        compute.backend(),
        &execution,
    )?
    else {
        return Err("Expected a concrete original LTX audio VAE".into());
    };
    let physical_state_bytes = compute.backend().memory_snapshot().current_bytes;
    println!(
        "Original 296-state LTX audio VAE constructed with {physical_state_bytes} physical backend bytes"
    );
    assert!(
        physical_state_bytes < 1024 * 1024,
        "Canonical expanded parameter path unexpectedly materialized full physical state"
    );
    Ok(Arc::new(NativeModelPayload::native_vae(Arc::new(vae))?))
}
fn empty_audio_prompt_failure_field<'a>(
    fixture: &'a Value,
    case: &Value,
) -> Result<&'a str, Box<dyn Error>> {
    let prompt = fixture["empty_audio_prompt_validation"]["cases"]
        .as_array()
        .ok_or("Missing original prompt validation")?
        .iter()
        .find(|prompt| prompt["input"] == case["input"])
        .ok_or("Missing exact original prompt case")?;
    assert_eq!(prompt["valid"], false);
    let error = prompt["errors"]
        .as_array()
        .and_then(|errors| errors.first())
        .ok_or("Missing original prompt rejection")?;
    assert_eq!(error["type"], "value_smaller_than_min");
    assert_eq!(error["extra_info"]["input_config"][1]["min"], 1);
    Ok(error["extra_info"]["input_name"]
        .as_str()
        .ok_or("Missing original rejected source field")?)
}
fn empty_audio_inputs(
    context: &NativeNodeContext,
    case: &Value,
    model: Arc<NativeModelPayload>,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let handle = context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
            Arc::new(NativeDiffusionPayload::vae(model)?),
        )?)),
        &context.cancellation,
    )?;
    let mut inputs = inputs(context, case)?;
    inputs.insert("audio_vae".into(), NativeValue::Handle { value: handle });
    Ok(inputs)
}
#[test]
fn task631_empty_audio_registry_uses_checked_vae_and_original_ceiling_geometry()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LTXVEmptyLatentAudio")
        .ok_or("Missing empty audio production binding")?;
    let descriptor = registry
        .descriptor("LTXVEmptyLatentAudio")
        .ok_or("Missing empty audio descriptor")?;
    descriptor.validate_exact_schema_v2()?;
    assert!(
        descriptor
            .inputs
            .iter()
            .all(|input| input.cardinality == sim_nodes::NativePortCardinality::Mapped)
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(TrackedStore::new(
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x631680))),
    ));
    let context = make_context(
        backend,
        &authority,
        store.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    let model = empty_audio_model_payload(&context, &fixture)?;
    let cases = fixture["empty_audio_cases"]
        .as_array()
        .ok_or("Missing actual empty audio cases")?;
    assert_eq!(cases.len(), 8);
    for case in cases {
        let inputs = empty_audio_inputs(&context, case, model.clone())?;
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        node.cache_change_token(&inputs)?;
        assert_eq!(
            node.cache_dependencies(&context, &inputs)?
                .artifact_digests
                .len(),
            1
        );
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = smol::block_on(node.execute(context.clone(), inputs))?
        else {
            return Err("Expected empty audio values".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        assert_eq!(
            Value::Array(
                outputs
                    .iter()
                    .map(|value| encode_value(&context, value))
                    .collect::<Result<Vec<_>, _>>()?
            ),
            case["output"]
        );
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing audio output")? else {
            return Err("Missing latent handle".into());
        };
        let payload =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
            return Err("Expected audio latent payload".into());
        };
        let NativeLatentSamples::Tensor(samples) = latent.samples() else {
            return Err("Expected ordinary audio latent".into());
        };
        assert_eq!(samples.descriptor().dtype(), sim_tensor::DType::F32);
        assert_eq!(case["dtype"], "torch.float32");
    }
    for case in fixture["empty_audio_errors"]
        .as_array()
        .ok_or("Missing original empty audio errors")?
    {
        let inputs = empty_audio_inputs(&context, case, model.clone())?;
        let count = store.publications.load(Ordering::Acquire);
        let error = smol::block_on(node.execute(context.clone(), inputs))
            .err()
            .ok_or("Expected original prompt rejection")?;
        let field = empty_audio_prompt_failure_field(&fixture, case)?;
        assert_eq!(error.code, "invalid_latent_part_five_inputs");
        assert_eq!(error.message, format!("{field} violates its source schema"));
        assert_eq!(error.kind, NativeNodeFailureKind::Failure);
        assert_eq!(store.publications.load(Ordering::Acquire), count);
    }
    for case in fixture["empty_audio_capability_cases"]
        .as_array()
        .ok_or("Missing source capability cases")?
        .iter()
        .take(2)
    {
        let inputs = empty_audio_inputs(&context, case, model.clone())?;
        let count = store.publications.load(Ordering::Acquire);
        let error = smol::block_on(node.execute(context.clone(), inputs))
            .err()
            .ok_or("Expected original prompt rejection before empty body execution")?;
        let field = empty_audio_prompt_failure_field(&fixture, case)?;
        assert_eq!(error.code, "invalid_latent_part_five_inputs");
        assert_eq!(error.message, format!("{field} violates its source schema"));
        assert_eq!(error.kind, NativeNodeFailureKind::Failure);
        assert_eq!(store.publications.load(Ordering::Acquire), count);
    }
    Ok(())
}
#[test]
fn task631_empty_audio_publication_cancellation_budget_and_stale_models_fail_closed()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let case = fixture["empty_audio_cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Missing empty audio source case")?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LTXVEmptyLatentAudio")
        .ok_or("Missing node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(TrackedStore::new(
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x631681))),
    ));
    let context = make_context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    let model = empty_audio_model_payload(&context, &fixture)?;
    let inputs = empty_audio_inputs(&context, case, model)?;
    let count = store.publications.load(Ordering::Acquire);
    store.reject_at.store(count + 1, Ordering::Release);
    assert!(smol::block_on(node.execute(context.clone(), inputs.clone())).is_err());
    store.reject_at.store(usize::MAX, Ordering::Release);
    let (constrained_backend, constrained_authority) = CpuWorkspaceAuthority::create_backend(1)?;
    let constrained = make_context(
        Arc::new(constrained_backend),
        &constrained_authority,
        store.clone(),
        CancellationToken::default(),
        1,
    )?;
    assert!(smol::block_on(node.execute(constrained, inputs.clone())).is_err());
    let foreign_generation = NativeHandleStoreGeneration::new()?;
    let foreign = make_context(
        backend.clone(),
        &authority,
        foreign_generation.handle_store_for_attempt(context.attempt_id),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    assert!(smol::block_on(node.execute(foreign, inputs.clone())).is_err());
    let cancellation = CancellationToken::default();
    let cancelled = make_context(
        backend,
        &authority,
        store.clone(),
        cancellation.clone(),
        8 * 1024 * 1024,
    )?;
    cancellation.cancel();
    let error = smol::block_on(node.execute(cancelled, inputs.clone()))
        .err()
        .ok_or("Expected pre-cancelled failure")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    store.cancel_at.store(
        store.publications.load(Ordering::Acquire) + 1,
        Ordering::Release,
    );
    let error = smol::block_on(node.execute(context, inputs))
        .err()
        .ok_or("Expected late cancellation")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    let trace = store
        .published
        .lock()
        .map_err(|_| "Poisoned trace")?
        .clone();
    let cleanup = CancellationToken::default();
    for handle in trace.iter().skip(count) {
        assert!(matches!(
            store.resolve(handle, handle.handle_type(), &cleanup),
            Err(NativeHandleStoreError::Missing(_))
        ));
    }
    for handle in trace.iter().take(count) {
        store.resolve(handle, handle.handle_type(), &cleanup)?;
    }
    Ok(())
}

#[derive(Debug)]
struct LatentFileAssets {
    identity: NativeNodeServiceIdentity,
    files: Mutex<BTreeMap<String, Arc<[u8]>>>,
    fault: AtomicUsize,
    reads: AtomicUsize,
}
impl LatentFileAssets {
    fn new(attempt: AttemptId, fixture: &Value) -> Result<Self, Box<dyn Error>> {
        let contract = &fixture["load_latent_contract"];
        let mut files = BTreeMap::new();
        for group in ["cases", "errors", "capability_gaps"] {
            for case in contract[group]
                .as_array()
                .ok_or("Missing actual file cases")?
            {
                let name = case["input"]["latent"]
                    .as_str()
                    .ok_or("Missing source file name")?;
                let bytes = serde_json::from_value::<Vec<u8>>(case["encoded_bytes"].clone())?;
                assert_eq!(
                    format!("{:x}", Sha256::digest(&bytes)),
                    case["encoded_sha256"]
                );
                files.insert(name.to_owned(), Arc::from(bytes));
            }
        }
        Ok(Self {
            identity: NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x631344),
                attempt,
                NodeId("latent-five".into()),
            )?,
            files: Mutex::new(files),
            fault: AtomicUsize::new(0),
            reads: AtomicUsize::new(0),
        })
    }
    fn reference(
        &self,
        index: usize,
        bytes: &[u8],
    ) -> Result<NativeAssetReference, NativeAssetServiceError> {
        NativeAssetReference::checked(
            self.identity.service_id(),
            Uuid::from_u128(0x631344000 + index as u128),
            "LATENT",
            u64::try_from(bytes.len()).map_err(|_| NativeAssetServiceError::TooLarge)?,
            format!("{:x}", Sha256::digest(bytes)),
        )
    }
}
impl NativeAssetResolver for LatentFileAssets {
    fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }
    fn list_names(
        &self,
        request: &NativeAssetNameListRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameList, NativeAssetServiceError> {
        cancellation
            .check()
            .map_err(|_| NativeAssetServiceError::Cancelled)?;
        if request.folder_category() != "input" {
            return Err(NativeAssetServiceError::InvalidFolderCategory);
        }
        let names = self
            .files
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?
            .keys()
            .cloned()
            .collect();
        NativeAssetNameList::checked("input", names)
    }
    fn resolve_names(
        &self,
        request: &NativeAssetNameResolutionRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeAssetNameResolution, NativeAssetServiceError> {
        cancellation
            .check()
            .map_err(|_| NativeAssetServiceError::Cancelled)?;
        if request.folder_category() != "input"
            || request.source_type_id() != "LATENT"
            || request.names().len() != 1
        {
            return Err(NativeAssetServiceError::InvalidRequest);
        }
        let name = request
            .names()
            .first()
            .ok_or(NativeAssetServiceError::InvalidRequest)?;
        let files = self
            .files
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?;
        let (index, (_, bytes)) = files
            .iter()
            .enumerate()
            .find(|(_, (candidate, _))| *candidate == name)
            .ok_or(NativeAssetServiceError::Missing)?;
        let reference = self.reference(index, bytes)?;
        let fault = self.fault.load(Ordering::Acquire);
        let reference = if matches!(fault, 1 | 5) {
            NativeAssetReference::checked(
                if fault == 1 {
                    Uuid::from_u128(0x631bad)
                } else {
                    reference.service_id()
                },
                reference.reference_id(),
                if fault == 5 { "IMAGE" } else { "LATENT" },
                reference.byte_length(),
                reference.sha256(),
            )?
        } else {
            reference
        };
        let asset = NativeNamedAssetReference::checked(
            if fault == 2 { "changed.latent" } else { name },
            reference,
        )?;
        let assets = if fault == 4 {
            vec![asset.clone(), asset]
        } else {
            vec![asset]
        };
        NativeAssetNameResolution::checked(if fault == 3 { "output" } else { "input" }, assets)
    }
    fn read_verified(
        &self,
        request: &NativeAssetReadRequest,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedAsset, NativeAssetServiceError> {
        cancellation
            .check()
            .map_err(|_| NativeAssetServiceError::Cancelled)?;
        self.reads.fetch_add(1, Ordering::AcqRel);
        match self.fault.load(Ordering::Acquire) {
            7 => return Err(NativeAssetServiceError::DigestMismatch),
            8 => return Err(NativeAssetServiceError::Cancelled),
            9 => return Err(NativeAssetServiceError::PermissionDenied),
            _ => {}
        }
        let files = self
            .files
            .lock()
            .map_err(|_| NativeAssetServiceError::Rejected)?;
        for (index, bytes) in files.values().enumerate() {
            if bytes.is_empty() {
                continue;
            }
            let reference = self.reference(index, bytes)?;
            if &reference == request.reference() {
                if reference.byte_length() > request.maximum_bytes() {
                    return Err(NativeAssetServiceError::TooLarge);
                }
                let reference = if self.fault.load(Ordering::Acquire) == 6 {
                    NativeAssetReference::checked(
                        reference.service_id(),
                        Uuid::from_u128(0x631badd),
                        "LATENT",
                        reference.byte_length(),
                        reference.sha256(),
                    )?
                } else {
                    reference
                };
                let sha256 = reference.sha256().to_owned();
                let result = NativeResolvedAsset::checked(reference, bytes.clone(), sha256)?;
                if self.fault.load(Ordering::Acquire) == 10 {
                    cancellation.cancel();
                }
                return Ok(result);
            }
        }
        Err(NativeAssetServiceError::InvalidReference)
    }
}
fn latent_file_context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    assets: Arc<dyn NativeAssetResolver>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("latent-five".into());
    let scratch = authority.authorize_workspace(budget)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x63103), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x63101)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(Some(assets), None, Some(compute))?,
    )?)
}
fn file_inputs(name: &str) -> BTreeMap<String, NativeValue> {
    BTreeMap::from([(
        "latent".into(),
        NativeValue::PreservedUnknown {
            type_name: "COMBO".into(),
            value: Value::String(name.into()),
        },
    )])
}
#[test]
fn task631_load_latent_matches_original_bytes_float_and_marker_presence()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LoadLatent")
        .ok_or("Missing LoadLatent binding")?;
    let descriptor = registry
        .descriptor("LoadLatent")
        .ok_or("Missing descriptor")?;
    descriptor.validate_exact_schema_v2()?;
    assert_eq!(
        descriptor.effect,
        sim_nodes::NativeEffectClass::ReadsArtifact
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(TrackedStore::new(
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x6313441))),
    ));
    let assets = Arc::new(LatentFileAssets::new(store.attempt_id(), &fixture)?);
    let context = latent_file_context(
        Arc::new(backend),
        &authority,
        store,
        assets,
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    let ordinary_cases = fixture["load_latent_contract"]["cases"]
        .as_array()
        .ok_or("Missing original load cases")?;
    let float8_cases = fixture["load_latent_contract"]["capability_gaps"]
        .as_array()
        .ok_or("Missing original capability cases")?
        .iter()
        .filter(|case| is_source_float8_case(case));
    let mut successes = 0;
    for (index, case) in ordinary_cases.iter().chain(float8_cases).enumerate() {
        let inputs = file_inputs(
            case["input"]["latent"]
                .as_str()
                .ok_or("Missing source filename")?,
        );
        let submission: sim_types::PromptSubmission =
            serde_json::from_value(json!({"prompt": {"latent": {
                "class_type": "LoadLatent", "inputs": case["input"]
            }}}))?;
        // Input admission runs before the intentionally absent output-node check.
        assert!(
            matches!(
                sim_runtime::PromptCompiler::new(&registry).compile(submission),
                Err(sim_runtime::PromptCompileError::NoOutputNode)
            ),
            "source input admission {index}"
        );
        let source_input = descriptor.inputs.first().ok_or("Missing source input")?;
        assert!(
            source_input
                .accepted_types
                .accepts(inputs.get("latent").ok_or("Missing latent input")?)
        );
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        let dependencies = node.cache_dependencies(&context, &inputs)?;
        assert_eq!(
            dependencies
                .artifact_digests
                .get("latent")
                .map(String::as_str),
            case["change_token"].as_str()
        );
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = smol::block_on(node.execute(context.clone(), inputs))?
        else {
            return Err("Expected source latent output".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        assert_source_loaded_latent(&context, &outputs, &case["output"], index)?;
        assert_eq!(case["dtype"], "torch.float32");
        successes += 1;
    }
    assert_eq!(successes, 26);
    Ok(())
}
fn assert_source_loaded_latent(
    context: &NativeNodeContext,
    outputs: &[NativeValue],
    expected: &Value,
    case_index: usize,
) -> Result<(), Box<dyn Error>> {
    let expected = expected.as_array().ok_or("Missing source output array")?;
    assert_eq!(outputs.len(), expected.len());
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    for (output, expected) in outputs.iter().zip(expected) {
        let NativeValue::Handle { value } = output else {
            return Err("Missing native source latent handle".into());
        };
        let payload =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
            return Err("Missing native source latent payload".into());
        };
        let NativeLatentSamples::Tensor(samples) = latent.samples() else {
            return Err("Unexpected nested load samples".into());
        };
        assert_eq!(samples.descriptor().dtype(), sim_tensor::DType::F32);
        let actual_values = tensor_to_f32(compute.backend(), samples, &execution)?;
        let actual_bits = actual_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let expected_bits =
            serde_json::from_value::<Vec<f32>>(expected["samples"]["values"].clone())?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
        let mut actual = encode_value(context, output)?;
        let mut expected = expected.clone();
        // Decimal JSON parsing can round an exact Torch F32 value in F64;
        // compare the original F32 bits while retaining exact dictionary structure.
        actual["samples"]["values"] = serde_json::to_value(actual_bits)?;
        expected["samples"]["values"] = serde_json::to_value(expected_bits)?;
        assert_eq!(actual, expected, "original load case {case_index}");
    }
    Ok(())
}
fn is_source_float8_case(case: &Value) -> bool {
    matches!(
        case["input"]["latent"].as_str(),
        Some("torch.float8_e4m3fn.latent" | "torch.float8_e5m2.latent")
    )
}
#[test]
fn task631_load_latent_raw_file_digest_controls_effective_cache_identity()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("LoadLatent").ok_or("Missing node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x6313442)));
    let assets = Arc::new(LatentFileAssets::new(store.attempt_id(), &fixture)?);
    let context = latent_file_context(
        Arc::new(backend),
        &authority,
        store,
        assets.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    let cases = fixture["load_latent_contract"]["cases"]
        .as_array()
        .ok_or("Missing cases")?;
    let first = cases.first().ok_or("Missing first")?;
    let second = cases.get(1).ok_or("Missing second")?;
    let name = first["input"]["latent"]
        .as_str()
        .ok_or("Missing source filename")?;
    let inputs = file_inputs(name);
    let key = || -> Result<sim_runtime::CacheKey, Box<dyn Error>> {
        let dependencies = node.cache_dependencies(&context, &inputs)?;
        Ok(sim_runtime::CacheKey::from_inputs(
            "LoadLatent",
            node.implementation_version(),
            &inputs,
            dependencies.artifact_digests,
            "cpu",
            "f32",
            None,
            None,
            "source-load-latent",
            "generated",
            node.cache_change_token(&inputs)?,
        )?)
    };
    let original = key()?;
    assert_eq!(original, key()?);
    assets
        .files
        .lock()
        .map_err(|_| "Poisoned asset files")?
        .insert(
            name.into(),
            Arc::from(serde_json::from_value::<Vec<u8>>(
                second["encoded_bytes"].clone(),
            )?),
        );
    let changed = key()?;
    assert_ne!(original, changed);
    assert_eq!(
        original.artifact_digests,
        vec![(
            "latent".into(),
            first["encoded_sha256"]
                .as_str()
                .ok_or("Missing SHA")?
                .into()
        )]
    );
    assert_eq!(
        changed.artifact_digests,
        vec![(
            "latent".into(),
            second["encoded_sha256"]
                .as_str()
                .ok_or("Missing SHA")?
                .into()
        )]
    );
    Ok(())
}
#[test]
fn task631_load_latent_source_errors_and_canonical_representation_gaps_publish_nothing()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("LoadLatent").ok_or("Missing node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(TrackedStore::new(
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x6313443))),
    ));
    let assets = Arc::new(LatentFileAssets::new(store.attempt_id(), &fixture)?);
    let context = latent_file_context(
        Arc::new(backend),
        &authority,
        store.clone(),
        assets,
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    for group in ["errors", "capability_gaps"] {
        for case in fixture["load_latent_contract"][group]
            .as_array()
            .ok_or("Missing source error/capability cases")?
        {
            if group == "capability_gaps" && is_source_float8_case(case) {
                continue;
            }
            let name = case["input"]["latent"].as_str().ok_or("Missing name")?;
            assert!(
                smol::block_on(node.execute(context.clone(), file_inputs(name))).is_err(),
                "source boundary {name}"
            );
            assert_eq!(store.publications.load(Ordering::Acquire), 0);
        }
    }
    assert!(
        smol::block_on(node.execute(context.clone(), file_inputs("missing-file.latent"))).is_err()
    );
    for name in ["../outside.latent", "/outside.latent", "bad\\name.latent"] {
        assert!(smol::block_on(node.execute(context.clone(), file_inputs(name))).is_err());
    }
    assert_eq!(store.publications.load(Ordering::Acquire), 0);
    Ok(())
}
#[test]
fn task631_load_latent_wrong_asset_domains_cancel_budget_and_publication_fail_closed()
-> Result<(), Box<dyn Error>> {
    let fixture = fixture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("LoadLatent").ok_or("Missing node")?;
    let first = fixture["load_latent_contract"]["cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Missing source case")?;
    let name = first["input"]["latent"].as_str().ok_or("Missing name")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(TrackedStore::new(
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x6313444))),
    ));
    let assets = Arc::new(LatentFileAssets::new(store.attempt_id(), &fixture)?);
    let context = latent_file_context(
        backend.clone(),
        &authority,
        store.clone(),
        assets.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    for value in [
        NativeValue::Primitive {
            value: NativePrimitive::String(name.into()),
        },
        NativeValue::PreservedUnknown {
            type_name: "STRING".into(),
            value: Value::String(name.into()),
        },
        NativeValue::PreservedUnknown {
            type_name: "COMBO".into(),
            value: Value::Bool(true),
        },
        NativeValue::PreservedUnknown {
            type_name: "COMBO".into(),
            value: Value::Null,
        },
    ] {
        let inputs = BTreeMap::from([("latent".into(), value)]);
        assert!(node.demanded_lazy_inputs(&context, &inputs).is_err());
        assert!(node.cache_change_token(&inputs).is_err());
        assert!(node.cache_dependencies(&context, &inputs).is_err());
        assert!(smol::block_on(node.execute(context.clone(), inputs)).is_err());
        assert_eq!(assets.reads.load(Ordering::Acquire), 0);
        assert_eq!(store.publications.load(Ordering::Acquire), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
    }
    let unbound = make_context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    assert!(smol::block_on(node.execute(unbound.clone(), file_inputs(name))).is_err());
    assert!(
        node.cache_dependencies(&unbound, &file_inputs(name))
            .is_err()
    );
    let foreign_assets = Arc::new(LatentFileAssets::new(
        AttemptId(Uuid::from_u128(0x631bad)),
        &fixture,
    )?);
    assert!(
        latent_file_context(
            backend.clone(),
            &authority,
            store.clone(),
            foreign_assets,
            CancellationToken::default(),
            8 * 1024 * 1024,
        )
        .is_err()
    );
    let mut wrong_node_assets = LatentFileAssets::new(store.attempt_id(), &fixture)?;
    wrong_node_assets.identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x631344),
        store.attempt_id(),
        NodeId("other-node".into()),
    )?;
    assert!(
        latent_file_context(
            backend.clone(),
            &authority,
            store.clone(),
            Arc::new(wrong_node_assets),
            CancellationToken::default(),
            8 * 1024 * 1024,
        )
        .is_err()
    );
    for fault in 1..=9 {
        assets.fault.store(fault, Ordering::Release);
        let before = assets.reads.load(Ordering::Acquire);
        let error = smol::block_on(node.execute(context.clone(), file_inputs(name)))
            .err()
            .ok_or("Expected rejected asset authority")?;
        if fault <= 5 {
            assert_eq!(assets.reads.load(Ordering::Acquire), before);
        }
        assert_eq!(
            error.kind,
            if fault == 8 {
                NativeNodeFailureKind::Interrupted
            } else {
                NativeNodeFailureKind::Failure
            }
        );
        assert_eq!(store.publications.load(Ordering::Acquire), 0);
    }
    assets.fault.store(0, Ordering::Release);
    let cancellation = CancellationToken::default();
    let cancelled = latent_file_context(
        backend.clone(),
        &authority,
        store.clone(),
        assets.clone(),
        cancellation.clone(),
        8 * 1024 * 1024,
    )?;
    cancellation.cancel();
    assert_eq!(
        smol::block_on(node.execute(cancelled, file_inputs(name)))
            .err()
            .ok_or("Expected pre cancellation")?
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    let (small_backend, small_authority) = CpuWorkspaceAuthority::create_backend(1)?;
    let small_backend = Arc::new(small_backend);
    let small = latent_file_context(
        small_backend.clone(),
        &small_authority,
        store.clone(),
        assets.clone(),
        CancellationToken::default(),
        1,
    )?;
    assert_eq!(
        smol::block_on(node.execute(small, file_inputs(name)))
            .err()
            .ok_or("Expected actual node allocation failure")?
            .kind,
        NativeNodeFailureKind::Failure
    );
    assert_eq!(small_backend.memory_snapshot().current_bytes, 0);
    store.reject_at.store(1, Ordering::Release);
    assert!(smol::block_on(node.execute(context, file_inputs(name))).is_err());
    store.reject_at.store(usize::MAX, Ordering::Release);
    store.cancel_at.store(
        store.publications.load(Ordering::Acquire) + 1,
        Ordering::Release,
    );
    let late = latent_file_context(
        backend.clone(),
        &authority,
        store.clone(),
        assets.clone(),
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    assert_eq!(
        smol::block_on(node.execute(late, file_inputs(name)))
            .err()
            .ok_or("Expected publication cancellation")?
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    for handle in store.published.lock().map_err(|_| "Poisoned trace")?.iter() {
        assert!(matches!(
            store.resolve(handle, handle.handle_type(), &CancellationToken::default()),
            Err(NativeHandleStoreError::Missing(_))
        ));
    }
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assets.fault.store(10, Ordering::Release);
    let during_read = latent_file_context(
        backend,
        &authority,
        store,
        assets,
        CancellationToken::default(),
        8 * 1024 * 1024,
    )?;
    assert_eq!(
        smol::block_on(node.execute(during_read, file_inputs(name)))
            .err()
            .ok_or("Expected verified read cancellation")?
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    Ok(())
}
