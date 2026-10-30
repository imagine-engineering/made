use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_model::{
    NativeModelPayload,
    clip::NativeTokenizer,
    conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningMetadataKey,
        ConditioningMetadataValue, ConditioningSet, ConditioningTimestepRange, ConditioningValue,
    },
};
use sim_nodes::{
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
    NativeNode, NativeNodeBindingDisposition, NativeNodeComputeSession, NativeNodeContext,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices,
    NativeOpaqueHandle, NativePrimitive, NativeResolvedPayload, NativeStoredModelPayload,
    NativeStoredPayload, NativeValue,
};
use sim_runtime::{
    CacheKey, NativeHandleStoreGeneration, NativeImageExecutor, NativeImageRuntimeError,
    generated_native_node_registry_projection,
};
use sim_sampler::NativeDiffusionPayload;
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, StreamId, Tensor,
    TensorBackend, TensorDescriptor, generated_native_diffusion::tensor_from_f32,
};
use sim_types::{AttemptId, NodeId, ProfileId, PromptId};
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use uuid::Uuid;

type TestResult = Result<(), Box<dyn Error>>;
const LIMIT: u64 = 128 * 1024 * 1024;
const SOURCE_BYTES: &str = include_str!(
    "../fixtures/nodes/model-conditioning-comfy-node-0008/source-body-observations.json"
);

fn source() -> Result<Value, Box<dyn Error>> {
    assert_eq!(
        format!("{:x}", Sha256::digest(SOURCE_BYTES.as_bytes())),
        "0c99462b861116119ec03b058caccd243bcb26425cbcba604e1f7575fdb67b80"
    );
    let source: Value = serde_json::from_str(SOURCE_BYTES)?;
    assert_eq!(
        source["source_sha256"],
        "9a4d0cc1f60bd7e29413c610cd300bd86c2f7426431d81a3546275b3804d8d48"
    );
    assert_eq!(source["full_v3_bootstrap"], false);
    assert_eq!(
        source["cases"]
            .as_array()
            .ok_or("Missing source cases")?
            .len(),
        5
    );
    Ok(source)
}

fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("conditioning-part-one".to_owned());
    let scratch = authority.authorize_workspace(budget)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x61603), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x61601)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn inputs(
    context: &NativeNodeContext,
    backend: Arc<CpuBackend>,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let bundle = sim_test_support::NativeDiffusionFixture::checked_in()
        .load_bundle_with_context(backend, &execution)?;
    let vae = Arc::new(NativeModelPayload::native_vae(bundle.vae().clone())?);
    let vae = context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
            Arc::new(NativeDiffusionPayload::vae(vae)?),
        )?)),
        &context.cancellation,
    )?;
    let mut inputs = BTreeMap::from([("vae".to_owned(), NativeValue::Handle { value: vae })]);
    for (name, values) in [("positive", [1.0, -2.0]), ("negative", [3.0, 4.0])] {
        let tensor = tensor_from_f32(compute.backend(), &[1, 1, 2], &values, &execution)?;
        let conditioning = Arc::new(ConditioningSet::checked(
            bundle.conditioning().identity().clone(),
            vec![ConditioningEntry::checked(
                name,
                ConditioningValue::regular(tensor)?,
                ConditioningEntryOptions::default(),
            )?],
            &context.cancellation,
        )?);
        let handle = context.handle_store().publish(
            NativeStoredPayload::Conditioning(conditioning),
            &context.cancellation,
        )?;
        inputs.insert(name.to_owned(), NativeValue::Handle { value: handle });
    }
    for (name, value) in [
        ("width", 16),
        ("height", 16),
        ("length", 1),
        ("batch_size", 1),
    ] {
        inputs.insert(
            name.to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(value),
            },
        );
    }
    Ok(inputs)
}

fn latent_output(outcome: NativeNodeOutcome) -> Result<Vec<NativeValue>, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected Bernini value output".into());
    };
    assert_eq!(outputs.len(), 3);
    assert!(ui.is_none());
    assert!(effects.is_empty());
    Ok(outputs)
}

#[test]
fn task616_registry_uses_unique_source_descriptors_and_the_existing_clip_owner() -> TestResult {
    let families = sim_nodes::generated_family_node_bindings()?;
    for class_type in ["AudioEncoderEncode", "BerniniConditioning"] {
        let matches = families
            .iter()
            .filter(|binding| binding.descriptor().class_type == class_type)
            .collect::<Vec<_>>();
        let [binding] = matches.as_slice() else {
            return Err("Expected exactly one family binding".into());
        };
        binding.descriptor().validate_exact_schema_v2()?;
        assert_eq!(
            binding.disposition(),
            NativeNodeBindingDisposition::Executable
        );
    }
    assert!(!sim_nodes::GENERATED_FAMILY_DESCRIPTOR_IDS.contains(&"CLIPTextEncode"));
    assert_eq!(
        sim_nodes::GENERATED_DESCRIPTOR_IDS
            .iter()
            .filter(|class_type| **class_type == "CLIPTextEncode")
            .count(),
        1
    );
    let registry = generated_native_node_registry_projection(Some(Arc::new(
        sim_test_support::NativeDiffusionFixture::checked_in(),
    )))?;
    registry.validate_comprehensive_bindings()?;
    for class_type in [
        "AudioEncoderEncode",
        "BerniniConditioning",
        "CLIPTextEncode",
    ] {
        registry
            .descriptor(class_type)
            .ok_or("Missing registry descriptor")?
            .validate_exact_schema_v2()?;
        assert_eq!(
            registry.binding_disposition(class_type),
            Some(NativeNodeBindingDisposition::Executable)
        );
        assert!(registry.node(class_type).is_some());
    }
    Ok(())
}

#[test]
fn task616_original_bernini_cases_keep_handles_and_actual_f32_bits() -> TestResult {
    let source = source()?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("BerniniConditioning")
        .ok_or("Bernini executable missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61611))),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let mut inputs = inputs(&context, backend)?;
    let dependencies = node.cache_dependencies(&context, &inputs)?;
    for name in ["positive.0", "negative.0", "vae.0"] {
        assert!(dependencies.artifact_digests.contains_key(name));
    }
    for case in source["cases"].as_array().ok_or("Missing source cases")? {
        for (name, value) in ["width", "height", "length", "batch_size"]
            .into_iter()
            .zip(case["inputs"].as_array().ok_or("Missing original inputs")?)
        {
            inputs.insert(
                name.to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(
                        value.as_i64().ok_or("Missing source integer")?,
                    ),
                },
            );
        }
        let before = generation.len();
        let outputs = latent_output(smol::block_on(
            node.execute(context.clone(), inputs.clone()),
        )?)?;
        assert_eq!(outputs.first(), inputs.get("positive"));
        assert_eq!(outputs.get(1), inputs.get("negative"));
        let Some(NativeValue::Handle { value: handle }) = outputs.get(2) else {
            return Err("Missing latent handle".into());
        };
        let payload =
            context
                .handle_store()
                .resolve(handle, handle.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
            return Err("Missing canonical latent".into());
        };
        let tensor = latent.samples().tensor().ok_or("Missing actual tensor")?;
        let shape = serde_json::from_value::<Vec<u64>>(case["shape"].clone())?;
        assert_eq!(tensor.descriptor().shape(), shape.as_slice());
        assert_eq!(tensor.descriptor().dtype(), sim_tensor::DType::F32);
        assert_eq!(
            format!("{:x}", Sha256::digest(tensor.contiguous_bytes()?)),
            case["f32_bytes_sha256"]
                .as_str()
                .ok_or("Missing genuine Source bit digest")?
        );
        assert_eq!(generation.len(), before + 1);
        drop(payload);
        context
            .handle_store()
            .revoke(handle, &context.cancellation)?;
        assert_eq!(generation.len(), before);
        assert_eq!(node.cache_dependencies(&context, &inputs)?, dependencies);
    }
    Ok(())
}

#[derive(Debug)]
struct CancelAfterPublish {
    inner: Arc<dyn NativeHandleStore>,
    armed: AtomicBool,
}
impl NativeHandleStore for CancelAfterPublish {
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
        let handle = self.inner.publish(payload, cancellation)?;
        if self.armed.swap(false, Ordering::AcqRel) {
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
fn task616_bernini_cancellation_budget_and_post_publish_failure_are_atomic_and_recoverable()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("BerniniConditioning")
        .ok_or("Bernini executable missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = Arc::new(CancelAfterPublish {
        inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61621))),
        armed: AtomicBool::new(false),
    });
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let inputs = inputs(&setup, backend.clone())?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for mode in ["before_execute", "resident_capacity", "after_publish"] {
        let resident_capacity_guard = if mode == "resident_capacity" {
            let compute = setup.compute_session()?;
            let execution = compute.execution_context(&setup)?;
            let remaining = LIMIT
                .checked_sub(backend.memory_snapshot().current_bytes)
                .ok_or("Resident allocation exceeds backend limit")?;
            let descriptor = TensorDescriptor::contiguous(
                vec![remaining],
                DType::U8,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?;
            Some(backend.allocate(descriptor, &execution)?.0)
        } else {
            None
        };
        let invocation_before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        if mode == "resident_capacity" {
            assert_eq!(invocation_before.2, LIMIT);
        }
        let cancellation = CancellationToken::default();
        if mode == "before_execute" {
            cancellation.cancel();
        }
        store
            .armed
            .store(mode == "after_publish", Ordering::Release);
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation,
            LIMIT / 2,
        )?;
        let result = smol::block_on(node.execute(invocation, inputs.clone()));
        let error = result
            .err()
            .ok_or_else(|| format!("Failure boundary {mode} unexpectedly produced output"))?;
        assert_eq!(
            error.kind,
            if mode == "resident_capacity" {
                NativeNodeFailureKind::Failure
            } else {
                NativeNodeFailureKind::Interrupted
            }
        );
        if mode == "resident_capacity" {
            assert_eq!(error.code, "conditioning_execution_failed");
            assert!(
                error
                    .message
                    .contains("tensor allocation of 256 bytes failed")
            );
            assert!(error.message.contains(&format!(
                "backend limit is {LIMIT} bytes with {LIMIT} bytes already reserved"
            )));
        }
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            invocation_before
        );
        drop(resident_capacity_guard);
        assert_eq!(backend.memory_snapshot().current_bytes, before.2);
    }
    let recovered = context(
        backend,
        &authority,
        store,
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let outputs = latent_output(smol::block_on(
        node.execute(recovered.clone(), inputs.clone()),
    )?)?;
    assert_eq!(outputs.first(), inputs.get("positive"));
    assert_eq!(outputs.get(1), inputs.get("negative"));
    let Some(NativeValue::Handle { value: handle }) = outputs.get(2) else {
        return Err("Missing recovered output".into());
    };
    recovered
        .handle_store()
        .revoke(handle, &recovered.cancellation)?;
    assert_eq!(generation.len(), before.0);
    Ok(())
}

#[test]
fn task616_reference_capability_and_wrong_conditioning_type_fail_before_publication() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("BerniniConditioning")
        .ok_or("Bernini executable missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61631))),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let base_inputs = inputs(&context, backend.clone())?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let image = tensor_from_f32(compute.backend(), &[1, 16, 16, 3], &[0.0; 768], &execution)?;
    let image = context.handle_store().publish(
        NativeStoredPayload::Tensor(Arc::new(sim_tensor::NativeTensorPayload::from_image(
            sim_tensor::NativeTensorRole::Image,
            sim_tensor::ImageTensor::from_tensor(image)?,
        )?)),
        &context.cancellation,
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let mut reference_inputs = base_inputs.clone();
    reference_inputs.insert(
        "source_video".to_owned(),
        NativeValue::Handle { value: image },
    );
    let error = smol::block_on(node.execute(context.clone(), reference_inputs))
        .err()
        .ok_or("Unimplemented context stream unexpectedly succeeded")?;
    assert_eq!(error.code, "bernini_context_latents_unbound");
    assert_eq!(error.kind, NativeNodeFailureKind::Failure);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    let mut wrong_type = base_inputs.clone();
    wrong_type.insert(
        "positive".to_owned(),
        base_inputs.get("vae").ok_or("Missing actual VAE")?.clone(),
    );
    let error = smol::block_on(node.execute(context.clone(), wrong_type))
        .err()
        .ok_or("VAE handle admitted as conditioning")?;
    assert_eq!(error.code, "invalid_conditioning_inputs");
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    Ok(())
}

struct RegisteredClipPipeline {
    executor: NativeImageExecutor,
    checkpoint: Arc<dyn NativeNode>,
    set_layer: Arc<dyn NativeNode>,
    encode_text: Arc<dyn NativeNode>,
    controlnet_text: Arc<dyn NativeNode>,
}

fn registered_clip_pipeline(
    backend: Arc<CpuBackend>,
) -> Result<RegisteredClipPipeline, Box<dyn Error>> {
    let mut selected = None;
    let executor = NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
        ProfileId(Uuid::from_u128(0x61641)),
        BTreeMap::new(),
        true,
        backend,
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in()),
        |registry| {
            let mut nodes = Vec::new();
            for class_type in [
                "CheckpointLoaderSimple",
                "CLIPSetLastLayer",
                "CLIPTextEncode",
                "CLIPTextEncodeControlnet",
            ] {
                let descriptor = registry.descriptor(class_type).ok_or_else(|| {
                    NativeImageRuntimeError::Registry(format!(
                        "Missing actual registered {class_type}"
                    ))
                })?;
                if class_type != "CheckpointLoaderSimple" {
                    descriptor
                        .validate_exact_schema_v2()
                        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
                }
                if registry.binding_disposition(class_type)
                    != Some(NativeNodeBindingDisposition::Executable)
                {
                    return Err(NativeImageRuntimeError::Registry(format!(
                        "Non-executable {class_type}"
                    )));
                }
                nodes.push(registry.node(class_type).ok_or_else(|| {
                    NativeImageRuntimeError::Registry(format!(
                        "Missing actual executable {class_type}"
                    ))
                })?);
            }
            let [checkpoint, set_layer, encode_text, controlnet_text] = nodes.as_slice() else {
                return Err(NativeImageRuntimeError::Registry(
                    "Actual CLIP registry selection is incomplete".to_owned(),
                ));
            };
            selected = Some((
                checkpoint.clone(),
                set_layer.clone(),
                encode_text.clone(),
                controlnet_text.clone(),
            ));
            Ok(())
        },
    )?;
    let (checkpoint, set_layer, encode_text, controlnet_text) =
        selected.ok_or("Missing sole runtime registry selection")?;
    assert_eq!(
        sim_nodes::GENERATED_FAMILY_DESCRIPTOR_IDS
            .iter()
            .filter(|class_type| **class_type == "CLIPSetLastLayer")
            .count(),
        1,
    );
    assert!(!sim_nodes::GENERATED_FAMILY_DESCRIPTOR_IDS.contains(&"CLIPTextEncode"));
    Ok(RegisteredClipPipeline {
        executor,
        checkpoint,
        set_layer,
        encode_text,
        controlnet_text,
    })
}

fn one_clip_handle(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected real CLIP value output".into());
    };
    assert!(ui.is_none());
    assert!(effects.is_empty());
    let [NativeValue::Handle { value }] = outputs.as_slice() else {
        return Err("Expected exactly one actual published handle".into());
    };
    Ok(value.clone())
}

fn one_conditioning_handle(
    outcome: NativeNodeOutcome,
    expected_tokens: &[u32],
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected actual CLIPTextEncode conditioning output".into());
    };
    assert_eq!(ui, Some(serde_json::json!({"tokens": expected_tokens})));
    assert!(effects.is_empty());
    let [NativeValue::Handle { value }] = outputs.as_slice() else {
        return Err("Expected exactly one actual conditioning handle".into());
    };
    Ok(value.clone())
}

fn clip_inputs(clip: &NativeOpaqueHandle, source_layer: i64) -> BTreeMap<String, NativeValue> {
    BTreeMap::from([
        (
            "clip".to_owned(),
            NativeValue::Handle {
                value: clip.clone(),
            },
        ),
        (
            "stop_at_clip_layer".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(source_layer),
            },
        ),
    ])
}

fn encode_inputs(clip: &NativeOpaqueHandle) -> BTreeMap<String, NativeValue> {
    BTreeMap::from([
        (
            "clip".to_owned(),
            NativeValue::Handle {
                value: clip.clone(),
            },
        ),
        (
            "text".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::String("a test".to_owned()),
            },
        ),
    ])
}

fn resolve_clip(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Arc<NativeStoredModelPayload>, Box<dyn Error>> {
    let resolved =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Model(model) = resolved.as_ref() else {
        return Err("Registered CLIP output has no actual canonical model payload".into());
    };
    assert!(model.diffusion().is_some());
    model.validate()?;
    Ok(model.clone())
}

fn registered_checkpoint_handles(
    pipeline: &RegisteredClipPipeline,
    context: &NativeNodeContext,
) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
    let outcome = smol::block_on(pipeline.checkpoint.execute(
        context.clone(),
        BTreeMap::from([(
            "ckpt_name".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::String("model.safetensors".to_owned()),
            },
        )]),
    ))?;
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected real reduced checkpoint outputs".into());
    };
    assert_eq!(outputs.len(), 3);
    assert!(ui.is_some());
    assert!(effects.is_empty());
    outputs
        .into_iter()
        .map(|value| {
            let NativeValue::Handle { value } = value else {
                return Err("Reduced checkpoint output is not a real handle".into());
            };
            Ok(value)
        })
        .collect()
}

fn cross_attention(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Tensor, Box<dyn Error>> {
    let resolved =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Conditioning(conditioning) = resolved.as_ref() else {
        return Err("Shared CLIPTextEncode did not publish canonical conditioning".into());
    };
    let [entry] = conditioning.entries() else {
        return Err("Unexpected actual conditioning entry count".into());
    };
    let ConditioningValue::CrossAttention(tensor) = entry.value() else {
        return Err(
            "Shared CLIPTextEncode did not retain the canonical cross-attention tensor".into(),
        );
    };
    Ok(tensor.clone())
}

fn clip_cache_key(
    node: &dyn NativeNode,
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<CacheKey, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let capability_sha = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(compute.backend().capabilities())?)
    );
    let dependencies = node.cache_dependencies(context, inputs)?;
    Ok(CacheKey::from_inputs(
        node.class_type(),
        node.implementation_version(),
        inputs,
        dependencies.artifact_digests,
        capability_sha,
        "cpu-f32",
        None,
        None,
        "task616-retained-sd1-in-process",
        node.implementation_version(),
        node.cache_change_token(inputs)?,
    )?)
}

#[test]
fn task616_registered_retained_clip_source_layer_consumption_identity_and_reconstruction()
-> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let pipeline = registered_clip_pipeline(backend.clone())?;
    let generation = pipeline.executor.handle_store_generation_for_test();
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61642))),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let mut handles = registered_checkpoint_handles(&pipeline, &context)?;
    let original_handle = handles
        .get(1)
        .ok_or("Actual loader CLIP handle missing")?
        .clone();
    let original = resolve_clip(&context, &original_handle)?;
    let (tokenizer, encoder) = original
        .model_payload()
        .clip()
        .ok_or("Actual retained SD1 backing missing")?;
    let original_identity = original.model_payload().identity().clone();
    let original_storage = encoder.resident_tensor_allocations();
    let original_plan = encoder.plan().digest().to_owned();
    let baseline_len = generation.len();
    let baseline_resident = generation.resident_bytes();
    let before_tensors = backend.memory_snapshot().current_bytes;
    let first = one_clip_handle(smol::block_on(
        pipeline
            .set_layer
            .execute(context.clone(), clip_inputs(&original_handle, -2)),
    )?)?;
    let first_model = resolve_clip(&context, &first)?;
    let overwritten = one_clip_handle(smol::block_on(
        pipeline
            .set_layer
            .execute(context.clone(), clip_inputs(&first, -1)),
    )?)?;
    let independent = one_clip_handle(smol::block_on(
        pipeline
            .set_layer
            .execute(context.clone(), clip_inputs(&original_handle, -1)),
    )?)?;
    let overwritten_model = resolve_clip(&context, &overwritten)?;
    let independent_model = resolve_clip(&context, &independent)?;
    assert_eq!(
        overwritten_model.model_payload().identity(),
        independent_model.model_payload().identity()
    );
    assert_eq!(overwritten.digest_sha256(), independent.digest_sha256());
    assert_ne!(first.digest_sha256(), overwritten.digest_sha256());
    assert_ne!(original_handle.digest_sha256(), first.digest_sha256());
    assert_eq!(generation.len(), baseline_len + 3);
    assert!(generation.resident_bytes() > baseline_resident);
    assert_eq!(backend.memory_snapshot().current_bytes, before_tensors);
    for (model, marker) in [
        (&first_model, -2),
        (&overwritten_model, -1),
        (&independent_model, -1),
    ] {
        assert_eq!(
            model.model_payload().identity().artifact_sha256(),
            original_identity.artifact_sha256()
        );
        assert_eq!(
            model.model_payload().identity().format(),
            original_identity.format()
        );
        let (actual_tokenizer, actual_encoder) = model
            .model_payload()
            .clip()
            .ok_or("Shared SD1 backing lost")?;
        assert!(Arc::ptr_eq(tokenizer, actual_tokenizer));
        assert!(Arc::ptr_eq(encoder, actual_encoder));
        let resource = model
            .model_payload()
            .native_clip_resource()
            .ok_or("Sole CLIP wrapper missing")?;
        assert_eq!(resource.source_layer_override(), Some(marker));
        assert_eq!(resource.resident_tensor_allocations()?, original_storage);
        let old_parts = original.model_payload().resident_parts()?;
        let new_parts = model.model_payload().resident_parts()?;
        assert!(new_parts.tensor_allocations().is_empty());
        for old in old_parts.backing_allocations() {
            assert_eq!(
                new_parts
                    .backing_allocations()
                    .iter()
                    .filter(|new| **new == *old)
                    .count(),
                1
            );
        }
        assert_eq!(
            new_parts.backing_allocations().len(),
            old_parts.backing_allocations().len() + 1
        );
    }
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let resource = overwritten_model
        .model_payload()
        .native_clip_resource()
        .ok_or("Overwrite resource missing")?;
    let reconstructed = NativeModelPayload::native_clip_with_cancellation(
        Arc::new(resource.restart(compute.backend(), &execution)?),
        &context.cancellation,
    )?;
    assert_eq!(
        reconstructed.identity(),
        overwritten_model.model_payload().identity()
    );
    let (reconstructed_tokenizer, reconstructed_encoder) = reconstructed
        .clip()
        .ok_or("In-memory reconstruction lost retained backing")?;
    assert!(Arc::ptr_eq(tokenizer, reconstructed_tokenizer));
    assert!(Arc::ptr_eq(encoder, reconstructed_encoder));
    let batch = tokenizer.tokenize_batch(
        &[vec![sim_model::clip::WeightedText::checked("a test", 1.0)?]],
        &context.cancellation,
    )?;
    let expected_tokens = batch
        .rows()
        .first()
        .ok_or("Actual tokenizer returned no row")?
        .sd1_token_ids()?;
    let direct_original = encoder.execute(&batch, &execution)?;
    let direct_earlier = encoder.execute_with_source_layer(&batch, Some(-2), &execution)?;
    let direct_last = encoder.execute_with_source_layer(&batch, Some(-1), &execution)?;
    let mut actual = Vec::new();
    for (handle, expected) in [
        (&original_handle, direct_original.conditioning()),
        (&first, direct_earlier.conditioning()),
        (&overwritten, direct_last.conditioning()),
    ] {
        let inputs = encode_inputs(handle);
        let output = one_conditioning_handle(
            smol::block_on(pipeline.encode_text.execute(context.clone(), inputs))?,
            &expected_tokens,
        )?;
        let tensor = cross_attention(&context, &output)?;
        assert_eq!(tensor.descriptor(), expected.descriptor());
        assert_eq!(tensor.contiguous_bytes()?, expected.contiguous_bytes()?);
        actual.push(tensor);
        handles.push(output);
    }
    let [original_output, earlier_output, last_output] = actual.as_slice() else {
        return Err("Missing actual three shared consumer tensors".into());
    };
    assert_ne!(
        original_output.contiguous_bytes()?,
        earlier_output.contiguous_bytes()?
    );
    assert_eq!(
        original_output.contiguous_bytes()?,
        last_output.contiguous_bytes()?
    );
    assert_eq!(encoder.plan().digest(), original_plan);
    assert_eq!(original.model_payload().identity(), &original_identity);
    let original_key = clip_cache_key(
        pipeline.encode_text.as_ref(),
        &context,
        &encode_inputs(&original_handle),
    )?;
    let changed_key = clip_cache_key(
        pipeline.encode_text.as_ref(),
        &context,
        &encode_inputs(&first),
    )?;
    assert_eq!(original_key.artifact_digests, changed_key.artifact_digests);
    assert_ne!(original_key.identity()?, changed_key.identity()?);
    handles.extend([first, overwritten, independent]);
    for handle in handles.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &context.cancellation)?;
        assert!(
            context
                .handle_store()
                .resolve(handle, handle.handle_type(), &context.cancellation)
                .is_err()
        );
    }
    assert_eq!(generation.len(), 0);
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn task616_registered_clip_layer_cancellation_memory_and_post_publish_cleanup_recover() -> TestResult
{
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let pipeline = registered_clip_pipeline(backend.clone())?;
    let generation = pipeline.executor.handle_store_generation_for_test();
    let store = Arc::new(CancelAfterPublish {
        inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61643))),
        armed: AtomicBool::new(false),
    });
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let checkpoint_handles = registered_checkpoint_handles(&pipeline, &setup)?;
    let original = checkpoint_handles
        .get(1)
        .ok_or("Actual original CLIP missing")?;
    let inputs = clip_inputs(original, -2);
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for mode in ["before_execute", "one_byte_budget", "after_publish"] {
        let cancellation = CancellationToken::default();
        if mode == "before_execute" {
            cancellation.cancel();
        }
        store
            .armed
            .store(mode == "after_publish", Ordering::Release);
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation,
            if mode == "one_byte_budget" {
                1
            } else {
                LIMIT / 2
            },
        )?;
        let result = smol::block_on(
            pipeline
                .set_layer
                .execute(invocation.clone(), inputs.clone()),
        );
        let error = result
            .err()
            .ok_or("Actual CLIP failure boundary published a success")?;
        assert_eq!(
            error.kind,
            if mode == "one_byte_budget" {
                NativeNodeFailureKind::Failure
            } else {
                NativeNodeFailureKind::Interrupted
            }
        );
        assert_eq!(
            error.code,
            if mode == "one_byte_budget" {
                "conditioning_execution_failed"
            } else {
                "execution_interrupted"
            }
        );
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        assert_eq!(invocation.scratch.in_use_bytes(), 0);
    }
    let recovered = context(
        backend,
        &authority,
        store,
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let changed = one_clip_handle(smol::block_on(
        pipeline.set_layer.execute(recovered.clone(), inputs),
    )?)?;
    let changed_model = resolve_clip(&recovered, &changed)?;
    let (tokenizer, encoder) = changed_model
        .model_payload()
        .clip()
        .ok_or("Recovered actual SD1 backing missing")?;
    let compute = recovered.compute_session()?;
    let execution = compute.execution_context(&recovered)?;
    let batch = tokenizer.tokenize_batch(
        &[vec![sim_model::clip::WeightedText::checked("a test", 1.0)?]],
        &recovered.cancellation,
    )?;
    let expected_tokens = batch
        .rows()
        .first()
        .ok_or("Recovered actual tokenizer returned no row")?
        .sd1_token_ids()?;
    let output = one_conditioning_handle(
        smol::block_on(
            pipeline
                .encode_text
                .execute(recovered.clone(), encode_inputs(&changed)),
        )?,
        &expected_tokens,
    )?;
    let tensor = cross_attention(&recovered, &output)?;
    let expected = encoder.execute_with_source_layer(&batch, Some(-2), &execution)?;
    assert_eq!(
        tensor.contiguous_bytes()?,
        expected.conditioning().contiguous_bytes()?
    );
    for handle in [&output, &changed] {
        recovered
            .handle_store()
            .revoke(handle, &recovered.cancellation)?;
    }
    assert_eq!(generation.len(), before.0);
    assert_eq!(generation.resident_bytes(), before.1);
    for handle in checkpoint_handles.iter().rev() {
        recovered
            .handle_store()
            .revoke(handle, &recovered.cancellation)?;
    }
    assert_eq!(generation.len(), 0);
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(recovered.scratch.in_use_bytes(), 0);
    Ok(())
}

fn controlnet_inputs(
    clip: &NativeOpaqueHandle,
    conditioning: &NativeOpaqueHandle,
    text: &str,
) -> BTreeMap<String, NativeValue> {
    BTreeMap::from([
        (
            "clip".to_owned(),
            NativeValue::Handle {
                value: clip.clone(),
            },
        ),
        (
            "conditioning".to_owned(),
            NativeValue::Handle {
                value: conditioning.clone(),
            },
        ),
        (
            "text".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::String(text.to_owned()),
            },
        ),
    ])
}

fn resolve_conditioning(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Arc<ConditioningSet>, Box<dyn Error>> {
    let value =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Conditioning(value) = value.as_ref() else {
        return Err("Actual conditioning owner missing".into());
    };
    value.validate()?;
    Ok(value.clone())
}

#[test]
fn task616_controlnet_text_copies_metadata_preserves_primary_aliases_and_consumes_retained_layer()
-> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let pipeline = registered_clip_pipeline(backend.clone())?;
    let generation = pipeline.executor.handle_store_generation_for_test();
    let context = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61651))),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let mut handles = registered_checkpoint_handles(&pipeline, &context)?;
    let clip = handles.get(1).ok_or("Original CLIP missing")?.clone();
    let model = resolve_clip(&context, &clip)?;
    let (tokenizer, encoder) = model
        .model_payload()
        .clip()
        .ok_or("Real SD1 owner missing")?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let batch = tokenizer.tokenize_batch(
        &[vec![sim_model::clip::WeightedText::checked("a test", 1.0)?]],
        &context.cancellation,
    )?;
    let tokens = batch
        .rows()
        .first()
        .ok_or("Actual token row missing")?
        .sd1_token_ids()?;
    let primary_handle = one_conditioning_handle(
        smol::block_on(
            pipeline
                .encode_text
                .execute(context.clone(), encode_inputs(&clip)),
        )?,
        &tokens,
    )?;
    let primary = resolve_conditioning(&context, &primary_handle)?;
    let entry = primary
        .entries()
        .first()
        .ok_or("Actual primary conditioning missing")?;
    let mut entries = Vec::new();
    for identifier in ["first", "second"] {
        let mut metadata = entry.metadata().clone();
        metadata.insert(
            ConditioningMetadataKey::TimestepRange,
            ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(0.2, 0.8)?),
        )?;
        metadata.insert(
            ConditioningMetadataKey::CrossAttentionControlnet,
            ConditioningMetadataValue::Tensor(cross_attention(&context, &primary_handle)?),
        )?;
        metadata.insert(
            ConditioningMetadataKey::PooledOutputControlnet,
            ConditioningMetadataValue::ControlnetPooledOutput(None),
        )?;
        entries.push(ConditioningEntry::checked_with_metadata(
            identifier,
            entry.value().clone(),
            entry.options().clone(),
            entry.references().clone(),
            metadata,
        )?);
    }
    let input_set = Arc::new(ConditioningSet::checked(
        primary.identity().clone(),
        entries,
        &context.cancellation,
    )?);
    let input_digest = input_set.digest().to_owned();
    let input_handle = context.handle_store().publish(
        NativeStoredPayload::Conditioning(input_set.clone()),
        &context.cancellation,
    )?;
    let changed_clip = one_clip_handle(smol::block_on(
        pipeline
            .set_layer
            .execute(context.clone(), clip_inputs(&clip, -2)),
    )?)?;
    let text = "(a test:1.25)";
    let segments = sim_model::clip_tokenizer::parse_prompt_weights(text, &context.cancellation)?
        .iter()
        .map(|segment| sim_model::clip::WeightedText::checked(segment.text(), segment.weight()))
        .collect::<Result<Vec<_>, _>>()?;
    let weighted_batch = tokenizer.tokenize_batch(&[segments], &context.cancellation)?;
    let expected = encoder.execute_with_source_layer(&weighted_batch, Some(-2), &execution)?;
    let inputs = controlnet_inputs(&changed_clip, &input_handle, text);
    let output_handle = one_clip_handle(smol::block_on(
        pipeline
            .controlnet_text
            .execute(context.clone(), inputs.clone()),
    )?)?;
    let output = resolve_conditioning(&context, &output_handle)?;
    assert_eq!(input_set.digest(), input_digest);
    assert_eq!(output.identity(), input_set.identity());
    assert_eq!(output.entries().len(), 2);
    let mut encoded_storage = None;
    for (before, after) in input_set.entries().iter().zip(output.entries()) {
        assert_eq!(before.identifier(), after.identifier());
        let (
            ConditioningValue::CrossAttention(before_tensor),
            ConditioningValue::CrossAttention(after_tensor),
        ) = (before.value(), after.value())
        else {
            return Err("Primary conditioning changed kind".into());
        };
        assert_eq!(before_tensor.storage_id(), after_tensor.storage_id());
        assert_eq!(
            before_tensor.contiguous_bytes()?,
            after_tensor.contiguous_bytes()?
        );
        let Some(ConditioningMetadataValue::Tensor(encoded)) = after
            .metadata()
            .get(ConditioningMetadataKey::CrossAttentionControlnet)
        else {
            return Err("Actual Controlnet cross attention missing".into());
        };
        assert_eq!(encoded.descriptor(), expected.conditioning().descriptor());
        assert_eq!(
            encoded.contiguous_bytes()?,
            expected.conditioning().contiguous_bytes()?
        );
        if let Some(previous) = encoded_storage {
            assert_eq!(encoded.storage_id(), previous);
        } else {
            encoded_storage = Some(encoded.storage_id());
        }
        let Some(ConditioningMetadataValue::ControlnetPooledOutput(pooled)) = after
            .metadata()
            .get(ConditioningMetadataKey::PooledOutputControlnet)
        else {
            return Err("Explicit pooled key missing".into());
        };
        match (pooled.as_ref(), expected.pooled()) {
            (Some(actual), Some(expected)) => {
                assert_eq!(actual.descriptor(), expected.descriptor());
                assert_eq!(actual.contiguous_bytes()?, expected.contiguous_bytes()?);
            }
            (None, None) => {}
            _ => return Err("Actual pooled presence differs from canonical encoding".into()),
        }
        assert!(matches!(
            before
                .metadata()
                .get(ConditioningMetadataKey::PooledOutputControlnet),
            Some(ConditioningMetadataValue::ControlnetPooledOutput(None))
        ));
        assert!(
            matches!(after.metadata().get(ConditioningMetadataKey::TimestepRange), Some(ConditioningMetadataValue::TimestepRange(range)) if *range == ConditioningTimestepRange::source(0.2, 0.8)?)
        );
    }
    let parts = output.resident_parts()?;
    let encoded_storage = encoded_storage.ok_or("Actual encoded storage missing")?;
    assert_eq!(
        parts
            .tensor_allocations()
            .iter()
            .filter(|allocation| allocation.storage_id() == encoded_storage)
            .count(),
        1
    );
    let key = clip_cache_key(pipeline.controlnet_text.as_ref(), &context, &inputs)?;
    let different = clip_cache_key(
        pipeline.controlnet_text.as_ref(),
        &context,
        &controlnet_inputs(&changed_clip, &input_handle, "a different test"),
    )?;
    assert_eq!(key.artifact_digests, different.artifact_digests);
    assert_ne!(key.identity()?, different.identity()?);
    let encoded_key = serde_json::to_vec(&key)?;
    let restored_key: CacheKey = serde_json::from_slice(&encoded_key)?;
    assert_eq!(restored_key.identity()?, key.identity()?);
    let mut with_output = inputs;
    with_output.insert(
        "conditioning".to_owned(),
        NativeValue::Handle {
            value: output_handle.clone(),
        },
    );
    let dependencies = pipeline
        .controlnet_text
        .cache_dependencies(&context, &with_output)?;
    assert_eq!(
        dependencies
            .artifact_digests
            .get("conditioning.0")
            .map(String::as_str),
        Some(output.digest())
    );
    handles.extend([primary_handle, input_handle, changed_clip, output_handle]);
    for handle in handles.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &context.cancellation)?;
    }
    assert_eq!(generation.len(), 0);
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn task616_controlnet_text_validation_cancellation_budget_post_publish_failure_and_recovery()
-> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let pipeline = registered_clip_pipeline(backend.clone())?;
    let generation = pipeline.executor.handle_store_generation_for_test();
    let store = Arc::new(CancelAfterPublish {
        inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61652))),
        armed: AtomicBool::new(false),
    });
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let mut handles = registered_checkpoint_handles(&pipeline, &setup)?;
    let clip = handles.get(1).ok_or("Actual CLIP missing")?.clone();
    let model = resolve_clip(&setup, &clip)?;
    let (tokenizer, _) = model
        .model_payload()
        .clip()
        .ok_or("Retained backing missing")?;
    let batch = tokenizer.tokenize_batch(
        &[vec![sim_model::clip::WeightedText::checked("a test", 1.0)?]],
        &setup.cancellation,
    )?;
    let tokens = batch
        .rows()
        .first()
        .ok_or("Token row missing")?
        .sd1_token_ids()?;
    let primary = one_conditioning_handle(
        smol::block_on(
            pipeline
                .encode_text
                .execute(setup.clone(), encode_inputs(&clip)),
        )?,
        &tokens,
    )?;
    let inputs = controlnet_inputs(&clip, &primary, "a test");
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for mode in [
        "wrong_text_type",
        "before_execute",
        "one_byte_budget",
        "after_publish",
    ] {
        let cancellation = CancellationToken::default();
        if mode == "before_execute" {
            cancellation.cancel();
        }
        store
            .armed
            .store(mode == "after_publish", Ordering::Release);
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation,
            if mode == "one_byte_budget" {
                1
            } else {
                LIMIT / 2
            },
        )?;
        let mut inputs = inputs.clone();
        if mode == "wrong_text_type" {
            inputs.insert(
                "text".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(3),
                },
            );
        }
        let error = smol::block_on(pipeline.controlnet_text.execute(invocation.clone(), inputs))
            .err()
            .ok_or("Actual boundary incorrectly succeeded")?;
        assert_eq!(
            error.kind,
            if matches!(mode, "before_execute" | "after_publish") {
                NativeNodeFailureKind::Interrupted
            } else {
                NativeNodeFailureKind::Failure
            }
        );
        assert_eq!(
            error.code,
            match mode {
                "wrong_text_type" => "invalid_conditioning_inputs",
                "one_byte_budget" => "conditioning_execution_failed",
                _ => "execution_interrupted",
            }
        );
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        assert_eq!(invocation.scratch.in_use_bytes(), 0);
    }

    for text in ["embedding:name", "a (embedding:name) b"] {
        let error = smol::block_on(
            pipeline
                .controlnet_text
                .execute(setup.clone(), controlnet_inputs(&clip, &primary, text)),
        )
        .err()
        .ok_or("Textual inversion was silently tokenized as literal text")?;
        assert_eq!(error.code, "controlnet_textual_inversion_unbound");
        assert_eq!(error.kind, NativeNodeFailureKind::Failure);
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
    }
    let long_inputs = controlnet_inputs(&clip, &primary, &"a ".repeat(200));
    let error = smol::block_on(pipeline.controlnet_text.execute(setup, long_inputs))
        .err()
        .ok_or("Source multi-chunk prompt was silently truncated")?;
    assert_eq!(error.code, "controlnet_text_chunking_unbound");
    assert_eq!(error.kind, NativeNodeFailureKind::Failure);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    let recovered = context(
        backend,
        &authority,
        store,
        CancellationToken::default(),
        LIMIT / 2,
    )?;
    let output = one_clip_handle(smol::block_on(
        pipeline.controlnet_text.execute(recovered.clone(), inputs),
    )?)?;
    resolve_conditioning(&recovered, &output)?.validate()?;
    recovered
        .handle_store()
        .revoke(&output, &recovered.cancellation)?;
    assert_eq!(generation.len(), before.0);
    assert_eq!(generation.resident_bytes(), before.1);
    handles.push(primary);
    for handle in handles.iter().rev() {
        recovered
            .handle_store()
            .revoke(handle, &recovered.cancellation)?;
    }
    assert_eq!(generation.len(), 0);
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(recovered.scratch.in_use_bytes(), 0);
    Ok(())
}
