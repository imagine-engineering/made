use sha2::{Digest, Sha256};
use sim_model::conditioning::{
    ConditioningEntry, ConditioningEntryOptions, ConditioningIdentity,
    ConditioningMetadataKey as Key, ConditioningMetadataMap, ConditioningMetadataValue as Value,
    ConditioningReferences, ConditioningSet, ConditioningValue,
};
use sim_model::generated_native_diffusion::{
    sd15_latent_format_identity, sd15_model_family_identity,
};
use sim_nodes::{
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
    NativeNode, NativeNodeBindingDisposition, NativeNodeComputeSession, NativeNodeContext,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices,
    NativeOpaqueHandle, NativePrimitive, NativeResolvedPayload, NativeStoredPayload, NativeValue,
};
use sim_runtime::{
    CacheKey, NativeHandleStoreGeneration, generated_native_node_registry_projection,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, NativeLatentBundle, NativeLatentMetadata,
    StreamId, TensorBackend,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use uuid::Uuid;

type TestResult = Result<(), Box<dyn Error>>;
const LIMIT: u64 = 8 * 1024 * 1024;
const CLASS: &str = "NormalizeVideoLatentStart";

fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("conditioning-part-six".to_owned());
    let scratch = authority.authorize_workspace(LIMIT / 2)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x62103), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x62101)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn output(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected NormalizeVideoLatentStart value outcome".into());
    };
    assert!(ui.is_none());
    assert!(effects.is_empty());
    let [NativeValue::Handle { value }] = outputs.as_slice() else {
        return Err("Expected sole latent handle".into());
    };
    Ok(value.clone())
}
fn key(
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
        "task621-noop-in-process",
        node.implementation_version(),
        node.cache_change_token(inputs)?,
    )?)
}

fn integer(inputs: &mut BTreeMap<String, NativeValue>, name: &str, value: i64) {
    inputs.insert(
        name.to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        },
    );
}
fn inputs(handle: NativeOpaqueHandle) -> BTreeMap<String, NativeValue> {
    let mut inputs = BTreeMap::from([("latent".to_owned(), NativeValue::Handle { value: handle })]);
    integer(&mut inputs, "start_frame_count", 4);
    integer(&mut inputs, "reference_frame_count", 5);
    inputs
}
fn latent(
    context: &NativeNodeContext,
    shape: &[u64],
    with_dictionary: bool,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or("Shape overflow")?;
    let count = usize::try_from(count)?;
    let values = (0..count)
        .map(|index| {
            if index % 3 == 0 {
                -0.0
            } else {
                index as f32 - 3.5
            }
        })
        .collect::<Vec<_>>();
    let tensor = tensor_from_f32(compute.backend(), shape, &values, &execution)?;
    let mask = if with_dictionary {
        Some(tensor_from_f32(
            compute.backend(),
            &[1, 1, 1, 2, 2],
            &[0.0, 0.25, 0.5, 1.0],
            &execution,
        )?)
    } else {
        None
    };
    let indices = with_dictionary.then_some(vec![73]);
    let bundle = NativeLatentBundle::single(
        tensor,
        mask,
        indices,
        NativeLatentMetadata::default(),
        &execution,
    )?;
    Ok(context.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(bundle)),
        &context.cancellation,
    )?)
}

#[test]
fn task621_normalize_noop_is_the_unique_generated_source_binding() -> TestResult {
    let bindings = sim_nodes::generated_family_node_bindings()?;
    let matches = bindings
        .iter()
        .filter(|binding| binding.descriptor().class_type == CLASS)
        .collect::<Vec<_>>();
    let [binding] = matches.as_slice() else {
        return Err("Expected one canonical node owner".into());
    };
    assert_eq!(binding.feature_id(), "COMFY-NODE-0457");
    assert_eq!(
        binding.disposition(),
        NativeNodeBindingDisposition::Executable
    );
    binding.descriptor().validate_exact_schema_v2()?;
    let registry = generated_native_node_registry_projection(None)?;
    let descriptor = registry
        .descriptor(CLASS)
        .ok_or("Registered descriptor missing")?;
    descriptor.validate_exact_schema_v2()?;
    assert!(registry.node(CLASS).is_some());
    assert_eq!(descriptor.inputs.len(), 3);
    assert!(
        descriptor
            .inputs
            .iter()
            .all(|input| input.required && !input.lazy)
    );
    assert_eq!(descriptor.outputs.len(), 1);
    assert_eq!(
        descriptor.outputs.first().ok_or("Missing output")?.name,
        "latent"
    );
    Ok(())
}

#[test]
fn task621_normalize_noop_keeps_exact_handle_dictionary_bits_and_full_store_capacity() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Registered node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for (index, shape) in [
        vec![1, 2, 1, 2, 2],
        vec![2, 1, 1, 1, 2],
        vec![1, 2, 1],
        vec![1, 2, 1, 2],
    ]
    .into_iter()
    .enumerate()
    {
        let generation = NativeHandleStoreGeneration::with_capacity(1)?;
        let setup = context(
            backend.clone(),
            &authority,
            generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621100 + index as u128))),
            CancellationToken::default(),
        )?;
        let handle = latent(&setup, &shape, index == 0)?;
        let before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        assert_eq!(before.0, 1);
        {
            let original =
                setup
                    .handle_store()
                    .resolve(&handle, handle.handle_type(), &setup.cancellation)?;
            let NativeStoredPayload::Latent(original) = original.as_ref() else {
                return Err("Missing canonical latent".into());
            };
            let compute = setup.compute_session()?;
            let execution = compute.execution_context(&setup)?;
            let original_words = tensor_to_f32(
                compute.backend(),
                original.samples().tensor().ok_or("No tensor")?,
                &execution,
            )?
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
            for (start, reference) in [(1, 1), (4, 5), (100, 100)] {
                let mut values = inputs(handle.clone());
                integer(&mut values, "start_frame_count", start);
                integer(&mut values, "reference_frame_count", reference);
                let returned = output(smol::block_on(node.execute(setup.clone(), values))?)?;
                assert_eq!(returned, handle);
                let resolved = setup.handle_store().resolve(
                    &returned,
                    returned.handle_type(),
                    &setup.cancellation,
                )?;
                let NativeStoredPayload::Latent(returned) = resolved.as_ref() else {
                    return Err("Missing returned canonical latent".into());
                };
                assert!(Arc::ptr_eq(original, returned));
                assert_eq!(
                    tensor_to_f32(
                        compute.backend(),
                        returned.samples().tensor().ok_or("No tensor")?,
                        &execution
                    )?
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                    original_words
                );
                assert_eq!(
                    (
                        generation.len(),
                        generation.resident_bytes(),
                        backend.memory_snapshot().current_bytes
                    ),
                    before
                );
            }
        }
        setup.handle_store().revoke(&handle, &setup.cancellation)?;
        assert_eq!(generation.len(), 0);
        assert_eq!(generation.resident_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
    }
    Ok(())
}

#[test]
fn task621_normalize_noop_rejects_multiframe_schema_wrong_foreign_stale_and_recovers() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Registered node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621201))),
        CancellationToken::default(),
    )?;
    let handle = latent(&setup, &[1, 2, 1, 2, 2], true)?;
    let multi = latent(&setup, &[1, 2, 2, 2, 2], false)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let error = smol::block_on(node.execute(setup.clone(), inputs(multi.clone())))
        .err()
        .ok_or("Multi-frame numerical branch executed")?;
    assert_eq!(error.code, "source_video_normalization_unbound");
    for (name, value) in [("start_frame_count", 0), ("reference_frame_count", -1)] {
        let mut invalid = inputs(handle.clone());
        integer(&mut invalid, name, value);
        let error = smol::block_on(node.execute(setup.clone(), invalid))
            .err()
            .ok_or("Source schema minimum bypassed")?;
        assert_eq!(error.code, "invalid_normalize_video_latent_inputs");
    }
    let mut wrong = inputs(handle.clone());
    wrong.insert(
        "latent".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(1),
        },
    );
    let error = smol::block_on(node.execute(setup.clone(), wrong))
        .err()
        .ok_or("Wrong source type admitted")?;
    assert_eq!(error.code, "invalid_normalize_video_latent_inputs");
    let foreign = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621202))),
        CancellationToken::default(),
    )?;
    let error = smol::block_on(node.execute(foreign, inputs(handle.clone())))
        .err()
        .ok_or("Foreign attempt handle admitted")?;
    assert_eq!(error.code, "normalize_video_latent_execution_failed");
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    setup.handle_store().revoke(&handle, &setup.cancellation)?;
    let error = smol::block_on(node.execute(setup.clone(), inputs(handle)))
        .err()
        .ok_or("Revoked handle admitted")?;
    assert_eq!(error.code, "normalize_video_latent_execution_failed");
    setup.handle_store().revoke(&multi, &setup.cancellation)?;
    let recovered = latent(&setup, &[1, 2, 1, 2, 2], true)?;
    assert_eq!(
        output(smol::block_on(
            node.execute(setup, inputs(recovered.clone()))
        )?)?,
        recovered
    );
    Ok(())
}

#[test]
fn task621_normalize_noop_cancellation_has_no_publication_and_active_recovery_is_exact()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Registered node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(1)?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621301))),
        CancellationToken::default(),
    )?;
    let handle = latent(&setup, &[1, 2, 1, 2, 2], true)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(setup.attempt_id),
        cancelled,
    )?;
    let error = smol::block_on(node.execute(cancelled_context, inputs(handle.clone())))
        .err()
        .ok_or("Cancelled execution succeeded")?;
    assert_eq!(error.code, "execution_interrupted");
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    let without_compute = NativeNodeContext::new(
        setup.prompt_id,
        setup.attempt_id,
        setup.node_id.clone(),
        CancellationToken::default(),
        authority.authorize_workspace(1)?,
        generation.handle_store_for_attempt(setup.attempt_id),
    )?;
    assert_eq!(
        output(smol::block_on(
            node.execute(without_compute, inputs(handle.clone()))
        )?)?,
        handle
    );
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

#[test]
fn task621_normalize_noop_cache_tracks_preserved_latent_and_unused_source_inputs() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Registered node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621401))),
        CancellationToken::default(),
    )?;
    let handle = latent(&setup, &[1, 2, 1, 2, 2], true)?;
    let mut values = inputs(handle.clone());
    let original = key(node.as_ref(), &setup, &values)?;
    assert_eq!(key(node.as_ref(), &setup, &values)?, original);
    assert!(
        node.cache_dependencies(&setup, &values)?
            .artifact_digests
            .contains_key("latent.0")
    );
    integer(&mut values, "start_frame_count", 5);
    assert_ne!(key(node.as_ref(), &setup, &values)?, original);
    integer(&mut values, "start_frame_count", 4);
    integer(&mut values, "reference_frame_count", 6);
    assert_ne!(key(node.as_ref(), &setup, &values)?, original);
    assert_eq!(
        output(smol::block_on(node.execute(setup, values))?)?,
        handle
    );
    Ok(())
}

const LTXV_CLASS: &str = "LTXVConditioning";
fn ltxv_inputs(
    context: &NativeNodeContext,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let identity = ConditioningIdentity::new(
        "part-six-frame-rate-metadata",
        sd15_model_family_identity()?,
        sd15_latent_format_identity()?,
    )?;
    let tensor = tensor_from_f32(compute.backend(), &[1, 1, 2], &[1.0, -0.0], &execution)?;
    let reference = tensor_from_f32(compute.backend(), &[1, 1, 1, 1], &[0.25], &execution)?;
    let mut inputs = BTreeMap::new();
    for (name, count) in [("positive", 2_usize), ("negative", 3_usize)] {
        let entries = (0..count)
            .map(|index| {
                ConditioningEntry::checked_with_metadata(
                    format!("{name}.{index}"),
                    ConditioningValue::regular(tensor.clone())?,
                    ConditioningEntryOptions::default(),
                    ConditioningReferences::default(),
                    ConditioningMetadataMap::checked([
                        (Key::FrameRate, Value::FiniteF64(9.0 + index as f64)),
                        (Key::PooledOutput, Value::PooledOutput(None)),
                        (
                            Key::ReferenceLatents,
                            Value::TensorList(vec![reference.clone()]),
                        ),
                    ])?,
                )
            })
            .collect::<Result<Vec<_>, sim_model::conditioning::ConditioningError>>()?;
        let set = ConditioningSet::checked(identity.clone(), entries, &context.cancellation)?;
        let handle = context.handle_store().publish(
            NativeStoredPayload::Conditioning(Arc::new(set)),
            &context.cancellation,
        )?;
        inputs.insert(name.to_owned(), NativeValue::Handle { value: handle });
    }
    inputs.insert(
        "frame_rate".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Number(25.0),
        },
    );
    Ok(inputs)
}
fn ltxv_outputs(outcome: NativeNodeOutcome) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected real CONDITIONING outputs".into());
    };
    assert!(ui.is_none());
    assert!(effects.is_empty());
    let [
        NativeValue::Handle { value: positive },
        NativeValue::Handle { value: negative },
    ] = outputs.as_slice()
    else {
        return Err("Expected positive then negative CONDITIONING handles".into());
    };
    Ok(vec![positive.clone(), negative.clone()])
}
fn check_ltxv_set(set: &ConditioningSet, frame_rate: f64) -> TestResult {
    for entry in set.entries() {
        let Some(Value::FiniteF64(stored)) = entry.metadata().get(Key::FrameRate) else {
            return Err("Missing canonical FrameRate".into());
        };
        assert_eq!(stored.to_bits(), frame_rate.to_bits());
    }
    Ok(())
}

#[test]
fn task621_ltxv_binding_matches_source_and_updates_both_ordered_dictionaries_without_tensor_copy()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node(LTXV_CLASS)
        .ok_or("Registered LTXV node missing")?;
    let bindings = sim_nodes::generated_family_node_bindings()?;
    let matches = bindings
        .iter()
        .filter(|binding| binding.descriptor().class_type == LTXV_CLASS)
        .collect::<Vec<_>>();
    let [binding] = matches.as_slice() else {
        return Err("LTXV owner collision/missing".into());
    };
    assert_eq!(binding.feature_id(), "COMFY-NODE-0365");
    binding.descriptor().validate_exact_schema_v2()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621501))),
        CancellationToken::default(),
    )?;
    let mut inputs = ltxv_inputs(&setup)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for frame_rate in [-0.0_f64, 0.0, 25.0, 1000.0] {
        inputs.insert(
            "frame_rate".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(frame_rate),
            },
        );
        let handles = ltxv_outputs(smol::block_on(node.execute(setup.clone(), inputs.clone()))?)?;
        for ((name, count), handle) in [("positive", 2), ("negative", 3)].into_iter().zip(&handles)
        {
            let Some(NativeValue::Handle { value: original }) = inputs.get(name) else {
                return Err("Input handle missing".into());
            };
            assert_ne!(original.identifier(), handle.identifier());
            {
                let original = setup.handle_store().resolve(
                    original,
                    original.handle_type(),
                    &setup.cancellation,
                )?;
                let updated = setup.handle_store().resolve(
                    handle,
                    handle.handle_type(),
                    &setup.cancellation,
                )?;
                let (
                    NativeStoredPayload::Conditioning(original),
                    NativeStoredPayload::Conditioning(updated),
                ) = (original.as_ref(), updated.as_ref())
                else {
                    return Err("Canonical sets missing".into());
                };
                assert_eq!(updated.entries().len(), count);
                assert_eq!(updated.identity(), original.identity());
                check_ltxv_set(updated, frame_rate)?;
                for (before, after) in original.entries().iter().zip(updated.entries()) {
                    assert_eq!(before.identifier(), after.identifier());
                    let (
                        ConditioningValue::Regular(before_tensor),
                        ConditioningValue::Regular(after_tensor),
                    ) = (before.value(), after.value())
                    else {
                        return Err("Original value form lost".into());
                    };
                    assert_eq!(before_tensor.storage_id(), after_tensor.storage_id());
                    let (Some(Value::TensorList(before_refs)), Some(Value::TensorList(after_refs))) = (
                        before.metadata().get(Key::ReferenceLatents),
                        after.metadata().get(Key::ReferenceLatents),
                    ) else {
                        return Err("Reference metadata lost".into());
                    };
                    assert_eq!(
                        before_refs
                            .first()
                            .ok_or("Missing original ref")?
                            .storage_id(),
                        after_refs
                            .first()
                            .ok_or("Missing retained ref")?
                            .storage_id()
                    );
                    assert!(matches!(
                        after.metadata().get(Key::PooledOutput),
                        Some(Value::PooledOutput(None))
                    ));
                    let Some(Value::FiniteF64(old_rate)) = before.metadata().get(Key::FrameRate)
                    else {
                        return Err("Original frame rate lost".into());
                    };
                    assert_eq!(
                        old_rate.to_bits(),
                        (9.0 + before
                            .identifier()
                            .rsplit('.')
                            .next()
                            .ok_or("No entry index")?
                            .parse::<f64>()?)
                        .to_bits()
                    );
                }
                assert!(
                    ConditioningSet::combined_resident_bytes(&[original, updated])?
                        < original.resident_bytes()? + updated.resident_bytes()?
                );
            }
            setup.handle_store().revoke(handle, &setup.cancellation)?;
        }
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
    }
    Ok(())
}

#[derive(Debug)]
struct LtxvCancelAfterPublish {
    inner: Arc<dyn NativeHandleStore>,
    remaining: AtomicUsize,
}
impl NativeHandleStore for LtxvCancelAfterPublish {
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
        match self
            .remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            }) {
            Ok(1) => {
                cancellation.cancel();
            }
            Ok(_) | Err(_) => {}
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
fn task621_ltxv_second_output_capacity_and_post_publish_cancellation_rollback_then_recover()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(LTXV_CLASS).ok_or("LTXV node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for (index, mode) in ["capacity", "after_first", "after_second", "before"]
        .into_iter()
        .enumerate()
    {
        let generation =
            NativeHandleStoreGeneration::with_capacity(if mode == "capacity" { 3 } else { 4 })?;
        let store = Arc::new(LtxvCancelAfterPublish {
            inner: generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621600 + index as u128))),
            remaining: AtomicUsize::new(0),
        });
        let setup = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
        )?;
        let inputs = ltxv_inputs(&setup)?;
        let before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        assert_eq!(before.0, 2);
        let cancellation = CancellationToken::default();
        if mode == "before" {
            cancellation.cancel();
        }
        store.remaining.store(
            match mode {
                "after_first" => 1,
                "after_second" => 2,
                _ => 0,
            },
            Ordering::Release,
        );
        let invocation = context(backend.clone(), &authority, store.clone(), cancellation)?;
        let error = smol::block_on(node.execute(invocation.clone(), inputs.clone()))
            .err()
            .ok_or("Rollback boundary succeeded")?;
        assert_eq!(
            error.kind,
            if mode == "capacity" {
                NativeNodeFailureKind::Failure
            } else {
                NativeNodeFailureKind::Interrupted
            }
        );
        assert_eq!(
            error.code,
            if mode == "capacity" {
                "ltxv_conditioning_execution_failed"
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
        if mode != "capacity" {
            store.remaining.store(0, Ordering::Release);
            let handles =
                ltxv_outputs(smol::block_on(node.execute(setup.clone(), inputs.clone()))?)?;
            for handle in handles {
                setup.handle_store().revoke(&handle, &setup.cancellation)?;
            }
            assert_eq!(
                (
                    generation.len(),
                    generation.resident_bytes(),
                    backend.memory_snapshot().current_bytes
                ),
                before
            );
        } else {
            let mut recovered_inputs = inputs.clone();
            let Some(NativeValue::Handle { value: negative }) = recovered_inputs.get("negative")
            else {
                return Err("Negative input missing".into());
            };
            setup.handle_store().revoke(negative, &setup.cancellation)?;
            let positive = recovered_inputs
                .get("positive")
                .ok_or("Positive input missing")?
                .clone();
            recovered_inputs.insert("negative".to_owned(), positive);
            let after_revoke = (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes,
            );
            assert_eq!(after_revoke.0, 1);
            let handles = ltxv_outputs(smol::block_on(
                node.execute(setup.clone(), recovered_inputs),
            )?)?;
            assert_eq!(generation.len(), 3);
            for handle in handles {
                setup.handle_store().revoke(&handle, &setup.cancellation)?;
            }
            assert_eq!(
                (
                    generation.len(),
                    generation.resident_bytes(),
                    backend.memory_snapshot().current_bytes
                ),
                after_revoke
            );
        }
    }
    Ok(())
}

#[test]
fn task621_ltxv_source_float_bounds_foreign_and_stale_inputs_fail_without_publication() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(LTXV_CLASS).ok_or("LTXV node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621701))),
        CancellationToken::default(),
    )?;
    let inputs = ltxv_inputs(&setup)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for rate in [f64::NAN, f64::INFINITY, -0.01, 1000.01] {
        let mut invalid = inputs.clone();
        invalid.insert(
            "frame_rate".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(rate),
            },
        );
        let error = smol::block_on(node.execute(setup.clone(), invalid))
            .err()
            .ok_or("Invalid source FLOAT admitted")?;
        assert_eq!(error.code, "invalid_ltxv_conditioning_inputs");
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
    }
    let wrong = latent(&setup, &[1, 1, 1, 1], false)?;
    let wrong_before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let mut wrong_inputs = inputs.clone();
    wrong_inputs.insert(
        "positive".to_owned(),
        NativeValue::Handle {
            value: wrong.clone(),
        },
    );
    let error = smol::block_on(node.execute(setup.clone(), wrong_inputs))
        .err()
        .ok_or("LATENT admitted as CONDITIONING")?;
    assert_eq!(error.code, "invalid_ltxv_conditioning_inputs");
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        wrong_before
    );
    setup.handle_store().revoke(&wrong, &setup.cancellation)?;
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    let foreign_generation = NativeHandleStoreGeneration::new()?;
    let foreign = context(
        backend.clone(),
        &authority,
        foreign_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621702))),
        CancellationToken::default(),
    )?;
    let error = smol::block_on(node.execute(foreign, inputs.clone()))
        .err()
        .ok_or("Foreign authority admitted")?;
    assert_eq!(error.code, "ltxv_conditioning_execution_failed");
    let Some(NativeValue::Handle { value: negative }) = inputs.get("negative") else {
        return Err("Negative handle missing".into());
    };
    setup.handle_store().revoke(negative, &setup.cancellation)?;
    let after_revoke = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let Some(NativeValue::Handle { value: positive }) = inputs.get("positive") else {
        return Err("Positive handle missing".into());
    };
    let positive = positive.clone();
    let positive_digest = setup
        .handle_store()
        .resolve(&positive, positive.handle_type(), &setup.cancellation)?
        .digest_sha256();
    let error = smol::block_on(node.execute(setup.clone(), inputs))
        .err()
        .ok_or("Stale second input admitted")?;
    assert_eq!(error.code, "ltxv_conditioning_execution_failed");
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        after_revoke
    );
    let retained =
        setup
            .handle_store()
            .resolve(&positive, positive.handle_type(), &setup.cancellation)?;
    assert_eq!(retained.digest_sha256(), positive_digest);
    Ok(())
}

#[test]
fn task621_ltxv_cache_includes_exact_frame_rate_and_unchanged_input_digests() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(LTXV_CLASS).ok_or("LTXV node missing")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621801))),
        CancellationToken::default(),
    )?;
    let mut inputs = ltxv_inputs(&setup)?;
    let first = key(node.as_ref(), &setup, &inputs)?;
    let dependencies = node.cache_dependencies(&setup, &inputs)?;
    assert_eq!(dependencies.artifact_digests.len(), 2);
    assert_eq!(key(node.as_ref(), &setup, &inputs)?, first);
    inputs.insert(
        "frame_rate".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Number(0.0),
        },
    );
    assert_ne!(key(node.as_ref(), &setup, &inputs)?, first);
    let handles = ltxv_outputs(smol::block_on(node.execute(setup.clone(), inputs.clone()))?)?;
    assert_eq!(node.cache_dependencies(&setup, &inputs)?, dependencies);
    for handle in handles {
        setup.handle_store().revoke(&handle, &setup.cancellation)?;
    }
    Ok(())
}

#[test]
fn task621_ltxv_real_mapped_executor_keeps_ordered_rates_cache_leases_and_terminal_recovery()
-> TestResult {
    fn create_executor(
        backend: Arc<CpuBackend>,
    ) -> Result<
        (
            sim_runtime::NativeImageExecutor,
            BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
        ),
        Box<dyn Error>,
    > {
        let mut descriptors = BTreeMap::new();
        let executor = sim_runtime::NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
            sim_types::ProfileId(Uuid::from_u128(0x621910)), BTreeMap::new(), true, backend,
            Arc::new(sim_test_support::NativeDiffusionFixture::checked_in()), |registry| {
                for class_type in ["CheckpointLoaderSimple", "CLIPTextEncode", LTXV_CLASS] {
                    let descriptor = registry.descriptor(class_type).ok_or_else(|| sim_runtime::NativeImageRuntimeError::Registry(format!("Missing actual generated node {class_type}")))?;
                    descriptors.insert(class_type.to_owned(), descriptor.clone());
                }
                Ok(())
            },
        )?;
        Ok((executor, descriptors))
    }
    fn plan(
        descriptors: &BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
    ) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
        use sim_runtime::InputBinding;
        let literal = |value| InputBinding::Literal { value };
        let link = |source: &str, index| InputBinding::Link {
            source: NodeId(source.to_owned()),
            output_index: index,
            lazy: false,
            mode: sim_nodes::NativePortCardinality::Mapped,
        };
        let mut nodes = BTreeMap::new();
        let mut order = Vec::new();
        for (identifier, class_type) in [
            ("checkpoint", "CheckpointLoaderSimple"),
            ("text", "CLIPTextEncode"),
            ("frame-rate", LTXV_CLASS),
        ] {
            let id = NodeId(identifier.to_owned());
            let mut descriptor = descriptors
                .get(class_type)
                .ok_or("Descriptor missing")?
                .clone();
            let inputs = match class_type {
                "CheckpointLoaderSimple" => BTreeMap::from([(
                    "ckpt_name".to_owned(),
                    literal(NativeValue::Primitive {
                        value: NativePrimitive::String("model.safetensors".to_owned()),
                    }),
                )]),
                "CLIPTextEncode" => BTreeMap::from([
                    ("clip".to_owned(), link("checkpoint", 1)),
                    (
                        "text".to_owned(),
                        literal(NativeValue::Primitive {
                            value: NativePrimitive::String("cat".to_owned()),
                        }),
                    ),
                ]),
                _ => {
                    descriptor.output_node = true;
                    BTreeMap::from([
                        ("positive".to_owned(), link("text", 0)),
                        ("negative".to_owned(), link("text", 0)),
                        (
                            "frame_rate".to_owned(),
                            literal(NativeValue::List {
                                values: [0.0_f64, 25.0, 1000.0]
                                    .into_iter()
                                    .map(|rate| NativeValue::Primitive {
                                        value: NativePrimitive::Number(rate),
                                    })
                                    .collect(),
                            }),
                        ),
                    ])
                }
            };
            nodes.insert(
                id.clone(),
                sim_runtime::CompiledNode {
                    id: id.clone(),
                    class_type: class_type.to_owned(),
                    descriptor,
                    inputs,
                    unknown: BTreeMap::new(),
                },
            );
            order.push(id);
        }
        let plan = sim_runtime::CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x621911)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes,
            static_required_nodes: order.iter().cloned().collect(),
            topological_order: order,
            output_nodes: vec![NodeId("frame-rate".to_owned())],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
        Ok(serde_json::from_slice(&serde_json::to_vec(&plan)?)?)
    }
    fn inspect(
        result: &sim_runtime::NativeImageExecutionResult,
        context: &NativeNodeContext,
    ) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
        let columns = result
            .report
            .outputs
            .get(&NodeId("frame-rate".to_owned()))
            .ok_or("Mapped LTXV output missing")?;
        let [
            NativeValue::List { values: positive },
            NativeValue::List { values: negative },
        ] = columns.as_slice()
        else {
            return Err("Expected two real mapped CONDITIONING columns".into());
        };
        assert_eq!(positive.len(), 3);
        assert_eq!(negative.len(), 3);
        let mut handles = Vec::new();
        for column in [positive, negative] {
            for (value, rate) in column.iter().zip([0.0_f64, 25.0, 1000.0]) {
                let NativeValue::Handle { value } = value else {
                    return Err("Expected canonical CONDITIONING handle".into());
                };
                {
                    let retained = context.handle_store().resolve(
                        value,
                        value.handle_type(),
                        &context.cancellation,
                    )?;
                    let NativeStoredPayload::Conditioning(set) = retained.as_ref() else {
                        return Err("Mapped canonical set missing".into());
                    };
                    assert!(!set.entries().is_empty());
                    check_ltxv_set(set, rate)?;
                }
                handles.push(value.clone());
            }
        }
        Ok(handles)
    }
    const MAPPED_LIMIT: u64 = 64 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MAPPED_LIMIT)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = create_executor(backend.clone())?;
    let compiled = plan(&descriptors)?;
    let first = executor.execute_blocking(
        &compiled,
        AttemptId(Uuid::from_u128(0x621912)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(MAPPED_LIMIT / 2)?,
    )?;
    assert_eq!(first.report.cache_hits, 0);
    let generation = executor.handle_store_generation_for_test();
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621912))),
        CancellationToken::default(),
    )?;
    let original = inspect(&first, &setup)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
        executor.cache_len_for_test(),
    );
    let cached = executor.execute_blocking(
        &compiled,
        AttemptId(Uuid::from_u128(0x621913)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(MAPPED_LIMIT / 2)?,
    )?;
    assert!(cached.report.cache_hits > 0);
    assert_eq!(inspect(&cached, &setup)?, original);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
            executor.cache_len_for_test()
        ),
        before
    );
    drop(executor);
    assert!(!generation.is_empty());
    drop(first);
    drop(cached);
    assert!(generation.is_empty());
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    let (recovered, descriptors) = create_executor(backend.clone())?;
    let restored = plan(&descriptors)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        recovered.execute_blocking(
            &restored,
            AttemptId(Uuid::from_u128(0x621914)),
            cancelled,
            0,
            authority.authorize_workspace(MAPPED_LIMIT / 2)?
        ),
        Err(sim_runtime::NativeImageRuntimeError::Cancelled)
    ));
    assert!(recovered.handle_store_generation_for_test().is_empty());
    assert_eq!(recovered.cache_len_for_test(), 0);
    let recovery = recovered.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x621915)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(MAPPED_LIMIT / 2)?,
    )?;
    assert_eq!(recovery.report.cache_hits, 0);
    let recovered_generation = recovered.handle_store_generation_for_test();
    let recovered_context = context(
        backend,
        &authority,
        recovered_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x621915))),
        CancellationToken::default(),
    )?;
    assert_eq!(inspect(&recovery, &recovered_context)?.len(), 6);
    Ok(())
}
