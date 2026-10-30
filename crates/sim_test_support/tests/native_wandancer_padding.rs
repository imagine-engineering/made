use serde_json::{Value, json};
use sim_media::NativeAudioPayload;
use sim_nodes::{
    NativeNodeComputeSession, NativeNodeContext, NativeNodeServiceIdentity, NativeNodeServices,
};
use sim_runtime::{
    NativeHandleStoreGeneration, NativeNodeFailureKind, NativeNodeOutcome, NativeOpaqueHandle,
    NativePrimitive, NativeStoredPayload, NativeValue, generated_native_node_registry_projection,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ImageTensor,
    NativeTensorPayload, NativeTensorRole, StreamId, TensorDescriptor,
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;

const LIMIT: u64 = 64 * 1024 * 1024;
const FIXTURE: &str = include_str!("../fixtures/nodes/image-video-comfy-node-0762/fixture.json");

fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn sim_nodes::NativeHandleStore>,
    cancellation: CancellationToken,
    scratch_bytes: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("padding".to_owned());
    let scratch = authority.authorize_workspace(scratch_bytes)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x61503), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x61501)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

fn input_payloads(
    case: &Value,
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &CancellationToken,
) -> Result<[NativeStoredPayload; 2], Box<dyn Error>> {
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        cancellation,
    );
    let count = case["images"].as_u64().ok_or("image count absent")?;
    let frames = case["audio_frames"].as_u64().ok_or("audio count absent")?;
    let rate = u32::try_from(case["sample_rate"].as_u64().ok_or("sample rate absent")?)?;
    let mut values = Vec::new();
    for frame in 0..count {
        values.extend(std::iter::repeat_n((frame + 1) as f32, 18));
    }
    let image = ImageTensor::from_f32(backend, &execution, count, 2, 3, 3, &values)?;
    let audio_values = (0..frames).map(|index| index as f32).collect::<Vec<_>>();
    let descriptor = TensorDescriptor::contiguous(
        vec![1, 1, frames],
        DType::F32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let (waveform, _) = backend.upload_f32(descriptor, &audio_values, &execution)?;
    Ok([
        NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            image,
        )?)),
        NativeStoredPayload::Audio(Arc::new(NativeAudioPayload::checked(waveform, rate)?)),
    ])
}

fn inputs(
    case: &Value,
    handles: &[NativeOpaqueHandle],
    list: bool,
) -> Result<BTreeMap<String, NativeValue>, Box<dyn Error>> {
    let image = handles.first().ok_or("image absent")?.clone();
    let audio = handles.get(1).ok_or("audio absent")?.clone();
    Ok(BTreeMap::from([
        ("images".to_owned(), NativeValue::Handle { value: image }),
        ("audio".to_owned(), NativeValue::Handle { value: audio }),
        (
            "segment_length".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(
                    case["segment_length"].as_u64().ok_or("length absent")?,
                ),
            },
        ),
        (
            if list {
                "num_segments"
            } else {
                "segment_index"
            }
            .to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(if list {
                    3
                } else {
                    case["segment_index"].as_u64().ok_or("index absent")?
                }),
            },
        ),
    ]))
}

fn output_handles(outcome: NativeNodeOutcome) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        effects,
        ui,
    } = outcome
    else {
        return Err("padding did not return values".into());
    };
    assert!(effects.is_empty() && ui.is_none());
    outputs
        .into_iter()
        .map(|value| match value {
            NativeValue::Handle { value } => Ok(value),
            _ => Err("scalar output handle absent".into()),
        })
        .collect()
}

#[test]
fn task615_native_padding_matches_source_and_recovers_atomically() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let cases = fixture["padding_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("WanDancerPadKeyframes")
        .ok_or("padding node absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let mut reviewed = Vec::new();
    for case in cases {
        let generation = NativeHandleStoreGeneration::with_capacities(16, usize::try_from(LIMIT)?)?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61502)));
        let payloads = input_payloads(case, &backend, &authority, &cancellation)?;
        let handles = payloads
            .iter()
            .cloned()
            .map(|payload| store.publish(payload, &cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        let values = inputs(case, &handles, false)?;
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?;
        assert!(node.demanded_lazy_inputs(&invocation, &values)?.is_empty());
        let dependencies = node.cache_dependencies(&invocation, &values)?;
        assert_eq!(
            dependencies.artifact_digests.get("images"),
            Some(&payloads[0].digest_sha256())
        );
        assert_eq!(
            dependencies.artifact_digests.get("audio"),
            Some(&payloads[1].digest_sha256())
        );
        let outputs = output_handles(smol::block_on(node.execute(invocation, values.clone()))?)?;
        assert_eq!(outputs.len(), 3);
        let image = store.resolve(&outputs[0], outputs[0].handle_type(), &cancellation)?;
        let mask = store.resolve(&outputs[1], outputs[1].handle_type(), &cancellation)?;
        let audio = store.resolve(&outputs[2], outputs[2].handle_type(), &cancellation)?;
        let NativeStoredPayload::Tensor(image) = image.as_ref() else {
            return Err("image absent".into());
        };
        let NativeStoredPayload::Tensor(mask) = mask.as_ref() else {
            return Err("mask absent".into());
        };
        let NativeStoredPayload::Audio(audio) = audio.as_ref() else {
            return Err("audio absent".into());
        };
        let image_values = image
            .image()
            .ok_or("ImageTensor absent")?
            .as_f32_slice()?
            .chunks_exact(18)
            .map(|frame| f64::from(frame[0]))
            .collect::<Vec<_>>();
        let mask_values = (0..case["segment_length"].as_u64().ok_or("length absent")?)
            .map(|frame| {
                Ok::<f64, Box<dyn Error>>(f64::from(f32::from_le_bytes(
                    mask.tensor().element_bytes(&[frame, 0, 0])?.try_into()?,
                )))
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            serde_json::to_value(&image_values)?,
            case["keyframe_values"],
            "{case}"
        );
        assert_eq!(
            serde_json::to_value(&mask_values)?,
            case["mask_values"],
            "{case}"
        );
        let frames = audio.waveform().descriptor().shape()[2];
        assert_eq!(Some(frames), case["audio_length"].as_u64());
        if frames > 0 {
            for (frame, key) in [(0, "audio_first"), (frames - 1, "audio_last")] {
                let value =
                    f32::from_le_bytes(audio.waveform().element_bytes(&[0, 0, frame])?.try_into()?);
                assert_eq!(Some(f64::from(value)), case[key].as_f64());
            }
        }
        reviewed.push(json!({"case": case, "actual_keyframes": image_values, "actual_mask": mask_values, "actual_audio_length": frames}));
        for (handle, payload) in handles.iter().zip(&payloads) {
            assert_eq!(
                store
                    .resolve(handle, handle.handle_type(), &cancellation)?
                    .digest_sha256(),
                payload.digest_sha256()
            );
        }
        let serialized = serde_json::to_vec(&values)?;
        let restored: BTreeMap<String, NativeValue> = serde_json::from_slice(&serialized)?;
        let recovery = generation.fresh_generation()?;
        let recovered_store = recovery.handle_store_for_attempt(store.attempt_id());
        assert!(
            smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    recovered_store.clone(),
                    cancellation.clone(),
                    LIMIT
                )?,
                restored
            ))
            .is_err()
        );
        assert!(recovery.is_empty());
        let new_handles = payloads
            .iter()
            .cloned()
            .map(|payload| recovered_store.publish(payload, &cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        let recovered = output_handles(smol::block_on(node.execute(
            context(
                backend.clone(),
                &authority,
                recovered_store,
                cancellation.clone(),
                LIMIT,
            )?,
            inputs(case, &new_handles, false)?,
        ))?)?;
        for (original, recovered) in outputs.iter().zip(&recovered) {
            assert_eq!(original.digest_sha256(), recovered.digest_sha256());
        }
        let before = generation.len();
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let error = smol::block_on(node.execute(
            context(backend.clone(), &authority, store.clone(), cancelled, LIMIT)?,
            values.clone(),
        ))
        .expect_err("cancelled node");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(generation.len(), before);
        assert!(
            smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    store.clone(),
                    cancellation.clone(),
                    1
                )?,
                values.clone()
            ))
            .is_err()
        );
        assert_eq!(generation.len(), before);
        store.revoke(&handles[0], &cancellation)?;
        assert!(
            smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    store,
                    cancellation.clone(),
                    LIMIT
                )?,
                values
            ))
            .is_err()
        );
    }
    std::fs::write(
        std::env::temp_dir().join("sim-wandancer-padding-native-review.json"),
        serde_json::to_vec_pretty(&reviewed)?,
    )?;
    Ok(())
}

#[test]
fn task615_padding_lists_match_scalar_segments_and_rollback_every_output()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let case = fixture["padding_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .get(4)
        .ok_or("case absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let scalar = registry
        .node("WanDancerPadKeyframes")
        .ok_or("scalar node absent")?;
    let list = registry
        .node("WanDancerPadKeyframesList")
        .ok_or("list node absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let payloads = input_payloads(case, &backend, &authority, &cancellation)?;
    let generation = NativeHandleStoreGeneration::with_capacities(64, usize::try_from(LIMIT)?)?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61504)));
    let handles = payloads
        .iter()
        .cloned()
        .map(|payload| store.publish(payload, &cancellation))
        .collect::<Result<Vec<_>, _>>()?;
    let list_inputs = inputs(case, &handles, true)?;
    let NativeNodeOutcome::Values {
        outputs,
        effects,
        ui,
    } = smol::block_on(list.execute(
        context(
            backend.clone(),
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?,
        list_inputs.clone(),
    ))?
    else {
        return Err("list outputs absent".into());
    };
    assert!(effects.is_empty() && ui.is_none());
    assert_eq!(outputs.len(), 3);
    for column in &outputs {
        let NativeValue::List { values } = column else {
            return Err("list column absent".into());
        };
        assert_eq!(values.len(), 3);
    }
    for segment_index in 0..3 {
        let mut segment = case.clone();
        segment["segment_index"] = json!(segment_index);
        let expected = output_handles(smol::block_on(scalar.execute(
            context(
                backend.clone(),
                &authority,
                store.clone(),
                cancellation.clone(),
                LIMIT,
            )?,
            inputs(&segment, &handles, false)?,
        ))?)?;
        for (column, expected) in outputs.iter().zip(expected) {
            let NativeValue::List { values } = column else {
                return Err("list column absent".into());
            };
            let Some(NativeValue::Handle { value }) = values.get(segment_index) else {
                return Err("list element absent".into());
            };
            assert_eq!(value.digest_sha256(), expected.digest_sha256());
        }
    }
    let restored: BTreeMap<String, NativeValue> =
        serde_json::from_slice(&serde_json::to_vec(&list_inputs)?)?;
    let recovery = generation.fresh_generation()?;
    let recovered_store = recovery.handle_store_for_attempt(store.attempt_id());
    assert!(
        smol::block_on(list.execute(
            context(
                backend.clone(),
                &authority,
                recovered_store.clone(),
                cancellation.clone(),
                LIMIT
            )?,
            restored
        ))
        .is_err()
    );
    assert!(recovery.is_empty());
    let recovered_handles = payloads
        .iter()
        .cloned()
        .map(|payload| recovered_store.publish(payload, &cancellation))
        .collect::<Result<Vec<_>, _>>()?;
    let NativeNodeOutcome::Values {
        outputs: recovered_outputs,
        ..
    } = smol::block_on(list.execute(
        context(
            backend.clone(),
            &authority,
            recovered_store,
            cancellation.clone(),
            LIMIT,
        )?,
        inputs(case, &recovered_handles, true)?,
    ))?
    else {
        return Err("recovered list absent".into());
    };
    assert_eq!(recovered_outputs.len(), outputs.len());
    for (original, recovered) in outputs.iter().zip(&recovered_outputs) {
        let (NativeValue::List { values: original }, NativeValue::List { values: recovered }) =
            (original, recovered)
        else {
            return Err("recovered list column absent".into());
        };
        assert_eq!(original.len(), recovered.len());
        for (original, recovered) in original.iter().zip(recovered) {
            let (NativeValue::Handle { value: original }, NativeValue::Handle { value: recovered }) =
                (original, recovered)
            else {
                return Err("recovered list handle absent".into());
            };
            assert_eq!(original.digest_sha256(), recovered.digest_sha256());
        }
    }
    for (node, is_list, capacity) in [(scalar, false, 4), (list.clone(), true, 6)] {
        let tight =
            NativeHandleStoreGeneration::with_capacities(capacity, usize::try_from(LIMIT)?)?;
        let tight_store = tight.handle_store_for_attempt(store.attempt_id());
        let tight_handles = payloads
            .iter()
            .cloned()
            .map(|payload| tight_store.publish(payload, &cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        assert!(
            smol::block_on(node.execute(
                context(
                    backend.clone(),
                    &authority,
                    tight_store.clone(),
                    cancellation.clone(),
                    LIMIT
                )?,
                inputs(case, &tight_handles, is_list)?
            ))
            .is_err()
        );
        assert_eq!(tight.len(), 2);
        for (handle, payload) in tight_handles.iter().zip(&payloads) {
            assert_eq!(
                tight_store
                    .resolve(handle, handle.handle_type(), &cancellation)?
                    .digest_sha256(),
                payload.digest_sha256()
            );
        }
    }
    let before = generation.len();
    for (name, value) in [
        (
            "segment_length",
            NativeValue::Primitive {
                value: NativePrimitive::Integer(0),
            },
        ),
        (
            "segment_length",
            NativeValue::Primitive {
                value: NativePrimitive::Integer(10001),
            },
        ),
        (
            "num_segments",
            NativeValue::Primitive {
                value: NativePrimitive::Integer(0),
            },
        ),
        (
            "num_segments",
            NativeValue::Primitive {
                value: NativePrimitive::Integer(101),
            },
        ),
        (
            "num_segments",
            NativeValue::Primitive {
                value: NativePrimitive::Number(2.0),
            },
        ),
        ("images", NativeValue::List { values: Vec::new() }),
        (
            "audio",
            NativeValue::Handle {
                value: handles[0].clone(),
            },
        ),
        (
            "unexpected",
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(false),
            },
        ),
    ] {
        let mut invalid = list_inputs.clone();
        invalid.insert(name.to_owned(), value);
        assert!(list.cache_change_token(&invalid).is_err());
        assert!(
            smol::block_on(list.execute(
                context(
                    backend.clone(),
                    &authority,
                    store.clone(),
                    cancellation.clone(),
                    LIMIT
                )?,
                invalid
            ))
            .is_err()
        );
    }
    for name in ["images", "audio", "segment_length", "num_segments"] {
        let mut missing = list_inputs.clone();
        missing.remove(name);
        assert!(list.cache_change_token(&missing).is_err());
    }
    assert_eq!(generation.len(), before);
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
fn task615_late_cancellation_revokes_complete_scalar_and_list_results() -> Result<(), Box<dyn Error>>
{
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let case = fixture["padding_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .get(4)
        .ok_or("case absent")?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for (class_type, list, output_count) in [
        ("WanDancerPadKeyframes", false, 3),
        ("WanDancerPadKeyframesList", true, 9),
    ] {
        let cancellation = CancellationToken::default();
        let generation = NativeHandleStoreGeneration::with_capacities(16, usize::try_from(LIMIT)?)?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61505)));
        let payloads = input_payloads(case, &backend, &authority, &cancellation)?;
        let handles = payloads
            .iter()
            .cloned()
            .map(|payload| store.publish(payload, &cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        let cancelling = Arc::new(CancelOnPublication {
            inner: store.clone(),
            remaining: std::sync::atomic::AtomicUsize::new(output_count),
        });
        let node = registry.node(class_type).ok_or("node absent")?;
        let error = smol::block_on(node.execute(
            context(backend.clone(), &authority, cancelling, cancellation, LIMIT)?,
            inputs(case, &handles, list)?,
        ))
        .expect_err("late cancellation");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(generation.len(), 2);
        for (handle, payload) in handles.iter().zip(&payloads) {
            assert_eq!(
                store
                    .resolve(handle, handle.handle_type(), &CancellationToken::default())?
                    .digest_sha256(),
                payload.digest_sha256()
            );
        }
    }
    Ok(())
}

#[test]
fn task615_padding_executes_through_native_engine_with_spatial_pixels_and_multichannel_audio()
-> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    use sim_types::ProfileId;
    use std::collections::BTreeSet;
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let case = fixture["padding_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .get(15)
        .ok_or("duplicate placement case absent")?;
    assert_eq!(case["images"], 100);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(LIMIT)?,
        &cancellation,
    );
    let pixels = (0..100)
        .flat_map(|frame| (0..18).map(move |offset| (frame * 100 + offset) as f32))
        .collect::<Vec<_>>();
    let image = ImageTensor::from_f32(&backend, &execution, 100, 2, 3, 3, &pixels)?;
    let samples = (0..4)
        .flat_map(|channel| (0..100).map(move |sample| (channel * 1000 + sample) as f32))
        .collect::<Vec<_>>();
    let (waveform, _) = backend.upload_f32(
        TensorDescriptor::contiguous(
            vec![2, 2, 100],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        &samples,
        &execution,
    )?;
    let payloads = [
        NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
            NativeTensorRole::Image,
            image,
        )?)),
        NativeStoredPayload::Audio(Arc::new(NativeAudioPayload::checked(waveform, 100)?)),
    ];
    let registry = generated_native_node_registry_projection(None)?;
    for (class_type, list) in [
        ("WanDancerPadKeyframes", false),
        ("WanDancerPadKeyframesList", true),
    ] {
        let executor = NativeImageExecutor::new_with_generated_registry(
            ProfileId(Uuid::from_u128(0x61506)),
            BTreeMap::new(),
            true,
            backend.clone(),
        )?;
        let attempt = AttemptId(Uuid::from_u128(0x61507));
        let store = executor
            .handle_store_generation_for_test()
            .handle_store_for_attempt(attempt);
        let handles = payloads
            .iter()
            .cloned()
            .map(|payload| store.publish(payload, &cancellation))
            .collect::<Result<Vec<_>, _>>()?;
        let mut descriptor = registry
            .descriptor(class_type)
            .ok_or("descriptor absent")?
            .clone();
        descriptor.output_node = true;
        let node_id = NodeId("padding".to_owned());
        let plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x61508)),
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
                    inputs: inputs(case, &handles, list)?
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
        let outputs = result
            .report
            .outputs
            .get(&node_id)
            .ok_or("engine outputs absent")?;
        let mut first_segment = Vec::new();
        for column in outputs {
            let column = if list {
                let NativeValue::List { values } = column else {
                    return Err("engine list absent".into());
                };
                assert_eq!(values.len(), 3);
                values.first().ok_or("first segment absent")?
            } else {
                column
            };
            let NativeValue::Handle { value } = column else {
                return Err("engine handle absent".into());
            };
            first_segment.push(store.resolve(value, value.handle_type(), &cancellation)?);
        }
        let NativeStoredPayload::Tensor(image) =
            first_segment.first().ok_or("image absent")?.as_ref()
        else {
            return Err("image payload absent".into());
        };
        let actual = image.image().ok_or("ImageTensor absent")?.as_f32_slice()?;
        for (frame, expected) in actual.chunks_exact(18).zip(
            case["keyframe_values"]
                .as_array()
                .ok_or("keyframes absent")?,
        ) {
            let source = expected.as_f64().ok_or("source absent")? as u64;
            for (offset, value) in frame.iter().enumerate() {
                assert_eq!(
                    *value,
                    if source == 0 {
                        0.0
                    } else {
                        ((source - 1) * 100 + offset as u64) as f32
                    }
                );
            }
        }
        let NativeStoredPayload::Audio(audio) =
            first_segment.get(2).ok_or("audio absent")?.as_ref()
        else {
            return Err("audio payload absent".into());
        };
        assert_eq!(audio.waveform().descriptor().shape(), &[2, 2, 16]);
        for batch in 0..2 {
            for channel in 0..2 {
                for sample in 0..16 {
                    let value = f32::from_le_bytes(
                        audio
                            .waveform()
                            .element_bytes(&[batch, channel, sample])?
                            .try_into()?,
                    );
                    assert_eq!(value, ((batch * 2 + channel) * 1000 + sample) as f32);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn task615_padding_list_accepts_the_source_segment_limit() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(FIXTURE)?;
    let case = fixture["padding_reference"]["cases"]
        .as_array()
        .ok_or("cases absent")?
        .get(9)
        .ok_or("single-frame case absent")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let cancellation = CancellationToken::default();
    let generation = NativeHandleStoreGeneration::with_capacities(512, usize::try_from(LIMIT)?)?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61509)));
    let payloads = input_payloads(case, &backend, &authority, &cancellation)?;
    let handles = payloads
        .into_iter()
        .map(|payload| store.publish(payload, &cancellation))
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = inputs(case, &handles, true)?;
    values.insert(
        "num_segments".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(100),
        },
    );
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("WanDancerPadKeyframesList")
        .ok_or("list node absent")?;
    let NativeNodeOutcome::Values { outputs, .. } = smol::block_on(node.execute(
        context(
            backend,
            &authority,
            store.clone(),
            cancellation.clone(),
            LIMIT,
        )?,
        values,
    ))?
    else {
        return Err("list outputs absent".into());
    };
    assert_eq!(outputs.len(), 3);
    for column in outputs {
        let NativeValue::List { values } = column else {
            return Err("list absent".into());
        };
        assert_eq!(values.len(), 100);
        for value in values {
            let NativeValue::Handle { value } = value else {
                return Err("segment handle absent".into());
            };
            store.resolve(&value, value.handle_type(), &cancellation)?;
        }
    }
    assert_eq!(generation.len(), 302);
    drop(store);
    assert_eq!(generation.len(), 0);
    Ok(())
}
