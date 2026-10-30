use serde_json::Value;
use sim_nodes::{
    NativeHandleStore, NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices, NativePrimitive,
    NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;

fn context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    scratch_bytes: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("latent".to_owned());
    let scratch = authority.authorize_workspace(scratch_bytes)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x62703), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x62701)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

#[test]
fn task627_registered_empty_latents_match_pinned_source() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0018/source_empty_latents.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing cases")?;
    assert_eq!(cases.len(), 112);
    for (index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62710 + index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            64 * 1024 * 1024,
        )?;
        let inputs = case["inputs"]
            .as_object()
            .ok_or("Missing inputs")?
            .iter()
            .map(|(name, value)| {
                Ok((
                    name.clone(),
                    NativeValue::Primitive {
                        value: if name == "seconds" {
                            NativePrimitive::Number(value.as_f64().ok_or("Missing seconds")?)
                        } else {
                            NativePrimitive::UnsignedInteger(
                                value.as_u64().ok_or("Missing integer")?,
                            )
                        },
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        let serialized = serde_json::to_vec(&inputs)?;
        let name = case["class_type"].as_str().ok_or("Missing class")?;
        let node = registry.node(name).ok_or("Missing node")?;
        let token = node.cache_change_token(&inputs)?;
        let restored = serde_json::from_slice(&serialized)?;
        assert_eq!(token, node.cache_change_token(&restored)?);
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        node.cache_dependencies(&context, &inputs)?;
        let outcome = smol::block_on(node.execute(context.clone(), restored))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("Unexpected node outcome".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        assert_eq!(outputs.len(), 1);
        assert_eq!(generation.len(), 1);
        let NativeValue::Handle { value: handle } =
            outputs.first().ok_or("Missing latent output")?
        else {
            return Err("Nonhandle output".into());
        };
        let payload =
            context
                .handle_store()
                .resolve(handle, handle.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
            return Err("Nonlatent output".into());
        };
        let samples = latent.samples().tensor().ok_or("Missing tensor")?;
        assert_eq!(samples.descriptor().dtype(), sim_tensor::DType::F32);
        assert_eq!(samples.descriptor().device(), sim_tensor::DeviceId::CPU);
        assert!(latent.noise_mask().is_none() && latent.batch_indices().is_none());
        let shape: Vec<u64> = serde_json::from_value(case["shape"].clone())?;
        assert_eq!(samples.descriptor().shape(), shape, "{name}/{index}");
        assert_eq!(
            serde_json::to_value(latent.metadata())?,
            case["metadata"],
            "{name}/{index}"
        );
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        assert!(
            sim_tensor::generated_native_diffusion::tensor_to_f32(
                compute.backend(),
                samples,
                &execution
            )?
            .iter()
            .all(|x| x.to_bits() == 0),
            "{name}/{index}"
        );
        assert_eq!(serialized, serde_json::to_vec(&inputs)?);
    }
    Ok(())
}

#[test]
fn task627_registered_batch_latents_preserve_source_dictionaries() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0018/source_batch_latents.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing batch cases")?;
    assert_eq!(cases.len(), 64);
    let node = registry
        .node("BatchLatentsNode")
        .ok_or("Missing batch node")?;
    let mut successes = 0;
    let mut rejections = 0;
    for (case_index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let attempt = AttemptId(Uuid::from_u128(0x627100 + case_index as u128));
        let store = generation.handle_store_for_attempt(attempt);
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
            64 * 1024 * 1024,
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let first_shape: Vec<u64> = serde_json::from_value(case["first_shape"].clone())?;
        let second_shape: Vec<u64> = serde_json::from_value(case["second_shape"].clone())?;
        let mask_shape: Vec<u64> = serde_json::from_value(case["mask_shape"].clone())?;
        let make = |shape: &[u64], divisor: f32| {
            sim_tensor::generated_native_diffusion::tensor_from_f32(
                compute.backend(),
                shape,
                &(0..shape.iter().product::<u64>())
                    .map(|i| i as f32 / divisor)
                    .collect::<Vec<_>>(),
                &execution,
            )
        };
        let mask = sim_tensor::generated_native_diffusion::tensor_from_f32(
            compute.backend(),
            &mask_shape,
            &vec![1.0; mask_shape.iter().product::<u64>() as usize],
            &execution,
        )?;
        let indices = (0..*first_shape.first().ok_or("Missing batch")?)
            .map(|i| 11 + i)
            .collect::<Vec<_>>();
        let first = Arc::new(sim_tensor::NativeLatentBundle::single(
            make(&first_shape, 32.0)?,
            Some(mask.clone()),
            Some(indices.clone()),
            sim_tensor::NativeLatentMetadata::checked(None, None, Some(8), None)?,
            &execution,
        )?);
        let second = Arc::new(sim_tensor::NativeLatentBundle::single(
            make(&second_shape, 16.0)?,
            None,
            None,
            sim_tensor::NativeLatentMetadata::checked(None, None, Some(16), None)?,
            &execution,
        )?);
        let mut inputs = BTreeMap::new();
        for (name, latent) in [("latent1", first.clone()), ("latent2", second)] {
            let handle = context
                .handle_store()
                .publish(NativeStoredPayload::Latent(latent), &context.cancellation)?;
            inputs.insert(name.to_owned(), NativeValue::Handle { value: handle });
        }
        let before = serde_json::to_vec(&inputs)?;
        node.cache_change_token(&inputs)?;
        node.cache_dependencies(&context, &inputs)?;
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        let result = smol::block_on(node.execute(context.clone(), inputs.clone()));
        if case.get("error").is_some() {
            let error = result.err().ok_or("Expected source batch failure")?;
            assert_eq!(
                error.kind,
                NativeNodeFailureKind::Failure,
                "{case_index}: {error}"
            );
            assert_eq!(generation.len(), 2);
            rejections += 1;
            continue;
        }
        let NativeNodeOutcome::Values { outputs, .. } =
            result.map_err(|e| format!("case {case_index}: {e}"))?
        else {
            return Err("Unexpected batch outcome".into());
        };
        assert_eq!(generation.len(), 3);
        let NativeValue::Handle { value: handle } = outputs.first().ok_or("Missing output")? else {
            return Err("Expected output handle".into());
        };
        let retained =
            context
                .handle_store()
                .resolve(handle, handle.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(output) = retained.as_ref() else {
            return Err("Expected output latent".into());
        };
        let expected_shape: Vec<u64> = serde_json::from_value(case["output_shape"].clone())?;
        let expected_indices: Vec<u64> = serde_json::from_value(case["batch_indices"].clone())?;
        let samples = output.samples().tensor().ok_or("Missing batch tensor")?;
        assert_eq!(samples.descriptor().shape(), expected_shape, "{case_index}");
        assert_eq!(
            output.batch_indices(),
            Some(expected_indices.as_slice()),
            "{case_index}"
        );
        assert_eq!(first.batch_indices(), Some(indices.as_slice()));
        assert_eq!(output.metadata(), first.metadata());
        assert_eq!(
            output
                .noise_mask()
                .and_then(|x| x.tensor())
                .ok_or("Missing mask")?
                .storage_id(),
            mask.storage_id()
        );
        let actual = sim_tensor::generated_native_diffusion::tensor_to_f32(
            compute.backend(),
            samples,
            &execution,
        )?;
        let expected: Vec<f32> = serde_json::from_value(case["output_values"].clone())?;
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                "{case_index}/{index}: {a} vs {b}"
            );
        }
        assert_eq!(before, serde_json::to_vec(&inputs)?);
        successes += 1;
    }
    assert_eq!((successes, rejections), (22, 42));
    Ok(())
}

fn scalar(value: u64) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::UnsignedInteger(value),
    }
}

fn empty_inputs(name: &str) -> BTreeMap<String, NativeValue> {
    let mut inputs = BTreeMap::from([("batch_size".to_owned(), scalar(1))]);
    if name.contains("AceStep") {
        inputs.insert(
            "seconds".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(1.0),
            },
        );
    } else {
        let minimum = if matches!(
            name,
            "EmptyHiDreamO1LatentImage" | "EmptyHunyuanImageLatent"
        ) {
            64
        } else {
            16
        };
        inputs.insert("width".to_owned(), scalar(minimum));
        inputs.insert("height".to_owned(), scalar(minimum));
        if name.ends_with("Video") || name == "EmptyARVideoLatent" {
            inputs.insert("length".to_owned(), scalar(1));
        }
    }
    inputs
}

fn output_handle(
    outcome: NativeNodeOutcome,
) -> Result<sim_nodes::NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Expected values".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    assert_eq!(outputs.len(), 1);
    let Some(NativeValue::Handle { value }) = outputs.first() else {
        return Err("Expected latent handle".into());
    };
    Ok(value.clone())
}

fn publish_latent(
    context: &NativeNodeContext,
    value: f32,
    batch: u64,
) -> Result<NativeValue, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let tensor = sim_tensor::generated_native_diffusion::tensor_from_f32(
        compute.backend(),
        &[batch, 4, 1, 1],
        &vec![value; (batch * 4) as usize],
        &execution,
    )?;
    let latent =
        sim_tensor::NativeLatentBundle::single(tensor, None, None, Default::default(), &execution)?;
    Ok(NativeValue::Handle {
        value: context.handle_store().publish(
            NativeStoredPayload::Latent(Arc::new(latent)),
            &context.cancellation,
        )?,
    })
}

#[test]
fn task627_boundaries_cancel_and_publication_failures_leave_no_output() -> Result<(), Box<dyn Error>>
{
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0018/fixture.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for row in fixture["nodes"].as_array().ok_or("Missing rows")? {
        let name = row["class_type"].as_str().ok_or("Missing class")?;
        let node = registry.node(name).ok_or("Missing node")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x627300)));
        let good_context = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
            4 * 1024 * 1024,
        )?;
        let good = if name == "BatchLatentsNode" {
            BTreeMap::from([("latent1".to_owned(), publish_latent(&good_context, 0.5, 1)?)])
        } else {
            empty_inputs(name)
        };
        node.cache_change_token(&good)?;
        let initial = generation.len();
        let mut bad_cases = vec![BTreeMap::new()];
        let mut extra = good.clone();
        extra.insert("unknown".to_owned(), scalar(1));
        bad_cases.push(extra);
        for key in good.keys() {
            let mut missing = good.clone();
            missing.remove(key);
            bad_cases.push(missing);
            for value in [
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(true),
                },
                NativeValue::Primitive {
                    value: NativePrimitive::Number(f64::NAN),
                },
                NativeValue::List { values: vec![] },
                NativeValue::List {
                    values: vec![
                        good.get(key).ok_or("Missing input")?.clone(),
                        scalar(u64::MAX),
                    ],
                },
            ] {
                let mut bad = good.clone();
                bad.insert(key.clone(), value);
                bad_cases.push(bad);
            }
        }
        if name == "BatchLatentsNode" {
            for key in [
                "latent0", "latent2", "latent01", "latent51", "latent-1", "latents",
            ] {
                bad_cases.push(BTreeMap::from([(
                    key.to_owned(),
                    good.get("latent1").ok_or("Missing latent")?.clone(),
                )]));
            }
            let mut gap = good.clone();
            gap.insert(
                "latent3".to_owned(),
                good.get("latent1").ok_or("Missing latent")?.clone(),
            );
            bad_cases.push(gap);
        } else {
            for key in good.keys() {
                let (minimum, maximum) = match key.as_str() {
                    "seconds" => (1, 1000),
                    "batch_size" => (
                        1,
                        if matches!(name, "EmptyARVideoLatent" | "EmptyHiDreamO1LatentImage") {
                            64
                        } else {
                            4096
                        },
                    ),
                    "length" => (
                        1,
                        if name == "EmptyARVideoLatent" {
                            1024
                        } else {
                            16384
                        },
                    ),
                    "width" | "height" => (
                        if matches!(
                            name,
                            "EmptyHiDreamO1LatentImage" | "EmptyHunyuanImageLatent"
                        ) {
                            64
                        } else {
                            16
                        },
                        match name {
                            "EmptyHiDreamO1LatentImage" => 4096,
                            "EmptyARVideoLatent" => 8192,
                            _ => 16384,
                        },
                    ),
                    _ => return Err("Unexpected parameter".into()),
                };
                for value in [minimum, maximum] {
                    let mut bound = good.clone();
                    bound.insert(key.clone(), scalar(value));
                    node.cache_change_token(&bound)
                        .map_err(|error| format!("{name}/{key}/{value}: {error}"))?;
                    node.cache_dependencies(&good_context, &bound)?;
                }
                for value in [minimum - 1, maximum + 1] {
                    let mut bad = good.clone();
                    bad.insert(key.clone(), scalar(value));
                    bad_cases.push(bad);
                }
            }
        }
        for bad in bad_cases {
            assert!(node.cache_change_token(&bad).is_err(), "{name}/{bad:?}");
            assert!(
                node.cache_dependencies(&good_context, &bad).is_err(),
                "{name}"
            );
            assert!(
                node.demanded_lazy_inputs(&good_context, &bad).is_err(),
                "{name}"
            );
            assert!(
                smol::block_on(node.execute(good_context.clone(), bad)).is_err(),
                "{name}"
            );
            assert_eq!(generation.len(), initial, "{name}");
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
            smol::block_on(node.execute(cancelled, good.clone()))
                .expect_err("Cancelled node executed")
                .kind,
            NativeNodeFailureKind::Interrupted
        );
        let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(1)?;
        assert!(
            smol::block_on(node.execute(
                context(
                    Arc::new(limited_backend),
                    &limited_authority,
                    store.clone(),
                    CancellationToken::default(),
                    0
                )?,
                good.clone()
            ))
            .is_err(),
            "{name}"
        );
        assert_eq!(generation.len(), initial);
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
                    4 * 1024 * 1024,
                )?,
                good.clone(),
            ))
            .expect_err("Publication boundary admitted output");
            assert_eq!(
                error.kind,
                if reject {
                    NativeNodeFailureKind::Failure
                } else {
                    NativeNodeFailureKind::Interrupted
                }
            );
            assert_eq!(generation.len(), initial, "{name}");
        }
        output_handle(smol::block_on(node.execute(good_context, good))?)?;
        assert_eq!(generation.len(), initial + 1, "{name}");
    }
    Ok(())
}

#[test]
fn task627_dynamic_order_and_stale_handles_are_checked() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("BatchLatentsNode")
        .ok_or("Missing batch node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let attempt = AttemptId(Uuid::from_u128(0x627400));
    let store = generation.handle_store_for_attempt(attempt);
    let context = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        4 * 1024 * 1024,
    )?;
    let inputs = (1..=50)
        .map(|index| {
            Ok((
                format!("latent{index}"),
                publish_latent(&context, index as f32, 1)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
    let dependencies = node.cache_dependencies(&context, &inputs)?;
    assert_eq!(dependencies.artifact_digests.len(), 50);
    let serialized = serde_json::to_vec(&inputs)?;
    let restored = serde_json::from_slice(&serialized)?;
    assert_eq!(dependencies, node.cache_dependencies(&context, &restored)?);
    let output = output_handle(smol::block_on(node.execute(context.clone(), restored))?)?;
    let retained = store.resolve(&output, output.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Latent(latent) = retained.as_ref() else {
        return Err("Expected LATENT".into());
    };
    assert_eq!(latent.batch_indices(), Some(vec![0; 50].as_slice()));
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let actual = sim_tensor::generated_native_diffusion::tensor_to_f32(
        compute.backend(),
        latent.samples().tensor().ok_or("Missing samples")?,
        &execution,
    )?;
    assert_eq!(
        &*actual,
        (1..=50)
            .flat_map(|index| vec![index as f32; 4])
            .collect::<Vec<_>>()
    );
    assert_eq!(serialized, serde_json::to_vec(&inputs)?);
    let fresh_generation = NativeHandleStoreGeneration::new()?;
    let fresh_context = crate::context(
        backend,
        &authority,
        fresh_generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        4 * 1024 * 1024,
    )?;
    assert!(node.cache_dependencies(&fresh_context, &inputs).is_err());
    assert!(smol::block_on(node.execute(fresh_context.clone(), inputs.clone())).is_err());
    assert_eq!(fresh_generation.len(), 0);
    let recovered = BTreeMap::from([(
        "latent1".to_owned(),
        publish_latent(&fresh_context, 1.0, 1)?,
    )]);
    let recovered_output = output_handle(smol::block_on(
        node.execute(fresh_context.clone(), recovered),
    )?)?;
    fresh_context.handle_store().resolve(
        &recovered_output,
        recovered_output.handle_type(),
        &fresh_context.cancellation,
    )?;
    assert_eq!(fresh_generation.len(), 2);
    let Some(NativeValue::Handle { value }) = inputs.get("latent1") else {
        return Err("Missing input".into());
    };
    store.revoke(value, &context.cancellation)?;
    let retained_count = generation.len();
    assert!(node.cache_dependencies(&context, &inputs).is_err());
    assert!(smol::block_on(node.execute(context, inputs)).is_err());
    assert_eq!(generation.len(), retained_count);
    Ok(())
}

#[test]
fn task627_nested_batch_failures_match_source_without_publication() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0018/source_nested_failures.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("BatchLatentsNode")
        .ok_or("Missing batch node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for case in fixture["cases"].as_array().ok_or("Missing cases")? {
        let side = case["second_side"].as_u64().ok_or("Missing side")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x627500))),
            CancellationToken::default(),
            4 * 1024 * 1024,
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let make = |shape: &[u64]| {
            sim_tensor::generated_native_diffusion::tensor_from_f32(
                compute.backend(),
                shape,
                &vec![0.0; shape.iter().product::<u64>() as usize],
                &execution,
            )
        };
        let mut inputs = BTreeMap::new();
        for (index, side) in [2, side].into_iter().enumerate() {
            let latent = sim_tensor::NativeLatentBundle::audio_video(
                make(&[1, 16, 2, side, side])?,
                make(&[1, 64, 4])?,
                None,
                None,
                None,
                Default::default(),
                &execution,
            )?;
            inputs.insert(
                format!("latent{}", index + 1),
                NativeValue::Handle {
                    value: context.handle_store().publish(
                        NativeStoredPayload::Latent(Arc::new(latent)),
                        &context.cancellation,
                    )?,
                },
            );
        }
        node.cache_dependencies(&context, &inputs)?;
        let error = smol::block_on(node.execute(context.clone(), inputs))
            .expect_err("Nested source failure accepted");
        assert_eq!(error.kind, NativeNodeFailureKind::Failure);
        assert!(error.message.contains("nested LATENT"));
        assert_eq!(generation.len(), 2);
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
        handle: &sim_nodes::NativeOpaqueHandle,
        expected: &sim_nodes::NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<sim_nodes::NativeResolvedPayload, sim_nodes::NativeHandleStoreError> {
        self.inner.resolve(handle, expected, cancellation)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<sim_nodes::NativeOpaqueHandle, sim_nodes::NativeHandleStoreError> {
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
        handle: &sim_nodes::NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), sim_nodes::NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)
    }
}

#[test]
fn task627_engine_maps_latents_repeats_last_and_recovers_persisted_plans()
-> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0018/fixture.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for row in fixture["nodes"].as_array().ok_or("Missing rows")? {
        let name = row["class_type"].as_str().ok_or("Missing class")?;
        if name == "BatchLatentsNode" {
            continue;
        }
        let base = empty_inputs(name);
        let varying_key = if name.contains("AceStep") {
            "seconds"
        } else {
            "width"
        };
        let first_value = base
            .get(varying_key)
            .ok_or("Missing varying field")?
            .clone();
        let second_value = match &first_value {
            NativeValue::Primitive {
                value: NativePrimitive::Number(value),
            } => NativeValue::Primitive {
                value: NativePrimitive::Number(value * 2.0),
            },
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(value),
            } => scalar(value * 2),
            _ => return Err("Unexpected varying parameter".into()),
        };
        let mut mapped_inputs = base.clone();
        mapped_inputs.insert(
            "batch_size".to_owned(),
            NativeValue::List {
                values: vec![scalar(1), scalar(2), scalar(3)],
            },
        );
        mapped_inputs.insert(
            varying_key.to_owned(),
            NativeValue::List {
                values: vec![first_value, second_value.clone()],
            },
        );
        let mut second_inputs = base.clone();
        second_inputs.insert(
            "batch_size".to_owned(),
            NativeValue::List {
                values: vec![scalar(2), scalar(4)],
            },
        );
        second_inputs.insert(
            varying_key.to_owned(),
            NativeValue::List {
                values: vec![
                    base.get(varying_key)
                        .ok_or("Missing varying field")?
                        .clone(),
                    second_value.clone(),
                ],
            },
        );
        let empty_id = NodeId("empty".to_owned());
        let second_id = NodeId("second".to_owned());
        let batch_id = NodeId("batch".to_owned());
        let make_node = |id: &NodeId,
                         inputs: BTreeMap<String, NativeValue>|
         -> Result<CompiledNode, Box<dyn Error>> {
            Ok(CompiledNode {
                id: id.clone(),
                class_type: name.to_owned(),
                descriptor: registry
                    .descriptor(name)
                    .ok_or("Missing empty descriptor")?
                    .clone(),
                inputs: inputs
                    .into_iter()
                    .map(|(key, value)| (key, InputBinding::Literal { value }))
                    .collect(),
                unknown: BTreeMap::new(),
            })
        };
        let mut batch_descriptor = registry
            .descriptor("BatchLatentsNode")
            .ok_or("Missing batch descriptor")?
            .clone();
        batch_descriptor.output_node = true;
        let link = |source| InputBinding::Link {
            source,
            output_index: 0,
            lazy: false,
            mode: sim_nodes::NativePortCardinality::Mapped,
        };
        let plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x627600)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([
                (empty_id.clone(), make_node(&empty_id, mapped_inputs)?),
                (second_id.clone(), make_node(&second_id, second_inputs)?),
                (
                    batch_id.clone(),
                    CompiledNode {
                        id: batch_id.clone(),
                        class_type: "BatchLatentsNode".to_owned(),
                        descriptor: batch_descriptor,
                        inputs: BTreeMap::from([
                            ("latent1".to_owned(), link(empty_id.clone())),
                            ("latent2".to_owned(), link(second_id.clone())),
                        ]),
                        unknown: BTreeMap::new(),
                    },
                ),
            ]),
            topological_order: vec![empty_id.clone(), second_id.clone(), batch_id.clone()],
            static_required_nodes: std::collections::BTreeSet::from([
                empty_id,
                second_id,
                batch_id.clone(),
            ]),
            output_nodes: vec![batch_id.clone()],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
        let serialized = serde_json::to_vec(&plan)?;
        let mut previous_handles: Option<Vec<NativeValue>> = None;
        for restart in 0..2 {
            let plan: CompiledPlan = serde_json::from_slice(&serialized)?;
            let executor = NativeImageExecutor::new_with_generated_registry(
                sim_types::ProfileId(Uuid::from_u128(0x627610)),
                BTreeMap::new(),
                true,
                backend.clone(),
            )?;
            let execute = |iteration| {
                executor.execute_blocking(
                    &plan,
                    AttemptId(Uuid::from_u128(0x627620 + restart * 10 + iteration)),
                    CancellationToken::default(),
                    0,
                    authority
                        .authorize_workspace(16 * 1024 * 1024)
                        .expect("Valid fixed workspace"),
                )
            };
            let result = execute(0)?;
            assert_eq!(
                result.report.state,
                AttemptState::Succeeded,
                "{name}: {:?}",
                result.report.error
            );
            assert_eq!(result.report.cache_hits, 0);
            let Some(NativeValue::List { values }) = result
                .report
                .outputs
                .get(&batch_id)
                .and_then(|outputs| outputs.first())
            else {
                return Err("Missing mapped outputs".into());
            };
            assert_eq!(values.len(), 3);
            let generation = executor.handle_store_generation_for_test();
            let context = context(
                backend.clone(),
                &authority,
                generation
                    .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x627620 + restart * 10))),
                CancellationToken::default(),
                16 * 1024 * 1024,
            )?;
            if let Some(previous) = &previous_handles {
                for value in previous {
                    if let NativeValue::Handle { value } = value {
                        assert!(
                            context
                                .handle_store()
                                .resolve(value, value.handle_type(), &context.cancellation)
                                .is_err()
                        );
                    }
                }
            }
            let compute = context.compute_session()?;
            let execution = compute.execution_context(&context)?;
            for (index, value) in values.iter().enumerate() {
                let NativeValue::Handle { value } = value else {
                    return Err("Missing mapped handle".into());
                };
                let payload = context.handle_store().resolve(
                    value,
                    value.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                    return Err("Missing LATENT".into());
                };
                let mut direct = base.clone();
                direct.insert("batch_size".to_owned(), scalar(index as u64 + 1));
                if index > 0 {
                    direct.insert(varying_key.to_owned(), second_value.clone());
                }
                let expected_handle = output_handle(smol::block_on(
                    registry
                        .node(name)
                        .ok_or("Missing node")?
                        .execute(context.clone(), direct),
                )?)?;
                let expected_payload = context.handle_store().resolve(
                    &expected_handle,
                    expected_handle.handle_type(),
                    &context.cancellation,
                )?;
                let NativeStoredPayload::Latent(expected) = expected_payload.as_ref() else {
                    return Err("Missing expected LATENT".into());
                };
                let samples = latent.samples().tensor().ok_or("Missing tensor")?;
                let mut shape = expected
                    .samples()
                    .tensor()
                    .ok_or("Missing expected tensor")?
                    .descriptor()
                    .shape()
                    .to_vec();
                let other_batch = if index == 0 { 2 } else { 4 };
                *shape.first_mut().ok_or("Missing batch dimension")? += other_batch;
                assert_eq!(samples.descriptor().shape(), shape, "{name}/{index}");
                assert_eq!(latent.metadata(), expected.metadata());
                assert_eq!(
                    latent.batch_indices(),
                    Some(
                        (0..index as u64 + 1)
                            .chain(0..other_batch)
                            .collect::<Vec<_>>()
                            .as_slice()
                    )
                );
                assert!(
                    sim_tensor::generated_native_diffusion::tensor_to_f32(
                        compute.backend(),
                        samples,
                        &execution
                    )?
                    .iter()
                    .all(|value| value.to_bits() == 0)
                );
                context
                    .handle_store()
                    .revoke(&expected_handle, &context.cancellation)?;
            }
            let cached = execute(1)?;
            assert_eq!(
                cached.report.state,
                AttemptState::Succeeded,
                "{name}: {:?}",
                cached.report.error
            );
            assert_eq!(cached.report.outputs, result.report.outputs);
            assert_eq!(cached.report.cache_hits, 3);
            for empty in [false, true] {
                let mut invalid_plan = plan.clone();
                let empty_node = invalid_plan
                    .nodes
                    .get_mut(&NodeId("empty".to_owned()))
                    .ok_or("Missing empty node")?;
                empty_node.inputs.insert(
                    "batch_size".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::List {
                            values: if empty {
                                vec![]
                            } else {
                                vec![scalar(1), scalar(4097)]
                            },
                        },
                    },
                );
                let before = generation.len();
                let failed = executor.execute_blocking(
                    &invalid_plan,
                    AttemptId(Uuid::from_u128(0x627650 + u128::from(empty))),
                    CancellationToken::default(),
                    0,
                    authority.authorize_workspace(16 * 1024 * 1024)?,
                )?;
                assert_eq!(failed.report.state, AttemptState::Failed, "{name}/{empty}");
                assert!(failed.report.outputs.is_empty());
                assert_eq!(generation.len(), before);
            }
            previous_handles = Some(values.clone());
        }
    }
    Ok(())
}
