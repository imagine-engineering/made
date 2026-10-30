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
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x62803), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x62801)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

#[test]
fn task628_registered_empty_latents_match_pinned_source() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_empty_latents.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing cases")?;
    assert_eq!(cases.len(), 120);
    for (index, case) in cases.iter().enumerate() {
        if case["class_type"] == "EmptyLatentImage" {
            continue;
        }
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62810 + index as u128)));
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
fn task628_registered_latent_add_preserves_source_dictionaries() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_latent_add.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing batch cases")?;
    assert_eq!(cases.len(), 100);
    let node = registry.node("LatentAdd").ok_or("Missing batch node")?;
    let mut successes = 0;
    let mut rejections = 0;
    for (case_index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let attempt = AttemptId(Uuid::from_u128(0x628100 + case_index as u128));
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
        for (name, latent) in [("samples1", first.clone()), ("samples2", second)] {
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
    assert_eq!((successes, rejections), (54, 46));
    Ok(())
}

fn scalar(value: u64) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::UnsignedInteger(value),
    }
}

#[test]
fn task628_nested_latent_add_matches_source_broadcasting_and_failures() -> Result<(), Box<dyn Error>>
{
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_nested_add.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("LatentAdd").ok_or("Missing add node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let cases = fixture["cases"].as_array().ok_or("Missing nested cases")?;
    assert_eq!(cases.len(), 7);
    for (index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let execution_context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x628b00))),
            CancellationToken::default(),
            4 * 1024 * 1024,
        )?;
        let compute = execution_context.compute_session()?;
        let execution = compute.execution_context(&execution_context)?;
        let make = |shape: &Value, value: f32| -> Result<sim_tensor::Tensor, Box<dyn Error>> {
            let shape: Vec<u64> = serde_json::from_value(shape.clone())?;
            let count = usize::try_from(shape.iter().product::<u64>())?;
            Ok(sim_tensor::generated_native_diffusion::tensor_from_f32(
                compute.backend(),
                &shape,
                &vec![value; count],
                &execution,
            )?)
        };
        let mut inputs = BTreeMap::new();
        for (prefix, name, value, audio_value) in [
            ("first", "samples1", 2.0, 3.0),
            ("second", "samples2", 5.0, 7.0),
        ] {
            let tensor = make(&case[format!("{prefix}_shape")], value)?;
            let latent = if case[format!("{prefix}_nested")]
                .as_bool()
                .ok_or("Missing nested flag")?
            {
                sim_tensor::NativeLatentBundle::audio_video(
                    tensor,
                    make(&case[format!("{prefix}_audio_shape")], audio_value)?,
                    None,
                    None,
                    None,
                    Default::default(),
                    &execution,
                )?
            } else {
                sim_tensor::NativeLatentBundle::single(
                    tensor,
                    None,
                    None,
                    Default::default(),
                    &execution,
                )?
            };
            inputs.insert(
                name.to_owned(),
                NativeValue::Handle {
                    value: execution_context.handle_store().publish(
                        NativeStoredPayload::Latent(Arc::new(latent)),
                        &execution_context.cancellation,
                    )?,
                },
            );
        }
        let result = smol::block_on(node.execute(execution_context.clone(), inputs));
        if case.get("error").is_some() {
            assert!(result.is_err(), "{index}");
            assert_eq!(generation.len(), 2);
            continue;
        }
        let NativeNodeOutcome::Values { outputs, .. } =
            result.map_err(|error| format!("case {index}: {error}"))?
        else {
            return Err("Missing result".into());
        };
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
            return Err("Missing handle".into());
        };
        let retained = execution_context.handle_store().resolve(
            value,
            value.handle_type(),
            &execution_context.cancellation,
        )?;
        let NativeStoredPayload::Latent(latent) = retained.as_ref() else {
            return Err("Missing latent".into());
        };
        let (video, audio) = latent
            .samples()
            .audio_video()
            .ok_or("Expected nested output")?;
        let expected = case["outputs"].as_array().ok_or("Missing source outputs")?;
        assert_eq!(expected.len(), 2);
        for (tensor, expected) in [video, audio].into_iter().zip(expected) {
            let shape: Vec<u64> = serde_json::from_value(expected["shape"].clone())?;
            let values: Vec<f32> = serde_json::from_value(expected["values"].clone())?;
            assert_eq!(tensor.descriptor().shape(), shape, "{index}");
            let actual = sim_tensor::generated_native_diffusion::tensor_to_f32(
                compute.backend(),
                tensor,
                &execution,
            )?;
            assert_eq!(&*actual, values.as_slice(), "{index}");
        }
        assert_eq!(generation.len(), 3);
    }
    Ok(())
}

#[test]
fn task628_all_empty_nodes_map_cache_and_recover_through_latent_add() -> Result<(), Box<dyn Error>>
{
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_empty_latents.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing cases")?;
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> =
        Arc::new(sim_test_support::NativeDiffusionFixture::checked_in());
    let registry = generated_native_node_registry_projection(Some(provider.clone()))?;
    let names = cases
        .iter()
        .map(|case| case["class_type"].as_str().ok_or("Missing class"))
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    assert_eq!(names.len(), 8);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let mut scenarios = Vec::new();
    for name in names {
        let selected = cases
            .iter()
            .filter(|case| case["class_type"] == name)
            .collect::<Vec<_>>();
        if name == "EmptyLatentImage" {
            assert_eq!(selected.len(), 6);
            for pair in selected.chunks(2) {
                let [first, last] = pair else {
                    return Err("Missing retained source pair".into());
                };
                scenarios.push((name, *first, *last));
            }
        } else {
            scenarios.push((
                name,
                *selected.first().ok_or("Missing first case")?,
                *selected.last().ok_or("Missing last case")?,
            ));
        }
    }
    for (name, first, last) in scenarios {
        let value = |key: &str, case: &Value| -> Result<NativeValue, Box<dyn Error>> {
            Ok(if key == "seconds" {
                NativeValue::Primitive {
                    value: NativePrimitive::Number(
                        case["inputs"][key].as_f64().ok_or("Missing seconds")?,
                    ),
                }
            } else {
                scalar(case["inputs"][key].as_u64().ok_or("Missing integer")?)
            })
        };
        let inputs = first["inputs"]
            .as_object()
            .ok_or("Missing inputs")?
            .keys()
            .map(|key| {
                let mut values = vec![value(key, first)?, value(key, last)?];
                if key == "batch_size" {
                    values.push(value(key, last)?);
                }
                Ok((
                    key.clone(),
                    InputBinding::Literal {
                        value: NativeValue::List { values },
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        let empty = NodeId("empty".to_owned());
        let add = NodeId("add".to_owned());
        let mut add_descriptor = registry
            .descriptor("LatentAdd")
            .ok_or("Missing add descriptor")?
            .clone();
        add_descriptor.output_node = true;
        let link = || InputBinding::Link {
            source: empty.clone(),
            output_index: 0,
            lazy: false,
            mode: sim_nodes::NativePortCardinality::Mapped,
        };
        let plan = CompiledPlan {
            prompt_id: PromptId(Uuid::from_u128(0x628a00)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([
                (
                    empty.clone(),
                    CompiledNode {
                        id: empty.clone(),
                        class_type: name.to_owned(),
                        descriptor: registry
                            .descriptor(name)
                            .ok_or("Missing empty descriptor")?
                            .clone(),
                        inputs,
                        unknown: BTreeMap::new(),
                    },
                ),
                (
                    add.clone(),
                    CompiledNode {
                        id: add.clone(),
                        class_type: "LatentAdd".to_owned(),
                        descriptor: add_descriptor,
                        inputs: BTreeMap::from([
                            ("samples1".to_owned(), link()),
                            ("samples2".to_owned(), link()),
                        ]),
                        unknown: BTreeMap::new(),
                    },
                ),
            ]),
            topological_order: vec![empty.clone(), add.clone()],
            static_required_nodes: std::collections::BTreeSet::from([empty.clone(), add.clone()]),
            output_nodes: vec![add.clone()],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        plan.validate_integrity()?;
        let encoded = serde_json::to_vec(&plan)?;
        let mut previous: Option<Vec<NativeValue>> = None;
        for restart in 0..2 {
            let plan: CompiledPlan = serde_json::from_slice(&encoded)?;
            let executor = NativeImageExecutor::new_with_generated_registry_and_diffusion_provider(
                sim_types::ProfileId(Uuid::from_u128(0x628a10)),
                BTreeMap::new(),
                true,
                backend.clone(),
                provider.clone(),
            )?;
            let attempt = AttemptId(Uuid::from_u128(0x628a20 + restart * 10));
            let result = executor.execute_blocking(
                &plan,
                attempt,
                CancellationToken::default(),
                0,
                authority.authorize_workspace(64 * 1024 * 1024)?,
            )?;
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
                .get(&add)
                .and_then(|outputs| outputs.first())
            else {
                return Err("Missing mapped add outputs".into());
            };
            assert_eq!(values.len(), 3);
            let generation = executor.handle_store_generation_for_test();
            let execution_context = context(
                backend.clone(),
                &authority,
                generation.handle_store_for_attempt(attempt),
                CancellationToken::default(),
                64 * 1024 * 1024,
            )?;
            if let Some(previous) = &previous {
                for value in previous {
                    let NativeValue::Handle { value } = value else {
                        return Err("Expected previous handle".into());
                    };
                    assert!(
                        execution_context
                            .handle_store()
                            .resolve(value, value.handle_type(), &execution_context.cancellation)
                            .is_err()
                    );
                }
            }
            let compute = execution_context.compute_session()?;
            let execution = compute.execution_context(&execution_context)?;
            for (value, expected) in values.iter().zip([first, last, last]) {
                let NativeValue::Handle { value } = value else {
                    return Err("Expected LATENT handle".into());
                };
                let retained = execution_context.handle_store().resolve(
                    value,
                    value.handle_type(),
                    &execution_context.cancellation,
                )?;
                let NativeStoredPayload::Latent(latent) = retained.as_ref() else {
                    return Err("Expected LATENT bundle".into());
                };
                let samples = latent.samples().tensor().ok_or("Missing tensor")?;
                let shape: Vec<u64> = serde_json::from_value(expected["shape"].clone())?;
                assert_eq!(samples.descriptor().shape(), shape, "{name}");
                assert_eq!(
                    serde_json::to_value(latent.metadata())?,
                    expected["metadata"]
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
            }
            let cached = executor.execute_blocking(
                &plan,
                AttemptId(Uuid::from_u128(0x628a21 + restart * 10)),
                CancellationToken::default(),
                0,
                authority.authorize_workspace(64 * 1024 * 1024)?,
            )?;
            assert_eq!(
                cached.report.state,
                AttemptState::Succeeded,
                "{name}: {:?}",
                cached.report.error
            );
            assert_eq!(cached.report.cache_hits, 2);
            assert_eq!(cached.report.outputs, result.report.outputs);
            for empty_list in [false, true] {
                let mut invalid = plan.clone();
                invalid
                    .nodes
                    .get_mut(&empty)
                    .ok_or("Missing empty node")?
                    .inputs
                    .insert(
                        "batch_size".to_owned(),
                        InputBinding::Literal {
                            value: NativeValue::List {
                                values: if empty_list {
                                    vec![]
                                } else {
                                    vec![scalar(1), scalar(4097)]
                                },
                            },
                        },
                    );
                let before = generation.len();
                let failed = executor.execute_blocking(
                    &invalid,
                    AttemptId(Uuid::from_u128(0x628a40 + u128::from(empty_list))),
                    CancellationToken::default(),
                    0,
                    authority.authorize_workspace(64 * 1024 * 1024)?,
                )?;
                assert_eq!(
                    failed.report.state,
                    AttemptState::Failed,
                    "{name}/{empty_list}"
                );
                assert!(failed.report.outputs.is_empty());
                assert_eq!(generation.len(), before);
            }
            previous = Some(values.clone());
        }
    }
    Ok(())
}

fn hunyuan_fixture_payloads(
    execution_context: &NativeNodeContext,
    fixture: &Value,
    model_scale: f32,
) -> Result<
    (
        sim_nodes::NativeStoredModelPayload,
        Arc<sim_tensor::NativeLatentBundle>,
    ),
    Box<dyn Error>,
> {
    use sim_model::{
        NativeModelPayload,
        latent_upscale_model::{NativeLatentUpscaleCheckpoint, NativeLatentUpscaleModelResource},
    };
    let compute = execution_context.compute_session()?;
    let execution = compute.execution_context(execution_context)?;
    let make = |shape: &[u64], values: &[f32]| {
        sim_tensor::generated_native_diffusion::tensor_from_f32(
            compute.backend(),
            shape,
            values,
            &execution,
        )
    };
    let state = fixture["state"]
        .as_object()
        .ok_or("Missing checkpoint state")?
        .iter()
        .map(|(key, tensor)| {
            let shape: Vec<u64> = serde_json::from_value(tensor["shape"].clone())?;
            let mut values: Vec<f32> = serde_json::from_value(tensor["values"].clone())?;
            for value in &mut values {
                *value *= model_scale;
            }
            Ok((key.clone(), make(&shape, &values)?))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let resource = NativeLatentUpscaleModelResource::from_checkpoint(
        NativeLatentUpscaleCheckpoint {
            artifact_sha256: if model_scale == 1.0 { "1" } else { "2" }.repeat(64),
            metadata: BTreeMap::new(),
            ordered_state: state,
            memory_budget_bytes: 64 * 1024 * 1024,
        },
        &execution,
    )?;
    let model = sim_nodes::NativeStoredModelPayload::model_resource(Arc::new(
        NativeModelPayload::latent_upscale_model(Arc::new(resource))?,
    ))?;
    let input_shape: Vec<u64> = serde_json::from_value(fixture["input_shape"].clone())?;
    let input_values: Vec<f32> = serde_json::from_value(fixture["input_values"].clone())?;
    let input_bundle = Arc::new(sim_tensor::NativeLatentBundle::single(
        make(&input_shape, &input_values)?,
        Some(make(&[1, 2, 3], &[1.0; 6])?),
        Some(vec![11]),
        sim_tensor::NativeLatentMetadata::checked(None, None, Some(16), None)?,
        &execution,
    )?);
    Ok((model, input_bundle))
}

struct HunyuanFixtureNode;

impl sim_nodes::NativeNode for HunyuanFixtureNode {
    fn class_type(&self) -> &str {
        "Task628HunyuanFixture"
    }

    fn implementation_version(&self) -> &str {
        "task628-fixture-v1"
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<NativeNodeOutcome, sim_nodes::NativeNodeFailure>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let failure = |error: String| sim_nodes::NativeNodeFailure {
                code: "fixture_failed".to_owned(),
                message: error,
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            };
            let fixture: Value = serde_json::from_str(include_str!(
                "../fixtures/nodes/model-latent-comfy-node-0159/source_hunyuan_upscale.json"
            ))
            .map_err(|error| failure(error.to_string()))?;
            let scale = if inputs.get("variant") == Some(&scalar(1)) {
                1.0
            } else {
                0.5
            };
            let (model, latent) = hunyuan_fixture_payloads(&context, &fixture, scale)
                .map_err(|error| failure(error.to_string()))?;
            let mut handles = Vec::new();
            for payload in [
                NativeStoredPayload::Model(Arc::new(model)),
                NativeStoredPayload::Latent(latent),
            ] {
                match context
                    .handle_store()
                    .publish(payload, &context.cancellation)
                {
                    Ok(handle) => handles.push(handle),
                    Err(error) => {
                        for handle in &handles {
                            context
                                .handle_store()
                                .revoke(handle, &CancellationToken::default())
                                .map_err(|error| failure(error.to_string()))?;
                        }
                        return Err(failure(error.to_string()));
                    }
                }
            }
            Ok(NativeNodeOutcome::Values {
                outputs: handles
                    .into_iter()
                    .map(|value| NativeValue::Handle { value })
                    .collect(),
                effects: Vec::new(),
                ui: None,
            })
        })
    }
}

#[test]
fn task628_hunyuan_engine_maps_model_changes_and_recovers() -> Result<(), Box<dyn Error>> {
    use sim_runtime::{
        AttemptState, CompiledNode, CompiledPlan, InputBinding, NativeImageExecutor,
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_hunyuan_upscale.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let mut upscale_descriptor = registry
        .descriptor("HunyuanVideo15LatentUpscaleWithModel")
        .ok_or("Missing Hunyuan descriptor")?
        .clone();
    let mut producer_descriptor = upscale_descriptor.clone();
    producer_descriptor.class_type = "Task628HunyuanFixture".to_owned();
    producer_descriptor.implementation_version = "task628-fixture-v1".to_owned();
    producer_descriptor.schema_version = sim_nodes::LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
    producer_descriptor.source_schema = None;
    let mut variant = upscale_descriptor
        .inputs
        .iter()
        .find(|input| input.name == "width")
        .ok_or("Missing width input")?
        .clone();
    variant.name = "variant".to_owned();
    variant.cardinality = sim_nodes::NativePortCardinality::Scalar;
    producer_descriptor.inputs = vec![variant];
    producer_descriptor.outputs.insert(
        0,
        sim_nodes::NativeOutputDescriptor {
            name: "model".to_owned(),
            produced_type: upscale_descriptor
                .inputs
                .iter()
                .find(|input| input.name == "model")
                .and_then(|input| input.accepted_types.members().first())
                .ok_or("Missing model type")?
                .clone(),
            is_list: false,
        },
    );
    upscale_descriptor.output_node = true;
    producer_descriptor
        .validate()
        .map_err(|error| format!("Fixture descriptor: {error}"))?;
    upscale_descriptor
        .validate()
        .map_err(|error| format!("Hunyuan descriptor: {error}"))?;
    let producer = NodeId("fixture".to_owned());
    let upscale = NodeId("upscale".to_owned());
    let cases = fixture["cases"].as_array().ok_or("Missing cases")?;
    let selected = cases
        .iter()
        .filter(|case| case["inputs"]["width"] == 64 && case["inputs"]["height"] == 80)
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 10);
    let inputs = ["width", "height", "upscale_method", "crop"]
        .into_iter()
        .map(|name| {
            let values = selected
                .iter()
                .map(|case| {
                    Ok(if let Some(value) = case["inputs"][name].as_str() {
                        NativeValue::Primitive {
                            value: NativePrimitive::String(value.to_owned()),
                        }
                    } else {
                        scalar(case["inputs"][name].as_u64().ok_or("Missing dimension")?)
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            Ok((
                name.to_owned(),
                InputBinding::Literal {
                    value: NativeValue::List { values },
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
    let mut inputs = inputs;
    for (name, output_index) in [("model", 0), ("samples", 1)] {
        inputs.insert(
            name.to_owned(),
            InputBinding::Link {
                source: producer.clone(),
                output_index,
                lazy: false,
                mode: sim_nodes::NativePortCardinality::Mapped,
            },
        );
    }
    let plan = CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x628c00)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([
            (
                producer.clone(),
                CompiledNode {
                    id: producer.clone(),
                    class_type: producer_descriptor.class_type.clone(),
                    descriptor: producer_descriptor.clone(),
                    inputs: BTreeMap::from([(
                        "variant".to_owned(),
                        InputBinding::Literal { value: scalar(1) },
                    )]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                upscale.clone(),
                CompiledNode {
                    id: upscale.clone(),
                    class_type: upscale_descriptor.class_type.clone(),
                    descriptor: upscale_descriptor,
                    inputs,
                    unknown: BTreeMap::new(),
                },
            ),
        ]),
        topological_order: vec![producer.clone(), upscale.clone()],
        static_required_nodes: [producer.clone(), upscale.clone()].into_iter().collect(),
        output_nodes: vec![upscale.clone()],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()
        .map_err(|error| format!("Hunyuan compiled fixture plan: {error}"))?;
    let encoded = serde_json::to_vec(&plan)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let mut previous_handles: Vec<sim_nodes::NativeOpaqueHandle> = Vec::new();
    for restart in 0..2 {
        let plan: CompiledPlan = serde_json::from_slice(&encoded)?;
        let executor = NativeImageExecutor::new_with_generated_registry_and_registration(
            sim_types::ProfileId(Uuid::from_u128(0x628c01)),
            BTreeMap::new(),
            true,
            backend.clone(),
            |registry| {
                registry
                    .register_bound_batch_with_presentations([(
                        producer_descriptor.clone(),
                        Arc::new(HunyuanFixtureNode) as Arc<dyn sim_nodes::NativeNode>,
                        sim_nodes::NativeNodePresentation {
                            display_name: "Task 628 Hunyuan fixture".to_owned(),
                            category: "test".to_owned(),
                            description: String::new(),
                            output_names: vec!["model".to_owned(), "latent".to_owned()],
                            search_aliases: Vec::new(),
                            is_deprecated: false,
                            is_experimental: false,
                        },
                    )])
                    .map_err(|error| {
                        sim_runtime::NativeImageRuntimeError::Registry(error.to_string())
                    })?;
                Ok(())
            },
        )?;
        let execute = |plan: &CompiledPlan,
                       sequence|
         -> Result<sim_runtime::NativeImageExecutionResult, Box<dyn Error>> {
            Ok(executor.execute_blocking(
                plan,
                AttemptId(Uuid::from_u128(0x628c10 + restart * 16 + sequence)),
                CancellationToken::default(),
                0,
                authority.authorize_workspace(64 * 1024 * 1024)?,
            )?)
        };
        let result = execute(&plan, 0)?;
        assert_eq!(
            result.report.state,
            AttemptState::Succeeded,
            "{:?}",
            result.report.error
        );
        assert_eq!(result.report.cache_hits, 0);
        let Some(NativeValue::List { values }) = result
            .report
            .outputs
            .get(&upscale)
            .and_then(|outputs| outputs.first())
        else {
            return Err("Missing mapped Hunyuan outputs".into());
        };
        assert_eq!(values.len(), 10);
        let generation = executor.handle_store_generation_for_test();
        let context = context(
            backend.clone(),
            &authority,
            generation
                .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x628c10 + restart * 16))),
            CancellationToken::default(),
            64 * 1024 * 1024,
        )?;
        for handle in &previous_handles {
            assert!(
                context
                    .handle_store()
                    .resolve(handle, handle.handle_type(), &context.cancellation)
                    .is_err()
            );
        }
        previous_handles.clear();
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let mut original_digests = Vec::new();
        for (value, expected) in values.iter().zip(&selected) {
            let NativeValue::Handle { value: handle } = value else {
                return Err("Missing Hunyuan handle".into());
            };
            previous_handles.push(handle.clone());
            let payload = context.handle_store().resolve(
                handle,
                handle.handle_type(),
                &context.cancellation,
            )?;
            original_digests.push(payload.digest_sha256());
            let NativeStoredPayload::Latent(bundle) = payload.as_ref() else {
                return Err("Missing Hunyuan bundle".into());
            };
            assert!(bundle.noise_mask().is_none() && bundle.batch_indices().is_none());
            let samples = bundle.samples().tensor().ok_or("Missing samples")?;
            let shape: Vec<u64> = serde_json::from_value(expected["output_shape"].clone())?;
            assert_eq!(samples.descriptor().shape(), shape);
            let actual = sim_tensor::generated_native_diffusion::tensor_to_f32(
                compute.backend(),
                samples,
                &execution,
            )?;
            let expected: Vec<f32> = serde_json::from_value(expected["output_values"].clone())?;
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() <= 2e-5 * expected.abs().max(1.0));
            }
        }
        let cached = execute(&plan, 1)?;
        assert_eq!(cached.report.state, AttemptState::Succeeded);
        assert_eq!(cached.report.cache_hits, 2);
        assert_eq!(cached.report.outputs, result.report.outputs);
        let mut changed = plan.clone();
        changed
            .nodes
            .get_mut(&producer)
            .ok_or("Missing producer")?
            .inputs
            .insert(
                "variant".to_owned(),
                InputBinding::Literal { value: scalar(2) },
            );
        let changed = execute(&changed, 2)?;
        assert_eq!(
            changed.report.state,
            AttemptState::Succeeded,
            "{:?}",
            changed.report.error
        );
        assert_eq!(changed.report.cache_hits, 0);
        assert_ne!(
            changed.report.outputs.get(&upscale),
            result.report.outputs.get(&upscale)
        );
        let Some(NativeValue::List { values }) = changed
            .report
            .outputs
            .get(&upscale)
            .and_then(|outputs| outputs.first())
        else {
            return Err("Missing changed Hunyuan outputs".into());
        };
        assert_eq!(values.len(), original_digests.len());
        for (value, original) in values.iter().zip(&original_digests) {
            let NativeValue::Handle { value: handle } = value else {
                return Err("Missing changed handle".into());
            };
            let payload = context.handle_store().resolve(
                handle,
                handle.handle_type(),
                &context.cancellation,
            )?;
            assert_ne!(
                &payload.digest_sha256(),
                original,
                "Changed model must change actual latent content"
            );
        }
        let recovered = execute(&plan, 3)?;
        assert_eq!(recovered.report.state, AttemptState::Succeeded);
        assert_eq!(recovered.report.cache_hits, 2);
        assert_eq!(recovered.report.outputs, result.report.outputs);
        let count = generation.len();
        let mut invalid = plan.clone();
        invalid
            .nodes
            .get_mut(&upscale)
            .ok_or("Missing upscale")?
            .inputs
            .insert(
                "width".to_owned(),
                InputBinding::Literal {
                    value: NativeValue::List {
                        values: vec![scalar(64), scalar(16385)],
                    },
                },
            );
        let rejected = execute(&invalid, 4)?;
        assert_eq!(rejected.report.state, AttemptState::Failed);
        assert!(!rejected.report.outputs.contains_key(&upscale));
        assert_eq!(generation.len(), count);
    }
    Ok(())
}

#[test]
fn task628_registered_hunyuan_upscale_matches_source_model_and_noop() -> Result<(), Box<dyn Error>>
{
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_hunyuan_upscale.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("HunyuanVideo15LatentUpscaleWithModel")
        .ok_or("Missing Hunyuan node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let execution_context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x628800))),
        CancellationToken::default(),
        64 * 1024 * 1024,
    )?;
    let compute = execution_context.compute_session()?;
    let execution = compute.execution_context(&execution_context)?;
    let (model, input_bundle) = hunyuan_fixture_payloads(&execution_context, &fixture, 1.0)?;
    let model = execution_context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(model)),
        &execution_context.cancellation,
    )?;
    let samples = execution_context.handle_store().publish(
        NativeStoredPayload::Latent(input_bundle.clone()),
        &execution_context.cancellation,
    )?;
    let cases = fixture["cases"].as_array().ok_or("Missing cases")?;
    assert_eq!(cases.len(), 60);
    for (index, case) in cases.iter().enumerate() {
        let mut inputs = case["inputs"]
            .as_object()
            .ok_or("Missing source inputs")?
            .iter()
            .map(|(name, value)| {
                Ok((
                    name.clone(),
                    if let Some(value) = value.as_str() {
                        NativeValue::Primitive {
                            value: NativePrimitive::String(value.to_owned()),
                        }
                    } else {
                        scalar(value.as_u64().ok_or("Missing dimension")?)
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        inputs.insert(
            "model".to_owned(),
            NativeValue::Handle {
                value: model.clone(),
            },
        );
        inputs.insert(
            "samples".to_owned(),
            NativeValue::Handle {
                value: samples.clone(),
            },
        );
        let before_inputs = serde_json::to_vec(&inputs)?;
        let dependencies = node.cache_dependencies(&execution_context, &inputs)?;
        assert_eq!(dependencies.artifact_digests.len(), 2);
        let before_count = generation.len();
        let NativeNodeOutcome::Values {
            outputs,
            effects,
            ui,
        } = smol::block_on(node.execute(execution_context.clone(), inputs.clone()))?
        else {
            return Err("Unexpected upscale outcome".into());
        };
        assert_eq!(outputs.len(), 1);
        assert!(ui.is_none() && effects.is_empty());
        let NativeValue::Handle {
            value: output_handle,
        } = outputs.first().ok_or("Missing output")?
        else {
            return Err("Missing output handle".into());
        };
        let alias = case["aliases_input"]
            .as_bool()
            .ok_or("Missing alias flag")?;
        assert_eq!(output_handle == &samples, alias, "{index}");
        assert_eq!(generation.len(), before_count + usize::from(!alias));
        let output = execution_context.handle_store().resolve(
            output_handle,
            output_handle.handle_type(),
            &execution_context.cancellation,
        )?;
        let NativeStoredPayload::Latent(output) = output.as_ref() else {
            return Err("Expected LATENT".into());
        };
        if alias {
            assert_eq!(output.projection(), input_bundle.projection());
        } else {
            assert!(output.noise_mask().is_none() && output.batch_indices().is_none());
            assert_eq!(
                output.metadata(),
                &sim_tensor::NativeLatentMetadata::default()
            );
        }
        let tensor = output.samples().tensor().ok_or("Missing tensor")?;
        let shape: Vec<u64> = serde_json::from_value(case["output_shape"].clone())?;
        assert_eq!(tensor.descriptor().shape(), shape, "{index}");
        assert_eq!(tensor.descriptor().dtype(), sim_tensor::DType::F32);
        assert_eq!(tensor.descriptor().device(), sim_tensor::DeviceId::CPU);
        let actual = sim_tensor::generated_native_diffusion::tensor_to_f32(
            compute.backend(),
            tensor,
            &execution,
        )?;
        let expected: Vec<f32> = serde_json::from_value(case["output_values"].clone())?;
        assert_eq!(actual.len(), expected.len());
        for (position, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                "{index}/{position}: {actual} vs {expected}"
            );
        }
        assert_eq!(serde_json::to_vec(&inputs)?, before_inputs);
        assert_eq!(input_bundle.batch_indices(), Some([11].as_slice()));
        assert_eq!(input_bundle.metadata().spatial_downscale_ratio(), Some(16));
        if !alias {
            for reject in [true, false] {
                let count = generation.len();
                let publication_context = context(
                    backend.clone(),
                    &authority,
                    Arc::new(PublicationBoundary {
                        inner: generation.handle_store_for_attempt(execution_context.attempt_id),
                        remaining: std::sync::atomic::AtomicUsize::new(1),
                        reject,
                    }),
                    CancellationToken::default(),
                    64 * 1024 * 1024,
                )?;
                let error = smol::block_on(node.execute(publication_context, inputs.clone()))
                    .err()
                    .ok_or("Hunyuan publication fault succeeded")?;
                if !reject {
                    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
                }
                assert_eq!(generation.len(), count, "{index}: partial Hunyuan output");
            }
        }
    }
    Ok(())
}

#[test]
fn task628_empty_latent_validation_cancellation_and_allocation_are_atomic()
-> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0159/source_empty_latents.json"
    ))?;
    let registry = generated_native_node_registry_projection(None)?;
    let mut visited = std::collections::BTreeSet::new();
    for case in fixture["cases"].as_array().ok_or("Missing cases")? {
        let name = case["class_type"].as_str().ok_or("Missing class")?;
        if name == "EmptyLatentImage" || !visited.insert(name) {
            continue;
        }
        let node = registry.node(name).ok_or("Missing empty node")?;
        let valid = case["inputs"]
            .as_object()
            .ok_or("Missing inputs")?
            .iter()
            .map(|(key, value)| {
                Ok((
                    key.clone(),
                    if key == "seconds" {
                        NativeValue::Primitive {
                            value: NativePrimitive::Number(
                                value.as_f64().ok_or("Missing seconds")?,
                            ),
                        }
                    } else {
                        scalar(value.as_u64().ok_or("Missing integer")?)
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let backend = Arc::new(backend);
        let generation = NativeHandleStoreGeneration::new()?;
        let attempt = AttemptId(Uuid::from_u128(0x628900));
        let execution_context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(attempt),
            CancellationToken::default(),
            16 * 1024 * 1024,
        )?;
        let mut invalid_inputs = Vec::new();
        for (key, value) in &valid {
            let mut missing = valid.clone();
            missing.remove(key);
            invalid_inputs.push(missing);
            for invalid_value in [
                NativeValue::List { values: Vec::new() },
                NativeValue::Primitive {
                    value: NativePrimitive::String("wrong type".to_owned()),
                },
                NativeValue::List {
                    values: vec![value.clone(), scalar(1_000_000)],
                },
            ] {
                let mut malformed = valid.clone();
                malformed.insert(key.clone(), invalid_value);
                invalid_inputs.push(malformed);
            }
            let mut mapped = valid.clone();
            mapped.insert(
                key.clone(),
                NativeValue::List {
                    values: vec![value.clone(), value.clone()],
                },
            );
            node.cache_change_token(&mapped)?;
        }
        let mut extra = valid.clone();
        extra.insert("unexpected".to_owned(), scalar(1));
        invalid_inputs.push(extra);
        for inputs in invalid_inputs {
            assert!(node.cache_change_token(&inputs).is_err(), "{name}");
            assert!(
                node.demanded_lazy_inputs(&execution_context, &inputs)
                    .is_err(),
                "{name}"
            );
            assert!(
                smol::block_on(node.execute(execution_context.clone(), inputs)).is_err(),
                "{name}"
            );
            assert_eq!(generation.len(), 0);
        }
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let cancelled = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(attempt),
            cancellation,
            16 * 1024 * 1024,
        )?;
        let error = smol::block_on(node.execute(cancelled, valid.clone()))
            .err()
            .ok_or("Cancelled node succeeded")?;
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(generation.len(), 0);
        for reject in [true, false] {
            let publication_context = context(
                backend.clone(),
                &authority,
                Arc::new(PublicationBoundary {
                    inner: generation.handle_store_for_attempt(attempt),
                    remaining: std::sync::atomic::AtomicUsize::new(1),
                    reject,
                }),
                CancellationToken::default(),
                16 * 1024 * 1024,
            )?;
            let error = smol::block_on(node.execute(publication_context, valid.clone()))
                .err()
                .ok_or("Publication fault succeeded")?;
            if !reject {
                assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
            }
            assert_eq!(generation.len(), 0, "{name}: publication must be atomic");
        }
        smol::block_on(node.execute(execution_context.clone(), valid.clone()))?;
        assert_eq!(generation.len(), 1);
        drop(execution_context);

        let (limited_backend, limited_authority) = CpuWorkspaceAuthority::create_backend(1)?;
        let limited_generation = NativeHandleStoreGeneration::new()?;
        let limited = context(
            Arc::new(limited_backend),
            &limited_authority,
            limited_generation.handle_store_for_attempt(attempt),
            CancellationToken::default(),
            1,
        )?;
        assert!(
            smol::block_on(node.execute(limited, valid)).is_err(),
            "{name}"
        );
        assert_eq!(limited_generation.len(), 0);
    }
    assert_eq!(visited.len(), 7);
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
