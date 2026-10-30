use serde_json::Value;
use sim_model::{NativeLatentOperation, NativeModelPayload};
use sim_nodes::{
    NativeHandleStore, NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices, NativePrimitive,
    NativeStoredModelPayload, NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, NativeLatentBundle, NativeLatentMetadata,
    NativeTensorPayload, NativeTensorRole, StreamId,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;

fn context(
    backend: Arc<sim_tensor::CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let attempt = store.attempt_id();
    let node = NodeId("seed-behavior".to_owned());
    let scratch = authority.authorize_workspace(8 * 1024 * 1024)?;
    let identity =
        NativeNodeServiceIdentity::checked(Uuid::from_u128(0x62903), attempt, node.clone())?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x62901)),
        attempt,
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}

#[test]
fn task629_batch_seed_behavior_matches_pinned_source_cases() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0311/seed_behavior.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing source cases")?;
    assert_eq!(cases.len(), 4);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentBatchSeedBehavior")
        .ok_or("Missing native seed node")?;
    let descriptor = registry
        .descriptor("LatentBatchSeedBehavior")
        .ok_or("Missing descriptor")?;
    descriptor.validate_exact_schema_v2()?;
    assert_eq!(descriptor.inputs.len(), 2);
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629100 + index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let shape = [3, 4, 8, 8];
        let samples = tensor_from_f32(
            compute.backend(),
            &shape,
            &vec![0.25; 3 * 4 * 8 * 8],
            &execution,
        )?;
        let mask = tensor_from_f32(
            compute.backend(),
            &[3, 8, 8],
            &vec![1.0; 3 * 8 * 8],
            &execution,
        )?;
        let indices =
            serde_json::from_value::<Option<Vec<u64>>>(case["input_batch_index"].clone())?;
        let latent = Arc::new(NativeLatentBundle::single(
            samples,
            Some(mask),
            indices.clone(),
            NativeLatentMetadata::checked(None, None, Some(8), None)?,
            &execution,
        )?);
        let source_digest = latent.semantic_digest_sha256().to_owned();
        let source = context.handle_store().publish(
            NativeStoredPayload::Latent(latent.clone()),
            &context.cancellation,
        )?;
        let inputs = BTreeMap::from([
            ("samples".to_owned(), NativeValue::Handle { value: source }),
            (
                "seed_behavior".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String(
                        case["seed_behavior"]
                            .as_str()
                            .ok_or("Missing behavior")?
                            .to_owned(),
                    ),
                },
            ),
        ]);
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
        } = smol::block_on(node.execute(context.clone(), inputs.clone()))?
        else {
            return Err("Unexpected output kind".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        assert_eq!(outputs.len(), 1);
        let NativeValue::Handle { value } = &outputs[0] else {
            return Err("Expected output handle".into());
        };
        let result =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(result) = result.as_ref() else {
            return Err("Expected latent bundle".into());
        };
        let expected =
            serde_json::from_value::<Option<Vec<u64>>>(case["expected_batch_index"].clone())?;
        assert_eq!(result.batch_indices(), expected.as_deref(), "case {index}");
        assert_eq!(result.metadata(), latent.metadata());
        assert_eq!(result.noise_mask().is_some(), latent.noise_mask().is_some());
        assert_eq!(
            result
                .samples()
                .tensor()
                .ok_or("Expected tensor")?
                .descriptor()
                .shape(),
            shape
        );
        assert_eq!(latent.batch_indices(), indices.as_deref());
        assert_eq!(latent.semantic_digest_sha256(), source_digest);
        assert_eq!(generation.len(), 2);
    }
    Ok(())
}

#[test]
fn task629_batch_seed_behavior_rejects_unknown_option_and_cancellation_atomically()
-> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentBatchSeedBehavior")
        .ok_or("Missing native seed node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629200)));
    let cancellation = CancellationToken::default();
    let context = context(backend, &authority, store, cancellation.clone())?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let samples = tensor_from_f32(
        compute.backend(),
        &[1, 4, 8, 8],
        &vec![0.0; 4 * 8 * 8],
        &execution,
    )?;
    let latent = Arc::new(NativeLatentBundle::single(
        samples,
        None,
        None,
        NativeLatentMetadata::default(),
        &execution,
    )?);
    let source = context
        .handle_store()
        .publish(NativeStoredPayload::Latent(latent), &context.cancellation)?;
    let mut inputs = BTreeMap::from([
        ("samples".to_owned(), NativeValue::Handle { value: source }),
        (
            "seed_behavior".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::String("unsupported".to_owned()),
            },
        ),
    ]);
    let error = smol::block_on(node.execute(context.clone(), inputs.clone()))
        .err()
        .ok_or("Expected invalid option")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Failure);
    assert_eq!(generation.len(), 1);
    inputs.insert(
        "seed_behavior".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::String("fixed".to_owned()),
        },
    );
    cancellation.cancel();
    let error = smol::block_on(node.execute(context, inputs))
        .err()
        .ok_or("Expected cancellation")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(generation.len(), 0);
    Ok(())
}

#[test]
fn task629_latent_cut_matches_pinned_source_boundaries() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0311/cut.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing cut cases")?;
    assert_eq!(cases.len(), 4);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentCut")
        .ok_or("Missing native LatentCut")?;
    registry
        .descriptor("LatentCut")
        .ok_or("Missing LatentCut descriptor")?
        .validate_exact_schema_v2()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (case_index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629300 + case_index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let samples = tensor_from_f32(
            compute.backend(),
            &[2, 4, 3, 5, 7],
            &vec![0.25; 2 * 4 * 3 * 5 * 7],
            &execution,
        )?;
        let source = Arc::new(NativeLatentBundle::single(
            samples,
            None,
            None,
            NativeLatentMetadata::default(),
            &execution,
        )?);
        let source_digest = source.semantic_digest_sha256().to_owned();
        let handle = context.handle_store().publish(
            NativeStoredPayload::Latent(source.clone()),
            &context.cancellation,
        )?;
        let inputs = BTreeMap::from([
            ("samples".to_owned(), NativeValue::Handle { value: handle }),
            (
                "dim".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String(
                        case["dim"].as_str().ok_or("Missing dimension")?.to_owned(),
                    ),
                },
            ),
            (
                "index".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(case["index"].as_i64().ok_or("Missing index")?),
                },
            ),
            (
                "amount".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(
                        case["amount"].as_i64().ok_or("Missing amount")?,
                    ),
                },
            ),
        ]);
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
            return Err("Unexpected output kind".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
            return Err("Expected handle".into());
        };
        let result =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(result) = result.as_ref() else {
            return Err("Expected latent".into());
        };
        let expected_shape: Vec<u64> = serde_json::from_value(case["expected_shape"].clone())?;
        assert_eq!(
            result
                .samples()
                .tensor()
                .ok_or("Expected tensor")?
                .descriptor()
                .shape(),
            expected_shape
        );
        assert_eq!(
            source
                .samples()
                .tensor()
                .ok_or("Expected source tensor")?
                .descriptor()
                .shape(),
            &[2, 4, 3, 5, 7]
        );
        assert_eq!(source.semantic_digest_sha256(), source_digest);
        assert_eq!(generation.len(), 2);
    }
    Ok(())
}

#[test]
fn task629_latent_crop_matches_pinned_source_boundaries() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0311/crop.json"
    ))?;
    let cases = fixture["cases"].as_array().ok_or("Missing crop cases")?;
    assert_eq!(cases.len(), 4);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentCrop")
        .ok_or("Missing native LatentCrop")?;
    registry
        .descriptor("LatentCrop")
        .ok_or("Missing crop descriptor")?
        .validate_exact_schema_v2()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (case_index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629400 + case_index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let samples = tensor_from_f32(
            compute.backend(),
            &[1, 4, 16, 20],
            &vec![0.25; 4 * 16 * 20],
            &execution,
        )?;
        let source = Arc::new(NativeLatentBundle::single(
            samples,
            None,
            None,
            NativeLatentMetadata::default(),
            &execution,
        )?);
        let source_digest = source.semantic_digest_sha256().to_owned();
        let handle = context.handle_store().publish(
            NativeStoredPayload::Latent(source.clone()),
            &context.cancellation,
        )?;
        let mut inputs =
            BTreeMap::from([("samples".to_owned(), NativeValue::Handle { value: handle })]);
        for name in ["width", "height", "x", "y"] {
            inputs.insert(
                name.to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(
                        case[name].as_i64().ok_or("Missing crop coordinate")?,
                    ),
                },
            );
        }
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
            return Err("Unexpected output kind".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
            return Err("Expected handle".into());
        };
        let result =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(result) = result.as_ref() else {
            return Err("Expected latent".into());
        };
        let expected_shape: Vec<u64> = serde_json::from_value(case["expected_shape"].clone())?;
        assert_eq!(
            result
                .samples()
                .tensor()
                .ok_or("Expected tensor")?
                .descriptor()
                .shape(),
            expected_shape
        );
        assert_eq!(
            source
                .samples()
                .tensor()
                .ok_or("Expected source tensor")?
                .descriptor()
                .shape(),
            &[1, 4, 16, 20]
        );
        assert_eq!(source.semantic_digest_sha256(), source_digest);
        assert_eq!(generation.len(), 2);
    }
    Ok(())
}

#[test]
fn task629_latent_batch_and_concat_match_pinned_source_cases() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0311/batch_concat.json"
    ))?;
    let cases = fixture["cases"]
        .as_array()
        .ok_or("Missing batch/concat cases")?;
    assert_eq!(cases.len(), 5);
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (case_index, case) in cases.iter().enumerate() {
        let class_type = case["class_type"].as_str().ok_or("Missing class type")?;
        let node = registry
            .node(class_type)
            .ok_or("Missing native batch/concat node")?;
        registry
            .descriptor(class_type)
            .ok_or("Missing batch/concat descriptor")?
            .validate_exact_schema_v2()?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629500 + case_index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let mut inputs = BTreeMap::new();
        for (index, name) in ["samples1", "samples2"].iter().enumerate() {
            let key = if index == 0 { "first" } else { "second" };
            let shape: Vec<u64> = serde_json::from_value(case[format!("{key}_shape")].clone())?;
            let count = usize::try_from(shape.iter().product::<u64>())?;
            let samples = tensor_from_f32(
                compute.backend(),
                &shape,
                &vec![0.25 + index as f32; count],
                &execution,
            )?;
            let indices: Option<Vec<u64>> =
                serde_json::from_value(case[format!("{key}_batch_index")].clone())?;
            let latent = if let Some(indices) = &indices
                && indices.len() != usize::try_from(shape[0])?
            {
                let seed_shape = [u64::try_from(indices.len())?, shape[1], shape[2], shape[3]];
                let seed_count = usize::try_from(seed_shape.iter().product::<u64>())?;
                let seed_samples = tensor_from_f32(
                    compute.backend(),
                    &seed_shape,
                    &vec![0.0; seed_count],
                    &execution,
                )?;
                NativeLatentBundle::single(
                    seed_samples,
                    None,
                    Some(indices.clone()),
                    NativeLatentMetadata::default(),
                    &execution,
                )?
                .replaced_samples_source_dictionary(
                    sim_tensor::NativeLatentSamples::Tensor(samples),
                    &execution,
                )?
            } else {
                NativeLatentBundle::single(
                    samples,
                    None,
                    indices,
                    NativeLatentMetadata::default(),
                    &execution,
                )?
            };
            let latent = Arc::new(latent);
            let handle = context
                .handle_store()
                .publish(NativeStoredPayload::Latent(latent), &context.cancellation)?;
            inputs.insert((*name).to_owned(), NativeValue::Handle { value: handle });
        }
        if let Some(dim) = case["dim"].as_str() {
            inputs.insert(
                "dim".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String(dim.to_owned()),
                },
            );
        }
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        node.cache_change_token(&inputs)?;
        assert_eq!(
            node.cache_dependencies(&context, &inputs)?
                .artifact_digests
                .len(),
            2
        );
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = smol::block_on(node.execute(context.clone(), inputs))?
        else {
            return Err("Unexpected output kind".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
            return Err("Expected handle".into());
        };
        let result =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(result) = result.as_ref() else {
            return Err("Expected latent".into());
        };
        let expected_shape: Vec<u64> = serde_json::from_value(case["expected_shape"].clone())?;
        let expected_indices: Option<Vec<u64>> =
            serde_json::from_value(case["expected_batch_index"].clone())?;
        assert_eq!(
            result
                .samples()
                .tensor()
                .ok_or("Expected tensor")?
                .descriptor()
                .shape(),
            expected_shape
        );
        assert_eq!(result.batch_indices(), expected_indices.as_deref());
        assert_eq!(generation.len(), 3);
    }
    Ok(())
}

#[test]
fn task629_cut_to_batch_preserves_source_indices_and_noop_handle() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-latent-comfy-node-0311/cut_to_batch.json"
    ))?;
    let cases = fixture["cases"]
        .as_array()
        .ok_or("Missing cut-to-batch cases")?;
    assert_eq!(cases.len(), 4);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentCutToBatch")
        .ok_or("Missing CutToBatch node")?;
    registry
        .descriptor("LatentCutToBatch")
        .ok_or("Missing CutToBatch descriptor")?
        .validate_exact_schema_v2()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (case_index, case) in cases.iter().enumerate() {
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629600 + case_index as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let shape: Vec<u64> = serde_json::from_value(case["shape"].clone())?;
        let count = usize::try_from(shape.iter().product::<u64>())?;
        let samples = tensor_from_f32(compute.backend(), &shape, &vec![0.5; count], &execution)?;
        let indices: Option<Vec<u64>> = serde_json::from_value(case["batch_index"].clone())?;
        let latent = Arc::new(NativeLatentBundle::single(
            samples,
            None,
            indices,
            NativeLatentMetadata::default(),
            &execution,
        )?);
        let source = context
            .handle_store()
            .publish(NativeStoredPayload::Latent(latent), &context.cancellation)?;
        let inputs = BTreeMap::from([
            (
                "samples".to_owned(),
                NativeValue::Handle {
                    value: source.clone(),
                },
            ),
            (
                "dim".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String(
                        case["dim"].as_str().ok_or("Missing dim")?.to_owned(),
                    ),
                },
            ),
            (
                "slice_size".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(
                        case["slice_size"].as_i64().ok_or("Missing slice_size")?,
                    ),
                },
            ),
        ]);
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
            return Err("Unexpected output kind".into());
        };
        assert!(ui.is_none() && effects.is_empty());
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
            return Err("Expected handle".into());
        };
        let result =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(result) = result.as_ref() else {
            return Err("Expected latent".into());
        };
        let expected_shape: Vec<u64> = serde_json::from_value(case["expected_shape"].clone())?;
        let expected_indices: Option<Vec<u64>> =
            serde_json::from_value(case["expected_batch_index"].clone())?;
        assert_eq!(
            result
                .samples()
                .tensor()
                .ok_or("Expected tensor")?
                .descriptor()
                .shape(),
            expected_shape
        );
        assert_eq!(result.batch_indices(), expected_indices.as_deref());
        result.validate_retained()?;
        if case["returns_original"].as_bool() == Some(true) {
            assert_eq!(value, &source);
            assert_eq!(generation.len(), 1);
        } else {
            assert_ne!(value, &source);
            assert_eq!(generation.len(), 2);
        }
    }
    Ok(())
}

#[test]
fn task629_latent_operation_resource_and_direct_consumer() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentApplyOperation")
        .ok_or("Missing operation consumer")?;
    registry
        .descriptor("LatentApplyOperation")
        .ok_or("Missing operation descriptor")?
        .validate_exact_schema_v2()?;
    assert!(registry.node("LatentOperationTonemapReinhard").is_some());
    assert!(registry.node("LatentOperationSharpen").is_some());
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629900)));
    let context = context(backend, &authority, store, CancellationToken::default())?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let samples = tensor_from_f32(
        compute.backend(),
        &[1, 1, 1, 3],
        &[1.0, 2.0, 3.0],
        &execution,
    )?;
    let latent = Arc::new(NativeLatentBundle::single(
        samples,
        None,
        Some(vec![42]),
        NativeLatentMetadata::default(),
        &execution,
    )?);
    let source_digest = latent.semantic_digest_sha256().to_owned();
    let source = context.handle_store().publish(
        NativeStoredPayload::Latent(latent.clone()),
        &context.cancellation,
    )?;
    let operation = Arc::new(NativeLatentOperation::tonemap_reinhard(1.0)?);
    let resource = Arc::new(NativeModelPayload::latent_operation(operation)?);
    let operation = context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(
            resource,
        )?)),
        &context.cancellation,
    )?;
    let inputs = BTreeMap::from([
        ("samples".to_owned(), NativeValue::Handle { value: source }),
        (
            "operation".to_owned(),
            NativeValue::Handle { value: operation },
        ),
    ]);
    assert_eq!(
        node.cache_dependencies(&context, &inputs)?
            .artifact_digests
            .len(),
        2
    );
    let NativeNodeOutcome::Values { outputs, .. } =
        smol::block_on(node.execute(context.clone(), inputs))?
    else {
        return Err("Expected values".into());
    };
    let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
        return Err("Expected handle".into());
    };
    let payload =
        context
            .handle_store()
            .resolve(value, value.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Latent(output) = payload.as_ref() else {
        return Err("Expected latent".into());
    };
    let values = tensor_to_f32(
        compute.backend(),
        output.samples().tensor().ok_or("Missing samples")?,
        &execution,
    )?;
    for (actual, expected) in values.iter().zip([0.875, 14.0 / 9.0, 2.1]) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }
    assert_eq!(output.batch_indices(), Some([42].as_slice()));
    assert_eq!(latent.semantic_digest_sha256(), source_digest);
    assert_eq!(generation.len(), 3);
    assert!(NativeLatentOperation::tonemap_reinhard(101.0).is_err());
    let sharpen = NativeLatentOperation::sharpen(1, 1.0, 0.1)?;
    let impulse = tensor_from_f32(
        compute.backend(),
        &[1, 1, 3, 3],
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &execution,
    )?;
    let sharpened = sharpen.apply(&impulse, compute.backend(), &execution)?;
    let sharpened = tensor_to_f32(compute.backend(), &sharpened, &execution)?;
    let gaussian_center = 1.0_f32 / (1.0 + 4.0 * (-0.5_f32).exp() + 4.0 * (-1.0_f32).exp());
    let expected_center = 2.0 - gaussian_center;
    assert!((sharpened[4] - expected_center).abs() < 1.0e-5);
    assert!(
        sharpened
            .iter()
            .enumerate()
            .all(|(index, value)| index == 4 || value.abs() < 1.0e-6)
    );
    Ok(())
}

#[test]
fn task629_latent_composites_preserve_destination_dictionary() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (offset, class_type) in ["LatentComposite", "LatentCompositeMasked"]
        .iter()
        .enumerate()
    {
        let node = registry.node(class_type).ok_or("Missing composite")?;
        registry
            .descriptor(class_type)
            .ok_or("Missing composite descriptor")?
            .validate_exact_schema_v2()?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629a00 + offset as u128)));
        let context = context(
            backend.clone(),
            &authority,
            store,
            CancellationToken::default(),
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let destination = Arc::new(NativeLatentBundle::single(
            tensor_from_f32(compute.backend(), &[1, 1, 4, 4], &[0.0; 16], &execution)?,
            None,
            Some(vec![7]),
            NativeLatentMetadata::default(),
            &execution,
        )?);
        let source = Arc::new(NativeLatentBundle::single(
            tensor_from_f32(compute.backend(), &[1, 1, 2, 2], &[1.0; 4], &execution)?,
            None,
            None,
            NativeLatentMetadata::default(),
            &execution,
        )?);
        let destination_digest = destination.semantic_digest_sha256().to_owned();
        let destination_handle = context.handle_store().publish(
            NativeStoredPayload::Latent(destination.clone()),
            &context.cancellation,
        )?;
        let source_handle = context
            .handle_store()
            .publish(NativeStoredPayload::Latent(source), &context.cancellation)?;
        let mut inputs = BTreeMap::from([
            (
                (if offset == 0 {
                    "samples_to"
                } else {
                    "destination"
                })
                .to_owned(),
                NativeValue::Handle {
                    value: destination_handle,
                },
            ),
            (
                (if offset == 0 {
                    "samples_from"
                } else {
                    "source"
                })
                .to_owned(),
                NativeValue::Handle {
                    value: source_handle,
                },
            ),
            (
                "x".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(8),
                },
            ),
            (
                "y".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(8),
                },
            ),
        ]);
        if offset == 0 {
            inputs.insert(
                "feather".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(16),
                },
            );
        } else {
            inputs.insert(
                "resize_source".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(false),
                },
            );
            let mask = tensor_from_f32(
                compute.backend(),
                &[1, 2, 2],
                &[1.0, 0.0, 0.0, 1.0],
                &execution,
            )?;
            let handle = context.handle_store().publish(
                NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_tensor(
                    NativeTensorRole::Mask,
                    mask,
                )?)),
                &context.cancellation,
            )?;
            inputs.insert("mask".to_owned(), NativeValue::Handle { value: handle });
        }
        let NativeNodeOutcome::Values { outputs, .. } =
            smol::block_on(node.execute(context.clone(), inputs.clone()))?
        else {
            return Err("Expected values".into());
        };
        let NativeValue::Handle { value } = outputs.first().ok_or("Missing output")? else {
            return Err("Expected handle".into());
        };
        let payload =
            context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Latent(output) = payload.as_ref() else {
            return Err("Expected latent".into());
        };
        let values = tensor_to_f32(
            compute.backend(),
            output.samples().tensor().ok_or("Missing tensor")?,
            &execution,
        )?;
        let expected = if offset == 0 {
            [0.25, 0.25, 0.25, 0.25]
        } else {
            [1.0, 0.0, 0.0, 1.0]
        };
        for (index, expected) in [5, 6, 9, 10].into_iter().zip(expected) {
            assert!((values[index] - expected).abs() < 1.0e-6);
        }
        assert_eq!(output.batch_indices(), Some([7].as_slice()));
        assert_eq!(destination.semantic_digest_sha256(), destination_digest);
        if offset == 0 {
            inputs.insert(
                "feather".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(32),
                },
            );
            let NativeNodeOutcome::Values { outputs, .. } =
                smol::block_on(node.execute(context.clone(), inputs))?
            else {
                return Err("Expected oversized feather values".into());
            };
            let NativeValue::Handle { value } = outputs.first().ok_or("Missing feather output")?
            else {
                return Err("Expected handle".into());
            };
            let payload = context.handle_store().resolve(
                value,
                value.handle_type(),
                &context.cancellation,
            )?;
            let NativeStoredPayload::Latent(output) = payload.as_ref() else {
                return Err("Expected feather latent".into());
            };
            let values = tensor_to_f32(
                compute.backend(),
                output.samples().tensor().ok_or("Missing feather tensor")?,
                &execution,
            )?;
            for index in [5, 6, 9, 10] {
                assert!((values[index] - 0.015625).abs() < 1.0e-6, "index {index}");
            }
        }
    }
    Ok(())
}

#[test]
fn task629_cfg_operation_consumer_clones_native_model_with_modifier() -> Result<(), Box<dyn Error>>
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("LatentApplyOperationCFG")
        .ok_or("Missing CFG consumer")?;
    registry
        .descriptor("LatentApplyOperationCFG")
        .ok_or("Missing CFG descriptor")?
        .validate_exact_schema_v2()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(512 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629b00)));
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(256 * 1024 * 1024)?,
        &cancellation,
    );
    let bundle = sim_test_support::NativeDiffusionFixture::checked_in()
        .load_bundle_with_context(backend.clone(), &execution)?;
    let model = Arc::new(NativeModelPayload::sd15_model(bundle.model().clone())?);
    let conditioning = Arc::new(sim_sampler::NativeConditioningPayload::checked_sd15(
        bundle.model_digest(),
        bundle.model(),
        bundle.conditioning().patch_graph().clone(),
        None,
    )?);
    let diffusion = Arc::new(sim_sampler::NativeDiffusionPayload::model(
        model,
        conditioning,
    )?);
    let source = context(backend, &authority, store, cancellation)?;
    let model_handle = source.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::native_diffusion(
            diffusion.clone(),
        )?)),
        &source.cancellation,
    )?;
    let operation = Arc::new(NativeLatentOperation::tonemap_reinhard(1.0)?);
    let operation_handle = source.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(
            Arc::new(NativeModelPayload::latent_operation(operation.clone())?),
        )?)),
        &source.cancellation,
    )?;
    let inputs = BTreeMap::from([
        (
            "model".to_owned(),
            NativeValue::Handle {
                value: model_handle,
            },
        ),
        (
            "operation".to_owned(),
            NativeValue::Handle {
                value: operation_handle,
            },
        ),
    ]);
    let NativeNodeOutcome::Values { outputs, .. } =
        smol::block_on(node.execute(source.clone(), inputs))?
    else {
        return Err("Expected values".into());
    };
    let NativeValue::Handle { value } = outputs.first().ok_or("Missing CFG output")? else {
        return Err("Expected handle".into());
    };
    let payload =
        source
            .handle_store()
            .resolve(value, value.handle_type(), &source.cancellation)?;
    let NativeStoredPayload::Model(model) = payload.as_ref() else {
        return Err("Expected model".into());
    };
    let changed = model.diffusion().ok_or("Missing diffusion")?;
    assert_eq!(
        changed.guidance_modifiers().modifiers(),
        &[sim_sampler::NativeGuidanceModifier::LatentOperation {
            operation: operation.kind()
        },]
    );
    assert!(Arc::ptr_eq(
        changed.model_payload(),
        diffusion.model_payload()
    ));
    assert_ne!(changed.digest_sha256(), diffusion.digest_sha256());
    assert_eq!(generation.len(), 3);
    Ok(())
}

#[test]
fn task629_all_ten_nodes_fail_closed_on_cancelled_attempt() -> Result<(), Box<dyn Error>> {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    for (index, class_type) in [
        "LatentApplyOperation",
        "LatentApplyOperationCFG",
        "LatentBatch",
        "LatentBatchSeedBehavior",
        "LatentComposite",
        "LatentCompositeMasked",
        "LatentConcat",
        "LatentCrop",
        "LatentCut",
        "LatentCutToBatch",
    ]
    .into_iter()
    .enumerate()
    {
        let descriptor = registry
            .descriptor(class_type)
            .ok_or("Missing assigned descriptor")?;
        descriptor.validate_exact_schema_v2()?;
        let node = registry
            .node(class_type)
            .ok_or("Missing assigned adapter")?;
        let generation = NativeHandleStoreGeneration::new()?;
        let store = generation
            .handle_store_for_attempt(AttemptId(Uuid::from_u128(0x629c00 + index as u128)));
        let cancellation = CancellationToken::default();
        let context = context(backend.clone(), &authority, store, cancellation.clone())?;
        cancellation.cancel();
        let error = smol::block_on(node.execute(context, BTreeMap::new()))
            .err()
            .ok_or("Cancelled node unexpectedly succeeded")?;
        assert_eq!(
            error.kind,
            NativeNodeFailureKind::Interrupted,
            "{class_type}"
        );
        assert_eq!(generation.len(), 0, "{class_type}");
    }
    Ok(())
}
