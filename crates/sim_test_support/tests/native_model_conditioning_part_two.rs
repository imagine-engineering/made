use sim_model::{
    conditioning::{
        ConditioningConstant, ConditioningEntry, ConditioningEntryOptions,
        ConditioningHookReference, ConditioningIdentity, ConditioningMetadataKey,
        ConditioningMetadataMap, ConditioningMetadataValue, ConditioningReferences,
        ConditioningRegion, ConditioningSet, ConditioningTimestepRange, ConditioningValue,
    },
    generated_native_diffusion::{sd15_latent_format_identity, sd15_model_family_identity},
};
use sim_nodes::{
    NativeHandleStore, NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle,
    NativePrimitive, NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, StreamId, Tensor,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;
type TestResult = Result<(), Box<dyn Error>>;
const BUDGET: u64 = 8 * 1024 * 1024;
fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let node = NodeId("conditioning-part-two".to_owned());
    let scratch = authority.authorize_workspace(budget)?;
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x61703),
        store.attempt_id(),
        node.clone(),
    )?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x61701)),
        store.attempt_id(),
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}
fn identity(namespace: &str) -> Result<ConditioningIdentity, Box<dyn Error>> {
    Ok(ConditioningIdentity::new(
        namespace,
        sd15_model_family_identity()?,
        sd15_latent_format_identity()?,
    )?)
}
fn conditioning(
    context: &NativeNodeContext,
    namespace: &str,
    entries: &[(usize, f32)],
    cross_attention: bool,
) -> Result<ConditioningSet, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let mut output = Vec::new();
    for (index, (tokens, value)) in entries.iter().enumerate() {
        let tokens_u64 = u64::try_from(*tokens)?;
        let tensor = tensor_from_f32(
            compute.backend(),
            &[1, tokens_u64, 2],
            &vec![*value; tokens * 2],
            &execution,
        )?;
        let value = if cross_attention {
            ConditioningValue::cross_attention(tensor)?
        } else {
            ConditioningValue::regular(tensor)?
        };
        let mut metadata = ConditioningMetadataMap::default();
        metadata.insert(
            ConditioningMetadataKey::TimestepRange,
            ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(0.2, 0.8)?),
        )?;
        output.push(ConditioningEntry::checked_with_metadata(
            format!("original.{index}"),
            value,
            ConditioningEntryOptions {
                strength: f64::from(0.5 + index as f32),
                region: Some(ConditioningRegion::percentage(
                    vec![0.5, 0.75],
                    vec![0.0, 0.1],
                )?),
                ..Default::default()
            },
            ConditioningReferences::checked(
                None,
                vec![ConditioningHookReference::checked(format!("hook.{index}"))?],
            )?,
            metadata,
        )?);
    }
    Ok(ConditioningSet::checked(
        identity(namespace)?,
        output,
        &context.cancellation,
    )?)
}
fn publish(
    context: &NativeNodeContext,
    value: ConditioningSet,
) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    Ok(context.handle_store().publish(
        NativeStoredPayload::Conditioning(Arc::new(value)),
        &context.cancellation,
    )?)
}
fn inputs(
    class_type: &str,
    first: &NativeOpaqueHandle,
    second: &NativeOpaqueHandle,
) -> BTreeMap<String, NativeValue> {
    let names = if class_type == "ConditioningCombine" {
        ["conditioning_1", "conditioning_2"]
    } else {
        ["conditioning_to", "conditioning_from"]
    };
    BTreeMap::from([
        (
            names[0].to_owned(),
            NativeValue::Handle {
                value: first.clone(),
            },
        ),
        (
            names[1].to_owned(),
            NativeValue::Handle {
                value: second.clone(),
            },
        ),
    ])
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
    let NativeValue::Handle { value } = outputs.into_iter().next().ok_or("Missing output")? else {
        return Err("Handle output required".into());
    };
    Ok(value)
}
fn resolved(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Arc<ConditioningSet>, Box<dyn Error>> {
    let payload =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Conditioning(value) = payload.as_ref() else {
        return Err("Conditioning required".into());
    };
    Ok(value.clone())
}
fn tensor(entry: &ConditioningEntry) -> Result<&Tensor, Box<dyn Error>> {
    match entry.value() {
        ConditioningValue::Regular(tensor) | ConditioningValue::CrossAttention(tensor) => {
            Ok(tensor)
        }
        _ => Err("Tensor conditioning required".into()),
    }
}
fn tensor_values(context: &NativeNodeContext, tensor: &Tensor) -> Result<Vec<f32>, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    Ok(tensor_to_f32(compute.backend(), tensor, &execution)?.to_vec())
}
#[test]
fn task617_exact_four_descriptors_aliases_mapped_cache_and_pure_outputs() -> TestResult {
    let bindings = (sim_nodes::generated_model_conditioning_02::NATIVE_NODE_BINDINGS)()?;
    assert_eq!(bindings.len(), 5);
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61711))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let handle = publish(
        &context,
        conditioning(&context, "stable", &[(1, 1.0)], true)?,
    )?;
    for (binding, feature) in bindings.iter().zip(["COMFY-NODE-0089", "COMFY-NODE-0090"]) {
        assert_eq!(binding.feature_id(), feature);
        binding.descriptor().validate_exact_schema_v2()?;
        assert!(
            binding
                .descriptor()
                .inputs
                .iter()
                .all(|input| !input.allows_literal
                    && input.cardinality == sim_nodes::NativePortCardinality::Mapped)
        );
        let class_type = &binding.descriptor().class_type;
        assert_eq!(
            sim_nodes::generated_family_node_bindings()?
                .iter()
                .filter(|binding| binding.descriptor().class_type == *class_type)
                .count(),
            1
        );
        let node = registry.node(class_type).ok_or("Missing node")?;
        let mut mapped = inputs(class_type, &handle, &handle);
        for value in mapped.values_mut() {
            *value = NativeValue::List {
                values: vec![value.clone(), value.clone()],
            };
        }
        assert_eq!(
            node.cache_dependencies(&context, &mapped)?
                .artifact_digests
                .len(),
            4
        );
        assert!(node.demanded_lazy_inputs(&context, &mapped)?.is_empty());
    }
    assert_eq!(
        bindings
            .first()
            .ok_or("Combine binding")?
            .presentation()
            .search_aliases,
        [
            "combine",
            "merge conditioning",
            "combine prompts",
            "merge prompts",
            "mix prompts",
            "add prompt"
        ]
    );
    assert!(
        bindings
            .get(1)
            .ok_or("Concat binding")?
            .presentation()
            .search_aliases
            .is_empty()
    );
    Ok(())
}
#[test]
fn task617_combine_preserves_order_metadata_storage_aliases_and_original_identifiers() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningCombine")
        .ok_or("Missing combine")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61721))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let first = conditioning(&context, "stable", &[(1, 1.0), (2, 2.0)], true)?;
    let second = conditioning(&context, "stable", &[(3, 3.0), (1, 4.0)], false)?;
    let first_digest = first.digest().to_owned();
    let second_digest = second.digest().to_owned();
    let first_handle = publish(&context, first.clone())?;
    let second_handle = publish(&context, second.clone())?;
    let before = backend.memory_snapshot().current_bytes;
    let handle = output(smol::block_on(node.execute(
        context.clone(),
        inputs("ConditioningCombine", &first_handle, &second_handle),
    ))?)?;
    let result = resolved(&context, &handle)?;
    assert_eq!(result.entries().len(), 4);
    for (index, (actual, original)) in result
        .entries()
        .iter()
        .zip(first.entries().iter().chain(second.entries()))
        .enumerate()
    {
        assert_eq!(actual.identifier(), format!("combined.{index}"));
        assert_eq!(tensor(actual)?.storage_id(), tensor(original)?.storage_id());
        assert_eq!(actual.options().strength, original.options().strength);
        assert_eq!(actual.options().region, original.options().region);
        assert_eq!(actual.references(), original.references());
        assert_eq!(actual.metadata().len(), original.metadata().len());
        assert!(
            matches!(actual.metadata().get(ConditioningMetadataKey::TimestepRange),Some(ConditioningMetadataValue::TimestepRange(range)) if range.start_percent()==0.2 && range.end_percent()==0.8)
        );
    }
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    assert_eq!(first.digest(), first_digest);
    assert_eq!(second.digest(), second_digest);
    assert_eq!(
        first
            .entries()
            .first()
            .ok_or("First input entry")?
            .identifier(),
        "original.0"
    );
    assert_eq!(
        second
            .entries()
            .first()
            .ok_or("Second input entry")?
            .identifier(),
        "original.0"
    );
    assert_eq!(resolved(&context, &first_handle)?.digest(), first_digest);
    assert_eq!(resolved(&context, &second_handle)?.digest(), second_digest);
    Ok(())
}
#[test]
fn task617_combine_self_alias_duplicates_entries_without_new_tensor_storage() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningCombine")
        .ok_or("Missing combine")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61731))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let original = publish(
        &context,
        conditioning(&context, "stable", &[(1, 7.0)], true)?,
    )?;
    let before = backend.memory_snapshot().current_bytes;
    let result = resolved(
        &context,
        &output(smol::block_on(node.execute(
            context.clone(),
            inputs("ConditioningCombine", &original, &original),
        ))?)?,
    )?;
    assert_eq!(result.entries().len(), 2);
    assert_eq!(
        tensor(result.entries().first().ok_or("First result entry")?)?.storage_id(),
        tensor(result.entries().get(1).ok_or("Second result entry")?)?.storage_id()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    assert_eq!(generation.len(), 2);
    Ok(())
}
#[test]
fn task617_concat_uses_first_source_axis_one_and_preserves_each_target_metadata() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningConcat")
        .ok_or("Missing concat")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61741))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let first = conditioning(&context, "stable", &[(2, 1.0), (3, 2.0)], true)?;
    let first_digest = first.digest().to_owned();
    let second = conditioning(&context, "stable", &[(3, 8.0), (5, 999.0)], true)?;
    let second_digest = second.digest().to_owned();
    let first_handle = publish(&context, first.clone())?;
    let second_handle = publish(&context, second)?;
    let result = resolved(
        &context,
        &output(smol::block_on(node.execute(
            context.clone(),
            inputs("ConditioningConcat", &first_handle, &second_handle),
        ))?)?,
    )?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    for (index, actual) in result.entries().iter().enumerate() {
        let original = first.entries().get(index).ok_or("Original entry")?;
        let tokens = index + 2;
        assert_eq!(
            tensor(actual)?.descriptor().shape(),
            &[1, u64::try_from(tokens + 3)?, 2]
        );
        let expected = [vec![(index + 1) as f32; tokens * 2], vec![8.0; 6]].concat();
        assert_eq!(
            &*tensor_to_f32(compute.backend(), tensor(actual)?, &execution)?,
            expected.as_slice()
        );
        assert_eq!(actual.identifier(), original.identifier());
        assert_eq!(actual.options().strength, original.options().strength);
        assert_eq!(actual.options().region, original.options().region);
        assert_eq!(actual.references(), original.references());
        assert_eq!(actual.metadata().len(), original.metadata().len());
    }
    assert_eq!(resolved(&context, &first_handle)?.digest(), first_digest);
    assert_eq!(resolved(&context, &second_handle)?.digest(), second_digest);
    Ok(())
}
#[test]
fn task617_input_schema_identity_and_unsupported_value_fail_before_publication() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61751))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let first = publish(
        &context,
        conditioning(&context, "first", &[(1, 1.0)], true)?,
    )?;
    let second = publish(
        &context,
        conditioning(&context, "second", &[(1, 1.0)], true)?,
    )?;
    let before = generation.len();
    for class_type in ["ConditioningCombine", "ConditioningConcat"] {
        let node = registry.node(class_type).ok_or("Missing node")?;
        let actual = inputs(class_type, &first, &second);
        let error = smol::block_on(node.execute(context.clone(), actual.clone()))
            .err()
            .ok_or("Identity mismatch succeeded")?;
        assert_eq!(error.code, "unsupported_conditioning_representation");
        for value in [
            NativeValue::Primitive {
                value: NativePrimitive::String("wrong".to_owned()),
            },
            NativeValue::List { values: vec![] },
        ] {
            let mut malformed = actual.clone();
            let key = malformed.keys().next().ok_or("Input key")?.clone();
            malformed.insert(key, value);
            assert!(node.cache_change_token(&malformed).is_err());
            assert!(smol::block_on(node.execute(context.clone(), malformed)).is_err());
        }
        let mut unknown = actual;
        unknown.insert(
            "unknown".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(true),
            },
        );
        assert!(node.cache_change_token(&unknown).is_err());
    }
    let constant = ConditioningSet::checked(
        identity("first")?,
        vec![ConditioningEntry::checked(
            "constant",
            ConditioningValue::constant(ConditioningConstant::Null)?,
            ConditioningEntryOptions::default(),
        )?],
        &context.cancellation,
    )?;
    let constant = publish(&context, constant)?;
    let node = registry
        .node("ConditioningConcat")
        .ok_or("Missing concat")?;
    let error = smol::block_on(node.execute(
        context.clone(),
        inputs("ConditioningConcat", &first, &constant),
    ))
    .err()
    .ok_or("Constant concat succeeded")?;
    assert_eq!(error.code, "unsupported_conditioning_representation");
    for handle in [&first, &second, &constant] {
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    }
    assert_eq!(generation.len(), before + 1);
    Ok(())
}
#[test]
fn task617_stale_foreign_handles_and_new_generation_recovery_are_real() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningCombine")
        .ok_or("Missing combine")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context_old = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61761))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let old = publish(
        &context_old,
        conditioning(&context_old, "stable", &[(1, 1.0)], true)?,
    )?;
    for other_generation in [
        NativeHandleStoreGeneration::new()?,
        generation.fresh_generation()?,
    ] {
        let context = context(
            backend.clone(),
            &authority,
            other_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61762))),
            CancellationToken::default(),
            BUDGET,
        )?;
        assert!(
            node.cache_dependencies(&context, &inputs("ConditioningCombine", &old, &old))
                .is_err()
        );
        assert!(
            smol::block_on(
                node.execute(context.clone(), inputs("ConditioningCombine", &old, &old))
            )
            .is_err()
        );
        assert!(other_generation.is_empty());
        let fresh = publish(
            &context,
            conditioning(&context, "stable", &[(1, 2.0)], true)?,
        )?;
        let result = output(smol::block_on(node.execute(
            context.clone(),
            inputs("ConditioningCombine", &fresh, &fresh),
        ))?)?;
        assert_eq!(resolved(&context, &result)?.entries().len(), 2);
    }
    context_old
        .handle_store()
        .revoke(&old, &context_old.cancellation)?;
    assert!(
        smol::block_on(node.execute(context_old, inputs("ConditioningCombine", &old, &old)))
            .is_err()
    );
    Ok(())
}
#[test]
fn task617_precancel_and_capacity_rejection_leave_no_output_or_tensor_leak() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    for class_type in ["ConditioningCombine", "ConditioningConcat"] {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
        let backend = Arc::new(backend);
        let generation = NativeHandleStoreGeneration::with_capacities(2, usize::try_from(BUDGET)?)?;
        let cancellation = CancellationToken::default();
        let context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61771))),
            cancellation.clone(),
            BUDGET,
        )?;
        let first = publish(
            &context,
            conditioning(&context, "stable", &[(2, 1.0)], true)?,
        )?;
        let second = publish(
            &context,
            conditioning(&context, "stable", &[(3, 2.0)], true)?,
        )?;
        let before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        let node = registry.node(class_type).ok_or("Missing node")?;
        assert!(
            smol::block_on(node.execute(context.clone(), inputs(class_type, &first, &second)))
                .is_err()
        );
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        cancellation.cancel();
        let error =
            smol::block_on(node.execute(context.clone(), inputs(class_type, &first, &second)))
                .err()
                .ok_or("Cancelled operation succeeded")?;
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
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
#[test]
fn task617_concat_backend_resident_exhaustion_leaves_inputs_and_memory_intact() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningConcat")
        .ok_or("Missing concat")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let store = generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61781)));
    let setup = context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        BUDGET,
    )?;
    let first = publish(
        &setup,
        conditioning(&setup, "stable", &[(1024, 1.0)], true)?,
    )?;
    let second = publish(
        &setup,
        conditioning(&setup, "stable", &[(1024, 2.0)], true)?,
    )?;
    let compute = setup.compute_session()?;
    let execution = compute.execution_context(&setup)?;
    let remaining = BUDGET
        .checked_sub(backend.memory_snapshot().current_bytes)
        .ok_or("Backend accounting exceeds budget")?;
    let lease = backend.reserve_workspace(&execution, remaining)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(setup.clone(), inputs("ConditioningConcat", &first, &second)))
            .is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    drop(lease);
    let recovered = output(smol::block_on(
        node.execute(setup.clone(), inputs("ConditioningConcat", &first, &second)),
    )?)?;
    setup
        .handle_store()
        .revoke(&recovered, &CancellationToken::default())?;
    assert_eq!(generation.len(), before.0);
    assert_eq!(generation.resident_bytes(), before.1);
    Ok(())
}

#[test]
fn task617_concat_late_shape_failure_drops_earlier_allocations_without_publishing() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningConcat")
        .ok_or("Missing concat")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61791))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let target = conditioning(&context, "stable", &[(2, 1.0)], true)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let incompatible = tensor_from_f32(compute.backend(), &[1, 2, 3], &[4.0; 6], &execution)?;
    let mut entries = target.entries().to_vec();
    entries.push(ConditioningEntry::checked(
        "later.incompatible",
        ConditioningValue::cross_attention(incompatible)?,
        ConditioningEntryOptions::default(),
    )?);
    let target = publish(
        &context,
        ConditioningSet::checked(target.identity().clone(), entries, &context.cancellation)?,
    )?;
    let source = publish(
        &context,
        conditioning(&context, "stable", &[(3, 8.0)], true)?,
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(
            context.clone(),
            inputs("ConditioningConcat", &target, &source)
        ))
        .is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    assert_eq!(resolved(&context, &target)?.entries().len(), 2);
    Ok(())
}

fn vision_inputs(
    model: &NativeOpaqueHandle,
    image: &NativeOpaqueHandle,
    crop: &str,
) -> BTreeMap<String, NativeValue> {
    BTreeMap::from([
        (
            "clip_vision".to_owned(),
            NativeValue::Handle {
                value: model.clone(),
            },
        ),
        (
            "image".to_owned(),
            NativeValue::Handle {
                value: image.clone(),
            },
        ),
        (
            "crop".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::String(crop.to_owned()),
            },
        ),
    ])
}
fn vision_input_handles(
    context: &NativeNodeContext,
    batch: usize,
) -> Result<(NativeOpaqueHandle, NativeOpaqueHandle), Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let resource = sim_model::clip_vision::deterministic_reduced_clip_vision_encode_fixture(
        compute.backend(),
        &execution,
    )?;
    let model = sim_model::NativeModelPayload::clip_vision(Arc::new(resource))?;
    let model = sim_nodes::NativeStoredModelPayload::model_resource(Arc::new(model))?;
    let model = context.handle_store().publish(
        NativeStoredPayload::Model(Arc::new(model)),
        &context.cancellation,
    )?;
    let mut values = Vec::new();
    for batch_index in 0..batch {
        for height in 0..3 {
            for width in 0..5 {
                for channel in 0..3 {
                    values.push(
                        ((batch_index * 7 + height * 13 + width * 3 + channel * 5) % 23) as f32
                            / 22.0,
                    );
                }
            }
        }
    }
    let image = tensor_from_f32(
        compute.backend(),
        &[u64::try_from(batch)?, 3, 5, 3],
        &values,
        &execution,
    )?;
    let image = sim_tensor::NativeTensorPayload::from_image(
        sim_tensor::NativeTensorRole::Image,
        sim_tensor::ImageTensor::from_tensor(image)?,
    )?;
    let image = context.handle_store().publish(
        NativeStoredPayload::Tensor(Arc::new(image)),
        &context.cancellation,
    )?;
    Ok((model, image))
}
fn vision_output(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Arc<sim_model::clip_vision::ClipVisionOutput>, Box<dyn Error>> {
    let payload =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::ClipVisionOutput(output) = payload.as_ref() else {
        return Err("Canonical vision output required".into());
    };
    Ok(output.clone())
}
#[test]
fn task617_vision_exact_source_descriptor_crop_schema_and_output_projection() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let binding = (sim_nodes::generated_model_conditioning_02::NATIVE_NODE_BINDINGS)()?
        .into_iter()
        .find(|binding| binding.descriptor().class_type == "CLIPVisionEncode")
        .ok_or("Missing vision binding")?;
    assert_eq!(binding.feature_id(), "COMFY-NODE-0075");
    binding.descriptor().validate_exact_schema_v2()?;
    assert_eq!(binding.descriptor().inputs.len(), 3);
    for input in &binding.descriptor().inputs {
        assert_eq!(input.cardinality, sim_nodes::NativePortCardinality::Mapped);
        assert_eq!(input.allows_literal, input.name == "crop");
    }
    assert_eq!(binding.descriptor().outputs.len(), 1);
    assert_eq!(
        sim_nodes::generated_family_node_bindings()?
            .iter()
            .filter(|binding| binding.descriptor().class_type == "CLIPVisionEncode")
            .count(),
        1
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617751))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let (model, image) = vision_input_handles(&context, 1)?;
    let node = registry
        .node("CLIPVisionEncode")
        .ok_or("Missing vision node")?;
    for crop in ["other", "CENTER", "", " center"] {
        assert!(
            node.demanded_lazy_inputs(&context, &vision_inputs(&model, &image, crop))
                .is_err()
        );
    }
    let mut bad = vision_inputs(&model, &image, "center");
    bad.insert(
        "crop".to_owned(),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(1),
        },
    );
    assert!(node.cache_dependencies(&context, &bad).is_err());
    assert_eq!(generation.len(), 2);
    Ok(())
}
#[test]
fn task617_vision_crop_center_none_preserve_all_batches_and_delegate_exact_owner_output()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("CLIPVisionEncode")
        .ok_or("Missing vision node")?;
    for batch in [1, 2] {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
        let backend = Arc::new(backend);
        let generation = NativeHandleStoreGeneration::new()?;
        let context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617752))),
            CancellationToken::default(),
            BUDGET,
        )?;
        let (model, image) = vision_input_handles(&context, batch)?;
        let model_resolved =
            context
                .handle_store()
                .resolve(&model, model.handle_type(), &context.cancellation)?;
        let image_resolved =
            context
                .handle_store()
                .resolve(&image, image.handle_type(), &context.cancellation)?;
        let NativeStoredPayload::Model(model_payload) = model_resolved.as_ref() else {
            return Err("Model required".into());
        };
        let NativeStoredPayload::Tensor(image_payload) = image_resolved.as_ref() else {
            return Err("Image required".into());
        };
        let resource = model_payload
            .model_payload()
            .clip_vision_resource()
            .ok_or("Vision backing required")?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        for crop in ["center", "none"] {
            let inputs = vision_inputs(&model, &image, crop);
            let dependencies = node.cache_dependencies(&context, &inputs)?;
            assert_eq!(dependencies.artifact_digests.len(), 2);
            let pixels = resource.preprocess(
                compute.backend(),
                image_payload.tensor(),
                crop == "center",
                &execution,
            )?;
            let expected = resource.execution_session(&context.cancellation)?.forward(
                compute.backend(),
                &pixels,
                sim_model::clip_vision::ClipVisionIntermediate::Layer(-2),
                &execution,
            )?;
            let handle = output(smol::block_on(node.execute(context.clone(), inputs))?)?;
            assert_eq!(handle.handle_type().type_id, "CLIP_VISION_OUTPUT");
            let actual = vision_output(&context, &handle)?;
            actual.validate()?;
            assert_eq!(
                actual.semantic_digest_sha256(),
                expected.semantic_digest_sha256()
            );
            assert_eq!(
                actual.last_hidden_state.descriptor().shape(),
                &[u64::try_from(batch)?, 5, 2]
            );
            assert_eq!(
                actual
                    .intermediate
                    .as_ref()
                    .ok_or("Source penultimate missing")?
                    .descriptor()
                    .shape(),
                &[u64::try_from(batch)?, 5, 2]
            );
            assert_eq!(
                actual.image_embeds.descriptor().shape(),
                &[u64::try_from(batch)?, 2]
            );
            assert_eq!(actual.image_sizes(), vec![[3, 2, 2]; batch].as_slice());
            assert!(actual.projected_intermediate.is_none());
            context
                .handle_store()
                .revoke(&handle, &context.cancellation)?;
            drop(actual);
            drop(expected);
            drop(pixels);
            assert_eq!(
                (
                    generation.len(),
                    generation.resident_bytes(),
                    backend.memory_snapshot().current_bytes
                ),
                before
            );
            assert_eq!(
                node.cache_dependencies(&context, &vision_inputs(&model, &image, crop))?
                    .artifact_digests,
                dependencies.artifact_digests
            );
        }
        let center =
            resource.preprocess(compute.backend(), image_payload.tensor(), true, &execution)?;
        let noncrop =
            resource.preprocess(compute.backend(), image_payload.tensor(), false, &execution)?;
        let center_values = tensor_values(&context, &center)?;
        let noncrop_values = tensor_values(&context, &noncrop)?;
        assert_ne!(&*center_values, &*noncrop_values);
    }
    Ok(())
}
#[test]
fn task617_vision_wrong_port_foreign_stale_and_recovery_fail_without_output() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("CLIPVisionEncode")
        .ok_or("Missing vision node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context_old = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617753))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let (model, image) = vision_input_handles(&context_old, 1)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(context_old.clone(), vision_inputs(&image, &model, "center")))
            .is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    for other in [
        NativeHandleStoreGeneration::new()?,
        generation.fresh_generation()?,
    ] {
        let context_new = context(
            backend.clone(),
            &authority,
            other.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617754))),
            CancellationToken::default(),
            BUDGET,
        )?;
        assert!(
            node.cache_dependencies(&context_new, &vision_inputs(&model, &image, "center"))
                .is_err()
        );
        assert!(
            smol::block_on(
                node.execute(context_new.clone(), vision_inputs(&model, &image, "center"))
            )
            .is_err()
        );
        assert!(other.is_empty());
        let (fresh_model, fresh_image) = vision_input_handles(&context_new, 1)?;
        let handle = output(smol::block_on(node.execute(
            context_new.clone(),
            vision_inputs(&fresh_model, &fresh_image, "none"),
        ))?)?;
        vision_output(&context_new, &handle)?.validate()?;
    }
    context_old
        .handle_store()
        .revoke(&image, &context_old.cancellation)?;
    assert!(
        smol::block_on(node.execute(context_old, vision_inputs(&model, &image, "center"))).is_err()
    );
    Ok(())
}
#[test]
fn task617_vision_precancel_and_output_capacity_rollback_preserve_inputs_and_memory() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("CLIPVisionEncode")
        .ok_or("Missing vision node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacities(2, usize::try_from(BUDGET)?)?;
    let cancellation = CancellationToken::default();
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617755))),
        cancellation.clone(),
        BUDGET,
    )?;
    let (model, image) = vision_input_handles(&context, 1)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(context.clone(), vision_inputs(&model, &image, "center")))
            .is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    cancellation.cancel();
    let error =
        smol::block_on(node.execute(context.clone(), vision_inputs(&model, &image, "none")))
            .err()
            .ok_or("Cancelled execution succeeded")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    let inspection_cancellation = CancellationToken::default();
    for handle in [&model, &image] {
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &inspection_cancellation)?;
    }
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
fn task617_vision_workspace_exhaustion_drops_temporaries_without_output() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("CLIPVisionEncode")
        .ok_or("Missing vision node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let attempt = AttemptId(Uuid::from_u128(0x617756));
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        BUDGET,
    )?;
    let (model, image) = vision_input_handles(&setup, 2)?;
    let limited = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        1,
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(limited, vision_inputs(&model, &image, "center"))).is_err()
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

fn multiply_inputs(handle: &NativeOpaqueHandle, multiplier: f64) -> BTreeMap<String, NativeValue> {
    BTreeMap::from([
        (
            "conditioning".to_owned(),
            NativeValue::Handle {
                value: handle.clone(),
            },
        ),
        (
            "multiplier".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(multiplier),
            },
        ),
    ])
}
#[test]
fn task617_multiply_source_descriptor_aliases_range_and_scalar_cache_boundary() -> TestResult {
    let bindings = (sim_nodes::generated_model_conditioning_02::NATIVE_NODE_BINDINGS)()?;
    let binding = bindings
        .iter()
        .find(|binding| binding.descriptor().class_type == "ConditioningMultiply")
        .ok_or("Multiply binding")?;
    assert_eq!(binding.feature_id(), "COMFY-NODE-0091");
    binding.descriptor().validate_exact_schema_v2()?;
    assert_eq!(
        binding.presentation().search_aliases,
        [
            "scale conditioning",
            "scale prompt",
            "multiply conditioning",
            "multiply prompt"
        ]
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        Arc::new(backend),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617911))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let handle = publish(
        &context,
        conditioning(&context, "stable", &[(2, 1.0)], true)?,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningMultiply")
        .ok_or("Multiply node")?;
    for scalar in [-100.0001, 100.0001, f64::NAN, f64::INFINITY] {
        assert!(
            node.demanded_lazy_inputs(&context, &multiply_inputs(&handle, scalar))
                .is_err()
        );
    }
    assert_eq!(
        node.cache_dependencies(&context, &multiply_inputs(&handle, 1.0))?
            .artifact_digests
            .len(),
        1
    );
    assert_eq!(generation.len(), 1);
    Ok(())
}
#[test]
fn task617_multiply_f32_values_signed_zero_fractional_scalars_and_original_input_immutability()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningMultiply")
        .ok_or("Multiply node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617912))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let source = conditioning(&context, "stable", &[(2, 2.0), (1, -3.0)], true)?;
    let digest = source.digest().to_owned();
    let handle = publish(&context, source.clone())?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for scalar in [-100.0, -1.0, -0.0, 0.0, 0.1, 1.0, 100.0, 2.0_f64.powi(-54)] {
        let result_handle = output(smol::block_on(
            node.execute(context.clone(), multiply_inputs(&handle, scalar)),
        )?)?;
        let result = resolved(&context, &result_handle)?;
        for (index, entry) in result.entries().iter().enumerate() {
            let value = if index == 0 { 2.0_f32 } else { -3.0_f32 };
            let expected = (value * scalar as f32).to_bits();
            let actual = tensor_values(&context, tensor(entry)?)?;
            assert!(actual.iter().all(|value| value.to_bits() == expected));
        }
        assert_eq!(source.digest(), digest);
        assert_eq!(resolved(&context, &handle)?.digest(), digest);
        context
            .handle_store()
            .revoke(&result_handle, &context.cancellation)?;
        drop(result);
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
#[test]
fn task617_multiply_preserves_metadata_options_references_identity_and_allocates_new_tensor()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningMultiply")
        .ok_or("Multiply node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617913))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let source = conditioning(&context, "stable", &[(1, 2.0), (2, 3.0)], false)?;
    let handle = publish(&context, source.clone())?;
    let result_handle = output(smol::block_on(
        node.execute(context.clone(), multiply_inputs(&handle, 1.0)),
    )?)?;
    let result = resolved(&context, &result_handle)?;
    assert_eq!(result.identity(), source.identity());
    for (actual, original) in result.entries().iter().zip(source.entries()) {
        assert_eq!(actual.identifier(), original.identifier());
        assert_eq!(actual.options().strength, original.options().strength);
        assert_eq!(actual.options().region, original.options().region);
        assert_eq!(actual.references(), original.references());
        assert_eq!(actual.metadata().len(), original.metadata().len());
        assert!(
            matches!(actual.metadata().get(ConditioningMetadataKey::TimestepRange),Some(ConditioningMetadataValue::TimestepRange(range)) if range.start_percent()==0.2 && range.end_percent()==0.8)
        );
        assert_ne!(tensor(actual)?.storage_id(), tensor(original)?.storage_id());
    }
    Ok(())
}
#[test]
fn task617_multiply_precancel_capacity_and_foreign_handle_fail_without_publication() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningMultiply")
        .ok_or("Multiply node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacities(1, usize::try_from(BUDGET)?)?;
    let cancellation = CancellationToken::default();
    let run_context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617914))),
        cancellation.clone(),
        BUDGET,
    )?;
    let handle = publish(
        &run_context,
        conditioning(&run_context, "stable", &[(1, 1.0)], true)?,
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(run_context.clone(), multiply_inputs(&handle, 2.0))).is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    let foreign = NativeHandleStoreGeneration::new()?;
    let other = context(
        backend.clone(),
        &authority,
        foreign.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617916))),
        CancellationToken::default(),
        BUDGET,
    )?;
    assert!(
        node.cache_dependencies(&other, &multiply_inputs(&handle, 2.0))
            .is_err()
    );
    assert!(smol::block_on(node.execute(other, multiply_inputs(&handle, 2.0))).is_err());
    assert!(foreign.is_empty());
    cancellation.cancel();
    let error = smol::block_on(node.execute(run_context.clone(), multiply_inputs(&handle, 2.0)))
        .err()
        .ok_or("Cancelled operation succeeded")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    run_context.handle_store().resolve(
        &handle,
        handle.handle_type(),
        &CancellationToken::default(),
    )?;
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
fn task617_multiply_actual_backend_saturation_and_late_unsupported_entry_drop_temporaries()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningMultiply")
        .ok_or("Multiply node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617915))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let input = conditioning(&context, "stable", &[(2, 1.0)], true)?;
    let handle = publish(&context, input.clone())?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let remaining = BUDGET
        .checked_sub(backend.memory_snapshot().current_bytes)
        .ok_or("Backend accounting exceeds budget")?;
    let lease = backend.reserve_workspace(&execution, remaining)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(smol::block_on(node.execute(context.clone(), multiply_inputs(&handle, 2.0))).is_err());
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    drop(lease);
    let mut entries = input.entries().to_vec();
    entries.push(ConditioningEntry::checked(
        "unsupported-later",
        ConditioningValue::constant(ConditioningConstant::Null)?,
        ConditioningEntryOptions::default(),
    )?);
    let unsupported = publish(
        &context,
        ConditioningSet::checked(input.identity().clone(), entries, &context.cancellation)?,
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let error = smol::block_on(node.execute(context.clone(), multiply_inputs(&unsupported, 2.0)))
        .err()
        .ok_or("Unsupported source representation succeeded")?;
    assert_eq!(error.code, "unsupported_conditioning_representation");
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

const WORKFLOW_BUDGET: u64 = 128 * 1024 * 1024;

fn workflow_executor(
    backend: Arc<CpuBackend>,
    diffusion: bool,
) -> Result<
    (
        sim_runtime::NativeImageExecutor,
        BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
    ),
    Box<dyn Error>,
> {
    let mut descriptors = BTreeMap::new();
    let mut capture = |registry: &mut sim_runtime::NativeNodeRegistry| {
        for class_type in ["ConditioningMultiply", "ConditioningAverage"]
            .into_iter()
            .chain(diffusion.then_some("CLIPTextEncode"))
            .chain(diffusion.then_some("CheckpointLoaderSimple"))
        {
            let descriptor = registry.descriptor(class_type).ok_or_else(|| {
                sim_runtime::NativeImageRuntimeError::Registry(format!(
                    "Missing genuine registered {class_type}"
                ))
            })?;
            if class_type != "CheckpointLoaderSimple" {
                descriptor.validate_exact_schema_v2().map_err(|error| {
                    sim_runtime::NativeImageRuntimeError::Registry(error.to_string())
                })?;
            }
            if registry.binding_disposition(class_type)
                != Some(sim_nodes::NativeNodeBindingDisposition::Executable)
            {
                return Err(sim_runtime::NativeImageRuntimeError::Registry(format!(
                    "Non-executable {class_type}"
                )));
            }
            descriptors.insert(class_type.to_owned(), descriptor.clone());
        }
        Ok(())
    };
    let executor = if diffusion {
        sim_runtime::NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
            sim_types::ProfileId(Uuid::from_u128(0x617610)),
            BTreeMap::new(),
            true,
            backend,
            Arc::new(sim_test_support::NativeDiffusionFixture::checked_in()),
            &mut capture,
        )?
    } else {
        sim_runtime::NativeImageExecutor::new_with_generated_registry_and_registration(
            sim_types::ProfileId(Uuid::from_u128(0x617610)),
            BTreeMap::new(),
            true,
            backend,
            &mut capture,
        )?
    };
    Ok((executor, descriptors))
}
fn workflow_literal(value: NativeValue) -> sim_runtime::InputBinding {
    sim_runtime::InputBinding::Literal { value }
}
fn workflow_number(value: f64) -> NativeValue {
    NativeValue::Primitive {
        value: NativePrimitive::Number(value),
    }
}
fn workflow_link(source: &str, output_index: usize) -> sim_runtime::InputBinding {
    sim_runtime::InputBinding::Link {
        source: NodeId(source.to_owned()),
        output_index,
        lazy: false,
        mode: sim_nodes::NativePortCardinality::Mapped,
    }
}
fn workflow_plan(
    descriptors: &BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
    definitions: Vec<(&str, &str, BTreeMap<String, sim_runtime::InputBinding>)>,
) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
    let mut nodes = BTreeMap::new();
    let mut order = Vec::new();
    let mut outputs = Vec::new();
    for (identifier, class_type, inputs) in definitions {
        let identifier = NodeId(identifier.to_owned());
        let mut descriptor = descriptors
            .get(class_type)
            .ok_or("Actual workflow descriptor missing")?
            .clone();
        if class_type != "CheckpointLoaderSimple" {
            descriptor.output_node = true;
            outputs.push(identifier.clone());
        }
        nodes.insert(
            identifier.clone(),
            sim_runtime::CompiledNode {
                id: identifier.clone(),
                class_type: class_type.to_owned(),
                descriptor,
                inputs,
                unknown: BTreeMap::new(),
            },
        );
        order.push(identifier);
    }
    outputs.sort();
    let plan = sim_runtime::CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x617611)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes,
        static_required_nodes: order.iter().cloned().collect(),
        topological_order: order,
        output_nodes: outputs,
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    Ok(plan)
}
fn workflow_column(columns: &[NativeValue]) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
    let [NativeValue::List { values }] = columns else {
        return Err("Genuine mapped single output column required".into());
    };
    values
        .iter()
        .map(|value| match value {
            NativeValue::Handle { value } => Ok(value.clone()),
            _ => Err("Mapped conditioning handle required".into()),
        })
        .collect()
}
fn workflow_pool(entry: &ConditioningEntry) -> Result<Option<&Tensor>, Box<dyn Error>> {
    match entry.metadata().get(ConditioningMetadataKey::PooledOutput) {
        Some(ConditioningMetadataValue::PooledOutput(value)) => Ok(value.as_ref()),
        None => Ok(None),
        _ => Err("Ordinary pooled canonical value required".into()),
    }
}
fn workflow_with_pool(
    input: &ConditioningSet,
    value: Option<Option<Tensor>>,
    cancellation: &CancellationToken,
) -> Result<ConditioningSet, Box<dyn Error>> {
    let mut metadata = input
        .entries()
        .first()
        .ok_or("Source entry required")?
        .metadata()
        .clone();
    if let Some(value) = value {
        metadata.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(value),
        )?;
    }
    Ok(input.with_entry_metadata(vec![metadata], cancellation)?)
}
fn workflow_transform_plan(
    descriptors: &BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
    targets: &[NativeOpaqueHandle],
    sources: &[NativeOpaqueHandle],
) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
    let handles = |handles: &[NativeOpaqueHandle]| NativeValue::List {
        values: handles
            .iter()
            .map(|handle| NativeValue::Handle {
                value: handle.clone(),
            })
            .collect(),
    };
    workflow_plan(
        descriptors,
        vec![
            (
                "multiply",
                "ConditioningMultiply",
                BTreeMap::from([
                    (
                        "conditioning".to_owned(),
                        workflow_literal(handles(targets)),
                    ),
                    (
                        "multiplier".to_owned(),
                        workflow_literal(workflow_number(2.0)),
                    ),
                ]),
            ),
            (
                "average",
                "ConditioningAverage",
                BTreeMap::from([
                    ("conditioning_to".to_owned(), workflow_link("multiply", 0)),
                    (
                        "conditioning_from".to_owned(),
                        workflow_literal(handles(sources)),
                    ),
                    (
                        "conditioning_to_strength".to_owned(),
                        workflow_literal(workflow_number(0.25)),
                    ),
                ]),
            ),
        ],
    )
}

#[test]
fn task617_registered_mapped_clip_producer_multiply_average_retains_explicit_none_and_cache()
-> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKFLOW_BUDGET)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = workflow_executor(backend.clone(), true)?;
    assert_eq!(
        descriptors
            .get("CLIPTextEncode")
            .ok_or("CLIP descriptor")?
            .implementation_version,
        "source-sd15-clip-text-pooled-v1"
    );
    let plan = workflow_plan(
        &descriptors,
        vec![
            (
                "checkpoint",
                "CheckpointLoaderSimple",
                BTreeMap::from([(
                    "ckpt_name".to_owned(),
                    workflow_literal(NativeValue::Primitive {
                        value: NativePrimitive::String("model.safetensors".to_owned()),
                    }),
                )]),
            ),
            (
                "encode",
                "CLIPTextEncode",
                BTreeMap::from([
                    ("clip".to_owned(), workflow_link("checkpoint", 1)),
                    (
                        "text".to_owned(),
                        workflow_literal(NativeValue::List {
                            values: ["a test", ""]
                                .into_iter()
                                .map(|text| NativeValue::Primitive {
                                    value: NativePrimitive::String(text.to_owned()),
                                })
                                .collect(),
                        }),
                    ),
                ]),
            ),
            (
                "multiply",
                "ConditioningMultiply",
                BTreeMap::from([
                    ("conditioning".to_owned(), workflow_link("encode", 0)),
                    (
                        "multiplier".to_owned(),
                        workflow_literal(NativeValue::List {
                            values: vec![workflow_number(2.0), workflow_number(-1.0)],
                        }),
                    ),
                ]),
            ),
            (
                "average",
                "ConditioningAverage",
                BTreeMap::from([
                    ("conditioning_to".to_owned(), workflow_link("multiply", 0)),
                    ("conditioning_from".to_owned(), workflow_link("encode", 0)),
                    (
                        "conditioning_to_strength".to_owned(),
                        workflow_literal(workflow_number(0.25)),
                    ),
                ]),
            ),
        ],
    )?;
    let restored: sim_runtime::CompiledPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        executor.execute_blocking(
            &restored,
            AttemptId(Uuid::from_u128(0x617614)),
            cancelled,
            0,
            authority.authorize_workspace(WORKFLOW_BUDGET / 2)?,
        ),
        Err(sim_runtime::NativeImageRuntimeError::Cancelled)
    ));
    assert!(executor.handle_store_generation_for_test().is_empty());
    assert_eq!(executor.cache_len_for_test(), 0);
    let result = executor.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x617612)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(WORKFLOW_BUDGET / 2)?,
    )?;
    assert_eq!(
        result.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    assert_eq!(result.report.cache_hits, 0);
    let generation = executor.handle_store_generation_for_test();
    let context = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617612))),
        CancellationToken::default(),
        WORKFLOW_BUDGET / 2,
    )?;
    let columns = |identifier: &str| {
        workflow_column(
            result
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Actual workflow output missing")?,
        )
    };
    let encoded = columns("encode")?;
    let multiplied = columns("multiply")?;
    let averaged = columns("average")?;
    assert_eq!((encoded.len(), multiplied.len(), averaged.len()), (2, 2, 2));
    for (index, scalar) in [2.0_f32, -1.0].into_iter().enumerate() {
        let original = resolved(&context, encoded.get(index).ok_or("Mapped encode")?)?;
        let scaled = resolved(&context, multiplied.get(index).ok_or("Mapped multiply")?)?;
        let blended = resolved(&context, averaged.get(index).ok_or("Mapped average")?)?;
        let original_entry = original.entries().first().ok_or("Encoded entry")?;
        let scaled_entry = scaled.entries().first().ok_or("Scaled entry")?;
        let blended_entry = blended.entries().first().ok_or("Blended entry")?;
        for entry in [original_entry, scaled_entry, blended_entry] {
            assert!(matches!(
                entry.metadata().get(ConditioningMetadataKey::PooledOutput),
                Some(ConditioningMetadataValue::PooledOutput(None))
            ));
        }
        assert_eq!(blended.identity(), original.identity());
        assert_eq!(blended_entry.identifier(), original_entry.identifier());
        assert_ne!(
            tensor(scaled_entry)?.storage_id(),
            tensor(original_entry)?.storage_id()
        );
        let originals = tensor_values(&context, tensor(original_entry)?)?;
        let scaled_values = tensor_values(&context, tensor(scaled_entry)?)?;
        let blended_values = tensor_values(&context, tensor(blended_entry)?)?;
        assert_eq!(originals.len(), scaled_values.len());
        assert_eq!(originals.len(), blended_values.len());
        for ((original, scaled), blended) in originals
            .iter()
            .zip(scaled_values.iter())
            .zip(blended_values.iter())
        {
            assert_eq!(scaled.to_bits(), (original * scalar).to_bits());
            assert_eq!(
                blended.to_bits(),
                ((scaled * 0.25) + (original * 0.75)).to_bits()
            );
        }
    }
    let cached = executor.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x617613)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(WORKFLOW_BUDGET / 2)?,
    )?;
    assert_eq!(
        cached.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        cached.report.error
    );
    assert!(cached.report.cache_hits > 0);
    for identifier in ["encode", "multiply", "average"] {
        let original = workflow_column(
            result
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Original mapped column")?,
        )?;
        let replayed = workflow_column(
            cached
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Cached mapped column")?,
        )?;
        assert_eq!(original.len(), replayed.len());
        for (original, replayed) in original.iter().zip(&replayed) {
            assert_eq!(original.digest_sha256(), replayed.digest_sha256());
        }
    }
    Ok(())
}

#[test]
fn task617_registered_mapped_pool_states_scale_publish_and_retain_source_aliases() -> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = workflow_executor(backend.clone(), false)?;
    let generation = executor.handle_store_generation_for_test();
    let attempt = AttemptId(Uuid::from_u128(0x617620));
    let context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        BUDGET,
    )?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let target_pool = tensor_from_f32(&backend, &[2], &[2.0, 4.0], &execution)?;
    let source_pool = tensor_from_f32(&backend, &[2], &[8.0, 12.0], &execution)?;
    let base = conditioning(&context, "mapped-pooled", &[(1, 2.0)], true)?;
    let source = workflow_with_pool(
        &conditioning(&context, "mapped-pooled", &[(1, 10.0)], true)?,
        Some(Some(source_pool.clone())),
        &context.cancellation,
    )?;
    let targets = [
        workflow_with_pool(&base, None, &context.cancellation)?,
        workflow_with_pool(&base, Some(None), &context.cancellation)?,
        workflow_with_pool(
            &base,
            Some(Some(target_pool.clone())),
            &context.cancellation,
        )?,
    ];
    let digests = targets
        .iter()
        .map(|value| value.digest().to_owned())
        .collect::<Vec<_>>();
    let target_handles = targets
        .iter()
        .cloned()
        .map(|value| publish(&context, value))
        .collect::<Result<Vec<_>, _>>()?;
    let source_handle = publish(&context, source)?;
    let plan = workflow_transform_plan(&descriptors, &target_handles, &[source_handle])?;
    let result = executor.execute_blocking(
        &plan,
        attempt,
        CancellationToken::default(),
        0,
        authority.authorize_workspace(BUDGET)?,
    )?;
    assert_eq!(
        result.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    let multiplied = workflow_column(
        result
            .report
            .outputs
            .get(&NodeId("multiply".to_owned()))
            .ok_or("Multiply column")?,
    )?;
    let averaged = workflow_column(
        result
            .report
            .outputs
            .get(&NodeId("average".to_owned()))
            .ok_or("Average column")?,
    )?;
    assert_eq!((multiplied.len(), averaged.len()), (3, 3));
    for index in 0..3 {
        let scaled = resolved(&context, multiplied.get(index).ok_or("Scaled handle")?)?;
        let blended = resolved(&context, averaged.get(index).ok_or("Average handle")?)?;
        let scaled_entry = scaled.entries().first().ok_or("Scaled entry")?;
        let blended_entry = blended.entries().first().ok_or("Blended entry")?;
        assert!(
            tensor_values(&context, tensor(blended_entry)?)?
                .iter()
                .all(|value| *value == 8.5)
        );
        assert_eq!(
            blended_entry.identifier(),
            targets
                .get(index)
                .ok_or("Original input")?
                .entries()
                .first()
                .ok_or("Original entry")?
                .identifier()
        );
        let pooled = workflow_pool(blended_entry)?.ok_or("Source Some must produce Some")?;
        if index == 2 {
            assert_eq!(&*tensor_values(&context, pooled)?, &[7.0, 11.0]);
            assert_ne!(
                workflow_pool(scaled_entry)?
                    .ok_or("Scaled pool")?
                    .storage_id(),
                target_pool.storage_id()
            );
        } else {
            assert_eq!(&*tensor_values(&context, pooled)?, &[8.0, 12.0]);
            if index == 1 {
                assert!(matches!(
                    scaled_entry
                        .metadata()
                        .get(ConditioningMetadataKey::PooledOutput),
                    Some(ConditioningMetadataValue::PooledOutput(None))
                ));
                assert_eq!(pooled.storage_id(), source_pool.storage_id());
            } else {
                assert!(
                    scaled_entry
                        .metadata()
                        .get(ConditioningMetadataKey::PooledOutput)
                        .is_none()
                );
                assert_ne!(pooled.storage_id(), source_pool.storage_id());
            }
        }
        assert_eq!(
            targets.get(index).ok_or("Input")?.digest(),
            digests.get(index).ok_or("Digest")?
        );
    }
    assert_eq!(&*tensor_values(&context, &target_pool)?, &[2.0, 4.0]);
    assert_eq!(&*tensor_values(&context, &source_pool)?, &[8.0, 12.0]);
    Ok(())
}

#[test]
fn task617_registered_mapped_late_average_failure_rolls_back_then_recovers() -> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = workflow_executor(backend.clone(), false)?;
    let generation = executor.handle_store_generation_for_test();
    let attempt = AttemptId(Uuid::from_u128(0x617630));
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        BUDGET,
    )?;
    let compute = setup.compute_session()?;
    let execution = compute.execution_context(&setup)?;
    let base = conditioning(&setup, "mapped-rollback", &[(1, 2.0)], true)?;
    let pool = tensor_from_f32(&backend, &[2], &[8.0, 12.0], &execution)?;
    let good = workflow_with_pool(&base, Some(Some(pool)), &setup.cancellation)?;
    let integer = backend
        .upload_bytes(
            sim_tensor::TensorDescriptor::contiguous(
                vec![2],
                sim_tensor::DType::I32,
                sim_tensor::DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &[0; 8],
            &execution,
        )?
        .0;
    let bad = workflow_with_pool(&base, Some(Some(integer)), &setup.cancellation)?;
    let original_digest = base.digest().to_owned();
    let bad_digest = bad.digest().to_owned();
    let handles = [
        publish(&setup, base.clone())?,
        publish(&setup, good.clone())?,
        publish(&setup, bad.clone())?,
    ];
    let plan = workflow_transform_plan(
        &descriptors,
        &[handles.first().ok_or("Target")?.clone()],
        handles.get(1..).ok_or("Source handles")?,
    )?;
    let before = backend.memory_snapshot().current_bytes;
    let failed = executor.execute_blocking(
        &plan,
        attempt,
        CancellationToken::default(),
        0,
        authority.authorize_workspace(BUDGET)?,
    )?;
    assert_eq!(failed.report.state, sim_runtime::AttemptState::Failed);
    assert!(
        failed
            .report
            .error
            .as_deref()
            .is_some_and(|error| error.contains("zed.cpu.binary-scalar.multiply"))
    );
    assert!(failed.report.outputs.is_empty());
    assert!(generation.is_empty());
    assert_eq!(executor.cache_len_for_test(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    assert_eq!(base.digest(), original_digest);
    assert_eq!(bad.digest(), bad_digest);
    let recovered_attempt = AttemptId(Uuid::from_u128(0x617631));
    let recovered_context = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(recovered_attempt),
        CancellationToken::default(),
        BUDGET,
    )?;
    let target = publish(&recovered_context, base)?;
    let source = publish(&recovered_context, good)?;
    let repaired = workflow_transform_plan(&descriptors, &[target], &[source])?;
    let recovered = executor.execute_blocking(
        &repaired,
        recovered_attempt,
        CancellationToken::default(),
        0,
        authority.authorize_workspace(BUDGET)?,
    )?;
    assert_eq!(
        recovered.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        recovered.report.error
    );
    assert_eq!(
        workflow_column(
            recovered
                .report
                .outputs
                .get(&NodeId("average".to_owned()))
                .ok_or("Recovered average")?
        )?
        .len(),
        1
    );
    Ok(())
}

#[test]
fn task617_registered_average_cancel_capacity_and_backend_exhaustion_preserve_inputs_then_recover()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningAverage").ok_or("Average node")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacities(2, usize::try_from(BUDGET)?)?;
    let cancellation = CancellationToken::default();
    let run_context = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617640))),
        cancellation.clone(),
        BUDGET,
    )?;
    let target = conditioning(&run_context, "average-authority", &[(1, 2.0)], true)?;
    let source = conditioning(&run_context, "average-authority", &[(1, 10.0)], true)?;
    let target_handle = publish(&run_context, target.clone())?;
    let source_handle = publish(&run_context, source.clone())?;
    let make_inputs = |target: &NativeOpaqueHandle, source: &NativeOpaqueHandle| {
        let mut inputs = inputs("ConditioningAverage", target, source);
        inputs.insert("conditioning_to_strength".to_owned(), workflow_number(0.25));
        inputs
    };
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let capacity_error = smol::block_on(node.execute(
        run_context.clone(),
        make_inputs(&target_handle, &source_handle),
    ))
    .err()
    .ok_or("Full canonical handle store accepted Average publication")?;
    assert!(
        capacity_error
            .message
            .contains("native handle store capacity is exhausted")
    );
    assert_eq!(capacity_error.kind, NativeNodeFailureKind::Failure);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    cancellation.cancel();
    let error = smol::block_on(node.execute(
        run_context.clone(),
        make_inputs(&target_handle, &source_handle),
    ))
    .err()
    .ok_or("Cancelled Average succeeded")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    let inspection_cancellation = CancellationToken::default();
    for handle in [&target_handle, &source_handle] {
        run_context.handle_store().resolve(
            handle,
            handle.handle_type(),
            &inspection_cancellation,
        )?;
    }
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    let recovered_generation = NativeHandleStoreGeneration::new()?;
    let recovered = context(
        backend.clone(),
        &authority,
        recovered_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617641))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let target_handle = publish(&recovered, target)?;
    let source_handle = publish(&recovered, source)?;
    let compute = recovered.compute_session()?;
    let execution = compute.execution_context(&recovered)?;
    let remaining = BUDGET
        .checked_sub(backend.memory_snapshot().current_bytes)
        .ok_or("Backend memory exceeds limit")?;
    let reservation = backend.reserve_workspace(&execution, remaining)?;
    let saturated = (
        recovered_generation.len(),
        recovered_generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(
            recovered.clone(),
            make_inputs(&target_handle, &source_handle)
        ))
        .is_err()
    );
    assert_eq!(
        (
            recovered_generation.len(),
            recovered_generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        saturated
    );
    drop(reservation);
    let result = output(smol::block_on(node.execute(
        recovered.clone(),
        make_inputs(&target_handle, &source_handle),
    ))?)?;
    let result = resolved(&recovered, &result)?;
    assert!(
        tensor_values(
            &recovered,
            tensor(result.entries().first().ok_or("Recovered entry")?)?
        )?
        .iter()
        .all(|value| *value == 8.0)
    );
    Ok(())
}

#[derive(serde::Deserialize)]
struct SourceTensorRecord {
    shape: Vec<u64>,
    values: Vec<f32>,
}
#[derive(serde::Deserialize)]
struct SourceConditioningEntry {
    primary: SourceTensorRecord,
    pool_state: String,
    pooled: Option<SourceTensorRecord>,
}
#[derive(serde::Deserialize)]
struct SourceConditioningCase {
    class_type: String,
    scalar: Option<f64>,
    first: Vec<SourceConditioningEntry>,
    second: Vec<SourceConditioningEntry>,
    output: Vec<SourceConditioningEntry>,
}
#[derive(serde::Deserialize)]
struct SourceConditioningObservations {
    cases: Vec<SourceConditioningCase>,
    case_count: usize,
}
fn source_conditioning_set(
    context: &NativeNodeContext,
    entries: &[SourceConditioningEntry],
) -> Result<ConditioningSet, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let mut output = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let primary = tensor_from_f32(
            compute.backend(),
            &entry.primary.shape,
            &entry.primary.values,
            &execution,
        )?;
        let mut metadata = ConditioningMetadataMap::default();
        match entry.pool_state.as_str() {
            "absent" => assert!(entry.pooled.is_none()),
            "none" => {
                assert!(entry.pooled.is_none());
                metadata.insert(
                    ConditioningMetadataKey::PooledOutput,
                    ConditioningMetadataValue::PooledOutput(None),
                )?;
            }
            "some" => {
                let pooled = entry
                    .pooled
                    .as_ref()
                    .ok_or("Source pooled tensor missing")?;
                let pooled =
                    tensor_from_f32(compute.backend(), &pooled.shape, &pooled.values, &execution)?;
                metadata.insert(
                    ConditioningMetadataKey::PooledOutput,
                    ConditioningMetadataValue::PooledOutput(Some(pooled)),
                )?;
            }
            _ => return Err("Unknown Source pool state".into()),
        }
        output.push(ConditioningEntry::checked_with_metadata(
            format!("source.{index}"),
            ConditioningValue::cross_attention(primary)?,
            Default::default(),
            Default::default(),
            metadata,
        )?);
    }
    Ok(ConditioningSet::checked(
        identity("source-numerical")?,
        output,
        &context.cancellation,
    )?)
}
fn compare_source_tensor_exact(
    context: &NativeNodeContext,
    actual: &Tensor,
    expected: &SourceTensorRecord,
) -> TestResult {
    assert_eq!(actual.descriptor().shape(), expected.shape);
    let actual = tensor_values(context, actual)?;
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}
#[test]
fn task617_original_torch_conditioning_numerical_outputs_match_canonical_nodes() -> TestResult {
    use sha2::{Digest, Sha256};
    let bytes = include_bytes!("../fixtures/nodes/conditioning-part-two-original.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "edc65d0fa4cc99c928d7a5f7a983fc1e64be3a3b18dd37261c63794b626665d0"
    );
    let source: SourceConditioningObservations = serde_json::from_slice(bytes)?;
    assert_eq!(source.case_count, 41);
    assert_eq!(source.cases.len(), 41);
    let registry = generated_native_node_registry_projection(None)?;
    for (index, case) in source.cases.iter().enumerate() {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
        let backend = Arc::new(backend);
        let generation = NativeHandleStoreGeneration::new()?;
        let context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(
                0x617900 + u128::try_from(index)?,
            ))),
            CancellationToken::default(),
            BUDGET,
        )?;
        let first = source_conditioning_set(&context, &case.first)?;
        let first_digest = first.digest().to_owned();
        let first_handle = publish(&context, first)?;
        let mut arguments = if case.class_type == "ConditioningMultiply" {
            multiply_inputs(
                &first_handle,
                case.scalar.ok_or("Source multiplier missing")?,
            )
        } else {
            let second = source_conditioning_set(&context, &case.second)?;
            let second_handle = publish(&context, second)?;
            inputs(&case.class_type, &first_handle, &second_handle)
        };
        if case.class_type == "ConditioningAverage" {
            arguments.insert(
                "conditioning_to_strength".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(case.scalar.ok_or("Source strength missing")?),
                },
            );
        }
        let node = registry
            .node(&case.class_type)
            .ok_or("Source node missing")?;
        let handle = output(smol::block_on(node.execute(context.clone(), arguments))?)?;
        let actual = resolved(&context, &handle)?;
        assert_eq!(actual.entries().len(), case.output.len());
        for (actual, expected) in actual.entries().iter().zip(&case.output) {
            compare_source_tensor_exact(&context, tensor(actual)?, &expected.primary)?;
            match (
                actual.metadata().get(ConditioningMetadataKey::PooledOutput),
                expected.pool_state.as_str(),
            ) {
                (None, "absent")
                | (Some(ConditioningMetadataValue::PooledOutput(None)), "none") => {
                    assert!(expected.pooled.is_none())
                }
                (Some(ConditioningMetadataValue::PooledOutput(Some(actual))), "some") => {
                    compare_source_tensor_exact(
                        &context,
                        actual,
                        expected.pooled.as_ref().ok_or("Source pool missing")?,
                    )?
                }
                _ => return Err(format!("Source pool state mismatch in case {index}").into()),
            }
        }
        assert_eq!(resolved(&context, &first_handle)?.digest(), first_digest);
    }
    Ok(())
}
#[derive(serde::Deserialize)]
struct SourceVisionCase {
    batch: usize,
    crop: String,
    input: SourceTensorRecord,
    pixels: SourceTensorRecord,
    last_hidden_state: SourceTensorRecord,
    penultimate_hidden_states: SourceTensorRecord,
    image_embeds: SourceTensorRecord,
    image_sizes: Vec<[u64; 3]>,
}
#[derive(serde::Deserialize)]
struct SourceVisionStateTensor {
    shape: Vec<u64>,
    f32_values: Vec<f32>,
}
#[derive(serde::Deserialize)]
struct SourceVisionObservations {
    configuration: serde_json::Value,
    state: BTreeMap<String, SourceVisionStateTensor>,
    state_count: usize,
    state_loaded_exactly: bool,
    cases: Vec<SourceVisionCase>,
}
fn compare_source_vision_tensor(
    context: &NativeNodeContext,
    actual: &Tensor,
    expected: &SourceTensorRecord,
    field: &str,
) -> TestResult {
    assert_eq!(actual.descriptor().shape(), expected.shape);
    let actual = tensor_values(context, actual)?;
    assert_eq!(actual.len(), expected.values.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected.values).enumerate() {
        assert!(actual.is_finite() && expected.is_finite());
        assert!(
            (actual - expected).abs() <= 0.00002 + 0.00002 * expected.abs(),
            "Source vision {field} element {index}: actual {actual}, expected {expected}"
        );
    }
    Ok(())
}
#[test]
fn task617_original_torch_clip_vision_numerical_outputs_match_canonical_node() -> TestResult {
    use sha2::{Digest, Sha256};
    use sim_model::clip_vision::{NativeClipVision, NativeClipVisionRecipeConfiguration};
    let bytes = include_bytes!("../fixtures/nodes/clip-vision-encode-original.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "ebf440f0e3b5a50d414c30735251bf656f6b4b89082c43102d7a9ba40ecef6b5"
    );
    let source: SourceVisionObservations = serde_json::from_slice(bytes)?;
    assert_eq!(source.state_count, 39);
    assert_eq!(source.state.len(), 39);
    assert!(source.state_loaded_exactly);
    assert_eq!(source.cases.len(), 4);
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("CLIPVisionEncode")
        .ok_or("Missing vision node")?;
    for case in &source.cases {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
        let backend = Arc::new(backend);
        let generation = NativeHandleStoreGeneration::new()?;
        let context = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x617975))),
            CancellationToken::default(),
            BUDGET,
        )?;
        let compute = context.compute_session()?;
        let execution = compute.execution_context(&context)?;
        let state = source
            .state
            .iter()
            .map(|(name, record)| {
                Ok::<_, Box<dyn Error>>((
                    name.clone(),
                    tensor_from_f32(
                        compute.backend(),
                        &record.shape,
                        &record.f32_values,
                        &execution,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let configuration = serde_json::to_vec(&source.configuration)?;
        let resource = NativeClipVision::from_recipe_state(
            sim_model::model_family::NativeModelRecipeSourceForm::Standalone,
            vec![NativeClipVisionRecipeConfiguration {
                artifact_sha256: format!("{:x}", Sha256::digest(&configuration)),
                bytes: configuration,
                parser_limits: sim_model::ParserLimits::default(),
            }],
            state,
            &context.cancellation,
        )?;
        let image = tensor_from_f32(
            compute.backend(),
            &case.input.shape,
            &case.input.values,
            &execution,
        )?;
        let pixels =
            resource.preprocess(compute.backend(), &image, case.crop == "center", &execution)?;
        compare_source_vision_tensor(&context, &pixels, &case.pixels, "pixels")?;
        let model = sim_model::NativeModelPayload::clip_vision(Arc::new(resource))?;
        let model = sim_nodes::NativeStoredModelPayload::model_resource(Arc::new(model))?;
        let model = context.handle_store().publish(
            NativeStoredPayload::Model(Arc::new(model)),
            &context.cancellation,
        )?;
        let image = sim_tensor::NativeTensorPayload::from_image(
            sim_tensor::NativeTensorRole::Image,
            sim_tensor::ImageTensor::from_tensor(image)?,
        )?;
        let image = context.handle_store().publish(
            NativeStoredPayload::Tensor(Arc::new(image)),
            &context.cancellation,
        )?;
        let handle = output(smol::block_on(
            node.execute(context.clone(), vision_inputs(&model, &image, &case.crop)),
        )?)?;
        let actual = vision_output(&context, &handle)?;
        assert_eq!(
            case.input.shape.first().copied(),
            Some(u64::try_from(case.batch)?)
        );
        assert_eq!(actual.image_sizes(), case.image_sizes);
        assert!(actual.projected_intermediate.is_none());
        compare_source_vision_tensor(
            &context,
            &actual.last_hidden_state,
            &case.last_hidden_state,
            "last_hidden_state",
        )?;
        compare_source_vision_tensor(
            &context,
            actual
                .intermediate
                .as_ref()
                .ok_or("Penultimate hidden missing")?,
            &case.penultimate_hidden_states,
            "penultimate_hidden_states",
        )?;
        compare_source_vision_tensor(
            &context,
            &actual.image_embeds,
            &case.image_embeds,
            "image_embeds",
        )?;
    }
    Ok(())
}
