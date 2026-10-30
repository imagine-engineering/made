use sim_model::{
    conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningHookReference,
        ConditioningIdentity, ConditioningMetadataKey, ConditioningMetadataMap,
        ConditioningMetadataValue, ConditioningReferences, ConditioningRegion, ConditioningSet,
        ConditioningTimestepRange, ConditioningValue,
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
const WORKFLOW_BUDGET: u64 = 128 * 1024 * 1024;
fn node_context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
    budget: u64,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let node = NodeId("conditioning-part-three".to_owned());
    let scratch = authority.authorize_workspace(budget)?;
    let identity = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x61803),
        store.attempt_id(),
        node.clone(),
    )?;
    let compute =
        NativeNodeComputeSession::checked(identity, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x61801)),
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

fn primitive(value: NativePrimitive) -> NativeValue {
    NativeValue::Primitive { value }
}
fn number(value: f64) -> NativeValue {
    primitive(NativePrimitive::Number(value))
}
fn unsigned(value: u64) -> NativeValue {
    primitive(NativePrimitive::UnsignedInteger(value))
}
fn inputs(class_type: &str, handle: &NativeOpaqueHandle) -> BTreeMap<String, NativeValue> {
    let mut inputs = BTreeMap::from([(
        "conditioning".to_owned(),
        NativeValue::Handle {
            value: handle.clone(),
        },
    )]);
    match class_type {
        "ConditioningSetArea" => {
            for (name, value) in [("width", 65), ("height", 79), ("x", 17), ("y", 9)] {
                inputs.insert(name.to_owned(), unsigned(value));
            }
            inputs.insert("strength".to_owned(), number(0.5));
        }
        "ConditioningSetAreaPercentage" | "ConditioningSetAreaPercentageVideo" => {
            for (name, value) in [
                ("width", 0.5),
                ("height", 0.75),
                ("x", 0.25),
                ("y", 0.0),
                ("strength", 0.5),
            ] {
                inputs.insert(name.to_owned(), number(value));
            }
            if class_type.ends_with("Video") {
                inputs.insert("temporal".to_owned(), number(0.25));
                inputs.insert("z".to_owned(), number(0.5));
            }
        }
        "ConditioningSetAreaStrength" => {
            inputs.insert("strength".to_owned(), number(2.0));
        }
        "ConditioningSetTimestepRange" => {
            inputs.insert("start".to_owned(), number(0.8));
            inputs.insert("end".to_owned(), number(0.2));
        }
        "ConditioningStableAudio" => {
            inputs.clear();
            for name in ["positive", "negative"] {
                inputs.insert(
                    name.to_owned(),
                    NativeValue::Handle {
                        value: handle.clone(),
                    },
                );
            }
            inputs.insert("seconds_start".to_owned(), number(0.1));
            inputs.insert("seconds_total".to_owned(), number(47.125));
        }
        _ => {}
    }
    inputs
}
fn output_handles(outcome: NativeNodeOutcome) -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Value outputs required".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    outputs
        .into_iter()
        .map(|value| match value {
            NativeValue::Handle { value } => Ok(value),
            _ => Err("Handle output required".into()),
        })
        .collect()
}
fn first_output(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let mut handles = output_handles(outcome)?;
    if handles.len() != 1 {
        return Err("Exactly one output required".into());
    }
    handles.pop().ok_or_else(|| "Missing output".into())
}
fn setup(
    capacity: usize,
) -> Result<
    (
        Arc<CpuBackend>,
        CpuWorkspaceAuthority,
        NativeHandleStoreGeneration,
        NativeNodeContext,
        NativeOpaqueHandle,
        ConditioningSet,
    ),
    Box<dyn Error>,
> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let backend = Arc::new(backend);
    let generation =
        NativeHandleStoreGeneration::with_capacities(capacity, usize::try_from(BUDGET)?)?;
    let context = node_context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61831))),
        CancellationToken::default(),
        BUDGET,
    )?;
    let source = conditioning(&context, "part-three", &[(1, 1.0), (2, 2.0)], true)?;
    let handle = publish(&context, source.clone())?;
    Ok((backend, authority, generation, context, handle, source))
}
fn assert_aliases(result: &ConditioningSet, source: &ConditioningSet) -> TestResult {
    assert_eq!(result.identity(), source.identity());
    assert_eq!(result.entries().len(), source.entries().len());
    for (actual, original) in result.entries().iter().zip(source.entries()) {
        assert_eq!(actual.identifier(), original.identifier());
        assert_eq!(tensor(actual)?.storage_id(), tensor(original)?.storage_id());
        assert_eq!(actual.references(), original.references());
    }
    Ok(())
}
#[test]
fn task618_exact_seven_source_bindings_mapped_cache_and_aliases() -> TestResult {
    let bindings = (sim_nodes::generated_model_conditioning_03::NATIVE_NODE_BINDINGS)()?;
    assert_eq!(bindings.len(), 7);
    let registry = generated_native_node_registry_projection(None)?;
    let (_, _, _, context, handle, _) = setup(30)?;
    for (binding, feature) in bindings.iter().zip([
        "COMFY-NODE-0092",
        "COMFY-NODE-0093",
        "COMFY-NODE-0094",
        "COMFY-NODE-0095",
        "COMFY-NODE-0100",
        "COMFY-NODE-0101",
        "COMFY-NODE-0103",
    ]) {
        assert_eq!(binding.feature_id(), feature);
        let descriptor = binding.descriptor();
        descriptor.validate_exact_schema_v2()?;
        assert_eq!(
            descriptor.cache,
            sim_nodes::NativeCachePolicy::InputIdentity
        );
        assert_eq!(descriptor.effect, sim_nodes::NativeEffectClass::Pure);
        assert!(
            descriptor
                .inputs
                .iter()
                .all(|input| input.cardinality == sim_nodes::NativePortCardinality::Mapped)
        );
        assert_eq!(
            sim_nodes::generated_family_node_bindings()?
                .iter()
                .filter(|candidate| candidate.feature_id() == feature)
                .count(),
            1
        );
        let node = registry
            .node(&descriptor.class_type)
            .ok_or("Executable node missing")?;
        let mut mapped = inputs(&descriptor.class_type, &handle);
        for value in mapped.values_mut() {
            *value = NativeValue::List {
                values: vec![value.clone(), value.clone()],
            };
        }
        assert!(node.demanded_lazy_inputs(&context, &mapped)?.is_empty());
        let dependencies = node.cache_dependencies(&context, &mapped)?;
        assert_eq!(
            dependencies.artifact_digests.len(),
            if descriptor.class_type == "ConditioningStableAudio" {
                4
            } else {
                2
            }
        );
        assert_eq!(
            node.cache_change_token(&mapped)?,
            node.implementation_version()
        );
    }
    assert_eq!(
        bindings
            .first()
            .ok_or("Area binding")?
            .presentation()
            .search_aliases,
        [
            "regional prompt",
            "area prompt",
            "spatial conditioning",
            "localized prompt"
        ]
    );
    Ok(())
}
#[test]
fn task618_absolute_area_floor_order_updates_mask_bounds_and_retains_aliases() -> TestResult {
    let (backend, _, _, context, _, source) = setup(20)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let mask = tensor_from_f32(&backend, &[1, 2, 2], &[0.0, 1.0, 0.5, 1.0], &execution)?;
    let entry = source.entries().first().ok_or("Source entry")?;
    let mut options = entry.options().clone();
    options.mask = Some(sim_model::conditioning::ConditioningMask::new(
        mask.clone(),
        0.75,
        vec![0, 0],
        true,
    )?);
    let mut metadata = entry.metadata().clone();
    metadata.insert(
        ConditioningMetadataKey::Region,
        ConditioningMetadataValue::Region(options.region.clone().ok_or("Region")?),
    )?;
    metadata.insert(
        ConditioningMetadataKey::SetAreaToBounds,
        ConditioningMetadataValue::Boolean(true),
    )?;
    metadata.insert(
        ConditioningMetadataKey::Mask,
        ConditioningMetadataValue::Tensor(mask.clone()),
    )?;
    metadata.insert(
        ConditioningMetadataKey::MaskStrength,
        ConditioningMetadataValue::FiniteF64(0.75),
    )?;
    let source = ConditioningSet::checked(
        source.identity().clone(),
        vec![ConditioningEntry::checked_with_metadata(
            entry.identifier(),
            entry.value().clone(),
            options,
            entry.references().clone(),
            metadata,
        )?],
        &context.cancellation,
    )?;
    let digest = source.digest().to_owned();
    let handle = publish(&context, source.clone())?;
    let before = backend.memory_snapshot().current_bytes;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningSetArea").ok_or("Area")?;
    let output = first_output(smol::block_on(
        node.execute(context.clone(), inputs("ConditioningSetArea", &handle)),
    )?)?;
    let result = resolved(&context, &output)?;
    assert_aliases(&result, &source)?;
    let actual = result.entries().first().ok_or("Entry")?;
    let region = ConditioningRegion::absolute(vec![9, 8], vec![1, 2])?;
    assert_eq!(actual.options().region.as_ref(), Some(&region));
    assert_eq!(actual.options().strength, 0.5);
    assert!(
        matches!(actual.metadata().get(ConditioningMetadataKey::Region),Some(ConditioningMetadataValue::Region(value)) if value==&region)
    );
    assert!(matches!(
        actual
            .metadata()
            .get(ConditioningMetadataKey::SetAreaToBounds),
        Some(ConditioningMetadataValue::Boolean(false))
    ));
    let actual_mask = actual.options().mask.as_ref().ok_or("Mask")?;
    assert!(!actual_mask.set_region_to_nonzero_bounds());
    assert_eq!(actual_mask.tensor().storage_id(), mask.storage_id());
    assert_eq!(actual_mask.strength(), 0.75);
    let mut upper = inputs("ConditioningSetArea", &handle);
    upper.insert("width".to_owned(), unsigned(16_384));
    upper.insert("height".to_owned(), unsigned(16_384));
    upper.insert("x".to_owned(), unsigned(16_384));
    upper.insert("y".to_owned(), unsigned(16_384));
    let upper = first_output(smol::block_on(node.execute(context.clone(), upper))?)?;
    let upper = resolved(&context, &upper)?;
    assert_eq!(
        upper
            .entries()
            .first()
            .ok_or("Upper bound entry")?
            .options()
            .region
            .as_ref(),
        Some(&ConditioningRegion::absolute(
            vec![2048, 2048],
            vec![2048, 2048]
        )?)
    );
    assert_aliases(&upper, &source)?;
    assert_eq!(source.digest(), digest);
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    Ok(())
}
#[test]
fn task618_percentage_and_video_axes_zero_sizes_and_original_metadata() -> TestResult {
    let (backend, _, _, context, handle, source) = setup(20)?;
    let before = backend.memory_snapshot().current_bytes;
    let registry = generated_native_node_registry_projection(None)?;
    for class_type in [
        "ConditioningSetAreaPercentage",
        "ConditioningSetAreaPercentageVideo",
    ] {
        let node = registry.node(class_type).ok_or("Percentage node")?;
        let mut arguments = inputs(class_type, &handle);
        arguments.insert("width".to_owned(), number(0.0));
        let output = first_output(smol::block_on(node.execute(context.clone(), arguments))?)?;
        let result = resolved(&context, &output)?;
        assert_aliases(&result, &source)?;
        let expected = if class_type.ends_with("Video") {
            ConditioningRegion::percentage(vec![0.25, 0.75, 0.0], vec![0.5, 0.0, 0.25])?
        } else {
            ConditioningRegion::percentage(vec![0.75, 0.0], vec![0.0, 0.25])?
        };
        for actual in result.entries() {
            assert_eq!(actual.options().region.as_ref(), Some(&expected));
            assert_eq!(actual.options().strength, 0.5);
            assert!(
                actual
                    .metadata()
                    .get(ConditioningMetadataKey::TimestepRange)
                    .is_some()
            );
        }
        assert!(
            expected
                .resolve(if class_type.ends_with("Video") {
                    &[4, 8, 8]
                } else {
                    &[8, 8]
                })
                .is_err()
        );
    }
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    Ok(())
}
#[test]
fn task618_strength_only_preserves_region_metadata_and_negative_zero() -> TestResult {
    let (_, _, _, context, handle, source) = setup(20)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningSetAreaStrength")
        .ok_or("Strength node")?;
    for strength in [0.0, -0.0, 0.5, 1.0, 2.0, 10.0] {
        let mut arguments = inputs("ConditioningSetAreaStrength", &handle);
        arguments.insert("strength".to_owned(), number(strength));
        let output = first_output(smol::block_on(node.execute(context.clone(), arguments))?)?;
        let result = resolved(&context, &output)?;
        assert_aliases(&result, &source)?;
        for (actual, original) in result.entries().iter().zip(source.entries()) {
            assert_eq!(actual.options().strength.to_bits(), strength.to_bits());
            assert_eq!(actual.options().region, original.options().region);
            assert_eq!(actual.metadata().len(), original.metadata().len());
        }
    }
    Ok(())
}
#[test]
fn task618_timestep_reversed_equal_and_endpoint_source_metadata_retained() -> TestResult {
    let (_, _, _, context, handle, source) = setup(20)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningSetTimestepRange")
        .ok_or("Range node")?;
    for (start, end) in [
        (0.0, 1.0),
        (0.8, 0.2),
        (1.0, 0.0),
        (0.25, 0.25),
        (-0.0, 1.0),
    ] {
        let mut arguments = inputs("ConditioningSetTimestepRange", &handle);
        arguments.insert("start".to_owned(), number(start));
        arguments.insert("end".to_owned(), number(end));
        let output = first_output(smol::block_on(node.execute(context.clone(), arguments))?)?;
        let result = resolved(&context, &output)?;
        assert_aliases(&result, &source)?;
        for (actual, original) in result.entries().iter().zip(source.entries()) {
            let Some(ConditioningMetadataValue::TimestepRange(range)) = actual
                .metadata()
                .get(ConditioningMetadataKey::TimestepRange)
            else {
                return Err("Range required".into());
            };
            assert_eq!(range.start_percent().to_bits(), start.to_bits());
            assert_eq!(range.end_percent().to_bits(), end.to_bits());
            assert_eq!(actual.options().strength, original.options().strength);
            assert_eq!(actual.options().region, original.options().region);
        }
    }
    Ok(())
}
#[test]
fn task618_stable_audio_two_columns_replace_timing_together_and_keep_identities() -> TestResult {
    let (backend, _, _, context, _, source) = setup(20)?;
    let negative_source = conditioning(&context, "negative", &[(3, 3.0)], false)?;
    let negative = publish(&context, negative_source.clone())?;
    let mut values = Vec::new();
    for entry in source.entries() {
        let metadata = entry.metadata();
        let metadata = sim_model::conditioning::ConditioningMetadataMap::checked(
            metadata
                .iter()
                .map(|(key, value)| (*key, value.clone()))
                .chain([
                    (
                        ConditioningMetadataKey::SecondsStart,
                        ConditioningMetadataValue::FiniteF64(1.0),
                    ),
                    (
                        ConditioningMetadataKey::SecondsTotal,
                        ConditioningMetadataValue::FiniteF64(2.0),
                    ),
                ]),
        )?;
        values.push(metadata);
    }
    let source = source.with_entry_metadata(values, &context.cancellation)?;
    let positive = publish(&context, source.clone())?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningStableAudio")
        .ok_or("Audio node")?;
    let mut arguments = inputs("ConditioningStableAudio", &positive);
    arguments.insert(
        "negative".to_owned(),
        NativeValue::Handle { value: negative },
    );
    arguments.insert("seconds_start".to_owned(), number(3.0));
    arguments.insert("seconds_total".to_owned(), number(4.0));
    let before = backend.memory_snapshot().current_bytes;
    let outputs = output_handles(smol::block_on(node.execute(context.clone(), arguments))?)?;
    assert_eq!(outputs.len(), 2);
    for (handle, original) in outputs.iter().zip([&source, &negative_source]) {
        let result = resolved(&context, handle)?;
        assert_aliases(&result, original)?;
        for entry in result.entries() {
            assert!(
                matches!(entry.metadata().get(ConditioningMetadataKey::SecondsStart),Some(ConditioningMetadataValue::FiniteF64(value)) if *value==3.0)
            );
            assert!(
                matches!(entry.metadata().get(ConditioningMetadataKey::SecondsTotal),Some(ConditioningMetadataValue::FiniteF64(value)) if *value==4.0)
            );
        }
    }
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    Ok(())
}
#[test]
fn task618_binary64_strength_preserves_raw_bits_aliases_and_cache_version() -> TestResult {
    let (backend, _, generation, context, handle, source) = setup(64)?;
    let registry = generated_native_node_registry_projection(None)?;
    let digest = source.digest().to_owned();
    let before = backend.memory_snapshot().current_bytes;
    for class_type in [
        "ConditioningSetArea",
        "ConditioningSetAreaPercentage",
        "ConditioningSetAreaPercentageVideo",
        "ConditioningSetAreaStrength",
    ] {
        let node = registry
            .node(class_type)
            .ok_or("Registered Source strength node")?;
        assert_eq!(
            node.implementation_version(),
            "source-conditioning-part-three-strength-f64-v3"
        );
        for strength in [
            0.1_f64,
            f64::from(0.1_f32),
            f64::from_bits(0.1_f64.to_bits() + 1),
            -0.0,
            1e-46,
        ] {
            let mut arguments = inputs(class_type, &handle);
            arguments.insert("strength".to_owned(), number(strength));
            assert!(node.demanded_lazy_inputs(&context, &arguments)?.is_empty());
            let output = first_output(smol::block_on(node.execute(context.clone(), arguments))?)?;
            {
                let result = resolved(&context, &output)?;
                assert_aliases(&result, &source)?;
                for entry in result.entries() {
                    assert_eq!(entry.options().strength.to_bits(), strength.to_bits());
                }
            }
            context
                .handle_store()
                .revoke(&output, &context.cancellation)?;
        }
    }
    assert_eq!(generation.len(), 1);
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    assert_eq!(source.digest(), digest);
    Ok(())
}
#[test]
fn task618_schema_bounds_and_empty_mapped_column_reject_before_publication() -> TestResult {
    let (_, _, generation, context, handle, _) = setup(20)?;
    let registry = generated_native_node_registry_projection(None)?;
    for (class_type, name, value) in [
        ("ConditioningSetArea", "width", unsigned(63)),
        ("ConditioningSetArea", "width", unsigned(16_385)),
        ("ConditioningSetArea", "height", unsigned(16_385)),
        ("ConditioningSetArea", "x", unsigned(16_385)),
        ("ConditioningSetArea", "y", unsigned(16_385)),
        ("ConditioningSetAreaPercentage", "width", number(1.01)),
        ("ConditioningSetAreaStrength", "strength", number(-0.1)),
        ("ConditioningSetTimestepRange", "start", number(1.01)),
        ("ConditioningStableAudio", "seconds_total", number(1000.1)),
    ] {
        let node = registry.node(class_type).ok_or("Node")?;
        let mut arguments = inputs(class_type, &handle);
        arguments.insert(name.to_owned(), value);
        assert!(node.cache_change_token(&arguments).is_err());
        assert!(smol::block_on(node.execute(context.clone(), arguments)).is_err());
        assert_eq!(generation.len(), 1);
    }
    let node = registry
        .node("ConditioningSetAreaStrength")
        .ok_or("Strength")?;
    let mut arguments = inputs("ConditioningSetAreaStrength", &handle);
    arguments.insert(
        "conditioning".to_owned(),
        NativeValue::List { values: Vec::new() },
    );
    assert!(node.demanded_lazy_inputs(&context, &arguments).is_err());
    assert_eq!(generation.len(), 1);
    Ok(())
}
#[test]
fn task618_foreign_stale_and_wrong_kind_handles_fail_and_valid_recovery_succeeds() -> TestResult {
    let (backend, authority, generation, context, handle, _) = setup(20)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ConditioningSetAreaStrength")
        .ok_or("Strength")?;
    for fresh in [
        NativeHandleStoreGeneration::new()?,
        generation.fresh_generation()?,
    ] {
        let context = node_context(
            backend.clone(),
            &authority,
            fresh.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x61844))),
            CancellationToken::default(),
            BUDGET,
        )?;
        assert!(
            node.cache_dependencies(&context, &inputs("ConditioningSetAreaStrength", &handle))
                .is_err()
        );
        assert!(
            smol::block_on(node.execute(
                context.clone(),
                inputs("ConditioningSetAreaStrength", &handle)
            ))
            .is_err()
        );
        assert!(fresh.is_empty());
        let good = publish(
            &context,
            conditioning(&context, "recovery", &[(1, 4.0)], true)?,
        )?;
        let output = first_output(smol::block_on(node.execute(
            context.clone(),
            inputs("ConditioningSetAreaStrength", &good),
        ))?)?;
        assert_eq!(resolved(&context, &output)?.entries().len(), 1);
    }
    let wrong = context.handle_store().publish(
        NativeStoredPayload::Tensor(Arc::new(sim_tensor::NativeTensorPayload::from_tensor(
            sim_tensor::NativeTensorRole::Mask,
            tensor(
                resolved(&context, &handle)?
                    .entries()
                    .first()
                    .ok_or("Source entry")?,
            )?
            .clone(),
        )?)),
        &context.cancellation,
    )?;
    assert!(
        smol::block_on(node.execute(
            context.clone(),
            inputs("ConditioningSetAreaStrength", &wrong)
        ))
        .is_err()
    );
    context
        .handle_store()
        .revoke(&handle, &context.cancellation)?;
    assert!(
        smol::block_on(node.execute(context, inputs("ConditioningSetAreaStrength", &handle)))
            .is_err()
    );
    Ok(())
}
#[test]
fn task618_precancel_and_single_output_capacity_failure_leave_inputs_intact() -> TestResult {
    let (backend, _, generation, context, handle, source) = setup(1)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let registry = generated_native_node_registry_projection(None)?;
    for class_type in [
        "ConditioningSetArea",
        "ConditioningSetAreaPercentage",
        "ConditioningSetAreaPercentageVideo",
        "ConditioningSetAreaStrength",
        "ConditioningSetTimestepRange",
        "ConditioningStableAudio",
    ] {
        let node = registry.node(class_type).ok_or("Node")?;
        assert!(
            smol::block_on(node.execute(context.clone(), inputs(class_type, &handle))).is_err()
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
    context.cancellation.cancel();
    let error = smol::block_on(
        registry
            .node("ConditioningSetArea")
            .ok_or("Area")?
            .execute(context.clone(), inputs("ConditioningSetArea", &handle)),
    )
    .err()
    .ok_or("Cancelled success")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    assert_eq!(source.entries().len(), 2);
    assert!(context.cancellation.is_cancelled());
    Ok(())
}
#[test]
fn task618_audio_second_publication_capacity_failure_rolls_back_then_recovers() -> TestResult {
    let (backend, _, generation, context, positive, source) = setup(3)?;
    let negative = publish(
        &context,
        conditioning(&context, "negative", &[(1, 3.0)], false)?,
    )?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningStableAudio").ok_or("Audio")?;
    let mut arguments = inputs("ConditioningStableAudio", &positive);
    arguments.insert(
        "negative".to_owned(),
        NativeValue::Handle {
            value: negative.clone(),
        },
    );
    assert!(smol::block_on(node.execute(context.clone(), arguments)).is_err());
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    assert_eq!(resolved(&context, &positive)?.digest(), source.digest());
    context
        .handle_store()
        .revoke(&negative, &context.cancellation)?;
    let outputs = output_handles(smol::block_on(node.execute(
        context.clone(),
        inputs("ConditioningStableAudio", &positive),
    ))?)?;
    assert_eq!(outputs.len(), 2);
    assert_eq!(generation.len(), 3);
    assert_eq!(resolved(&context, &positive)?.digest(), source.digest());
    Ok(())
}

fn workflow_executor(
    backend: Arc<CpuBackend>,
) -> Result<
    (
        sim_runtime::NativeImageExecutor,
        BTreeMap<String, sim_nodes::NativeNodeDescriptor>,
    ),
    Box<dyn Error>,
> {
    let mut descriptors = BTreeMap::new();
    let executor =
        sim_runtime::NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
            sim_types::ProfileId(Uuid::from_u128(0x618501)),
            BTreeMap::new(),
            true,
            backend,
            Arc::new(sim_test_support::NativeDiffusionFixture::checked_in()),
            &mut |registry: &mut sim_runtime::NativeNodeRegistry| {
                for class_type in [
                    "CheckpointLoaderSimple",
                    "CLIPTextEncode",
                    "ConditioningSetAreaStrength",
                    "ConditioningSetTimestepRange",
                ] {
                    let descriptor = registry.descriptor(class_type).ok_or_else(|| {
                        sim_runtime::NativeImageRuntimeError::Registry(
                            "Genuine registered conditioning node required".to_owned(),
                        )
                    })?;
                    if class_type != "CheckpointLoaderSimple" {
                        descriptor.validate_exact_schema_v2().map_err(|error| {
                            sim_runtime::NativeImageRuntimeError::Registry(error.to_string())
                        })?;
                    }
                    if registry.binding_disposition(class_type)
                        != Some(sim_nodes::NativeNodeBindingDisposition::Executable)
                    {
                        return Err(sim_runtime::NativeImageRuntimeError::Registry(
                            "Executable conditioning binding required".to_owned(),
                        ));
                    }
                    descriptors.insert(class_type.to_owned(), descriptor.clone());
                }
                Ok(())
            },
        )?;
    Ok((executor, descriptors))
}
fn workflow_literal(value: NativeValue) -> sim_runtime::InputBinding {
    sim_runtime::InputBinding::Literal { value }
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
        prompt_id: PromptId(Uuid::from_u128(0x618611)),
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

#[test]
fn task618_registered_mapper_strength_to_range_persistence_cache_and_aliases() -> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKFLOW_BUDGET)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = workflow_executor(backend.clone())?;
    let attempt = AttemptId(Uuid::from_u128(0x618502));
    let generation = executor.handle_store_generation_for_test();
    let context = node_context(
        backend,
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        WORKFLOW_BUDGET,
    )?;
    let plan = workflow_plan(
        &descriptors,
        vec![
            (
                "checkpoint",
                "CheckpointLoaderSimple",
                BTreeMap::from([(
                    "ckpt_name".to_owned(),
                    workflow_literal(primitive(NativePrimitive::String(
                        "model.safetensors".to_owned(),
                    ))),
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
                            values: vec![primitive(NativePrimitive::String("a test".to_owned()))],
                        }),
                    ),
                ]),
            ),
            (
                "strength",
                "ConditioningSetAreaStrength",
                BTreeMap::from([
                    ("conditioning".to_owned(), workflow_link("encode", 0)),
                    (
                        "strength".to_owned(),
                        workflow_literal(NativeValue::List {
                            values: vec![number(0.5), number(2.0)],
                        }),
                    ),
                ]),
            ),
            (
                "range",
                "ConditioningSetTimestepRange",
                BTreeMap::from([
                    ("conditioning".to_owned(), workflow_link("strength", 0)),
                    ("start".to_owned(), workflow_literal(number(0.8))),
                    ("end".to_owned(), workflow_literal(number(0.2))),
                ]),
            ),
        ],
    )?;
    let restored: sim_runtime::CompiledPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
    let result = executor.execute_blocking(
        &restored,
        attempt,
        CancellationToken::default(),
        0,
        authority.authorize_workspace(WORKFLOW_BUDGET)?,
    )?;
    assert_eq!(
        result.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    assert_eq!(result.report.cache_hits, 0);
    let source_column = workflow_column(
        result
            .report
            .outputs
            .get(&NodeId("encode".to_owned()))
            .ok_or("Retained CLIP source column")?,
    )?;
    assert_eq!(source_column.len(), 1);
    let source = resolved(&context, source_column.first().ok_or("CLIP source handle")?)?;
    for entry in source.entries() {
        assert!(matches!(
            entry.metadata().get(ConditioningMetadataKey::PooledOutput),
            Some(ConditioningMetadataValue::PooledOutput(None))
        ));
    }
    let columns = workflow_column(
        result
            .report
            .outputs
            .get(&NodeId("range".to_owned()))
            .ok_or("Range column")?,
    )?;
    assert_eq!(columns.len(), 2);
    for (handle, strength) in columns.iter().zip([0.5, 2.0]) {
        let actual = resolved(&context, handle)?;
        assert_aliases(&actual, &source)?;
        for entry in actual.entries() {
            assert_eq!(entry.options().strength, strength);
            let Some(ConditioningMetadataValue::TimestepRange(range)) =
                entry.metadata().get(ConditioningMetadataKey::TimestepRange)
            else {
                return Err("Mapped range missing".into());
            };
            assert_eq!((range.start_percent(), range.end_percent()), (0.8, 0.2));
        }
    }
    let cached = executor.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x618503)),
        CancellationToken::default(),
        1,
        authority.authorize_workspace(WORKFLOW_BUDGET)?,
    )?;
    assert_eq!(
        cached.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        cached.report.error
    );
    assert!(cached.report.cache_hits > 0);
    for identifier in ["encode", "strength", "range"] {
        let original = workflow_column(
            result
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Original column")?,
        )?;
        let replay = workflow_column(
            cached
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Cached column")?,
        )?;
        assert_eq!(original.len(), replay.len());
        for (original, replay) in original.iter().zip(&replay) {
            assert_eq!(original.digest_sha256(), replay.digest_sha256());
        }
    }
    Ok(())
}

#[test]
fn task618_source_percentage_endpoints_and_zero_reversed_audio_pairs_are_stored() -> TestResult {
    let (backend, _, _, context, handle, source) = setup(40)?;
    let before = backend.memory_snapshot().current_bytes;
    let registry = generated_native_node_registry_projection(None)?;
    for (class_type, offset_name) in [
        ("ConditioningSetAreaPercentage", "x"),
        ("ConditioningSetAreaPercentageVideo", "z"),
    ] {
        let mut arguments = inputs(class_type, &handle);
        arguments.insert(offset_name.to_owned(), number(1.0));
        let node = registry.node(class_type).ok_or("Percentage node")?;
        let output = first_output(smol::block_on(node.execute(context.clone(), arguments))?)?;
        let result = resolved(&context, &output)?;
        assert_aliases(&result, &source)?;
        let area = result
            .entries()
            .first()
            .ok_or("Percentage entry")?
            .options()
            .region
            .as_ref()
            .ok_or("Region")?;
        let ConditioningRegion::Percentage { offsets, .. } = area else {
            return Err("Percentage required".into());
        };
        assert_eq!(
            offsets.get(if class_type.ends_with("Video") { 0 } else { 1 }),
            Some(&1.0)
        );
        assert!(
            area.resolve(if class_type.ends_with("Video") {
                &[4, 8, 8]
            } else {
                &[8, 8]
            })
            .is_err()
        );
    }
    let node = registry
        .node("ConditioningStableAudio")
        .ok_or("Audio node")?;
    for (start, total) in [(0.0, 0.0), (-0.0, -0.0), (50.0, 47.125), (1000.0, 0.0)] {
        let mut arguments = inputs("ConditioningStableAudio", &handle);
        arguments.insert("seconds_start".to_owned(), number(start));
        arguments.insert("seconds_total".to_owned(), number(total));
        let outputs = output_handles(smol::block_on(node.execute(context.clone(), arguments))?)?;
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            let result = resolved(&context, &output)?;
            assert_aliases(&result, &source)?;
            for entry in result.entries() {
                assert!(
                    matches!(entry.metadata().get(ConditioningMetadataKey::SecondsStart), Some(ConditioningMetadataValue::FiniteF64(value)) if value.to_bits() == start.to_bits())
                );
                assert!(
                    matches!(entry.metadata().get(ConditioningMetadataKey::SecondsTotal), Some(ConditioningMetadataValue::FiniteF64(value)) if value.to_bits() == total.to_bits())
                );
            }
        }
    }
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    Ok(())
}

fn zero_source(
    context: &NativeNodeContext,
    kind: usize,
    pooled_kind: usize,
) -> Result<ConditioningSet, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let primary = tensor_from_f32(
        compute.backend(),
        &[1, 1, 2],
        &[f32::NAN, f32::INFINITY],
        &execution,
    )?;
    let pooled = tensor_from_f32(
        compute.backend(),
        &[1, 2],
        &[f32::NEG_INFINITY, -3.0],
        &execution,
    )?;
    let value = match kind {
        0 => ConditioningValue::regular(primary)?,
        1 => ConditioningValue::cross_attention(primary)?,
        2 => ConditioningValue::noise_shape(primary)?,
        _ => return Err("Unknown test kind".into()),
    };
    let mut metadata = ConditioningMetadataMap::checked([
        (
            ConditioningMetadataKey::PooledOutputControlnet,
            ConditioningMetadataValue::ControlnetPooledOutput(Some(pooled.clone())),
        ),
        (
            ConditioningMetadataKey::TimestepRange,
            ConditioningMetadataValue::TimestepRange(ConditioningTimestepRange::source(0.8, 0.2)?),
        ),
    ])?;
    match pooled_kind {
        0 => {}
        1 => metadata.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(None),
        )?,
        2 => metadata.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(pooled)),
        )?,
        _ => return Err("Unknown pooled test kind".into()),
    }
    let entry = ConditioningEntry::checked_with_metadata(
        "zero.original",
        value,
        ConditioningEntryOptions {
            strength: 0.5,
            ..Default::default()
        },
        ConditioningReferences::checked(
            None,
            vec![ConditioningHookReference::checked("zero.hook")?],
        )?,
        metadata,
    )?;
    Ok(ConditioningSet::checked(
        identity("zero-out")?,
        vec![entry],
        &context.cancellation,
    )?)
}
fn zero_values(
    context: &NativeNodeContext,
    tensor: &Tensor,
) -> Result<sim_tensor::CpuWorkspaceVec<f32>, Box<dyn Error>> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    Ok(tensor_to_f32(compute.backend(), tensor, &execution)?)
}
fn zero_tensor(entry: &ConditioningEntry) -> Result<&Tensor, Box<dyn Error>> {
    match entry.value() {
        ConditioningValue::Regular(value)
        | ConditioningValue::CrossAttention(value)
        | ConditioningValue::NoiseShape(value) => Ok(value),
        _ => Err("Tensor required".into()),
    }
}

#[test]
fn task618_zero_out_primary_pooled_absent_none_some_preserves_metadata_and_inputs() -> TestResult {
    let (_, _, _, context, _, _) = setup(40)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut node")?;
    for kind in 0..3 {
        for pooled_kind in 0..3 {
            let source = zero_source(&context, kind, pooled_kind)?;
            let digest = source.digest().to_owned();
            let handle = publish(&context, source.clone())?;
            let output = first_output(smol::block_on(
                node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
            )?)?;
            let result = resolved(&context, &output)?;
            assert_eq!(result.identity(), source.identity());
            let original = source.entries().first().ok_or("Original entry")?;
            let actual = result.entries().first().ok_or("Zero entry")?;
            assert_eq!(actual.identifier(), original.identifier());
            assert_eq!(actual.references(), original.references());
            assert_eq!(actual.options().strength, original.options().strength);
            assert_eq!(actual.options().region, original.options().region);
            assert_eq!(actual.options().window, original.options().window);
            assert_eq!(
                actual.options().default_region,
                original.options().default_region
            );
            let original_tensor = zero_tensor(original)?;
            let actual_tensor = zero_tensor(actual)?;
            assert_eq!(actual_tensor.descriptor(), original_tensor.descriptor());
            assert_ne!(actual_tensor.storage_id(), original_tensor.storage_id());
            assert!(
                zero_values(&context, actual_tensor)?
                    .iter()
                    .all(|value| value.to_bits() == 0)
            );
            let original_values = zero_values(&context, original_tensor)?;
            assert!(original_values.first().is_some_and(|value| value.is_nan()));
            assert_eq!(original_values.get(1), Some(&f32::INFINITY));
            match (
                actual.metadata().get(ConditioningMetadataKey::PooledOutput),
                original
                    .metadata()
                    .get(ConditioningMetadataKey::PooledOutput),
            ) {
                (None, None)
                | (
                    Some(ConditioningMetadataValue::PooledOutput(None)),
                    Some(ConditioningMetadataValue::PooledOutput(None)),
                ) => {}
                (
                    Some(ConditioningMetadataValue::PooledOutput(Some(actual))),
                    Some(ConditioningMetadataValue::PooledOutput(Some(original))),
                ) => {
                    assert_ne!(actual.storage_id(), original.storage_id());
                    assert_eq!(actual.descriptor(), original.descriptor());
                    assert!(
                        zero_values(&context, actual)?
                            .iter()
                            .all(|value| value.to_bits() == 0)
                    );
                    assert_eq!(
                        &*zero_values(&context, original)?,
                        &[f32::NEG_INFINITY, -3.0]
                    );
                }
                _ => return Err("Pooled absence/None/Some changed".into()),
            }
            let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(original_control))) =
                original
                    .metadata()
                    .get(ConditioningMetadataKey::PooledOutputControlnet)
            else {
                return Err("Original control pool".into());
            };
            let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(actual_control))) =
                actual
                    .metadata()
                    .get(ConditioningMetadataKey::PooledOutputControlnet)
            else {
                return Err("Control pool".into());
            };
            assert_eq!(actual_control.storage_id(), original_control.storage_id());
            assert_eq!(source.digest(), digest);
            result.validate()?;
        }
    }
    Ok(())
}

#[test]
fn task618_zero_out_precancel_publication_and_backend_capacity_roll_back_then_recover() -> TestResult
{
    let (backend, authority, generation, context, existing, _) = setup(2)?;
    context
        .handle_store()
        .revoke(&existing, &context.cancellation)?;
    let source = zero_source(&context, 1, 2)?;
    let digest = source.digest().to_owned();
    let handle = publish(&context, source.clone())?;
    let filler = publish(&context, source)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let snapshot = || {
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        )
    };
    let before = snapshot();
    let error =
        smol::block_on(node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("Full store accepted output")?;
    assert!(
        error
            .message
            .contains("native handle store capacity is exhausted")
    );
    assert_eq!(snapshot(), before);
    context
        .handle_store()
        .revoke(&filler, &context.cancellation)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let remaining = BUDGET
        .checked_sub(backend.memory_snapshot().current_bytes)
        .ok_or("Budget")?;
    let reservation = backend.reserve_workspace(&execution, remaining)?;
    let before = snapshot();
    let error =
        smol::block_on(node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("Saturated backend accepted output")?;
    assert_eq!(error.code, "conditioning_execution_failed");
    assert_eq!(snapshot(), before);
    drop(reservation);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = node_context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(context.attempt_id),
        cancelled,
        BUDGET,
    )?;
    let before = snapshot();
    let error =
        smol::block_on(node.execute(cancelled_context, inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("Cancelled success")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(snapshot(), before);
    let output = first_output(smol::block_on(
        node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
    )?)?;
    assert_eq!(generation.len(), 2);
    let result = resolved(&context, &output)?;
    assert!(
        zero_values(
            &context,
            zero_tensor(result.entries().first().ok_or("Recovered entry")?)?
        )?
        .iter()
        .all(|value| value.to_bits() == 0)
    );
    assert_eq!(resolved(&context, &handle)?.digest(), digest);
    Ok(())
}

#[test]
fn task618_zero_out_shared_input_aliases_create_distinct_zero_storages_per_entry() -> TestResult {
    let (_, _, _, context, _, _) = setup(10)?;
    let single = zero_source(&context, 0, 2)?;
    let original = single.entries().first().ok_or("Original")?;
    let source = ConditioningSet::checked(
        single.identity().clone(),
        vec![
            original.clone(),
            ConditioningEntry::checked_with_metadata(
                "zero.second",
                original.value().clone(),
                original.options().clone(),
                original.references().clone(),
                original.metadata().clone(),
            )?,
        ],
        &context.cancellation,
    )?;
    let handle = publish(&context, source.clone())?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let output = first_output(smol::block_on(
        node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
    )?)?;
    let result = resolved(&context, &output)?;
    let first = result.entries().first().ok_or("First")?;
    let second = result.entries().get(1).ok_or("Second")?;
    assert_ne!(
        zero_tensor(first)?.storage_id(),
        zero_tensor(second)?.storage_id()
    );
    let pool = |entry: &ConditioningEntry| match entry
        .metadata()
        .get(ConditioningMetadataKey::PooledOutput)
    {
        Some(ConditioningMetadataValue::PooledOutput(Some(value))) => Ok(value.storage_id()),
        _ => Err("Pool missing"),
    };
    assert_ne!(pool(first)?, pool(second)?);
    for (actual, original) in result.entries().iter().zip(source.entries()) {
        assert_eq!(actual.identifier(), original.identifier());
        assert_eq!(actual.references(), original.references());
        assert!(
            zero_values(&context, zero_tensor(actual)?)?
                .iter()
                .all(|value| value.to_bits() == 0)
        );
    }
    assert_eq!(
        zero_tensor(source.entries().first().ok_or("Source")?)?.storage_id(),
        zero_tensor(source.entries().get(1).ok_or("Second source")?)?.storage_id()
    );
    Ok(())
}

#[test]
fn task618_registered_mapper_binary64_strength_bits_persistence_cache_and_aliases() -> TestResult {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKFLOW_BUDGET)?;
    let backend = Arc::new(backend);
    let (executor, descriptors) = workflow_executor(backend.clone())?;
    let attempt = AttemptId(Uuid::from_u128(0x618652));
    let generation = executor.handle_store_generation_for_test();
    let context = node_context(
        backend,
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
        WORKFLOW_BUDGET,
    )?;
    let plan = workflow_plan(
        &descriptors,
        vec![
            (
                "checkpoint",
                "CheckpointLoaderSimple",
                BTreeMap::from([(
                    "ckpt_name".to_owned(),
                    workflow_literal(primitive(NativePrimitive::String(
                        "model.safetensors".to_owned(),
                    ))),
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
                            values: vec![primitive(NativePrimitive::String("a test".to_owned()))],
                        }),
                    ),
                ]),
            ),
            (
                "strength",
                "ConditioningSetAreaStrength",
                BTreeMap::from([
                    ("conditioning".to_owned(), workflow_link("encode", 0)),
                    (
                        "strength".to_owned(),
                        workflow_literal(NativeValue::List {
                            values: vec![
                                number(0.1),
                                number(f64::from_bits(0.1_f64.to_bits() + 1)),
                                number(-0.0),
                            ],
                        }),
                    ),
                ]),
            ),
            (
                "range",
                "ConditioningSetTimestepRange",
                BTreeMap::from([
                    ("conditioning".to_owned(), workflow_link("strength", 0)),
                    ("start".to_owned(), workflow_literal(number(0.8))),
                    ("end".to_owned(), workflow_literal(number(0.2))),
                ]),
            ),
        ],
    )?;
    let restored: sim_runtime::CompiledPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
    let result = executor.execute_blocking(
        &restored,
        attempt,
        CancellationToken::default(),
        0,
        authority.authorize_workspace(WORKFLOW_BUDGET)?,
    )?;
    assert_eq!(
        result.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        result.report.error
    );
    assert_eq!(result.report.cache_hits, 0);
    let source_column = workflow_column(
        result
            .report
            .outputs
            .get(&NodeId("encode".to_owned()))
            .ok_or("Retained CLIP source column")?,
    )?;
    assert_eq!(source_column.len(), 1);
    let source = resolved(&context, source_column.first().ok_or("CLIP source handle")?)?;
    for entry in source.entries() {
        assert!(matches!(
            entry.metadata().get(ConditioningMetadataKey::PooledOutput),
            Some(ConditioningMetadataValue::PooledOutput(None))
        ));
    }
    let columns = workflow_column(
        result
            .report
            .outputs
            .get(&NodeId("range".to_owned()))
            .ok_or("Range column")?,
    )?;
    assert_eq!(columns.len(), 3);
    for (handle, strength) in
        columns
            .iter()
            .zip([0.1_f64, f64::from_bits(0.1_f64.to_bits() + 1), -0.0])
    {
        let actual = resolved(&context, handle)?;
        assert_aliases(&actual, &source)?;
        for entry in actual.entries() {
            assert_eq!(entry.options().strength.to_bits(), strength.to_bits());
            let Some(ConditioningMetadataValue::TimestepRange(range)) =
                entry.metadata().get(ConditioningMetadataKey::TimestepRange)
            else {
                return Err("Mapped range missing".into());
            };
            assert_eq!((range.start_percent(), range.end_percent()), (0.8, 0.2));
        }
    }
    let cached = executor.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x618653)),
        CancellationToken::default(),
        1,
        authority.authorize_workspace(WORKFLOW_BUDGET)?,
    )?;
    assert_eq!(
        cached.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        cached.report.error
    );
    assert!(cached.report.cache_hits > 0);
    for identifier in ["encode", "strength", "range"] {
        let original = workflow_column(
            result
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Original column")?,
        )?;
        let replay = workflow_column(
            cached
                .report
                .outputs
                .get(&NodeId(identifier.to_owned()))
                .ok_or("Cached column")?,
        )?;
        assert_eq!(original.len(), replay.len());
        for (original, replay) in original.iter().zip(&replay) {
            assert_eq!(original.digest_sha256(), replay.digest_sha256());
        }
    }
    Ok(())
}

fn captured_f64(value: &serde_json::Value) -> Result<f64, Box<dyn Error>> {
    Ok(f64::from_bits(
        value
            .get("f64_bits")
            .and_then(serde_json::Value::as_u64)
            .ok_or("Source binary64 bits required")?,
    ))
}
fn captured_tensor(
    context: &NativeNodeContext,
    value: &serde_json::Value,
) -> Result<Tensor, Box<dyn Error>> {
    let shape = value
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source tensor shape required")?
        .iter()
        .map(|dimension| dimension.as_u64().ok_or("Source dimension required"))
        .collect::<Result<Vec<_>, _>>()?;
    let values = value
        .get("words32")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source tensor bits required")?
        .iter()
        .map(|value| {
            let word = u32::try_from(value.as_u64().ok_or("Source F32 word required")?)?;
            Ok::<_, Box<dyn Error>>(f32::from_bits(word))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        value.get("dtype").and_then(serde_json::Value::as_str),
        Some("F32")
    );
    assert_eq!(
        value.get("device").and_then(serde_json::Value::as_str),
        Some("CPU")
    );
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    Ok(tensor_from_f32(
        compute.backend(),
        &shape,
        &values,
        &execution,
    )?)
}
fn assert_captured_tensor(
    context: &NativeNodeContext,
    actual: &Tensor,
    expected: &serde_json::Value,
) -> TestResult {
    let shape = expected
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source shape required")?
        .iter()
        .map(|value| value.as_u64().ok_or("Source dimension required"))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(actual.descriptor().shape(), shape.as_slice());
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let values = tensor_to_f32(compute.backend(), actual, &execution)?;
    let expected = expected
        .get("words32")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source F32 words required")?;
    assert_eq!(values.len(), expected.len());
    for (actual, expected) in values.iter().zip(expected) {
        assert_eq!(
            u64::from(actual.to_bits()),
            expected.as_u64().ok_or("Source word")?
        );
    }
    Ok(())
}
fn captured_conditioning(
    context: &NativeNodeContext,
    column: &serde_json::Value,
) -> Result<ConditioningSet, Box<dyn Error>> {
    let mut entries = Vec::new();
    for (index, value) in column
        .as_array()
        .ok_or("Source column required")?
        .iter()
        .enumerate()
    {
        let primary = captured_tensor(context, value.get("primary").ok_or("Source primary")?)?;
        let metadata = value
            .get("metadata")
            .and_then(serde_json::Value::as_object)
            .ok_or("Source metadata")?;
        let mut stored_metadata = ConditioningMetadataMap::default();
        if let Some(pooled) = metadata.get("pooled_output") {
            let pooled = if pooled.is_null() {
                None
            } else {
                Some(captured_tensor(
                    context,
                    pooled.get("tensor").ok_or("Pooled tensor")?,
                )?)
            };
            stored_metadata.insert(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(pooled),
            )?;
        }
        if metadata
            .keys()
            .any(|key| !matches!(key.as_str(), "strength" | "pooled_output"))
        {
            return Err("Source input metadata needs an additional canonical owner".into());
        }
        entries.push(ConditioningEntry::checked_with_metadata(
            format!("source.{index}"),
            ConditioningValue::cross_attention(primary)?,
            ConditioningEntryOptions {
                strength: captured_f64(metadata.get("strength").ok_or("Source strength")?)?,
                ..Default::default()
            },
            ConditioningReferences::default(),
            stored_metadata,
        )?);
    }
    Ok(ConditioningSet::checked(
        identity("original-part-three")?,
        entries,
        &context.cancellation,
    )?)
}
fn assert_captured_entry(
    context: &NativeNodeContext,
    actual: &ConditioningEntry,
    expected: &serde_json::Value,
) -> TestResult {
    assert_captured_tensor(
        context,
        zero_tensor(actual)?,
        expected.get("primary").ok_or("Primary")?,
    )?;
    let metadata = expected
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .ok_or("Metadata")?;
    assert_eq!(
        actual.options().strength.to_bits(),
        captured_f64(metadata.get("strength").ok_or("Strength")?)?.to_bits()
    );
    let mut expected_metadata_count = 0;
    if let Some(pooled) = metadata.get("pooled_output") {
        expected_metadata_count += 1;
        match (
            actual.metadata().get(ConditioningMetadataKey::PooledOutput),
            pooled.is_null(),
        ) {
            (Some(ConditioningMetadataValue::PooledOutput(None)), true) => {}
            (Some(ConditioningMetadataValue::PooledOutput(Some(actual))), false) => {
                assert_captured_tensor(context, actual, pooled.get("tensor").ok_or("Pooled")?)?;
            }
            _ => return Err("Source pooled absent/None/Some was changed".into()),
        }
    } else {
        assert!(
            actual
                .metadata()
                .get(ConditioningMetadataKey::PooledOutput)
                .is_none()
        );
    }
    if let Some(area) = metadata.get("area") {
        expected_metadata_count += 2;
        let area = area
            .get("tuple")
            .and_then(serde_json::Value::as_array)
            .ok_or("Area tuple")?;
        let expected = if area.first().and_then(serde_json::Value::as_str) == Some("percentage") {
            let numbers = area
                .iter()
                .skip(1)
                .map(captured_f64)
                .collect::<Result<Vec<_>, _>>()?;
            let middle = numbers.len() / 2;
            ConditioningRegion::percentage(
                numbers.get(..middle).ok_or("Sizes")?.to_vec(),
                numbers.get(middle..).ok_or("Offsets")?.to_vec(),
            )?
        } else {
            let numbers = area
                .iter()
                .map(|value| value.as_u64().ok_or("Absolute area"))
                .collect::<Result<Vec<_>, _>>()?;
            let middle = numbers.len() / 2;
            ConditioningRegion::absolute(
                numbers.get(..middle).ok_or("Sizes")?.to_vec(),
                numbers.get(middle..).ok_or("Offsets")?.to_vec(),
            )?
        };
        assert_eq!(actual.options().region.as_ref(), Some(&expected));
        if let ConditioningRegion::Percentage { sizes, offsets } = &expected {
            let Some(ConditioningRegion::Percentage {
                sizes: actual_sizes,
                offsets: actual_offsets,
            }) = actual.options().region.as_ref()
            else {
                return Err("Percentage options missing".into());
            };
            for (actual, expected) in actual_sizes
                .iter()
                .chain(actual_offsets)
                .zip(sizes.iter().chain(offsets))
            {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            let Some(ConditioningMetadataValue::Region(ConditioningRegion::Percentage {
                sizes: actual_sizes,
                offsets: actual_offsets,
            })) = actual.metadata().get(ConditioningMetadataKey::Region)
            else {
                return Err("Percentage metadata missing".into());
            };
            for (actual, expected) in actual_sizes
                .iter()
                .chain(actual_offsets)
                .zip(sizes.iter().chain(offsets))
            {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        assert!(
            matches!(actual.metadata().get(ConditioningMetadataKey::Region), Some(ConditioningMetadataValue::Region(region)) if region == &expected)
        );
        assert!(matches!(
            actual
                .metadata()
                .get(ConditioningMetadataKey::SetAreaToBounds),
            Some(ConditioningMetadataValue::Boolean(false))
        ));
        assert_eq!(
            metadata
                .get("set_area_to_bounds")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
    } else {
        assert!(actual.options().region.is_none());
    }
    if let Some(start) = metadata.get("start_percent") {
        expected_metadata_count += 1;
        let Some(ConditioningMetadataValue::TimestepRange(range)) = actual
            .metadata()
            .get(ConditioningMetadataKey::TimestepRange)
        else {
            return Err("Source timestep metadata missing".into());
        };
        assert_eq!(
            range.start_percent().to_bits(),
            captured_f64(start)?.to_bits()
        );
        assert_eq!(
            range.end_percent().to_bits(),
            captured_f64(metadata.get("end_percent").ok_or("End")?)?.to_bits()
        );
    }
    for (name, key) in [
        ("seconds_start", ConditioningMetadataKey::SecondsStart),
        ("seconds_total", ConditioningMetadataKey::SecondsTotal),
    ] {
        if let Some(expected) = metadata.get(name) {
            expected_metadata_count += 1;
            let Some(ConditioningMetadataValue::FiniteF64(actual)) = actual.metadata().get(key)
            else {
                return Err("Source timing metadata missing".into());
            };
            assert_eq!(actual.to_bits(), captured_f64(expected)?.to_bits());
        }
    }
    assert_eq!(actual.metadata().len(), expected_metadata_count);
    Ok(())
}
fn original_capture() -> Result<serde_json::Value, Box<dyn Error>> {
    let path = std::env::var_os("SIM_CONDITIONING_PART_THREE_SOURCE_CAPTURE")
        .ok_or("Required genuine original Source capture not provided")?;
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)?;
    let expected_sha = std::env::var("SIM_CONDITIONING_PART_THREE_SOURCE_CAPTURE_SHA256")
        .map_err(|_| "Reviewed actual Source capture SHA256 not provided")?;
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected_sha);
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    assert_eq!(
        value
            .get("producer_sha256")
            .and_then(serde_json::Value::as_str),
        Some("2458d05f8f21b200f56bac01cc7a8f1cb2aa175e034db27f16c0cd24834cf5f6")
    );
    assert_eq!(
        value
            .get("python_version")
            .and_then(serde_json::Value::as_str),
        Some("3.12.14")
    );
    assert_eq!(
        value
            .pointer("/package_versions/torch")
            .and_then(serde_json::Value::as_str),
        Some("2.8.0+cpu")
    );
    assert_eq!(
        value.get("format").and_then(serde_json::Value::as_str),
        Some("comfy.source.conditioning-part-three.original-cpu-f32.v1")
    );
    assert_eq!(
        value.get("case_count").and_then(serde_json::Value::as_u64),
        Some(34)
    );
    assert_eq!(
        value
            .pointer("/source_sha256/nodes.py")
            .and_then(serde_json::Value::as_str),
        Some("b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b")
    );
    Ok(value)
}
#[test]
#[ignore = "Requires an independently produced genuine CPU/F32 original Source capture; coordinator must run --ignored --exact after capture and hash verification"]
fn task618_original_cpu_f32_seven_node_storage_and_zero_bits() -> TestResult {
    let capture = original_capture()?;
    let registry = generated_native_node_registry_projection(None)?;
    let mut compared = 0;
    let mut gaps = 0;
    for case in capture
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or("Cases")?
    {
        let gap = case.get("gap").and_then(serde_json::Value::as_str);
        if matches!(gap, Some("native-lyrics-metadata-unrepresented")) {
            assert!(
                case.get("output")
                    .and_then(serde_json::Value::as_array)
                    .is_some()
            );
            gaps += 1;
            continue;
        }
        let (backend, _, _, context, _, _) = setup(8)?;
        let class_type = case
            .get("class_type")
            .and_then(serde_json::Value::as_str)
            .ok_or("Class")?;
        let source_columns = case
            .get("input")
            .and_then(serde_json::Value::as_array)
            .ok_or("Inputs")?;
        let mut arguments = BTreeMap::new();
        let mut sources = Vec::new();
        for (index, column) in source_columns.iter().enumerate() {
            let source = captured_conditioning(&context, column)?;
            let handle = publish(&context, source.clone())?;
            let name = if class_type == "ConditioningStableAudio" {
                if index == 0 { "positive" } else { "negative" }
            } else {
                "conditioning"
            };
            arguments.insert(name.to_owned(), NativeValue::Handle { value: handle });
            sources.push(source);
        }
        for (name, value) in case
            .get("arguments")
            .and_then(serde_json::Value::as_object)
            .ok_or("Arguments")?
        {
            let value = if let Some(value) = value.as_u64() {
                unsigned(value)
            } else {
                number(value.as_f64().ok_or("Numeric argument")?)
            };
            arguments.insert(name.clone(), value);
        }
        let node = registry.node(class_type).ok_or("Registered node")?;
        let output = output_handles(smol::block_on(node.execute(context.clone(), arguments))?)?;
        let expected = case
            .get("output")
            .and_then(serde_json::Value::as_array)
            .ok_or("Outputs")?;
        assert_eq!(output.len(), expected.len());
        for ((handle, expected), source) in output.iter().zip(expected).zip(&sources) {
            let result = resolved(&context, handle)?;
            let expected = expected.as_array().ok_or("Column")?;
            assert_eq!(result.entries().len(), expected.len());
            assert_eq!(result.identity(), source.identity());
            for ((entry, expected), original) in
                result.entries().iter().zip(expected).zip(source.entries())
            {
                assert_eq!(entry.identifier(), original.identifier());
                assert_captured_entry(&context, entry, expected)?;
                if class_type == "ConditioningZeroOut" {
                    assert_ne!(
                        zero_tensor(entry)?.storage_id(),
                        zero_tensor(original)?.storage_id()
                    );
                } else {
                    assert_eq!(
                        zero_tensor(entry)?.storage_id(),
                        zero_tensor(original)?.storage_id()
                    );
                }
            }
        }
        for (source, expected) in sources.iter().zip(source_columns) {
            for (entry, expected) in source
                .entries()
                .iter()
                .zip(expected.as_array().ok_or("Input column")?)
            {
                assert_captured_entry(&context, entry, expected)?;
            }
        }
        assert!(backend.memory_snapshot().current_bytes <= BUDGET);
        compared += 1;
    }
    assert_eq!((compared, gaps), (33, 1));
    Ok(())
}
#[test]
#[ignore = "Requires the independent genuine Source capture and reviewed canonical f64 owner; no inferred sampler acceptance"]
fn task618_original_strength_consumer_bits_and_storage_only_rejections() -> TestResult {
    let capture = original_capture()?;
    let (backend, _, _, context, _, _) = setup(32)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let target = sim_tensor::TensorDescriptor::contiguous(
        vec![1, 1, 16, 16],
        sim_tensor::DType::F32,
        sim_tensor::DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let mut weights = 0;
    let mut areas = 0;
    let mut windows = 0;
    let mut equal_windows = 0;
    for case in capture
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or("Cases")?
    {
        let class_type = case
            .get("class_type")
            .and_then(serde_json::Value::as_str)
            .ok_or("Class")?;
        if !matches!(
            class_type,
            "ConditioningSetAreaStrength"
                | "ConditioningSetAreaPercentage"
                | "ConditioningSetAreaPercentageVideo"
                | "ConditioningSetTimestepRange"
        ) {
            continue;
        }
        let output = case.pointer("/output/0/0").ok_or("Source output")?;
        if class_type == "ConditioningSetAreaStrength" {
            let input = case.pointer("/input/0").ok_or("Source input")?;
            let source = captured_conditioning(&context, input)?;
            let original = source.entries().first().ok_or("Source entry")?;
            let mut options = original.options().clone();
            options.strength = captured_f64(
                output
                    .pointer("/metadata/strength")
                    .ok_or("Source strength")?,
            )?;
            let entry = ConditioningEntry::checked_with_metadata(
                original.identifier(),
                original.value().clone(),
                options,
                original.references().clone(),
                original.metadata().clone(),
            )?;
            let actual = entry.resolve(&target, compute.backend(), &execution)?;
            let expected = case
                .pointer("/sampler/results/0/mult/words32")
                .and_then(serde_json::Value::as_array)
                .ok_or("Actual original sampler weights")?;
            assert_eq!(expected.len(), 256);
            for y in 0..16 {
                for x in 0..16 {
                    let index = usize::try_from(y * 16 + x)?;
                    assert_eq!(
                        u64::from(actual.contribution_weight(&[y, x], None)?.to_bits()),
                        expected
                            .get(index)
                            .and_then(serde_json::Value::as_u64)
                            .ok_or("Source weight")?
                    );
                }
            }
            assert_eq!(
                actual.strength().to_bits(),
                entry.options().strength.to_bits()
            );
            weights += 1;
        } else if case
            .get("name")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|name| matches!(name, "percentage-zero-endpoint" | "video-zero-endpoint"))
        {
            assert_eq!(
                case.pointer("/sampler/status")
                    .and_then(serde_json::Value::as_str),
                Some("success")
            );
            let area = output
                .pointer("/metadata/area/tuple")
                .and_then(serde_json::Value::as_array)
                .ok_or("Area")?;
            let values = area
                .iter()
                .skip(1)
                .map(captured_f64)
                .collect::<Result<Vec<_>, _>>()?;
            let middle = values.len() / 2;
            let region = ConditioningRegion::percentage(
                values.get(..middle).ok_or("Sizes")?.to_vec(),
                values.get(middle..).ok_or("Offsets")?.to_vec(),
            )?;
            let dimensions = case
                .get("dimensions")
                .and_then(serde_json::Value::as_array)
                .ok_or("Dimensions")?
                .iter()
                .map(|value| value.as_u64().ok_or("Dimension"))
                .collect::<Result<Vec<_>, _>>()?;
            assert!(
                region.resolve(&dimensions).is_err(),
                "Existing canonical consumer limitation unexpectedly changed; require Source re-review, not rejection parity"
            );
            areas += 1;
        } else if class_type == "ConditioningSetTimestepRange" {
            let start = captured_f64(output.pointer("/metadata/start_percent").ok_or("Start")?)?;
            let end = captured_f64(output.pointer("/metadata/end_percent").ok_or("End")?)?;
            let range = ConditioningTimestepRange::source(start, end)?;
            if start > end {
                assert!(
                    range.sampler_window().is_err(),
                    "Documented strict consumer prerequisite unexpectedly changed"
                );
                windows += 1;
            } else {
                let window = range.sampler_window()?;
                assert_eq!(window.start_percent().to_bits(), start.to_bits());
                assert_eq!(window.end_percent().to_bits(), end.to_bits());
                assert!(window.contains(start));
                assert!(window.contains(end));
                if start == end {
                    let outside = if start == 0.0 { 1.0 } else { 0.0 };
                    assert!(!window.contains(outside));
                    equal_windows += 1;
                }
            }
            assert_eq!(
                case.pointer("/sampler/status")
                    .and_then(serde_json::Value::as_str),
                Some("success")
            );
        }
    }
    assert_eq!((weights, areas, windows, equal_windows), (8, 2, 1, 2));
    assert!(backend.memory_snapshot().current_bytes <= BUDGET);
    Ok(())
}
struct EmptyZeroOutFixture {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
    generation: NativeHandleStoreGeneration,
    context: NativeNodeContext,
    input: NativeOpaqueHandle,
    source: Arc<ConditioningSet>,
}

fn empty_zero_out_context(
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    Ok(NativeNodeContext::new(
        PromptId(Uuid::from_u128(0x618e01)),
        store.attempt_id(),
        NodeId("conditioning-empty-zero-out".to_owned()),
        cancellation,
        authority.authorize_workspace(0)?,
        store,
    )?)
}

fn empty_zero_out_fixture(
    capacity: usize,
    limit_residency_to_input: bool,
) -> Result<EmptyZeroOutFixture, Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(BUDGET)?;
    let cancellation = CancellationToken::default();
    let source = Arc::new(ConditioningSet::checked(
        identity("conditioning.empty.zero")?,
        Vec::new(),
        &cancellation,
    )?);
    let payload = NativeStoredPayload::Conditioning(source.clone());
    let input_residency = payload.residency()?.resident_bytes()?;
    assert!(input_residency > 0);
    let byte_capacity = if limit_residency_to_input {
        input_residency
    } else {
        usize::try_from(BUDGET)?
    };
    let generation = NativeHandleStoreGeneration::with_capacities(capacity, byte_capacity)?;
    let context = empty_zero_out_context(
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x618e31))),
        cancellation,
    )?;
    let input = context
        .handle_store()
        .publish(payload, &context.cancellation)?;
    assert_eq!(generation.resident_bytes(), input_residency);
    assert!(context.compute_session().is_err());
    Ok(EmptyZeroOutFixture {
        backend,
        authority,
        generation,
        context,
        input,
        source,
    })
}

fn empty_zero_out_snapshot(fixture: &EmptyZeroOutFixture) -> (usize, usize, u64, u64) {
    let memory = fixture.backend.memory_snapshot();
    (
        fixture.generation.len(),
        fixture.generation.resident_bytes(),
        memory.current_bytes,
        memory.peak_bytes,
    )
}

#[test]
fn task618_empty_zero_out_new_handle_same_digest_without_compute_or_tensor_allocation() -> TestResult
{
    let fixture = empty_zero_out_fixture(2, false)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let arguments = inputs("ConditioningZeroOut", &fixture.input);
    let dependencies = node.cache_dependencies(&fixture.context, &arguments)?;
    dependencies.validate()?;
    assert_eq!(dependencies.artifact_digests.len(), 1);
    let before = empty_zero_out_snapshot(&fixture);
    assert_eq!((before.2, before.3), (0, 0));
    let output = first_output(smol::block_on(
        node.execute(fixture.context.clone(), arguments.clone()),
    )?)?;
    assert_ne!(output, fixture.input);
    assert_eq!(output.handle_type(), fixture.input.handle_type());
    assert_eq!(output.digest_sha256(), fixture.input.digest_sha256());
    assert_eq!(fixture.generation.len(), 2);
    assert!(fixture.generation.resident_bytes() > before.1);
    {
        let result = resolved(&fixture.context, &output)?;
        result.validate()?;
        assert!(result.entries().is_empty());
        assert_eq!(result.identity(), fixture.source.identity());
        assert_eq!(result.digest(), fixture.source.digest());
        assert!(!Arc::ptr_eq(&result, &fixture.source));
        let parts = result.resident_parts()?;
        assert!(parts.owned_bytes() > 0);
        assert!(parts.tensor_allocations().is_empty());
        assert_eq!(parts.resident_bytes()?, parts.owned_bytes());
        assert_eq!(
            resolved(&fixture.context, &fixture.input)?.digest(),
            fixture.source.digest()
        );
    }
    assert_eq!(
        node.cache_dependencies(&fixture.context, &arguments)?
            .artifact_digests,
        dependencies.artifact_digests
    );
    assert_eq!(fixture.backend.memory_snapshot().current_bytes, 0);
    assert_eq!(fixture.backend.memory_snapshot().peak_bytes, 0);
    fixture
        .context
        .handle_store()
        .revoke(&output, &fixture.context.cancellation)?;
    assert_eq!(empty_zero_out_snapshot(&fixture), before);
    Ok(())
}

#[test]
fn task618_empty_zero_out_precancel_retains_input_and_fresh_token_recovers() -> TestResult {
    let fixture = empty_zero_out_fixture(2, false)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let context = empty_zero_out_context(
        &fixture.authority,
        fixture
            .generation
            .handle_store_for_attempt(fixture.context.attempt_id),
        cancelled,
    )?;
    let before = empty_zero_out_snapshot(&fixture);
    let failure =
        smol::block_on(node.execute(context, inputs("ConditioningZeroOut", &fixture.input)))
            .err()
            .ok_or("Cancelled empty ZeroOut succeeded")?;
    assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(failure.code, "execution_interrupted");
    assert_eq!(empty_zero_out_snapshot(&fixture), before);
    assert_eq!(
        resolved(&fixture.context, &fixture.input)?.digest(),
        fixture.source.digest()
    );
    let recovered = empty_zero_out_context(
        &fixture.authority,
        fixture
            .generation
            .handle_store_for_attempt(fixture.context.attempt_id),
        CancellationToken::default(),
    )?;
    let output = first_output(smol::block_on(node.execute(
        recovered.clone(),
        inputs("ConditioningZeroOut", &fixture.input),
    ))?)?;
    assert!(resolved(&recovered, &output)?.entries().is_empty());
    recovered
        .handle_store()
        .revoke(&output, &recovered.cancellation)?;
    assert_eq!(empty_zero_out_snapshot(&fixture), before);
    Ok(())
}

#[derive(Debug)]
struct EmptyZeroOutCancelAfterPublish {
    inner: Arc<dyn NativeHandleStore>,
    armed: std::sync::atomic::AtomicBool,
    published: std::sync::atomic::AtomicUsize,
    revoked: std::sync::atomic::AtomicUsize,
}

impl NativeHandleStore for EmptyZeroOutCancelAfterPublish {
    fn identity(&self) -> sim_nodes::NativeHandleStoreIdentity {
        self.inner.identity()
    }
    fn attempt_id(&self) -> AttemptId {
        self.inner.attempt_id()
    }
    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected_type: &sim_nodes::NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<sim_nodes::NativeResolvedPayload, sim_nodes::NativeHandleStoreError> {
        self.inner.resolve(handle, expected_type, cancellation)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, sim_nodes::NativeHandleStoreError> {
        let handle = self.inner.publish(payload, cancellation)?;
        self.published
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if self.armed.swap(false, std::sync::atomic::Ordering::AcqRel) {
            cancellation.cancel();
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
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(())
    }
}

#[test]
fn task618_empty_zero_out_postpublication_cancel_revokes_and_recovers() -> TestResult {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let fixture = empty_zero_out_fixture(2, false)?;
    let store = Arc::new(EmptyZeroOutCancelAfterPublish {
        inner: fixture
            .generation
            .handle_store_for_attempt(fixture.context.attempt_id),
        armed: AtomicBool::new(true),
        published: AtomicUsize::new(0),
        revoked: AtomicUsize::new(0),
    });
    let cancellation = CancellationToken::default();
    let context = empty_zero_out_context(&fixture.authority, store.clone(), cancellation.clone())?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let before = empty_zero_out_snapshot(&fixture);
    let failure =
        smol::block_on(node.execute(context, inputs("ConditioningZeroOut", &fixture.input)))
            .err()
            .ok_or("Late-cancelled empty ZeroOut succeeded")?;
    assert!(cancellation.is_cancelled());
    assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(failure.code, "execution_interrupted");
    assert_eq!(store.published.load(Ordering::Acquire), 1);
    assert_eq!(store.revoked.load(Ordering::Acquire), 1);
    assert_eq!(empty_zero_out_snapshot(&fixture), before);
    assert_eq!(
        resolved(&fixture.context, &fixture.input)?.digest(),
        fixture.source.digest()
    );
    let recovered = empty_zero_out_context(
        &fixture.authority,
        store.clone(),
        CancellationToken::default(),
    )?;
    let output = first_output(smol::block_on(node.execute(
        recovered.clone(),
        inputs("ConditioningZeroOut", &fixture.input),
    ))?)?;
    assert_eq!(store.published.load(Ordering::Acquire), 2);
    assert_eq!(store.revoked.load(Ordering::Acquire), 1);
    assert!(resolved(&recovered, &output)?.entries().is_empty());
    assert_eq!(output.digest_sha256(), fixture.input.digest_sha256());
    recovered
        .handle_store()
        .revoke(&output, &recovered.cancellation)?;
    assert_eq!(store.revoked.load(Ordering::Acquire), 2);
    assert_eq!(empty_zero_out_snapshot(&fixture), before);
    Ok(())
}

#[test]
fn task618_empty_zero_out_handle_and_resident_capacity_fail_without_residue_then_fresh_recovery()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    for (capacity, limit_residency_to_input) in [(1, false), (2, true)] {
        let fixture = empty_zero_out_fixture(capacity, limit_residency_to_input)?;
        let before = empty_zero_out_snapshot(&fixture);
        let failure = smol::block_on(node.execute(
            fixture.context.clone(),
            inputs("ConditioningZeroOut", &fixture.input),
        ))
        .err()
        .ok_or("Full store accepted empty ZeroOut")?;
        assert_eq!(failure.code, "conditioning_execution_failed");
        assert_eq!(
            failure.message,
            "native handle store rejected the operation: native handle store capacity is exhausted"
        );
        assert_eq!(empty_zero_out_snapshot(&fixture), before);
        assert_eq!(
            resolved(&fixture.context, &fixture.input)?.digest(),
            fixture.source.digest()
        );
        let recovered = empty_zero_out_fixture(2, false)?;
        assert_ne!(
            recovered.input.store_identity(),
            fixture.input.store_identity()
        );
        assert_eq!(recovered.source.identity(), fixture.source.identity());
        assert_eq!(recovered.source.digest(), fixture.source.digest());
        let recovered_before = empty_zero_out_snapshot(&recovered);
        let output = first_output(smol::block_on(node.execute(
            recovered.context.clone(),
            inputs("ConditioningZeroOut", &recovered.input),
        ))?)?;
        assert!(resolved(&recovered.context, &output)?.entries().is_empty());
        recovered
            .context
            .handle_store()
            .revoke(&output, &recovered.context.cancellation)?;
        assert_eq!(empty_zero_out_snapshot(&recovered), recovered_before);
        assert_eq!(empty_zero_out_snapshot(&fixture), before);
    }
    Ok(())
}

fn exact_lyrics_source(
    context: &NativeNodeContext,
    lyrics: Option<Option<Tensor>>,
) -> Result<ConditioningSet, Box<dyn Error>> {
    let source = zero_source(context, 1, 2)?;
    let mut metadata = source
        .entries()
        .first()
        .ok_or("source entry")?
        .metadata()
        .clone();
    if let Some(lyrics) = lyrics {
        metadata.insert(
            ConditioningMetadataKey::ConditioningLyrics,
            ConditioningMetadataValue::ConditioningLyrics(lyrics),
        )?;
    }
    Ok(source.with_entry_metadata(vec![metadata], &context.cancellation)?)
}

#[test]
fn task618_exact_lyrics_zero_out_nullable_ace_dtypes_preserves_inputs() -> TestResult {
    use sim_tensor::{DType, DeviceId, TensorDescriptor};
    let (_, _, _, context, _, _) = setup(40)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let mut cases = vec![None, Some(None)];
    for (dtype, bytes) in [
        (
            DType::F32,
            [1.0_f32, -3.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            DType::F64,
            [1.0000000000000002_f64, -0.0]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            DType::I64,
            [7_i64, -3]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
    ] {
        let lyrics = compute
            .backend()
            .upload_bytes(
                TensorDescriptor::contiguous(vec![1, 2], dtype, DeviceId::CPU, StreamId::DEFAULT)?,
                &bytes,
                &execution,
            )?
            .0;
        cases.push(Some(Some(lyrics)));
    }
    for lyrics in cases {
        let source = exact_lyrics_source(&context, lyrics.clone())?;
        let digest = source.digest().to_owned();
        let original_bytes = lyrics
            .as_ref()
            .and_then(Option::as_ref)
            .map(Tensor::contiguous_bytes)
            .transpose()?
            .map(<[u8]>::to_vec);
        let handle = publish(&context, source.clone())?;
        let output = first_output(smol::block_on(
            node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
        )?)?;
        let result = resolved(&context, &output)?;
        let entry = result.entries().first().ok_or("output entry")?;
        let original = source.entries().first().ok_or("input entry")?;
        match (
            lyrics.as_ref(),
            entry
                .metadata()
                .get(ConditioningMetadataKey::ConditioningLyrics),
        ) {
            (None, None)
            | (Some(None), Some(ConditioningMetadataValue::ConditioningLyrics(None))) => {}
            (
                Some(Some(input)),
                Some(ConditioningMetadataValue::ConditioningLyrics(Some(output))),
            ) => {
                assert_eq!(output.descriptor().shape(), input.descriptor().shape());
                assert_eq!(output.descriptor().dtype(), input.descriptor().dtype());
                assert_eq!(output.descriptor().device(), input.descriptor().device());
                assert_ne!(output.storage_id(), input.storage_id());
                assert!(output.contiguous_bytes()?.iter().all(|byte| *byte == 0));
                assert_eq!(Some(input.contiguous_bytes()?.to_vec()), original_bytes);
            }
            _ => return Err("exact lyrics presence changed".into()),
        }
        assert_eq!(entry.identifier(), original.identifier());
        assert_eq!(entry.references(), original.references());
        assert_eq!(
            entry.options().strength.to_bits(),
            original.options().strength.to_bits()
        );
        assert_eq!(
            entry.options().window.start_percent().to_bits(),
            original.options().window.start_percent().to_bits()
        );
        assert_eq!(
            entry.options().window.end_percent().to_bits(),
            original.options().window.end_percent().to_bits()
        );
        assert_eq!(
            entry.options().default_region,
            original.options().default_region
        );
        assert!(entry.options().region.is_none() && original.options().region.is_none());
        assert!(entry.options().mask.is_none() && original.options().mask.is_none());
        assert!(
            zero_tensor(entry)?
                .contiguous_bytes()?
                .iter()
                .all(|byte| *byte == 0)
        );
        let Some(ConditioningMetadataValue::PooledOutput(Some(pool))) =
            entry.metadata().get(ConditioningMetadataKey::PooledOutput)
        else {
            return Err("pool".into());
        };
        assert!(pool.contiguous_bytes()?.iter().all(|byte| *byte == 0));
        let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(retained))) = entry
            .metadata()
            .get(ConditioningMetadataKey::PooledOutputControlnet)
        else {
            return Err("retained control pool".into());
        };
        let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(input))) = original
            .metadata()
            .get(ConditioningMetadataKey::PooledOutputControlnet)
        else {
            return Err("original control pool".into());
        };
        assert_eq!(retained.storage_id(), input.storage_id());
        assert_eq!(retained.contiguous_bytes()?, input.contiguous_bytes()?);
        assert_eq!(resolved(&context, &handle)?.digest(), digest);
    }
    Ok(())
}

#[test]
fn task618_exact_lyrics_zero_out_shared_three_way_inputs_allocate_six_distinct_outputs()
-> TestResult {
    let (_, _, _, context, _, _) = setup(10)?;
    let source = zero_source(&context, 0, 2)?;
    let original = source.entries().first().ok_or("input")?;
    let shared = zero_tensor(original)?.clone();
    let metadata = ConditioningMetadataMap::checked([
        (
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(shared.clone())),
        ),
        (
            ConditioningMetadataKey::ConditioningLyrics,
            ConditioningMetadataValue::ConditioningLyrics(Some(shared.clone())),
        ),
    ])?;
    let entries = ["first", "second"]
        .into_iter()
        .map(|identifier| {
            ConditioningEntry::checked_with_metadata(
                identifier,
                original.value().clone(),
                original.options().clone(),
                original.references().clone(),
                metadata.clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source =
        ConditioningSet::checked(source.identity().clone(), entries, &context.cancellation)?;
    assert_eq!(source.resident_parts()?.tensor_allocations().len(), 1);
    let handle = publish(&context, source.clone())?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let output = first_output(smol::block_on(
        node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
    )?)?;
    let result = resolved(&context, &output)?;
    let allocations = result.resident_parts()?;
    assert_eq!(allocations.tensor_allocations().len(), 6);
    let mut storages = std::collections::HashSet::new();
    for entry in result.entries() {
        let Some(ConditioningMetadataValue::PooledOutput(Some(pool))) =
            entry.metadata().get(ConditioningMetadataKey::PooledOutput)
        else {
            return Err("pool".into());
        };
        let Some(ConditioningMetadataValue::ConditioningLyrics(Some(lyrics))) = entry
            .metadata()
            .get(ConditioningMetadataKey::ConditioningLyrics)
        else {
            return Err("lyrics".into());
        };
        for tensor in [zero_tensor(entry)?, pool, lyrics] {
            assert!(storages.insert(tensor.storage_id()));
            assert_ne!(tensor.storage_id(), shared.storage_id());
            assert!(tensor.contiguous_bytes()?.iter().all(|byte| *byte == 0));
        }
    }
    assert_eq!(resolved(&context, &handle)?.digest(), source.digest());
    Ok(())
}

#[test]
fn task618_exact_lyrics_zero_out_precancel_publication_backend_capacity_and_recovery() -> TestResult
{
    let (backend, authority, generation, context, existing, _) = setup(2)?;
    context
        .handle_store()
        .revoke(&existing, &context.cancellation)?;
    let initial = zero_source(&context, 1, 2)?;
    let source = exact_lyrics_source(
        &context,
        Some(Some(
            zero_tensor(initial.entries().first().ok_or("initial")?)?.clone(),
        )),
    )?;
    let digest = source.digest().to_owned();
    let handle = publish(&context, source.clone())?;
    let filler = publish(&context, source)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let snapshot = || {
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        )
    };
    let before = snapshot();
    let error =
        smol::block_on(node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("Full store accepted output")?;
    assert!(
        error
            .message
            .contains("native handle store capacity is exhausted")
    );
    assert_eq!(snapshot(), before);
    context
        .handle_store()
        .revoke(&filler, &context.cancellation)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let remaining = BUDGET
        .checked_sub(backend.memory_snapshot().current_bytes)
        .ok_or("Budget")?;
    let reservation = backend.reserve_workspace(&execution, remaining)?;
    let before = snapshot();
    let error =
        smol::block_on(node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("Saturated backend accepted output")?;
    assert_eq!(error.code, "conditioning_execution_failed");
    assert_eq!(snapshot(), before);
    drop(reservation);
    let retained = resolved(&context, &handle)?;
    let Some(ConditioningMetadataValue::PooledOutput(Some(pool))) = retained
        .entries()
        .first()
        .ok_or("partial allocation source")?
        .metadata()
        .get(ConditioningMetadataKey::PooledOutput)
    else {
        return Err("partial allocation pool".into());
    };
    let parts = retained.resident_parts()?;
    let pooled_resident = parts
        .tensor_allocations()
        .iter()
        .find(|allocation| allocation.storage_id() == pool.storage_id())
        .ok_or("pool allocation")?
        .resident_bytes();
    let partial_remaining = BUDGET
        .checked_sub(backend.memory_snapshot().current_bytes)
        .and_then(|remaining| remaining.checked_sub(pooled_resident))
        .ok_or("partial allocation budget")?;
    let reservation = backend.reserve_workspace(&execution, partial_remaining)?;
    let before_partial = snapshot();
    let error =
        smol::block_on(node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("partial allocation capacity accepted lyrics")?;
    assert_eq!(error.code, "conditioning_execution_failed");
    assert_eq!(snapshot(), before_partial);
    drop(reservation);
    drop(retained);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = node_context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(context.attempt_id),
        cancelled,
        BUDGET,
    )?;
    let before = snapshot();
    let error =
        smol::block_on(node.execute(cancelled_context, inputs("ConditioningZeroOut", &handle)))
            .err()
            .ok_or("Cancelled success")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(snapshot(), before);
    let output = first_output(smol::block_on(
        node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
    )?)?;
    assert_eq!(generation.len(), 2);
    let result = resolved(&context, &output)?;
    assert!(
        zero_values(
            &context,
            zero_tensor(result.entries().first().ok_or("Recovered entry")?)?
        )?
        .iter()
        .all(|value| value.to_bits() == 0)
    );
    assert_eq!(resolved(&context, &handle)?.digest(), digest);
    Ok(())
}

#[test]
fn task618_exact_lyrics_zero_out_postpublication_cancel_revokes_then_recovers() -> TestResult {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let (backend, authority, generation, initial_context, _, _) = setup(3)?;
    let initial = zero_source(&initial_context, 1, 2)?;
    let source = exact_lyrics_source(
        &initial_context,
        Some(Some(
            zero_tensor(initial.entries().first().ok_or("initial")?)?.clone(),
        )),
    )?;
    let handle = publish(&initial_context, source.clone())?;
    let store = Arc::new(EmptyZeroOutCancelAfterPublish {
        inner: generation.handle_store_for_attempt(initial_context.attempt_id),
        armed: AtomicBool::new(true),
        published: AtomicUsize::new(0),
        revoked: AtomicUsize::new(0),
    });
    let cancellation = CancellationToken::default();
    let context = node_context(
        backend.clone(),
        &authority,
        store.clone(),
        cancellation.clone(),
        BUDGET,
    )?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let snapshot = || {
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        )
    };
    let before = snapshot();
    let failure = smol::block_on(node.execute(context, inputs("ConditioningZeroOut", &handle)))
        .err()
        .ok_or("late cancelled lyrics succeeded")?;
    assert!(cancellation.is_cancelled());
    assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(store.published.load(Ordering::Acquire), 1);
    assert_eq!(store.revoked.load(Ordering::Acquire), 1);
    assert_eq!(snapshot(), before);
    assert_eq!(
        resolved(&initial_context, &handle)?.digest(),
        source.digest()
    );
    let recovered = node_context(
        backend.clone(),
        &authority,
        store.clone(),
        CancellationToken::default(),
        BUDGET,
    )?;
    let output = first_output(smol::block_on(
        node.execute(recovered.clone(), inputs("ConditioningZeroOut", &handle)),
    )?)?;
    let result = resolved(&recovered, &output)?;
    let Some(ConditioningMetadataValue::ConditioningLyrics(Some(lyrics))) = result
        .entries()
        .first()
        .ok_or("recovered")?
        .metadata()
        .get(ConditioningMetadataKey::ConditioningLyrics)
    else {
        return Err("recovered lyrics".into());
    };
    assert!(lyrics.contiguous_bytes()?.iter().all(|byte| *byte == 0));
    assert_eq!(store.published.load(Ordering::Acquire), 2);
    assert_eq!(store.revoked.load(Ordering::Acquire), 1);
    drop(result);
    recovered
        .handle_store()
        .revoke(&output, &recovered.cancellation)?;
    assert_eq!(store.revoked.load(Ordering::Acquire), 2);
    assert_eq!(snapshot(), before);
    Ok(())
}

#[test]
fn task618_exact_lyrics_zero_out_preserves_pooled_none_and_absent_with_present_lyrics() -> TestResult
{
    let (_, _, _, context, _, _) = setup(10)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(&context)?;
    let lyrics = tensor_from_f32(compute.backend(), &[1, 2], &[0.1, -3.5], &execution)?;
    let original_lyrics_bytes = lyrics.contiguous_bytes()?.to_vec();
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    for pooled_kind in [0, 1] {
        let initial = zero_source(&context, 1, pooled_kind)?;
        let mut metadata = initial
            .entries()
            .first()
            .ok_or("input entry")?
            .metadata()
            .clone();
        metadata.insert(
            ConditioningMetadataKey::ConditioningLyrics,
            ConditioningMetadataValue::ConditioningLyrics(Some(lyrics.clone())),
        )?;
        let source = initial.with_entry_metadata(vec![metadata], &context.cancellation)?;
        let digest = source.digest().to_owned();
        let original = source.entries().first().ok_or("source entry")?;
        let original_primary_bytes = zero_tensor(original)?.contiguous_bytes()?.to_vec();
        let handle = publish(&context, source.clone())?;
        let output = first_output(smol::block_on(
            node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
        )?)?;
        let result = resolved(&context, &output)?;
        let entry = result.entries().first().ok_or("output entry")?;
        match (
            pooled_kind,
            entry.metadata().get(ConditioningMetadataKey::PooledOutput),
        ) {
            (0, None) | (1, Some(ConditioningMetadataValue::PooledOutput(None))) => {}
            _ => return Err("pooled None/absence changed with present lyrics".into()),
        }
        let Some(ConditioningMetadataValue::ConditioningLyrics(Some(zero_lyrics))) = entry
            .metadata()
            .get(ConditioningMetadataKey::ConditioningLyrics)
        else {
            return Err("present typed lyrics lost".into());
        };
        assert_eq!(zero_lyrics.descriptor(), lyrics.descriptor());
        assert_ne!(zero_lyrics.storage_id(), lyrics.storage_id());
        assert_ne!(zero_lyrics.storage_id(), zero_tensor(entry)?.storage_id());
        assert!(
            zero_lyrics
                .contiguous_bytes()?
                .iter()
                .all(|byte| *byte == 0)
        );
        assert_eq!(lyrics.contiguous_bytes()?, original_lyrics_bytes.as_slice());
        assert_eq!(
            zero_tensor(entry)?.descriptor(),
            zero_tensor(original)?.descriptor()
        );
        assert_ne!(
            zero_tensor(entry)?.storage_id(),
            zero_tensor(original)?.storage_id()
        );
        assert!(
            zero_tensor(entry)?
                .contiguous_bytes()?
                .iter()
                .all(|byte| *byte == 0)
        );
        assert_eq!(
            zero_tensor(original)?.contiguous_bytes()?,
            original_primary_bytes.as_slice()
        );
        assert_eq!(entry.identifier(), original.identifier());
        assert_eq!(entry.references(), original.references());
        assert_eq!(
            entry.options().strength.to_bits(),
            original.options().strength.to_bits()
        );
        assert_eq!(
            entry.options().window.start_percent().to_bits(),
            original.options().window.start_percent().to_bits()
        );
        assert_eq!(
            entry.options().window.end_percent().to_bits(),
            original.options().window.end_percent().to_bits()
        );
        assert_eq!(
            entry.options().default_region,
            original.options().default_region
        );
        assert!(entry.options().region.is_none() && original.options().region.is_none());
        assert!(entry.options().mask.is_none() && original.options().mask.is_none());
        let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(retained))) = entry
            .metadata()
            .get(ConditioningMetadataKey::PooledOutputControlnet)
        else {
            return Err("retained control pool".into());
        };
        let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(input))) = original
            .metadata()
            .get(ConditioningMetadataKey::PooledOutputControlnet)
        else {
            return Err("original control pool".into());
        };
        assert_eq!(retained.storage_id(), input.storage_id());
        assert_eq!(retained.contiguous_bytes()?, input.contiguous_bytes()?);
        assert_eq!(resolved(&context, &handle)?.digest(), digest);
        result.validate()?;
        drop(result);
        context
            .handle_store()
            .revoke(&output, &context.cancellation)?;
        context
            .handle_store()
            .revoke(&handle, &context.cancellation)?;
    }
    Ok(())
}

fn source13_capture() -> Result<serde_json::Value, Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    let path = std::env::var_os("SIM_CONDITIONING_SOURCE13_CAPTURE")
        .ok_or("Reviewed genuine Source13 capture path required")?;
    let expected = std::env::var("SIM_CONDITIONING_SOURCE13_CAPTURE_SHA256")
        .map_err(|_| "Reviewed actual Source13 capture SHA256 required")?;
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("Full lowercase reviewed Source13 capture SHA256 required".into());
    }
    if std::fs::metadata(&path)?.len() > 8 * 1024 * 1024 {
        return Err("Bounded genuine Source13 capture exceeds eight MiB".into());
    }
    let bytes = std::fs::read(path)?;
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    for (key, expected) in [
        (
            "format",
            "comfy.original.conditioning-zero-out.exact-lyrics.v1",
        ),
        (
            "producer_sha256",
            "df782f6a1e45e759a5c8796478e3a3a0c20d971750a695976d0264f4010e56c7",
        ),
        (
            "required_profile_manifest_sha256",
            "49f9df72a49310cf844512277d34ecc33739d66b7f5575eee76d46b27edd0b4a",
        ),
    ] {
        assert_eq!(
            value.get(key).and_then(serde_json::Value::as_str),
            Some(expected)
        );
    }
    for (path, expected) in [
        (
            "/source_pins/nodes.py",
            "b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b",
        ),
        (
            "/source_pins/node_helpers.py",
            "22317a033cce556331bfedc5f4ab6c2833184fd2bbdf9496c9182a4973a7098a",
        ),
        (
            "/source_pins/comfy_api~1latest~1_io.py",
            "ac986092cace7600461b8547e21f770126958d8c15eef59d79d6709cca4b7717",
        ),
    ] {
        assert_eq!(
            value.pointer(path).and_then(serde_json::Value::as_str),
            Some(expected)
        );
    }
    for (name, expected) in [
        ("torch", "2.8.0+cpu"),
        ("numpy", "1.26.4"),
        ("Pillow", "12.3.0"),
        ("safetensors", "0.8.0"),
        ("av", "19.0.0"),
        ("torchaudio", "2.8.0+cpu"),
        ("torchvision", "0.23.0+cpu"),
        ("transformers", "5.18.0"),
        ("torchsde", "0.2.6"),
    ] {
        assert_eq!(
            value
                .get("package_versions")
                .and_then(|versions| versions.get(name))
                .and_then(serde_json::Value::as_str),
            Some(expected)
        );
    }
    assert_eq!(
        value.get("case_count").and_then(serde_json::Value::as_u64),
        Some(13)
    );
    let cases = value
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source13 cases required")?;
    let names = cases
        .iter()
        .map(|case| {
            case.get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or("Source13 case name required")
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        names,
        [
            "absent",
            "none",
            "f32",
            "i64",
            "f64",
            "scalar",
            "empty",
            "primary-pooled-lyrics-shared",
            "shared-two-entries",
            "pool-none-lyrics-f32",
            "pool-absent-lyrics-f32",
            "list-error",
            "integer-error"
        ]
    );
    for case in cases {
        assert_eq!(
            case.get("input_immutable")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        let name = case
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or("case name")?;
        let expected_gap = if matches!(name, "scalar" | "empty" | "list-error" | "integer-error") {
            Some(name)
        } else {
            None
        };
        assert_eq!(
            case.get("native_representation_gap")
                .and_then(serde_json::Value::as_str),
            expected_gap
        );
        if matches!(name, "list-error" | "integer-error") {
            assert_eq!(
                case.get("status").and_then(serde_json::Value::as_str),
                Some("error")
            );
            assert_eq!(
                case.get("error_type").and_then(serde_json::Value::as_str),
                Some("builtins.TypeError")
            );
        } else {
            assert_eq!(
                case.get("status").and_then(serde_json::Value::as_str),
                Some("success")
            );
        }
    }
    Ok(value)
}

fn source13_tensor_spec(
    value: &serde_json::Value,
) -> Result<(Vec<u64>, sim_tensor::DType, Vec<u8>), Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    use sim_tensor::DType;
    let shape = value
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source13 shape")?
        .iter()
        .map(|dimension| dimension.as_u64().ok_or("Source13 dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    let (dtype, width) = match value.get("dtype").and_then(serde_json::Value::as_str) {
        Some("torch.float32") => (DType::F32, 4_usize),
        Some("torch.float64") => (DType::F64, 8_usize),
        Some("torch.int64") => (DType::I64, 8_usize),
        _ => return Err("Source13 supported exact dtype required".into()),
    };
    assert_eq!(
        value.get("word_bits").and_then(serde_json::Value::as_u64),
        Some(u64::try_from(width)? * 8)
    );
    let words = value
        .get("words")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source13 words")?;
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or("Source13 element count overflow")
    })?;
    assert_eq!(u64::try_from(words.len())?, count);
    let mut bytes = Vec::with_capacity(
        words
            .len()
            .checked_mul(width)
            .ok_or("Source13 byte count overflow")?,
    );
    for word in words {
        let word = word.as_u64().ok_or("Source13 unsigned word")?;
        if width == 4 {
            bytes.extend_from_slice(&u32::try_from(word)?.to_le_bytes());
        } else {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    assert_eq!(
        value
            .get("raw_le_sha256")
            .and_then(serde_json::Value::as_str),
        Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
    );
    Ok((shape, dtype, bytes))
}

fn source13_upload(
    context: &NativeNodeContext,
    value: &serde_json::Value,
) -> Result<Tensor, Box<dyn Error>> {
    use sim_tensor::{DeviceId, TensorDescriptor};
    let (shape, dtype, bytes) = source13_tensor_spec(value)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    Ok(compute
        .backend()
        .upload_bytes(
            TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, StreamId::DEFAULT)?,
            &bytes,
            &execution,
        )?
        .0)
}

fn source13_assert_tensor(actual: &Tensor, expected: &serde_json::Value) -> TestResult {
    let (shape, dtype, bytes) = source13_tensor_spec(expected)?;
    assert_eq!(actual.descriptor().shape(), shape.as_slice());
    assert_eq!(actual.descriptor().dtype(), dtype);
    assert_eq!(actual.descriptor().device(), sim_tensor::DeviceId::CPU);
    assert_eq!(actual.contiguous_bytes()?, bytes.as_slice());
    Ok(())
}

fn source13_alias(
    aliases: &serde_json::Value,
    field: &str,
    row: &str,
    column_aliases: &serde_json::Value,
    column: &str,
) -> Result<bool, Box<dyn Error>> {
    let index = |value: &serde_json::Value, role: &str| -> Result<usize, Box<dyn Error>> {
        value
            .get("role_order")
            .and_then(serde_json::Value::as_array)
            .ok_or("Source13 alias roles")?
            .iter()
            .position(|value| value.as_str() == Some(role))
            .ok_or_else(|| format!("Source13 alias role missing: {role}").into())
    };
    let row = index(aliases, row)?;
    let column = index(column_aliases, column)?;
    aliases
        .get(field)
        .and_then(serde_json::Value::as_array)
        .and_then(|rows| rows.get(row))
        .and_then(serde_json::Value::as_array)
        .and_then(|columns| columns.get(column))
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| "Source13 nonempty tensor alias boolean required".into())
}

fn source13_input_tensor(
    context: &NativeNodeContext,
    role: &str,
    expected: &serde_json::Value,
    aliases: &serde_json::Value,
    prior: &mut BTreeMap<String, Tensor>,
) -> Result<Tensor, Box<dyn Error>> {
    let mut shared: Option<Tensor> = None;
    for (earlier, tensor) in prior.iter() {
        if source13_alias(aliases, "storage_alias_to_input", role, aliases, earlier)? {
            source13_assert_tensor(tensor, expected)?;
            if let Some(storage) = shared.as_ref().map(Tensor::storage_id) {
                assert_eq!(storage, tensor.storage_id());
            }
            shared = Some(tensor.clone());
        }
    }
    let tensor = match shared {
        Some(tensor) => tensor,
        None => source13_upload(context, expected)?,
    };
    prior.insert(role.to_owned(), tensor.clone());
    Ok(tensor)
}

fn source13_conditioning(
    context: &NativeNodeContext,
    case: &serde_json::Value,
) -> Result<ConditioningSet, Box<dyn Error>> {
    let mut prior = BTreeMap::new();
    let mut entries = Vec::new();
    let aliases = case.get("input_aliases").ok_or("Source13 input aliases")?;
    for (index, entry) in case
        .get("input")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source13 input column")?
        .iter()
        .enumerate()
    {
        let pair = entry.as_array().ok_or("Source13 entry pair")?;
        assert_eq!(pair.len(), 2);
        let primary = source13_input_tensor(
            context,
            &format!("{index}:primary"),
            pair.first()
                .and_then(|value| value.get("tensor"))
                .ok_or("Source13 primary tensor")?,
            aliases,
            &mut prior,
        )?;
        let metadata = pair
            .get(1)
            .and_then(serde_json::Value::as_object)
            .ok_or("Source13 metadata")?;
        let keys = metadata
            .keys()
            .filter(|key| {
                !matches!(
                    key.as_str(),
                    "strength" | "pooled_output" | "conditioning_lyrics"
                )
            })
            .map(String::as_str)
            .collect::<Vec<_>>();
        assert_eq!(keys, ["lyrics", "retained"]);
        println!(
            "Source13 case {} entry {index}: unsupported dictionary fields {keys:?}; typed tensor projection only",
            case.get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or("case name")?
        );
        let mut typed = ConditioningMetadataMap::default();
        for (name, key) in [
            ("pooled_output", ConditioningMetadataKey::PooledOutput),
            (
                "conditioning_lyrics",
                ConditioningMetadataKey::ConditioningLyrics,
            ),
        ] {
            if let Some(value) = metadata.get(name) {
                let tensor = if value.is_null() {
                    None
                } else {
                    Some(source13_input_tensor(
                        context,
                        &format!("{index}:{name}"),
                        value
                            .get("tensor")
                            .ok_or("Source13 typed metadata tensor")?,
                        aliases,
                        &mut prior,
                    )?)
                };
                typed.insert(
                    key,
                    if key == ConditioningMetadataKey::PooledOutput {
                        ConditioningMetadataValue::PooledOutput(tensor)
                    } else {
                        ConditioningMetadataValue::ConditioningLyrics(tensor)
                    },
                )?;
            }
        }
        entries.push(ConditioningEntry::checked_with_metadata(
            format!("source13.{index}"),
            ConditioningValue::cross_attention(primary)?,
            ConditioningEntryOptions {
                strength: captured_f64(metadata.get("strength").ok_or("Source13 strength")?)?,
                ..Default::default()
            },
            ConditioningReferences::default(),
            typed,
        )?);
    }
    Ok(ConditioningSet::checked(
        identity("source13-typed-projection")?,
        entries,
        &context.cancellation,
    )?)
}

fn source13_roles(set: &ConditioningSet) -> Result<Vec<(String, &Tensor)>, Box<dyn Error>> {
    let mut roles = Vec::new();
    for (index, entry) in set.entries().iter().enumerate() {
        roles.push((format!("{index}:primary"), zero_tensor(entry)?));
        if let Some(ConditioningMetadataValue::PooledOutput(Some(tensor))) =
            entry.metadata().get(ConditioningMetadataKey::PooledOutput)
        {
            roles.push((format!("{index}:pooled_output"), tensor));
        }
        if let Some(ConditioningMetadataValue::ConditioningLyrics(Some(tensor))) = entry
            .metadata()
            .get(ConditioningMetadataKey::ConditioningLyrics)
        {
            roles.push((format!("{index}:conditioning_lyrics"), tensor));
        }
    }
    Ok(roles)
}

fn source13_assert_entry(actual: &ConditioningEntry, expected: &serde_json::Value) -> TestResult {
    let pair = expected.as_array().ok_or("Source13 output pair")?;
    assert_eq!(pair.len(), 2);
    source13_assert_tensor(
        zero_tensor(actual)?,
        pair.first()
            .and_then(|value| value.get("tensor"))
            .ok_or("Source13 output primary")?,
    )?;
    let metadata = pair
        .get(1)
        .and_then(serde_json::Value::as_object)
        .ok_or("Source13 output metadata")?;
    assert_eq!(
        actual.options().strength.to_bits(),
        captured_f64(metadata.get("strength").ok_or("Source13 output strength")?)?.to_bits()
    );
    let mut count = 0;
    for (name, key) in [
        ("pooled_output", ConditioningMetadataKey::PooledOutput),
        (
            "conditioning_lyrics",
            ConditioningMetadataKey::ConditioningLyrics,
        ),
    ] {
        let tensor = match (key, actual.metadata().get(key)) {
            (_, None) => None,
            (
                ConditioningMetadataKey::PooledOutput,
                Some(ConditioningMetadataValue::PooledOutput(value)),
            )
            | (
                ConditioningMetadataKey::ConditioningLyrics,
                Some(ConditioningMetadataValue::ConditioningLyrics(value)),
            ) => Some(value.as_ref()),
            _ => return Err("Source13 typed metadata kind changed".into()),
        };
        match (metadata.get(name), tensor) {
            (None, None) => {}
            (Some(value), Some(None)) if value.is_null() => {
                count += 1;
            }
            (Some(value), Some(Some(tensor))) => {
                count += 1;
                source13_assert_tensor(
                    tensor,
                    value
                        .get("tensor")
                        .ok_or("Source13 output metadata tensor")?,
                )?;
            }
            _ => return Err("Source13 output typed metadata presence changed".into()),
        }
    }
    assert_eq!(actual.metadata().len(), count);
    Ok(())
}

#[test]
#[ignore = "Requires separately reviewed genuine Source13 capture and exact SHA; compares only declared typed projection, never unsupported dictionary/shape/value behavior"]
fn task618_original_source13_exact_typed_zero_words_and_alias_graphs() -> TestResult {
    let capture = source13_capture()?;
    let (_, _, _, context, _, _) = setup(40)?;
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ConditioningZeroOut").ok_or("ZeroOut")?;
    let mut compared = 0;
    let mut unsupported_shapes = 0;
    let mut untyped_errors = 0;
    for case in capture
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source13 cases")?
    {
        let name = case
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or("Source13 name")?;
        if matches!(name, "scalar" | "empty") {
            let value = case
                .pointer("/input/0/1/conditioning_lyrics/tensor")
                .ok_or("Source13 unsupported lyrics tensor")?;
            let (shape, _, _) = source13_tensor_spec(value)?;
            assert_eq!(shape, if name == "scalar" { vec![] } else { vec![1, 0] });
            let tensor = source13_upload(&context, value)?;
            assert!(
                ConditioningMetadataMap::checked([(
                    ConditioningMetadataKey::ConditioningLyrics,
                    ConditioningMetadataValue::ConditioningLyrics(Some(tensor))
                )])
                .is_err()
            );
            println!(
                "Source13 {name}: Source succeeds; canonical lyrics owner rejects exact captured shape. No node/numerical equivalence."
            );
            unsupported_shapes += 1;
            continue;
        }
        if matches!(name, "list-error" | "integer-error") {
            let value = case
                .pointer("/input/0/1/conditioning_lyrics")
                .ok_or("Source13 untyped value")?;
            assert!(if name == "list-error" {
                value.is_array()
            } else {
                value.as_u64() == Some(7)
            });
            println!(
                "Source13 {name}: Source builtins.TypeError; untyped value has no sealed native representation. No native error parity."
            );
            untyped_errors += 1;
            continue;
        }
        let source = source13_conditioning(&context, case)?;
        let digest = source.digest().to_owned();
        let input_roles = source13_roles(&source)?;
        let input_aliases = case.get("input_aliases").ok_or("Source13 input aliases")?;
        for (row, left) in &input_roles {
            for (column, right) in &input_roles {
                assert_eq!(
                    left.storage_id() == right.storage_id(),
                    source13_alias(
                        input_aliases,
                        "storage_alias_to_input",
                        row,
                        input_aliases,
                        column
                    )?
                );
            }
        }
        let handle = publish(&context, source.clone())?;
        let output = first_output(smol::block_on(
            node.execute(context.clone(), inputs("ConditioningZeroOut", &handle)),
        )?)?;
        let result = resolved(&context, &output)?;
        let columns = case
            .get("output")
            .and_then(serde_json::Value::as_array)
            .ok_or("Source13 output columns")?;
        assert_eq!(columns.len(), 1);
        let expected = columns
            .first()
            .and_then(serde_json::Value::as_array)
            .ok_or("Source13 output column")?;
        assert_eq!(result.entries().len(), expected.len());
        for ((actual, original), expected) in
            result.entries().iter().zip(source.entries()).zip(expected)
        {
            source13_assert_entry(actual, expected)?;
            assert_eq!(actual.identifier(), original.identifier());
            assert_eq!(actual.references(), original.references());
            assert_eq!(
                actual.options().strength.to_bits(),
                original.options().strength.to_bits()
            );
            assert_eq!(
                actual.options().window.start_percent().to_bits(),
                original.options().window.start_percent().to_bits()
            );
            assert_eq!(
                actual.options().window.end_percent().to_bits(),
                original.options().window.end_percent().to_bits()
            );
            assert_eq!(
                actual.options().default_region,
                original.options().default_region
            );
            assert!(actual.options().region.is_none() && original.options().region.is_none());
            assert!(actual.options().mask.is_none() && original.options().mask.is_none());
        }
        let aliases = case
            .get("output_aliases")
            .and_then(serde_json::Value::as_array)
            .ok_or("Source13 output aliases")?;
        assert_eq!(aliases.len(), 1);
        let output_aliases = aliases.first().ok_or("Source13 first output aliases")?;
        let output_roles = source13_roles(&result)?;
        assert_eq!(
            input_roles.iter().map(|(role, _)| role).collect::<Vec<_>>(),
            output_roles
                .iter()
                .map(|(role, _)| role)
                .collect::<Vec<_>>()
        );
        for (row, left) in &output_roles {
            for (column, right) in &output_roles {
                assert_eq!(
                    left.storage_id() == right.storage_id(),
                    source13_alias(
                        output_aliases,
                        "output_storage_alias_matrix",
                        row,
                        output_aliases,
                        column
                    )?
                );
            }
            for (column, right) in &input_roles {
                assert_eq!(
                    left.storage_id() == right.storage_id(),
                    source13_alias(
                        output_aliases,
                        "storage_alias_to_input",
                        row,
                        input_aliases,
                        column
                    )?
                );
            }
        }
        let retained = resolved(&context, &handle)?;
        assert_eq!(retained.digest(), digest);
        let input = case
            .get("input")
            .and_then(serde_json::Value::as_array)
            .ok_or("Source13 input column")?;
        for (actual, expected) in retained.entries().iter().zip(input) {
            source13_assert_entry(actual, expected)?;
        }
        println!(
            "Source13 {name}: exact captured typed words/shape/dtype/presence plus supported input/output alias graph asserted; unrelated lyrics/retained fields not represented."
        );
        compared += 1;
        drop(output_roles);
        drop(input_roles);
        drop(retained);
        drop(result);
        context
            .handle_store()
            .revoke(&output, &context.cancellation)?;
        context
            .handle_store()
            .revoke(&handle, &context.cancellation)?;
    }
    assert_eq!((compared, unsupported_shapes, untyped_errors), (9, 2, 2));
    println!(
        "Source13 typed projection9, exact unsupported shape rejections2, untyped representation gaps2; unsupported dictionary/object identity in all13. No full Source/native/app/learned parity claim."
    );
    Ok(())
}
