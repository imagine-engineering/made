use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativeResolvedPayload, NativeStoredPayload,
    NativeValue, NodeRegistry, built_in_source_schema, native_value_matches_input_schema,
    native_value_type_for_output_schema, native_value_types_for_input_schema,
};
use sim_model::conditioning::{
    ConditioningMetadataKey, ConditioningMetadataValue, ConditioningSet,
};
use sim_tensor::{CancellationToken, Tensor};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["ReferenceLatent", "ReferenceTimbreAudio"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
struct ReferenceLatentNode;
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    Ok(vec![
        binding(
            "COMFY-NODE-0527",
            Arc::new(ReferenceLatentNode),
            vec!["conditioning".to_owned()],
        )?,
        binding(
            "COMFY-NODE-0528",
            Arc::new(ReferenceTimbreAudioNode),
            vec!["conditioning".to_owned()],
        )?,
    ])
}
fn binding(
    feature_id: &str,
    node: Arc<dyn NativeNode>,
    output_names: Vec<String>,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let class_type = node.class_type();
    let catalog = built_in_source_schema(class_type).map_err(schema_error)?;
    if catalog.node.feature_id.as_deref() != Some(feature_id) {
        return Err(schema_error(
            "Conditioning feature identity does not match the source",
        ));
    }
    let registry = NodeRegistry::built_in().map_err(schema_error)?;
    let row = registry
        .descriptor(class_type)
        .ok_or_else(|| schema_error("Missing conditioning source row"))?;
    let names = catalog
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let source_schema = catalog
        .bind_execution_ports(&names, &catalog.dynamic_inputs, &output_names)
        .map_err(schema_error)?;
    let inputs = catalog
        .inputs
        .iter()
        .map(|input| {
            let accepted_types =
                native_value_types_for_input_schema(&input.schema).map_err(schema_error)?;
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                allows_literal: input.schema.source_type_names.iter().all(|name| {
                    matches!(
                        name.as_str(),
                        "INT" | "STRING" | "FLOAT" | "BOOLEAN" | "COMBO"
                    )
                }),
                accepted_types,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: input.requirement == NativeInputRequirement::Hidden,
                lazy: false,
                cardinality: NativePortCardinality::Mapped,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    if !catalog.dynamic_inputs.is_empty() {
        return Err(schema_error(
            "ReferenceLatent has no original dynamic inputs",
        ));
    }
    let outputs = source_schema
        .outputs
        .iter()
        .map(|output| {
            Ok(NativeOutputDescriptor {
                name: output.name.clone(),
                produced_type: native_value_type_for_output_schema(output).map_err(schema_error)?,
                is_list: false,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let result = NativeNodeBinding::Executable {
        feature_id: feature_id.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: node.implementation_version().to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs,
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: row.display_name.clone(),
            category: row.category.clone(),
            description: catalog.presentation.description.clone().unwrap_or_default(),
            output_names,
            search_aliases: Vec::new(),
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node,
    };
    result.validate()?;
    Ok(result)
}

fn mapped_values(value: &NativeValue) -> Result<&[NativeValue], NativeNodeFailure> {
    match value {
        NativeValue::List { values } if values.is_empty() => {
            Err(invalid("Mapped inputs cannot be empty"))
        }
        NativeValue::List { values } => Ok(values),
        value => Ok(std::slice::from_ref(value)),
    }
}
fn validate(inputs: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
    let schema = built_in_source_schema("ReferenceLatent").map_err(invalid)?;
    for input in &schema.inputs {
        let Some(value) = inputs.get(&input.schema.name) else {
            if input.requirement == NativeInputRequirement::Required {
                return Err(invalid(format!(
                    "Missing original input {}",
                    input.schema.name
                )));
            }
            continue;
        };
        value.validate().map_err(invalid)?;
        let accepted = native_value_types_for_input_schema(&input.schema).map_err(invalid)?;
        for value in mapped_values(value)? {
            if !accepted.accepts(value) || !native_value_matches_input_schema(value, &input.schema)
            {
                return Err(invalid(format!(
                    "{} violates its original schema",
                    input.schema.name
                )));
            }
        }
    }
    if inputs
        .keys()
        .any(|name| !schema.inputs.iter().any(|input| &input.schema.name == name))
    {
        return Err(invalid("Unknown original ReferenceLatent input"));
    }
    Ok(())
}
fn resolve(
    context: &NativeNodeContext,
    value: &NativeValue,
    kind: NativeHandleKind,
    source_type: &str,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    let NativeValue::Handle { value } = value else {
        return Err(invalid("Mapped canonical handle required"));
    };
    let expected = NativeHandleType::new(kind, source_type).map_err(invalid)?;
    context
        .handle_store()
        .resolve(value, &expected, &context.cancellation)
        .map_err(|error| failure(context, error))
}
fn append(
    context: &NativeNodeContext,
    input: &ConditioningSet,
    reference: &Tensor,
) -> Result<ConditioningSet, NativeNodeFailure> {
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(input.entries().len())
        .map_err(|error| failure(context, error))?;
    for entry in input.entries() {
        check(context)?;
        let mut values = entry.metadata().clone();
        let mut references = match values.get(ConditioningMetadataKey::ReferenceLatents) {
            Some(ConditioningMetadataValue::TensorList(values)) => values.clone(),
            None => Vec::new(),
            Some(_) => return Err(invalid("Canonical reference_latents list required")),
        };
        references
            .try_reserve_exact(1)
            .map_err(|error| failure(context, error))?;
        references.push(reference.clone());
        values
            .insert(
                ConditioningMetadataKey::ReferenceLatents,
                ConditioningMetadataValue::TensorList(references),
            )
            .map_err(|error| failure(context, error))?;
        metadata.push(values);
    }
    input
        .with_entry_metadata(metadata, &context.cancellation)
        .map_err(|error| failure(context, error))
}
fn outcome(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check(context)?;
    let outcome = NativeNodeOutcome::Values {
        outputs: vec![NativeValue::Handle {
            value: handle.clone(),
        }],
        ui: None,
        effects: Vec::new(),
    };
    outcome.validate().map_err(invalid)?;
    Ok(outcome)
}
impl NativeNode for ReferenceLatentNode {
    fn class_type(&self) -> &str {
        "ReferenceLatent"
    }
    fn implementation_version(&self) -> &str {
        "source-reference-latent-0527-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate(inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, kind, source_type) in [
            (
                "conditioning",
                NativeHandleKind::Conditioning,
                "CONDITIONING",
            ),
            ("latent", NativeHandleKind::Latent, "LATENT"),
        ] {
            if let Some(value) = inputs.get(name) {
                for (index, value) in mapped_values(value)?.iter().enumerate() {
                    check(context)?;
                    let retained = resolve(context, value, kind, source_type)?;
                    let compatible = match source_type {
                        "CONDITIONING" => {
                            matches!(retained.as_ref(), NativeStoredPayload::Conditioning(_))
                        }
                        "LATENT" => matches!(retained.as_ref(), NativeStoredPayload::Latent(_)),
                        _ => false,
                    };
                    if !compatible {
                        return Err(invalid("Canonical source payload required"));
                    }
                    artifact_digests.insert(format!("{name}.{index}"), retained.digest_sha256());
                }
            }
        }
        Ok(NativeCacheDependencies {
            artifact_digests,
            ..Default::default()
        })
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            validate(&inputs)?;
            let conditioning_value = inputs
                .get("conditioning")
                .ok_or_else(|| invalid("Missing conditioning"))?;
            let retained = resolve(
                &context,
                conditioning_value,
                NativeHandleKind::Conditioning,
                "CONDITIONING",
            )?;
            let NativeStoredPayload::Conditioning(conditioning) = retained.as_ref() else {
                return Err(invalid("Canonical conditioning required"));
            };
            let Some(latent_value) = inputs.get("latent") else {
                let NativeValue::Handle { value } = conditioning_value else {
                    return Err(invalid("Canonical conditioning handle required"));
                };
                return outcome(&context, value);
            };
            let latent = resolve(&context, latent_value, NativeHandleKind::Latent, "LATENT")?;
            let NativeStoredPayload::Latent(latent) = latent.as_ref() else {
                return Err(invalid("Canonical latent bundle required"));
            };
            let Some(samples) = latent.samples().tensor() else {
                return Err(NativeNodeFailure {
                    code: "source_reference_latent_samples_unbound".to_owned(),
                    message: "Original nested samples cannot be represented by canonical reference_latents TensorList".to_owned(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                });
            };
            let output = append(&context, conditioning, samples)?;
            check(&context)?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Conditioning(Arc::new(output)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            match outcome(&context, &handle) {
                Ok(outcome) => Ok(outcome),
                Err(mut original) => {
                    if let Err(error) = context
                        .handle_store()
                        .revoke(&handle, &CancellationToken::default())
                    {
                        original
                            .message
                            .push_str(&format!("; canonical rollback failed: {error}"));
                    }
                    Err(original)
                }
            }
        })
    }
}
fn schema_error(error: impl std::fmt::Display) -> NativeNodeContractError {
    NativeNodeContractError::InvalidSourceSchema(error.to_string())
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_reference_latent_inputs".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: if context.cancellation.is_cancelled() {
            "execution_interrupted"
        } else {
            "conditioning_execution_failed"
        }
        .to_owned(),
        message: error.to_string(),
        kind: if context.cancellation.is_cancelled() {
            NativeNodeFailureKind::Interrupted
        } else {
            NativeNodeFailureKind::Failure
        },
        retryable: context.cancellation.is_cancelled(),
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}

struct ReferenceTimbreAudioNode;
fn validate_reference_audio(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    let schema = built_in_source_schema("ReferenceTimbreAudio").map_err(invalid)?;
    for input in &schema.inputs {
        let Some(value) = inputs.get(&input.schema.name) else {
            if input.requirement == NativeInputRequirement::Required {
                return Err(invalid_reference_audio(format!(
                    "Missing original input {}",
                    input.schema.name
                )));
            }
            continue;
        };
        value.validate().map_err(invalid)?;
        let accepted = native_value_types_for_input_schema(&input.schema).map_err(invalid)?;
        for value in mapped_values(value)? {
            if !accepted.accepts(value) || !native_value_matches_input_schema(value, &input.schema)
            {
                return Err(invalid_reference_audio(format!(
                    "{} violates its original schema",
                    input.schema.name
                )));
            }
        }
    }
    if inputs
        .keys()
        .any(|name| !schema.inputs.iter().any(|input| &input.schema.name == name))
    {
        return Err(invalid_reference_audio(
            "Unknown original ReferenceTimbreAudio input",
        ));
    }
    Ok(())
}
fn append_reference_audio(
    context: &NativeNodeContext,
    input: &ConditioningSet,
    reference: &Tensor,
) -> Result<ConditioningSet, NativeNodeFailure> {
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(input.entries().len())
        .map_err(|error| failure(context, error))?;
    for entry in input.entries() {
        check(context)?;
        let mut values = entry.metadata().clone();
        let mut references = match values.get(ConditioningMetadataKey::ReferenceAudioTimbreLatents)
        {
            Some(ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(values))) => {
                let mut retained = Vec::new();
                retained
                    .try_reserve_exact(values.len().saturating_add(1))
                    .map_err(|error| failure(context, error))?;
                retained.extend(values.iter().cloned());
                retained
            }
            None | Some(ConditioningMetadataValue::ReferenceAudioTimbreLatents(None)) => Vec::new(),
            Some(_) => {
                return Err(invalid_reference_audio(
                    "Canonical reference_audio_timbre_latents list required",
                ));
            }
        };
        references
            .try_reserve_exact(1)
            .map_err(|error| failure(context, error))?;
        references.push(reference.clone());
        values
            .insert(
                ConditioningMetadataKey::ReferenceAudioTimbreLatents,
                ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(references)),
            )
            .map_err(|error| failure(context, error))?;
        metadata.push(values);
    }
    input
        .with_entry_metadata(metadata, &context.cancellation)
        .map_err(|error| failure(context, error))
}
impl NativeNode for ReferenceTimbreAudioNode {
    fn class_type(&self) -> &str {
        "ReferenceTimbreAudio"
    }
    fn implementation_version(&self) -> &str {
        "source-reference-audio-timbre-0528-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate_reference_audio(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_reference_audio(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate_reference_audio(inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, kind, source_type) in [
            (
                "conditioning",
                NativeHandleKind::Conditioning,
                "CONDITIONING",
            ),
            ("latent", NativeHandleKind::Latent, "LATENT"),
        ] {
            if let Some(value) = inputs.get(name) {
                for (index, value) in mapped_values(value)?.iter().enumerate() {
                    check(context)?;
                    let retained = resolve(context, value, kind, source_type)?;
                    let compatible = match source_type {
                        "CONDITIONING" => {
                            matches!(retained.as_ref(), NativeStoredPayload::Conditioning(_))
                        }
                        "LATENT" => matches!(retained.as_ref(), NativeStoredPayload::Latent(_)),
                        _ => false,
                    };
                    if !compatible {
                        return Err(invalid_reference_audio("Canonical source payload required"));
                    }
                    artifact_digests.insert(format!("{name}.{index}"), retained.digest_sha256());
                }
            }
        }
        Ok(NativeCacheDependencies {
            artifact_digests,
            ..Default::default()
        })
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            validate_reference_audio(&inputs)?;
            let conditioning_value = inputs
                .get("conditioning")
                .ok_or_else(|| invalid_reference_audio("Missing conditioning"))?;
            let retained = resolve(
                &context,
                conditioning_value,
                NativeHandleKind::Conditioning,
                "CONDITIONING",
            )?;
            let NativeStoredPayload::Conditioning(conditioning) = retained.as_ref() else {
                return Err(invalid_reference_audio("Canonical conditioning required"));
            };
            let Some(latent_value) = inputs.get("latent") else {
                let NativeValue::Handle { value } = conditioning_value else {
                    return Err(invalid_reference_audio(
                        "Canonical conditioning handle required",
                    ));
                };
                return outcome(&context, value);
            };
            let latent = resolve(&context, latent_value, NativeHandleKind::Latent, "LATENT")?;
            let NativeStoredPayload::Latent(latent) = latent.as_ref() else {
                return Err(invalid_reference_audio("Canonical latent bundle required"));
            };
            let Some(samples) = latent.samples().tensor() else {
                return Err(NativeNodeFailure {
                    code: "source_reference_audio_timbre_samples_unbound".to_owned(),
                    message: "Original nested samples cannot be represented by canonical reference_audio_timbre_latents tensor list".to_owned(),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: false,
                });
            };
            let output = append_reference_audio(&context, conditioning, samples)?;
            check(&context)?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Conditioning(Arc::new(output)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            match outcome(&context, &handle) {
                Ok(outcome) => Ok(outcome),
                Err(mut original) => {
                    if let Err(error) = context
                        .handle_store()
                        .revoke(&handle, &CancellationToken::default())
                    {
                        original
                            .message
                            .push_str(&format!("; canonical rollback failed: {error}"));
                    }
                    Err(original)
                }
            }
        })
    }
}
fn invalid_reference_audio(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_reference_audio_timbre_inputs".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
