use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeHandleKind, NativeHandleType,
    NativeInputDescriptor, NativeInputRequirement, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativeResolvedPayload, NativeStoredPayload, NativeValue, NodeRegistry, built_in_source_schema,
    native_value_matches_input_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};
use sim_model::conditioning::{
    ConditioningEntry, ConditioningMask, ConditioningMetadataKey, ConditioningMetadataValue,
    ConditioningRegion, ConditioningSet, ConditioningTimestepRange, ConditioningValue,
};
use sim_tensor::generated_elementwise_or_runtime_operation_22::zeros_like_with_context_exact_native;
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "ConditioningSetArea",
    "ConditioningSetAreaPercentage",
    "ConditioningSetAreaPercentageVideo",
    "ConditioningSetAreaStrength",
    "ConditioningSetTimestepRange",
    "ConditioningStableAudio",
    "ConditioningZeroOut",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
struct ConditioningMetadataNode {
    class_type: &'static str,
}
fn conditioning_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Conditioning, "CONDITIONING")
}
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        ("ConditioningSetArea", "COMFY-NODE-0092"),
        ("ConditioningSetAreaPercentage", "COMFY-NODE-0093"),
        ("ConditioningSetAreaPercentageVideo", "COMFY-NODE-0094"),
        ("ConditioningSetAreaStrength", "COMFY-NODE-0095"),
        ("ConditioningSetTimestepRange", "COMFY-NODE-0100"),
        ("ConditioningStableAudio", "COMFY-NODE-0101"),
        ("ConditioningZeroOut", "COMFY-NODE-0103"),
    ]
    .into_iter()
    .map(|(class_type, feature)| {
        let outputs = if class_type == "ConditioningStableAudio" {
            vec!["positive".to_owned(), "negative".to_owned()]
        } else {
            vec!["conditioning".to_owned()]
        };
        binding(
            feature,
            Arc::new(ConditioningMetadataNode { class_type }),
            outputs,
        )
    })
    .collect()
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
    let dynamic_inputs = catalog
        .dynamic_inputs
        .iter()
        .map(|dynamic| {
            Ok(NativeDynamicInputDescriptor {
                name_template: dynamic.identity.clone(),
                start_index: dynamic.start_index,
                minimum_count: dynamic.minimum_count,
                maximum_count: dynamic.maximum_count,
                input: NativeInputDescriptor {
                    name: dynamic.input.name.clone(),
                    accepted_types: native_value_types_for_input_schema(&dynamic.input)
                        .map_err(schema_error)?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Mapped,
                    allows_literal: false,
                },
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
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
            dynamic_inputs,
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
            search_aliases: if class_type == "ConditioningSetArea" {
                [
                    "regional prompt",
                    "area prompt",
                    "spatial conditioning",
                    "localized prompt",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect()
            } else if class_type == "ConditioningZeroOut" {
                ["null conditioning", "clear conditioning"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect()
            } else {
                Vec::new()
            },
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node,
    };
    result.validate()?;
    Ok(result)
}

fn validate(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    let catalog = built_in_source_schema(class_type).map_err(invalid)?;
    for name in inputs.keys() {
        if !catalog
            .inputs
            .iter()
            .any(|input| &input.schema.name == name)
        {
            return Err(invalid(format!("Unknown conditioning input {name}")));
        }
    }
    for input in &catalog.inputs {
        let value = inputs
            .get(&input.schema.name)
            .ok_or_else(|| invalid(format!("Missing source input {}", input.schema.name)))?;
        value.validate().map_err(invalid)?;
        let values = match value {
            NativeValue::List { values } if values.is_empty() => {
                return Err(invalid("Mapped conditioning inputs cannot be empty"));
            }
            NativeValue::List { values } => values.as_slice(),
            value => std::slice::from_ref(value),
        };
        let accepted = native_value_types_for_input_schema(&input.schema).map_err(invalid)?;
        for value in values {
            if !accepted.accepts(value) || !native_value_matches_input_schema(value, &input.schema)
            {
                return Err(invalid(format!(
                    "{} violates its source schema",
                    input.schema.name
                )));
            }
        }
    }
    if class_type == "ConditioningSetArea" {
        // The catalog retains MAX_RESOLUTION as provenance rather than an executable bound.
        for name in ["width", "height", "x", "y"] {
            let value = inputs
                .get(name)
                .ok_or_else(|| invalid(format!("Missing source input {name}")))?;
            let values = match value {
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for value in values {
                let exceeds_source_maximum = match value {
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(value),
                    } => *value > 16_384,
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(value),
                    } => *value > 16_384,
                    _ => false,
                };
                if exceeds_source_maximum {
                    return Err(invalid(format!(
                        "{name} exceeds the source MAX_RESOLUTION of 16384"
                    )));
                }
            }
        }
    }

    Ok(())
}
fn resolve(
    context: &NativeNodeContext,
    value: &NativeValue,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    let NativeValue::Handle { value } = value else {
        return Err(invalid("Mapped conditioning handle required"));
    };
    context
        .handle_store()
        .resolve(
            value,
            &conditioning_type().map_err(invalid)?,
            &context.cancellation,
        )
        .map_err(|error| failure(context, error))
}
fn number(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<f64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value as f64),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Ok(*value as f64),
        _ => Err(invalid(format!("Mapped numeric input {name} required"))),
    }
}
fn unsigned(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).map_err(invalid),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Ok(*value),
        _ => Err(invalid(format!(
            "Mapped nonnegative integer input {name} required"
        ))),
    }
}
fn source_strength(inputs: &BTreeMap<String, NativeValue>) -> Result<f64, NativeNodeFailure> {
    number(inputs, "strength")
}
fn region(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<ConditioningRegion, NativeNodeFailure> {
    if class_type == "ConditioningSetArea" {
        return ConditioningRegion::absolute(
            vec![
                unsigned(inputs, "height")? / 8,
                unsigned(inputs, "width")? / 8,
            ],
            vec![unsigned(inputs, "y")? / 8, unsigned(inputs, "x")? / 8],
        )
        .map_err(unsupported);
    }
    let (sizes, offsets) = if class_type == "ConditioningSetAreaPercentageVideo" {
        (
            vec![
                number(inputs, "temporal")?,
                number(inputs, "height")?,
                number(inputs, "width")?,
            ],
            vec![
                number(inputs, "z")?,
                number(inputs, "y")?,
                number(inputs, "x")?,
            ],
        )
    } else {
        (
            vec![number(inputs, "height")?, number(inputs, "width")?],
            vec![number(inputs, "y")?, number(inputs, "x")?],
        )
    };
    ConditioningRegion::percentage(sizes, offsets).map_err(unsupported)
}
fn transform(
    context: &NativeNodeContext,
    class_type: &str,
    source: &ConditioningSet,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<ConditioningSet, NativeNodeFailure> {
    check(context)?;
    if class_type == "ConditioningZeroOut" {
        return zero_out(context, source);
    }
    let strength = if matches!(
        class_type,
        "ConditioningSetArea"
            | "ConditioningSetAreaPercentage"
            | "ConditioningSetAreaPercentageVideo"
            | "ConditioningSetAreaStrength"
    ) {
        Some(source_strength(inputs)?)
    } else {
        None
    };
    let area = if matches!(
        class_type,
        "ConditioningSetArea"
            | "ConditioningSetAreaPercentage"
            | "ConditioningSetAreaPercentageVideo"
    ) {
        Some(region(class_type, inputs)?)
    } else {
        None
    };
    let range = if class_type == "ConditioningSetTimestepRange" {
        Some(
            ConditioningTimestepRange::source(number(inputs, "start")?, number(inputs, "end")?)
                .map_err(invalid)?,
        )
    } else {
        None
    };
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(source.entries().len())
        .map_err(invalid)?;
    for entry in source.entries() {
        check(context)?;
        let mut options = entry.options().clone();
        let mut metadata = entry.metadata().clone();
        if let Some(strength) = strength {
            options.strength = strength;
        }
        if let Some(area) = &area {
            options.region = Some(area.clone());
            options.default_region = false;
            if let Some(mask) = &options.mask {
                options.mask = Some(
                    ConditioningMask::new(
                        mask.tensor().clone(),
                        mask.strength(),
                        mask.feather().to_vec(),
                        false,
                    )
                    .map_err(invalid)?,
                );
            }
            metadata
                .insert(
                    ConditioningMetadataKey::Region,
                    ConditioningMetadataValue::Region(area.clone()),
                )
                .map_err(unsupported)?;
            metadata
                .insert(
                    ConditioningMetadataKey::SetAreaToBounds,
                    ConditioningMetadataValue::Boolean(false),
                )
                .map_err(unsupported)?;
        }
        if let Some(range) = range {
            metadata
                .insert(
                    ConditioningMetadataKey::TimestepRange,
                    ConditioningMetadataValue::TimestepRange(range),
                )
                .map_err(unsupported)?;
        }
        if class_type == "ConditioningStableAudio" {
            // The map validates paired timing values, so replace them together without transiently validating one old endpoint.
            let mut values = metadata
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key,
                        ConditioningMetadataKey::SecondsStart
                            | ConditioningMetadataKey::SecondsTotal
                    )
                })
                .map(|(key, value)| (*key, value.clone()))
                .collect::<Vec<_>>();
            values.push((
                ConditioningMetadataKey::SecondsStart,
                ConditioningMetadataValue::FiniteF64(number(inputs, "seconds_start")?),
            ));
            values.push((
                ConditioningMetadataKey::SecondsTotal,
                ConditioningMetadataValue::FiniteF64(number(inputs, "seconds_total")?),
            ));
            metadata = sim_model::conditioning::ConditioningMetadataMap::checked(values)
                .map_err(unsupported)?;
        }
        entries.push(
            ConditioningEntry::checked_with_metadata(
                entry.identifier().to_owned(),
                entry.value().clone(),
                options,
                entry.references().clone(),
                metadata,
            )
            .map_err(|error| failure(context, error))?,
        );
    }
    ConditioningSet::checked(source.identity().clone(), entries, &context.cancellation)
        .map_err(|error| failure(context, error))
}
fn zero_out(
    context: &NativeNodeContext,
    source: &ConditioningSet,
) -> Result<ConditioningSet, NativeNodeFailure> {
    if source.entries().is_empty() {
        return ConditioningSet::checked(
            source.identity().clone(),
            Vec::new(),
            &context.cancellation,
        )
        .map_err(|error| failure(context, error));
    }
    let compute = context
        .compute_session()
        .map_err(|error| failure(context, error))?;
    let execution = compute
        .execution_context(context)
        .map_err(|error| failure(context, error))?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(source.entries().len())
        .map_err(invalid)?;
    for entry in source.entries() {
        check(context)?;
        let mut metadata = entry.metadata().clone();
        if let Some(ConditioningMetadataValue::PooledOutput(Some(pooled))) =
            metadata.get(ConditioningMetadataKey::PooledOutput)
        {
            let pooled = zeros_like_with_context_exact_native(
                compute.backend(),
                pooled,
                None,
                None,
                None,
                &execution,
            )
            .map_err(|error| failure(context, error))?;
            metadata
                .insert(
                    ConditioningMetadataKey::PooledOutput,
                    ConditioningMetadataValue::PooledOutput(Some(pooled)),
                )
                .map_err(|error| failure(context, error))?;
        }
        if let Some(ConditioningMetadataValue::ConditioningLyrics(Some(lyrics))) =
            metadata.get(ConditioningMetadataKey::ConditioningLyrics)
        {
            let lyrics = zeros_like_with_context_exact_native(
                compute.backend(),
                lyrics,
                None,
                None,
                None,
                &execution,
            )
            .map_err(|error| failure(context, error))?;
            metadata
                .insert(
                    ConditioningMetadataKey::ConditioningLyrics,
                    ConditioningMetadataValue::ConditioningLyrics(Some(lyrics)),
                )
                .map_err(|error| failure(context, error))?;
        }
        let zero = |tensor: &sim_tensor::Tensor| {
            zeros_like_with_context_exact_native(
                compute.backend(),
                tensor,
                None,
                None,
                None,
                &execution,
            )
            .map_err(|error| failure(context, error))
        };
        let value = match entry.value() {
            ConditioningValue::Regular(tensor) => ConditioningValue::regular(zero(tensor)?),
            ConditioningValue::CrossAttention(tensor) => ConditioningValue::cross_attention(zero(tensor)?),
            ConditioningValue::NoiseShape(tensor) => ConditioningValue::noise_shape(zero(tensor)?),
            _ => return Err(unsupported("Source ZeroOut requires a tensor primary value; native constant/list values are not that representation")),
        }.map_err(|error| failure(context, error))?;
        entries.push(
            ConditioningEntry::checked_with_metadata(
                entry.identifier().to_owned(),
                value,
                entry.options().clone(),
                entry.references().clone(),
                metadata,
            )
            .map_err(|error| failure(context, error))?,
        );
    }
    ConditioningSet::checked(source.identity().clone(), entries, &context.cancellation)
        .map_err(|error| failure(context, error))
}

fn publish(
    context: &NativeNodeContext,
    outputs: Vec<ConditioningSet>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let mut handles: Vec<NativeOpaqueHandle> = Vec::new();
    handles.try_reserve_exact(outputs.len()).map_err(invalid)?;
    let result = (|| {
        for output in outputs {
            check(context)?;
            handles.push(
                context
                    .handle_store()
                    .publish(
                        NativeStoredPayload::Conditioning(Arc::new(output)),
                        &context.cancellation,
                    )
                    .map_err(|error| failure(context, error))?,
            );
        }
        check(context)?;
        let outcome = NativeNodeOutcome::Values {
            outputs: handles
                .iter()
                .cloned()
                .map(|value| NativeValue::Handle { value })
                .collect(),
            ui: None,
            effects: Vec::new(),
        };
        outcome.validate().map_err(invalid)?;
        Ok(outcome)
    })();
    if let Err(original) = result {
        let mut errors = Vec::new();
        for handle in handles.iter().rev() {
            if let Err(error) = context
                .handle_store()
                .revoke(handle, &sim_types::CancellationToken::default())
            {
                errors.push(error.to_string());
            }
        }
        if errors.is_empty() {
            return Err(original);
        }
        return Err(failure(
            context,
            format!("{}; rollback: {}", original.message, errors.join("; ")),
        ));
    }
    result
}
impl NativeNode for ConditioningMetadataNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        match self.class_type {
            "ConditioningZeroOut" => "source-conditioning-part-three-zero-v1",
            "ConditioningSetArea"
            | "ConditioningSetAreaPercentage"
            | "ConditioningSetAreaPercentageVideo"
            | "ConditioningSetAreaStrength" => "source-conditioning-part-three-strength-f64-v3",
            _ => "source-conditioning-part-three-metadata-v2",
        }
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type, inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate(self.class_type, inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type, inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for name in ["conditioning", "positive", "negative"] {
            if let Some(value) = inputs.get(name) {
                let values = match value {
                    NativeValue::List { values } => values.as_slice(),
                    value => std::slice::from_ref(value),
                };
                for (index, value) in values.iter().enumerate() {
                    artifact_digests.insert(
                        format!("{name}.{index}"),
                        resolve(context, value)?.digest_sha256(),
                    );
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
            validate(self.class_type, &inputs)?;
            let names: &[&str] = if self.class_type == "ConditioningStableAudio" {
                &["positive", "negative"]
            } else {
                &["conditioning"]
            };
            let mut outputs = Vec::new();
            outputs.try_reserve_exact(names.len()).map_err(invalid)?;
            for name in names {
                let payload = resolve(
                    &context,
                    inputs
                        .get(*name)
                        .ok_or_else(|| invalid("Missing conditioning input"))?,
                )?;
                let NativeStoredPayload::Conditioning(source) = payload.as_ref() else {
                    return Err(invalid("Canonical conditioning set required"));
                };
                outputs.push(transform(&context, self.class_type, source, &inputs)?);
            }
            publish(&context, outputs)
        })
    }
}
fn schema_error(error: impl std::fmt::Display) -> NativeNodeContractError {
    NativeNodeContractError::InvalidSourceSchema(error.to_string())
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    error_failure("invalid_conditioning_inputs", error)
}
fn unsupported(error: impl std::fmt::Display) -> NativeNodeFailure {
    error_failure("unsupported_conditioning_representation", error)
}
fn error_failure(code: &str, error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: code.to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message: error.to_string(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        error_failure("conditioning_execution_failed", error)
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}
