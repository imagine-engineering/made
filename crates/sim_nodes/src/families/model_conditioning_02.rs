use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeHandleKind, NativeHandleType,
    NativeInputDescriptor, NativeInputRequirement, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativeResolvedPayload,
    NativeStoredPayload, NativeValue, NodeRegistry, built_in_source_schema,
    native_value_matches_input_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};
use sim_model::clip_vision::{ClipVisionIntermediate, ClipVisionModelType};
use sim_model::conditioning::{
    ConditioningEntry, ConditioningMetadataKey, ConditioningMetadataMap, ConditioningMetadataValue,
    ConditioningSet, ConditioningValue,
};
use sim_tensor::generated_shape_layout_transform_02::torch_cat_with_context_exact_native;
use sim_tensor::{DType, DeviceId};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "ConditioningCombine",
    "ConditioningConcat",
    "CLIPVisionEncode",
    "ConditioningMultiply",
    "ConditioningAverage",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
struct ConditioningTransformNode {
    class_type: &'static str,
}
fn conditioning_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Conditioning, "CONDITIONING")
}
fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        ("ConditioningCombine", "COMFY-NODE-0089"),
        ("ConditioningConcat", "COMFY-NODE-0090"),
    ]
    .into_iter()
    .map(|(class_type, feature_id)| {
        binding(
            feature_id,
            Arc::new(ConditioningTransformNode { class_type }),
            vec!["conditioning".to_owned()],
        )
    })
    .collect::<Result<Vec<_>, _>>()
    .and_then(|mut bindings| {
        bindings.push(binding(
            "COMFY-NODE-0075",
            Arc::new(ClipVisionEncodeNode),
            vec!["clip_vision_output".to_owned()],
        )?);
        bindings.push(binding(
            "COMFY-NODE-0091",
            Arc::new(ConditioningMultiplyNode),
            vec!["conditioning".to_owned()],
        )?);
        bindings.push(binding(
            "COMFY-NODE-0088",
            Arc::new(ConditioningAverageNode),
            vec!["conditioning".to_owned()],
        )?);
        Ok(bindings)
    })
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
            search_aliases: if class_type == "ConditioningCombine" {
                [
                    "combine",
                    "merge conditioning",
                    "combine prompts",
                    "merge prompts",
                    "mix prompts",
                    "add prompt",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect()
            } else if class_type == "ConditioningMultiply" {
                [
                    "scale conditioning",
                    "scale prompt",
                    "multiply conditioning",
                    "multiply prompt",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect()
            } else if class_type == "ConditioningAverage" {
                [
                    "blend prompts",
                    "interpolate conditioning",
                    "mix prompts",
                    "style fusion",
                    "weighted blend",
                ]
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
            if class_type == "ConditioningMultiply" && input.schema.name == "multiplier" {
                let scalar = match value {
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(value),
                    } => *value,
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(value),
                    } => *value as f64,
                    _ => return Err(invalid("Mapped source multiplier required")),
                };
                // The catalog retains the unary negative literal as provenance rather than an executable bound.
                if !(-100.0..=100.0).contains(&scalar) {
                    return Err(invalid("Multiplier must be between -100 and 100"));
                }
            }
            if !accepted.accepts(value) || !native_value_matches_input_schema(value, &input.schema)
            {
                return Err(invalid(format!(
                    "{} violates its source schema",
                    input.schema.name
                )));
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
fn entry_with_value(
    entry: &ConditioningEntry,
    identifier: String,
    value: ConditioningValue,
) -> Result<ConditioningEntry, NativeNodeFailure> {
    ConditioningEntry::checked_with_metadata(
        identifier,
        value,
        entry.options().clone(),
        entry.references().clone(),
        entry.metadata().clone(),
    )
    .map_err(invalid)
}
fn combine(
    context: &NativeNodeContext,
    first: &ConditioningSet,
    second: &ConditioningSet,
) -> Result<ConditioningSet, NativeNodeFailure> {
    let count = first
        .entries()
        .len()
        .checked_add(second.entries().len())
        .ok_or_else(|| invalid("Combined conditioning entry count overflow"))?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(count).map_err(invalid)?;
    for (index, entry) in first.entries().iter().chain(second.entries()).enumerate() {
        check(context)?;
        // Source lists have no entry identifiers; assign internal ordinals without mutating either input.
        entries.push(entry_with_value(
            entry,
            format!("combined.{index}"),
            entry.value().clone(),
        )?);
    }
    ConditioningSet::checked(first.identity().clone(), entries, &context.cancellation)
        .map_err(|error| failure(context, error))
}
fn concat(
    context: &NativeNodeContext,
    target: &ConditioningSet,
    source: &ConditioningSet,
) -> Result<ConditioningSet, NativeNodeFailure> {
    let first = source.entries().first().ok_or_else(|| {
        unsupported(
            "An empty Source conditioning list is not representable by the checked set owner",
        )
    })?;
    let source_tensor = match first.value() {
        ConditioningValue::Regular(tensor) | ConditioningValue::CrossAttention(tensor) => tensor,
        _ => {
            return Err(unsupported(
                "Source concat requires tensor-backed conditioning",
            ));
        }
    };
    let compute = context.compute_session().map_err(invalid)?;
    let execution = compute.execution_context(context).map_err(invalid)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(target.entries().len())
        .map_err(invalid)?;
    for entry in target.entries() {
        check(context)?;
        let (tensor, cross_attention) = match entry.value() {
            ConditioningValue::Regular(tensor) => (tensor, false),
            ConditioningValue::CrossAttention(tensor) => (tensor, true),
            _ => {
                return Err(unsupported(
                    "Source concat requires tensor-backed conditioning",
                ));
            }
        };
        let value = torch_cat_with_context_exact_native(
            compute.backend(),
            &[tensor.clone(), source_tensor.clone()],
            1,
            &execution,
        )
        .map_err(|error| failure(context, error))?;
        let value = if cross_attention {
            ConditioningValue::cross_attention(value)
        } else {
            ConditioningValue::regular(value)
        }
        .map_err(invalid)?;
        entries.push(entry_with_value(
            entry,
            entry.identifier().to_owned(),
            value,
        )?);
    }
    ConditioningSet::checked(target.identity().clone(), entries, &context.cancellation)
        .map_err(|error| failure(context, error))
}
impl NativeNode for ConditioningTransformNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-conditioning-part-two-list-transforms-v1"
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
        for (name, value) in inputs {
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
            let (first_name, second_name) = if self.class_type == "ConditioningCombine" {
                ("conditioning_1", "conditioning_2")
            } else {
                ("conditioning_to", "conditioning_from")
            };
            let first = resolve(
                &context,
                inputs
                    .get(first_name)
                    .ok_or_else(|| invalid("Missing target conditioning"))?,
            )?;
            let second = resolve(
                &context,
                inputs
                    .get(second_name)
                    .ok_or_else(|| invalid("Missing source conditioning"))?,
            )?;
            let NativeStoredPayload::Conditioning(first) = first.as_ref() else {
                return Err(invalid("Canonical conditioning set required"));
            };
            let NativeStoredPayload::Conditioning(second) = second.as_ref() else {
                return Err(invalid("Canonical conditioning set required"));
            };
            if first.identity() != second.identity() {
                return Err(unsupported(
                    "Different conditioning identities cannot be represented in one checked set",
                ));
            }
            let output = if self.class_type == "ConditioningCombine" {
                combine(&context, first, second)?
            } else {
                concat(&context, first, second)?
            };
            publish(
                &context,
                NativeStoredPayload::Conditioning(Arc::new(output)),
            )
        })
    }
}
fn publish(
    context: &NativeNodeContext,
    payload: NativeStoredPayload,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check(context)?;
    let handle = context
        .handle_store()
        .publish(payload, &context.cancellation)
        .map_err(|error| failure(context, error))?;
    let result = check(context).and_then(|()| {
        let outcome = NativeNodeOutcome::Values {
            outputs: vec![NativeValue::Handle {
                value: handle.clone(),
            }],
            ui: None,
            effects: Vec::new(),
        };
        outcome.validate().map_err(invalid)?;
        Ok(outcome)
    });
    if result.is_err() {
        context
            .handle_store()
            .revoke(&handle, &sim_types::CancellationToken::default())
            .map_err(|error| failure(context, error))?;
    }
    result
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

struct ClipVisionEncodeNode;
fn vision_resolve(
    context: &NativeNodeContext,
    value: &NativeValue,
    name: &str,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    let NativeValue::Handle { value } = value else {
        return Err(invalid("Mapped vision handle required"));
    };
    let expected = match name {
        "clip_vision" => NativeHandleType::new(NativeHandleKind::Clip, "CLIP_VISION"),
        "image" => NativeHandleType::new(NativeHandleKind::Image, "IMAGE"),
        _ => return Err(invalid("Unknown vision handle port")),
    }
    .map_err(invalid)?;
    context
        .handle_store()
        .resolve(value, &expected, &context.cancellation)
        .map_err(|error| failure(context, error))
}
impl NativeNode for ClipVisionEncodeNode {
    fn class_type(&self) -> &str {
        "CLIPVisionEncode"
    }
    fn implementation_version(&self) -> &str {
        "source-conditioning-part-two-clip-vision-cpu-f32-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type(), inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate(self.class_type(), inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type(), inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for name in ["clip_vision", "image"] {
            let value = inputs
                .get(name)
                .ok_or_else(|| invalid("Missing vision handle"))?;
            let values = match value {
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for (index, value) in values.iter().enumerate() {
                artifact_digests.insert(
                    format!("{name}.{index}"),
                    vision_resolve(context, value, name)?.digest_sha256(),
                );
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
            validate(self.class_type(), &inputs)?;
            let Some(NativeValue::Primitive {
                value: NativePrimitive::String(crop),
            }) = inputs.get("crop")
            else {
                return Err(invalid("Mapped crop string required"));
            };
            let vision = vision_resolve(
                &context,
                inputs
                    .get("clip_vision")
                    .ok_or_else(|| invalid("Missing vision model"))?,
                "clip_vision",
            )?;
            let image = vision_resolve(
                &context,
                inputs
                    .get("image")
                    .ok_or_else(|| invalid("Missing image"))?,
                "image",
            )?;
            let NativeStoredPayload::Model(model) = vision.as_ref() else {
                return Err(invalid("Canonical vision model required"));
            };
            let resource = model
                .model_payload()
                .clip_vision_resource()
                .ok_or_else(|| unsupported("Canonical vision backing unavailable"))?;
            if resource.configuration().model_type != ClipVisionModelType::Clip
                || resource.dtype() != DType::F32
                || resource.device() != DeviceId::CPU
            {
                return Err(unsupported(
                    "CLIPVisionEncode currently supports the canonical CPU/F32 CLIP profile; other source output/dtype profiles are not reconciled",
                ));
            }
            let NativeStoredPayload::Tensor(image) = image.as_ref() else {
                return Err(invalid("Canonical image tensor required"));
            };
            if image.image().is_none() {
                return Err(invalid("Canonical IMAGE storage required"));
            }
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let pixel_values = resource
                .preprocess(
                    compute.backend(),
                    image.tensor(),
                    crop == "center",
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
            let mut session = resource
                .execution_session(&context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let output = session
                .forward(
                    compute.backend(),
                    &pixel_values,
                    ClipVisionIntermediate::Layer(-2),
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
            publish(
                &context,
                NativeStoredPayload::ClipVisionOutput(Arc::new(output)),
            )
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum ConditioningMultiplyMetadataError {
    #[error(transparent)]
    Tensor(#[from] sim_tensor::TensorError),
    #[error(transparent)]
    Metadata(#[from] sim_model::conditioning::ConditioningError),
}

fn scaled_multiply_metadata(
    input: &ConditioningMetadataMap,
    multiplier: f64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ConditioningMetadataMap, ConditioningMultiplyMetadataError> {
    execution.check()?;
    let mut metadata = input.clone();
    if let Some(ConditioningMetadataValue::PooledOutput(Some(tensor))) =
        input.get(ConditioningMetadataKey::PooledOutput)
    {
        let descriptor = tensor.descriptor();
        let output_descriptor = sim_tensor::TensorDescriptor::contiguous(
            descriptor.shape().to_vec(),
            descriptor.dtype(),
            descriptor.device(),
            descriptor.stream(),
        )?;
        let pooled = sim_tensor::TensorBackend::binary_scalar(
            backend,
            sim_tensor::BinaryOperation::Multiply,
            tensor,
            sim_tensor::Scalar::Float(multiplier),
            sim_tensor::ScalarSide::Right,
            output_descriptor,
            execution,
        )?
        .0;
        metadata.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(pooled)),
        )?;
    }
    execution.check()?;
    Ok(metadata)
}

struct ConditioningMultiplyNode;
fn multiplier(inputs: &BTreeMap<String, NativeValue>) -> Result<f64, NativeNodeFailure> {
    match inputs.get("multiplier") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value as f64),
        _ => Err(invalid("Mapped source multiplier required")),
    }
}
impl NativeNode for ConditioningMultiplyNode {
    fn class_type(&self) -> &str {
        "ConditioningMultiply"
    }
    fn implementation_version(&self) -> &str {
        "source-conditioning-part-two-pooled-multiply-v2"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type(), inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate(self.class_type(), inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type(), inputs)?;
        let value = inputs
            .get("conditioning")
            .ok_or_else(|| invalid("Missing conditioning"))?;
        let values = match value {
            NativeValue::List { values } => values.as_slice(),
            value => std::slice::from_ref(value),
        };
        let mut artifact_digests = BTreeMap::new();
        for (index, value) in values.iter().enumerate() {
            artifact_digests.insert(
                format!("conditioning.{index}"),
                resolve(context, value)?.digest_sha256(),
            );
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
            validate(self.class_type(), &inputs)?;
            let multiplier = multiplier(&inputs)?;
            let payload = resolve(
                &context,
                inputs
                    .get("conditioning")
                    .ok_or_else(|| invalid("Missing conditioning"))?,
            )?;
            let NativeStoredPayload::Conditioning(input) = payload.as_ref() else {
                return Err(invalid("Canonical conditioning set required"));
            };
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(input.entries().len())
                .map_err(invalid)?;
            for entry in input.entries() {
                check(&context)?;
                let tensor = match entry.value() {
                    ConditioningValue::Regular(tensor)
                    | ConditioningValue::CrossAttention(tensor)
                    | ConditioningValue::NoiseShape(tensor) => tensor,
                    _ => {
                        return Err(unsupported(
                            "Source Multiply requires tensor-backed conditioning; constant/list representations are unsupported",
                        ));
                    }
                };
                let metadata = scaled_multiply_metadata(
                    entry.metadata(),
                    multiplier,
                    compute.backend(),
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
                let descriptor = tensor.descriptor();
                let output_descriptor = sim_tensor::TensorDescriptor::contiguous(
                    descriptor.shape().to_vec(),
                    descriptor.dtype(),
                    descriptor.device(),
                    descriptor.stream(),
                )
                .map_err(|error| failure(&context, error))?;
                let value = sim_tensor::TensorBackend::binary_scalar(
                    compute.backend(),
                    sim_tensor::BinaryOperation::Multiply,
                    tensor,
                    sim_tensor::Scalar::Float(multiplier),
                    sim_tensor::ScalarSide::Right,
                    output_descriptor,
                    &execution,
                )
                .map_err(|error| failure(&context, error))?
                .0;
                let value = match entry.value() {
                    ConditioningValue::Regular(_) => ConditioningValue::regular(value),
                    ConditioningValue::CrossAttention(_) => {
                        ConditioningValue::cross_attention(value)
                    }
                    ConditioningValue::NoiseShape(_) => ConditioningValue::noise_shape(value),
                    _ => {
                        return Err(unsupported(
                            "Source Multiply requires tensor-backed conditioning",
                        ));
                    }
                }
                .map_err(invalid)?;
                entries.push(
                    ConditioningEntry::checked_with_metadata(
                        entry.identifier().to_owned(),
                        value,
                        entry.options().clone(),
                        entry.references().clone(),
                        metadata,
                    )
                    .map_err(invalid)?,
                );
            }
            let output =
                ConditioningSet::checked(input.identity().clone(), entries, &context.cancellation)
                    .map_err(|error| failure(&context, error))?;
            publish(
                &context,
                NativeStoredPayload::Conditioning(Arc::new(output)),
            )
        })
    }
}

#[cfg(test)]
mod pooled_multiply_tests {
    use super::*;
    use sim_tensor::{
        CpuWorkspaceAuthority, StreamId,
        generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    };
    use sim_types::CancellationToken;
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn pooled_multiply_preserves_absence_none_aliases_and_scales_only_ordinary_values() -> TestResult
    {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let tensor = tensor_from_f32(&backend, &[1, 2], &[2.0, -3.0], &execution)?;
        let absent = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::PooledOutputControlnet,
            ConditioningMetadataValue::ControlnetPooledOutput(Some(tensor.clone())),
        )])?;
        let mut none = absent.clone();
        none.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(None),
        )?;
        let mut some = absent.clone();
        some.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(tensor.clone())),
        )?;
        let before = backend.memory_snapshot().current_bytes;
        for input in [&absent, &none] {
            let output = scaled_multiply_metadata(input, -2.0, &backend, &execution)?;
            assert_eq!(output.len(), input.len());
            assert_eq!(backend.memory_snapshot().current_bytes, before);
            assert!(
                matches!(output.get(ConditioningMetadataKey::PooledOutputControlnet),Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(value))) if value.storage_id()==tensor.storage_id())
            );
        }
        assert!(
            scaled_multiply_metadata(&absent, -2.0, &backend, &execution)?
                .get(ConditioningMetadataKey::PooledOutput)
                .is_none()
        );
        assert!(matches!(
            scaled_multiply_metadata(&none, -2.0, &backend, &execution)?
                .get(ConditioningMetadataKey::PooledOutput),
            Some(ConditioningMetadataValue::PooledOutput(None))
        ));
        for scalar in [-100.0, -0.0, 0.0, 0.1, 1.0, 100.0, 2.0_f64.powi(-54)] {
            let output = scaled_multiply_metadata(&some, scalar, &backend, &execution)?;
            let Some(ConditioningMetadataValue::PooledOutput(Some(value))) =
                output.get(ConditioningMetadataKey::PooledOutput)
            else {
                return Err("pooled output lost".into());
            };
            assert_ne!(value.storage_id(), tensor.storage_id());
            let values = tensor_to_f32(&backend, value, &execution)?;
            assert_eq!(
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [2.0_f32, -3.0]
                    .into_iter()
                    .map(|value| (value * scalar as f32).to_bits())
                    .collect::<Vec<_>>()
            );
            assert!(
                matches!(output.get(ConditioningMetadataKey::PooledOutputControlnet),Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(value))) if value.storage_id()==tensor.storage_id())
            );
            assert!(
                matches!(some.get(ConditioningMetadataKey::PooledOutput),Some(ConditioningMetadataValue::PooledOutput(Some(value))) if value.storage_id()==tensor.storage_id())
            );
            drop(values);
            drop(output);
            assert_eq!(backend.memory_snapshot().current_bytes, before);
        }
        Ok(())
    }

    #[test]
    fn pooled_multiply_retains_typed_dtype_cancel_and_backend_capacity_failures_without_temporaries()
    -> TestResult {
        let budget = 1 << 20;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(budget)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(budget)?,
            &cancellation,
        );
        let tensor = tensor_from_f32(&backend, &[3], &[1.0, 2.0, 3.0], &execution)?;
        let input = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(tensor.clone())),
        )])?;
        let before = backend.memory_snapshot().current_bytes;
        let remaining = budget.checked_sub(before).ok_or("backend usage overflow")?;
        let reservation = backend.reserve_workspace(&execution, remaining)?;
        assert!(matches!(
            scaled_multiply_metadata(&input, 2.0, &backend, &execution),
            Err(ConditioningMultiplyMetadataError::Tensor(
                sim_tensor::TensorError::AllocationFailed { .. }
            ))
        ));
        drop(reservation);
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(budget)?,
            &cancelled,
        );
        assert!(matches!(
            scaled_multiply_metadata(&input, 2.0, &backend, &cancelled_execution),
            Err(ConditioningMultiplyMetadataError::Tensor(
                sim_tensor::TensorError::Cancelled
            ))
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        let integer = backend
            .upload_bytes(
                sim_tensor::TensorDescriptor::contiguous(
                    vec![3],
                    DType::I32,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &[0; 12],
                &execution,
            )?
            .0;
        let integer_input = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(integer)),
        )])?;
        let integer_before = backend.memory_snapshot().current_bytes;
        assert!(matches!(
            scaled_multiply_metadata(&integer_input, 2.0, &backend, &execution),
            Err(ConditioningMultiplyMetadataError::Tensor(
                sim_tensor::TensorError::UnsupportedCapability {
                    operation,
                    device,
                    ..
                }
            )) if operation == "zed.cpu.binary-scalar.multiply" && device == DeviceId::CPU
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, integer_before);
        assert!(
            matches!(input.get(ConditioningMetadataKey::PooledOutput),Some(ConditioningMetadataValue::PooledOutput(Some(value))) if value.storage_id()==tensor.storage_id())
        );
        drop(scaled_multiply_metadata(&input, 2.0, &backend, &execution)?);
        assert_eq!(backend.memory_snapshot().current_bytes, integer_before);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
enum ConditioningAverageError {
    #[error(transparent)]
    Tensor(#[from] sim_tensor::TensorError),
    #[error(transparent)]
    Narrow(#[from] sim_tensor::generated_indexing_masking_01::IndexingMaskingPartOneError),
    #[error(transparent)]
    Creation(#[from] sim_tensor::generated_tensor_creation_01::TensorCreationPartOneError),
    #[error(transparent)]
    Concatenation(
        #[from] sim_tensor::generated_shape_layout_transform_02::ShapeLayoutTransformPartTwoError,
    ),
    #[error(transparent)]
    Metadata(#[from] sim_model::conditioning::ConditioningError),
    #[error("conditioning Average representation is unsupported: {0}")]
    Unsupported(&'static str),
}

fn average_tensor_value(
    entry: &ConditioningEntry,
) -> Result<&sim_tensor::Tensor, ConditioningAverageError> {
    let tensor = match entry.value() {
        ConditioningValue::Regular(tensor)
        | ConditioningValue::CrossAttention(tensor)
        | ConditioningValue::NoiseShape(tensor) => tensor,
        _ => {
            return Err(ConditioningAverageError::Unsupported(
                "tensor-backed primary conditioning required",
            ));
        }
    };
    if tensor.descriptor().rank() != 3 {
        return Err(ConditioningAverageError::Unsupported(
            "only rank-three primary conditioning is bound",
        ));
    }
    Ok(tensor)
}

fn average_blend(
    target: &sim_tensor::Tensor,
    source: &sim_tensor::Tensor,
    strength: f64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, ConditioningAverageError> {
    use sim_tensor::{BinaryOperation, Scalar, ScalarSide, TensorBackend, TensorDescriptor};
    let descriptor = |tensor: &sim_tensor::Tensor| {
        let descriptor = tensor.descriptor();
        TensorDescriptor::contiguous(
            descriptor.shape().to_vec(),
            descriptor.dtype(),
            descriptor.device(),
            descriptor.stream(),
        )
    };
    let left = backend
        .binary_scalar(
            BinaryOperation::Multiply,
            target,
            Scalar::Float(strength),
            ScalarSide::Right,
            descriptor(target)?,
            execution,
        )?
        .0;
    let right = backend
        .binary_scalar(
            BinaryOperation::Multiply,
            source,
            Scalar::Float(1.0 - strength),
            ScalarSide::Right,
            descriptor(source)?,
            execution,
        )?
        .0;
    let shape = sim_tensor::binary_broadcast_shape(
        left.descriptor().shape(),
        right.descriptor().shape(),
    )?;
    let output = TensorDescriptor::contiguous(
        shape,
        DType::F32,
        left.descriptor().device(),
        left.descriptor().stream(),
    )?;
    Ok(backend
        .binary(BinaryOperation::Add, &left, &right, output, execution)?
        .0)
}

fn average_conditioning(
    target: &ConditioningSet,
    source: &ConditioningSet,
    strength: f64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ConditioningSet, ConditioningAverageError> {
    execution.check()?;
    if target.identity() != source.identity() {
        return Err(ConditioningAverageError::Unsupported(
            "different conditioning identities cannot form one checked set",
        ));
    }
    let first = source
        .entries()
        .first()
        .ok_or(ConditioningAverageError::Unsupported(
            "empty source conditioning is not representable by the checked set owner",
        ))?;
    let source_tensor = average_tensor_value(first)?;
    let source_pooled = match first.metadata().get(ConditioningMetadataKey::PooledOutput) {
        Some(ConditioningMetadataValue::PooledOutput(value)) => value.as_ref(),
        None => None,
        _ => {
            return Err(ConditioningAverageError::Unsupported(
                "ordinary pooled metadata must retain its canonical typed value",
            ));
        }
    };
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(target.entries().len())
        .map_err(|error| {
            sim_model::conditioning::ConditioningError::Invalid(error.to_string())
        })?;
    for entry in target.entries() {
        execution.check()?;
        let tensor = average_tensor_value(entry)?;
        let target_pooled = match entry.metadata().get(ConditioningMetadataKey::PooledOutput) {
            Some(ConditioningMetadataValue::PooledOutput(value)) => value.as_ref(),
            None => source_pooled,
            _ => {
                return Err(ConditioningAverageError::Unsupported(
                    "ordinary pooled metadata must retain its canonical typed value",
                ));
            }
        };
        let target_tokens = tensor
            .descriptor()
            .shape()
            .get(1)
            .copied()
            .ok_or(sim_tensor::TensorError::ShapeOverflow)?;
        let source_tokens = source_tensor
            .descriptor()
            .shape()
            .get(1)
            .copied()
            .ok_or(sim_tensor::TensorError::ShapeOverflow)?;
        let mut prepared =
            sim_tensor::generated_indexing_masking_01::narrow_function_exact_native(
                source_tensor,
                1,
                0,
                source_tokens.min(target_tokens),
                execution.cancellation,
            )?;
        if source_tokens < target_tokens {
            let channels = tensor
                .descriptor()
                .shape()
                .get(2)
                .copied()
                .ok_or(sim_tensor::TensorError::ShapeOverflow)?;
            let padding =
                sim_tensor::generated_tensor_creation_01::zeros_with_context_exact_native(
                    backend,
                    &[1, target_tokens - source_tokens, channels],
                    DType::F32,
                    sim_tensor::Layout::Strided,
                    DeviceId::CPU,
                    false,
                    None,
                    execution,
                )?;
            prepared =
                torch_cat_with_context_exact_native(backend, &[prepared, padding], 1, execution)?;
        }
        let value = average_blend(tensor, &prepared, strength, backend, execution)?;
        let mut metadata = entry.metadata().clone();
        if let Some(source_pooled) = source_pooled {
            let pooled = if let Some(target_pooled) = target_pooled {
                average_blend(target_pooled, source_pooled, strength, backend, execution)?
            } else {
                source_pooled.clone()
            };
            metadata.insert(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(Some(pooled)),
            )?;
        }
        let value = match entry.value() {
            ConditioningValue::Regular(_) => ConditioningValue::regular(value)?,
            ConditioningValue::CrossAttention(_) => ConditioningValue::cross_attention(value)?,
            ConditioningValue::NoiseShape(_) => ConditioningValue::noise_shape(value)?,
            _ => {
                return Err(ConditioningAverageError::Unsupported(
                    "tensor-backed primary conditioning required",
                ));
            }
        };
        entries.push(ConditioningEntry::checked_with_metadata(
            entry.identifier().to_owned(),
            value,
            entry.options().clone(),
            entry.references().clone(),
            metadata,
        )?);
    }
    Ok(ConditioningSet::checked(
        target.identity().clone(),
        entries,
        execution.cancellation,
    )?)
}

struct ConditioningAverageNode;
impl NativeNode for ConditioningAverageNode {
    fn class_type(&self) -> &str {
        "ConditioningAverage"
    }
    fn implementation_version(&self) -> &str {
        "source-conditioning-part-two-average-cpu-f32-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type(), inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate(self.class_type(), inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type(), inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for name in ["conditioning_to", "conditioning_from"] {
            let value = inputs
                .get(name)
                .ok_or_else(|| invalid("Missing Average conditioning"))?;
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
            validate(self.class_type(), &inputs)?;
            let strength = match inputs.get("conditioning_to_strength") {
                Some(NativeValue::Primitive {
                    value: NativePrimitive::Number(value),
                }) => *value,
                Some(NativeValue::Primitive {
                    value: NativePrimitive::Integer(value),
                }) => *value as f64,
                _ => return Err(invalid("Mapped source Average strength required")),
            };
            let target = resolve(
                &context,
                inputs
                    .get("conditioning_to")
                    .ok_or_else(|| invalid("Missing target conditioning"))?,
            )?;
            let source = resolve(
                &context,
                inputs
                    .get("conditioning_from")
                    .ok_or_else(|| invalid("Missing source conditioning"))?,
            )?;
            let (
                NativeStoredPayload::Conditioning(target),
                NativeStoredPayload::Conditioning(source),
            ) = (target.as_ref(), source.as_ref())
            else {
                return Err(invalid("Canonical conditioning sets required"));
            };
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let output =
                average_conditioning(target, source, strength, compute.backend(), &execution)
                    .map_err(|error| match error {
                        ConditioningAverageError::Unsupported(_) => unsupported(error),
                        _ => failure(&context, error),
                    })?;
            publish(
                &context,
                NativeStoredPayload::Conditioning(Arc::new(output)),
            )
        })
    }
}

#[cfg(test)]
mod average_tests {
    use super::*;
    use sim_model::{
        conditioning::{ConditioningEntryOptions, ConditioningIdentity, ConditioningTimestepRange},
        generated_native_diffusion::{sd15_latent_format_identity, sd15_model_family_identity},
    };
    use sim_tensor::{
        CpuWorkspaceAuthority, StreamId,
        generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    };
    use sim_types::CancellationToken;
    type TestResult = Result<(), Box<dyn std::error::Error>>;
    fn set(
        backend: &sim_tensor::CpuBackend,
        execution: &sim_tensor::ExecutionContext<'_>,
        entries: &[(&str, &[u64], &[f32])],
    ) -> Result<ConditioningSet, Box<dyn std::error::Error>> {
        let entries = entries
            .iter()
            .map(|(identifier, shape, values)| {
                let tensor = tensor_from_f32(backend, shape, values, execution)?;
                Ok(ConditioningEntry::checked_with_metadata(
                    *identifier,
                    ConditioningValue::cross_attention(tensor)?,
                    ConditioningEntryOptions {
                        strength: 0.75,
                        ..Default::default()
                    },
                    Default::default(),
                    ConditioningMetadataMap::checked([(
                        ConditioningMetadataKey::TimestepRange,
                        ConditioningMetadataValue::TimestepRange(
                            ConditioningTimestepRange::source(0.2, 0.8)?,
                        ),
                    )])?,
                )?)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        Ok(ConditioningSet::checked(
            ConditioningIdentity::new(
                "average.test",
                sd15_model_family_identity()?,
                sd15_latent_format_identity()?,
            )?,
            entries,
            execution.cancellation,
        )?)
    }
    fn values(
        set: &ConditioningSet,
        backend: &sim_tensor::CpuBackend,
        execution: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let entry = set.entries().first().ok_or("entry missing")?;
        Ok(tensor_to_f32(backend, average_tensor_value(entry)?, execution)?.to_vec())
    }

    #[test]
    fn average_descriptor_binds_source_inputs_aliases_and_only_the_assigned_row() -> TestResult {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 5);
        let binding = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "ConditioningAverage")
            .ok_or("Average binding missing")?;
        binding.descriptor().validate_exact_schema_v2()?;
        assert_eq!(binding.feature_id(), "COMFY-NODE-0088");
        assert_eq!(
            binding
                .descriptor()
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            [
                "conditioning_to",
                "conditioning_from",
                "conditioning_to_strength"
            ]
        );
        assert_eq!(
            binding
                .descriptor()
                .outputs
                .first()
                .ok_or("output missing")?
                .name,
            "conditioning"
        );
        assert_eq!(
            ConditioningAverageNode.implementation_version(),
            "source-conditioning-part-two-average-cpu-f32-v1"
        );
        for strength in [-1.0, 2.0, f64::NAN, f64::INFINITY] {
            let input = built_in_source_schema("ConditioningAverage")?
                .inputs
                .into_iter()
                .find(|input| input.schema.name == "conditioning_to_strength")
                .ok_or("strength schema missing")?;
            let value = NativeValue::Primitive {
                value: NativePrimitive::Number(strength),
            };
            assert!(
                value.validate().is_err()
                    || !native_value_matches_input_schema(&value, &input.schema)
            );
        }
        Ok(())
    }

    #[test]
    fn average_narrows_or_pads_first_source_and_broadcasts_without_repeating_other_entries()
    -> TestResult {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let target = set(&backend, &execution, &[("target", &[1, 2, 2], &[2.0; 4])])?;
        let short = set(
            &backend,
            &execution,
            &[
                ("first", &[1, 1, 2], &[8.0; 2]),
                ("ignored", &[1, 2, 2], &[99.0; 4]),
            ],
        )?;
        let input_digest = target.digest().to_owned();
        let source_digest = short.digest().to_owned();
        let padded = average_conditioning(&target, &short, 0.25, &backend, &execution)?;
        assert_eq!(values(&padded, &backend, &execution)?, [6.5, 6.5, 0.5, 0.5]);
        let long = set(
            &backend,
            &execution,
            &[("first", &[1, 3, 2], &[8.0, 8.0, 4.0, 4.0, 99.0, 99.0])],
        )?;
        assert_eq!(
            values(
                &average_conditioning(&target, &long, 0.25, &backend, &execution)?,
                &backend,
                &execution
            )?,
            [6.5, 6.5, 3.5, 3.5]
        );
        let batch_target = set(&backend, &execution, &[("target", &[2, 2, 2], &[2.0; 8])])?;
        let broadcast = average_conditioning(&batch_target, &short, 0.25, &backend, &execution)?;
        assert_eq!(
            values(&broadcast, &backend, &execution)?,
            [6.5, 6.5, 0.5, 0.5, 6.5, 6.5, 0.5, 0.5]
        );
        let channel_source = set(&backend, &execution, &[("first", &[1, 2, 1], &[8.0, 4.0])])?;
        assert_eq!(
            values(
                &average_conditioning(&target, &channel_source, 0.25, &backend, &execution)?,
                &backend,
                &execution
            )?,
            [6.5, 6.5, 3.5, 3.5]
        );
        assert_eq!(
            padded
                .entries()
                .first()
                .ok_or("missing entry")?
                .identifier(),
            "target"
        );
        assert_eq!(padded.identity(), target.identity());
        assert_eq!(
            padded
                .entries()
                .first()
                .ok_or("missing entry")?
                .options()
                .strength,
            0.75
        );
        assert!(
            matches!(padded.entries().first().ok_or("missing entry")?.metadata().get(ConditioningMetadataKey::TimestepRange),Some(ConditioningMetadataValue::TimestepRange(range)) if range.start_percent()==0.2 && range.end_percent()==0.8)
        );
        assert_eq!(target.digest(), input_digest);
        assert_eq!(short.digest(), source_digest);
        Ok(())
    }

    #[test]
    fn average_preserves_all_nine_absent_none_tensor_pool_fallback_states() -> TestResult {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let base = set(&backend, &execution, &[("target", &[1, 1, 2], &[2.0; 2])])?;
        let target_pool = tensor_from_f32(&backend, &[2], &[2.0, 2.0], &execution)?;
        let source_pool = tensor_from_f32(&backend, &[2], &[6.0, 6.0], &execution)?;
        for target_state in 0..3 {
            for source_state in 0..3 {
                let make = |state,
                            pool: &sim_tensor::Tensor|
                 -> Result<ConditioningSet, Box<dyn std::error::Error>> {
                    let mut metadata = base
                        .entries()
                        .first()
                        .ok_or("entry missing")?
                        .metadata()
                        .clone();
                    match state {
                        0 => {}
                        1 => metadata.insert(
                            ConditioningMetadataKey::PooledOutput,
                            ConditioningMetadataValue::PooledOutput(None),
                        )?,
                        _ => metadata.insert(
                            ConditioningMetadataKey::PooledOutput,
                            ConditioningMetadataValue::PooledOutput(Some(pool.clone())),
                        )?,
                    }
                    Ok(base.with_entry_metadata(vec![metadata], &cancellation)?)
                };
                let target = make(target_state, &target_pool)?;
                let source = make(source_state, &source_pool)?;
                let target_digest = target.digest().to_owned();
                let source_digest = source.digest().to_owned();
                let output = average_conditioning(&target, &source, 0.25, &backend, &execution)?;
                let value = output
                    .entries()
                    .first()
                    .ok_or("entry missing")?
                    .metadata()
                    .get(ConditioningMetadataKey::PooledOutput);
                if source_state == 2 {
                    let Some(ConditioningMetadataValue::PooledOutput(Some(pool))) = value else {
                        return Err("Source pool fallback lost".into());
                    };
                    assert_eq!(
                        tensor_to_f32(&backend, pool, &execution)?.to_vec(),
                        if target_state == 2 {
                            vec![5.0; 2]
                        } else {
                            vec![6.0; 2]
                        }
                    );
                    if target_state == 1 {
                        assert_eq!(pool.storage_id(), source_pool.storage_id());
                    } else {
                        assert_ne!(pool.storage_id(), source_pool.storage_id());
                    }
                } else {
                    match target_state {
                        0 => assert!(value.is_none()),
                        1 => assert!(matches!(
                            value,
                            Some(ConditioningMetadataValue::PooledOutput(None))
                        )),
                        _ => assert!(
                            matches!(value,Some(ConditioningMetadataValue::PooledOutput(Some(pool))) if pool.storage_id()==target_pool.storage_id())
                        ),
                    }
                }
                assert_eq!(target.digest(), target_digest);
                assert_eq!(source.digest(), source_digest);
            }
        }
        Ok(())
    }

    #[test]
    fn average_shape_errors_and_late_failures_drop_results_without_mutating_inputs() -> TestResult {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1 << 20)?,
            &cancellation,
        );
        let target = set(
            &backend,
            &execution,
            &[
                ("valid", &[1, 1, 2], &[2.0; 2]),
                ("late_invalid", &[1, 1, 3], &[2.0; 3]),
            ],
        )?;
        let source = set(&backend, &execution, &[("source", &[1, 1, 2], &[8.0; 2])])?;
        let before = backend.memory_snapshot().current_bytes;
        let digest = target.digest().to_owned();
        assert!(average_conditioning(&target, &source, 0.5, &backend, &execution).is_err());
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        assert_eq!(target.digest(), digest);
        let target = set(&backend, &execution, &[("target", &[1, 2, 2], &[2.0; 4])])?;
        for source in [
            set(&backend, &execution, &[("batch", &[2, 1, 2], &[8.0; 4])])?,
            set(&backend, &execution, &[("channel", &[1, 1, 1], &[8.0])])?,
        ] {
            let before = backend.memory_snapshot().current_bytes;
            assert!(matches!(
                average_conditioning(&target, &source, 0.5, &backend, &execution),
                Err(ConditioningAverageError::Concatenation(_))
            ));
            assert_eq!(backend.memory_snapshot().current_bytes, before);
        }
        let integer = backend
            .upload_bytes(
                sim_tensor::TensorDescriptor::contiguous(
                    vec![2],
                    DType::I32,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &[0; 8],
                &execution,
            )?
            .0;
        let mut metadata = source
            .entries()
            .first()
            .ok_or("entry missing")?
            .metadata()
            .clone();
        metadata.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(integer)),
        )?;
        let integer_pool_source = source.with_entry_metadata(vec![metadata], &cancellation)?;
        let before = backend.memory_snapshot().current_bytes;
        assert!(
            matches!(average_conditioning(&target,&integer_pool_source,0.5,&backend,&execution),Err(ConditioningAverageError::Tensor(sim_tensor::TensorError::UnsupportedCapability{operation,device,..})) if operation=="zed.cpu.binary-scalar.multiply" && device==DeviceId::CPU)
        );
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        Ok(())
    }

    #[test]
    fn average_precancel_and_real_backend_capacity_preserve_aliases_then_recover() -> TestResult {
        let budget = 1 << 20;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(budget)?;
        let cancellation = CancellationToken::default();
        let execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(budget)?,
            &cancellation,
        );
        let target = set(&backend, &execution, &[("target", &[1, 1, 2], &[2.0; 2])])?;
        let source = set(&backend, &execution, &[("source", &[1, 1, 2], &[8.0; 2])])?;
        let before = backend.memory_snapshot().current_bytes;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_execution = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(budget)?,
            &cancelled,
        );
        assert!(matches!(
            average_conditioning(&target, &source, 0.5, &backend, &cancelled_execution),
            Err(ConditioningAverageError::Tensor(
                sim_tensor::TensorError::Cancelled
            ))
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        let lease = backend.reserve_workspace(
            &execution,
            budget.checked_sub(before).ok_or("accounting overflow")?,
        )?;
        assert!(matches!(
            average_conditioning(&target, &source, 0.5, &backend, &execution),
            Err(ConditioningAverageError::Tensor(
                sim_tensor::TensorError::AllocationFailed { .. }
            ))
        ));
        drop(lease);
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        assert_eq!(
            values(
                &average_conditioning(&target, &source, 0.5, &backend, &execution)?,
                &backend,
                &execution
            )?,
            [5.0; 2]
        );
        Ok(())
    }
}
