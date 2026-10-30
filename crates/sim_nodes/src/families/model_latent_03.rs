use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativeStoredPayload, NativeValue, NativeValueType,
    NodeRegistry, built_in_source_schema, native_value_matches_input_schema,
    native_value_types_for_input_schema,
};
use sim_sampler::{NativeGuidanceModifier, NativeGuidanceModifierChain};
use sim_tensor::{
    Layout, MemoryFormatReference, NativeLatentBundle, NativeLatentCompositeMode,
    NativeLatentMetadataRetention, NativeLatentSamples, NativeTensorRole,
    composite_latents_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_shape_layout_transform_02::{
        tensor_repeat_with_context_exact_native, tensor_reshape_with_context_exact_native,
        torch_cat_with_context_exact_native, torch_movedim_exact_native,
    },
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
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
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

struct LatentBatchSeedBehavior;
struct LatentTransform {
    class_type: &'static str,
}
struct LatentOperationNode {
    class_type: &'static str,
}
struct LatentCompositeNode {
    class_type: &'static str,
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let class_type = "LatentBatchSeedBehavior";
    let catalog = built_in_source_schema(class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let registry = NodeRegistry::built_in()
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let row = registry.descriptor(class_type).ok_or_else(|| {
        NativeNodeContractError::InvalidSourceSchema("Missing latent source row".to_owned())
    })?;
    let names = catalog
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let source_schema = catalog
        .bind_execution_ports(&names, &[], &["latent".to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog
        .inputs
        .iter()
        .map(|input| {
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types: native_value_types_for_input_schema(&input.schema).map_err(
                    |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                )?,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Mapped,
                allows_literal: input.schema.name != "samples",
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let mut bindings = vec![NativeNodeBinding::Executable {
        feature_id: "COMFY-NODE-0314".to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: "source-latent-batch-seed-behavior-v1".to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "latent".to_owned(),
                produced_type: NativeValueType::Handle(NativeHandleType::new(
                    NativeHandleKind::Latent,
                    "LATENT",
                )?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: catalog
                .presentation
                .display_name
                .clone()
                .unwrap_or_else(|| class_type.to_owned()),
            category: row.category.clone(),
            description: catalog.presentation.description.clone().unwrap_or_default(),
            output_names: vec!["latent".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node: Arc::new(LatentBatchSeedBehavior),
    }];
    bindings.push(latent_transform_binding("LatentBatch", "COMFY-NODE-0313")?);
    bindings.push(latent_transform_binding("LatentConcat", "COMFY-NODE-0318")?);
    bindings.push(latent_transform_binding("LatentCrop", "COMFY-NODE-0319")?);
    bindings.push(latent_transform_binding("LatentCut", "COMFY-NODE-0320")?);
    bindings.push(latent_transform_binding(
        "LatentCutToBatch",
        "COMFY-NODE-0321",
    )?);
    bindings.push(latent_operation_binding(
        "LatentApplyOperation",
        "COMFY-NODE-0311",
    )?);
    bindings.push(latent_operation_binding(
        "LatentApplyOperationCFG",
        "COMFY-NODE-0312",
    )?);
    bindings.push(latent_composite_binding(
        "LatentComposite",
        "COMFY-NODE-0316",
    )?);
    bindings.push(latent_composite_binding(
        "LatentCompositeMasked",
        "COMFY-NODE-0317",
    )?);
    Ok(bindings)
}

fn latent_composite_binding(
    class_type: &'static str,
    feature_id: &str,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog = built_in_source_schema(class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let registry = NodeRegistry::built_in()
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let row = registry.descriptor(class_type).ok_or_else(|| {
        NativeNodeContractError::InvalidSourceSchema("Missing latent composite row".to_owned())
    })?;
    let source_schema = catalog
        .bind_execution_ports(
            &catalog
                .inputs
                .iter()
                .map(|input| input.schema.name.clone())
                .collect::<Vec<_>>(),
            &[],
            &["latent".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog
        .inputs
        .iter()
        .map(|input| {
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types: native_value_types_for_input_schema(&input.schema).map_err(
                    |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                )?,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Mapped,
                allows_literal: !matches!(
                    input.schema.name.as_str(),
                    "samples_to" | "samples_from" | "destination" | "source" | "mask"
                ),
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    Ok(NativeNodeBinding::Executable {
        feature_id: feature_id.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: "source-latent-composite-v1".to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "latent".to_owned(),
                produced_type: NativeValueType::Handle(NativeHandleType::new(
                    NativeHandleKind::Latent,
                    "LATENT",
                )?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: row.display_name.clone(),
            category: row.category.clone(),
            description: catalog.presentation.description.clone().unwrap_or_default(),
            output_names: vec!["latent".to_owned()],
            search_aliases: vec![
                "overlay latent".to_owned(),
                "layer latent".to_owned(),
                "paste latent".to_owned(),
            ],
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node: Arc::new(LatentCompositeNode { class_type }),
    })
}

fn latent_operation_binding(
    class_type: &'static str,
    feature_id: &str,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog = built_in_source_schema(class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let registry = NodeRegistry::built_in()
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let row = registry.descriptor(class_type).ok_or_else(|| {
        NativeNodeContractError::InvalidSourceSchema("Missing latent operation row".to_owned())
    })?;
    let output_type = if class_type == "LatentApplyOperation" {
        "LATENT"
    } else {
        "MODEL"
    };
    let output_kind = if class_type == "LatentApplyOperation" {
        NativeHandleKind::Latent
    } else {
        NativeHandleKind::Model
    };
    let output_name = if class_type == "LatentApplyOperation" {
        "latent"
    } else {
        "model"
    };
    let source_schema = catalog
        .bind_execution_ports(
            &catalog
                .inputs
                .iter()
                .map(|input| input.schema.name.clone())
                .collect::<Vec<_>>(),
            &[],
            &[output_name.to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog
        .inputs
        .iter()
        .map(|input| {
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types: native_value_types_for_input_schema(&input.schema).map_err(
                    |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                )?,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Mapped,
                allows_literal: false,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    Ok(NativeNodeBinding::Executable {
        feature_id: feature_id.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: "source-latent-operation-v1".to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: output_name.to_owned(),
                produced_type: NativeValueType::Handle(NativeHandleType::new(
                    output_kind,
                    output_type,
                )?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: row.display_name.clone(),
            category: row.category.clone(),
            description: catalog.presentation.description.clone().unwrap_or_default(),
            output_names: vec![output_name.to_owned()],
            search_aliases: if class_type == "LatentApplyOperation" {
                vec!["transform latent".to_owned()]
            } else {
                Vec::new()
            },
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node: Arc::new(LatentOperationNode { class_type }),
    })
}

fn latent_transform_binding(
    class_type: &'static str,
    feature_id: &str,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog = built_in_source_schema(class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let registry = NodeRegistry::built_in()
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let row = registry.descriptor(class_type).ok_or_else(|| {
        NativeNodeContractError::InvalidSourceSchema("Missing latent transform row".to_owned())
    })?;
    let names = catalog
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let source_schema = catalog
        .bind_execution_ports(&names, &[], &["latent".to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog
        .inputs
        .iter()
        .map(|input| {
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types: native_value_types_for_input_schema(&input.schema).map_err(
                    |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                )?,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Mapped,
                allows_literal: !matches!(
                    input.schema.name.as_str(),
                    "samples" | "samples1" | "samples2"
                ),
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    Ok(NativeNodeBinding::Executable {
        feature_id: feature_id.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: "source-latent-transform-v1".to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "latent".to_owned(),
                produced_type: NativeValueType::Handle(NativeHandleType::new(
                    NativeHandleKind::Latent,
                    "LATENT",
                )?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: row.display_name.clone(),
            category: row.category.clone(),
            description: catalog.presentation.description.clone().unwrap_or_default(),
            output_names: vec!["latent".to_owned()],
            search_aliases: match class_type {
                "LatentCut" => vec![
                    "crop latent".to_owned(),
                    "slice latent".to_owned(),
                    "extract region".to_owned(),
                ],
                "LatentCrop" => vec!["trim latent".to_owned(), "cut latent".to_owned()],
                "LatentConcat" => vec!["join latents".to_owned(), "stitch latents".to_owned()],
                "LatentBatch" => vec![
                    "combine latents".to_owned(),
                    "merge latents".to_owned(),
                    "join latents".to_owned(),
                ],
                "LatentCutToBatch" => vec![
                    "slice to batch".to_owned(),
                    "split latent".to_owned(),
                    "tile latent".to_owned(),
                ],
                _ => Vec::new(),
            },
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node: Arc::new(LatentTransform { class_type }),
    })
}

fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_latent_inputs".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".to_owned(),
            message: error.to_string(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        invalid(error)
    }
}

fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}

fn validate_inputs(inputs: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
    let catalog = built_in_source_schema("LatentBatchSeedBehavior").map_err(invalid)?;
    if inputs.len() != catalog.inputs.len()
        || catalog
            .inputs
            .iter()
            .any(|input| !inputs.contains_key(&input.schema.name))
    {
        return Err(invalid("Missing or unknown latent input"));
    }
    for source in &catalog.inputs {
        let value = &inputs[&source.schema.name];
        value.validate().map_err(invalid)?;
        if !native_value_matches_input_schema(value, &source.schema) {
            return Err(invalid(format!(
                "{} violates its source schema",
                source.schema.name
            )));
        }
    }
    Ok(())
}

fn behavior(inputs: &BTreeMap<String, NativeValue>) -> Result<&str, NativeNodeFailure> {
    match inputs.get("seed_behavior") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => Ok(value),
        _ => Err(invalid("seed_behavior must be a source option")),
    }
}

fn batch_size(samples: &NativeLatentSamples) -> Result<usize, NativeNodeFailure> {
    let tensor = match samples {
        NativeLatentSamples::Tensor(tensor) => tensor,
        NativeLatentSamples::AudioVideo { video, .. } => video,
    };
    tensor
        .descriptor()
        .shape()
        .first()
        .copied()
        .ok_or_else(|| invalid("Missing latent batch dimension"))
        .and_then(|size| usize::try_from(size).map_err(invalid))
}

impl NativeNode for LatentBatchSeedBehavior {
    fn class_type(&self) -> &str {
        "LatentBatchSeedBehavior"
    }
    fn implementation_version(&self) -> &str {
        "source-latent-batch-seed-behavior-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate_inputs(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_inputs(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate_inputs(inputs)?;
        let Some(NativeValue::Handle { value }) = inputs.get("samples") else {
            return Err(invalid("LATENT handle required"));
        };
        let payload = context
            .handle_store()
            .resolve(value, value.handle_type(), &context.cancellation)
            .map_err(|error| failure(context, error))?;
        Ok(NativeCacheDependencies {
            artifact_digests: BTreeMap::from([("samples".to_owned(), payload.digest_sha256())]),
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
            validate_inputs(&inputs)?;
            let Some(NativeValue::Handle { value }) = inputs.get("samples") else {
                return Err(invalid("LATENT handle required"));
            };
            let payload = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                return Err(invalid("LATENT bundle required"));
            };
            let indices = match behavior(&inputs)? {
                "random" => None,
                "fixed" => Some(vec![
                    latent
                        .batch_indices()
                        .and_then(|indices| indices.first().copied())
                        .unwrap_or(0);
                    batch_size(latent.samples())?
                ]),
                _ => return Err(invalid("Unknown seed_behavior")),
            };
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let result = NativeLatentBundle::checked(
                latent.samples().clone(),
                latent.noise_mask().cloned(),
                indices,
                latent.metadata().clone(),
                &execution,
            )
            .map_err(|error| failure(&context, error))?;
            check(&context)?;
            let output = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Latent(Arc::new(result)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: output.clone(),
                }],
                ui: None,
                effects: Vec::new(),
            };
            let validated = check(&context)
                .and_then(|()| outcome.validate().map_err(invalid).map(|()| outcome));
            if validated.is_err() {
                context
                    .handle_store()
                    .revoke(&output, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
            }
            validated
        })
    }
}

fn validate_transform_inputs(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    let catalog = built_in_source_schema(class_type).map_err(invalid)?;
    if inputs.keys().any(|name| {
        !catalog
            .inputs
            .iter()
            .any(|input| input.schema.name == *name)
    }) || catalog.inputs.iter().any(|input| {
        input.requirement == NativeInputRequirement::Required
            && !inputs.contains_key(&input.schema.name)
    }) {
        return Err(invalid("Missing or unknown latent transform input"));
    }
    for source in &catalog.inputs {
        let Some(value) = inputs.get(&source.schema.name) else {
            continue;
        };
        value.validate().map_err(invalid)?;
        if !native_value_matches_input_schema(value, &source.schema) {
            return Err(invalid(format!(
                "{} violates its source schema",
                source.schema.name
            )));
        }
    }
    Ok(())
}

fn cut_integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<i64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => i64::try_from(*value).map_err(invalid),
        _ => Err(invalid(format!("{name} must be an integer"))),
    }
}

impl NativeNode for LatentTransform {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-latent-transform-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate_transform_inputs(self.class_type, inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_transform_inputs(self.class_type, inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate_transform_inputs(self.class_type, inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, input) in inputs {
            let NativeValue::Handle { value } = input else {
                continue;
            };
            let payload = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(context, error))?;
            artifact_digests.insert(name.clone(), payload.digest_sha256());
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
            validate_transform_inputs(self.class_type, &inputs)?;
            let source_name = if matches!(self.class_type, "LatentBatch" | "LatentConcat") {
                "samples1"
            } else {
                "samples"
            };
            let Some(NativeValue::Handle { value }) = inputs.get(source_name) else {
                return Err(invalid("LATENT handle required"));
            };
            let payload = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                return Err(invalid("LATENT bundle required"));
            };
            let NativeLatentSamples::Tensor(samples) = latent.samples() else {
                return Err(invalid("LatentCut requires tensor samples"));
            };
            let shape = samples.descriptor().shape();
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let narrowed = if self.class_type == "LatentCut" {
                let dimension = match inputs.get("dim") {
                    Some(NativeValue::Primitive {
                        value: NativePrimitive::String(value),
                    }) => match value.as_str() {
                        "x" => shape.len().checked_sub(1),
                        "y" => shape.len().checked_sub(2),
                        "t" => shape.len().checked_sub(3),
                        _ => None,
                    },
                    _ => None,
                }
                .ok_or_else(|| invalid("Unknown LatentCut dimension"))?;
                let dimension_size = i64::try_from(
                    *shape
                        .get(dimension)
                        .ok_or_else(|| invalid("Missing cut dimension"))?,
                )
                .map_err(invalid)?;
                if dimension_size == 0 {
                    return Err(invalid("Cannot cut an empty dimension"));
                }
                let index = cut_integer(&inputs, "index")?;
                let amount = cut_integer(&inputs, "amount")?;
                let (start, length) = if index >= 0 {
                    let start = index.min(dimension_size - 1);
                    (start, amount.min(dimension_size - start))
                } else {
                    let start = index.max(-dimension_size);
                    (dimension_size + start, amount.min(-start))
                };
                narrow_method_exact_native(
                    samples,
                    i64::try_from(dimension).map_err(invalid)?,
                    start,
                    u64::try_from(length).map_err(invalid)?,
                    &context.cancellation,
                )
                .map_err(invalid)?
            } else if self.class_type == "LatentCutToBatch" {
                let dimension = match inputs.get("dim") {
                    Some(NativeValue::Primitive {
                        value: NativePrimitive::String(value),
                    }) => match value.as_str() {
                        "x" => shape.len().checked_sub(1),
                        "y" => shape.len().checked_sub(2),
                        "t" => shape.len().checked_sub(3),
                        _ => None,
                    },
                    _ => None,
                }
                .ok_or_else(|| invalid("Unknown LatentCutToBatch dimension"))?;
                if dimension < 2 {
                    check(&context)?;
                    let outcome = NativeNodeOutcome::Values {
                        outputs: vec![NativeValue::Handle {
                            value: value.clone(),
                        }],
                        ui: None,
                        effects: Vec::new(),
                    };
                    outcome.validate().map_err(invalid)?;
                    return Ok(outcome);
                }
                let slice_size =
                    u64::try_from(cut_integer(&inputs, "slice_size")?).map_err(invalid)?;
                let moved = torch_movedim_exact_native(
                    samples,
                    &[i64::try_from(dimension).map_err(invalid)?],
                    &[1],
                    &context.cancellation,
                )
                .map_err(invalid)?;
                let length = *moved
                    .descriptor()
                    .shape()
                    .get(1)
                    .ok_or_else(|| invalid("Missing slice dimension"))?;
                let slice_size = slice_size.min(length);
                if slice_size == 0 {
                    return Err(invalid("Cannot slice an empty dimension"));
                }
                let adjusted_length = (length / slice_size) * slice_size;
                let trimmed = if adjusted_length != length {
                    narrow_method_exact_native(&moved, 1, 0, adjusted_length, &context.cancellation)
                        .map_err(invalid)?
                } else {
                    moved
                };
                let mut new_shape = vec![-1, i64::try_from(slice_size).map_err(invalid)?];
                new_shape.extend(
                    trimmed
                        .descriptor()
                        .shape()
                        .iter()
                        .skip(2)
                        .map(|value| i64::try_from(*value).map_err(invalid))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let reshaped = tensor_reshape_with_context_exact_native(
                    compute.backend(),
                    &trimmed,
                    &new_shape,
                    &execution,
                )
                .map_err(invalid)?;
                torch_movedim_exact_native(
                    &reshaped,
                    &[1],
                    &[i64::try_from(dimension).map_err(invalid)?],
                    &context.cancellation,
                )
                .map_err(invalid)?
            } else if self.class_type == "LatentCrop" {
                if shape.len() != 4 {
                    return Err(invalid("LatentCrop requires rank-four samples"));
                }
                let x = cut_integer(&inputs, "x")? / 8;
                let y = cut_integer(&inputs, "y")? / 8;
                let width = cut_integer(&inputs, "width")? / 8;
                let height = cut_integer(&inputs, "height")? / 8;
                let sample_width = i64::try_from(shape[3]).map_err(invalid)?;
                let sample_height = i64::try_from(shape[2]).map_err(invalid)?;
                let x = if x > sample_width - 8 {
                    sample_width - 8
                } else {
                    x
                };
                let y = if y > sample_height - 8 {
                    sample_height - 8
                } else {
                    y
                };
                let start_x = x.max(0);
                let start_y = y.max(0);
                let end_x = x.saturating_add(width).clamp(0, sample_width);
                let end_y = y.saturating_add(height).clamp(0, sample_height);
                let row = narrow_method_exact_native(
                    samples,
                    2,
                    start_y,
                    u64::try_from(end_y.saturating_sub(start_y)).map_err(invalid)?,
                    &context.cancellation,
                )
                .map_err(invalid)?;
                narrow_method_exact_native(
                    &row,
                    3,
                    start_x,
                    u64::try_from(end_x.saturating_sub(start_x)).map_err(invalid)?,
                    &context.cancellation,
                )
                .map_err(invalid)?
            } else {
                let Some(NativeValue::Handle { value }) = inputs.get("samples2") else {
                    return Err(invalid("Second LATENT handle required"));
                };
                let second_payload = context
                    .handle_store()
                    .resolve(value, value.handle_type(), &context.cancellation)
                    .map_err(|error| failure(&context, error))?;
                let NativeStoredPayload::Latent(second) = second_payload.as_ref() else {
                    return Err(invalid("Second LATENT bundle required"));
                };
                let NativeLatentSamples::Tensor(second_samples) = second.samples() else {
                    return Err(invalid("Nested LATENT samples are unsupported by source"));
                };
                let second_shape = second_samples.descriptor().shape();
                if self.class_type == "LatentBatch" {
                    let resized = if shape.get(1..) != second_shape.get(1..) {
                        crate::generated_model_latent_01::resize_latent(
                            second_samples,
                            shape,
                            compute.backend(),
                            &execution,
                        )?
                    } else {
                        second_samples.clone()
                    };
                    torch_cat_with_context_exact_native(
                        compute.backend(),
                        &[samples.clone(), resized],
                        0,
                        &execution,
                    )
                    .map_err(invalid)?
                } else {
                    let target_batch = *shape
                        .first()
                        .ok_or_else(|| invalid("Missing target batch"))?;
                    let source_batch = *second_shape
                        .first()
                        .ok_or_else(|| invalid("Missing source batch"))?;
                    if source_batch == 0 {
                        return Err(invalid("Cannot repeat an empty batch"));
                    }
                    let repeated = if source_batch < target_batch {
                        let mut repeats = vec![1; second_shape.len()];
                        repeats[0] =
                            i64::try_from(target_batch.div_ceil(source_batch)).map_err(invalid)?;
                        tensor_repeat_with_context_exact_native(
                            compute.backend(),
                            second_samples,
                            &repeats,
                            &execution,
                        )
                        .map_err(invalid)?
                    } else {
                        second_samples.clone()
                    };
                    let second = if source_batch != target_batch {
                        narrow_method_exact_native(
                            &repeated,
                            0,
                            0,
                            target_batch,
                            &context.cancellation,
                        )
                        .map_err(invalid)?
                    } else {
                        repeated
                    };
                    let direction = match inputs.get("dim") {
                        Some(NativeValue::Primitive {
                            value: NativePrimitive::String(value),
                        }) => value.as_str(),
                        _ => return Err(invalid("Concat dimension required")),
                    };
                    let dimension = match direction {
                        "x" | "-x" => -1,
                        "y" | "-y" => -2,
                        "t" | "-t" => -3,
                        _ => return Err(invalid("Unknown concat dimension")),
                    };
                    let (left, right) = if direction.starts_with('-') {
                        (second, samples.clone())
                    } else {
                        (samples.clone(), second)
                    };
                    torch_cat_with_context_exact_native(
                        compute.backend(),
                        &[left, right],
                        dimension,
                        &execution,
                    )
                    .map_err(invalid)?
                }
            };
            let output_samples = contiguous_with_context_exact_native(
                compute.backend(),
                &narrowed,
                MemoryFormatReference::Layout(Layout::Contiguous),
                &execution,
            )
            .map_err(invalid)?;
            let output_latent = if self.class_type == "LatentBatch" {
                let Some(NativeValue::Handle { value }) = inputs.get("samples2") else {
                    return Err(invalid("Second LATENT handle required"));
                };
                let second_payload = context
                    .handle_store()
                    .resolve(value, value.handle_type(), &context.cancellation)
                    .map_err(|error| failure(&context, error))?;
                let NativeStoredPayload::Latent(second) = second_payload.as_ref() else {
                    return Err(invalid("Second LATENT bundle required"));
                };
                let first_batch = usize::try_from(
                    *shape
                        .first()
                        .ok_or_else(|| invalid("Missing first batch"))?,
                )
                .map_err(invalid)?;
                let second_batch = usize::try_from(
                    second
                        .samples()
                        .tensor()
                        .ok_or_else(|| invalid("Nested LATENT samples unsupported"))?
                        .descriptor()
                        .shape()[0],
                )
                .map_err(invalid)?;
                let indices = latent.batch_indices().map_or_else(
                    || (0..first_batch as u64).collect::<Vec<_>>(),
                    |value| value.to_vec(),
                );
                let mut indices = indices;
                indices.extend(second.batch_indices().map_or_else(
                    || (0..second_batch as u64).collect::<Vec<_>>(),
                    |value| value.to_vec(),
                ));
                latent
                    .replaced_samples_and_indices_source_dictionary(
                        NativeLatentSamples::Tensor(output_samples),
                        Some(indices),
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?
            } else if self.class_type == "LatentCutToBatch" {
                latent
                    .replaced_samples_source_dictionary(
                        NativeLatentSamples::Tensor(output_samples),
                        &execution,
                    )
                    .map_err(invalid)?
            } else {
                latent
                    .replaced_samples(
                        NativeLatentSamples::Tensor(output_samples),
                        NativeLatentMetadataRetention::Preserve,
                        &execution,
                    )
                    .map_err(invalid)?
            };
            check(&context)?;
            let output = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Latent(Arc::new(output_latent)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let result = check(&context).and_then(|()| {
                let outcome = NativeNodeOutcome::Values {
                    outputs: vec![NativeValue::Handle {
                        value: output.clone(),
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
                    .revoke(&output, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
            }
            result
        })
    }
}

impl NativeNode for LatentOperationNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-latent-operation-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate_transform_inputs(self.class_type, inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_transform_inputs(self.class_type, inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate_transform_inputs(self.class_type, inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, input) in inputs {
            let NativeValue::Handle { value } = input else {
                continue;
            };
            let payload = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(context, error))?;
            artifact_digests.insert(name.clone(), payload.digest_sha256());
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
            validate_transform_inputs(self.class_type, &inputs)?;
            let Some(NativeValue::Handle {
                value: operation_handle,
            }) = inputs.get("operation")
            else {
                return Err(invalid("LATENT_OPERATION handle required"));
            };
            let operation_payload = context
                .handle_store()
                .resolve(
                    operation_handle,
                    operation_handle.handle_type(),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let NativeStoredPayload::Model(operation_model) = operation_payload.as_ref() else {
                return Err(invalid("LATENT_OPERATION resource required"));
            };
            let operation = operation_model
                .model_payload()
                .latent_operation_resource()
                .ok_or_else(|| invalid("Concrete LATENT_OPERATION resource required"))?;
            let source_name = if self.class_type == "LatentApplyOperation" {
                "samples"
            } else {
                "model"
            };
            let Some(NativeValue::Handle {
                value: source_handle,
            }) = inputs.get(source_name)
            else {
                return Err(invalid("Source handle required"));
            };
            let source_payload = context
                .handle_store()
                .resolve(
                    source_handle,
                    source_handle.handle_type(),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let output_payload = if self.class_type == "LatentApplyOperation" {
                let NativeStoredPayload::Latent(latent) = source_payload.as_ref() else {
                    return Err(invalid("LATENT bundle required"));
                };
                let samples = latent
                    .samples()
                    .tensor()
                    .ok_or_else(|| invalid("Tensor LATENT samples required"))?;
                let compute = context.compute_session().map_err(invalid)?;
                let execution = compute.execution_context(&context).map_err(invalid)?;
                let transformed = operation
                    .apply(samples, compute.backend(), &execution)
                    .map_err(|error| failure(&context, error))?;
                let result = latent
                    .replaced_samples(
                        NativeLatentSamples::Tensor(transformed),
                        NativeLatentMetadataRetention::Preserve,
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?;
                NativeStoredPayload::Latent(Arc::new(result))
            } else {
                let NativeStoredPayload::Model(model) = source_payload.as_ref() else {
                    return Err(invalid("MODEL resource required"));
                };
                let diffusion = model
                    .diffusion()
                    .ok_or_else(|| invalid("Native diffusion MODEL required"))?;
                let mut modifiers = diffusion.guidance_modifiers().modifiers().to_vec();
                modifiers.push(NativeGuidanceModifier::LatentOperation {
                    operation: operation.kind(),
                });
                let chain =
                    Arc::new(NativeGuidanceModifierChain::checked(modifiers).map_err(invalid)?);
                let changed = Arc::new(
                    diffusion
                        .as_ref()
                        .clone()
                        .with_guidance_modifiers(chain)
                        .map_err(invalid)?,
                );
                let changed = match model.compiled_model() {
                    Some(compiled) => crate::NativeStoredModelPayload::native_compiled_diffusion(
                        changed,
                        compiled.clone(),
                    ),
                    None => crate::NativeStoredModelPayload::native_diffusion(changed),
                }
                .map_err(invalid)?;
                NativeStoredPayload::Model(Arc::new(changed))
            };
            check(&context)?;
            let output = context
                .handle_store()
                .publish(output_payload, &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let result = check(&context).and_then(|()| {
                let outcome = NativeNodeOutcome::Values {
                    outputs: vec![NativeValue::Handle {
                        value: output.clone(),
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
                    .revoke(&output, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
            }
            result
        })
    }
}

impl NativeNode for LatentCompositeNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-latent-composite-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        validate_transform_inputs(self.class_type, inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_transform_inputs(self.class_type, inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate_transform_inputs(self.class_type, inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, input) in inputs {
            let NativeValue::Handle { value } = input else {
                continue;
            };
            let payload = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(context, error))?;
            artifact_digests.insert(name.clone(), payload.digest_sha256());
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
            validate_transform_inputs(self.class_type, &inputs)?;
            let masked = self.class_type == "LatentCompositeMasked";
            let (destination_name, source_name) = if masked {
                ("destination", "source")
            } else {
                ("samples_to", "samples_from")
            };
            let resolve_latent =
                |name: &str| -> Result<Arc<NativeLatentBundle>, NativeNodeFailure> {
                    let Some(NativeValue::Handle { value }) = inputs.get(name) else {
                        return Err(invalid(format!("{name} LATENT handle required")));
                    };
                    let payload = context
                        .handle_store()
                        .resolve(value, value.handle_type(), &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                    let NativeStoredPayload::Latent(latent) = payload.as_ref() else {
                        return Err(invalid(format!("{name} LATENT bundle required")));
                    };
                    Ok(latent.clone())
                };
            let destination = resolve_latent(destination_name)?;
            let source = resolve_latent(source_name)?;
            let destination_tensor = destination
                .samples()
                .tensor()
                .ok_or_else(|| invalid("Destination tensor required"))?;
            let source_tensor = source
                .samples()
                .tensor()
                .ok_or_else(|| invalid("Source tensor required"))?;
            let mask = if let Some(NativeValue::Handle { value }) = inputs.get("mask") {
                let payload = context
                    .handle_store()
                    .resolve(value, value.handle_type(), &context.cancellation)
                    .map_err(|error| failure(&context, error))?;
                let NativeStoredPayload::Tensor(tensor) = payload.as_ref() else {
                    return Err(invalid("MASK tensor required"));
                };
                if tensor.role() != NativeTensorRole::Mask {
                    return Err(invalid("MASK role required"));
                }
                Some(tensor.clone())
            } else {
                None
            };
            let x = u64::try_from(cut_integer(&inputs, "x")?).map_err(invalid)?;
            let y = u64::try_from(cut_integer(&inputs, "y")?).map_err(invalid)?;
            let mode = if masked {
                let resize_source = match inputs.get("resize_source") {
                    Some(NativeValue::Primitive {
                        value: NativePrimitive::Boolean(value),
                    }) => *value,
                    _ => return Err(invalid("resize_source must be a boolean")),
                };
                NativeLatentCompositeMode::Masked { resize_source }
            } else {
                NativeLatentCompositeMode::Feather {
                    pixels: u64::try_from(cut_integer(&inputs, "feather")?).map_err(invalid)?,
                }
            };
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let result = composite_latents_exact_native(
                compute.backend(),
                destination_tensor,
                source_tensor,
                mask.as_deref().map(|mask| mask.tensor()),
                x,
                y,
                mode,
                &execution,
            )
            .map_err(|error| failure(&context, error))?;
            let output_latent = destination
                .replaced_samples(
                    NativeLatentSamples::Tensor(result),
                    NativeLatentMetadataRetention::Preserve,
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
            check(&context)?;
            let output = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Latent(Arc::new(output_latent)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let result = check(&context).and_then(|()| {
                let outcome = NativeNodeOutcome::Values {
                    outputs: vec![NativeValue::Handle {
                        value: output.clone(),
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
                    .revoke(&output, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
            }
            result
        })
    }
}
