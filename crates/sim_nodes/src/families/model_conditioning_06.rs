use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeInputDescriptor, NativeInputRequirement,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativeStoredPayload, NativeValue, NodeRegistry, built_in_source_schema,
    native_value_matches_input_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};
use sim_model::conditioning::{
    ConditioningError, ConditioningMetadataKey, ConditioningMetadataValue, ConditioningSet,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["NormalizeVideoLatentStart", "LTXVConditioning"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

struct NormalizeVideoLatentNoopNode;

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    Ok(vec![
        binding(
            "COMFY-NODE-0457",
            Arc::new(NormalizeVideoLatentNoopNode),
            vec!["latent".to_owned()],
        )?,
        binding(
            "COMFY-NODE-0365",
            Arc::new(LtxvConditioningNode),
            vec!["positive".to_owned(), "negative".to_owned()],
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
        return Err(schema_error(format!(
            "{class_type} feature identity does not match the source"
        )));
    }
    let registry = NodeRegistry::built_in().map_err(schema_error)?;
    let row = registry
        .descriptor(class_type)
        .ok_or_else(|| schema_error(format!("Missing {class_type} source row")))?;
    let names = catalog
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let source_schema = catalog
        .bind_execution_ports(&names, &catalog.dynamic_inputs, &output_names)
        .map_err(schema_error)?;
    let inputs =
        catalog
            .inputs
            .iter()
            .map(|input| {
                let accepted_types =
                    native_value_types_for_input_schema(&input.schema).map_err(schema_error)?;
                Ok(NativeInputDescriptor {
                    name: input.schema.name.clone(),
                    allows_literal: input.schema.source_type_names.iter().all(|name| {
                        matches!(name.as_str(), "INT" | "STRING" | "FLOAT" | "BOOLEAN")
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
            search_aliases: Vec::new(),
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node,
    };
    result.validate()?;
    Ok(result)
}

impl NormalizeVideoLatentNoopNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type()).map_err(invalid)?;
        for (name, value) in inputs {
            value.validate().map_err(invalid)?;
            let input = catalog
                .inputs
                .iter()
                .find(|input| input.schema.name == *name)
                .ok_or_else(|| {
                    invalid(format!("Unknown NormalizeVideoLatentStart input {name}"))
                })?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid(
                        "Mapped NormalizeVideoLatentStart inputs cannot be empty",
                    ));
                }
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            let accepted_types =
                native_value_types_for_input_schema(&input.schema).map_err(invalid)?;
            for value in values {
                if !accepted_types.accepts(value)
                    || !native_value_matches_input_schema(value, &input.schema)
                {
                    return Err(invalid(format!("{name} violates its source schema")));
                }
            }
        }
        for input in &catalog.inputs {
            if input.requirement == NativeInputRequirement::Required
                && !inputs.contains_key(&input.schema.name)
            {
                return Err(invalid(format!(
                    "Missing source input {}",
                    input.schema.name
                )));
            }
        }
        Ok(())
    }
}

impl NativeNode for NormalizeVideoLatentNoopNode {
    fn class_type(&self) -> &str {
        "NormalizeVideoLatentStart"
    }
    fn implementation_version(&self) -> &str {
        "source-normalize-video-latent-noop-part-six-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        self.validate_inputs(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.validate_inputs(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        self.validate_inputs(inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, value) in inputs {
            let values = match value {
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for (index, value) in values.iter().enumerate() {
                if let NativeValue::Handle { value } = value {
                    let payload = context
                        .handle_store()
                        .resolve(value, value.handle_type(), &context.cancellation)
                        .map_err(|error| failure(context, error))?;
                    artifact_digests.insert(format!("{name}.{index}"), payload.digest_sha256());
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
            self.validate_inputs(&inputs)?;
            let latent = inputs
                .get("latent")
                .ok_or_else(|| invalid("Missing source latent"))?;
            let NativeValue::Handle { value } = latent else {
                return Err(invalid("Canonical latent handle required"));
            };
            let retained = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let NativeStoredPayload::Latent(bundle) = retained.as_ref() else {
                return Err(unbound(
                    "source_latent_bundle_required",
                    "Canonical single-sample latent bundle required",
                ));
            };
            let samples = bundle.samples().tensor().ok_or_else(|| unbound(
                "source_single_sample_tensor_unbound",
                "The source reads the samples tensor shape; nested audio/video samples are not admitted by this slice",
            ))?;
            let frames = samples
                .descriptor()
                .shape()
                .get(2)
                .copied()
                .ok_or_else(|| invalid("Source samples shape has no third dimension"))?;
            if frames > 1 {
                return Err(unbound(
                    "source_video_normalization_unbound",
                    "Multi-frame source normalization requires canonical reduction, clamp and source-dictionary update integration",
                ));
            }
            check(&context)?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![latent.clone()],
                ui: None,
                effects: Vec::new(),
            };
            outcome.validate().map_err(invalid)?;
            Ok(outcome)
        })
    }
}

fn schema_error(error: impl std::fmt::Display) -> NativeNodeContractError {
    NativeNodeContractError::InvalidSourceSchema(error.to_string())
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    unbound("invalid_normalize_video_latent_inputs", error)
}
fn unbound(code: &str, error: impl std::fmt::Display) -> NativeNodeFailure {
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
        unbound("normalize_video_latent_execution_failed", error)
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}

struct LtxvConditioningNode;
impl LtxvConditioningNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type()).map_err(invalid_ltxv)?;
        for (name, value) in inputs {
            value.validate().map_err(invalid_ltxv)?;
            let input = catalog
                .inputs
                .iter()
                .find(|input| input.schema.name == *name)
                .ok_or_else(|| invalid_ltxv(format!("Unknown LTXVConditioning input {name}")))?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid_ltxv(
                        "Mapped LTXVConditioning inputs cannot be empty",
                    ));
                }
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            let accepted_types =
                native_value_types_for_input_schema(&input.schema).map_err(invalid_ltxv)?;
            for value in values {
                if !accepted_types.accepts(value)
                    || !native_value_matches_input_schema(value, &input.schema)
                {
                    return Err(invalid_ltxv(format!("{name} violates its source schema")));
                }
            }
        }
        for input in &catalog.inputs {
            if input.requirement == NativeInputRequirement::Required
                && !inputs.contains_key(&input.schema.name)
            {
                return Err(invalid_ltxv(format!(
                    "Missing source input {}",
                    input.schema.name
                )));
            }
        }
        Ok(())
    }
}

impl NativeNode for LtxvConditioningNode {
    fn class_type(&self) -> &str {
        "LTXVConditioning"
    }
    fn implementation_version(&self) -> &str {
        "source-ltxv-frame-rate-part-six-canonical-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_ltxv(context)?;
        self.validate_inputs(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.validate_inputs(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_ltxv(context)?;
        self.validate_inputs(inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, value) in inputs {
            let values = match value {
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for (index, value) in values.iter().enumerate() {
                if let NativeValue::Handle { value } = value {
                    let payload = context
                        .handle_store()
                        .resolve(value, value.handle_type(), &context.cancellation)
                        .map_err(|error| failure_ltxv(context, error))?;
                    artifact_digests.insert(format!("{name}.{index}"), payload.digest_sha256());
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
            check_ltxv(&context)?;
            self.validate_inputs(&inputs)?;
            let Some(NativeValue::Primitive {
                value: NativePrimitive::Number(frame_rate),
            }) = inputs.get("frame_rate")
            else {
                return Err(invalid_ltxv("Mapped source FLOAT frame_rate required"));
            };
            let mut handles = Vec::new();
            handles.try_reserve_exact(2).map_err(invalid_ltxv)?;
            let result = (|| {
                for name in ["positive", "negative"] {
                    check_ltxv(&context)?;
                    let Some(NativeValue::Handle { value }) = inputs.get(name) else {
                        return Err(invalid_ltxv("Source CONDITIONING handle required"));
                    };
                    let retained = context
                        .handle_store()
                        .resolve(value, value.handle_type(), &context.cancellation)
                        .map_err(|error| failure_ltxv(&context, error))?;
                    let NativeStoredPayload::Conditioning(conditioning) = retained.as_ref() else {
                        return Err(invalid_ltxv("Canonical CONDITIONING set required"));
                    };
                    let updated =
                        ltxv_frame_rate_metadata(conditioning, *frame_rate, &context.cancellation)
                            .map_err(|error| failure_ltxv(&context, error))?;
                    let handle = context
                        .handle_store()
                        .publish(
                            NativeStoredPayload::Conditioning(Arc::new(updated)),
                            &context.cancellation,
                        )
                        .map_err(|error| failure_ltxv(&context, error))?;
                    handles.push(handle);
                    check_ltxv(&context)?;
                }
                let outcome = NativeNodeOutcome::Values {
                    outputs: handles
                        .iter()
                        .cloned()
                        .map(|value| NativeValue::Handle { value })
                        .collect(),
                    ui: None,
                    effects: Vec::new(),
                };
                outcome.validate().map_err(invalid_ltxv)?;
                Ok(outcome)
            })();
            if let Err(original) = result {
                let mut rollback_failures = Vec::new();
                for handle in handles.iter().rev() {
                    if let Err(error) = context
                        .handle_store()
                        .revoke(handle, &sim_types::CancellationToken::default())
                    {
                        rollback_failures.push(error.to_string());
                    }
                }
                if !rollback_failures.is_empty() {
                    return Err(unbound(
                        "ltxv_conditioning_rollback_failed",
                        format!(
                            "Original {}: {}; rollback: {}",
                            original.code,
                            original.message,
                            rollback_failures.join("; ")
                        ),
                    ));
                }
                return Err(original);
            }
            result
        })
    }
}

fn ltxv_frame_rate_metadata(
    conditioning: &ConditioningSet,
    frame_rate: f64,
    cancellation: &sim_types::CancellationToken,
) -> Result<ConditioningSet, ConditioningError> {
    cancellation.check()?;
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(conditioning.entries().len())
        .map_err(|_| ConditioningError::ShapeOverflow("LTXV frame-rate metadata entries"))?;
    for entry in conditioning.entries() {
        cancellation.check()?;
        let mut values = entry.metadata().clone();
        values.insert(
            ConditioningMetadataKey::FrameRate,
            ConditioningMetadataValue::FiniteF64(frame_rate),
        )?;
        metadata.push(values);
    }
    conditioning.with_entry_metadata(metadata, cancellation)
}
fn invalid_ltxv(error: impl std::fmt::Display) -> NativeNodeFailure {
    unbound("invalid_ltxv_conditioning_inputs", error)
}
fn failure_ltxv(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message: error.to_string(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        unbound("ltxv_conditioning_execution_failed", error)
    }
}
fn check_ltxv(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure_ltxv(context, error))
}
