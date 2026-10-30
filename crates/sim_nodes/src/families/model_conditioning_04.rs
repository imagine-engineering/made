use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeInputDescriptor, NativeInputRequirement,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativeResolvedPayload, NativeStoredPayload, NativeValue, NodeRegistry,
    built_in_source_schema, native_value_matches_input_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, Layout, NativeLatentBundle,
    NativeLatentMetadata, generated_tensor_creation_01::zeros_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "CosmosImageToVideoLatent",
    "CosmosPredict2ImageToVideoLatent",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

struct CosmosEmptyNode {
    class_type: &'static str,
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        ("CosmosImageToVideoLatent", "COMFY-NODE-0112"),
        ("CosmosPredict2ImageToVideoLatent", "COMFY-NODE-0113"),
    ]
    .into_iter()
    .map(|(class_type, feature_id)| {
        binding(
            feature_id,
            Arc::new(CosmosEmptyNode { class_type }),
            vec!["latent".to_owned()],
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
            "Cosmos feature identity does not match the source",
        ));
    }
    let registry = NodeRegistry::built_in().map_err(schema_error)?;
    let row = registry
        .descriptor(class_type)
        .ok_or_else(|| schema_error("Missing Cosmos source row"))?;
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

impl CosmosEmptyNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        for (name, value) in inputs {
            value.validate().map_err(invalid)?;
            let input = catalog
                .inputs
                .iter()
                .find(|input| input.schema.name == *name)
                .ok_or_else(|| invalid(format!("Unknown Cosmos input {name}")))?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid("Mapped Cosmos inputs cannot be empty"));
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

impl NativeNode for CosmosEmptyNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-cosmos-empty-part-four-canonical-v1"
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
            let vae = resolve(&context, &inputs, "vae")?;
            if !matches!(vae.as_ref(), NativeStoredPayload::Model(_)) {
                return Err(invalid("Canonical VAE resource required"));
            }
            if inputs.contains_key("start_image") || inputs.contains_key("end_image") {
                return Err(unbound(
                    "cosmos_video_conditioning_unbound",
                    "Source image-conditioned Cosmos preparation requires canonical video VAE padding, slice assignment, and Predict2 source latent transforms",
                ));
            }
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let divisor = match self.class_type {
                "CosmosImageToVideoLatent" => 8,
                "CosmosPredict2ImageToVideoLatent" => 4,
                _ => return Err(invalid("Unassigned Cosmos node")),
            };
            let samples = cosmos_empty_latent(
                compute.backend(),
                &execution,
                integer(&inputs, "width")?,
                integer(&inputs, "height")?,
                integer(&inputs, "length")?,
                divisor,
            )
            .map_err(|error| failure(&context, error))?;
            // Both source bodies return before batch repetition and VAE execution when no images exist.
            publish(
                &context,
                NativeStoredPayload::Latent(Arc::new(samples)),
                Vec::new(),
            )
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum CosmosEmptyError {
    #[error(transparent)]
    Creation(#[from] sim_tensor::generated_tensor_creation_01::TensorCreationPartOneError),
    #[error(transparent)]
    Latent(#[from] sim_tensor::NativeLatentBundleError),
}

fn cosmos_empty_latent(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    width: u64,
    height: u64,
    length: u64,
    temporal_divisor: u64,
) -> Result<NativeLatentBundle, CosmosEmptyError> {
    let frames = length
        .checked_sub(1)
        .and_then(|value| value.checked_div(temporal_divisor))
        .and_then(|value| value.checked_add(1))
        .ok_or(sim_tensor::NativeLatentBundleError::InvalidSamplesShape)?;
    let tensor = zeros_with_context_exact_native(
        backend,
        &[1, 16, frames, height / 8, width / 8],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        context,
    )?;
    NativeLatentBundle::single(tensor, None, None, NativeLatentMetadata::default(), context)
        .map_err(CosmosEmptyError::from)
}

fn integer(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).map_err(invalid),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Ok(*value),
        _ => Err(invalid(format!("Source integer {name} required"))),
    }
}
fn resolve(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = inputs.get(name) else {
        return Err(invalid(format!("Source handle {name} required")));
    };
    context
        .handle_store()
        .resolve(value, value.handle_type(), &context.cancellation)
        .map_err(|error| failure(context, error))
}
fn publish(
    context: &NativeNodeContext,
    payload: NativeStoredPayload,
    mut outputs: Vec<NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check(context)?;
    let handle = context
        .handle_store()
        .publish(payload, &context.cancellation)
        .map_err(|error| failure(context, error))?;
    outputs.push(NativeValue::Handle {
        value: handle.clone(),
    });
    let result = check(context).and_then(|()| {
        let outcome = NativeNodeOutcome::Values {
            outputs,
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
    unbound("invalid_cosmos_inputs", error)
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
        unbound("cosmos_execution_failed", error)
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}
