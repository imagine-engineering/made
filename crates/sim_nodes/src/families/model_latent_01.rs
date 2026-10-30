use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeHandleKind, NativeHandleType,
    NativeInputDescriptor, NativeInputRequirement, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
    native_value_matches_input_schema, native_value_types_for_input_schema,
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, Layout, NativeLatentBundle,
    NativeLatentMetadata, NativeLatentSamples, NativeLatentType, ResizeCrop, ResizeMode,
    ResizeSpec, Tensor, TensorBackend, TensorDescriptor,
    generated_shape_layout_transform_02::{
        tensor_reshape_with_context_exact_native, torch_cat_with_context_exact_native,
        torch_movedim_exact_native,
    },
    generated_tensor_creation_01::zeros_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "BatchLatentsNode",
    "EmptyAceStep1.5LatentAudio",
    "EmptyAceStepLatentAudio",
    "EmptyARVideoLatent",
    "EmptyChromaRadianceLatentImage",
    "EmptyCosmosLatentVideo",
    "EmptyFlux2LatentImage",
    "EmptyHiDreamO1LatentImage",
    "EmptyHunyuanImageLatent",
    "EmptyHunyuanLatentVideo",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const FEATURES: [&str; 10] = [
    "COMFY-NODE-0018",
    "COMFY-NODE-0149",
    "COMFY-NODE-0150",
    "COMFY-NODE-0151",
    "COMFY-NODE-0153",
    "COMFY-NODE-0154",
    "COMFY-NODE-0155",
    "COMFY-NODE-0156",
    "COMFY-NODE-0157",
    "COMFY-NODE-0158",
];

#[derive(Clone, Copy)]
struct LatentNode {
    class_type: &'static str,
}

fn latent_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Latent, "LATENT")
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    NODE_DESCRIPTOR_IDS
        .iter()
        .zip(FEATURES)
        .map(|(class_type, feature_id)| {
            let catalog = built_in_source_schema(class_type)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let names = catalog
                .inputs
                .iter()
                .map(|input| input.schema.name.clone())
                .collect::<Vec<_>>();
            let source_schema = catalog
                .bind_execution_ports(&names, &catalog.dynamic_inputs, &["latent".to_owned()])
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let inputs = catalog
                .inputs
                .iter()
                .map(|input| {
                    Ok(NativeInputDescriptor {
                        name: input.schema.name.clone(),
                        accepted_types: native_value_types_for_input_schema(&input.schema)
                            .map_err(|error| {
                                NativeNodeContractError::InvalidSourceSchema(error.to_string())
                            })?,
                        required: input.requirement == NativeInputRequirement::Required,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Mapped,
                        allows_literal: true,
                    })
                })
                .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
            let dynamic_inputs = if *class_type == "BatchLatentsNode" {
                vec![NativeDynamicInputDescriptor {
                    name_template: "latent{index}".to_owned(),
                    start_index: 1,
                    minimum_count: 1,
                    maximum_count: 50,
                    input: NativeInputDescriptor {
                        name: "latent".to_owned(),
                        accepted_types: NativeTypeUnion::new(vec![NativeValueType::Handle(
                            latent_type()?,
                        )])?,
                        required: true,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Mapped,
                        allows_literal: false,
                    },
                }]
            } else {
                Vec::new()
            };
            let node = LatentNode { class_type };
            let category = match *class_type {
                "BatchLatentsNode" => "batch",
                "EmptyAceStep1.5LatentAudio" | "EmptyAceStepLatentAudio" => "ace",
                "EmptyARVideoLatent" => "autoregressive",
                "EmptyChromaRadianceLatentImage" => "chroma radiance",
                "EmptyCosmosLatentVideo" => "cosmos",
                "EmptyFlux2LatentImage" => "flux",
                "EmptyHiDreamO1LatentImage" => "hidream",
                "EmptyHunyuanImageLatent" => "hunyuan image",
                "EmptyHunyuanLatentVideo" => "hunyuan video",
                _ => {
                    return Err(NativeNodeContractError::InvalidSourceSchema(
                        "Unknown latent node".to_owned(),
                    ));
                }
            };
            Ok(NativeNodeBinding::Executable {
                feature_id: feature_id.to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: (*class_type).to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(source_schema),
                    inputs,
                    dynamic_inputs,
                    outputs: vec![NativeOutputDescriptor {
                        name: "latent".to_owned(),
                        produced_type: NativeValueType::Handle(latent_type()?),
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
                        .unwrap_or_else(|| (*class_type).to_owned()),
                    category: format!("model/latent/{category}"),
                    description: catalog.presentation.description.clone().unwrap_or_default(),
                    output_names: vec!["latent".to_owned()],
                    search_aliases: if *class_type == "BatchLatentsNode" {
                        ["combine latents", "stack latents", "merge latents"]
                            .into_iter()
                            .map(str::to_owned)
                            .collect()
                    } else {
                        Vec::new()
                    },
                    is_deprecated: catalog.presentation.is_deprecated,
                    is_experimental: catalog.presentation.is_experimental,
                },
                node: Arc::new(node),
            })
        })
        .collect()
}

fn dynamic_names(inputs: &BTreeMap<String, NativeValue>) -> Result<Vec<&str>, NativeNodeFailure> {
    if inputs.is_empty() || inputs.len() > 50 {
        return Err(invalid("BatchLatentsNode requires 1 through 50 inputs"));
    }
    let mut indexed = inputs
        .keys()
        .map(|name| {
            let index = name
                .strip_prefix("latent")
                .ok_or_else(|| invalid("Invalid dynamic LATENT input"))?
                .parse::<u32>()
                .map_err(invalid)?;
            if !(1..=50).contains(&index) || name != &format!("latent{index}") {
                return Err(invalid(
                    "Dynamic LATENT names must match latent1 through latent50",
                ));
            }
            Ok((index, name.as_str()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    indexed.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in (1..).zip(&indexed) {
        if expected != *actual {
            return Err(invalid("Dynamic LATENT inputs must be contiguous"));
        }
    }
    Ok(indexed.into_iter().map(|(_, name)| name).collect())
}

impl LatentNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        if self.class_type == "BatchLatentsNode" {
            dynamic_names(inputs)?;
        } else if inputs.len() != catalog.inputs.len()
            || catalog
                .inputs
                .iter()
                .any(|input| !inputs.contains_key(&input.schema.name))
        {
            return Err(invalid("Missing or unknown empty-LATENT input"));
        }
        for (name, value) in inputs {
            value.validate().map_err(invalid)?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid("Required mapped LATENT inputs cannot be empty"));
                }
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for value in values {
                if self.class_type == "BatchLatentsNode" {
                    latent_handle(value)?;
                } else {
                    let input = catalog
                        .inputs
                        .iter()
                        .find(|input| input.schema.name == *name)
                        .ok_or_else(|| invalid("Unknown empty-LATENT input"))?;
                    if !native_value_matches_input_schema(value, &input.schema) {
                        return Err(invalid(format!("{name} violates its source schema")));
                    }
                    if name == "seconds" {
                        number_value(value)?;
                    } else {
                        let value = integer_value(value)?;
                        if matches!(name.as_str(), "width" | "height" | "length") && value > 16_384
                        {
                            return Err(invalid("LATENT dimensions exceed source MAX_RESOLUTION"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl NativeNode for LatentNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-latent-part-one-v1"
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
        if self.class_type == "BatchLatentsNode" {
            for (name, value) in inputs {
                let values = match value {
                    NativeValue::List { values } => values.as_slice(),
                    value => std::slice::from_ref(value),
                };
                for (index, value) in values.iter().enumerate() {
                    let payload = context
                        .handle_store()
                        .resolve(
                            latent_handle(value)?,
                            &latent_type().map_err(invalid)?,
                            &context.cancellation,
                        )
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
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let payload = if self.class_type == "BatchLatentsNode" {
                let retained = dynamic_names(&inputs)?
                    .into_iter()
                    .map(|name| {
                        let value = inputs
                            .get(name)
                            .ok_or_else(|| invalid("Missing LATENT input"))?;
                        context
                            .handle_store()
                            .resolve(
                                latent_handle(value)?,
                                &latent_type().map_err(invalid)?,
                                &context.cancellation,
                            )
                            .map_err(|error| failure(&context, error))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let latents = retained
                    .iter()
                    .map(|payload| match payload.as_ref() {
                        NativeStoredPayload::Latent(latent) => Ok(latent.as_ref()),
                        _ => Err(invalid("LATENT bundle payload required")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                batch_latents(&latents, compute.backend(), &execution)
            } else {
                empty_latent(self.class_type, &inputs, compute.backend(), &execution)
            }
            .map_err(|error| failure(&context, error))?;
            check(&context)?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Latent(Arc::new(payload)),
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
            let result = check(&context).and_then(|()| outcome(handle.clone()));
            if result.is_err() {
                context
                    .handle_store()
                    .revoke(&handle, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
            }
            result
        })
    }
}

fn latent_handle(value: &NativeValue) -> Result<&NativeOpaqueHandle, NativeNodeFailure> {
    match value {
        NativeValue::Handle { value }
            if value.handle_type() == &latent_type().map_err(invalid)? =>
        {
            Ok(value)
        }
        _ => Err(invalid("LATENT handle required")),
    }
}
fn integer_value(value: &NativeValue) -> Result<u64, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => u64::try_from(*value).map_err(invalid),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => Ok(*value),
        _ => Err(invalid("Integer LATENT parameter required")),
    }
}
fn number_value(value: &NativeValue) -> Result<f64, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        } if value.is_finite() => Ok(*value),
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => Ok(*value as f64),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => Ok(*value as f64),
        _ => Err(invalid("Finite numeric LATENT parameter required")),
    }
}
fn outcome(value: NativeOpaqueHandle) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let result = NativeNodeOutcome::Values {
        outputs: vec![NativeValue::Handle { value }],
        ui: None,
        effects: Vec::new(),
    };
    result.validate().map_err(invalid)?;
    Ok(result)
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
    NativeNodeFailure {
        code: if context.cancellation.is_cancelled() {
            "interrupted"
        } else {
            "latent_failed"
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

fn empty_latent(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let integer = |name: &str| {
        integer_value(
            inputs
                .get(name)
                .ok_or_else(|| invalid(format!("Missing {name}")))?,
        )
    };
    let batch = integer("batch_size")?;
    let mut latent_type = None;
    let mut spatial_ratio = None;
    let mut temporal_ratio = None;
    let shape = match class_type {
        "EmptyAceStep1.5LatentAudio" | "EmptyAceStepLatentAudio" => {
            let seconds = number_value(
                inputs
                    .get("seconds")
                    .ok_or_else(|| invalid("Missing seconds"))?,
            )?;
            latent_type = Some(NativeLatentType::Audio);
            if class_type == "EmptyAceStep1.5LatentAudio" {
                temporal_ratio = Some(1764);
                vec![
                    batch,
                    64,
                    (seconds * 48000.0 / 1920.0).round_ties_even() as u64,
                ]
            } else {
                vec![batch, 8, 16, (seconds * 44100.0 / 512.0 / 8.0) as u64]
            }
        }
        _ => {
            let width = integer("width")?;
            let height = integer("height")?;
            match class_type {
                "EmptyChromaRadianceLatentImage" | "EmptyHiDreamO1LatentImage" => {
                    vec![batch, 3, height, width]
                }
                "EmptyFlux2LatentImage" => vec![batch, 128, height / 16, width / 16],
                "EmptyHunyuanImageLatent" => vec![batch, 64, height / 32, width / 32],
                "EmptyARVideoLatent" | "EmptyCosmosLatentVideo" | "EmptyHunyuanLatentVideo" => {
                    let length = integer("length")?;
                    let ratio = if class_type == "EmptyCosmosLatentVideo" {
                        8
                    } else {
                        4
                    };
                    if class_type == "EmptyHunyuanLatentVideo" {
                        spatial_ratio = Some(8);
                    }
                    vec![
                        batch,
                        16,
                        length
                            .checked_sub(1)
                            .ok_or_else(|| invalid("Video length must be positive"))?
                            / ratio
                            + 1,
                        height / 8,
                        width / 8,
                    ]
                }
                _ => return Err(invalid("Unknown empty LATENT node")),
            }
        }
    };
    let samples = zeros_with_context_exact_native(
        backend,
        &shape,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        execution,
    )
    .map_err(invalid)?;
    let metadata = NativeLatentMetadata::checked(latent_type, None, spatial_ratio, temporal_ratio)
        .map_err(invalid)?;
    NativeLatentBundle::single(samples, None, None, metadata, execution).map_err(invalid)
}

fn batch_latents(
    latents: &[&NativeLatentBundle],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let first = latents
        .first()
        .ok_or_else(|| invalid("At least one LATENT input is required"))?;
    let target = first.samples().tensor().ok_or_else(|| {
        invalid("Source BatchLatentsNode cannot concatenate nested LATENT tensors")
    })?;
    let target_shape = target.descriptor().shape();
    let mut tensors = Vec::new();
    let mut batch_indices = Vec::new();
    for latent in latents {
        execution.check().map_err(invalid)?;
        let samples = latent.samples().tensor().ok_or_else(|| {
            invalid("Source BatchLatentsNode cannot concatenate nested LATENT tensors")
        })?;
        let shape = samples.descriptor().shape();
        let batch = shape
            .first()
            .copied()
            .ok_or_else(|| invalid("LATENT batch dimension is missing"))?;
        let count = usize::try_from(batch).map_err(invalid)?;
        batch_indices.try_reserve(count).map_err(invalid)?;
        if let Some(indices) = latent.batch_indices() {
            batch_indices.extend_from_slice(indices);
        } else {
            batch_indices.extend(0..batch);
        }
        tensors.push(if shape.get(1..) == target_shape.get(1..) {
            samples.clone()
        } else {
            resize_latent(samples, target_shape, backend, execution)?
        });
    }
    let samples =
        torch_cat_with_context_exact_native(backend, &tensors, 0, execution).map_err(invalid)?;
    NativeLatentBundle::checked(
        NativeLatentSamples::Tensor(samples),
        first.noise_mask().cloned(),
        Some(batch_indices),
        first.metadata().clone(),
        execution,
    )
    .map_err(invalid)
}

pub(crate) fn resize_latent(
    samples: &Tensor,
    target: &[u64],
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let shape = samples.descriptor().shape();
    if !(4..=5).contains(&shape.len()) {
        return Err(invalid(
            "Source bilinear LATENT resizing requires image or video samples",
        ));
    }
    let [batch, channels, ..] = shape else {
        return Err(invalid("LATENT dimensions are missing"));
    };
    let width = target
        .last()
        .copied()
        .ok_or_else(|| invalid("Target width is missing"))?;
    let height = target
        .get(
            target
                .len()
                .checked_sub(2)
                .ok_or_else(|| invalid("Target height is missing"))?,
        )
        .copied()
        .ok_or_else(|| invalid("Target height is missing"))?;
    let reshape = |input: &Tensor, shape: &[u64]| {
        let shape = shape
            .iter()
            .map(|size| i64::try_from(*size).map_err(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        tensor_reshape_with_context_exact_native(backend, input, &shape, execution).map_err(invalid)
    };
    let (input, frames) = if let [_, _, frames, source_height, source_width] = shape {
        let reordered = torch_movedim_exact_native(samples, &[2], &[1], execution.cancellation)
            .map_err(invalid)?;
        let flattened_batch = batch
            .checked_mul(*frames)
            .ok_or_else(|| invalid("LATENT dimensions overflow"))?;
        (
            reshape(
                &reordered,
                &[flattened_batch, *channels, *source_height, *source_width],
            )?,
            Some(*frames),
        )
    } else {
        (samples.clone(), None)
    };
    let output_batch = input
        .descriptor()
        .shape()
        .first()
        .copied()
        .ok_or_else(|| invalid("LATENT batch dimension is missing"))?;
    let descriptor = TensorDescriptor::contiguous(
        vec![output_batch, *channels, height, width],
        input.descriptor().dtype(),
        input.descriptor().device(),
        input.descriptor().stream(),
    )
    .map_err(invalid)?;
    let (resized, _) = backend
        .resize(
            ResizeSpec {
                width,
                height,
                mode: ResizeMode::Bilinear,
                crop: ResizeCrop::Center,
                antialias: false,
                align_corners: false,
            },
            &input,
            descriptor,
            execution,
        )
        .map_err(invalid)?;
    if let Some(frames) = frames {
        let expanded = reshape(&resized, &[*batch, frames, *channels, height, width])?;
        torch_movedim_exact_native(&expanded, &[2], &[1], execution.cancellation).map_err(invalid)
    } else {
        Ok(resized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn latent_part_one_preserves_exact_source_contracts() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/model-latent-comfy-node-0018/fixture.json"
        )))?;
        let rows = fixture
            .get("nodes")
            .and_then(serde_json::Value::as_array)
            .ok_or("source rows absent")?;
        assert_eq!(rows.len(), 10);
        let bindings = native_node_bindings()?;
        crate::validate_generated_family_bindings(&bindings, NODE_DESCRIPTOR_IDS)?;
        let registry = crate::NodeRegistry::built_in()?;
        assert_eq!(bindings.len(), 10);
        for binding in bindings {
            registry.validate_native_binding(&binding)?;
            let descriptor = binding.descriptor();
            descriptor.validate_exact_schema_v2()?;
            let row = rows
                .iter()
                .find(|row| {
                    row.get("class_type").and_then(serde_json::Value::as_str)
                        == Some(descriptor.class_type.as_str())
                })
                .ok_or("assigned source row absent")?;
            assert_eq!(
                row.get("feature_id").and_then(serde_json::Value::as_str),
                Some(binding.feature_id())
            );
            assert_eq!(
                serde_json::to_value(built_in_source_schema(&descriptor.class_type)?)?,
                *row.get("schema").ok_or("source schema absent")?
            );
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert!(!descriptor.output_node);
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Mapped));
            assert_eq!(descriptor.outputs.len(), 1);
            assert_eq!(
                binding.presentation().category,
                row["category"].as_str().ok_or("category absent")?
            );
            if descriptor.class_type == "BatchLatentsNode" {
                let dynamic = descriptor
                    .dynamic_inputs
                    .first()
                    .ok_or("dynamic inputs absent")?;
                assert_eq!(
                    (
                        dynamic.start_index,
                        dynamic.minimum_count,
                        dynamic.maximum_count
                    ),
                    (1, 1, 50)
                );
                assert_eq!(dynamic.name_template, "latent{index}");
                assert_eq!(dynamic.input.cardinality, NativePortCardinality::Mapped);
                assert!(!dynamic.input.allows_literal);
                assert_eq!(
                    binding.presentation().search_aliases,
                    ["combine latents", "stack latents", "merge latents"]
                );
            } else {
                assert!(descriptor.dynamic_inputs.is_empty());
            }
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
            assert_eq!(
                *descriptor,
                serde_json::from_str::<NativeNodeDescriptor>(&serde_json::to_string(descriptor)?)?
            );
        }
        Ok(())
    }
}
