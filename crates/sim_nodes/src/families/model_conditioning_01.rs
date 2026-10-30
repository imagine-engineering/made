use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeInputDescriptor, NativeInputRequirement,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativeResolvedPayload, NativeStoredModelPayload, NativeStoredPayload,
    NativeValue, NodeRegistry, built_in_source_schema, native_value_matches_input_schema,
    native_value_type_for_output_schema, native_value_types_for_input_schema,
};
use sim_model::clip::{NativeTokenizer, SD1_CONTEXT_LENGTH, WeightedText};
use sim_model::clip_tokenizer::{parse_prompt_weights, source_prompt_requires_textual_inversion};
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
    "AudioEncoderEncode",
    "BerniniConditioning",
    "CLIPSetLastLayer",
    "CLIPTextEncodeControlnet",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

struct ConditioningNode {
    class_type: &'static str,
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        (
            "AudioEncoderEncode",
            "COMFY-NODE-0011",
            vec!["audio_encoder_output".to_owned()],
        ),
        (
            "BerniniConditioning",
            "COMFY-NODE-0022",
            vec![
                "positive".to_owned(),
                "negative".to_owned(),
                "latent".to_owned(),
            ],
        ),
        (
            "CLIPSetLastLayer",
            "COMFY-NODE-0063",
            vec!["clip".to_owned()],
        ),
        (
            "CLIPTextEncodeControlnet",
            "COMFY-NODE-0065",
            vec!["conditioning".to_owned()],
        ),
    ]
    .into_iter()
    .map(|(class_type, feature_id, outputs)| {
        binding(
            feature_id,
            Arc::new(ConditioningNode { class_type }),
            outputs,
        )
    })
    .collect()
}

pub fn clip_text_encode_binding(
    existing_node: Arc<dyn NativeNode>,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    if existing_node.class_type() != "CLIPTextEncode" {
        return Err(schema_error(
            "The existing CLIPTextEncode owner is required",
        ));
    }
    binding(
        "COMFY-NODE-0064",
        existing_node,
        vec!["conditioning".to_owned()],
    )
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
            search_aliases: if class_type == "CLIPTextEncode" {
                [
                    "text",
                    "prompt",
                    "text prompt",
                    "positive prompt",
                    "negative prompt",
                    "encode text",
                    "text encoder",
                    "encode prompt",
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

impl ConditioningNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        for (name, value) in inputs {
            value.validate().map_err(invalid)?;
            if self.class_type == "BerniniConditioning" && name.starts_with("reference_image_") {
                return Err(unbound(
                    "bernini_context_latents_unbound",
                    "Original Bernini reference streams require canonical context-latent metadata and source VAE geometry",
                ));
            }
            let input = catalog
                .inputs
                .iter()
                .find(|input| input.schema.name == *name)
                .ok_or_else(|| invalid(format!("Unknown conditioning input {name}")))?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid("Mapped conditioning inputs cannot be empty"));
                }
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for value in values {
                if !native_value_matches_input_schema(value, &input.schema) {
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
        if self.class_type == "BerniniConditioning"
            && (inputs.contains_key("source_video") || inputs.contains_key("reference_video"))
        {
            return Err(unbound(
                "bernini_context_latents_unbound",
                "Original Bernini video streams require canonical context-latent metadata and source VAE geometry",
            ));
        }
        Ok(())
    }
}

impl NativeNode for ConditioningNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-conditioning-part-one-canonical-v1"
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
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            match self.class_type {
                "CLIPSetLastLayer" => {
                    let payload = resolve(&context, &inputs, "clip")?;
                    let NativeStoredPayload::Model(model) = payload.as_ref() else {
                        return Err(invalid("Canonical CLIP resource required"));
                    };
                    let Some(NativeValue::Primitive {
                        value: NativePrimitive::Integer(source_layer),
                    }) = inputs.get("stop_at_clip_layer")
                    else {
                        return Err(invalid("Source signed stop_at_clip_layer required"));
                    };
                    let model = model
                        .model_payload()
                        .with_source_clip_layer(*source_layer, compute.backend(), &execution)
                        .map_err(|error| failure(&context, error))?;
                    check(&context)?;
                    let model = Arc::new(model);
                    let model = if model.clip().is_some() {
                        let diffusion = sim_sampler::NativeDiffusionPayload::clip(model)
                            .map_err(|error| failure(&context, error))?;
                        NativeStoredModelPayload::native_diffusion(Arc::new(diffusion))
                            .map_err(|error| failure(&context, error))?
                    } else {
                        NativeStoredModelPayload::model_resource(model)
                            .map_err(|error| failure(&context, error))?
                    };
                    publish(
                        &context,
                        NativeStoredPayload::Model(Arc::new(model)),
                        Vec::new(),
                    )
                }
                "CLIPTextEncodeControlnet" => {
                    let clip = resolve(&context, &inputs, "clip")?;
                    let NativeStoredPayload::Model(clip) = clip.as_ref() else {
                        return Err(invalid("Canonical CLIP resource required"));
                    };
                    let model = clip.model_payload();
                    let (tokenizer, encoder) = model.clip().ok_or_else(|| unbound(
                        "controlnet_clip_backing_unbound",
                        "Controlnet text encoding requires the retained canonical SD1 tokenizer and encoder",
                    ))?;
                    let Some(NativeValue::Primitive {
                        value: NativePrimitive::String(text),
                    }) = inputs.get("text")
                    else {
                        return Err(invalid("Source text string required"));
                    };

                    if source_prompt_requires_textual_inversion(text, &context.cancellation)
                        .map_err(|error| failure(&context, error))?
                    {
                        return Err(unbound(
                            "controlnet_textual_inversion_unbound",
                            "The retained SD1 owner cannot retain Source textual-inversion embedding resources",
                        ));
                    }
                    let weighted = parse_prompt_weights(text, &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                    let segments = weighted
                        .iter()
                        .map(|segment| WeightedText::checked(segment.text(), segment.weight()))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| failure(&context, error))?;
                    let content = tokenizer
                        .encode_weighted_content(&segments, &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                    if content.len() > SD1_CONTEXT_LENGTH - 2 {
                        return Err(unbound(
                            "controlnet_text_chunking_unbound",
                            "The retained SD1 owner cannot preserve Source multi-chunk text encoding",
                        ));
                    }
                    let prompts = vec![segments];
                    let tokens = tokenizer
                        .tokenize_batch(&prompts, &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                    let encoding = if let Some(resource) = model.native_clip_resource() {
                        resource.execute_retained_sd1_batch(&tokens, &execution)
                    } else {
                        encoder.execute(&tokens, &execution)
                    }
                    .map_err(|error| failure(&context, error))?;
                    let conditioning = resolve(&context, &inputs, "conditioning")?;
                    let NativeStoredPayload::Conditioning(conditioning) = conditioning.as_ref()
                    else {
                        return Err(invalid("Canonical conditioning set required"));
                    };
                    let output = conditioning
                        .with_controlnet_text_encoding(
                            encoding.conditioning(),
                            encoding.pooled(),
                            &context.cancellation,
                        )
                        .map_err(|error| failure(&context, error))?;
                    publish(
                        &context,
                        NativeStoredPayload::Conditioning(Arc::new(output)),
                        Vec::new(),
                    )
                }
                "AudioEncoderEncode" => {
                    let encoder = resolve(&context, &inputs, "audio_encoder")?;
                    let NativeStoredPayload::Model(encoder) = encoder.as_ref() else {
                        return Err(invalid("Canonical AUDIO_ENCODER resource required"));
                    };
                    let encoder = encoder
                        .model_payload()
                        .audio_encoder_resource()
                        .ok_or_else(|| invalid("Canonical native audio encoder required"))?;
                    let audio = resolve(&context, &inputs, "audio")?;
                    let NativeStoredPayload::Audio(audio) = audio.as_ref() else {
                        return Err(invalid("Canonical AUDIO waveform required"));
                    };
                    let output = encoder
                        .encode(compute.backend(), audio, &execution)
                        .map_err(|error| failure(&context, error))?;
                    publish(
                        &context,
                        NativeStoredPayload::AudioEncoderOutput(Arc::new(output)),
                        Vec::new(),
                    )
                }
                "BerniniConditioning" => {
                    let mut conditioning_outputs = Vec::new();
                    for name in ["positive", "negative"] {
                        let payload = resolve(&context, &inputs, name)?;
                        if !matches!(payload.as_ref(), NativeStoredPayload::Conditioning(_)) {
                            return Err(invalid("Canonical CONDITIONING set required"));
                        }
                        conditioning_outputs.push(
                            inputs
                                .get(name)
                                .ok_or_else(|| invalid("Missing conditioning input"))?
                                .clone(),
                        );
                    }
                    let samples = bernini_empty_latent(
                        compute.backend(),
                        &execution,
                        integer(&inputs, "width")?,
                        integer(&inputs, "height")?,
                        integer(&inputs, "length")?,
                        integer(&inputs, "batch_size")?,
                    )
                    .map_err(|error| failure(&context, error))?;
                    // The source does not inspect VAE or clone conditioning when no context stream exists.
                    publish(
                        &context,
                        NativeStoredPayload::Latent(Arc::new(samples)),
                        conditioning_outputs,
                    )
                }
                _ => Err(invalid("Unassigned conditioning node")),
            }
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum BerniniLatentError {
    #[error(transparent)]
    Creation(#[from] sim_tensor::generated_tensor_creation_01::TensorCreationPartOneError),
    #[error(transparent)]
    Latent(#[from] sim_tensor::NativeLatentBundleError),
}

fn bernini_empty_latent(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    width: u64,
    height: u64,
    length: u64,
    batch: u64,
) -> Result<NativeLatentBundle, BerniniLatentError> {
    let frames = length
        .checked_sub(1)
        .and_then(|length| length.checked_div(4))
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| sim_tensor::NativeLatentBundleError::InvalidSamplesShape)?;
    let tensor = zeros_with_context_exact_native(
        backend,
        &[batch, 16, frames, height / 8, width / 8],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        context,
    )?;
    NativeLatentBundle::single(tensor, None, None, NativeLatentMetadata::default(), context)
        .map_err(BerniniLatentError::from)
}
fn integer(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).map_err(invalid),
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
    unbound("invalid_conditioning_inputs", error)
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
        unbound("conditioning_execution_failed", error)
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use sim_types::CancellationToken;

    #[test]
    fn source_descriptors_preserve_order_dynamic_reference_schema_and_exact_outputs()
    -> Result<(), Box<dyn std::error::Error>> {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 4);
        for binding in &bindings {
            binding.descriptor().validate_exact_schema_v2()?;
        }
        assert_eq!(bindings[0].descriptor().class_type, "AudioEncoderEncode");
        let layer = bindings
            .get(2)
            .ok_or("Missing source layer binding")?
            .descriptor();
        assert_eq!(layer.class_type, "CLIPSetLastLayer");
        assert_eq!(
            layer.outputs.first().ok_or("Missing CLIP output")?.name,
            "clip"
        );
        assert_eq!(
            layer
                .outputs
                .first()
                .ok_or("Missing CLIP output")?
                .produced_type,
            crate::native_source_type_projection("CLIP")?.value_type()?
        );

        assert_eq!(
            bindings[0].descriptor().outputs[0].produced_type,
            crate::native_source_type_projection("AUDIO_ENCODER_OUTPUT")?.value_type()?
        );
        let bernini = &bindings[1].descriptor();
        assert_eq!(bernini.dynamic_inputs.len(), 1);
        assert_eq!(
            bernini.dynamic_inputs[0].name_template,
            "reference_image_{index}"
        );
        assert_eq!(
            bernini
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["positive", "negative", "latent"]
        );
        assert_eq!(
            bernini
                .inputs
                .iter()
                .find(|input| input.name == "source_video")
                .map(|input| input.required),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn bernini_source_frame_division_and_real_zero_tensor_are_exact_and_cancelled_allocations_do_not_escape()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        for (length, frames) in [(1, 1), (4, 1), (5, 2), (6, 2), (81, 21)] {
            let output = bernini_empty_latent(&backend, &context, 16, 24, length, 2)?;
            let samples = output.samples().tensor().ok_or("Tensor samples missing")?;
            assert_eq!(samples.descriptor().shape(), &[2, 16, frames, 3, 2]);
            assert_eq!(samples.descriptor().dtype(), DType::F32);
            assert!(samples.contiguous_bytes()?.iter().all(|value| *value == 0));
        }
        cancellation.cancel();
        let error = bernini_empty_latent(&backend, &context, 16, 16, 1, 1)
            .err()
            .ok_or("Cancelled preparation succeeded")?;
        assert!(matches!(
            error,
            BerniniLatentError::Creation(
                sim_tensor::generated_tensor_creation_01::TensorCreationPartOneError::Cancelled { .. }
            )
        ));
        Ok(())
    }
}
