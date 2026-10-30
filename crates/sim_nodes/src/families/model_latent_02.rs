use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativeResolvedPayload,
    NativeStoredPayload, NativeValue, NativeValueType, NodeRegistry, built_in_source_schema,
    generated_model_latent_01::resize_latent, native_value_matches_input_schema,
    native_value_types_for_input_schema,
};
use sim_model::latent_upscale_model::{HunyuanLatentUpscaleCrop, HunyuanLatentUpscaleMode};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, Layout, NativeLatentBundle,
    NativeLatentMetadata, NativeLatentMetadataRetention, NativeLatentSamples, NativeLatentType,
    Tensor, generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_16::add_method_with_context_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_shape_layout_transform_02::tensor_repeat_with_context_exact_native,
    generated_tensor_creation_01::zeros_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "EmptyHunyuanVideo15Latent",
    "EmptyLatentAudio",
    "EmptyLatentHunyuan3Dv2",
    "EmptyLTXVLatentVideo",
    "EmptyMochiLatentVideo",
    "EmptyQwenImageLayeredLatentImage",
    "EmptySD3LatentImage",
    "HunyuanVideo15LatentUpscaleWithModel",
    "LatentAdd",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const FEATURES: [&str; 9] = [
    "COMFY-NODE-0159",
    "COMFY-NODE-0161",
    "COMFY-NODE-0162",
    "COMFY-NODE-0164",
    "COMFY-NODE-0165",
    "COMFY-NODE-0166",
    "COMFY-NODE-0167",
    "COMFY-NODE-0233",
    "COMFY-NODE-0310",
];

#[derive(Clone, Copy)]
struct LatentNode {
    class_type: &'static str,
}

fn latent_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Latent, "LATENT")
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let registry = NodeRegistry::built_in()
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    NODE_DESCRIPTOR_IDS
        .iter()
        .zip(FEATURES)
        .map(|(class_type, feature_id)| {
            let catalog = built_in_source_schema(class_type)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let row = registry.descriptor(class_type).ok_or_else(|| {
                NativeNodeContractError::InvalidSourceSchema("Missing latent row".to_owned())
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
                        accepted_types: native_value_types_for_input_schema(&input.schema)
                            .map_err(|error| {
                                NativeNodeContractError::InvalidSourceSchema(error.to_string())
                            })?,
                        required: input.requirement == NativeInputRequirement::Required,
                        hidden: false,
                        lazy: false,
                        cardinality: NativePortCardinality::Mapped,
                        allows_literal: !matches!(
                            input.schema.name.as_str(),
                            "model" | "samples" | "samples1" | "samples2"
                        ),
                    })
                })
                .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
            let node = LatentNode { class_type };
            Ok(NativeNodeBinding::Executable {
                feature_id: feature_id.to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: (*class_type).to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(source_schema),
                    inputs,
                    dynamic_inputs: Vec::new(),
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
                    category: row.category.clone(),
                    description: catalog.presentation.description.clone().unwrap_or_default(),
                    output_names: vec!["latent".to_owned()],
                    search_aliases: if *class_type == "LatentAdd" {
                        vec!["combine latents".to_owned(), "sum latents".to_owned()]
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

impl LatentNode {
    fn validate_inputs(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        let catalog = built_in_source_schema(self.class_type).map_err(invalid)?;
        if inputs.len() != catalog.inputs.len()
            || catalog
                .inputs
                .iter()
                .any(|input| !inputs.contains_key(&input.schema.name))
        {
            return Err(invalid("Missing or unknown LATENT input"));
        }
        for input in &catalog.inputs {
            let value = inputs
                .get(&input.schema.name)
                .ok_or_else(|| invalid("Missing LATENT input"))?;
            value.validate().map_err(invalid)?;
            let values = match value {
                NativeValue::List { values } if values.is_empty() => {
                    return Err(invalid("Required mapped LATENT inputs cannot be empty"));
                }
                NativeValue::List { values } => values.as_slice(),
                value => std::slice::from_ref(value),
            };
            for value in values {
                if !native_value_matches_input_schema(value, &input.schema) {
                    return Err(invalid(format!(
                        "{} violates its source schema",
                        input.schema.name
                    )));
                }
                if matches!(
                    input.schema.name.as_str(),
                    "width" | "height" | "length" | "layers"
                ) && integer(value)? > 16_384
                {
                    return Err(invalid("LATENT dimensions exceed source MAX_RESOLUTION"));
                }
            }
        }
        Ok(())
    }
}

pub fn validate_empty_latent_image_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    LatentNode {
        class_type: "EmptyLatentImage",
    }
    .validate_inputs(inputs)
}

impl NativeNode for LatentNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        "source-latent-part-two-v1"
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
            let payload = match self.class_type {
                "LatentAdd" => {
                    let first = resolve(&context, &inputs, "samples1")?;
                    let second = resolve(&context, &inputs, "samples2")?;
                    add_latents(
                        latent(&first)?,
                        latent(&second)?,
                        compute.backend(),
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?
                }
                "HunyuanVideo15LatentUpscaleWithModel" => {
                    let samples = resolve(&context, &inputs, "samples")?;
                    let width = u32::try_from(input_integer(&inputs, "width")?).map_err(invalid)?;
                    let height =
                        u32::try_from(input_integer(&inputs, "height")?).map_err(invalid)?;
                    if width == 0 && height == 0 {
                        check(&context)?;
                        return outcome(handle(input(&inputs, "samples")?)?.clone());
                    }
                    let model = resolve(&context, &inputs, "model")?;
                    let NativeStoredPayload::Model(model) = model.as_ref() else {
                        return Err(invalid("LATENT_UPSCALE_MODEL payload required"));
                    };
                    let resource = model
                        .model_payload()
                        .latent_upscale_model_resource()
                        .ok_or_else(|| invalid("LATENT_UPSCALE_MODEL resource required"))?;
                    let mode = match string(input(&inputs, "upscale_method")?)? {
                        "nearest-exact" => HunyuanLatentUpscaleMode::NearestExact,
                        "bilinear" => HunyuanLatentUpscaleMode::Bilinear,
                        "area" => HunyuanLatentUpscaleMode::Area,
                        "bicubic" => HunyuanLatentUpscaleMode::Bicubic,
                        "bislerp" => HunyuanLatentUpscaleMode::Bislerp,
                        _ => return Err(invalid("Unknown Hunyuan upscale method")),
                    };
                    let crop = match string(input(&inputs, "crop")?)? {
                        "disabled" => HunyuanLatentUpscaleCrop::Disabled,
                        "center" => HunyuanLatentUpscaleCrop::Center,
                        _ => return Err(invalid("Unknown Hunyuan upscale crop")),
                    };
                    resource
                        .invoke_hunyuan_bundle(
                            compute.backend(),
                            latent(&samples)?,
                            width,
                            height,
                            mode,
                            crop,
                            &execution,
                        )
                        .map_err(|error| failure(&context, error))?
                }
                _ => empty_latent(self.class_type, &inputs, compute.backend(), &execution)
                    .map_err(|error| failure(&context, error))?,
            };
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

fn input<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a NativeValue, NativeNodeFailure> {
    inputs
        .get(name)
        .ok_or_else(|| invalid(format!("Missing {name}")))
}
fn handle(value: &NativeValue) -> Result<&NativeOpaqueHandle, NativeNodeFailure> {
    match value {
        NativeValue::Handle { value } => Ok(value),
        _ => Err(invalid("Handle required")),
    }
}
fn resolve(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    let handle = handle(input(inputs, name)?)?;
    context
        .handle_store()
        .resolve(handle, handle.handle_type(), &context.cancellation)
        .map_err(|error| failure(context, error))
}
fn latent(payload: &NativeStoredPayload) -> Result<&NativeLatentBundle, NativeNodeFailure> {
    match payload {
        NativeStoredPayload::Latent(value) => Ok(value),
        _ => Err(invalid("LATENT bundle required")),
    }
}
fn integer(value: &NativeValue) -> Result<u64, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => u64::try_from(*value).map_err(invalid),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => Ok(*value),
        _ => Err(invalid("Integer required")),
    }
}
fn input_integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<u64, NativeNodeFailure> {
    integer(input(inputs, name)?)
}
fn string(value: &NativeValue) -> Result<&str, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::String(value),
        } => Ok(value),
        _ => Err(invalid("String required")),
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

fn empty_latent(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let batch = input_integer(inputs, "batch_size")?;
    let mut latent_type = None;
    let mut spatial_ratio = None;
    let mut temporal_ratio = None;
    let shape = match class_type {
        "EmptyLatentAudio" => {
            let seconds = match input(inputs, "seconds")? {
                NativeValue::Primitive {
                    value: NativePrimitive::Number(value),
                } => *value,
                value => integer(value)? as f64,
            };
            latent_type = Some(NativeLatentType::Audio);
            temporal_ratio = Some(2048);
            vec![
                batch,
                64,
                (((seconds * 44100.0 / 2048.0) / 2.0).round_ties_even() as u64) * 2,
            ]
        }
        "EmptyLatentHunyuan3Dv2" => {
            latent_type = Some(NativeLatentType::Hunyuan3dV2);
            vec![batch, 64, input_integer(inputs, "resolution")?]
        }
        _ => {
            let width = input_integer(inputs, "width")?;
            let height = input_integer(inputs, "height")?;
            match class_type {
                "EmptySD3LatentImage" => {
                    spatial_ratio = Some(8);
                    vec![batch, 16, height / 8, width / 8]
                }
                "EmptyQwenImageLayeredLatentImage" => vec![
                    batch,
                    16,
                    input_integer(inputs, "layers")?
                        .checked_add(1)
                        .ok_or_else(|| invalid("Layer count overflow"))?,
                    height / 8,
                    width / 8,
                ],
                "EmptyHunyuanVideo15Latent" | "EmptyLTXVLatentVideo" | "EmptyMochiLatentVideo" => {
                    let (channels, temporal, spatial) = match class_type {
                        "EmptyHunyuanVideo15Latent" => {
                            spatial_ratio = Some(16);
                            (32, 4, 16)
                        }
                        "EmptyLTXVLatentVideo" => {
                            spatial_ratio = Some(32);
                            (128, 8, 32)
                        }
                        _ => (12, 6, 8),
                    };
                    let length = input_integer(inputs, "length")?
                        .checked_sub(1)
                        .ok_or_else(|| invalid("Video length must be positive"))?
                        / temporal
                        + 1;
                    vec![batch, channels, length, height / spatial, width / spatial]
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

fn add_latents(
    first: &NativeLatentBundle,
    second: &NativeLatentBundle,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let first_tensor = match first.samples() {
        NativeLatentSamples::Tensor(tensor)
        | NativeLatentSamples::AudioVideo { video: tensor, .. } => tensor,
    };
    let target = first_tensor.descriptor().shape();
    let reshape = |tensor: &Tensor| -> Result<Tensor, NativeNodeFailure> {
        let tensor = if tensor.descriptor().shape().get(1..) != target.get(1..) {
            resize_latent(tensor, target, backend, execution)?
        } else {
            tensor.clone()
        };
        let source_batch = *tensor
            .descriptor()
            .shape()
            .first()
            .ok_or_else(|| invalid("LATENT batch missing"))?;
        let batch = *target
            .first()
            .ok_or_else(|| invalid("Target batch missing"))?;
        if source_batch == batch {
            return Ok(tensor);
        }
        let tensor = if source_batch < batch {
            if source_batch == 0 {
                return Err(invalid("Cannot repeat an empty batch"));
            }
            let mut repeats = vec![1; tensor.descriptor().rank()];
            *repeats
                .first_mut()
                .ok_or_else(|| invalid("LATENT rank missing"))? =
                i64::try_from(batch.div_ceil(source_batch)).map_err(invalid)?;
            tensor_repeat_with_context_exact_native(backend, &tensor, &repeats, execution)
                .map_err(invalid)?
        } else {
            tensor
        };
        narrow_method_exact_native(&tensor, 0, 0, batch, execution.cancellation).map_err(invalid)
    };
    let add = |left: &Tensor, right: &Tensor| {
        add_method_with_context_exact_native(
            backend,
            left,
            ElementwiseOperand::Tensor(right),
            1.0,
            execution,
        )
        .map_err(invalid)
    };
    let samples = match (first.samples(), second.samples()) {
        (NativeLatentSamples::Tensor(left), NativeLatentSamples::Tensor(right)) => {
            NativeLatentSamples::Tensor(add(left, &reshape(right)?)?)
        }
        (NativeLatentSamples::AudioVideo { video, audio }, NativeLatentSamples::Tensor(right)) => {
            let right = reshape(right)?;
            NativeLatentSamples::AudioVideo {
                video: add(video, &right)?,
                audio: add(audio, &right)?,
            }
        }
        (
            NativeLatentSamples::AudioVideo { video, audio },
            NativeLatentSamples::AudioVideo {
                video: right_video,
                audio: right_audio,
            },
        ) if right_video.descriptor().shape() == target => NativeLatentSamples::AudioVideo {
            video: add(video, right_video)?,
            audio: add(audio, right_audio)?,
        },
        _ => {
            return Err(invalid(
                "Source nested LATENT addition does not support this reshape or operand order",
            ));
        }
    };
    first
        .replaced_samples(samples, NativeLatentMetadataRetention::Preserve, execution)
        .map_err(invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_part_two_preserves_exact_source_contracts() -> Result<(), Box<dyn std::error::Error>>
    {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/model-latent-comfy-node-0159/fixture.json"
        )))?;
        let rows = fixture["nodes"].as_array().ok_or("Missing rows")?;
        assert_eq!(rows.len(), 10);
        let bindings = native_node_bindings()?;
        crate::validate_generated_family_bindings(&bindings, NODE_DESCRIPTOR_IDS)?;
        let registry = NodeRegistry::built_in()?;
        assert_eq!(bindings.len(), 9);
        for binding in bindings {
            registry.validate_native_binding(&binding)?;
            let descriptor = binding.descriptor();
            descriptor.validate_exact_schema_v2()?;
            let row = rows
                .iter()
                .find(|row| row["class_type"] == descriptor.class_type)
                .ok_or("Missing source row")?;
            assert_eq!(row["feature_id"], binding.feature_id());
            assert_eq!(
                serde_json::to_value(built_in_source_schema(&descriptor.class_type)?)?,
                row["schema"]
            );
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert!(!descriptor.output_node);
            assert!(descriptor.dynamic_inputs.is_empty());
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Mapped));
            assert_eq!(descriptor.outputs.len(), 1);
            assert!(!descriptor.outputs.first().ok_or("Missing output")?.is_list);
            assert_eq!(binding.presentation().category, row["category"]);
            assert_eq!(
                *descriptor,
                serde_json::from_slice::<NativeNodeDescriptor>(&serde_json::to_vec(descriptor)?)?
            );
        }
        let retained = rows
            .iter()
            .find(|row| row["class_type"] == "EmptyLatentImage")
            .ok_or("Missing retained slice")?;
        assert_eq!(
            serde_json::to_value(built_in_source_schema("EmptyLatentImage")?)?,
            retained["schema"]
        );
        assert!(!NODE_DESCRIPTOR_IDS.contains(&"EmptyLatentImage"));
        Ok(())
    }
}
