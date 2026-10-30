use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeAssetNameResolutionRequest, NativeAssetReadRequest,
    NativeAssetServiceError, NativeCacheDependencies, NativeCachePolicy, NativeEffectClass,
    NativeHandleKind, NativeHandleType, NativeInputDescriptor, NativeInputRequirement, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativeResolvedAsset, NativeStoredPayload, NativeValue, NativeValueType,
    NodeRegistry, built_in_source_schema, native_value_matches_input_schema,
    native_value_matches_source_input_schema, native_value_types_for_input_schema,
};
use sim_model::{
    formats::parse_safetensors_bytes, parser_limits::ParserLimits, vae::VaeKernelProfile,
    vae_architecture::VaeLoaderConfiguration, vae_audio::audio_vae_source_plan,
};
use sim_tensor::{
    DType, DecodedScalar, DeviceId, ExecutionContext, Layout, MemoryFormatReference,
    NativeLatentBundle, NativeLatentMetadata, NativeLatentNoiseMask, NativeLatentSamples,
    NativeLatentType, NumericClass, Scalar, Tensor, ViewAccess,
    generated_elementwise_or_runtime_operation_09::{
        frombuffer_with_context_exact_native, mul_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_15::ones_like_with_context_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_reduction_01::tensor_mean_with_context_exact_native,
    generated_shape_layout_transform_02::{
        tensor_repeat_with_context_exact_native, tensor_reshape_with_context_exact_native,
        torch_cat_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        InterpolateConfiguration, InterpolateMode, interpolate_tensor_with_context_exact_native,
    },
    generated_storage_dtype_device_01::{
        clone_with_context_exact_native, copy_with_context_exact_native,
        float_with_context_exact_native,
    },
    generated_tensor_creation_01::{
        full_with_context_exact_native, ones_with_context_exact_native,
        zeros_with_context_exact_native,
    },
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "LoadLatent",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVEmptyLatentAudio",
    "RebatchLatents",
    "RepeatLatentBatch",
    "ReplaceVideoLatentFrames",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const VERSION: &str = "source-latent-part-five-v1";
const MAXIMUM_LATENT_ASSET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
struct LatentNode {
    class_type: &'static str,
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        ("LoadLatent", "COMFY-NODE-0344"),
        ("LTXVConcatAVLatent", "COMFY-NODE-0364"),
        ("LTXVSeparateAVLatent", "COMFY-NODE-0375"),
        ("LTXVEmptyLatentAudio", "COMFY-NODE-0368"),
        ("RebatchLatents", "COMFY-NODE-0507"),
        ("RepeatLatentBatch", "COMFY-NODE-0536"),
        ("ReplaceVideoLatentFrames", "COMFY-NODE-0538"),
    ]
    .into_iter()
    .map(|(class_type, feature_id)| binding(class_type, feature_id))
    .collect()
}
fn binding(
    class_type: &'static str,
    feature_id: &str,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog = built_in_source_schema(class_type).map_err(contract)?;
    let registry = NodeRegistry::built_in().map_err(contract)?;
    let row = registry
        .descriptor(class_type)
        .ok_or_else(|| contract("Missing latent source row"))?;
    let output_names: Vec<String> = if class_type == "LTXVSeparateAVLatent" {
        vec!["video_latent".into(), "audio_latent".into()]
    } else {
        vec!["latent".into()]
    };
    let input_names = catalog
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let schema = catalog
        .bind_execution_ports(&input_names, &[], &output_names)
        .map_err(contract)?;
    let inputs = catalog
        .inputs
        .iter()
        .map(|input| {
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types: native_value_types_for_input_schema(&input.schema)
                    .map_err(contract)?,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: false,
                lazy: false,
                cardinality: if class_type == "RebatchLatents" {
                    NativePortCardinality::List
                } else {
                    NativePortCardinality::Mapped
                },
                allows_literal: matches!(
                    input.schema.name.as_str(),
                    "amount" | "index" | "batch_size" | "frames_number" | "frame_rate"
                ) || (class_type == "LoadLatent" && input.schema.name == "latent"),
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let latent_type = NativeHandleType::new(NativeHandleKind::Latent, "LATENT")?;
    Ok(NativeNodeBinding::Executable {
        feature_id: feature_id.into(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.into(),
            implementation_version: VERSION.into(),
            source_schema: Some(schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: output_names
                .iter()
                .map(|name| NativeOutputDescriptor {
                    name: name.clone(),
                    produced_type: NativeValueType::Handle(latent_type.clone()),
                    is_list: class_type == "RebatchLatents",
                })
                .collect(),
            output_node: false,
            effect: if class_type == "LoadLatent" {
                NativeEffectClass::ReadsArtifact
            } else {
                NativeEffectClass::Pure
            },
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: row.display_name.clone(),
            category: row.category.clone(),
            description: catalog.presentation.description.clone().unwrap_or_default(),
            output_names,
            search_aliases: if class_type == "LoadLatent" {
                vec!["import latent".into(), "open latent".into()]
            } else if class_type == "RepeatLatentBatch" {
                vec!["duplicate latent".into(), "clone latent".into()]
            } else {
                Vec::new()
            },
            is_deprecated: catalog.presentation.is_deprecated,
            is_experimental: catalog.presentation.is_experimental,
        },
        node: Arc::new(LatentNode { class_type }),
    })
}
fn contract(error: impl std::fmt::Display) -> NativeNodeContractError {
    NativeNodeContractError::InvalidSourceSchema(error.to_string())
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_latent_part_five_inputs".into(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".into(),
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
fn validate(
    class_type: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    let catalog = built_in_source_schema(class_type).map_err(invalid)?;
    let source_schema = if class_type == "LoadLatent" {
        Some(
            catalog
                .bind_execution_ports(&["latent".to_owned()], &[], &["latent".to_owned()])
                .map_err(invalid)?,
        )
    } else {
        None
    };
    if inputs.keys().any(|name| {
        !catalog
            .inputs
            .iter()
            .any(|input| input.schema.name == *name)
    }) {
        return Err(invalid("Unknown latent input"));
    }
    for input in &catalog.inputs {
        let Some(value) = inputs.get(&input.schema.name) else {
            if input.requirement == NativeInputRequirement::Required {
                return Err(invalid("Missing latent input"));
            }
            continue;
        };
        value.validate().map_err(invalid)?;
        let values = if class_type == "RebatchLatents" {
            let NativeValue::List { values } = value else {
                return Err(invalid("Rebatch inputs must be lists"));
            };
            values.as_slice()
        } else {
            std::slice::from_ref(value)
        };
        for value in values {
            if class_type == "ReplaceVideoLatentFrames"
                && input.schema.name == "source"
                && matches!(
                    value,
                    NativeValue::Primitive {
                        value: NativePrimitive::Null
                    }
                )
            {
                continue;
            }
            let schema_matches = source_schema.as_ref().map_or_else(
                || native_value_matches_input_schema(value, &input.schema),
                |source_schema| {
                    native_value_matches_source_input_schema(
                        value,
                        class_type,
                        source_schema,
                        &input.schema,
                    )
                },
            );
            if !schema_matches {
                return Err(invalid(format!(
                    "{} violates its source schema",
                    input.schema.name
                )));
            }
        }
    }
    Ok(())
}
fn integer(value: Option<&NativeValue>) -> Result<i64, NativeNodeFailure> {
    match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => i64::try_from(*value).map_err(invalid),
        _ => Err(invalid("Integer required")),
    }
}
fn latent(
    context: &NativeNodeContext,
    value: Option<&NativeValue>,
) -> Result<Arc<NativeLatentBundle>, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid("LATENT handle required"));
    };
    let payload = context
        .handle_store()
        .resolve(value, value.handle_type(), &context.cancellation)
        .map_err(|error| failure(context, error))?;
    match payload.as_ref() {
        NativeStoredPayload::Latent(latent) => Ok(latent.clone()),
        _ => Err(invalid("LATENT bundle required")),
    }
}
fn tensor(latent: &NativeLatentBundle) -> Result<&Tensor, NativeNodeFailure> {
    latent
        .samples()
        .tensor()
        .ok_or_else(|| invalid("Ordinary tensor samples required"))
}
fn mask(latent: &NativeLatentBundle) -> Result<Option<&Tensor>, NativeNodeFailure> {
    match latent.noise_mask() {
        None => Ok(None),
        Some(NativeLatentNoiseMask::Tensor(mask)) => Ok(Some(mask)),
        _ => Err(invalid("Ordinary tensor mask required")),
    }
}
fn batch(tensor: &Tensor) -> Result<u64, NativeNodeFailure> {
    tensor
        .descriptor()
        .shape()
        .first()
        .copied()
        .ok_or_else(|| invalid("Missing batch dimension"))
}
fn repeat(
    context: &NativeNodeContext,
    tensor: &Tensor,
    amount: i64,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    let mut repeats = vec![1; tensor.descriptor().shape().len()];
    let first = repeats
        .first_mut()
        .ok_or_else(|| invalid("Missing repeat dimension"))?;
    *first = amount;
    tensor_repeat_with_context_exact_native(
        context.compute_session().map_err(invalid)?.backend(),
        tensor,
        &repeats,
        execution,
    )
    .map_err(|error| failure(context, error))
}
fn slice(
    context: &NativeNodeContext,
    tensor: &Tensor,
    start: u64,
    length: u64,
) -> Result<Tensor, NativeNodeFailure> {
    let size = batch(tensor)?;
    let start = start.min(size);
    let length = length.min(size - start);
    narrow_method_exact_native(
        tensor,
        0,
        i64::try_from(start).map_err(invalid)?,
        length,
        &context.cancellation,
    )
    .map_err(|error| failure(context, error))
}
fn collect_dependencies(
    context: &NativeNodeContext,
    name: &str,
    value: &NativeValue,
    result: &mut BTreeMap<String, String>,
) -> Result<(), NativeNodeFailure> {
    match value {
        NativeValue::Handle { value } => {
            let payload = context
                .handle_store()
                .resolve(value, value.handle_type(), &context.cancellation)
                .map_err(|error| failure(context, error))?;
            result.insert(name.to_owned(), payload.digest_sha256());
        }
        NativeValue::List { values } => {
            for (index, value) in values.iter().enumerate() {
                check(context)?;
                collect_dependencies(context, &format!("{name}/{index}"), value, result)?;
            }
        }
        _ => {}
    }
    Ok(())
}
impl NativeNode for LatentNode {
    fn class_type(&self) -> &str {
        self.class_type
    }
    fn implementation_version(&self) -> &str {
        VERSION
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
        Ok(VERSION.into())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        validate(self.class_type, inputs)?;
        let mut artifact_digests = BTreeMap::new();
        if self.class_type == "LoadLatent" {
            let asset = resolve_latent_asset(context, inputs)?;
            artifact_digests.insert("latent".into(), asset.sha256().to_owned());
        }
        for (name, value) in inputs {
            collect_dependencies(context, name, value, &mut artifact_digests)?;
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
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let outputs = match self.class_type {
                "LoadLatent" => vec![load_latent(&context, &inputs, &execution)?],
                "RepeatLatentBatch" => vec![repeat_latent(&context, &inputs, &execution)?],
                "LTXVConcatAVLatent" => vec![concat_av(&context, &inputs, &execution)?],
                "LTXVSeparateAVLatent" => separate_av(&context, &inputs, &execution)?,
                "ReplaceVideoLatentFrames" => {
                    let destination = latent(&context, inputs.get("destination"))?;
                    let source = match inputs.get("source") {
                        None
                        | Some(NativeValue::Primitive {
                            value: NativePrimitive::Null,
                        }) => None,
                        value => Some(latent(&context, value)?),
                    };
                    let index = integer(inputs.get("index"))?;
                    let Some(output) = replace_frames(
                        &context,
                        &destination,
                        source.as_deref(),
                        index,
                        &execution,
                    )?
                    else {
                        let value = inputs
                            .get("destination")
                            .cloned()
                            .ok_or_else(|| invalid("Missing destination"))?;
                        check(&context)?;
                        return outcome(vec![value]);
                    };
                    vec![output]
                }
                "RebatchLatents" => rebatch(&context, &inputs, &execution)?,
                "LTXVEmptyLatentAudio" => vec![empty_audio_latent(&context, &inputs, &execution)?],
                _ => return Err(invalid("Unknown latent node")),
            };
            publish_all(&context, outputs, self.class_type == "RebatchLatents")
        })
    }
}
fn asset_failure(context: &NativeNodeContext, error: NativeAssetServiceError) -> NativeNodeFailure {
    if error == NativeAssetServiceError::Cancelled {
        NativeNodeFailure {
            code: "interrupted".into(),
            message: error.to_string(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        failure(context, error)
    }
}
fn latent_asset_name(inputs: &BTreeMap<String, NativeValue>) -> Result<&str, NativeNodeFailure> {
    match inputs.get("latent") {
        Some(NativeValue::PreservedUnknown {
            type_name,
            value: serde_json::Value::String(name),
        }) if type_name == "COMBO" => Ok(name),
        _ => Err(invalid("A source latent filename is required")),
    }
}
fn resolve_latent_asset(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeResolvedAsset, NativeNodeFailure> {
    check(context)?;
    let name = latent_asset_name(inputs)?;
    let resolver = context
        .asset_resolver()
        .map_err(|error| asset_failure(context, error))?;
    let request =
        NativeAssetNameResolutionRequest::checked("input", vec![name.to_owned()], "LATENT")
            .map_err(|error| asset_failure(context, error))?;
    let resolution = resolver
        .resolve_names(&request, &context.cancellation)
        .map_err(|error| asset_failure(context, error))?;
    let asset = resolution
        .assets()
        .first()
        .filter(|asset| asset.name() == name)
        .ok_or_else(|| invalid("Latent asset resolution changed its source name"))?;
    if resolution.folder_category() != request.folder_category()
        || resolution.assets().len() != 1
        || asset.reference().service_id() != resolver.identity().service_id()
        || asset.reference().source_type_id() != request.source_type_id()
    {
        return Err(invalid("Latent asset resolution changed its source domain"));
    }
    let read =
        NativeAssetReadRequest::checked(asset.reference().clone(), MAXIMUM_LATENT_ASSET_BYTES)
            .map_err(|error| asset_failure(context, error))?;
    let resolved = resolver
        .read_verified(&read, &context.cancellation)
        .map_err(|error| asset_failure(context, error))?;
    if resolved.reference() != read.reference() || resolved.byte_length() > read.maximum_bytes() {
        return Err(invalid("Latent asset read changed its verified identity"));
    }
    check(context)?;
    Ok(resolved)
}
fn load_latent(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let asset = resolve_latent_asset(context, inputs)?;
    let limits = ParserLimits {
        maximum_tensor_bytes: MAXIMUM_LATENT_ASSET_BYTES,
        maximum_aggregate_tensor_bytes: MAXIMUM_LATENT_ASSET_BYTES,
        ..ParserLimits::default()
    };
    let parsed = parse_safetensors_bytes(
        asset.bytes(),
        Path::new(latent_asset_name(inputs)?),
        &limits,
        &context.cancellation,
    )
    .map_err(|error| failure(context, error))?;
    if parsed.source_sha256 != asset.sha256() || parsed.source_size != asset.byte_length() {
        return Err(invalid(
            "Latent parser changed the verified source identity",
        ));
    }
    for tensor in &parsed.tensors {
        let dtype = tensor.native_dtype().ok_or_else(|| NativeNodeFailure {
            code: "unsupported_latent_file_dtype".into(),
            message: format!(
                "Canonical source file dtype is unsupported: {}",
                tensor.data_type
            ),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        })?;
        if dtype.class() == NumericClass::Complex {
            return Err(invalid(
                "Source safetensors loader does not support complex file tensors",
            ));
        }
    }
    let metadata = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == "latent_tensor")
        .ok_or_else(|| invalid("Source latent_tensor is missing"))?;
    let dtype = metadata
        .native_dtype()
        .ok_or_else(|| invalid("Unsupported source latent tensor dtype"))?;
    let start = usize::try_from(metadata.storage.offset()).map_err(invalid)?;
    let end = metadata
        .storage
        .offset()
        .checked_add(metadata.storage.length())
        .ok_or_else(|| invalid("Latent tensor byte interval overflow"))?;
    let bytes = asset
        .bytes()
        .get(start..usize::try_from(end).map_err(invalid)?)
        .ok_or_else(|| invalid("Latent tensor byte interval exceeds verified asset"))?;
    let compute = context.compute_session().map_err(invalid)?;
    let source = frombuffer_with_context_exact_native(
        compute.backend(),
        bytes,
        dtype,
        None,
        0,
        compute.stream(),
        execution,
    )
    .map_err(|error| failure(context, error))?;
    let shape = metadata
        .shape
        .iter()
        .map(|dimension| i64::try_from(*dimension).map_err(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    let source =
        tensor_reshape_with_context_exact_native(compute.backend(), &source, &shape, execution)
            .map_err(|error| failure(context, error))?;
    let source = float_with_context_exact_native(
        compute.backend(),
        &source,
        MemoryFormatReference::PreserveFormat,
        execution,
    )
    .map_err(|error| failure(context, error))?;
    let multiplier = if parsed
        .tensors
        .iter()
        .any(|tensor| tensor.name == "latent_format_version_0")
    {
        1.0
    } else {
        1.0 / 0.18215
    };
    let multiplier = full_with_context_exact_native(
        compute.backend(),
        &[],
        Scalar::Float(multiplier),
        Some(DType::F32),
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        execution,
    )
    .map_err(|error| failure(context, error))?;
    let samples = mul_with_context_exact_native(compute.backend(), &source, &multiplier, execution)
        .map_err(|error| failure(context, error))?;
    NativeLatentBundle::single(
        samples,
        None,
        None,
        NativeLatentMetadata::default(),
        execution,
    )
    .map_err(|error| failure(context, error))
}
fn empty_audio_latent(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = inputs.get("audio_vae") else {
        return Err(invalid("A retained audio VAE is required"));
    };
    let resolved = context
        .handle_store()
        .resolve(value, value.handle_type(), &context.cancellation)
        .map_err(|error| failure(context, error))?;
    let NativeStoredPayload::Model(model) = resolved.as_ref() else {
        return Err(invalid("A retained audio VAE model is required"));
    };
    let vae = model
        .model_payload()
        .vae()
        .ok_or_else(|| invalid("A canonical native audio VAE is required"))?;
    let identity = vae.descriptor().identity();
    if identity.profile() != &VaeKernelProfile::LtxAudioV1 {
        return Err(invalid("Unsupported audio VAE profile"));
    }
    let VaeLoaderConfiguration::LtxAudio {
        latent_channels, ..
    } = identity.loader_configuration()
    else {
        return Err(invalid("A checked LTX audio configuration is required"));
    };
    let plan =
        audio_vae_source_plan(identity.profile()).map_err(|error| failure(context, error))?;
    let frequency_bins = plan
        .latent_frequency_bins()
        .ok_or_else(|| invalid("Audio VAE frequency bins are unavailable"))?;
    let frames = integer(inputs.get("frames_number"))?;
    let frame_rate = integer(inputs.get("frame_rate"))?;
    let batch = integer(inputs.get("batch_size"))?;
    let (numerator, denominator) = plan.sample_ratio();
    let latents_per_second =
        f64::from(plan.input_sample_rate()) * denominator as f64 / numerator as f64;
    let latent_frames = (frames as f64 / frame_rate as f64 * latents_per_second).ceil();
    if frame_rate == 0
        || batch < 0
        || !latent_frames.is_finite()
        || latent_frames < 0.0
        || latent_frames >= u64::MAX as f64
    {
        return Err(invalid("Invalid source audio latent dimensions"));
    }
    if batch == 0 || latent_frames == 0.0 {
        return Err(NativeNodeFailure {
            code: "unsupported_empty_audio_latent_representation".into(),
            message: "Canonical LATENT samples cannot represent source-valid empty dimensions"
                .into(),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        });
    }
    let samples = zeros_with_context_exact_native(
        context.compute_session().map_err(invalid)?.backend(),
        &[
            u64::try_from(batch).map_err(invalid)?,
            *latent_channels,
            latent_frames as u64,
            frequency_bins,
        ],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        false,
        None,
        execution,
    )
    .map_err(|error| failure(context, error))?;
    let metadata = NativeLatentMetadata::checked(Some(NativeLatentType::Audio), None, None, None)
        .map_err(|error| failure(context, error))?;
    NativeLatentBundle::single(samples, None, None, metadata, execution)
        .map_err(|error| failure(context, error))
}
fn source_dictionary(
    context: &NativeNodeContext,
    samples: Tensor,
    mask: Option<Tensor>,
    indices: Option<Vec<u64>>,
    metadata: NativeLatentMetadata,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let output = NativeLatentBundle::single(samples.clone(), mask, None, metadata, execution)
        .map_err(|error| failure(context, error))?;
    output
        .replaced_samples_and_indices_source_dictionary(
            NativeLatentSamples::Tensor(samples),
            indices,
            execution,
        )
        .map_err(|error| failure(context, error))
}
fn repeat_latent(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let source = latent(context, inputs.get("samples"))?;
    let samples = tensor(&source)?;
    let amount = integer(inputs.get("amount"))?;
    let output = repeat(context, samples, amount, execution)?;
    let output_mask = match mask(&source)? {
        Some(mask) if batch(mask)? > 1 => {
            if batch(mask)? < batch(samples)? {
                let times = batch(samples)?.div_ceil(batch(mask)?);
                let repeated = repeat(
                    context,
                    mask,
                    i64::try_from(times).map_err(invalid)?,
                    execution,
                )?;
                // The pinned source computes this normalized mask, then repeats the original mask instead.
                drop(slice(context, &repeated, 0, batch(samples)?)?);
            }
            Some(repeat(context, mask, amount, execution)?)
        }
        value => value.cloned(),
    };
    let indices = if let Some(indices) = source.batch_indices() {
        let minimum = indices
            .iter()
            .min()
            .ok_or_else(|| invalid("Cannot repeat empty batch indices"))?;
        let maximum = indices
            .iter()
            .max()
            .ok_or_else(|| invalid("Cannot repeat empty batch indices"))?;
        let offset = maximum
            .checked_sub(*minimum)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("Batch index overflow"))?;
        let count = indices
            .len()
            .checked_mul(usize::try_from(amount).map_err(invalid)?)
            .ok_or_else(|| invalid("Batch index overflow"))?;
        if count > 1_000_000 {
            return Err(invalid("Too many batch indices"));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(count).map_err(invalid)?;
        for ordinal in 0..u64::try_from(amount).map_err(invalid)? {
            check(context)?;
            let increment = ordinal
                .checked_mul(offset)
                .ok_or_else(|| invalid("Batch index overflow"))?;
            for index in indices {
                output.push(
                    index
                        .checked_add(increment)
                        .ok_or_else(|| invalid("Batch index overflow"))?,
                );
            }
        }
        Some(output)
    } else {
        None
    };
    source_dictionary(
        context,
        output,
        output_mask,
        indices,
        source.metadata().clone(),
        execution,
    )
}
fn concat_av(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    execution: &ExecutionContext<'_>,
) -> Result<NativeLatentBundle, NativeNodeFailure> {
    let video = latent(context, inputs.get("video_latent"))?;
    let audio = latent(context, inputs.get("audio_latent"))?;
    let video_samples = tensor(&video)?;
    let audio_samples = tensor(&audio)?;
    let (video_mask, audio_mask) = match (mask(&video)?, mask(&audio)?) {
        (None, None) => (None, None),
        (video_mask, audio_mask) => {
            let ones = |samples| {
                ones_like_with_context_exact_native(
                    context.compute_session().map_err(invalid)?.backend(),
                    samples,
                    None,
                    execution,
                )
                .map_err(|error| failure(context, error))
            };
            (
                Some(match video_mask {
                    Some(mask) => mask.clone(),
                    None => ones(video_samples)?,
                }),
                Some(match audio_mask {
                    Some(mask) => mask.clone(),
                    None => ones(audio_samples)?,
                }),
            )
        }
    };
    let metadata = NativeLatentMetadata::checked(
        audio
            .metadata()
            .latent_type()
            .or(video.metadata().latent_type()),
        audio
            .metadata()
            .sample_rate()
            .or(video.metadata().sample_rate()),
        audio
            .metadata()
            .spatial_downscale_ratio()
            .or(video.metadata().spatial_downscale_ratio()),
        audio
            .metadata()
            .temporal_downscale_ratio()
            .or(video.metadata().temporal_downscale_ratio()),
    )
    .map_err(invalid)?;
    let output = NativeLatentBundle::audio_video(
        video_samples.clone(),
        audio_samples.clone(),
        video_mask,
        audio_mask,
        None,
        metadata,
        execution,
    )
    .map_err(|error| failure(context, error))?;
    output
        .replaced_samples_and_indices_source_dictionary(
            output.samples().clone(),
            audio
                .batch_indices()
                .or(video.batch_indices())
                .map(<[u64]>::to_vec),
            execution,
        )
        .map_err(|error| failure(context, error))
}
fn unbind_pair(
    context: &NativeNodeContext,
    input: &Tensor,
    execution: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), NativeNodeFailure> {
    if batch(input)? < 2 {
        return Err(invalid("Source unbind requires at least two components"));
    }
    let shape = input
        .descriptor()
        .shape()
        .iter()
        .skip(1)
        .map(|size| i64::try_from(*size).map_err(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    let reshape = |start| {
        let view = slice(context, input, start, 1)?;
        tensor_reshape_with_context_exact_native(
            context.compute_session().map_err(invalid)?.backend(),
            &view,
            &shape,
            execution,
        )
        .map_err(|error| failure(context, error))
    };
    Ok((reshape(0)?, reshape(1)?))
}
fn separate_av(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    execution: &ExecutionContext<'_>,
) -> Result<Vec<NativeLatentBundle>, NativeNodeFailure> {
    let source = latent(context, inputs.get("av_latent"))?;
    let (video, audio) = match source.samples() {
        NativeLatentSamples::AudioVideo { video, audio } => (video.clone(), audio.clone()),
        NativeLatentSamples::Tensor(tensor) => unbind_pair(context, tensor, execution)?,
    };
    let (video_mask, audio_mask) = match source.noise_mask() {
        None => (None, None),
        Some(NativeLatentNoiseMask::AudioVideo { video, audio }) => {
            (Some(video.clone()), Some(audio.clone()))
        }
        Some(NativeLatentNoiseMask::Tensor(mask)) => {
            let (video, audio) = unbind_pair(context, mask, execution)?;
            (Some(video), Some(audio))
        }
    };
    [(video, video_mask), (audio, audio_mask)]
        .into_iter()
        .map(|(samples, mask)| {
            source_dictionary(
                context,
                samples,
                mask,
                source.batch_indices().map(<[u64]>::to_vec),
                source.metadata().clone(),
                execution,
            )
        })
        .collect()
}
fn replace_frames(
    context: &NativeNodeContext,
    destination: &NativeLatentBundle,
    source: Option<&NativeLatentBundle>,
    mut index: i64,
    execution: &ExecutionContext<'_>,
) -> Result<Option<NativeLatentBundle>, NativeNodeFailure> {
    let Some(source) = source else {
        return Ok(None);
    };
    let destination_samples = tensor(destination)?;
    let source_samples = tensor(source)?;
    let frames = |samples: &Tensor| {
        samples
            .descriptor()
            .shape()
            .get(2)
            .copied()
            .ok_or_else(|| invalid("Missing latent frame dimension"))
            .and_then(|value| i64::try_from(value).map_err(invalid))
    };
    let destination_frames = frames(destination_samples)?;
    let source_frames = frames(source_samples)?;
    if index < 0 {
        index = destination_frames
            .checked_add(index)
            .ok_or_else(|| invalid("Frame index overflow"))?;
    }
    let end = index
        .checked_add(source_frames)
        .ok_or_else(|| invalid("Frame index overflow"))?;
    if index > destination_frames || end > destination_frames {
        return Ok(None);
    }
    let normalize = |value: i64| {
        if value < 0 {
            destination_frames.saturating_add(value).max(0)
        } else {
            value.min(destination_frames)
        }
    };
    let start = normalize(index);
    let end = normalize(end);
    let output = clone_with_context_exact_native(
        context.compute_session().map_err(invalid)?.backend(),
        destination_samples,
        MemoryFormatReference::PreserveFormat,
        execution,
    )
    .map_err(|error| failure(context, error))?;
    let descriptor = output
        .descriptor()
        .narrowed_view(
            2,
            start,
            u64::try_from((end - start).max(0)).map_err(invalid)?,
        )
        .map_err(|error| failure(context, error))?;
    let mut view = output
        .view(descriptor, ViewAccess::Writable)
        .map_err(|error| failure(context, error))?;
    drop(
        copy_with_context_exact_native(
            context.compute_session().map_err(invalid)?.backend(),
            &mut view,
            source_samples,
            false,
            execution,
        )
        .map_err(|error| failure(context, error))?,
    );
    // Tensor writes detach shared storage; retain the written view's allocation.
    let output = view
        .view(output.descriptor().clone(), ViewAccess::Writable)
        .map_err(|error| failure(context, error))?;
    Ok(Some(
        source
            .replaced_samples_source_dictionary(NativeLatentSamples::Tensor(output), execution)
            .map_err(|error| failure(context, error))?,
    ))
}
struct Rebatch {
    samples: Tensor,
    mask: Tensor,
    indices: Vec<u64>,
}
fn prepare_batch(
    context: &NativeNodeContext,
    source: &NativeLatentBundle,
    processed: u64,
    execution: &ExecutionContext<'_>,
) -> Result<Rebatch, NativeNodeFailure> {
    let samples = tensor(source)?.clone();
    let shape = samples.descriptor().shape();
    let height = *shape
        .get(2)
        .ok_or_else(|| invalid("Missing latent height"))?;
    let width = *shape
        .get(3)
        .ok_or_else(|| invalid("Missing latent width"))?;
    let backend = context.compute_session().map_err(invalid)?.backend();
    let mut mask = match mask(source)? {
        Some(mask) => mask.clone(),
        None => ones_with_context_exact_native(
            backend,
            &[
                batch(&samples)?,
                1,
                height
                    .checked_mul(8)
                    .ok_or_else(|| invalid("Mask height overflow"))?,
                width
                    .checked_mul(8)
                    .ok_or_else(|| invalid("Mask width overflow"))?,
            ],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            execution,
        )
        .map_err(|error| failure(context, error))?,
    };
    let mask_shape = mask.descriptor().shape();
    let mask_width = *mask_shape
        .last()
        .ok_or_else(|| invalid("Missing mask width"))?;
    let mask_height = *mask_shape
        .get(
            mask_shape
                .len()
                .checked_sub(2)
                .ok_or_else(|| invalid("Missing mask height"))?,
        )
        .ok_or_else(|| invalid("Missing mask height"))?;
    if mask_width
        != width
            .checked_mul(8)
            .ok_or_else(|| invalid("Mask width overflow"))?
        || mask_height != height
    {
        let reshaped = tensor_reshape_with_context_exact_native(
            backend,
            &mask,
            &[
                -1,
                1,
                i64::try_from(mask_height).map_err(invalid)?,
                i64::try_from(mask_width).map_err(invalid)?,
            ],
            execution,
        )
        .map_err(|error| failure(context, error))?;
        // Source interpolation is executed for its failure/allocation behavior, but its result is discarded.
        drop(
            interpolate_tensor_with_context_exact_native(
                backend,
                &reshaped,
                &InterpolateConfiguration {
                    output_size: Some(vec![
                        usize::try_from(
                            height
                                .checked_mul(8)
                                .ok_or_else(|| invalid("Mask height overflow"))?,
                        )
                        .map_err(invalid)?,
                        usize::try_from(
                            width
                                .checked_mul(8)
                                .ok_or_else(|| invalid("Mask width overflow"))?,
                        )
                        .map_err(invalid)?,
                    ]),
                    scale_factor: None,
                    mode: InterpolateMode::Bilinear,
                    align_corners: None,
                    recompute_scale_factor: None,
                    antialias: false,
                },
                execution,
            )
            .map_err(|error| failure(context, error))?,
        );
    }
    if batch(&mask)? < batch(&samples)? {
        let count = batch(&mask)?;
        if count == 0 {
            return Err(invalid("Cannot repeat empty mask"));
        }
        mask = tensor_repeat_with_context_exact_native(
            backend,
            &mask,
            &[
                i64::try_from(batch(&samples)?.div_ceil(count)).map_err(invalid)?,
                1,
                1,
                1,
            ],
            execution,
        )
        .map_err(|error| failure(context, error))?;
        mask = slice(context, &mask, 0, batch(&samples)?)?;
    }
    let indices = match source.batch_indices() {
        Some(indices) => indices.to_vec(),
        None => (0..batch(&samples)?)
            .map(|index| {
                index
                    .checked_add(processed)
                    .ok_or_else(|| invalid("Batch index overflow"))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok(Rebatch {
        samples,
        mask,
        indices,
    })
}
fn slice_batch(
    context: &NativeNodeContext,
    source: &Rebatch,
    start: u64,
    length: u64,
) -> Result<Rebatch, NativeNodeFailure> {
    let index_start = usize::try_from(start)
        .map_err(invalid)?
        .min(source.indices.len());
    let index_end = usize::try_from(
        start
            .checked_add(length)
            .ok_or_else(|| invalid("Batch slice overflow"))?,
    )
    .map_err(invalid)?
    .min(source.indices.len());
    Ok(Rebatch {
        samples: slice(context, &source.samples, start, length)?,
        mask: slice(context, &source.mask, start, length)?,
        indices: source
            .indices
            .get(index_start..index_end)
            .ok_or_else(|| invalid("Invalid index slice"))?
            .to_vec(),
    })
}
fn rebatch(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    execution: &ExecutionContext<'_>,
) -> Result<Vec<NativeLatentBundle>, NativeNodeFailure> {
    let Some(NativeValue::List { values: latents }) = inputs.get("latents") else {
        return Err(invalid("Latent list required"));
    };
    let Some(NativeValue::List { values: sizes }) = inputs.get("batch_size") else {
        return Err(invalid("Batch size list required"));
    };
    let size = u64::try_from(integer(sizes.first())?).map_err(invalid)?;
    if size == 0 {
        return Err(invalid("Batch size must be positive"));
    }
    let backend = context.compute_session().map_err(invalid)?.backend();
    let mut processed = 0u64;
    let mut current: Option<Rebatch> = None;
    let mut outputs = Vec::new();
    for value in latents {
        check(context)?;
        let source = latent(context, Some(value))?;
        let next = prepare_batch(context, &source, processed, execution)?;
        processed = processed
            .checked_add(u64::try_from(next.indices.len()).map_err(invalid)?)
            .ok_or_else(|| invalid("Batch index overflow"))?;
        current = Some(match current.take() {
            None => next,
            Some(previous)
                if previous
                    .samples
                    .descriptor()
                    .shape()
                    .iter()
                    .rev()
                    .take(2)
                    .ne(next.samples.descriptor().shape().iter().rev().take(2)) =>
            {
                outputs.push(slice_batch(context, &previous, 0, size)?);
                next
            }
            Some(previous) => {
                let samples = torch_cat_with_context_exact_native(
                    backend,
                    &[previous.samples, next.samples],
                    0,
                    execution,
                )
                .map_err(|error| failure(context, error))?;
                let mask = torch_cat_with_context_exact_native(
                    backend,
                    &[previous.mask, next.mask],
                    0,
                    execution,
                )
                .map_err(|error| failure(context, error))?;
                let mut indices = previous.indices;
                if indices
                    .len()
                    .checked_add(next.indices.len())
                    .is_none_or(|length| length > 1_000_000)
                {
                    return Err(invalid("Too many rebatch indices"));
                }
                indices.try_reserve(next.indices.len()).map_err(invalid)?;
                indices.extend(next.indices);
                Rebatch {
                    samples,
                    mask,
                    indices,
                }
            }
        });
        if let Some(previous) = current.take() {
            let count = batch(&previous.samples)?;
            if count > size {
                let full = count / size;
                for ordinal in 0..full {
                    check(context)?;
                    if outputs.len() >= 1_000_000 {
                        return Err(invalid("Too many rebatch outputs"));
                    }
                    outputs.push(slice_batch(
                        context,
                        &previous,
                        ordinal
                            .checked_mul(size)
                            .ok_or_else(|| invalid("Batch slice overflow"))?,
                        size,
                    )?);
                }
                let start = full
                    .checked_mul(size)
                    .ok_or_else(|| invalid("Batch slice overflow"))?;
                if start < count {
                    current = Some(slice_batch(context, &previous, start, count - start)?);
                }
            } else {
                current = Some(previous);
            }
        }
    }
    if let Some(previous) = current {
        outputs.push(slice_batch(context, &previous, 0, size)?);
    }
    outputs
        .into_iter()
        .map(|output| {
            let mean = tensor_mean_with_context_exact_native(
                backend,
                &output.mask,
                None,
                false,
                None,
                execution,
            )
            .map_err(|error| failure(context, error))?;
            let decoded = mean
                .descriptor()
                .dtype()
                .decode_scalar(mean.element_bytes(&[]).map_err(invalid)?)
                .map_err(invalid)?;
            let mask = if decoded == DecodedScalar::Real(1.) {
                None
            } else {
                Some(output.mask)
            };
            source_dictionary(
                context,
                output.samples,
                mask,
                Some(output.indices),
                NativeLatentMetadata::default(),
                execution,
            )
        })
        .collect()
}
fn rollback(
    context: &NativeNodeContext,
    published: &[NativeOpaqueHandle],
) -> Result<(), NativeNodeFailure> {
    let cleanup = CancellationToken::default();
    for handle in published.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &cleanup)
            .map_err(|error| NativeNodeFailure {
                code: "native_latent_rollback_failed".into(),
                message: error.to_string(),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
    }
    Ok(())
}
fn publish_all(
    context: &NativeNodeContext,
    outputs: Vec<NativeLatentBundle>,
    is_list: bool,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    if outputs.len() > 1_000_000 {
        return Err(invalid("Too many latent outputs"));
    }
    let mut published = Vec::new();
    for output in outputs {
        if let Err(error) = check(context) {
            rollback(context, &published)?;
            return Err(error);
        }
        match context.handle_store().publish(
            NativeStoredPayload::Latent(Arc::new(output)),
            &context.cancellation,
        ) {
            Ok(handle) => published.push(handle),
            Err(error) => {
                rollback(context, &published)?;
                return Err(failure(context, error));
            }
        }
    }
    if let Err(error) = check(context) {
        rollback(context, &published)?;
        return Err(error);
    }
    let values = published
        .iter()
        .cloned()
        .map(|value| NativeValue::Handle { value })
        .collect();
    let result = outcome(if is_list {
        vec![NativeValue::List { values }]
    } else {
        values
    });
    if result.is_err() {
        rollback(context, &published)?;
    }
    result
}
fn outcome(outputs: Vec<NativeValue>) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let outcome = NativeNodeOutcome::Values {
        outputs,
        ui: None,
        effects: Vec::new(),
    };
    outcome.validate().map_err(invalid)?;
    Ok(outcome)
}
