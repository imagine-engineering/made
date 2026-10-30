use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor,
    NativeInputRequirement, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativeResolvedPayload,
    NativeStoredPayload, NativeValue, NativeValueType, NodeRegistry, built_in_source_schema,
    native_value_matches_input_schema, native_value_types_for_input_schema,
};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, Layout, MemoryFormatReference,
    NativeLatentBundle, NativeLatentMetadata, NativeLatentNoiseMask, NativeLatentSamples,
    NativeTensorPayload, NativeTensorRole, ResizeCrop, ResizeMode, Tensor,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_shape_layout_transform_02::{
        tensor_reshape_with_context_exact_native, torch_movedim_exact_native,
    },
    generated_storage_dtype_device_01::clone_with_context_exact_native,
    generated_tensor_creation_01::zeros_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "SetLatentNoiseMask",
    "TrimVideoLatent",
    "StableCascade_EmptyLatentImage",
    "StableCascade_StageC_VAEEncode",
    "VAEDecode",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
struct LatentNode {
    class_type: &'static str,
}
const FEATURES: [&str; 5] = [
    "COMFY-NODE-0618",
    "COMFY-NODE-0684",
    "COMFY-NODE-0635",
    "COMFY-NODE-0637",
    "COMFY-NODE-0710",
];
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
            let output_names = if matches!(
                *class_type,
                "StableCascade_EmptyLatentImage" | "StableCascade_StageC_VAEEncode"
            ) {
                vec!["stage_c".to_owned(), "stage_b".to_owned()]
            } else if *class_type == "VAEDecode" {
                vec!["image".to_owned()]
            } else {
                vec!["latent".to_owned()]
            };
            let source_schema = catalog
                .bind_execution_ports(&names, &[], &output_names)
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
                            "samples" | "mask" | "image" | "vae"
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
                    outputs: output_names
                        .iter()
                        .map(|name| {
                            Ok(NativeOutputDescriptor {
                                name: name.clone(),
                                produced_type: NativeValueType::Handle(
                                    if *class_type == "VAEDecode" {
                                        NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?
                                    } else {
                                        latent_type()?
                                    },
                                ),
                                is_list: false,
                            })
                        })
                        .collect::<Result<Vec<_>, NativeNodeContractError>>()?,
                    output_node: false,
                    effect: NativeEffectClass::Pure,
                    cache: NativeCachePolicy::InputIdentity,
                },
                presentation: NativeNodePresentation {
                    display_name: row.display_name.clone(),
                    category: row.category.clone(),
                    description: catalog.presentation.description.clone().unwrap_or_default(),
                    output_names,
                    search_aliases: if *class_type == "VAEDecode" {
                        [
                            "decode",
                            "decode latent",
                            "latent to image",
                            "render latent",
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
                node: Arc::new(node),
            })
        })
        .collect()
}

pub fn vae_decode_native_node() -> Arc<dyn NativeNode> {
    Arc::new(LatentNode {
        class_type: "VAEDecode",
    })
}

pub fn vae_decode_native_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    native_node_bindings()?
        .into_iter()
        .find(|binding| binding.feature_id() == "COMFY-NODE-0710")
        .ok_or_else(|| {
            NativeNodeContractError::InvalidSourceSchema("VAEDecode binding absent".to_owned())
        })
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
        if self.class_type == "VAEDecode" {
            "source-vae-decode-canonical-resource-v1"
        } else {
            "source-latent-part-six-three-tensor-leaves-v1"
        }
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
            if self.class_type == "VAEDecode" {
                let source = resolve(&context, &inputs, "samples")?;
                let bundle = latent(&source)?;
                let samples = match bundle.samples() {
                    NativeLatentSamples::Tensor(samples) => samples,
                    NativeLatentSamples::AudioVideo { video, .. } => video,
                };
                let model = resolve(&context, &inputs, "vae")?;
                let NativeStoredPayload::Model(model) = model.as_ref() else {
                    return Err(invalid("Canonical VAE model resource required"));
                };
                let vae = model
                    .model_payload()
                    .vae()
                    .ok_or_else(|| invalid("Canonical NativeVae resource required"))?;
                let formats = sim_model::LatentFormatRegistry::checked(
                    sim_model::GENERATED_LATENT_FORMATS,
                )
                .map_err(invalid)?;
                let format = formats
                    .get(&vae.descriptor().latent_format().identity)
                    .ok_or_else(|| invalid("VAE latent format unavailable"))?;
                // Workflow samples are source-unscaled; NativeVae.decode internally reverses this model-space transform.
                let model_samples =
                    sim_model::process_latent_in(format, compute.backend(), samples, &execution)
                        .map_err(|error| failure(&context, error))?;
                let decoded = vae
                    .decode(compute.backend(), &model_samples, &execution)
                    .map_err(|error| failure(&context, error))?;
                let images =
                    torch_movedim_exact_native(&decoded, &[1], &[-1], &context.cancellation)
                        .map_err(|error| failure(&context, error))?;
                let images = if images.descriptor().rank() == 5 {
                    let shape = images.descriptor().shape();
                    let flattened_batch = shape
                        .first()
                        .copied()
                        .and_then(|batch| {
                            shape.get(1).and_then(|frames| batch.checked_mul(*frames))
                        })
                        .ok_or_else(|| invalid("Decoded video batch overflow"))?;
                    let target = [
                        flattened_batch,
                        *shape
                            .get(2)
                            .ok_or_else(|| invalid("Decoded height absent"))?,
                        *shape
                            .get(3)
                            .ok_or_else(|| invalid("Decoded width absent"))?,
                        *shape
                            .get(4)
                            .ok_or_else(|| invalid("Decoded channels absent"))?,
                    ]
                    .into_iter()
                    .map(i64::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(invalid)?;
                    tensor_reshape_with_context_exact_native(
                        compute.backend(),
                        &images,
                        &target,
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?
                } else {
                    images
                };
                let images = clone_with_context_exact_native(
                    compute.backend(),
                    &images,
                    MemoryFormatReference::Layout(Layout::Contiguous),
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
                let image =
                    ImageTensor::from_tensor(images).map_err(|error| failure(&context, error))?;
                let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
                    .map_err(|error| failure(&context, error))?;
                check(&context)?;
                let handle = context
                    .handle_store()
                    .publish(
                        NativeStoredPayload::Tensor(Arc::new(payload)),
                        &context.cancellation,
                    )
                    .map_err(|error| failure(&context, error))?;
                let result = check(&context).and_then(|()| {
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
                        .map_err(|error| failure(&context, error))?;
                }
                return result;
            }
            if self.class_type == "StableCascade_EmptyLatentImage" {
                let integer = |name| match input(&inputs, name)? {
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(value),
                    } => u64::try_from(*value).map_err(invalid),
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(value),
                    } => Ok(*value),
                    _ => Err(invalid("Integer source dimensions required")),
                };
                let width = integer("width")?;
                let height = integer("height")?;
                let compression = integer("compression")?;
                let batch = integer("batch_size")?;
                let stage_c = [
                    batch,
                    16,
                    height
                        .checked_div(compression)
                        .ok_or_else(|| invalid("Zero compression"))?,
                    width
                        .checked_div(compression)
                        .ok_or_else(|| invalid("Zero compression"))?,
                ];
                let stage_b = [batch, 4, height / 4, width / 4];
                let mut outputs = Vec::new();
                for shape in [&stage_c, &stage_b] {
                    let samples = zeros_with_context_exact_native(
                        compute.backend(),
                        shape,
                        DType::F32,
                        Layout::Strided,
                        DeviceId::CPU,
                        false,
                        None,
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?;
                    outputs.push(
                        NativeLatentBundle::single(
                            samples,
                            None,
                            None,
                            NativeLatentMetadata::default(),
                            &execution,
                        )
                        .map_err(|error| failure(&context, error))?,
                    );
                }
                return publish_latents(&context, outputs);
            }
            if self.class_type == "StableCascade_StageC_VAEEncode" {
                let image_payload = resolve(&context, &inputs, "image")?;
                let NativeStoredPayload::Tensor(image_payload) = image_payload.as_ref() else {
                    return Err(invalid("Canonical IMAGE tensor required"));
                };
                if image_payload.role() != NativeTensorRole::Image {
                    return Err(invalid("IMAGE tensor role required"));
                }
                let image = image_payload
                    .image()
                    .ok_or_else(|| invalid("Canonical BHWC ImageTensor required"))?;
                let model_payload = resolve(&context, &inputs, "vae")?;
                let NativeStoredPayload::Model(model_payload) = model_payload.as_ref() else {
                    return Err(invalid("Canonical VAE model resource required"));
                };
                let vae = model_payload
                    .model_payload()
                    .vae()
                    .ok_or_else(|| invalid("Canonical NativeVae resource required"))?;
                let compression = match input(&inputs, "compression")? {
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(value),
                    } => u64::try_from(*value).map_err(invalid)?,
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(value),
                    } => *value,
                    _ => return Err(invalid("Integer compression required")),
                };
                let (_, height, width, _) = image.dimensions().map_err(invalid)?;
                let downscale_ratio = vae
                    .descriptor()
                    .source_image_encode_downscale_ratio()
                    .map_err(invalid)?;
                let dimension = |value: u64| {
                    value
                        .checked_div(compression)
                        .and_then(|value| value.checked_mul(downscale_ratio))
                        .ok_or_else(|| invalid("VAE resize dimension overflow or zero compression"))
                };
                let out_width = dimension(width)?;
                let out_height = dimension(height)?;
                // The source interpolator rejects zero targets; ImageTensor's zero means automatic sizing.
                if out_width == 0 || out_height == 0 {
                    return Err(invalid(
                        "Source bicubic VAE resize requires positive targets",
                    ));
                }
                let resized = image
                    .resize(
                        out_width,
                        out_height,
                        ResizeMode::Bicubic,
                        ResizeCrop::Center,
                        compute.backend(),
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?;
                let channels = resized.dimensions().map_err(invalid)?.3.min(3);
                let rgb = narrow_method_exact_native(
                    resized.tensor(),
                    3,
                    0,
                    channels,
                    &context.cancellation,
                )
                .map_err(|error| failure(&context, error))?;
                let pixels = torch_movedim_exact_native(&rgb, &[-1], &[1], &context.cancellation)
                    .map_err(|error| failure(&context, error))?;
                let encoded = vae
                    .encode(compute.backend(), &pixels, &execution)
                    .map_err(|error| failure(&context, error))?;
                let batch = *encoded
                    .descriptor()
                    .shape()
                    .first()
                    .ok_or_else(|| invalid("Encoded VAE batch is absent"))?;
                let stage_b = zeros_with_context_exact_native(
                    compute.backend(),
                    &[batch, 4, (height / 8) * 2, (width / 8) * 2],
                    DType::F32,
                    Layout::Strided,
                    DeviceId::CPU,
                    false,
                    None,
                    &execution,
                )
                .map_err(|error| failure(&context, error))?;
                let mut outputs = Vec::new();
                for samples in [encoded, stage_b] {
                    outputs.push(
                        NativeLatentBundle::single(
                            samples,
                            None,
                            None,
                            NativeLatentMetadata::default(),
                            &execution,
                        )
                        .map_err(|error| failure(&context, error))?,
                    );
                }
                return publish_latents(&context, outputs);
            }
            let source = resolve(&context, &inputs, "samples")?;
            let bundle = latent(&source)?;
            let samples = bundle
                .samples()
                .tensor()
                .ok_or_else(|| invalid("Tensor LATENT samples required"))?;
            let output = match self.class_type {
                "SetLatentNoiseMask" => {
                    let mask = resolve(&context, &inputs, "mask")?;
                    let NativeStoredPayload::Tensor(mask) = mask.as_ref() else {
                        return Err(invalid("Canonical MASK tensor required"));
                    };
                    if mask.role() != NativeTensorRole::Mask {
                        return Err(invalid("MASK tensor role required"));
                    }
                    let mask = mask.tensor();
                    let shape = mask.descriptor().shape();
                    let height = *shape
                        .get(
                            shape
                                .len()
                                .checked_sub(2)
                                .ok_or_else(|| invalid("MASK spatial rank"))?,
                        )
                        .ok_or_else(|| invalid("MASK height"))?;
                    let width = *shape.last().ok_or_else(|| invalid("MASK width"))?;
                    let mask = tensor_reshape_with_context_exact_native(
                        compute.backend(),
                        mask,
                        &[
                            -1,
                            1,
                            i64::try_from(height).map_err(invalid)?,
                            i64::try_from(width).map_err(invalid)?,
                        ],
                        &execution,
                    )
                    .map_err(|error| failure(&context, error))?;
                    bundle
                        .replaced_samples_mask_and_indices_source_dictionary(
                            NativeLatentSamples::Tensor(samples.clone()),
                            Some(NativeLatentNoiseMask::Tensor(mask)),
                            bundle.batch_indices().map(<[u64]>::to_vec),
                            &execution,
                        )
                        .map_err(|error| failure(&context, error))?
                }
                "TrimVideoLatent" => {
                    let amount = match input(&inputs, "trim_amount")? {
                        NativeValue::Primitive {
                            value: NativePrimitive::Integer(value),
                        } => *value,
                        NativeValue::Primitive {
                            value: NativePrimitive::UnsignedInteger(value),
                        } => i64::try_from(*value).map_err(invalid)?,
                        _ => return Err(invalid("Integer trim amount required")),
                    };
                    let extent = *samples
                        .descriptor()
                        .shape()
                        .get(2)
                        .ok_or_else(|| invalid("LATENT temporal axis required"))?;
                    let signed_extent = i64::try_from(extent).map_err(invalid)?;
                    let start = if amount < 0 {
                        amount.saturating_add(signed_extent)
                    } else {
                        amount
                    }
                    .clamp(0, signed_extent);
                    let view = narrow_method_exact_native(
                        samples,
                        2,
                        start,
                        extent - u64::try_from(start).map_err(invalid)?,
                        &context.cancellation,
                    )
                    .map_err(|error| failure(&context, error))?;
                    // Source slicing can return strided storage; the LATENT owner requires contiguous storage.
                    let output = cloned(&view, compute.backend(), &execution).map_err(|error| {
                        if context.cancellation.is_cancelled() {
                            failure(&context, error)
                        } else {
                            error
                        }
                    })?;
                    bundle
                        .replaced_samples_source_dictionary(
                            NativeLatentSamples::Tensor(output),
                            &execution,
                        )
                        .map_err(|error| failure(&context, error))?
                }
                _ => return Err(invalid("Unassigned latent node")),
            };
            publish_latents(&context, vec![output])
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
fn publish_latents(
    context: &NativeNodeContext,
    outputs: Vec<NativeLatentBundle>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let mut handles = Vec::new();
    let result = (|| {
        for output in outputs {
            check(context)?;
            handles.push(
                context
                    .handle_store()
                    .publish(
                        NativeStoredPayload::Latent(Arc::new(output)),
                        &context.cancellation,
                    )
                    .map_err(|error| failure(context, error))?,
            );
        }
        check(context)?;
        let result = NativeNodeOutcome::Values {
            outputs: handles
                .iter()
                .cloned()
                .map(|value| NativeValue::Handle { value })
                .collect(),
            ui: None,
            effects: Vec::new(),
        };
        result.validate().map_err(invalid)?;
        Ok(result)
    })();
    if result.is_err() {
        let mut errors = Vec::new();
        for handle in handles.iter().rev() {
            if let Err(error) = context
                .handle_store()
                .revoke(handle, &sim_types::CancellationToken::default())
            {
                errors.push(error.to_string());
            }
        }
        if !errors.is_empty() {
            return Err(failure(context, errors.join("; ")));
        }
    }
    result
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

fn cloned(
    input: &Tensor,
    backend: &CpuBackend,
    execution: &ExecutionContext<'_>,
) -> Result<Tensor, NativeNodeFailure> {
    clone_with_context_exact_native(
        backend,
        input,
        MemoryFormatReference::Layout(Layout::Contiguous),
        execution,
    )
    .map_err(invalid)
}
